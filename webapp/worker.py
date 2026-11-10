#!/usr/bin/env python3
"""Episode worker for the interactive eval page. One process per Python environment:

  worker.py cam   (.venv)     Diffusion Policy, ACT, DSRL       - plain YAM gripper, head + wrist cameras
  worker.py tac   (.venv-tw)  Touch in the Wild, DexScrew       - FlexiTac YAM gripper

Reads one JSON request per line on stdin: {"job", "model", "cond", "seed", "layout": {"nut": [x, y, yaw], "target": [x, y]}}
and answers with JSON lines on the ORIGINAL stdout (library prints are redirected to stderr):
  {"job", "type": "status", "msg"}                         loading a model, ...
  {"job", "type": "start", "target": [x, y, z]}
  {"job", "type": "frame", "t", "pose": b64 float32 (nbody x 7: pos, quat wxyz), "grip", "img": b64 jpeg | null,
   "tact": b64 uint8 (2 x 32 x 12) | null, "xy_mm", "h_mm", "contact": [l, r]}
  {"job", "type": "end", "success", "xy_err", "tilt", "steps", "error"?}
Models stay loaded on the GPU between requests. Episodes use the same success test as the offline evaluations.
"""
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTO = os.fdopen(os.dup(1), "w", buffering=1)      # protocol channel
os.dup2(2, 1)                                       # everything else printed (C or Python) goes to stderr
sys.stdout = sys.stderr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from registry import MODELS  # noqa: E402

FAMILY = sys.argv[1]
TABLE_TOP, NUT_REST = 0.98, 0.98 + 0.008


def send(**kw):
    PROTO.write(json.dumps(kw) + "\n")


def jpeg(a, q=80):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()


class Poser:
    """Body poses of the scene the page renders (scene.xml of the variant), from the episode's qpos."""

    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_path(str(xml)); self.d = mujoco.MjData(self.m)
        self.target = self.m.body("target").id

    def __call__(self, qpos, target):
        self.m.body_pos[self.target] = target
        self.d.qpos[:] = qpos; mujoco.mj_kinematics(self.m, self.d)
        P = np.concatenate([self.d.xpos, self.d.xquat], 1).astype(np.float32)
        return base64.b64encode(P.tobytes()).decode()


def tact_b64(t):            # (2, 32, 12) in [0, 1] -> uint8
    return base64.b64encode(np.clip(np.asarray(t) * 255, 0, 255).astype(np.uint8).tobytes()).decode()


# ============================================================================================ cam: DP / ACT / DSRL
if FAMILY == "cam":
    import torch
    sys.path.insert(0, str(REPO / "adamu_nut_yam" / "cotrain"))
    from act import ACT, CHUNK
    from cam_env import CamEnv
    from dp import TA, TO, DiffusionPolicy, augment
    DEV = "cuda"
    POSE = Poser(REPO / "adamu_nut_yam" / "assets" / "scene.xml")
    ENVS, POLS = {}, {}

    def policy(spec):
        if spec["id"] not in POLS:
            if spec["kind"] == "dsrl":
                from dsrl import Actor, Base
                ag = torch.load(Path(spec["path"]) / "agent.pt", map_location=DEV)
                base = Base(ag["args"]["base"], DEV)
                actor = Actor(ag["args"]["width"], ag["args"]["layers"]).to(DEV)
                actor.load_state_dict({k: v.float() for k, v in ag["actor"].items()}); actor.eval()
                POLS[spec["id"]] = (base.pol, (base, actor, ag["args"]["action_magnitude"]))
            else:
                pol = (ACT() if spec["kind"] == "act" else DiffusionPolicy()).to(DEV)
                pol.load_state_dict(torch.load(spec["path"], map_location=DEV)["state_dict"]); pol.eval()
                POLS[spec["id"]] = (pol, None)
        return POLS[spec["id"]]

    def episode(job, spec, cond, seed, layout):
        pol, dsrl = policy(spec)
        is_act = spec["kind"] == "act"
        if cond not in ENVS:
            ENVS[cond] = CamEnv(cond)
        env = ENVS[cond]; env.layout = layout
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        o = env.reset(seed)
        tgt = env.target_pos(); send(job=job, type="start", target=[float(x) for x in tgt])
        hist, plan, buf, t, info = [o] * TO, [], [], 0, {}

        def frame(g):
            p = env.nut_pos()
            send(job=job, type="frame", t=t, pose=POSE(env.d.qpos, tgt), grip=round(g, 3),
                 img=jpeg(np.concatenate(list(o["images"]), 1)), tact=None,
                 xy_mm=round(float(np.linalg.norm(p[:2] - tgt[:2])) * 1000, 1), h_mm=round(float(p[2] - NUT_REST) * 1000, 1),
                 contact=[int(env.grasped())] * 2)
        frame(0.0)
        while True:
            if is_act:       # temporal ensembling, as in act.py
                img = torch.from_numpy(o["images"])[None, None].to(DEV)
                prop = torch.from_numpy(o["proprio"])[None].to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    chunk = pol.act(augment(img, None, train=False)[:, 0], prop)[0].float().cpu().numpy()
                buf.append((t, chunk)); buf = [(t0, c) for t0, c in buf if t - t0 < CHUNK]
                preds = np.array([c[t - t0] for t0, c in buf]); w = np.exp(-0.01 * np.arange(len(preds)))
                plan = [(preds * w[:, None]).sum(0) / w.sum()]
            elif dsrl and not plan:
                base, actor, mag = dsrl
                s = base.encode([(np.stack([h["images"] for h in hist[-TO:]]), np.stack([h["proprio"] for h in hist[-TO:]]))])
                with torch.no_grad():
                    w = actor(s, deterministic=True)[0] * mag
                plan = list(base.unnorm(base.sample(s, w))[0, :TA].float().cpu().numpy())
            elif not plan:
                img = torch.from_numpy(np.stack([h["images"] for h in hist[-TO:]]))[None].to(DEV)
                prop = torch.from_numpy(np.stack([h["proprio"] for h in hist[-TO:]]))[None].to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    plan = list(pol.act(augment(img, None, train=False), prop, gen)[0].float().cpu().numpy()[:TA])
            a = plan.pop(0)
            o, done, info = env.step(a); hist.append(o); t += 1
            frame(float(np.clip(a[7], 0, 1)))
            if done:
                break
        return dict(success=bool(info["success"]), xy_err=float(info["xy_err"]), tilt=float(info["tilt"]), steps=t)

# ============================================================================================ tac: TW / DexScrew
else:
    import torch
    sys.path.insert(0, str(REPO / "tw")); sys.path.insert(0, str(REPO / "ds"))
    sys.path.insert(0, str(REPO / "ext" / "touch_in_the_wild")); sys.path.insert(0, str(REPO / "ext" / "dexscrew"))
    POSE = Poser(REPO / "adamu_nut_tactile" / "assets" / "scene.xml")
    POLS, ENVS = {}, {}

    def tw_episode(job, spec, cond, seed, layout):
        import eval_policy as ep
        from sim import TactileSim
        if spec["id"] not in POLS:
            POLS[spec["id"]] = ep.load(spec["path"])
        pol, cfg = POLS[spec["id"]]
        if "tw" not in ENVS:
            ENVS["tw"] = TactileSim()
        env = ENVS["tw"]; env.layout = layout
        o = env.reset(seed); o = env.observe()
        tgt = env.target_pos(); send(job=job, type="start", target=[float(x) for x in tgt])
        hist = [o, o]
        start = [np.concatenate([o["robot0_eef_pos"], o["robot0_eef_rot_axis_angle"]])]
        bumped, t, info, g = False, 0, {}, 0.0
        torch.manual_seed(seed)
        pol.reset()

        def frame():
            p = env.nut_pos(); tk = o["camera0_tactile"]                       # (12, 64): left | right
            tact = np.stack([tk[:, :32].T, tk[:, 32:].T])
            send(job=job, type="frame", t=t, pose=POSE(env.d.qpos, tgt), grip=round(g, 3), img=jpeg(o["camera0_rgb"]),
                 tact=tact_b64(tact), xy_mm=round(float(np.linalg.norm(p[:2] - tgt[:2])) * 1000, 1),
                 h_mm=round(float(p[2] - NUT_REST) * 1000, 1), contact=[int(tact[0].max() > 0), int(tact[1].max() > 0)])
        frame()
        while t < 320:
            obs = {k: np.stack([h[k] for h in hist[-2:]]) for k in ep.KEYS}
            with torch.no_grad():
                od = ep.get_real_umi_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta,
                                              obs_pose_repr=cfg.task.pose_repr.obs_pose_repr, tx_robot1_robot0=None,
                                              episode_start_pose=start)
                od = ep.dict_apply(od, lambda x: torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0).cuda())
                raw = pol.predict_action(od)["action_pred"][0].float().cpu().numpy()
            act = ep.get_real_umi_action(raw, obs, cfg.task.pose_repr.action_pose_repr)
            done = False
            for a in act[:8]:
                if cond == "bump" and not bumped and env.width_to_g(a[6]) > 0.15 and env.width() > 0.03:
                    p = env.nut_pos()
                    if np.linalg.norm(p[:2] - env.eef_pose()[0][:2]) < 0.02:          # closing around the nut
                        R = env.eef_pose()[1]
                        dv = R[:, 1].copy(); dv[2] = 0; dv /= np.linalg.norm(dv) + 1e-9
                        env.d.qpos[env.nut_q:env.nut_q + 2] += 0.03 * dv[:2]
                        mujoco.mj_forward(env.m, env.d); bumped = True
                _, done, info = env.step_pose(a[:3], a[3:6], a[6])
                o = env.observe(); hist.append(o); t += 1; g = float(env.width_to_g(a[6]))
                frame()
                if done:
                    break
            if (info and info.get("success")) or done or t >= 300:
                break
        return dict(success=bool(info.get("success", False)), xy_err=float(info.get("xy_err", np.nan)),
                    tilt=float(info.get("tilt", np.nan)), steps=t)

    def ds_episode(job, spec, cond, seed, layout):
        from eval_ds import build_policy
        from nut_rl_env import SHIFTED, NutRLEnv
        if spec["id"] not in POLS:
            POLS[spec["id"]] = build_policy(spec["path"], spec["kind"][3:])
        act = POLS[spec["id"]]
        key = "ds_shifted" if cond == "shifted" else "ds"
        if key not in ENVS:
            ENVS[key] = NutRLEnv(seed=0, randomize=False, domain=SHIFTED if cond == "shifted" else None)
        env = ENVS[key]; env.sim.layout = layout
        o = env.reset(seed); env.bump = cond == "bump"
        sim = env.sim; tgt = sim.target_pos(); send(job=job, type="start", target=[float(x) for x in tgt])
        t, fail, info = 0, False, {}

        def frame():
            p = sim.nut_pos()
            send(job=job, type="frame", t=t, pose=POSE(env.d.qpos, tgt), grip=round(float(env.g), 3), img=None,
                 tact=tact_b64(env.tact), xy_mm=round(float(np.linalg.norm(p[:2] - tgt[:2])) * 1000, 1),
                 h_mm=round(float(p[2] - NUT_REST) * 1000, 1), contact=[int(c) for c in env.contacts()])
        frame()
        while t < 300:
            o, r, done, inf = env.step(act(*o)); t += 1
            frame()
            if inf["fail"]:
                fail = True; break
            if sim.state.success_streak >= 10:
                break
        info = sim.state.info
        return dict(success=bool(sim.state.success_streak >= 10), xy_err=float(info.get("xy_err", np.nan)),
                    tilt=float(info.get("tilt", np.nan)), steps=t, fail=fail)

    def episode(job, spec, cond, seed, layout):
        return (tw_episode if spec["kind"] == "tw" else ds_episode)(job, spec, cond, seed, layout)


def main():
    send(type="ready", family=FAMILY)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line); job = req["job"]
        try:
            spec = MODELS[req["model"]]
            lay = req.get("layout") or {}
            layout = {k: tuple(float(x) for x in v) for k, v in lay.items()} or None
            send(job=job, type="status", msg="running" if spec["id"] in POLS else "loading model")
            t0 = time.time()
            res = episode(job, spec, req["cond"], int(req.get("seed", 2_000_000)), layout)
            send(job=job, type="end", seconds=round(time.time() - t0, 1), **res)
        except Exception as e:                                          # report and keep serving
            traceback.print_exc()
            send(job=job, type="end", success=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
