#!/usr/bin/env python3
"""Evaluate a Touch-in-the-Wild diffusion policy checkpoint in the MuJoCo FlexiTac scene.

Inference follows the repo's eval_real.py: 2-frame observation history -> get_real_umi_obs_dict (relative poses,
pose w.r.t. episode start) -> policy.predict_action (16 DDIM steps) -> get_real_umi_action (absolute targets) ->
execute the first n_action steps (8 = 0.4 s at 20 Hz). The Cartesian targets are executed with the arm IK.

Conditions:
  nominal  held-out seeds 2_000_000+
  bump     same seeds; when the gripper first starts closing near the nut, the nut is pushed 3 cm sideways
           (along the pad face), so the first grasp closes on air and the policy has to notice and regrasp.

python eval_policy.py --ckpt <run>/checkpoints/latest.ckpt --n 50 --cond nominal --out results/x.json [--record 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TW = HERE.parent / "ext" / "touch_in_the_wild"
sys.path.insert(0, str(TW)); sys.path.insert(0, str(HERE))

import dill  # noqa: E402
import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from umi.real_world.real_inference_util import get_real_umi_action, get_real_umi_obs_dict  # noqa: E402

from sim import NUT_HALF_H, TABLE_TOP, TactileSim  # noqa: E402

KEYS = ("camera0_rgb", "camera0_tactile", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")


def load(ckpt):
    """Either a full training checkpoint (<run>/checkpoints/latest.ckpt) or a slim release file (models/tw/*.pt:
    the EMA policy weights in fp16 + the training config)."""
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    cfg.policy.obs_encoder.pretrain_ckpt_path = None             # weights come from the checkpoint itself
    ws = hydra.utils.get_class(cfg._target_)(cfg)
    if "ema_model_fp16" in payload:
        pol = ws.ema_model if cfg.training.use_ema else ws.model
        pol.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in payload["ema_model_fp16"].items()})
    else:
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        pol = ws.ema_model if cfg.training.use_ema else ws.model
    pol.num_inference_steps = 16
    return pol.eval().cuda(), cfg


def run(pol, cfg, env, seed, cond, n_act=8, record=False, max_steps=320):
    o = env.reset(seed); o = env.observe()
    hist = [o, o]
    start = [np.concatenate([o["robot0_eef_pos"], o["robot0_eef_rot_axis_angle"]])]
    bumped, t, frames, info = False, 0, [], {}
    pol.reset()
    while t < max_steps:
        obs = {k: np.stack([h[k] for h in hist[-2:]]) for k in KEYS}
        with torch.no_grad():
            od = get_real_umi_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta, obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
                                       tx_robot1_robot0=None, episode_start_pose=start)
            od = dict_apply(od, lambda x: torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0).cuda())
            raw = pol.predict_action(od)["action_pred"][0].float().cpu().numpy()
        act = get_real_umi_action(raw, obs, cfg.task.pose_repr.action_pose_repr)
        for a in act[:n_act]:
            if cond == "bump" and not bumped and env.width_to_g(a[6]) > 0.15 and env.width() > 0.03:
                p = env.nut_pos()
                if np.linalg.norm(p[:2] - env.eef_pose()[0][:2]) < 0.02:       # closing around the nut
                    R = env.eef_pose()[1]
                    d = R[:, 1].copy(); d[2] = 0; d /= np.linalg.norm(d) + 1e-9      # along the pad face
                    env.d.qpos[env.nut_q:env.nut_q + 2] += 0.03 * d[:2]
                    import mujoco
                    mujoco.mj_forward(env.m, env.d); bumped = True
            _, done, info = env.step_pose(a[:3], a[3:6], a[6])
            o = env.observe(); hist.append(o); t += 1
            if record:
                frames.append(dict(rgb=o["camera0_rgb"], tact=o["camera0_tactile"], qpos=env.d.qpos.astype(np.float32).copy(),
                                   grip=env.width_to_g(a[6])))
            if done:
                break
        if (info and info.get("success")) or t >= 300:
            break
    p = env.nut_pos()
    res = dict(seed=seed, cond=cond, success=bool(info.get("success", False)), steps=t, bumped=bumped,
               xy_err=float(info.get("xy_err", np.nan)), tilt=float(info.get("tilt", np.nan)),
               nut_z=float(p[2] - TABLE_TOP - NUT_HALF_H))
    return res, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cond", default="nominal", choices=["nominal", "bump"]); ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--out", required=True); ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--record_dir", default=str(HERE / "rollouts"))
    a = ap.parse_args()
    pol, cfg = load(a.ckpt)
    env = TactileSim()
    out, t0 = [], time.time()
    for k in range(a.n):
        r, frames = run(pol, cfg, env, a.seed0 + k, a.cond, record=k < a.record)
        out.append(r); print(json.dumps(r), flush=True)
        if frames:
            d = Path(a.record_dir); d.mkdir(parents=True, exist_ok=True)
            tag = Path(a.out).stem
            np.savez_compressed(d / f"{tag}_{r['seed']}.npz", rgb=np.stack([f["rgb"] for f in frames]),
                                tact=np.stack([f["tact"] for f in frames]), qpos=np.stack([f["qpos"] for f in frames]),
                                grip=np.array([f["grip"] for f in frames]), target=env.target_pos(), meta=json.dumps(r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(json.dumps({"ckpt": a.ckpt, "cond": a.cond, "success_rate": float(np.mean([r["success"] for r in out])),
                      "n": len(out), "time": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
