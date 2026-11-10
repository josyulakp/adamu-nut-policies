#!/usr/bin/env python3
"""Roll out a Diffusion Policy checkpoint and save (a) replay episodes for the web viewer and (b) an mp4 of what
the policy saw (head | wrist camera).

python adamu_nut_yam/cotrain/record.py models/dp/co_r50_a05.pt dp_co real 2000000,2000001
python adamu_nut_yam/cotrain/record.py models/dsrl/dsrl_co_r50 dsrl_co ideal 2000000      # a DSRL run directory
Domain: real (stand-in), sim or ideal. Writes rollouts/cotrain/<kind>_<domain>_<seed>.npz (joint state per step for
replay, gripper, nut position, success) and <kind>_<domain>_<seed>_cams.mp4 (head | wrist camera, what the policy saw).
ACT checkpoints are recognised by a file name starting with act_.
"""
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cam_env import CamEnv  # noqa: E402
from dp import TA, TO, DiffusionPolicy, augment  # noqa: E402
from act import ACT, CHUNK  # noqa: E402

OUT = HERE.parents[1] / "rollouts" / "cotrain"     # <kind>_<domain>_<seed>.npz (joint state replay) + .mp4 (cameras)
VID = OUT


def main():
    ckpt, kind, domain = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")]
    dev = "cuda"
    is_act = Path(ckpt).name.startswith("act_")
    dsrl = None
    if Path(ckpt).is_dir():          # a DSRL run directory: frozen base DP + latent noise policy (mean noise)
        from dsrl import Actor, Base
        ag = torch.load(Path(ckpt) / "agent.pt", map_location=dev)
        base = Base(ag["args"]["base"], dev)
        actor = Actor(ag["args"]["width"], ag["args"]["layers"]).to(dev)
        actor.load_state_dict({k: v.float() for k, v in ag["actor"].items()}); actor.eval()
        dsrl, pol = (base, actor, ag["args"]["action_magnitude"]), base.pol
    else:
        pol = (ACT() if is_act else DiffusionPolicy()).to(dev)
        pol.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); pol.eval()
    env = CamEnv(domain)
    gen = torch.Generator(device=dev)
    OUT.mkdir(exist_ok=True); VID.mkdir(exist_ok=True)
    for seed in seeds:
        gen.manual_seed(seed)
        o = env.reset(seed)
        hist, plan, buf, t = [o] * TO, [], [], 0
        frames, cams = [], [np.concatenate(list(o["images"]), 1)]

        def fr(phase, g):
            p, t = env.nut_pos(), env.target_pos()
            frames.append(dict(qpos=env.d.qpos.astype(np.float32).copy(), grip=g, phase=phase, grasped=env.grasped(),
                               xy=float(np.linalg.norm(p[:2] - t[:2])), z=float(p[2])))
        fr("start", 0.0)
        while True:
            if is_act:   # temporal ensembling, as in act.py evaluate
                img = torch.from_numpy(o["images"])[None, None].to(dev)
                prop = torch.from_numpy(o["proprio"])[None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    chunk = pol.act(augment(img, None, train=False)[:, 0], prop)[0].float().cpu().numpy()
                buf.append((t, chunk)); buf = [(t0, c) for t0, c in buf if t - t0 < CHUNK]
                preds = np.array([c[t - t0] for t0, c in buf]); w = np.exp(-0.01 * np.arange(len(preds)))
                plan = [(preds * w[:, None]).sum(0) / w.sum()]
                t += 1
            elif dsrl and not plan:
                base, actor, mag = dsrl
                s = base.encode([(np.stack([h["images"] for h in hist[-TO:]]), np.stack([h["proprio"] for h in hist[-TO:]]))])
                with torch.no_grad():
                    w = actor(s, deterministic=True)[0] * mag
                plan = list(base.unnorm(base.sample(s, w))[0, :TA].float().cpu().numpy())
            elif not plan:
                img = torch.from_numpy(np.stack([h["images"] for h in hist[-TO:]]))[None].to(dev)
                prop = torch.from_numpy(np.stack([h["proprio"] for h in hist[-TO:]]))[None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    plan = list(pol.act(augment(img, None, train=False), prop, gen)[0].float().cpu().numpy()[:TA])
            a = plan.pop(0)
            o, done, info = env.step(a)
            hist.append(o); cams.append(np.concatenate(list(o["images"]), 1))
            fr("policy", float(np.clip(a[7], 0, 1)))
            if done:
                break
        eid = f"{kind}_{domain}_{seed}"
        np.savez_compressed(OUT / f"{eid}.npz", qpos=np.array([f["qpos"] for f in frames]),
                            grip=np.array([f["grip"] for f in frames]), grasped=np.array([f["grasped"] for f in frames]),
                            xy=np.array([f["xy"] for f in frames]), z=np.array([f["z"] for f in frames]),
                            phase=np.array([f["phase"] for f in frames]), target=env.target_pos(),
                            meta=json.dumps({"id": eid, "kind": kind, "seed": seed, "success": bool(info["success"]),
                                             "xy_err": info["xy_err"], "tilt": info["tilt"], "domain": domain}))
        big = [np.kron(c, np.ones((3, 3, 1), np.uint8)) for c in cams]
        imageio.mimwrite(VID / f"{eid}_cams.mp4", big, fps=20, quality=7, macro_block_size=8)
        print(eid, domain, "success" if info["success"] else "FAIL", len(frames), f"xy {info['xy_err'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
