#!/usr/bin/env python3
"""Collect expert demos in the MuJoCo FlexiTac scene and write a Touch-in-the-Wild / UMI replay buffer
(.zarr.zip, same keys and image codec as scripts_slam_pipeline/07_generate_replay_buffer.py).

Simulation gives exact poses, so the SLAM pipeline (GoPro + ORB-SLAM + ArUco gripper-width) is replaced by the
simulator's state; everything downstream (dataset class, MAE pretraining, policy training) is theirs unchanged.

python tw/collect.py --n 300 --out tw/data/nut_tactile.zarr.zip
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TW = HERE.parent / "ext" / "touch_in_the_wild"
sys.path.insert(0, str(TW))


POST_HOLD = 10


def worker(job):
    seeds, p_miss, p_bump = job
    from sim import RetryExpert, TactileSim
    env = TactileSim()
    eps = []
    for s in seeds:
        env.reset(s)
        ex = RetryExpert(env, np.random.default_rng(s + 10_000), p_miss=p_miss, p_bump=p_bump)
        keys = ("camera0_rgb", "camera0_tactile", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")
        E = {k: [] for k in keys + ("action",)}
        ok, post = False, 0
        while True:
            a = ex.act()
            o = env.observe()
            tgt, w = ex.target()
            for k in keys:
                E[k].append(o[k])
            E["action"].append(np.concatenate([tgt, [w]]).astype(np.float32))
            _, done, info = env.step(a)
            ok = ok or info["success"]
            # v2: keep recording through release + retreat and 10 hold steps after the plan ends (v1 stopped at the
            # first success, ~10 steps into the release, so the policy rarely saw what follows 'nut is down')
            if ex.done:
                post += 1
            if post >= POST_HOLD or env.state.t >= 299:
                break
        if not ok:
            continue
        E = {k: np.stack(v) for k, v in E.items()}
        start = np.concatenate([E["robot0_eef_pos"][0], E["robot0_eef_rot_axis_angle"][0]])
        end = np.concatenate([E["robot0_eef_pos"][-1], E["robot0_eef_rot_axis_angle"][-1]])
        E["robot0_demo_start_pose"] = np.repeat(start[None], len(E["action"]), 0)
        E["robot0_demo_end_pose"] = np.repeat(end[None], len(E["action"]), 0)
        eps.append((s, bool(ex.missed or ex.bumped), E))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--p_miss", type=float, default=0.3)
    ap.add_argument("--p_bump", type=float, default=0.2)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(HERE / "data" / "nut_tactile.zarr.zip"))
    a = ap.parse_args()
    import zarr
    from diffusion_policy.codecs.imagecodecs_numcodecs import JpegXl, register_codecs
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    register_codecs()
    t0 = time.time()
    seeds = np.arange(a.seed0, a.seed0 + a.n)
    jobs = [(list(map(int, c)), a.p_miss, a.p_bump) for c in np.array_split(seeds, a.workers)]
    with Pool(a.workers) as pool:
        res = pool.map(worker, jobs)
    eps = sorted([e for r in res for e in r], key=lambda e: e[0])
    print(f"collected {len(eps)}/{a.n} successful episodes ({sum(m for _, m, _ in eps)} with a regrasp), "
          f"{time.time() - t0:.0f}s", flush=True)
    rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    img_comp = JpegXl(level=99, numthreads=8)
    for s, missed, E in eps:
        chunks = {"camera0_rgb": (1, 224, 224, 3)}
        rb.add_episode(E, chunks=chunks, compressors={"camera0_rgb": img_comp})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with zarr.ZipStore(a.out, mode="w") as zs:
        rb.save_to_store(zs)
    meta = dict(n_episodes=len(eps), seeds=[int(s) for s, _, _ in eps], regrasp=[bool(m) for _, m, _ in eps],
                lengths=[int(len(E["action"])) for _, _, E in eps], p_miss=a.p_miss, hz=20)
    Path(a.out + ".json").write_text(json.dumps(meta))
    print(f"wrote {a.out}: {rb.n_steps} frames, {Path(a.out).stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
