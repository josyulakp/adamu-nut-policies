#!/usr/bin/env python3
"""Collect image demonstrations in one domain (keeps successful episodes only).

python cotrain/collect.py --domain sim  --n 300 --out cotrain/data/sim.npz
python cotrain/collect.py --domain real --n 100 --out cotrain/data/real.npz     # real-robot stand-in

Stored per step: images uint8 (2, H, W, 3) [head, wrist], proprio (8), action (8). Seeds: sim from 0,
real from 1_000_000 (evaluation uses 2_000_000+, never seen in training).
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

SEED0 = {"sim": 0, "real": 1_000_000}


def worker(job):
    domain, seeds, need = job
    os.environ.setdefault("MUJOCO_GL", "egl")
    from cam_env import CamEnv, make_expert
    env = CamEnv(domain)
    eps, tried = [], 0
    for s in seeds:
        if len(eps) >= need:
            break
        o = env.reset(s)
        ex = make_expert(env, np.random.default_rng(s + 10_000))
        I, P, A = [], [], []
        while True:
            a = ex.act()
            I.append(o["images"]); P.append(o["proprio"]); A.append(a.astype(np.float32))
            o, done, info = env.step(a)
            if done:
                break
        tried += 1
        if info["success"]:
            eps.append((s, np.stack(I), np.stack(P), np.stack(A)))
    return eps, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["sim", "real"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    per = int(np.ceil(a.n / a.workers))
    base = SEED0[a.domain]
    jobs = [(a.domain, range(base + w * 10_000, base + (w + 1) * 10_000), per) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        res = pool.map(worker, jobs)
    eps = [e for r, _ in res for e in r][:a.n]
    tried = sum(t for _, t in res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, images=np.concatenate([e[1] for e in eps]), proprio=np.concatenate([e[2] for e in eps]),
             action=np.concatenate([e[3] for e in eps]), ep_len=np.array([len(e[3]) for e in eps]),
             seeds=np.array([e[0] for e in eps]), tried=tried)
    L = np.array([len(e[3]) for e in eps])
    print(f"{a.domain}: kept {len(eps)} (tried {tried}), len {L.mean():.0f}, {L.sum()} frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
