#!/usr/bin/env python3
"""Evaluate a trained DSRL agent (runs/dsrl_*/agent.pt) on the 50 held-out seeds in any CamEnv domain.

  DSRL_DOMAIN=ideal python adamu_nut_yam/cotrain/eval_dsrl.py --run models/dsrl/dsrl_co_r50 --json results/x.json
DSRL_DOMAIN: real (default; the stand-in it was trained in), sim or ideal.
Uses the deterministic actor (the 'det' mode reported as the DSRL result).
"""
import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from dsrl import DSRL, Base, VecEnv, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--n_envs", type=int, default=10)
    ap.add_argument("--mode", default="det", choices=["det", "stoch", "base"])
    ap.add_argument("--base", default="", help="override the frozen Diffusion Policy checkpoint stored in agent.pt")
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / "agent.pt", map_location="cuda")
    args = Namespace(**ck["args"])
    if a.base:
        args.base = a.base
    base = Base(args.base, "cuda")
    agent = DSRL(base, args, "cuda")
    agent.actor.load_state_dict({k: v.float() for k, v in ck["actor"].items()})
    envs = VecEnv(a.n_envs, 0)
    res = evaluate(base, agent, envs, list(range(2_000_000, 2_000_000 + a.n)), a.mode)
    envs.close()
    sr = float(np.mean([r["success"] for r in res]))
    json.dump(res, open(a.json, "w"))
    print(json.dumps({"run": a.run, "domain": os.environ.get("DSRL_DOMAIN", "real"), "mode": a.mode, "success_rate": sr,
                      "n": len(res)}), flush=True)


if __name__ == "__main__":
    main()
