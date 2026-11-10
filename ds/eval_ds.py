#!/usr/bin/env python3
"""Evaluate a DexScrew teacher (stage 1, privileged) or student (stage 2, proprio + touch history) policy on the
same held-out seeds and the same success test as the Touch-in-the-Wild evaluation (tw/eval_policy.py):

  success = the base env's check (nut within 2 cm of the target, resting upright, gripper not touching it) held for
  10 consecutive control steps within 300 steps.  The RL env's own 5-step 'done' is ignored; only its hard
  failure (nut tipped over / off the table) ends an episode early.

Conditions: nominal (default physics), bump (nut pushed 3 cm along the pad face when the gripper first closes on
it, as in the TW eval), shifted (the 'real' stand-in physics: heavier, slipperier nut, weaker arm gains, 2-step
latency; nut_rl_env.SHIFTED).

  python eval_ds.py --ckpt .../stage1_nn/best.pth --kind teacher --cond nominal --n 50 --out results/x.json
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ext" / "dexscrew"))


def build_policy(ckpt, kind):
    import torch
    from dexscrew.algo.models.models import ActorCritic
    from dexscrew.algo.models.running_mean_std import RunningMeanStd
    from nut_rl_env import ACT_DIM, OBS_DIM, PRIV_DIM, PROP_DIM, PROP_LEN
    student = kind == "student"
    net = ActorCritic(dict(actor_units=[512, 256, 128], priv_mlp_units=[256, 128, 8], actions_num=ACT_DIM,
                           input_shape=(OBS_DIM,), priv_info=True, proprio_adapt=student, priv_info_dim=PRIV_DIM,
                           point_mlp_units=[32, 32, 32], use_point_cloud_info=False, proprio_len=PROP_LEN))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"]); net.eval()
    rms = RunningMeanStd((OBS_DIM,)); rms.load_state_dict(ck["running_mean_std"]); rms.eval()
    prms = RunningMeanStd(PRIV_DIM); prms.load_state_dict(ck["priv_mean_std"]); prms.eval()
    sa = None
    if student:
        sa = RunningMeanStd((PROP_LEN, PROP_DIM)); sa.load_state_dict(ck["sa_mean_std"]); sa.eval()

    @torch.no_grad()
    def act(obs, priv, prop):
        t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None]
        d = dict(obs=rms(t(obs)), point_cloud_info=torch.zeros(1, 1, 3))
        if student:
            d["proprio_hist"] = sa(t(prop))
        else:
            d["priv_info"] = prms(t(priv)); d["proprio_hist"] = t(prop)
        mu = net.act_inference(d)[0][0]
        return mu.clamp(-1, 1).numpy()
    return act


def run_episode(env, act, seed, cond, record=False, max_steps=300):
    o = env.reset(seed)
    env.bump = cond == "bump"
    frames, info, t, fail = [], {}, 0, False
    st = env.sim.state
    while t < max_steps:
        a = act(*o)
        o, r, done, inf = env.step(a)
        t += 1
        info = env.sim.state.info
        if record:
            frames.append(dict(qpos=env.d.qpos.astype(np.float32).copy(), tact=env.tact.copy(), grip=env.g))
        if inf["fail"]:
            fail = True; break
        if env.sim.state.success_streak >= 10:
            break
    st = env.sim.state
    return dict(seed=seed, cond=cond, success=bool(st.success_streak >= 10), fail=fail, steps=t, bumped=env.bumped,
                xy_err=float(info.get("xy_err", np.nan)), tilt=float(info.get("tilt", np.nan)),
                latent_err=None), frames


def _worker(args):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    ckpt, kind, cond, seeds, record_dir, n_record, tag, seed0 = args
    from nut_rl_env import SHIFTED, NutRLEnv
    env = NutRLEnv(seed=0, randomize=False, domain=SHIFTED if cond == "shifted" else None)
    act = build_policy(ckpt, kind)
    out = []
    for s in seeds:
        rec = record_dir is not None and s - seed0 < n_record
        r, frames = run_episode(env, act, s, cond, record=rec)
        out.append(r)
        if rec:
            d = Path(record_dir); d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{tag}_{cond}_{s}.npz", qpos=np.stack([f["qpos"] for f in frames]),
                                tact=np.stack([f["tact"] for f in frames]), grip=np.array([f["grip"] for f in frames]),
                                success=r["success"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--kind", choices=["teacher", "student"], required=True)
    ap.add_argument("--cond", default="nominal", choices=["nominal", "bump", "shifted"])
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--procs", type=int, default=25); ap.add_argument("--out", required=True)
    ap.add_argument("--record", type=int, default=0); ap.add_argument("--record_dir", default=None)
    ap.add_argument("--tag", default="ds")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    chunks = [seeds[i::a.procs] for i in range(a.procs) if seeds[i::a.procs]]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        res = [r for part in pool.map(_worker, [(a.ckpt, a.kind, a.cond, c, a.record_dir, a.record, a.tag, a.seed0)
                                               for c in chunks]) for r in part]
    res.sort(key=lambda r: r["seed"])
    sr = float(np.mean([r["success"] for r in res]))
    summary = dict(ckpt=a.ckpt, kind=a.kind, cond=a.cond, n=len(res), success_rate=sr,
                   fail_rate=float(np.mean([r["fail"] for r in res])),
                   mean_steps_success=float(np.mean([r["steps"] for r in res if r["success"]])) if sr > 0 else None,
                   episodes=res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
