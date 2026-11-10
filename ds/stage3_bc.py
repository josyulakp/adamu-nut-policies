#!/usr/bin/env python3
"""DexScrew stage 3 analogue: tactile behaviour cloning on data gathered in the 'real' domain with the learned skill.

In DexScrew the stage-2 skill assists a human teleoperator on the real robot and a tactile policy is then trained by
behaviour cloning on those demonstrations. Here the real robot is the SHIFTED physics stand-in (heavier, slipperier
nut, weaker arm gains, 2 control steps of latency) and there is no teleoperator, so the data come from the stage-2
student itself plus exploration noise, keeping only successful episodes (success-filtered BC). The fine-tuned
network is the student's (actor MLP + temporal conv over the 30-step proprio/touch history); it is trained to
reproduce the executed actions, with the observation normalisers frozen. Seeds 3_000_000+ (evaluation uses 2_000_000+).

  python stage3_bc.py --student .../stage2_nn/model_last.ckpt --episodes 400 --out .../stage3
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ext" / "dexscrew"))


def _collect(args):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    ckpt, seeds, noise = args
    from eval_ds import build_policy
    from nut_rl_env import SHIFTED, NutRLEnv
    env = NutRLEnv(seed=0, randomize=False, domain=SHIFTED)
    act = build_policy(ckpt, "student")
    out = []
    for s in seeds:
        o = env.reset(s); rng = np.random.default_rng(s)
        O, P, A = [], [], []
        ok = False
        for t in range(300):
            a = np.clip(act(*o) + rng.normal(0, noise, 5), -1, 1).astype(np.float32)
            O.append(o[0]); P.append(o[2]); A.append(a)
            o, r, done, inf = env.step(a)
            if inf["fail"]:
                break
            if env.sim.state.success_streak >= 10:
                ok = True; break
        out.append((s, ok, np.array(O), np.array(P), np.array(A)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=400); ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--keep_all", action="store_true", help="smoke test only: clone failed episodes too")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(3_000_000, 3_000_000 + a.episodes))
    chunks = [seeds[i::a.procs] for i in range(a.procs)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        eps = [e for part in pool.map(_collect, [(a.student, c, a.noise) for c in chunks]) for e in part]
    good = [e for e in eps if e[1] or a.keep_all]
    print(f"collected {len(eps)} episodes in the shifted domain, {len(good)} successful", flush=True)
    if not good:
        sys.exit("no successful episodes to clone; stage 3 needs a student that succeeds sometimes in the shifted domain")
    O = np.concatenate([e[2] for e in good]); P = np.concatenate([e[3] for e in good]); A = np.concatenate([e[4] for e in good])

    import torch
    from dexscrew.algo.models.models import ActorCritic
    from dexscrew.algo.models.running_mean_std import RunningMeanStd
    from nut_rl_env import ACT_DIM, OBS_DIM, PRIV_DIM, PROP_DIM, PROP_LEN
    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    net = ActorCritic(dict(actor_units=[512, 256, 128], priv_mlp_units=[256, 128, 8], actions_num=ACT_DIM,
                           input_shape=(OBS_DIM,), priv_info=True, proprio_adapt=True, priv_info_dim=PRIV_DIM,
                           point_mlp_units=[32, 32, 32], use_point_cloud_info=False, proprio_len=PROP_LEN)).cuda()
    net.load_state_dict(ck["model"])
    rms = RunningMeanStd((OBS_DIM,)).cuda(); rms.load_state_dict(ck["running_mean_std"]); rms.eval()
    sa = RunningMeanStd((PROP_LEN, PROP_DIM)).cuda(); sa.load_state_dict(ck["sa_mean_std"]); sa.eval()
    params = [p for n, p in net.named_parameters() if n.startswith(("actor_mlp", "mu", "adapt_tconv"))]
    opt = torch.optim.Adam(params, lr=a.lr)
    O, P, A = (torch.from_numpy(x).float().cuda() for x in (O, P, A))
    n = len(O); idx = np.random.default_rng(0).permutation(n); nv = max(1, n // 10)
    va, tr = idx[:nv], idx[nv:]
    log = []

    def loss_on(ii):
        with torch.no_grad():
            o, p = rms(O[ii]), sa(P[ii])
        mu = net._actor_critic(dict(obs=o, proprio_hist=p, point_cloud_info=torch.zeros(len(ii), 1, 3, device="cuda")))[0]
        return ((mu.clamp(-1, 1) - A[ii]) ** 2).mean()
    net.train()
    for ep in range(a.epochs):
        np.random.shuffle(tr); tl = []
        for k in range(0, len(tr), 512):
            l = loss_on(tr[k:k + 512]); opt.zero_grad(); l.backward(); opt.step(); tl.append(l.item())
        with torch.no_grad():
            vl = float(np.mean([loss_on(va[k:k + 2048]).item() for k in range(0, len(va), 2048)]))
        log.append(dict(epoch=ep, train=float(np.mean(tl)), val=vl)); print(log[-1], flush=True)
    net.eval()
    ck2 = dict(ck); ck2["model"] = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save(ck2, out / "model_ft.ckpt")
    (out / "stage3.json").write_text(json.dumps(dict(episodes=len(eps), successful=len(good), frames=int(n), noise=a.noise,
                                                     log=log), indent=1))


if __name__ == "__main__":
    main()
