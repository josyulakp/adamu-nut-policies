#!/usr/bin/env python3
"""DexScrew stage-1 teacher, warm-started by imitation (DAgger) of the scripted controller (heuristic.py).

Three PPO-from-scratch runs failed on this task by exploiting the dense shaping terms (hover; stall 6 cm off target;
farm the reach/align bonus without ever grasping), so the privileged teacher is first trained to imitate the hand-written
controller (8/10 on randomised physics) and can then be fine-tuned with PPO (train_ds.py teacher --init).

  round 0: the expert acts, with DART noise on the executed action; labels are the clean expert action
  round k: the current policy acts (beta-mixed with the expert); the expert labels every visited state (DAgger)
The network is DexScrew's ActorCritic exactly as stage 1 builds it (privileged MLP -> 8-d latent -> actor MLP), and the
checkpoint has the keys PPO.save writes, so train_ds.py student / eval_ds.py load it unchanged. Physics is randomised as in
PPO training, 30% of episodes have the bump; seeds 5_000_000+ (evaluation uses 2_000_000+).

  python ds/bc_teacher.py --out ds/runs/teacher_bc

Result (2026-11-10, eval_ds.py, 50 held-out seeds): round 0 (DART BC only) 50/50 nominal, 50/50 bump, 9/50 shifted;
the DAgger rounds got worse (round 5: 34/50 nominal, 1/50 bump), presumably because the expert's labels depend on its
hidden phase counters, which become inconsistent on policy-visited states. Round 0 is used as the stage-1 teacher.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ext" / "dexscrew"))


def _rollouts(args):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    seeds, ckpt, beta, noise = args
    from heuristic import heuristic
    from nut_rl_env import NutRLEnv
    env = NutRLEnv(seed=0, randomize=True)
    act = None
    if ckpt:
        from eval_ds import build_policy
        act = build_policy(ckpt, "teacher")
    O, PR, PH, A, stats = [], [], [], [], []
    for s in seeds:
        o = env.reset(s); env._hs = {"phase": "reach", "k": 0}
        rng = np.random.default_rng(s + 7)
        env.bump = rng.random() < 0.3                         # include bump-recovery episodes
        use_pol = act is not None and rng.random() > beta        # per-episode mixture
        ok = False
        for t in range(300):
            a_exp = heuristic(env).astype(np.float32)
            O.append(o[0]); PR.append(o[1]); PH.append(o[2]); A.append(a_exp)
            a = act(*o) if use_pol else a_exp.copy()
            if noise > 0:
                a[:4] = a[:4] + rng.normal(0, noise, 4)
            o, r, done, inf = env.step(np.clip(a, -1, 1))
            if inf["fail"]:
                break
            if env.sim.state.success_streak >= 10:
                ok = True; break
        stats.append((use_pol, ok))
    return np.array(O), np.array(PR), np.array(PH), np.array(A), stats


def set_rms(rms, x):
    import torch
    x = torch.as_tensor(x, dtype=torch.float64)
    rms.running_mean.copy_(x.mean(0)); rms.running_var.copy_(x.var(0) + 1e-6); rms.count.fill_(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1120); ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--noise", type=float, default=0.2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    from dexscrew.algo.models.models import ActorCritic
    from dexscrew.algo.models.running_mean_std import RunningMeanStd
    from nut_rl_env import ACT_DIM, OBS_DIM, PRIV_DIM, PROP_LEN
    net = ActorCritic(dict(actor_units=[512, 256, 128], priv_mlp_units=[256, 128, 8], actions_num=ACT_DIM,
                           input_shape=(OBS_DIM,), priv_info=True, proprio_adapt=False, priv_info_dim=PRIV_DIM,
                           point_mlp_units=[32, 32, 32], use_point_cloud_info=False, proprio_len=PROP_LEN)).cuda()
    rms, prms = RunningMeanStd((OBS_DIM,)).cuda(), RunningMeanStd(PRIV_DIM).cuda()
    vms = RunningMeanStd((1,)).cuda()
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith(("value", "sigma"))], lr=1e-3)
    data, log, ckpt, seed0 = None, [], "", 5_000_000
    for rnd in range(a.rounds):
        beta = 1.0 if rnd == 0 else 0.5 ** rnd
        seeds = list(range(seed0, seed0 + a.episodes)); seed0 += a.episodes
        chunks = [seeds[i::a.procs] for i in range(a.procs)]
        with mp.get_context("spawn").Pool(a.procs) as pool:
            parts = pool.map(_rollouts, [(c, ckpt, beta, a.noise if rnd == 0 else 0.05) for c in chunks])
        new = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        st = [s for p in parts for s in p[4]]
        pol = [ok for u, ok in st if u]; exp = [ok for u, ok in st if not u]
        rec = dict(round=rnd, beta=beta, frames=len(new[0]), expert_eps=len(exp), expert_succ=float(np.mean(exp)) if exp else None,
                   policy_eps=len(pol), policy_succ=float(np.mean(pol)) if pol else None)
        data = new if data is None else [np.concatenate([d, n]) for d, n in zip(data, new)]
        O, PR, A = (torch.from_numpy(x).float().cuda() for x in (data[0], data[1], data[3]))
        set_rms(rms, data[0]); set_rms(prms, data[1]); rms.eval(); prms.eval()
        n = len(O); idx = np.random.default_rng(rnd).permutation(n); nv = n // 20; va, tr = idx[:nv], idx[nv:]
        zeros = torch.zeros(1, 1, 3, device="cuda")

        def loss_on(ii):
            mu = net._actor_critic(dict(obs=rms(O[ii]), priv_info=prms(PR[ii]), point_cloud_info=zeros.expand(len(ii), 1, 3)))[0]
            return ((mu - A[ii]) ** 2).mean()
        net.train()
        for ep in range(a.epochs):
            np.random.shuffle(tr)
            for k in range(0, len(tr), 1024):
                l = loss_on(tr[k:k + 1024]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            rec["val_mse"] = float(np.mean([loss_on(va[k:k + 4096]).item() for k in range(0, len(va), 4096)]))
        with torch.no_grad():
            net.sigma.fill_(np.log(0.1))          # small exploration noise for a later PPO fine-tune
        ckpt = str(out / f"bc_round{rnd}.pth")
        torch.save(dict(model=net.state_dict(), running_mean_std=rms.state_dict(), priv_mean_std=prms.state_dict(),
                        value_mean_std=vms.state_dict()), ckpt)
        log.append(rec); print(rec, flush=True)
        (out / "bc_log.json").write_text(json.dumps(log, indent=1))
    Path(out / "bc_final.pth").write_bytes(Path(ckpt).read_bytes())


if __name__ == "__main__":
    main()
