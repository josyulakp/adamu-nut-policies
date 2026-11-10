"""Multiprocess MuJoCo vector env with the IsaacGym-style interface DexScrew's PPO / ProprioAdapt expect:
reset() -> obs_dict; step(actions) -> obs_dict, rewards, dones, infos (auto-reset; infos['time_outs']).
obs_dict: obs (N, 34), priv_info (N, 10), proprio_hist (N, 30, 24), point_cloud_info (N, 1, 3) (unused)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch


class _Box:
    def __init__(self, low, high, shape):
        self.shape, self.low, self.high = shape, np.full(shape, low, np.float32), np.full(shape, high, np.float32)

from nut_rl_env import ACT_DIM, OBS_DIM, PRIV_DIM, PROP_DIM, PROP_LEN


def _worker(remote, n, seed, kw):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    from nut_rl_env import NutRLEnv
    envs = [NutRLEnv(seed=seed + i, **kw) for i in range(n)]
    rng = np.random.default_rng(seed)

    def pack(obs):
        return [np.stack(x) for x in zip(*obs)]
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            seeds = arg
            remote.send(pack([e.reset(None if seeds is None else seeds[i]) for i, e in enumerate(envs)]))
        elif cmd == "step":
            out, R, D, TO, S, F = [], [], [], [], [], []
            for e, a in zip(envs, arg):
                o, r, d, info = e.step(a)
                if d:
                    o = e.reset(int(rng.integers(1 << 30)))
                out.append(o); R.append(r); D.append(d); TO.append(info["timeout"]); S.append(info["success"]); F.append(info["fail"])
            remote.send((pack(out), np.array(R, np.float32), np.array(D), np.array(TO), np.array(S), np.array(F)))
        elif cmd == "set":
            for e in envs:
                for k, v in arg.items():
                    setattr(e, k, v)
            remote.send(True)
        elif cmd == "close":
            return


class MjVecEnv:
    def __init__(self, n_procs=28, per_proc=8, seed=0, device="cuda:0", **kw):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n_procs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, per_proc, seed + 1000 * i, kw), daemon=True); p.start()
            self.remotes.append(a); self.procs.append(p)
        self.n_procs, self.per_proc, self.num_envs = n_procs, per_proc, n_procs * per_proc
        self.device = device
        self.observation_space = _Box(-np.inf, np.inf, (OBS_DIM,))
        self.action_space = _Box(-1.0, 1.0, (ACT_DIM,))
        self.priv_info_dim, self.prop_hist_len, self.point_cloud_buffer_dim = PRIV_DIM, PROP_LEN, 1
        self.with_camera, self.enable_camera_sensors = False, False
        self.stats = dict(episodes=0, success=0, fail=0)
        self.step_budget = None; self.steps = 0

    def _to(self, parts):
        obs, priv, prop = (np.concatenate([p[i] for p in parts]) for i in range(3))
        t = lambda x: torch.from_numpy(x).to(self.device)
        return dict(obs=t(obs), priv_info=t(priv), proprio_hist=t(prop),
                    point_cloud_info=torch.zeros(len(obs), 1, 3, device=self.device))

    def reset(self, seeds=None):
        for i, r in enumerate(self.remotes):
            r.send(("reset", None if seeds is None else seeds[i * self.per_proc:(i + 1) * self.per_proc]))
        return self._to([r.recv() for r in self.remotes])

    def step(self, actions, **_):
        if self.step_budget is not None and self.steps >= self.step_budget:
            raise StopIteration
        a = actions.detach().cpu().numpy()
        for i, r in enumerate(self.remotes):
            r.send(("step", a[i * self.per_proc:(i + 1) * self.per_proc]))
        res = [r.recv() for r in self.remotes]
        obs = self._to([x[0] for x in res])
        rew, done, to, succ, fail = (np.concatenate([x[i] for x in res]) for i in range(1, 6))
        self.steps += self.num_envs
        self.stats["episodes"] += int(done.sum()); self.stats["success"] += int(succ.sum()); self.stats["fail"] += int(fail.sum())
        t = lambda x, dt: torch.from_numpy(x.astype(dt)).to(self.device)
        infos = dict(time_outs=t(to, np.uint8), success_rate=float(succ.sum() / max(done.sum(), 1)))
        return obs, t(rew, np.float32), t(done, np.uint8), infos

    def set_attr(self, **kw):
        for r in self.remotes:
            r.send(("set", kw))
        [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
