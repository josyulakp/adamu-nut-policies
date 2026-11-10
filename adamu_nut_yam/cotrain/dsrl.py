#!/usr/bin/env python3
"""DSRL (Diffusion Steering via RL, Wagenmaker et al., CoRL 2025; https://github.com/ajwagen/dsrl) on top of a
frozen image Diffusion Policy, fine-tuned online in the real-robot stand-in domain.

The diffusion policy's weights never change. A small RL policy instead picks the initial noise w (16 x 8, the
whole predicted chunk) that the deterministic DDIM sampler turns into an action chunk. Algorithm = DSRL-NA (the
variant the authors recommend), following stable_baselines3/dsrl/dsrl.py in their repo:
  * Q_A(s, a): SAC-style critic on the executed action chunk (8 x 8, normalised), target uses
    a' = DP(s', pi_w(s'))
  * Q_W(s, w): noise-space critic distilled from Q_A on (w ~ N(0, I), a = DP(s, w)) pairs ("noise aliasing")
  * pi_w(s): tanh-Gaussian over w in [-1.5, 1.5]^128, trained against Q_W with automatic entropy tuning
  * state s = the frozen DP observation encoding (2 obs x (2 cams x 256 + 8 proprio) = 1040-d)
Hyperparameters are the repo's robomimic defaults: 3 x 2048 LayerNorm MLPs, 2 critics with min backup, lr 3e-4,
batch 256, tau 0.005, gamma 0.99, UTD 20 per vector step, 10 noise-critic steps, target entropy 0, action
magnitude 1.5, except (see --init_alpha / --backup_entropy) the entropy term is left out of the critic target and
the temperature starts at 0.1 instead of 1. Reward is sparse: 1 when the episode ends in success (terminal), else 0; timeouts are truncations.
One RL step = one executed chunk of 8 control steps (0.4 s).

python cotrain/dsrl.py --base runs/co_r50_a05.pt --out runs/dsrl_co_r50 --episodes 1500
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp import K_INFER, K_TRAIN, TA, TO, TP, DiffusionPolicy, augment

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ADIM = 8
WDIM = TP * ADIM          # 128: noise action
QDIM = TA * ADIM          # 64: executed actions seen by Q_A
SDIM = TO * (2 * 256 + 8)


# ------------------------------------------------------------------------------------------- envs
def worker(remote, seed0, stride):
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    from cam_env import CamEnv
    env = CamEnv(os.environ.get("DSRL_DOMAIN", "real"))   # online training / evaluation domain
    seed, hist = seed0, None

    def pack():
        return (np.stack([h["images"] for h in hist[-TO:]]), np.stack([h["proprio"] for h in hist[-TO:]]))

    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            if arg is not None:
                seed = arg
            o = env.reset(seed); seed += stride
            hist = [o] * TO
            remote.send(pack())
        elif cmd == "chunk":                     # arg: (TA, 8) raw actions
            done, info = False, {}
            for a in arg:
                o, done, info = env.step(a)
                hist.append(o)
                if done:
                    break
            hist = hist[-TO:]
            remote.send((pack(), done, bool(info.get("success", False))))
        elif cmd == "close":
            return


class VecEnv:
    def __init__(self, n, seed0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, seed0 + i, n), daemon=True); p.start()
            self.remotes.append(a); self.procs.append(p)
        self.n = n

    def reset(self, idx=None, seeds=None):
        idx = range(self.n) if idx is None else idx
        for j, i in enumerate(idx):
            self.remotes[i].send(("reset", None if seeds is None else seeds[j]))
        return [self.remotes[i].recv() for i in idx]

    def chunk(self, acts, idx=None):
        idx = range(self.n) if idx is None else idx
        for i, a in zip(idx, acts):
            self.remotes[i].send(("chunk", a))
        return [self.remotes[i].recv() for i in idx]

    def close(self):
        for r in self.remotes:
            r.send(("close", None))


# --------------------------------------------------------------------------------------- base policy
def resolve(p):
    """A checkpoint path as given, or relative to this directory, or to the repository root (release models)."""
    for c in (Path(p), Path(__file__).resolve().parent / p, Path(__file__).resolve().parents[2] / p):
        if c.exists():
            return str(c)
    return str(p)


class Base:
    """Frozen diffusion policy: obs -> state features, (state, noise) -> normalised chunk via deterministic DDIM."""

    def __init__(self, ckpt, dev):
        ckpt = resolve(ckpt)
        self.pol = DiffusionPolicy().to(dev)
        self.pol.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); self.pol.eval()
        for p in self.pol.parameters():
            p.requires_grad_(False)
        self.dev = dev
        self.ts = torch.linspace(K_TRAIN - 1, 0, K_INFER, device=dev).round().long()
        acp = self.pol.acp
        self.abs = [acp[t] for t in self.ts]
        self.abs_prev = [acp[t] for t in self.ts[1:]] + [torch.ones((), device=dev)]
        self.tvec = [t.view(1) for t in self.ts]
        self.graphs = {}

    @torch.no_grad()
    def sample(self, cond, w):
        """cond (B, SDIM), w (B, WDIM) -> normalised chunk (B, TP, ADIM) in [-1, 1]. CUDA-graphed per batch size."""
        B = cond.shape[0]
        if B not in self.graphs:
            sc, sw = cond.detach().clone(), w.detach().clone()
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    self._sample(sc, sw)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._sample(sc, sw)
            self.graphs[B] = (g, sc, sw, out)
        g, sc, sw, out = self.graphs[B]
        sc.copy_(cond); sw.copy_(w); g.replay()
        return out.clone()

    @torch.no_grad()
    def encode(self, obs):
        img = torch.from_numpy(np.stack([o[0] for o in obs])).to(self.dev)
        prop = torch.from_numpy(np.stack([o[1] for o in obs])).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.pol.encode(augment(img, None, train=False), prop).float()

    @torch.no_grad()
    def _sample(self, cond, w):
        x = w.view(-1, TP, ADIM).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t, ab, ab_prev in zip(self.tvec, self.abs, self.abs_prev):
                eps = self.pol.unet(x, t.expand(x.shape[0]), cond).float()
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.clamp(-1, 1)

    def unnorm(self, x):
        return self.pol.unnorm_a(x)


# -------------------------------------------------------------------------------------------- nets
def mlp(i, o, width, layers):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, width), nn.LayerNorm(width), nn.ReLU()]; d = width
    return nn.Sequential(*mods, nn.Linear(d, o))


class Critic(nn.Module):
    def __init__(self, adim, width, layers, n=2):
        super().__init__()
        self.qs = nn.ModuleList([mlp(SDIM + adim, 1, width, layers) for _ in range(n)])

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return torch.cat([q(x) for q in self.qs], -1)       # (B, n)


class Actor(nn.Module):
    """Squashed Gaussian in the scaled space [-1, 1]^WDIM (env-side noise = action_magnitude * this)."""

    def __init__(self, width, layers):
        super().__init__()
        self.body = mlp(SDIM, 2 * WDIM, width, layers)

    def forward(self, s, deterministic=False):
        mu, log_std = self.body(s).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        if deterministic:
            return torch.tanh(mu), None
        u = mu + torch.randn_like(mu) * log_std.exp()
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


# ------------------------------------------------------------------------------------------- buffer
class Buffer:
    def __init__(self, cap, dev):
        self.s = torch.zeros(cap, SDIM, device=dev); self.s2 = torch.zeros(cap, SDIM, device=dev)
        self.a = torch.zeros(cap, QDIM, device=dev)
        self.r = torch.zeros(cap, 1, device=dev); self.d = torch.zeros(cap, 1, device=dev)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, s, a, r, s2, d):
        k = s.shape[0]
        idx = (torch.arange(k, device=s.device) + self.i) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d
        self.i = (self.i + k) % self.cap; self.n = min(self.n + k, self.cap)

    def sample(self, B):
        j = torch.randint(0, self.n, (B,), device=self.s.device)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


# ------------------------------------------------------------------------------------------- DSRL-NA
class DSRL:
    def __init__(self, base, a, dev):
        self.base, self.a, self.dev = base, a, dev
        self.actor = Actor(a.width, a.layers).to(dev)
        self.qa = Critic(QDIM, a.width, a.layers).to(dev)
        self.qa_t = Critic(QDIM, a.width, a.layers).to(dev); self.qa_t.load_state_dict(self.qa.state_dict())
        self.qw = Critic(WDIM, a.width, a.layers).to(dev)
        self.log_alpha = torch.full((1,), math.log(a.init_alpha), device=dev, requires_grad=True)
        self.o_actor = torch.optim.Adam(self.actor.parameters(), lr=a.lr)
        self.o_qa = torch.optim.Adam(self.qa.parameters(), lr=a.lr)
        self.o_qw = torch.optim.Adam(self.qw.parameters(), lr=a.lr)
        self.o_alpha = torch.optim.Adam([self.log_alpha], lr=a.lr)
        self.mag = a.action_magnitude

    def noise(self, s, deterministic=False):
        with torch.no_grad():
            w, _ = self.actor(s, deterministic)
        return w * self.mag

    def update(self, buf, steps, B):
        logs = []
        for _ in range(steps):
            s, a, r, s2, d = buf.sample(B)
            w_pi, logp = self.actor(s)
            alpha = self.log_alpha.exp().detach()
            l_alpha = -(self.log_alpha * (logp + self.a.target_entropy).detach()).mean()
            self.o_alpha.zero_grad(); l_alpha.backward(); self.o_alpha.step()
            with torch.no_grad():
                w2, logp2 = self.actor(s2)
                a2 = self.base.sample(s2, w2 * self.mag)[:, :TA].flatten(1)
                q2 = self.qa_t(s2, a2).min(-1, keepdim=True)[0]
                if self.a.backup_entropy:
                    q2 = q2 - alpha * logp2[:, None]
                y = r + (1 - d) * self.a.gamma * q2
            q = self.qa(s, a)
            l_q = 0.5 * ((q - y) ** 2).sum(-1).mean()
            self.o_qa.zero_grad(); l_q.backward(); self.o_qa.step()
            l_pi = (alpha * logp - self.qw(s, w_pi).min(-1)[0]).mean()
            self.o_actor.zero_grad(); l_pi.backward(); self.o_actor.step()
            with torch.no_grad():
                for p, pt in zip(self.qa.parameters(), self.qa_t.parameters()):
                    pt.mul_(1 - self.a.tau).add_(self.a.tau * p)
            logs.append((l_q.item(), l_pi.item(), alpha.item(), q.mean().item()))
        for _ in range(self.a.noise_critic_steps):
            s = buf.sample(B)[0]
            with torch.no_grad():
                w = torch.randn(B, WDIM, device=self.dev)
                tq = self.qa(s, self.base.sample(s, w)[:, :TA].flatten(1))
            l_w = 0.5 * ((self.qw(s, w / self.mag) - tq) ** 2).sum(-1).mean()
            self.o_qw.zero_grad(); l_w.backward(); self.o_qw.step()
        return np.mean(logs, 0).tolist() + [l_w.item()]

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "qa": self.qa.state_dict(), "qw": self.qw.state_dict(),
                "log_alpha": self.log_alpha.detach(), "args": vars(self.a)}


# ---------------------------------------------------------------------------------------------- loop
def evaluate(base, agent, envs, seeds, mode):
    """mode: 'base' (w ~ N(0, I)), 'det' (actor mean), 'stoch' (actor sample). Returns per-seed results."""
    res, n = [], envs.n
    for k in range(0, len(seeds), n):
        ss = seeds[k:k + n]; idx = list(range(len(ss)))
        obs = dict(zip(idx, envs.reset(idx, ss)))
        alive, out = set(idx), {}
        steps = {i: 0 for i in idx}
        while alive:
            ids = sorted(alive)
            s = base.encode([obs[i] for i in ids])
            if mode == "base":
                w = torch.randn(len(ids), WDIM, device=s.device)
            else:
                w = agent.noise(s, deterministic=mode == "det")
            acts = base.unnorm(base.sample(s, w))[:, :TA].cpu().numpy()
            for i, (o, done, succ) in zip(ids, envs.chunk(acts, ids)):
                obs[i] = o; steps[i] += 1
                if done or steps[i] >= 40:
                    out[i] = succ; alive.discard(i)
        res += [{"seed": ss[i], "success": out[i]} for i in idx]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/co_r50_a05.pt"); ap.add_argument("--out", default="runs/dsrl_co_r50")
    ap.add_argument("--episodes", type=int, default=1500, help="online training episodes in the real stand-in")
    ap.add_argument("--n_envs", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_episodes", type=int, default=48, help="episodes with the plain base policy first")
    ap.add_argument("--width", type=int, default=2048); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--utd", type=int, default=20); ap.add_argument("--noise_critic_steps", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--target_entropy", type=float, default=0.0)
    ap.add_argument("--action_magnitude", type=float, default=1.5)
    # Deviations from the repo defaults (entropy 1.0, entropy in the backup): with a 128-d noise space the SAC entropy
    # bonus (~100 nats per step) swamped the 0/1 reward (Q ~ 500, falling success), so the critic learns the task
    # return only and the temperature starts at 0.1 (still auto-tuned towards target entropy 0 for the actor).
    ap.add_argument("--init_alpha", type=float, default=0.1)
    ap.add_argument("--backup_entropy", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--eval_n", type=int, default=50)
    ap.add_argument("--max_chunks", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = Base(a.base, dev)
    agent = DSRL(base, a, dev)
    buf = Buffer(200_000, dev)
    envs = VecEnv(a.n_envs, 3_000_000 + 100_000 * a.seed)     # training seeds, disjoint from eval (2_000_000+)
    eval_seeds = list(range(2_000_000, 2_000_000 + a.eval_n))
    log = open(out / "log.jsonl", "a")

    def L(d):
        d["time"] = round(time.time() - t0); print(json.dumps(d), flush=True); log.write(json.dumps(d) + "\n"); log.flush()

    t0 = time.time()
    obs = envs.reset()
    s = base.encode(obs)
    chunks = [0] * a.n_envs
    n_ep, n_succ_recent, recent, n_chunks, next_eval, upd = 0, 0, [], 0, 0, None
    while n_ep < a.episodes:
        if n_ep >= next_eval:
            r = evaluate(base, agent, envs, eval_seeds, "det" if n_ep else "base")
            L({"eval": n_ep, "mode": "det" if n_ep else "base", "success": float(np.mean([x["success"] for x in r]))})
            torch.save(agent.state_dict(), out / "agent.pt")
            next_eval += a.eval_every
            obs = envs.reset(); s = base.encode(obs); chunks = [0] * a.n_envs
        if n_ep < a.init_episodes:
            w = torch.randn(a.n_envs, WDIM, device=dev)
        else:
            w = agent.noise(s)
        x = base.sample(s, w)
        res = envs.chunk(base.unnorm(x)[:, :TA].cpu().numpy())
        n_chunks += a.n_envs
        nobs = [o for o, _, _ in res]
        s2 = base.encode(nobs)
        r = torch.tensor([[float(done and succ)] for _, done, succ in res], device=dev)
        term = torch.tensor([[float(done and succ)] for _, done, succ in res], device=dev)
        buf.add(s, x[:, :TA].flatten(1), r, s2, term)
        # episode ends: success / env timeout / chunk cap -> reset those envs
        ended = []
        for i, (_, done, succ) in enumerate(res):
            chunks[i] += 1
            if done or chunks[i] >= a.max_chunks:
                ended.append(i); recent.append(succ); n_ep += 1; chunks[i] = 0
        if ended:
            for i, o in zip(ended, envs.reset(ended)):
                nobs[i] = o
            s2 = base.encode(nobs)
        s = s2
        if n_ep >= a.init_episodes and buf.n >= a.batch:
            upd = agent.update(buf, a.utd, a.batch)
        if ended and len(recent) >= 50 and n_ep % 25 < len(ended):
            L({"episodes": n_ep, "chunks": n_chunks, "train_success_last50": float(np.mean(recent[-50:])),
               "q_loss": upd and round(upd[0], 4), "pi_loss": upd and round(upd[1], 3), "alpha": upd and upd[2],
               "q_mean": upd and round(upd[3], 3), "qw_loss": upd and round(upd[4], 4)})
    r = evaluate(base, agent, envs, eval_seeds, "det")
    L({"eval": n_ep, "mode": "det", "success": float(np.mean([x["success"] for x in r]))})
    json.dump(r, open(out / "final_det.json", "w"))
    r2 = evaluate(base, agent, envs, eval_seeds, "stoch")
    L({"eval": n_ep, "mode": "stoch", "success": float(np.mean([x["success"] for x in r2]))})
    json.dump(r2, open(out / "final_stoch.json", "w"))
    torch.save(agent.state_dict(), out / "agent.pt")
    envs.close()


if __name__ == "__main__":
    main()
