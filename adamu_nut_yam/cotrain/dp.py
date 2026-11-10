#!/usr/bin/env python3
"""Image-based Diffusion Policy (Chi et al. 2023) with sim-and-real co-training (Maddukuri et al. 2025).

Observation: head + wrist RGB (112 px, random 104 px crop) and proprio (7 joints + grip) for the last 2 steps.
Vision: one ResNet-18 per camera (GroupNorm instead of BatchNorm, trained from scratch), global average pool.
Action head: 1-D temporal U-Net with FiLM conditioning, epsilon prediction, 100-step cosine DDPM for training,
16-step DDIM at run time. Predicts 16 actions, executes 8, then re-plans.
Co-training: every batch contains round(alpha * B) samples from the real dataset and the rest from sim;
alpha = 1 is real-only, alpha = 0 is sim-only.

python cotrain/dp.py train --sim data/sim.npz --real data/real.npz --n_real 50 --alpha 0.5 --out runs/x.pt
python cotrain/dp.py eval --ckpt runs/x.pt --domain real --episodes 50 --json runs/x_real.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

TO, TP, TA = 2, 16, 8          # obs horizon, prediction horizon, executed actions
CROP = 104
K_TRAIN, K_INFER = 100, 16


# ---------------------------------------------------------------------------------------------- networks
def resnet18_gn(out=256):
    net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
    net.fc = nn.Linear(512, out)
    return net


class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        x = t.float()[:, None] * f[None]
        return torch.cat([x.sin(), x.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond, k=5, g=8):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(g, cout), nn.Mish())
        self.b2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(g, cout), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2 * cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = self.b2(h * (1 + s) + b)
        return h + self.skip(x)


class CondUnet1D(nn.Module):
    def __init__(self, adim, cond_dim, dims=(128, 256, 512), temb=128):
        super().__init__()
        self.temb = nn.Sequential(SinEmb(temb), nn.Linear(temb, temb * 4), nn.Mish(), nn.Linear(temb * 4, temb))
        c = cond_dim + temb
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        chans = [adim, *dims]
        for i in range(len(dims)):
            last = i == len(dims) - 1
            self.down.append(nn.ModuleList([ResBlock(chans[i], chans[i + 1], c), ResBlock(chans[i + 1], chans[i + 1], c),
                                            nn.Conv1d(chans[i + 1], chans[i + 1], 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([ResBlock(dims[-1], dims[-1], c), ResBlock(dims[-1], dims[-1], c)])
        for i in reversed(range(1, len(dims))):
            self.up.append(nn.ModuleList([ResBlock(dims[i] * 2, dims[i - 1], c), ResBlock(dims[i - 1], dims[i - 1], c),
                                          nn.ConvTranspose1d(dims[i - 1], dims[i - 1], 4, 2, 1)]))
        self.final = nn.Sequential(nn.Conv1d(dims[0], dims[0], 5, padding=2), nn.GroupNorm(8, dims[0]), nn.Mish(),
                                   nn.Conv1d(dims[0], adim, 1))

    def forward(self, x, t, cond):          # x (B, T, A)
        c = torch.cat([cond, self.temb(t)], -1)
        x = x.transpose(1, 2)
        hs = []
        for r1, r2, dn in self.down:
            x = r2(r1(x, c), c); hs.append(x); x = dn(x)
        for m in self.mid:
            x = m(x, c)
        for r1, r2, upm in self.up:
            x = torch.cat([x, hs.pop()], 1)
            x = upm(r2(r1(x, c), c))
        return self.final(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, pdim=8, adim=8, ncam=2, feat=256):
        super().__init__()
        self.enc = nn.ModuleList([resnet18_gn(feat) for _ in range(ncam)])
        cond = TO * (ncam * feat + pdim)
        self.unet = CondUnet1D(adim, cond)
        for k in ("p_lo", "p_hi", "a_lo", "a_hi"):
            self.register_buffer(k, torch.zeros(pdim if k[0] == "p" else adim))
        betas = cosine_betas(K_TRAIN)
        self.register_buffer("acp", torch.cumprod(1 - betas, 0))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def norm_p(self, p): return 2 * (p - self.p_lo) / (self.p_hi - self.p_lo) - 1
    def norm_a(self, a): return 2 * (a - self.a_lo) / (self.a_hi - self.a_lo) - 1
    def unnorm_a(self, a): return (a + 1) / 2 * (self.a_hi - self.a_lo) + self.a_lo

    def encode(self, imgs, prop):
        """imgs float (B, TO, ncam, 3, H, W) in [0,1], prop (B, TO, pdim) raw."""
        B = imgs.shape[0]
        feats = []
        for c, enc in enumerate(self.enc):
            x = imgs[:, :, c].flatten(0, 1)
            feats.append(enc((x - self.img_mean) / self.img_std).view(B, TO, -1))
        return torch.cat([*feats, self.norm_p(prop)], -1).flatten(1)

    def loss(self, imgs, prop, act):
        cond = self.encode(imgs, prop)
        a0 = self.norm_a(act)
        t = torch.randint(0, K_TRAIN, (a0.shape[0],), device=a0.device)
        eps = torch.randn_like(a0)
        ab = self.acp[t].view(-1, 1, 1)
        return F.mse_loss(self.unet(ab.sqrt() * a0 + (1 - ab).sqrt() * eps, t, cond), eps)

    @torch.no_grad()
    def act(self, imgs, prop, gen=None):
        cond = self.encode(imgs, prop)
        x = torch.randn((imgs.shape[0], TP, self.a_lo.numel()), device=imgs.device, generator=gen)
        ts = torch.linspace(K_TRAIN - 1, 0, K_INFER, device=imgs.device).round().long()
        for i, t in enumerate(ts):
            ab = self.acp[t]
            ab_prev = self.acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            eps = self.unet(x, t.expand(x.shape[0]), cond)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return self.unnorm_a(x)


def cosine_betas(K, s=0.008):
    t = torch.linspace(0, K, K + 1) / K
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return torch.clamp(1 - f[1:] / f[:-1], max=0.999)


# ---------------------------------------------------------------------------------------------- data
class Demos:
    """All frames of the first n episodes of one npz, on the GPU (uint8 images)."""

    def __init__(self, path, n, dev):
        z = np.load(path)
        L = z["ep_len"][:n]
        T = int(L.sum())
        self.n_eps, self.T = len(L), T
        self.img = torch.from_numpy(z["images"][:T]).to(dev)              # (T, cam, H, W, 3) uint8
        self.prop = torch.from_numpy(z["proprio"][:T]).to(dev)
        self.act = torch.from_numpy(z["action"][:T]).to(dev)
        start = np.repeat(np.cumsum(np.r_[0, L[:-1]]), L)
        end = np.repeat(np.cumsum(L) - 1, L)
        self.start, self.end = torch.from_numpy(start).to(dev), torch.from_numpy(end).to(dev)

    def sample(self, B, gen, to=TO, tp=TP):
        dev = self.img.device
        i = torch.randint(0, self.T, (B,), device=dev, generator=gen)
        oi = torch.maximum(i[:, None] + torch.arange(-to + 1, 1, device=dev)[None], self.start[i, None])
        ai = torch.minimum(i[:, None] + torch.arange(tp, device=dev)[None], self.end[i, None])
        return self.img[oi], self.prop[oi], self.act[ai]


def augment(img_u8, gen, train=True):
    """(B, TO, cam, H, W, 3) uint8 -> float (B, TO, cam, 3, CROP, CROP); random crop (same for the TO frames of a
    sample) + brightness/contrast jitter during training, centre crop otherwise."""
    B, To, C, H, W, _ = img_u8.shape
    x = img_u8.permute(0, 1, 2, 5, 3, 4).float() / 255
    if not train:
        o = (H - CROP) // 2
        return x[..., o:o + CROP, o:o + CROP]
    dev = x.device
    oy = torch.randint(0, H - CROP + 1, (B,), device=dev, generator=gen)
    ox = torch.randint(0, W - CROP + 1, (B,), device=dev, generator=gen)
    ar = torch.arange(CROP, device=dev)
    iy = (oy[:, None] + ar).view(B, 1, 1, 1, CROP, 1).expand(B, To, C, 3, CROP, W)
    x = torch.gather(x, 4, iy)
    ix = (ox[:, None] + ar).view(B, 1, 1, 1, 1, CROP).expand(B, To, C, 3, CROP, CROP)
    x = torch.gather(x, 5, ix)
    br = 1 + 0.2 * (torch.rand(B, 1, C, 1, 1, 1, device=dev, generator=gen) - 0.5)
    ct = 1 + 0.2 * (torch.rand(B, 1, C, 1, 1, 1, device=dev, generator=gen) - 0.5)
    m = x.mean((3, 4, 5), keepdim=True)
    return ((x - m) * ct + m * br).clamp(0, 1)


# ---------------------------------------------------------------------------------------------- train
def train(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    sets = {}
    if a.alpha < 1:
        sets["sim"] = Demos(a.sim, a.n_sim, dev)
    if a.alpha > 0:
        sets["real"] = Demos(a.real, a.n_real, dev)
    pol = DiffusionPolicy().to(dev)
    # normalisation from all data used (both domains)
    P = torch.cat([s.prop for s in sets.values()]); A = torch.cat([s.act for s in sets.values()])
    pol.p_lo[:], pol.p_hi[:] = P.min(0).values - 1e-3, P.max(0).values + 1e-3
    pol.a_lo[:], pol.a_hi[:] = A.min(0).values - 1e-3, A.max(0).values + 1e-3
    ema = torch.optim.swa_utils.AveragedModel(pol, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    nr = int(round(a.alpha * a.batch))
    print(f"train: sim {sets['sim'].n_eps if 'sim' in sets else 0} eps, real {sets['real'].n_eps if 'real' in sets else 0} eps, "
          f"alpha {a.alpha} ({nr}/{a.batch} real per batch)", flush=True)
    t0 = time.time()
    for step in range(a.steps):
        parts = []
        if nr:
            parts.append(sets["real"].sample(nr, gen))
        if a.batch - nr:
            parts.append(sets["sim"].sample(a.batch - nr, gen))
        img, prop, act = (torch.cat(x) for x in zip(*parts))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = pol.loss(augment(img, gen), prop, act)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step(); sched.step(); ema.update_parameters(pol)
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": ema.module.state_dict(), "args": vars(a)}, a.out)


# ---------------------------------------------------------------------------------------------- eval
def evaluate(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    from cam_env import CamEnv
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev)
    pol = DiffusionPolicy().to(dev); pol.load_state_dict(ck["state_dict"]); pol.eval()
    env = CamEnv(a.domain)
    gen = torch.Generator(device=dev)
    res = []
    for k in range(a.episodes):
        seed = a.eval_seed0 + k
        gen.manual_seed(seed)
        o = env.reset(seed)
        hist = [o] * TO
        plan, t = [], 0
        while True:
            if not plan:
                img = torch.from_numpy(np.stack([h["images"] for h in hist[-TO:]]))[None].to(dev)
                prop = torch.from_numpy(np.stack([h["proprio"] for h in hist[-TO:]]))[None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    acts = pol.act(augment(img, None, train=False), prop, gen)[0].float().cpu().numpy()
                plan = list(acts[:TA])
            o, done, info = env.step(plan.pop(0))
            hist.append(o); t += 1
            if done:
                break
        res.append({"seed": seed, "success": bool(info["success"]), "xy_err": info["xy_err"], "tilt": info["tilt"],
                    "nut_z": info["nut_z"], "steps": t})
    sr = float(np.mean([r["success"] for r in res]))
    print(json.dumps({"ckpt": a.ckpt, "domain": a.domain, "success_rate": sr, "n": len(res)}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--sim", default="data/sim.npz"); ap.add_argument("--real", default="data/real.npz")
    ap.add_argument("--n_sim", type=int, default=300); ap.add_argument("--n_real", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dp.pt")
    ap.add_argument("--ckpt", default="runs/dp.pt"); ap.add_argument("--domain", default="real")
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--eval_seed0", type=int, default=2_000_000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
