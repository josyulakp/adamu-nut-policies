#!/usr/bin/env python3
"""ACT (Action Chunking with Transformers, Zhao et al. 2023) with sim-and-real co-training.

Same data, mixing ratio, crop/colour augmentation and evaluation as dp.py, so the two compare directly.
  * observation: current head + wrist RGB (104 px crops) and proprio (7 joints + grip); no history (as in ACT)
  * backbone: one ResNet-18 (GroupNorm, from scratch) per camera; the 512 x 4 x 4 feature map -> 1x1 conv -> 16
    tokens of width 256 per camera, with learned position embeddings
  * CVAE: at training a 4-layer transformer encoder reads [CLS, proprio, action chunk] and gives a 32-d latent
    z (KL weight 10); at run time z = 0
  * transformer: 4 encoder layers over [z, proprio, image tokens], 7 decoder layers with 16 learned queries ->
    a chunk of 16 actions (0.8 s at 20 Hz); L1 loss
  * run time: a chunk is predicted every step and overlapping predictions are averaged with weights
    exp(-0.01 i) (temporal ensembling, oldest first)

python cotrain/act.py train --alpha 0.5 --n_real 50 --out runs/act_co_r50_a05.pt
python cotrain/act.py eval --ckpt runs/act_co_r50_a05.pt --domain real --episodes 50 --json runs/act_co_r50_a05_real.json
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

from dp import CROP, Demos, augment

CHUNK, D, ZDIM, KL_W = 16, 256, 32, 10.0


def backbone():
    net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
    return nn.Sequential(*list(net.children())[:-2])          # (B, 512, H/32, W/32)


class ACT(nn.Module):
    def __init__(self, pdim=8, adim=8, ncam=2):
        super().__init__()
        self.ncam, self.adim = ncam, adim
        self.backbones = nn.ModuleList([backbone() for _ in range(ncam)])
        self.proj = nn.Conv2d(512, D, 1)
        ntok = (math.ceil(CROP / 32)) ** 2
        self.img_pos = nn.Parameter(torch.randn(1, ncam * ntok, D) * 0.02)
        # CVAE encoder
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.enc_a = nn.Linear(adim, D); self.enc_p = nn.Linear(pdim, D)
        self.enc_pos = nn.Parameter(torch.randn(1, CHUNK + 2, D) * 0.02)
        lay = lambda: nn.TransformerEncoderLayer(D, 8, 2048, 0.1, batch_first=True, norm_first=True)
        self.vae = nn.TransformerEncoder(lay(), 4)
        self.to_latent = nn.Linear(D, 2 * ZDIM)
        # policy transformer
        self.z_in = nn.Linear(ZDIM, D); self.p_in = nn.Linear(pdim, D)
        self.extra_pos = nn.Parameter(torch.randn(1, 2, D) * 0.02)
        self.encoder = nn.TransformerEncoder(lay(), 4)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D, 8, 2048, 0.1, batch_first=True, norm_first=True), 7)
        self.queries = nn.Parameter(torch.randn(1, CHUNK, D) * 0.02)
        self.head = nn.Linear(D, adim)
        for k, n in (("p_mu", pdim), ("p_sd", pdim), ("a_mu", adim), ("a_sd", adim)):
            self.register_buffer(k, torch.zeros(n))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, prop, act=None):
        """imgs (B, cam, 3, H, W) in [0, 1]; prop (B, pdim) raw; act (B, CHUNK, adim) raw or None."""
        B = imgs.shape[0]
        p = (prop - self.p_mu) / self.p_sd
        if act is not None:
            a = (act - self.a_mu) / self.a_sd
            tok = torch.cat([self.cls.expand(B, -1, -1), self.enc_p(p)[:, None], self.enc_a(a)], 1) + self.enc_pos
            mu, logvar = self.to_latent(self.vae(tok)[:, 0]).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(B, ZDIM, device=imgs.device)
        feats = []
        for c, bb in enumerate(self.backbones):
            f = self.proj(bb((imgs[:, c] - self.img_mean) / self.img_std))
            feats.append(f.flatten(2).transpose(1, 2))
        src = torch.cat([torch.stack([self.z_in(z), self.p_in(p)], 1) + self.extra_pos,
                         torch.cat(feats, 1) + self.img_pos], 1)
        mem = self.encoder(src)
        out = self.head(self.decoder(self.queries.expand(B, -1, -1), mem))
        return out, mu, logvar

    def loss(self, imgs, prop, act):
        out, mu, logvar = self(imgs, prop, act)
        l1 = F.l1_loss(out, (act - self.a_mu) / self.a_sd)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + KL_W * kl, l1, kl

    @torch.no_grad()
    def act(self, imgs, prop):
        out, _, _ = self(imgs, prop)
        return out * self.a_sd + self.a_mu


def train(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    sets = {}
    if a.alpha < 1:
        sets["sim"] = Demos(a.sim, a.n_sim, dev)
    if a.alpha > 0:
        sets["real"] = Demos(a.real, a.n_real, dev)
    pol = ACT().to(dev)
    P = torch.cat([s.prop for s in sets.values()]); A = torch.cat([s.act for s in sets.values()])
    pol.p_mu[:], pol.p_sd[:] = P.mean(0), P.std(0) + 1e-3
    pol.a_mu[:], pol.a_sd[:] = A.mean(0), A.std(0) + 1e-3
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    nr = int(round(a.alpha * a.batch))
    print(f"ACT train: sim {sets['sim'].n_eps if 'sim' in sets else 0} eps, real "
          f"{sets['real'].n_eps if 'real' in sets else 0} eps, alpha {a.alpha} ({nr}/{a.batch} real per batch)", flush=True)
    t0 = time.time()
    for step in range(a.steps):
        parts = []
        if nr:
            parts.append(sets["real"].sample(nr, gen, to=1, tp=CHUNK))
        if a.batch - nr:
            parts.append(sets["sim"].sample(a.batch - nr, gen, to=1, tp=CHUNK))
        img, prop, act = (torch.cat(x) for x in zip(*parts))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, l1, kl = pol.loss(augment(img, gen)[:, 0], prop[:, 0], act)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} l1 {l1.item():.4f} kl {kl.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": pol.state_dict(), "args": vars(a)}, a.out)


def evaluate(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    from cam_env import CamEnv
    dev = "cuda"
    pol = ACT().to(dev); pol.load_state_dict(torch.load(a.ckpt, map_location=dev)["state_dict"]); pol.eval()
    env = CamEnv(a.domain)
    res = []
    for k in range(a.episodes):
        seed = a.eval_seed0 + k
        o = env.reset(seed)
        buf, t = [], 0
        while True:
            img = torch.from_numpy(o["images"])[None, None].to(dev)
            prop = torch.from_numpy(o["proprio"])[None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = pol.act(augment(img, None, train=False)[:, 0], prop)[0].float().cpu().numpy()
            buf.append((t, chunk)); buf = [(t0, c) for t0, c in buf if t - t0 < CHUNK]
            preds = np.array([c[t - t0] for t0, c in buf])            # oldest first
            w = np.exp(-a.ens_k * np.arange(len(preds)))
            act = (preds * w[:, None]).sum(0) / w.sum()
            o, done, info = env.step(act)
            t += 1
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
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/act.pt")
    ap.add_argument("--ckpt", default="runs/act.pt"); ap.add_argument("--domain", default="real")
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--eval_seed0", type=int, default=2_000_000)
    ap.add_argument("--ens_k", type=float, default=0.01)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
