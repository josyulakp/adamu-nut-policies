"""Figures for report/report.md, from the per-seed results in results/ (50 held-out seeds per cell)."""
import json
from pathlib import Path

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3, fixed order
INK, MUTED, GRID = "#1b2229", "#5d6873", "#e3e6ea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def rate(stem):
    f = REPO / "results" / f"{stem}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()); e = d["episodes"] if isinstance(d, dict) else d
    k, n = sum(bool(x["success"]) for x in e), len(e)
    z = 1.96; p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)          # Wilson 95% interval
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0, c - h), min(1, c + h)


def bars(rows, series, fname, title, height):
    fig, ax = plt.subplots(figsize=(6.6, height))
    y = np.arange(len(rows))[::-1]; bh = 0.8 / len(series)
    for j, (cond, label, col) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * bh
        for yi, (name, stem) in zip(y, rows):
            r = rate(f"{stem}_{cond}")
            if r is None:
                continue
            p, lo, hi = r
            ax.barh(yi - off, p * 100, height=bh * 0.86, color=col, label=label if yi == y[0] or not ax.get_legend_handles_labels()[1].count(label) else None)
            ax.errorbar(p * 100, yi - off, xerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.text(hi * 100 + 1.5, yi - off, f"{p * 100:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [n for n, _ in rows]); ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Success rate (50 held-out seeds; bars: 95% interval)")
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(seen), frameon=False, fontsize=8,
              handlelength=1.2, borderaxespad=0.2)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=22)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)


bars([("DP, sim only (300 sim)", "cotrain/sim_only"), ("DP, real only (50 real)", "cotrain/real_r50"),
      ("DP, real only (100 real)", "cotrain/real_r100"), ("DP, co-trained (300 sim + 50 real)", "cotrain/co_r50_a05"),
      ("DP + DSRL (co-trained base)", "cotrain/dsrl_co_r50"), ("ACT, co-trained (300 sim + 50 real)", "cotrain/act_co_r50_a05")],
     [("ideal", "Ideal sim", BLUE), ("real", "Real-robot stand-in", ORANGE)],
     "fig1_camera.png", "Camera policies, plain YAM gripper", 3.0)
bars([("TW, wrist camera only", "tw/dp_vision"), ("TW, camera + touch (MAE)", "tw/dp_tactile_mae"),
      ("TW, camera + touch (SimpleCNN)", "tw/dp_tactile_cnn"),
      ("DexScrew teacher (privileged state)", "ds/teacher"), ("DexScrew student (proprio + touch)", "ds/student"),
      ("DexScrew student + tactile fine-tune", "ds/ft")],
     [("nominal", "Ideal sim", BLUE), ("bump", "Bump at grasp", ORANGE)],
     "fig2_tactile.png", "Tactile policies, FlexiTac YAM gripper", 3.0)
print("ok")


# ---------------------------------------------------------------- Figure 3: success against number of demonstrations
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"


def episodes(stem):
    d = json.loads((REPO / "results" / f"{stem}.json").read_text())
    return d["episodes"] if isinstance(d, dict) else d


fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
bc = json.loads((Path(__file__).resolve().parent / "data" / "bc_scaling.json").read_text())["series"]
for s, col, mk in zip(bc, (BLUE, ORANGE), ("o", "s")):
    n = [p["n"] for p in s["scaling"]]; m = np.array([p["mean"] for p in s["scaling"]]) * 100
    ci = np.array([p["ci"] for p in s["scaling"]]) * 100
    a1.errorbar(n, m, yerr=ci, color=col, marker=mk, ms=4.5, lw=2, capsize=2, label=s["name"])
    a1.text(n[-1] * 1.08, m[-1], f"{m[-1]:.0f}%", va="center", fontsize=7.5, color=INK)
a1.set_xscale("log", base=2); a1.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"]); a1.set_xlim(20, 300)
a1.set_xlabel("Demonstrations"); a1.set_ylabel("Success rate")
a1.set_title("State-based BC, ideal sim\n(3 training seeds × 100 episodes)", loc="left", fontsize=8.5, color=INK)
for label, pts, col, mk in [("DP, real demos only", [(10, "real_r10"), (50, "real_r50"), (100, "real_r100")], AQUA, "o"),
                            ("DP, co-trained with 300 sim", [(0, "sim_only"), (10, "co_r10_a05"), (50, "co_r50_a05")], YELLOW, "s")]:
    xs, ms, lo, hi = [], [], [], []
    for n, stem in pts:
        p, l, h = rate(f"cotrain/{stem}_real"); xs.append(n); ms.append(p * 100); lo.append((p - l) * 100); hi.append((h - p) * 100)
    xp = [x if x > 0 else 5 for x in xs]                                     # 0 real demos drawn at the left edge
    a2.errorbar(xp, ms, yerr=[lo, hi], color=col, marker=mk, ms=4.5, lw=2, capsize=2, label=label)
    a2.text(xp[-1] * 1.12, ms[-1], f"{ms[-1]:.0f}%", va="center", fontsize=7.5, color=INK)
a2.set_xscale("log", base=2); a2.set_xticks([5, 10, 50, 100], ["0", "10", "50", "100"]); a2.set_xlim(4, 160)
a2.set_xlabel("Real-robot (stand-in) demonstrations")
a2.set_title("Image DP, real-robot stand-in\n(1 training seed × 50 episodes)", loc="left", fontsize=8.5, color=INK)
for a in (a1, a2):
    a.set_ylim(0, 105); a.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True); a.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig3_scaling.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: failure modes
starts = {**json.loads((Path(__file__).resolve().parent / "starts_cam.json").read_text()),
          **json.loads((Path(__file__).resolve().parent / "starts_tac.json").read_text())}
REST_ABS = 0.988                                     # camera results store absolute nut height; TW stores it relative to rest
MODES = [("Never picked up", BLUE), ("Off target", ORANGE), ("Tipped over", AQUA), ("Still held at timeout", YELLOW),
         ("Off the table", MAGENTA), ("At target, not settled", GREEN)]


def classify(e, start, absolute_z):
    if "nut_z" not in e:                             # DexScrew: no final height; 'fail' = tilt > 45 deg or off the table
        if e["fail"]:
            return "Tipped over" if e["tilt"] >= 15 else "Off the table"
        e = dict(e, nut_z=0.0); absolute_z = False
    z = e["nut_z"] - REST_ABS if absolute_z else e["nut_z"]
    nx, ny, tx, ty = start; d0 = float(np.hypot(nx - tx, ny - ty))
    if z < -0.05:
        return "Off the table"
    if z > 0.02:
        return "Still held at timeout"
    if e["tilt"] >= 15:
        return "Tipped over"
    if abs(e["xy_err"] - d0) < 0.01:
        return "Never picked up"
    if e["xy_err"] < 0.02:
        return "At target, not settled"
    return "Off target"


ROWS = [("DP sim only · stand-in", "cotrain/sim_only_real", "real", True),
        ("DP real only (100) · stand-in", "cotrain/real_r100_real", "real", True),
        ("DP co-trained · stand-in", "cotrain/co_r50_a05_real", "real", True),
        ("DP + DSRL · stand-in", "cotrain/dsrl_co_r50_real", "real", True),
        ("ACT co-trained · stand-in", "cotrain/act_co_r50_a05_real", "real", True),
        ("TW camera only · ideal", "tw/dp_vision_nominal", "tw", False), ("TW camera only · bump", "tw/dp_vision_bump", "tw", False),
        ("TW touch (MAE) · ideal", "tw/dp_tactile_mae_nominal", "tw", False), ("TW touch (MAE) · bump", "tw/dp_tactile_mae_bump", "tw", False),
        ("TW touch (CNN) · ideal", "tw/dp_tactile_cnn_nominal", "tw", False), ("TW touch (CNN) · bump", "tw/dp_tactile_cnn_bump", "tw", False),
        ("DexScrew student · ideal", "ds/student_nominal", "ds", False), ("DexScrew fine-tuned · ideal", "ds/ft_nominal", "ds", False),
        ("DexScrew fine-tuned · bump", "ds/ft_bump", "ds", False)]
table = {}
fig, ax = plt.subplots(figsize=(6.6, 4.1))
y = np.arange(len(ROWS))[::-1]
for yi, (name, stem, sk, absz) in zip(y, ROWS):
    eps = episodes(stem); counts = {m: 0 for m, _ in MODES}
    for e in eps:
        if not e["success"]:
            counts[classify(e, starts[sk][str(e["seed"])], absz)] += 1
    table[name] = dict(counts, success=sum(e["success"] for e in eps), n=len(eps))
    left = 0
    for m, col in MODES:
        c = counts[m]
        if c:
            ax.barh(yi, c, left=left, color=col, height=0.7, edgecolor="white", linewidth=1.5)
            if c >= 2:
                ax.text(left + c / 2, yi, str(c), ha="center", va="center", fontsize=7, color=INK)
            left += c
    ax.text(left + 0.8, yi, f"{table[name]['success']}/{len(eps)} succeeded", va="center", fontsize=7, color=MUTED)
ax.set_yticks(y, [r[0] for r in ROWS]); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 62)
ax.set_xlabel("Failed episodes out of 50, by final state"); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(handles=[matplotlib.patches.Patch(color=c, label=m) for m, c in MODES], loc="lower left", bbox_to_anchor=(0, 1.0),
          ncol=3, frameon=False, fontsize=7.5, handlelength=1.2)
ax.set_title("Failure modes", loc="left", fontsize=10, color=INK, fontweight="bold", pad=34)
fig.tight_layout(); fig.savefig(OUT / "fig4_failures.png", dpi=200); plt.close(fig)
(Path(__file__).resolve().parent / "data" / "failure_modes.json").write_text(json.dumps(table, indent=1))
for k, v in table.items():
    print(f"{k:32s}", v)
