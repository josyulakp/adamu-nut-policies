"""Start layout (nut xy, target xy) of every evaluation seed, per environment family: report/starts_<family>.json.
  .venv:     python report/starts.py cam     (CamEnv ideal / real / sim: DP, ACT, DSRL)
  .venv-tw:  python report/starts.py tac     (TactileSim: Touch in the Wild; NutRLEnv: DexScrew)"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
SEEDS = range(2_000_000, 2_000_050)
out = {}
if sys.argv[1] == "cam":
    sys.path.insert(0, str(REPO / "adamu_nut_yam" / "cotrain"))
    from cam_env import CamEnv
    for dom in ("ideal", "real", "sim"):
        env = CamEnv(dom); out[dom] = {}
        for s in SEEDS:
            env.reset(s); n, t = env.nut_pos(), env.target_pos(); out[dom][s] = [float(n[0]), float(n[1]), float(t[0]), float(t[1])]
else:
    sys.path.insert(0, str(REPO / "tw")); sys.path.insert(0, str(REPO / "ds"))
    from sim import TactileSim
    from nut_rl_env import SHIFTED, NutRLEnv
    env = TactileSim(render=False); out["tw"] = {}
    for s in SEEDS:
        env.reset(s); n, t = env.nut_pos(), env.target_pos(); out["tw"][s] = [float(n[0]), float(n[1]), float(t[0]), float(t[1])]
    for key, dom in (("ds", None), ("ds_shifted", SHIFTED)):
        e = NutRLEnv(seed=0, randomize=False, domain=dom); out[key] = {}
        for s in SEEDS:
            e.reset(s); n, t = e.sim.nut_pos(), e.sim.target_pos(); out[key][s] = [float(n[0]), float(n[1]), float(t[0]), float(t[1])]
(REPO / "report" / f"starts_{sys.argv[1]}.json").write_text(json.dumps(out))
print({k: len(v) for k, v in out.items()})
