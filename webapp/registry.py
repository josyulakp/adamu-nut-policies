"""Models the eval page can run, and the conditions each one supports (no torch / mujoco imports: the server and
both workers import this)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import os
DATA = Path(os.environ.get("ADAMU_DATA", REPO))       # folder holding models/ and results/
M = DATA / "models"

CAM_CONDS = [("ideal", "Ideal (nominal sim)"), ("real", "Real-robot stand-in"), ("sim", "Randomised training sim")]
TW_CONDS = [("nominal", "Ideal (nominal sim)"), ("bump", "Bump at grasp")]
DS_CONDS = TW_CONDS + [("shifted", "Shifted physics")]

# id -> (worker, gripper variant, group, label, kind, path, conditions, result-file stem for the ideal success rate)
MODELS = {}


def _add(mid, worker, variant, group, label, kind, path, conds, res):
    MODELS[mid] = dict(id=mid, worker=worker, variant=variant, group=group, label=label, kind=kind, path=str(path),
                       conds=conds, res=res)


for n, lab in [("sim_only", "300 sim demos"), ("co_r50_a05", "co-trained 300 sim + 50 real"),
               ("co_r10_a025", "co-trained 300 sim + 10 real, α 0.25"), ("co_r10_a05", "co-trained 300 sim + 10 real, α 0.5"),
               ("real_r100", "100 real demos"), ("real_r50", "50 real demos"), ("real_r10", "10 real demos")]:
    _add(f"dp_{n}", "cam", "yam", "Diffusion Policy", lab, "dp", M / "dp" / f"{n}.pt", CAM_CONDS, f"cotrain/{n}")
for n, lab in [("act_co_r50_a05", "co-trained 300 sim + 50 real"), ("act_sim_only", "300 sim demos"),
               ("act_co_r10_a05", "co-trained 300 sim + 10 real"), ("act_real_r50", "50 real demos")]:
    _add(n, "cam", "yam", "ACT", lab, "act", M / "act" / f"{n}.pt", CAM_CONDS, f"cotrain/{n}")
for n, lab in [("dsrl_co_r50", "on co-trained DP (300 sim + 50 real)"), ("dsrl_real_r50", "on real-only DP (50 real)")]:
    _add(n, "cam", "yam", "DP + DSRL", lab, "dsrl", M / "dsrl" / n, CAM_CONDS, f"cotrain/{n}")
for n, lab in [("dp_vision", "wrist camera only"), ("dp_tactile_mae", "camera + touch (visuo-tactile MAE)"),
               ("dp_tactile_cnn", "camera + touch (SimpleCNN)")]:
    _add(f"tw_{n}", "tac", "tac", "Touch in the Wild", lab, "tw", M / "tw" / f"{n}.pt", TW_CONDS, f"tw/{n}")
for n, f, kind, lab in [("teacher", "teacher.pth", "teacher", "stage 1 teacher (privileged state)"),
                        ("student", "student.ckpt", "student", "stage 2 student (proprio + touch)"),
                        ("ft", "stage3_ft.ckpt", "student", "stage 3 tactile BC fine-tune")]:
    _add(f"ds_{n}", "tac", "tac", "DexScrew", lab, f"ds_{kind}", M / "ds" / f, DS_CONDS, f"ds/{n}")

# table-top layout (world frame, metres; the robot's waist is at the origin, +x forward, +y to its left)
NUT_REGION = ((0.26, 0.38), (-0.34, -0.18))       # training distribution of the nut start (adamu_nut_yam/env.py)
TARGET_REGION = ((0.26, 0.36), (-0.08, 0.02))     # ... and of the target centre
PLACE_BOUNDS = ((0.20, 0.50), (-0.46, 0.14))      # where the page lets you put things (front part of the table)
TABLE = ((0.20, 0.64), (-0.48, 0.28))
