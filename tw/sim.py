"""MuJoCo FlexiTac nut pick-and-place, exposed in Touch-in-the-Wild / UMI conventions.

Observation (what the UMI replay buffer stores per frame, 20 Hz):
  camera0_rgb        (224, 224, 3) uint8   YAM D405 wrist camera (stands in for the UMI GoPro)
  camera0_tactile    (12, 64) float32      left | right FlexiTac pad, each 12 x 32 in [0, 1] (as in the paper)
  robot0_eef_pos     (3,)  pinch-point position from joint encoders + forward kinematics (base = world)
  robot0_eef_rot_axis_angle (3,)
  robot0_gripper_width (1,) gap between the two pad faces (m)
Action (UMI 'action' key, absolute): target pinch pose (pos + axis-angle) and target gripper width (m). The policy
predicts it relative to the current pose (the dataset converts); here it is executed with the arm IK.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "adamu_nut_yam")); sys.path.insert(0, str(ROOT / "adamu_nut_tactile"))
os.environ.setdefault("MUJOCO_GL", "egl")
from env import NUT_HALF_H, TABLE_TOP, NutPickPlaceEnv, ScriptedExpert  # noqa: E402
from kin import pinch_rot  # noqa: E402
from tactile import FlexiTacPads  # noqa: E402
from fast_render import make_renderer  # noqa: E402

SCENE = ROOT / "adamu_nut_tactile" / "assets" / "scene_cam.xml"
IMG = 224


def pose6(pos, R):
    return np.concatenate([pos, Rotation.from_matrix(R).as_rotvec()]).astype(np.float32)


class TactileSim(NutPickPlaceEnv):
    def __init__(self, render=True):
        super().__init__(scene=SCENE)
        self.pads = FlexiTacPads(self.m)
        self.rend = make_renderer(self.m, IMG, IMG) if render else None
        self.opt = mujoco.MjvOption(); self.opt.sitegroup[:] = 0
        self.sites = [self.m.site(s).id for s in ("left_tactile", "right_tactile")]
        self._calib_width()

    # gripper width <-> grip command g (linear finger travel), calibrated without the nut
    def _calib_width(self):
        m, d = self.m, self.d
        W = []
        for g in (0.0, 0.5):
            self.reset(0)
            d.qpos[self.nut_q:self.nut_q + 3] = [0, 0, 0.2]          # nut out of the way
            for _ in range(60):
                self._apply_grip(g); mujoco.mj_step(m, d)
            W.append(self.width())
        self.w_open, self.w_half = W

    def width(self):
        return float(np.linalg.norm(self.d.site_xpos[self.sites[0]] - self.d.site_xpos[self.sites[1]]))

    def g_to_width(self, g):
        return self.w_open + (self.w_half - self.w_open) * 2 * g

    def width_to_g(self, w):
        return float(np.clip((self.w_open - w) / (self.w_open - self.w_half) / 2, 0, 1))

    def eef_pose(self, q_arm=None):
        qpos = self.d.qpos.copy()
        if q_arm is not None:
            qpos[self.arm_q] = q_arm
        p, R = self.ik.pinch_pose(qpos)
        return p, R

    def tactile12x64(self):
        t = self.pads.read(self.d)                       # (2, 32 rows along finger, 12 across)
        return np.concatenate([t[0].T, t[1].T], 1)       # (12, 64): left | right

    def observe(self):
        p, R = self.eef_pose()
        o = dict(robot0_eef_pos=p.astype(np.float32), robot0_eef_rot_axis_angle=pose6(p, R)[3:],
                 robot0_gripper_width=np.array([self.width()], np.float32), camera0_tactile=self.tactile12x64())
        if self.rend is not None:
            self.rend.update_scene(self.d, "wrist_cam", self.opt)
            o["camera0_rgb"] = self.rend.render().copy()
        return o

    def step_pose(self, pos, rotvec, width):
        """Execute an absolute pinch-pose target with IK (warm-started at the current arm command)."""
        seed = self.d.qpos.copy(); seed[self.arm_q] = self.q_cmd
        q, _, _ = self.ik.solve(seed, np.asarray(pos, float), Rotation.from_rotvec(rotvec).as_matrix(), iters=60)
        self.q_cmd = q
        return self.step(np.concatenate([q, [self.width_to_g(width)]]))

    def reset(self, seed=None):
        o = super().reset(seed)
        self.q_cmd = self.d.qpos[self.arm_q].copy()
        return o


class RetryExpert(ScriptedExpert):
    """ScriptedExpert that, with probability p_miss, first closes 2 cm too high (on air), lifts a little,
    re-opens and grasps again: the demonstrations then contain the 'empty grasp -> regrasp' behaviour that touch
    makes observable."""
    MISS_DZ = 0.020

    def __init__(self, env, rng=None, p_miss=0.3, p_bump=0.0, **kw):
        self.p_miss, self.p_bump = p_miss, p_bump
        super().__init__(env, rng, **kw)

    def act(self):
        e = self.env
        if self.k < len(self.plan) and self.plan[self.k][0] == "replan":
            # bumped: grasp the nut where it is now (fresh plan from the current pinch pose and yaw)
            p, R = e.eef_pose(self.q_cmd); yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            ScriptedExpert.reset(self)
            nm, a, b, y0, y1, g0, g1, n = self.plan[0]
            self.plan[0] = (nm, p, b, yaw, y1, g0, g1, 24)
        if self.bump and not self.bumped and self.phase == "close" and self.g > 0.15 and e.width() > 0.03:
            q = e.d.qpos.copy(); q[e.arm_q] = self.q_cmd
            p, R = e.eef_pose(self.q_cmd); nut = e.nut_pos()
            if np.linalg.norm(nut[:2] - p[:2]) < 0.02:            # same perturbation as eval_policy.py 'bump'
                dv = R[:, 1].copy(); dv[2] = 0; dv /= np.linalg.norm(dv) + 1e-9
                e.d.qpos[e.nut_q:e.nut_q + 2] += 0.03 * dv[:2]
                mujoco.mj_forward(e.m, e.d); self.bumped = True
                G, yaw = self.CLOSE_G, self.plan[self.k][4]
                here = self.plan[self.k][2]
                self.plan = self.plan[:self.k + 1] + [
                    ("settle", here, here, yaw, yaw, G, G, 4),
                    ("check", here, here + [0, 0, 0.025], yaw, yaw, G, G, 10),
                    ("reopen", here + [0, 0, 0.025], here + [0, 0, 0.025], yaw, yaw, G, 0.0, 8),
                    ("replan", None, None, yaw, yaw, 0, 0, 1)]
        return super().act()

    def reset(self):
        super().reset()
        self.bump = bool(self.rng.random() < self.p_bump); self.bumped = False
        self.missed = (not self.bump) and bool(self.rng.random() < self.p_miss)
        if not self.missed:
            return
        P = {p[0]: p for p in self.plan}
        grasp = P["descend"][2]; yaw = P["descend"][4]; G = self.CLOSE_G
        miss = grasp + [0, 0, self.MISS_DZ]
        seq = [
            ("approach", *P["approach"][1:]),
            ("descend", None, miss, yaw, yaw, 0.0, 0.0, P["descend"][7]),
            ("close", None, miss, yaw, yaw, 0.0, G, P["close"][7]),
            ("settle", None, miss, yaw, yaw, G, G, 4),
            ("check", None, miss + [0, 0, 0.025], yaw, yaw, G, G, 10),
            ("reopen", None, miss + [0, 0, 0.025], yaw, yaw, G, 0.0, 8),
            ("redescend", None, grasp, yaw, yaw, 0.0, 0.0, 16),
            ("close", None, grasp, yaw, yaw, 0.0, G, P["close"][7]),
        ] + [p for p in self.plan if p[0] in ("settle", "lift", "carry", "lower", "open", "retreat")]
        plan, prev = [], seq[0][2]
        for name, a, b, y0, y1, g0, g1, n in seq:
            a = prev if a is None else a
            if name in ("settle", "lift", "carry", "lower", "open", "retreat") and plan and plan[-1][0] == "close" and name == "settle":
                a = grasp
            plan.append((name, a, np.asarray(b, float), y0, y1, g0, g1, n)); prev = np.asarray(b, float)
        self.plan = plan
        # the original settle/lift entries start from 'grasp' already; re-chain their start points
        for i in range(1, len(self.plan)):
            nm, a, b, *rest = self.plan[i]
            self.plan[i] = (nm, self.plan[i - 1][2], b, *rest)

    def target(self):
        """Commanded pinch pose and width for the current command (call after act())."""
        p, R = self.env.eef_pose(self.q_cmd)
        return pose6(p, R), self.env.g_to_width(self.g)
