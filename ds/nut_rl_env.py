"""Single MuJoCo environment for the DexScrew (Hsieh et al. 2025) pipeline on the FlexiTac nut pick-and-place task.

DexScrew trains an oracle with PPO on privileged state in an imperfect, randomised simulation, distils it into a
student that infers the privileged latent from 30 steps of proprioception (+ touch), and then fine-tunes with
tactile behaviour cloning on data collected in the real world with the learned skill as a prior. This file is the
simulation side, with the same observation contract as their IsaacGym task (obs / priv_info / proprio_hist).

Action (5, in [-1, 1]): pinch-point displacement (1 cm per step), yaw rate (0.08 rad per step), absolute grip
command g = (a + 1) / 2. Executed with the arm IK at 20 Hz (the robot's Cartesian controller).
obs (34): 3-frame history of [pinch pos 3, yaw sin/cos 2, gripper width 1, grip cmd 1] (21), last action (5),
  estimated nut position (3, 3 mm noise: perception), nut yaw as sin/cos(6 yaw) (2), target position (3).
priv_info (11): exact nut position relative to the pinch point (3), nut tilt (1), nut mass, nut friction, pad
  friction, arm gain scale (4), left / right pad contact (2), control latency / 2 (1).
proprio_hist (30, 24): [pinch pos 3, yaw sin/cos 2, width 1, grip cmd 1, last action 5, tactile features 12]
  (tactile features per pad: total force, contact area, centroid row/col, peak, row spread).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tw"))
from sim import NUT_HALF_H, TABLE_TOP, TactileSim  # noqa: E402
from kin import pinch_rot  # noqa: E402

PROP_LEN, PROP_DIM, OBS_DIM, PRIV_DIM, ACT_DIM = 30, 24, 34, 11, 5
EP_STEPS = 200
SUCC_STREAK = 10      # same as the base env's success test
WS_LO, WS_HI = np.array([0.18, -0.40, TABLE_TOP + 0.004]), np.array([0.46, 0.08, TABLE_TOP + 0.20])

NOMINAL = dict(nut_mass=0.030, nut_fric=1.5, pad_fric=1.0, kp_scale=1.0, latency=0)
SHIFTED = dict(nut_mass=0.040, nut_fric=0.9, pad_fric=0.7, kp_scale=0.8, latency=2)   # 'real' stand-in


def tactile_features(t):
    """t: (2, 32, 12) -> 12 features."""
    out = []
    rr, cc = np.meshgrid(np.arange(32) / 31, np.arange(12) / 11, indexing="ij")
    for p in range(2):
        f = t[p]; s = f.sum()
        if s > 1e-6:
            w = f / s
            r, c = (w * rr).sum(), (w * cc).sum()
            spread = np.sqrt((w * (rr - r) ** 2).sum())
        else:
            r = c = spread = 0.0
        out += [s / 30.0, (f > 0).mean() * 10, r, c, f.max(), spread * 5]
    return np.array(out, np.float32)


class NutRLEnv:
    def __init__(self, seed=0, randomize=True, domain=None):
        self.sim = TactileSim(render=False)
        self.m, self.d = self.sim.m, self.sim.d
        self.rng = np.random.default_rng(seed)
        self.randomize, self.domain = randomize, domain
        m = self.m
        self.nut_body, self.nut_geom = m.body("nut").id, m.geom("nut_col").id
        self.pad_geoms = [m.geom("left_pad").id, m.geom("right_pad").id]
        self.default = dict(mass=m.body_mass[self.nut_body], inertia=m.body_inertia[self.nut_body].copy(),
                            nut_fric=m.geom_friction[self.nut_geom].copy(), pad_fric=m.geom_friction[self.pad_geoms[0]].copy(),
                            gain=m.actuator_gainprm[:, 0].copy(), bias=m.actuator_biasprm[:, :3].copy())
        self.arm_act = self.sim.arm_act
        self.bump = False

    # ------------------------------------------------------------------ physics parameters
    def _params(self):
        if self.domain is not None:
            return dict(self.domain)
        if not self.randomize:
            return dict(NOMINAL)
        r = self.rng
        return dict(nut_mass=r.uniform(0.02, 0.045), nut_fric=r.uniform(0.8, 1.8), pad_fric=r.uniform(0.7, 1.3),
                    kp_scale=r.uniform(0.8, 1.15), latency=int(r.integers(0, 3)))

    def _apply(self, P):
        m, D = self.m, self.default
        k = P["nut_mass"] / D["mass"]
        m.body_mass[self.nut_body] = P["nut_mass"]; m.body_inertia[self.nut_body] = D["inertia"] * k
        m.geom_friction[self.nut_geom, 0] = P["nut_fric"]
        for g in self.pad_geoms:
            m.geom_friction[g, 0] = P["pad_fric"]
        m.actuator_gainprm[self.arm_act, 0] = D["gain"][self.arm_act] * P["kp_scale"]
        m.actuator_biasprm[self.arm_act, 1] = D["bias"][self.arm_act, 1] * P["kp_scale"]
        mujoco.mj_setConst(m, self.d)

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.P = self._params()
        # physics first: _apply ends with mj_setConst, which resets MjData to qpos0, so it must run before the scene
        # is laid out (calling it after the reset put the nut at its default spot and the arm at zero in every episode)
        self._apply(self.P)
        self.sim.reset(int(self.rng.integers(1 << 30)) if seed is None else seed)
        p, R = self.sim.eef_pose()
        self.pos_cmd = p.copy(); self.yaw_cmd = float(np.arctan2(R[1, 0], R[0, 0]))
        self.g = 0.0; self.t = 0; self.last_a = np.zeros(ACT_DIM, np.float32)
        self.queue = []
        self.z0 = self.sim.nut_pos()[2]
        self.succ_streak = 0; self.bumped = False
        f = self._frame()
        self.hist3 = [f] * 3
        self.prop = [self._prop(f)] * PROP_LEN
        return self._obs()

    def _frame(self):
        p, R = self.sim.eef_pose()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        self.tact = self.sim.pads.read(self.d)
        return np.array([*p, np.sin(yaw), np.cos(yaw), self.sim.width(), self.g], np.float32)

    def _prop(self, f):
        return np.concatenate([f, self.last_a, tactile_features(self.tact)]).astype(np.float32)

    def contacts(self):
        return [float(self.tact[i].sum() > 0) for i in range(2)]

    def _obs(self):
        nut = self.sim.nut_pos(); yaw = self.sim.nut_yaw()
        est = nut + self.rng.normal(0, 0.003, 3)
        obs = np.concatenate([np.concatenate(self.hist3), self.last_a, est, [np.sin(6 * yaw), np.cos(6 * yaw)],
                              self.sim.target_pos()]).astype(np.float32)
        p, R = self.sim.eef_pose()
        z = R[:, 2]
        tilt = np.degrees(np.arccos(np.clip(self.d.xmat[self.nut_body].reshape(3, 3)[2, 2], -1, 1)))
        priv = np.array([*(nut - p), tilt / 30, self.P["nut_mass"] * 30, self.P["nut_fric"], self.P["pad_fric"],
                         self.P["kp_scale"], *self.contacts(), self.P["latency"] / 2], np.float32)
        return obs, priv, np.stack(self.prop)

    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        self.queue.append(a)
        a_exec = self.queue.pop(0) if len(self.queue) > self.P["latency"] else np.zeros(ACT_DIM, np.float32)
        self.pos_cmd = np.clip(self.pos_cmd + a_exec[:3] * 0.01, WS_LO, WS_HI)
        self.yaw_cmd += float(a_exec[3]) * 0.08
        self.g = float((a_exec[4] + 1) / 2)
        seed = self.d.qpos.copy(); seed[self.sim.arm_q] = self.sim.q_cmd
        q, _, _ = self.sim.ik.solve(seed, self.pos_cmd, pinch_rot(self.yaw_cmd), iters=20)
        self.sim.q_cmd = q
        if self.bump and not self.bumped and self.g > 0.15 and self.sim.width() > 0.03:
            p, R = self.sim.eef_pose(); nut = self.sim.nut_pos()
            if np.linalg.norm(nut[:2] - p[:2]) < 0.02 and abs(nut[2] - p[2]) < 0.03:
                dv = R[:, 1].copy(); dv[2] = 0; dv /= np.linalg.norm(dv) + 1e-9
                self.d.qpos[self.sim.nut_q:self.sim.nut_q + 2] += 0.03 * dv[:2]
                mujoco.mj_forward(self.m, self.d); self.bumped = True
        _, _, info = self.sim.step(np.concatenate([q, [self.g]]))
        # anti-windup: the integrated position command may lead the real pinch point by at most 2 cm (v2 let it run
        # to the workspace box while the arm sat at its reach limit; the policy, which only sees the real pose,
        # then needed several reversed actions before anything moved and PPO stalled 6 cm from the target)
        p_now = self.sim.eef_pose()[0]
        self.pos_cmd = p_now + np.clip(self.pos_cmd - p_now, -0.02, 0.02)
        self.last_a = a
        self.t += 1
        f = self._frame()
        self.hist3 = self.hist3[1:] + [f]
        self.prop = self.prop[1:] + [self._prop(f)]
        r, success, fail = self._reward(info)
        done = success or fail or self.t >= EP_STEPS
        timeout = self.t >= EP_STEPS and not (success or fail)
        return self._obs(), r, done, dict(success=success, timeout=timeout, fail=fail)

    def _reward(self, info):
        sim = self.sim
        nut, tgt = sim.nut_pos(), sim.target_pos()
        p, R = sim.eef_pose()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        grasp_pt = nut + [0, 0, 0.004]
        d_reach = np.linalg.norm(p - grasp_pt)
        err = np.angle(np.exp(1j * 6 * (yaw - sim.nut_yaw() - np.pi / 6))) / 6      # mod 60 deg
        c = self.contacts(); grasped = c[0] and c[1]
        lift = float(np.clip((nut[2] - self.z0) / 0.05, 0, 1)) if grasped else 0.0
        dxy = np.linalg.norm(nut[:2] - tgt[:2])
        # shaping while carrying is capped at 7.5 / step; resting released on the target pays 10 / step, and the
        # terminal bonus pays the on-target rate for every remaining step, so finishing early never loses return
        # (v1 paid ~10 / step for holding the nut and ended the episode on release: the teacher learned to hover)
        r = (1 - np.tanh(10 * d_reach)) + 0.5 * (1 - abs(err) / (np.pi / 6)) * (d_reach < 0.05)
        r += 1.0 * grasped + 2.0 * lift + 3.0 * (1 - np.tanh(5 * dxy)) * (lift > 0.4 or (dxy < 0.03 and nut[2] < self.z0 + 0.01))
        on_target = dxy < 0.02 and abs(nut[2] - self.z0) < 0.004 and not any(c)
        tilt = np.degrees(np.arccos(np.clip(self.d.xmat[self.nut_body].reshape(3, 3)[2, 2], -1, 1)))
        if on_target and tilt < 15:
            r += 10.0; self.succ_streak += 1
        else:
            self.succ_streak = 0
        success = self.succ_streak >= SUCC_STREAK
        fail = bool(tilt > 45 or nut[2] < TABLE_TOP - 0.05)
        if success:
            r += 14.5 * (EP_STEPS - self.t) + 50.0
        if fail:
            r -= 10.0
        r -= 0.01 * float(np.sum(self.last_a[:4] ** 2))
        return float(r), bool(success), fail
