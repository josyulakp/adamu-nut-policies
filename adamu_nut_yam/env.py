"""ADAM-U + YAM linear_4310 gripper nut pick-and-place environment (MuJoCo), plus a scripted expert.

Same task as ../adamu_nut, with the right dexterous hand replaced by the YAM parallel-jaw gripper.
The grasp is purely physical (box pads with friction); there is no grasp assist.

Task
  The hex nut starts lying flat on the table, at a random position and yaw in a 12 x 16 cm region in
  front of the right arm. The YAM gripper on the right arm picks it with a parallel-jaw grasp across two flats and places it, upright,
  on a green target pad whose position is also randomised (on the robot's left of the start region).

Control: 20 Hz (25 physics steps of 2 ms).
Action (8): absolute position targets for the 7 right-arm joints (rad) + grip command g in [0, 1].
  The grip maps linearly to the linear_4310 finger position (kin.finger_pos). Waist, neck, left
  arm and the right ring/pinky fingers are held still.
Observation (dict, all float32):
  arm_q (7), arm_dq (7), grip (1: commanded g), hand_pos (3: pinch point), nut_pos (3),
  nut_yaw6 (2: sin/cos of 6*yaw, hex-symmetric), target_pos (3)  -> "state" concatenation (26).
Success (checked every step after the hand has released): nut centre within 2 cm of the target in
  the plane, resting on the table (z within 4 mm of rest height, tilt < 15 deg), not touching the
  hand. Episode ends on success held for 10 steps (0.5 s) or after 300 steps (15 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from kin import ARM, ArmIK, grip_targets, pinch_rot

SCENE = Path(__file__).resolve().parent / "assets" / "scene.xml"
CTRL_DT = 0.05
EP_LEN = 300
TABLE_TOP = 0.98
NUT_HALF_H = 0.008
NUT_REGION = ((0.26, 0.38), (-0.34, -0.18))       # x, y ranges of the nut start
TARGET_REGION = ((0.26, 0.36), (-0.08, 0.02))     # x, y ranges of the target centre
HOME_PINCH = np.array([0.28, -0.16, 1.14])
SUCCESS_XY = 0.02


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class EnvState:
    t: int = 0
    released: bool = False
    success_streak: int = 0
    info: dict = field(default_factory=dict)


class NutPickPlaceEnv:
    def __init__(self, render_size=None, scene=None):
        self.m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
        self.d = mujoco.MjData(self.m)
        self.ik = ArmIK(self.m)
        self.nsub = int(round(CTRL_DT / self.m.opt.timestep))
        m = self.m
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        self.arm_act = np.array([self.act[j] for j in ARM])
        self.arm_q = self.ik.qadr
        self.arm_dof = self.ik.dadr
        self.nut_q = m.jnt_qposadr[m.joint("nut_free").id]
        self.nut_body = m.body("nut").id
        self.target_body = m.body("target").id
        self.hand_bodies = set()
        root = m.body("wristRollRight").id
        for b in range(m.nbody):
            p = b
            while p > 0 and p != root:
                p = m.body_parentid[p]
            if p == root:
                self.hand_bodies.add(b)
        self.nut_geom = m.geom("nut_col").id
        self.finger_q = [m.jnt_qposadr[m.joint(j).id] for j in ("left_finger", "right_finger")]
        self.table_geom = m.geom("table_top").id
        # home arm configuration
        seed = m.qpos0.copy()
        seed[self.arm_q] = [-0.8, -0.2, 0.0, -1.2, 0, 0, 0]
        q, pe, _ = self.ik.solve(seed, HOME_PINCH, pinch_rot(np.deg2rad(30)))
        assert pe < 2e-3, pe
        self.home_q = q
        self.renderer = mujoco.Renderer(m, *render_size) if render_size else None
        self.state = EnvState()
        self.grip_cmd = 0.0

    # ------------------------------------------------------------------ core
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        q_home = self.home_q + rng.uniform(-0.03, 0.03, 7)
        d.qpos[self.arm_q] = q_home
        for a in self.finger_q:
            d.qpos[a] = grip_targets(0.0)["gripper"]
        nx, ny = rng.uniform(*NUT_REGION[0]), rng.uniform(*NUT_REGION[1])
        yaw = rng.uniform(0, np.pi / 3)
        d.qpos[self.nut_q:self.nut_q + 3] = [nx, ny, TABLE_TOP + NUT_HALF_H + 2e-4]
        d.qpos[self.nut_q + 3:self.nut_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        tx, ty = rng.uniform(*TARGET_REGION[0]), rng.uniform(*TARGET_REGION[1])
        m.body_pos[self.target_body] = [tx, ty, TABLE_TOP]
        # hold every actuator at its current joint position
        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        self.grip_cmd = 0.0
        self._apply_grip(0.0)
        mujoco.mj_forward(m, d)
        for _ in range(50):          # settle the nut
            mujoco.mj_step(m, d)
        self.state = EnvState()
        return self.obs()

    def _apply_grip(self, g):
        for name, v in grip_targets(g).items():
            self.d.ctrl[self.act[name]] = v

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        q = np.clip(action[:7], self.ik.lo, self.ik.hi)
        self.d.ctrl[self.arm_act] = q
        self.grip_cmd = float(np.clip(action[7], 0, 1))
        self._apply_grip(self.grip_cmd)
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)
        st = self.state
        st.t += 1
        ok, info = self._check_success()
        st.success_streak = st.success_streak + 1 if ok else 0
        done = st.success_streak >= 10 or st.t >= EP_LEN
        info["success"] = st.success_streak >= 10 or (st.t >= EP_LEN and ok)
        st.info = info
        return self.obs(), done, info

    # ------------------------------------------------------------------ observation / success
    def nut_pos(self):
        return self.d.xpos[self.nut_body].copy()

    def nut_yaw(self):
        return _quat_yaw(self.d.xquat[self.nut_body])

    def target_pos(self):
        return self.m.body_pos[self.target_body].copy()

    def hand_pos(self):
        return self.ik.pinch_pose(self.d.qpos)[0]

    def obs(self):
        yaw = self.nut_yaw()
        o = {
            "arm_q": self.d.qpos[self.arm_q].copy(),
            "arm_dq": self.d.qvel[self.arm_dof].copy(),
            "grip": np.array([self.grip_cmd]),
            "hand_pos": self.hand_pos(),
            "nut_pos": self.nut_pos(),
            "nut_yaw6": np.array([np.sin(6 * yaw), np.cos(6 * yaw)]),
            "target_pos": self.target_pos(),
        }
        o["state"] = np.concatenate([o[k] for k in STATE_KEYS]).astype(np.float32)
        return o

    def grasped(self):
        """Both pads touching the nut."""
        names = {self.m.geom(g).name for g in self._nut_contact_geoms()}
        return "left_pad" in names and "right_pad" in names

    def _nut_contact_geoms(self):
        d, out = self.d, set()
        for i in range(d.ncon):
            c = d.contact[i]
            if self.nut_geom in (c.geom1, c.geom2):
                out.add(int(c.geom2 if c.geom1 == self.nut_geom else c.geom1))
        return out

    def _nut_contact_bodies(self):
        d, out = self.d, set()
        for i in range(d.ncon):
            c = d.contact[i]
            if self.nut_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.nut_geom else c.geom1
                out.add(int(self.m.geom_bodyid[other]))
        return out

    def hand_touches_nut(self):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.nut_geom in (g1, g2):
                other = g2 if g1 == self.nut_geom else g1
                if self.m.geom_bodyid[other] in self.hand_bodies:
                    return True
        return False

    def _check_success(self):
        p = self.nut_pos()
        t = self.target_pos()
        R = self.d.xmat[self.nut_body].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        tilt = min(tilt, 180 - tilt)
        xy = float(np.linalg.norm(p[:2] - t[:2]))
        resting = abs(p[2] - (TABLE_TOP + NUT_HALF_H)) < 0.004 and tilt < 15
        touching = self.hand_touches_nut()
        lifted = p[2] > TABLE_TOP + NUT_HALF_H + 0.03
        info = {"xy_err": xy, "nut_z": float(p[2]), "tilt": float(tilt), "touching": touching, "lifted": lifted}
        return (xy < SUCCESS_XY and resting and not touching), info

    def render(self, cam=None):
        if cam is None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.30, -0.15, 1.02]
            cam.distance, cam.azimuth, cam.elevation = 0.9, 205, -32
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render()


STATE_KEYS = ["arm_q", "arm_dq", "grip", "hand_pos", "nut_pos", "nut_yaw6", "target_pos"]


# ---------------------------------------------------------------------- scripted expert
class ScriptedExpert:
    """Waypoint planner with IK on privileged state. Re-plans the grasp from the nut's current pose.

    Phases: approach above nut -> descend -> close -> lift -> carry above target -> lower -> open ->
    retreat -> hold. Cartesian segments are linearly interpolated and solved with warm-started IK.
    """
    PRE_H = 0.09
    GRASP_DZ = 0.004      # pad centre height above nut centre (keeps pads off the table)
    CLOSE_G = 0.85
    PLACE_DZ = 0.004      # release height of nut centre above its rest height

    def __init__(self, env: NutPickPlaceEnv, rng=None, speed=1.0, yaw_choice=None):
        self.env = env
        self.yaw_choice = yaw_choice      # optional f(candidate yaws, rng) -> yaw; default: nearest 30 deg
        self.rng = rng or np.random.default_rng()
        s = speed
        self.durations = dict(approach=int(36 / s), descend=int(22 / s), close=12, settle=4, lift=int(20 / s),
                              carry=int(34 / s), lower=int(20 / s), open=8, retreat=int(16 / s))
        self.reset()

    def reset(self):
        e = self.env
        self.q_cmd = e.d.qpos[e.arm_q].copy()
        self.g = 0.0
        self.plan = []
        nut = e.nut_pos()
        yaw_nut = e.nut_yaw()
        # pinch axis along a pair of flats: nut_yaw + 30 deg + k*60, choose the one nearest 30 deg
        cands = yaw_nut + np.pi / 6 + np.arange(-6, 7) * np.pi / 3
        yaw = (self.yaw_choice(cands, self.rng) if self.yaw_choice else
               cands[np.argmin(np.abs(np.angle(np.exp(1j * (cands - np.deg2rad(30))))))])
        tgt = e.target_pos()
        grasp = nut + [0, 0, self.GRASP_DZ]
        place = np.array([tgt[0], tgt[1], TABLE_TOP + NUT_HALF_H + self.PLACE_DZ + self.GRASP_DZ])
        jitter = lambda: self.rng.normal(0, 0.004, 3) * [1, 1, 0.5]
        start, _ = e.ik.pinch_pose(e.d.qpos)
        start_yaw = np.deg2rad(30)
        D = self.durations
        P = [
            ("approach", start, grasp + [0, 0, self.PRE_H] + jitter(), start_yaw, yaw, 0.0, 0.0, D["approach"]),
            ("descend", None, grasp, yaw, yaw, 0.0, 0.0, D["descend"]),
            ("close", None, grasp, yaw, yaw, 0.0, self.CLOSE_G, D["close"]),
            ("settle", None, grasp, yaw, yaw, self.CLOSE_G, self.CLOSE_G, D["settle"]),
            ("lift", None, grasp + [0, 0, self.PRE_H], yaw, yaw, self.CLOSE_G, self.CLOSE_G, D["lift"]),
            ("carry", None, place + [0, 0, self.PRE_H] + jitter(), yaw, yaw, self.CLOSE_G, self.CLOSE_G, D["carry"]),
            ("lower", None, place, yaw, yaw, self.CLOSE_G, self.CLOSE_G, D["lower"]),
            ("open", None, place, yaw, yaw, self.CLOSE_G, 0.0, D["open"]),
            ("retreat", None, place + [0, 0, self.PRE_H], yaw, yaw, 0.0, 0.0, D["retreat"]),
        ]
        prev = start
        for name, a, b, y0, y1, g0, g1, n in P:
            a = prev if a is None else a
            self.plan.append((name, a, np.asarray(b, float), y0, y1, g0, g1, n))
            prev = np.asarray(b, float)
        self.k = 0
        self.i = 0
        self.phase = self.plan[0][0]
        self._seed = e.d.qpos.copy()

    def act(self):
        e = self.env
        if self.k >= len(self.plan):
            return np.concatenate([self.q_cmd, [self.g]])
        name, a, b, y0, y1, g0, g1, n = self.plan[self.k]
        s = (self.i + 1) / n
        sm = s * s * (3 - 2 * s)                       # smoothstep
        pos = a + (b - a) * sm
        yaw = y0 + np.angle(np.exp(1j * (y1 - y0))) * sm
        self.g = g0 + (g1 - g0) * s
        seed = self._seed
        seed[e.arm_q] = self.q_cmd
        q, pe, re = e.ik.solve(seed, pos, pinch_rot(yaw), iters=60)
        self.q_cmd = q
        self.phase = name
        self.i += 1
        if self.i >= n:
            self.k += 1
            self.i = 0
        return np.concatenate([self.q_cmd, [self.g]])

    @property
    def done(self):
        return self.k >= len(self.plan)
