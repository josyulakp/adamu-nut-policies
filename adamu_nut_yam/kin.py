"""Kinematics helpers for ADAM-U's right arm with the YAM linear_4310 gripper: grasp frame, grip map, DLS IK."""
import mujoco
import numpy as np

ARM = ["shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right", "elbow_Right",
       "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]
WRIST_BODY = "wristRollRight"

# Grasp frame in the wrist-roll body frame: centre between the two YAM pads (171.2 mm below the wrist-roll
# joint, measured from the model), x along the pad separation, z the approach direction (out of the gripper).
PINCH_POS = np.array([0.0, 0.0, -0.1712])
_a = np.array([1.0, 0.0, 0.0])
_n = np.array([0.0, 0.0, -1.0])
PINCH_ROT = np.column_stack([_a, np.cross(_n, _a), _n])     # columns: pinch axis, side, approach

# Grip: g in [0, 1] -> linear_4310 finger position (m per finger). g=0 fully open (95 mm gap), g=1 is 5 mm past
# closed (squeeze). Contact on the 22 mm nut flats happens at g ~ 0.69.
FINGER_OPEN, FINGER_SQUEEZE = 0.0475, -0.005


def finger_pos(g: float) -> float:
    return FINGER_OPEN - float(np.clip(g, 0.0, 1.0)) * (FINGER_OPEN - FINGER_SQUEEZE)


def grip_targets(g: float) -> dict:
    return {"gripper": finger_pos(g)}


class ArmIK:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM], 1]
        self.wrist = model.body(WRIST_BODY).id

    def pinch_pose(self, qpos):
        d = self.d
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, d)
        R = d.xmat[self.wrist].reshape(3, 3)
        return d.xpos[self.wrist] + R @ PINCH_POS, R @ PINCH_ROT

    def solve(self, qpos0, pos, rot, iters=200, tol=1e-4, rot_weight=0.3, free_yaw_about_z=False):
        """DLS IK for the pinch frame. Returns (arm q, pos err m, rot err rad)."""
        m, d = self.m, self.d
        q = qpos0.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            R = d.xmat[self.wrist].reshape(3, 3)
            p = d.xpos[self.wrist] + R @ PINCH_POS
            Rp = R @ PINCH_ROT
            ep = pos - p
            er = 0.5 * sum(np.cross(Rp[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, p, self.wrist)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            e = np.concatenate([ep, rot_weight * er])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            dq = np.clip(dq, -0.2, 0.2)
            q[self.qadr] = np.clip(q[self.qadr] + dq, self.lo, self.hi)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R = d.xmat[self.wrist].reshape(3, 3)
        p = d.xpos[self.wrist] + R @ PINCH_POS
        Rp = R @ PINCH_ROT
        rerr = np.arccos(np.clip((np.trace(Rp.T @ rot) - 1) / 2, -1, 1))
        return q[self.qadr].copy(), float(np.linalg.norm(pos - p)), float(rerr)


def pinch_rot(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """World rotation for the pinch frame: approach straight down (tilted by `tilt` toward the
    robot), pinch axis horizontal at heading `yaw`."""
    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    n = np.array([0.0, 0.0, -1.0])
    if tilt:
        toward = np.array([-1.0, 0.0, 0.0]); toward -= a * (a @ toward)
        if np.linalg.norm(toward) > 1e-6:
            toward /= np.linalg.norm(toward)
            n = np.cos(tilt) * n + np.sin(tilt) * toward
    n -= a * (a @ n); n /= np.linalg.norm(n)
    return np.column_stack([a, np.cross(n, a), n])
