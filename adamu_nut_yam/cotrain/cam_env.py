"""Image-observation version of the YAM nut task, with two domains for sim-and-real co-training.

domain="sim"   the simulator we would normally train in: nominal dynamics with moderate per-episode
               randomisation, and per-episode visual randomisation (lighting, table material, nut/target
               colour, camera pose jitter, colour jitter, sensor noise).
domain="real"  a stand-in for the physical robot until real recordings exist. One fixed appearance the sim
               randomisation does not contain (unseen table material, warm light, head camera 2 cm / 2 deg off
               its nominal calibration, blur, noise) and dynamics effects the sim does not model or only
               partly covers:
                 - 150 ms action latency (sim: 0-100 ms)
                 - softer joint servos (kp x0.75) and imperfect gravity compensation (92 %): real tracking error
                 - extra joint friction / damping, pad friction 0.7, nut mass 36 g
                 - encoder noise on the joint readings.
               Its demonstrations are "teleoperated": they use either of the two reachable pairs of flats at
               random (multimodal), at a varying speed, and only successful takes are kept.

domain="ideal" evaluation only: the nominal simulator with nothing randomised or perturbed (default table, lights,
               colours and camera pose, no image or encoder noise, no latency, nominal masses / friction / servos).

Policy observation (no privileged state): two RGB images (head, wrist) + proprio (7 arm joints + grip command).
Action: unchanged, 7 joint targets + grip.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env import NutPickPlaceEnv, ScriptedExpert  # noqa: E402
from fast_render import make_renderer  # noqa: E402

SCENE_CAM = ROOT / "assets" / "scene_cam.xml"
CAMS = ("head_cam", "wrist_cam")
IMG = 112
SIM_TABLES = ["table", "tab_wood", "tab_check", "tab_noise", "tab_dark", "tab_white"]
REAL_TABLE = "tab_mdf"


class CamEnv(NutPickPlaceEnv):
    def __init__(self, domain="sim", img=IMG):
        super().__init__(scene=SCENE_CAM)
        assert domain in ("sim", "real", "ideal")
        self.domain, self.img = domain, img
        m = self.m
        self.renderer = make_renderer(m, img, img)
        self.vopt = mujoco.MjvOption(); self.vopt.sitegroup[:] = 0
        self.cam_ids = [m.camera(c).id for c in CAMS]
        self.head_cam = m.camera("head_cam").id
        self.pads = [m.geom("left_pad").id, m.geom("right_pad").id]
        self.lights = [m.light("key").id, m.light("fill").id]
        self.target_geom = m.geom("target_pad").id
        self.nut_vis = m.geom("nut_visual").id
        self.robot_bodies = [b for b in range(m.nbody) if m.body_gravcomp[b] > 0]
        self.nom = dict(
            friction=m.geom_friction[self.pads].copy(), armature=m.dof_armature.copy(), damping=m.dof_damping.copy(),
            frictionloss=m.dof_frictionloss.copy(), gravcomp=m.body_gravcomp.copy(), mass=m.body_mass.copy(),
            gainprm=m.actuator_gainprm.copy(), biasprm=m.actuator_biasprm.copy(),
            cam_pos=m.cam_pos[self.head_cam].copy(), cam_quat=m.cam_quat[self.head_cam].copy(),
            light_diffuse=m.light_diffuse[self.lights].copy(), light_dir=m.light_dir[self.lights].copy(),
            light_pos=m.light_pos[self.lights].copy(), target_rgba=m.geom_rgba[self.target_geom].copy(),
            nut_rgba=m.mat_rgba[m.geom_matid[self.nut_vis]].copy(), headlight=m.vis.headlight.diffuse.copy())
        self.lat, self.queue = 0, deque()
        self.img_fx = {}

    # ------------------------------------------------------------------ domain parameters
    def _restore(self):
        m, n = self.m, self.nom
        m.geom_friction[self.pads] = n["friction"]; m.dof_armature[:] = n["armature"]; m.dof_damping[:] = n["damping"]
        m.dof_frictionloss[:] = n["frictionloss"]; m.body_gravcomp[:] = n["gravcomp"]; m.body_mass[:] = n["mass"]
        m.actuator_gainprm[:] = n["gainprm"]; m.actuator_biasprm[:] = n["biasprm"]
        m.cam_pos[self.head_cam] = n["cam_pos"]; m.cam_quat[self.head_cam] = n["cam_quat"]
        m.light_diffuse[self.lights] = n["light_diffuse"]; m.light_dir[self.lights] = n["light_dir"]
        m.light_pos[self.lights] = n["light_pos"]; m.geom_rgba[self.target_geom] = n["target_rgba"]
        m.mat_rgba[m.geom_matid[self.nut_vis]] = n["nut_rgba"]; m.vis.headlight.diffuse[:] = n["headlight"]

    def _arm_servo_scale(self, k):
        m = self.m
        for i in self.arm_act:
            m.actuator_gainprm[i, 0] *= k; m.actuator_biasprm[i, 1] *= k; m.actuator_biasprm[i, 2] *= np.sqrt(k)

    def _cam_offset(self, dpos, drot_deg, rng):
        m = self.m
        m.cam_pos[self.head_cam] += dpos
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, axis, np.deg2rad(drot_deg))
        out = np.zeros(4); mujoco.mju_mulQuat(out, m.cam_quat[self.head_cam].copy(), q)
        m.cam_quat[self.head_cam] = out

    def _apply_domain(self, rng):
        m = self.m
        self._restore()
        nut_body = self.nut_body
        dof = self.arm_dof
        if self.domain == "ideal":
            self.lat = 0
            m.geom_matid[self.table_geom] = m.mat(SIM_TABLES[0]).id
            self.img_fx = dict(gain=np.ones(3), gamma=1.0, noise=0.0, blur=False)
            self.enc_noise = 0.0
        elif self.domain == "sim":
            self.lat = int(rng.integers(0, 3))
            m.geom_friction[self.pads, 0] = rng.uniform(0.6, 1.2)
            m.body_mass[nut_body] = rng.uniform(0.025, 0.040)
            m.dof_damping[dof] += rng.uniform(0, 0.5)
            m.dof_frictionloss[dof] += rng.uniform(0, 0.2)
            # visuals
            m.geom_matid[self.table_geom] = m.mat(str(rng.choice(SIM_TABLES))).id
            m.light_diffuse[self.lights[0]] = rng.uniform(0.4, 1.0) * rng.uniform(0.85, 1.0, 3)
            m.light_dir[self.lights[0]] = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), -1])
            m.light_diffuse[self.lights[1]] = rng.uniform(0.0, 0.5, 3)
            m.vis.headlight.diffuse[:] = rng.uniform(0.3, 0.7)
            m.geom_rgba[self.target_geom, :3] = np.clip(self.nom["target_rgba"][:3] + rng.normal(0, 0.08, 3), 0, 1)
            m.mat_rgba[m.geom_matid[self.nut_vis], :3] = np.clip(self.nom["nut_rgba"][:3] * rng.uniform(0.7, 1.15), 0, 1)
            self._cam_offset(rng.normal(0, 0.01, 3), rng.normal(0, 1.0), rng)
            self.img_fx = dict(gain=rng.uniform(0.85, 1.15, 3), gamma=rng.uniform(0.85, 1.15), noise=rng.uniform(0, 6),
                               blur=False)
            self.enc_noise = 0.0
        else:
            self.lat = 3
            self._arm_servo_scale(0.75)
            m.body_gravcomp[self.robot_bodies] = 0.92
            m.geom_friction[self.pads, 0] = 0.7
            m.body_mass[nut_body] = 0.036
            m.dof_damping[dof] += 0.6
            m.dof_frictionloss[dof] += 0.3
            m.dof_armature[dof] += 0.02
            m.geom_matid[self.table_geom] = m.mat(REAL_TABLE).id
            m.light_diffuse[self.lights[0]] = np.array([1.0, 0.88, 0.72]) * rng.uniform(0.8, 0.9)
            m.light_dir[self.lights[0]] = [0.35, 0.25, -1]
            m.light_diffuse[self.lights[1]] = [0.18, 0.18, 0.22]
            m.vis.headlight.diffuse[:] = 0.45
            m.cam_pos[self.head_cam] += [0.012, -0.015, 0.008]          # fixed calibration error
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.3, 1.0, 0.2]) / 1.06, np.deg2rad(2.0))
            out = np.zeros(4); mujoco.mju_mulQuat(out, m.cam_quat[self.head_cam].copy(), q); m.cam_quat[self.head_cam] = out
            self.img_fx = dict(gain=np.array([1.06, 1.0, 0.9]), gamma=1.1, noise=4.0, blur=True)
            self.enc_noise = 0.002
        # masses / armature changed at run time: recompute the derived constants (inverse weights used by the
        # constraint solver). Without this the nut contact goes unstable for some masses.
        mujoco.mj_setConst(m, self.d)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None):
        rng = np.random.default_rng(None if seed is None else seed + 7_777_777)
        self.rng_obs = np.random.default_rng(None if seed is None else seed + 3_333_333)
        self._apply_domain(rng)
        super().reset(seed)
        self.queue = deque()
        self.hold = np.concatenate([self.d.qpos[self.arm_q], [0.0]])
        return self.observe()

    def step(self, action):
        self.queue.append(np.asarray(action, dtype=np.float64).copy())
        a = self.queue.popleft() if len(self.queue) > self.lat else self.hold
        self.hold = a
        _, done, info = super().step(a)
        return self.observe(), done, info

    # ------------------------------------------------------------------ observation
    def render_images(self):
        ims = []
        for c in self.cam_ids:
            self.renderer.update_scene(self.d, c, self.vopt)
            ims.append(self.renderer.render())
        x = np.stack(ims).astype(np.float32) / 255.0
        fx = self.img_fx
        x = np.clip(x * fx["gain"], 0, 1) ** fx["gamma"]
        if fx["blur"]:
            x = (x + np.roll(x, 1, 1) + np.roll(x, 1, 2) + np.roll(np.roll(x, 1, 1), 1, 2)) / 4
        x = x * 255 + self.rng_obs.normal(0, fx["noise"], x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)

    def proprio(self):
        q = self.d.qpos[self.arm_q] + self.rng_obs.normal(0, self.enc_noise, 7) if self.enc_noise else self.d.qpos[self.arm_q]
        return np.concatenate([q, [self.grip_cmd]]).astype(np.float32)

    def observe(self):
        return {"images": self.render_images(), "proprio": self.proprio()}


def make_expert(env: CamEnv, rng):
    """Scripted demonstrator. In the real domain it imitates a teleoperator: random choice between the two
    reachable flat pairs (multimodal grasp yaw) and a varying speed."""
    if env.domain == "real":
        def choice(cands, r):
            d = np.abs(np.angle(np.exp(1j * (cands - np.deg2rad(30)))))
            return cands[np.argsort(d)[r.integers(0, 2)]]
        return ScriptedExpert(env, rng, speed=rng.uniform(0.8, 1.1), yaw_choice=choice)
    return ScriptedExpert(env, rng)
