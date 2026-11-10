"""Hand-written Cartesian controller in the RL action space (sanity check that the task is solvable)."""
import numpy as np
from nut_rl_env import NutRLEnv


def heuristic(env):
    sim = env.sim; nut = sim.nut_pos(); tgt = sim.target_pos(); p, R = sim.eef_pose()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    want_yaw = sim.nut_yaw() + np.pi / 6
    dy = np.angle(np.exp(1j * 6 * (want_yaw - yaw))) / 6
    grasped = all(c > 0 for c in env.contacts())
    st = env.__dict__.setdefault("_hs", {"phase": "reach", "k": 0})
    a = np.zeros(5); g = -1
    if st["phase"] in ("lift", "carry", "lower") and not grasped and np.linalg.norm(nut - p) > 0.015:
        st["phase"] = "reach"                   # grasp lost (e.g. the nut was bumped away): start over
    if st["phase"] == "close" and np.linalg.norm(nut[:2] - p[:2]) > 0.012:
        st["phase"] = "reach"
    ph = st["phase"]
    if ph == "reach":
        goal = nut + [0, 0, 0.06]
        if np.linalg.norm(p - goal) < 0.01 and abs(dy) < 0.05: st["phase"] = "descend"
    if ph == "descend":
        goal = nut + [0, 0, 0.004]
        if np.linalg.norm(p - goal) < 0.004: st["phase"] = "close"; st["k"] = 0
    if ph == "close":
        goal = nut + [0, 0, 0.004]; g = 0.7; st["k"] += 1
        if st["k"] > 12: st["phase"] = "lift"
    if ph == "lift":
        goal = np.array([nut[0], nut[1], env.z0 + 0.08]); g = 0.7
        if p[2] > env.z0 + 0.07: st["phase"] = "carry"
    if ph == "carry":
        goal = np.array([tgt[0], tgt[1], env.z0 + 0.08]) + (p - nut) * [1, 1, 0]; g = 0.7   # nut may sit off-centre in the grasp
        if np.linalg.norm(nut[:2] - tgt[:2]) < 0.006: st["phase"] = "lower"
    if ph == "lower":
        goal = np.array([tgt[0], tgt[1], env.z0 + 0.006]) + (p - nut) * [1, 1, 0]; g = 0.7
        if nut[2] < env.z0 + 0.003: st["phase"] = "open"; st["k"] = 0
    if ph == "open":
        goal = p.copy(); g = -1; st["k"] += 1
        if st["k"] > 8: st["phase"] = "retreat"
    if ph == "retreat":
        goal = p + [0, 0, 0.02]; g = -1
    # actions already queued by the control latency will still move the command: aim from the predicted command
    # (without this the P-loop oscillates at 1-2 steps of latency: 100% / 46% / 10% success at latency 0 / 1 / 2)
    pend = np.sum([q[:4] for q in env.queue], axis=0) if env.queue else np.zeros(4)
    a[:3] = np.clip((goal - (env.pos_cmd + pend[:3] * 0.01)) / 0.01, -1, 1)
    a[3] = np.clip((dy - pend[3] * 0.08) / 0.08, -1, 1); a[4] = g
    return a


if __name__ == "__main__":
    import time
    env = NutRLEnv(seed=0, randomize=True)
    ok, t0, steps = 0, time.time(), 0
    for s in range(10):
        env.reset(s); env._hs = {"phase": "reach", "k": 0}; R = 0
        while True:
            o, r, done, info = env.step(heuristic(env)); R += r; steps += 1
            if done: break
        ok += info["success"]; print(s, info, "return %.0f" % R, "steps", env.t, env.P)
    print("success", ok, "/10", "steps/s %.0f" % (steps / (time.time() - t0)))
