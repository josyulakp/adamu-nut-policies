"""FlexiTac-V2 tactile pads for the MuJoCo scene, using the same model as the FlexiTac Isaac Sim sensor
(isaaclab/sensors/warp_sdf_tactile in https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation):

  * a 12 x 32 grid of taxels at 2 mm pitch on each pad (cell centres, as in _create_local_grid_points);
  * every step each taxel's signed distance to the nut surface is computed (negative = inside the nut);
  * force = clamp(stiffness * max(-sdf, 0), 0, max_force) / max_force  in [0, 1].

The taxel plane sits PROUD m in front of the rigid pad face. Rigid contacts barely interpenetrate, so the offset is
what stands in for elastomer compression (the Isaac sensor's normal_offset plays the same role). With
stiffness 5000 N/m and max_force 10 N (FlexiTac Isaac defaults) a reading saturates at 2 mm of compression.

The nut SDF is exact inside the convex hull (max over the hull's face planes); the bore does not matter because the
fingers only touch the outside. Output layout: (2 pads, 32 rows along the finger from tip to root, 12 columns).
"""
import mujoco
import numpy as np
import trimesh

ROWS, COLS, PITCH = 32, 12, 0.002
STIFFNESS, MAX_FORCE = 5000.0, 10.0
PROUD = 0.0005          # taxel plane offset in front of the pad face (virtual elastomer thickness), m


def grid_local(rows=ROWS, cols=COLS, pitch=PITCH, proud=PROUD):
    """Taxel positions in the pad-site frame: x across (cols), z along the finger (rows, row 0 at the tip), y normal."""
    x = (np.arange(cols) - (cols - 1) / 2) * pitch
    z = (np.arange(rows) - (rows - 1) / 2) * pitch
    zz, xx = np.meshgrid(z, x, indexing="ij")
    return np.stack([xx.ravel(), np.full(xx.size, proud), zz.ravel()], -1)      # (rows*cols, 3)


class HullSDF:
    def __init__(self, stl):
        hull = trimesh.load(stl, force="mesh").convex_hull
        n, o = [], []
        for f, normal in zip(hull.facets, hull.facets_normal):       # merge coplanar triangles into faces
            n.append(normal); o.append(float(normal @ hull.vertices[hull.faces[f[0]][0]]))
        tri = np.setdiff1d(np.arange(len(hull.faces)), np.concatenate(hull.facets) if len(hull.facets) else [])
        for t in tri:
            n.append(hull.face_normals[t]); o.append(float(hull.face_normals[t] @ hull.vertices[hull.faces[t][0]]))
        self.n, self.o = np.array(n), np.array(o)

    def __call__(self, p):
        """p (N, 3) in the nut frame -> signed distance (exact for p inside / on the hull, a lower bound outside)."""
        return (p @ self.n.T - self.o).max(-1)


class FlexiTacPads:
    def __init__(self, m, nut_body="nut", nut_stl=None, sites=("left_tactile", "right_tactile")):
        from pathlib import Path
        self.m = m
        self.sites = [m.site(s).id for s in sites]
        self.nut = m.body(nut_body).id
        self.sdf = HullSDF(nut_stl or Path(__file__).resolve().parents[1] / "assets" / "meshes" / "nut_collision.stl")
        self.local = grid_local()
        hv = trimesh.load(nut_stl or Path(__file__).resolve().parents[1] / "assets" / "meshes" / "nut_collision.stl", force="mesh").vertices
        self.r2 = float((np.linalg.norm(hv[:, :2], axis=1).max() + 1e-4) ** 2)
        self.hz = float(np.abs(hv[:, 2]).max() + 1e-4)

    def points_world(self, d):
        return np.stack([d.site_xpos[s] + self.local @ d.site_xmat[s].reshape(3, 3).T for s in self.sites])

    def read(self, d):
        """-> (2, ROWS, COLS) normalised forces in [0, 1]."""
        # early out: pads (64 mm long) cannot touch the nut (~13 mm radius) if their centres are > 6 cm away
        if min(np.linalg.norm(d.site_xpos[s] - d.xpos[self.nut]) for s in self.sites) > 0.06:
            return np.zeros((2, ROWS, COLS), np.float32)
        pw = self.points_world(d)                                                  # (2, P, 3)
        R = d.xmat[self.nut].reshape(3, 3)
        pn = (pw - d.xpos[self.nut]) @ R                                           # world -> nut frame
        pn = pn.reshape(-1, 3)
        # only taxels inside the nut's bounding cylinder can be inside the nut (exact; saves the 536-plane test)
        near = (np.einsum("ij,ij->i", pn[:, :2], pn[:, :2]) < self.r2) & (np.abs(pn[:, 2]) < self.hz)
        sd = np.ones(len(pn))
        if near.any():
            sd[near] = self.sdf(pn[near])
        sd = sd.reshape(2, -1)
        f = np.clip(STIFFNESS * np.maximum(-sd, 0), 0, MAX_FORCE) / MAX_FORCE
        return f.reshape(2, ROWS, COLS).astype(np.float32)
