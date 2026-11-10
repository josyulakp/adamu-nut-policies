#!/usr/bin/env python3
"""Export the two MuJoCo scenes (plain YAM gripper, FlexiTac YAM gripper) as browser geometry for the eval page:
static/scene_{yam,tac}.json (one entry per visible geom: body, type, size, local pose, colour, mesh index) and
static/meshes_{yam,tac}.bin (float32 vertices + uint32 faces). Body indices match MjData.xpos of the same scene,
which is what the workers stream. Run once (server.py runs it if the files are missing)."""
import hashlib
import json
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCENES = {"yam": REPO / "adamu_nut_yam" / "assets" / "scene.xml", "tac": REPO / "adamu_nut_tactile" / "assets" / "scene.xml"}
MAX_FACES = 3000          # robot meshes are decimated to this, as in the published viewer (~5 MB per scene)
TYPE = {mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}


def export(variant, xml, out):
    m = mujoco.MjModel.from_xml_path(str(xml))
    blob, records, index, geoms = bytearray(), [], {}, []

    def add_mesh(mid):
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = np.asarray(m.mesh_vert[va:va + vn], np.float32); F = np.asarray(m.mesh_face[fa:fa + fn], np.uint32)
        if len(F) > MAX_FACES:
            V, F = fast_simplification.simplify(V, F, target_reduction=1 - MAX_FACES / len(F))
            V, F = np.asarray(V, np.float32), np.asarray(F, np.uint32)
            vn, fn = len(V), len(F)
        key = hashlib.sha1(V.tobytes() + F.tobytes()).hexdigest()
        if key not in index:
            r = {"v": len(blob), "nv": int(vn)}; blob.extend(V.tobytes())
            r["f"] = len(blob); r["nf"] = int(fn); blob.extend(F.tobytes())
            index[key] = len(records); records.append(r)
        return index[key]

    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        if t not in TYPE or m.geom_group[g] == 3:
            continue
        rgba = m.mat_rgba[m.geom_matid[g]].copy() if m.geom_matid[g] >= 0 else m.geom_rgba[g].copy()
        if rgba[3] < 0.05:
            continue
        name = m.geom(g).name
        if name == "floor":
            rgba = np.array([0.55, 0.58, 0.62, 1.0])
        if name.endswith("_pad_vis"):
            rgba = np.array([0.95, 0.55, 0.15, 1.0])            # FlexiTac pads
        e = {"b": int(m.geom_bodyid[g]), "t": TYPE[t], "s": [round(float(x), 5) for x in m.geom_size[g]],
             "p": [round(float(x), 6) for x in m.geom_pos[g]], "q": [round(float(x), 6) for x in m.geom_quat[g]],
             "c": [round(float(x), 3) for x in rgba], "n": name}
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            e["m"] = add_mesh(int(m.geom_dataid[g]))
        geoms.append(e)
    info = {"nbody": int(m.nbody), "geoms": geoms, "meshes": records, "nut_body": m.body("nut").id,
            "target_body": m.body("target").id, "look": [0.30, -0.14, 1.02]}
    (out / f"scene_{variant}.json").write_text(json.dumps(info, separators=(",", ":")))
    (out / f"meshes_{variant}.bin").write_bytes(bytes(blob))
    return len(geoms), len(blob)


def main():
    out = HERE / "static"; out.mkdir(exist_ok=True)
    for v, xml in SCENES.items():
        n, b = export(v, xml, out)
        print(f"{v}: {n} geoms, {b / 1e6:.1f} MB of meshes")


if __name__ == "__main__":
    main()
