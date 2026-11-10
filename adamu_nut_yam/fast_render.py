"""Faster camera rendering for CPU-only machines (software OpenGL), for the hosted eval page.

The robot meshes have 1.4 M triangles, which costs ~0.45 s per 112 x 112 image on llvmpipe with 2 vCPU. With
RENDER_MAX_FACES set (2000 on the Space; RENDER_SHADOW=1024 also shrinks the shadow maps), make_renderer() builds
the OpenGL context from a copy of the model whose large untextured meshes are decimated to at most that many faces,
then renders scenes of the ORIGINAL model with it. MuJoCo uploads mesh geometry once, at context creation, and
takes everything else (poses, colours, lights, cameras) from the model and data at update_scene time, so physics is
untouched. 2000 / 1024 brings a frame to ~0.08 s with a mean pixel difference of 1.5 / 255. Without
RENDER_MAX_FACES this is plain mujoco.Renderer.
"""
import copy
import os

import mujoco
import numpy as np


def cluster(V, F, max_faces):
    """Vertex clustering: merge vertices on a grid, coarsening it until at most max_faces triangles remain."""
    cell = float(np.linalg.norm(V.max(0) - V.min(0))) / 400
    while True:
        key, inv = np.unique(np.floor(V / cell).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        C = np.zeros((len(key), 3)); np.add.at(C, inv, V); C /= np.bincount(inv, minlength=len(key))[:, None]
        G = inv[F]
        G = G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])]
        G = G[np.unique(np.sort(G, 1), axis=0, return_index=True)[1]]
        if len(G) <= max_faces:
            return C.astype(np.float32), G.astype(np.int32)
        cell *= 1.3


def decimated_copy(m, max_faces):
    import fast_simplification
    m2 = copy.deepcopy(m)
    for i in range(m.nmesh):
        fn = int(m.mesh_facenum[i])
        if fn <= max_faces or m.mesh_texcoordnum[i] > 0:
            continue
        va, vn, fa, na = m.mesh_vertadr[i], m.mesh_vertnum[i], m.mesh_faceadr[i], m.mesh_normaladr[i]
        V, F = m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn]
        U, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)       # weld the split (sharp-edge) vertices
        F = inv.reshape(-1)[F]
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        V2, F2 = fast_simplification.simplify(U.astype(np.float32), F.astype(np.int32),
                                              target_reduction=max(0.0, 1 - max_faces / len(F)))
        V2, F2 = np.asarray(V2, np.float32), np.asarray(F2, np.int32)
        if len(F2) > 1.5 * max_faces:               # quadric simplification stalls on meshes made of many small parts
            V2, F2 = cluster(V2, F2, max_faces)
        fnorm = np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]])
        N = np.zeros_like(V2)
        for k in range(3):
            np.add.at(N, F2[:, k], fnorm)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        m2.mesh_vert[va:va + len(V2)] = V2; m2.mesh_vertnum[i] = len(V2)
        m2.mesh_normal[na:na + len(V2)] = N; m2.mesh_normalnum[i] = len(V2)
        m2.mesh_face[fa:fa + len(F2)] = F2; m2.mesh_facenormal[fa:fa + len(F2)] = F2; m2.mesh_facenum[i] = len(F2)
    return m2


def make_renderer(m, height, width):
    max_faces = int(os.environ.get("RENDER_MAX_FACES", 0))
    if not max_faces:
        return mujoco.Renderer(m, height, width)
    m2 = decimated_copy(m, max_faces)
    m2.vis.quality.shadowsize = int(os.environ.get("RENDER_SHADOW", m.vis.quality.shadowsize))
    r = mujoco.Renderer(m2, height, width)
    r._model = m            # scenes come from the real model; only the uploaded mesh buffers are the light ones
    r._light_model = m2
    return r
