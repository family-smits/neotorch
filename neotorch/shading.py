"""Vertex normals + procedural noise voor realistischer shading."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def compute_vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted smooth vertex normals."""
    verts = mesh.vertices
    faces = mesh.faces
    normals = np.zeros_like(verts)
    if mesh.face_count == 0:
        return normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    # Area-weight zit al in cross magnitude
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-12)
    return normals / lens


def value_noise3(points: np.ndarray, seed: int = 0, scale: float = 4.0) -> np.ndarray:
    """Goedkope 3D value-noise ∈ [0,1] op vertexposities."""
    rng = np.random.default_rng(seed)
    # Lattice hash via afgeronde coords
    p = points * scale
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    # Smoothstep
    u = f * f * (3.0 - 2.0 * f)

    def corner(ox, oy, oz):
        idx = i0 + np.array([ox, oy, oz], dtype=np.int64)
        # Deterministische hash
        h = (idx[:, 0] * 374761393 + idx[:, 1] * 668265263 + idx[:, 2] * 1274126177 + seed) & 0xFFFFFFFF
        # Map naar [0,1)
        return (h.astype(np.float64) % 10000) / 10000.0

    c000 = corner(0, 0, 0)
    c100 = corner(1, 0, 0)
    c010 = corner(0, 1, 0)
    c110 = corner(1, 1, 0)
    c001 = corner(0, 0, 1)
    c101 = corner(1, 0, 1)
    c011 = corner(0, 1, 1)
    c111 = corner(1, 1, 1)

    x00 = c000 * (1 - u[:, 0]) + c100 * u[:, 0]
    x10 = c010 * (1 - u[:, 0]) + c110 * u[:, 0]
    x01 = c001 * (1 - u[:, 0]) + c101 * u[:, 0]
    x11 = c011 * (1 - u[:, 0]) + c111 * u[:, 0]
    y0 = x00 * (1 - u[:, 1]) + x10 * u[:, 1]
    y1 = x01 * (1 - u[:, 1]) + x11 * u[:, 1]
    return y0 * (1 - u[:, 2]) + y1 * u[:, 2]


def fbm_noise(points: np.ndarray, seed: int = 0, octaves: int = 4) -> np.ndarray:
    total = np.zeros(len(points), dtype=np.float64)
    amp = 0.5
    freq = 1.0
    for o in range(octaves):
        total += amp * value_noise3(points, seed=seed + o * 97, scale=3.0 * freq)
        amp *= 0.5
        freq *= 2.0
    return np.clip(total, 0.0, 1.0)
