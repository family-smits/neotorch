"""Real-life realism pass: micro-AO, skin/fabric response, wetness, grit."""

from __future__ import annotations

from typing import Set

import numpy as np

from .life import make_alive, paint_life
from .mesh import Mesh
from .photoreal import apply_photoreal_materials, photoreal_refine
from .shading import compute_vertex_normals, fbm_noise
from .smooth import laplacian_smooth, refine

REAL_WORDS = {
    "real", "reallife", "realistisch", "realistic", "reality", "echt",
    "echte", "lifelike", "levensecht", "levensechte", "irl", "photo",
    "fotorealistisch", "hyperreal", "hyperrealistisch",
}


def wants_reallife(modifiers: Set[str]) -> bool:
    return bool(modifiers & {"reallife", "photoreal", "realistic", "cinematic", "alive", "ultra"})


def apply_reallife(mesh: Mesh, obj: str = "neural", seed: int = 0, ultra: bool = True) -> Mesh:
    """Maximale real-life look: refine → life → materials → micro detail."""
    if mesh.vertex_count == 0:
        return mesh
    # Alleen light refine als mesh nog "lowpoly" is — voorkomt 300k+ verts
    if ultra and mesh.vertex_count < 18_000:
        mesh = refine(mesh, levels=1, smooth_iters=1)
    mesh = photoreal_refine(mesh, strength="high")
    mesh = make_alive(mesh, seed=seed, strength=1.35)
    mesh = apply_photoreal_materials(mesh, obj=obj, seed=seed + 3)
    mesh = paint_life(mesh, seed=seed + 5, strength=1.3)
    mesh = _micro_realism(mesh, seed=seed + 9)
    mesh = laplacian_smooth(mesh, iterations=1, lambda_=0.22)
    return mesh


def _micro_realism(mesh: Mesh, seed: int) -> Mesh:
    albedo = mesh.colors.copy() if mesh.colors is not None else np.full((mesh.vertex_count, 3), 0.62)
    n = compute_vertex_normals(mesh)
    v = mesh.vertices
    noise = fbm_noise(v * 2.4, seed=seed, octaves=5)
    pore = fbm_noise(v * 9.0, seed=seed + 11, octaves=3)
    dirt = fbm_noise(v * 1.1 + 3.7, seed=seed + 23, octaves=3)

    # Bent-normal AO
    center = v.mean(axis=0)
    radial = v - center
    rlen = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(rlen, 1e-9)
    cavity = np.clip(1.0 - np.sum(n * radial, axis=1), 0.0, 1.0)
    ao = 1.0 - 0.55 * cavity

    # Contact darkening near ground
    y = v[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    yn = (y - y0) / max(1e-9, y1 - y0)
    ground_contact = np.clip(1.0 - yn * 4.0, 0.0, 1.0) ** 2

    # Poriën / grit
    albedo = albedo * (0.94 + 0.1 * noise - 0.05 * (pore - 0.5))[:, None]
    albedo *= ao[:, None]
    albedo *= (1.0 - 0.22 * ground_contact * dirt)[:, None]

    # Specular edge (dielectric)
    fres = (1.0 - np.clip(np.abs(n[:, 2]), 0, 1)) ** 2
    albedo = albedo + 0.06 * fres[:, None] * np.array([0.95, 0.95, 0.98])

    # Hemispheric sky/ground bounce bake
    up = np.clip(n[:, 1] * 0.5 + 0.5, 0, 1)
    sky = np.array([0.55, 0.62, 0.78])
    gnd = np.array([0.42, 0.38, 0.32])
    bounce = sky * up[:, None] + gnd * (1.0 - up)[:, None]
    albedo = 0.82 * albedo + 0.18 * bounce * albedo.mean(axis=1, keepdims=True)

    return mesh.paint(np.clip(albedo, 0.015, 1.0))
