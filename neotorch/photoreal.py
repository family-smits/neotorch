"""Photoreal-achtige material pass (lokaal, zonder externe renderer)."""

from __future__ import annotations

from typing import Set

import numpy as np

from .mesh import Mesh
from .shading import compute_vertex_normals, fbm_noise
from .smooth import laplacian_smooth, refine


# Materiaal-presets: (roughness, metalness, sheen)
MATERIAL_PRESETS = {
    "dragon": (0.35, 0.15, 0.4),
    "robot": (0.25, 0.85, 0.55),
    "humanoid": (0.55, 0.02, 0.35),
    "quadruped": (0.7, 0.0, 0.25),
    "bird": (0.65, 0.05, 0.3),
    "tree": (0.85, 0.0, 0.15),
    "house": (0.75, 0.05, 0.2),
    "castle": (0.8, 0.1, 0.15),
    "crystal": (0.08, 0.05, 0.9),
    "vehicle": (0.3, 0.7, 0.5),
    "sword": (0.2, 0.9, 0.65),
    "boat": (0.7, 0.1, 0.2),
    "rocket": (0.35, 0.6, 0.45),
    "mushroom": (0.75, 0.0, 0.25),
    "mountain": (0.9, 0.0, 0.1),
    "ground": (0.95, 0.0, 0.05),
    "water": (0.05, 0.0, 0.85),
    "neural": (0.4, 0.2, 0.4),
}


def apply_photoreal_materials(mesh: Mesh, obj: str = "neural", seed: int = 0) -> Mesh:
    """
    Bake micro-detail, curvature/AO en materiaalrespons in vertexkleuren.
    Geeft een veel 'fotografischer' albedo voor de preview/export.
    """
    if mesh.vertex_count == 0:
        return mesh
    albedo = mesh.colors.copy() if mesh.colors is not None else np.full((mesh.vertex_count, 3), 0.7)
    normals = compute_vertex_normals(mesh)
    noise = fbm_noise(mesh.vertices, seed=seed, octaves=5)
    fine = fbm_noise(mesh.vertices * 1.7, seed=seed + 11, octaves=3)

    # Curvature proxy: afwijking normal vs radial
    center = mesh.vertices.mean(axis=0)
    radial = mesh.vertices - center
    rlen = np.linalg.norm(radial, axis=1, keepdims=True)
    rlen = np.maximum(rlen, 1e-9)
    radial_n = radial / rlen
    cavity = np.clip(1.0 - np.sum(normals * radial_n, axis=1), 0.0, 1.0)

    roughness, metal, sheen = MATERIAL_PRESETS.get(obj, MATERIAL_PRESETS["neural"])

    # Micro-surface
    micro = 0.88 + 0.18 * noise + 0.08 * (fine - 0.5)
    albedo = albedo * micro[:, None]

    # Ambient occlusion in holtes
    ao = 1.0 - 0.45 * cavity
    albedo *= ao[:, None]

    # Subsurface-achtige warmte voor organische materialen
    if metal < 0.2:
        warm = np.array([1.06, 0.98, 0.94])
        albedo = albedo * (0.92 + 0.08 * noise)[:, None] * warm

    # Metalen: albedo dichter bij specular tint, iets donkerder diffuse
    if metal > 0.4:
        albedo = albedo * (0.75 + 0.25 * (1.0 - roughness))
        # Edge sheen bake
        fresnel = np.clip(1.0 - np.abs(normals[:, 1]), 0.0, 1.0) ** 2
        albedo = albedo + sheen * 0.25 * fresnel[:, None] * np.clip(albedo + 0.15, 0, 1)

    # Kristal / water: helderder + iriserende tint
    if obj in ("crystal", "water"):
        iridescence = np.stack([
            0.5 + 0.5 * np.sin(noise * 6.28),
            0.5 + 0.5 * np.sin(noise * 6.28 + 2.1),
            0.5 + 0.5 * np.sin(noise * 6.28 + 4.2),
        ], axis=1)
        albedo = 0.65 * albedo + 0.35 * iridescence

    # Top-light bake (lichte studio fill in albedo)
    up = np.clip(normals[:, 1] * 0.5 + 0.5, 0.0, 1.0)
    albedo = albedo * (0.82 + 0.28 * up)[:, None]

    return mesh.paint(np.clip(albedo, 0.02, 1.0))


def photoreal_refine(mesh: Mesh, strength: str = "high") -> Mesh:
    """Extra gladheid — licht genoeg om bruikbaar te blijven."""
    if strength == "ultra":
        # Alleen smooth als al dense; anders 1 lichte refine
        if mesh.vertex_count < 12_000:
            mesh = refine(mesh, levels=1, smooth_iters=1)
        mesh = laplacian_smooth(mesh, iterations=2, lambda_=0.38)
    else:
        mesh = laplacian_smooth(mesh, iterations=2, lambda_=0.4)
    return mesh


def wants_photoreal(modifiers: Set[str]) -> bool:
    return bool(modifiers & {"photoreal", "realistic", "cinematic"})
