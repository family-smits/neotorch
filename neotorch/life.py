"""Leven: organische asymmetrie, zachte weefsels, levende huid-tint."""

from __future__ import annotations

from typing import Set

import numpy as np

from .mesh import Mesh
from .shading import compute_vertex_normals, fbm_noise
from .smooth import laplacian_smooth

# Objecten die standaard “levend” voelen
LIVING_OBJECTS = {
    "humanoid", "creature", "quadruped", "dragon", "bird", "robot",
    "fruit", "mushroom", "tree", "freeform", "neural",
}

ALIVE_WORDS = {
    "alive", "leven", "levend", "levende", "living", "organic", "organisch",
    "organische", "adem", "ademhaling", "breathing", "breathe", "pulse",
    "hart", "heart", "soul", "ziel", "vital", "vitaal", "bloed", "blood",
    "skin", "huid", "flesh", "vlees", "creature", "wezen", "beast",
}


def wants_life(modifiers: Set[str], obj: str = "") -> bool:
    if modifiers & {"alive", "organic", "cute", "scary"}:
        return True
    return obj in LIVING_OBJECTS


def make_alive(mesh: Mesh, seed: int = 0, strength: float = 1.0) -> Mesh:
    """
    Organische asymmetrie + weke-delen-ruis zodat het niet als star plastic oogt.
    """
    if mesh.vertex_count < 8:
        return mesh
    rng = np.random.default_rng(seed)
    v = mesh.vertices.copy()
    n = compute_vertex_normals(mesh)
    noise = fbm_noise(v * 1.35, seed=seed, octaves=4)
    fine = fbm_noise(v * 3.1, seed=seed + 19, octaves=3)

    # Lichte links/rechts asymmetrie (levende wezens zijn nooit perfect)
    asym = 0.035 * strength * np.sign(v[:, 0] + 1e-6) * (0.4 + fine)
    v[:, 0] = v[:, 0] + asym * np.abs(v[:, 1]) * 0.15

    # Weke delen: opblazen langs normal met organische ruis
    soft = (0.028 * strength) * (noise - 0.45)[:, None] * n
    # Meer op torso/buik (midden-hoogte)
    y = v[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    yn = (y - y0) / max(1e-9, y1 - y0)
    belly = np.exp(-((yn - 0.45) ** 2) / 0.08)[:, None]
    v = v + soft * (0.55 + 0.9 * belly)

    # Micro-spier/ribbing
    ang = np.arctan2(v[:, 2], v[:, 0])
    rib = 0.012 * strength * np.sin(ang * (5 + (seed % 3)) + yn * 4.0)
    v = v + n * rib[:, None]

    # Heel lichte random jitter (niet mechanisch)
    v = v + rng.normal(0, 0.004 * strength, size=v.shape)

    out = Mesh(v, mesh.faces.copy(), name=mesh.name)
    if mesh.colors is not None:
        out = out.paint(mesh.colors.copy())
    out = laplacian_smooth(out, iterations=1, lambda_=0.28)
    return out


def paint_life(mesh: Mesh, seed: int = 0, strength: float = 1.0) -> Mesh:
    """Warme subsurface-achtige tint, blush, subtiele ‘bloed’-variatie."""
    if mesh.vertex_count == 0:
        return mesh
    albedo = mesh.colors.copy() if mesh.colors is not None else np.full((mesh.vertex_count, 3), 0.65)
    n = compute_vertex_normals(mesh)
    noise = fbm_noise(mesh.vertices * 2.0, seed=seed, octaves=4)
    fine = fbm_noise(mesh.vertices * 5.5, seed=seed + 7, octaves=3)

    # Warmte (SSS feel)
    warm = np.array([1.08, 0.96, 0.90])
    cool = np.array([0.92, 0.95, 1.05])
    mix = (0.55 + 0.45 * noise)[:, None]
    albedo = albedo * (warm * mix + cool * (1.0 - mix))

    # Blush op bolle / naar voren gerichte delen
    forward = np.clip(n[:, 2] * 0.5 + 0.5, 0, 1)
    up = np.clip(n[:, 1] * 0.5 + 0.5, 0, 1)
    blush = (0.12 * strength) * forward * (0.4 + 0.6 * up) * (0.7 + 0.3 * fine)
    albedo = albedo + blush[:, None] * np.array([0.35, 0.08, 0.05])

    # Holtes iets koeler/donkerder (diepte)
    cavity = np.clip(1.0 - up, 0, 1)
    albedo = albedo * (1.0 - 0.18 * strength * cavity)[:, None]

    # Micro-poriën / vlekjes
    speck = 0.04 * strength * (fine - 0.5)
    albedo = albedo + speck[:, None]

    return mesh.paint(np.clip(albedo, 0.02, 1.0))
