"""Prompt-conditionele mesh-generatie met NeoTorch (geen externe AI)."""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

from .mesh import Mesh
from .nn import Linear, Module, ReLU, Sequential, Tanh
from .primitives import box, cone, cylinder, pyramid, sphere, torus
from .prompt import PromptPlan, token_embedding
from .tensor import Tensor


class PromptEncoder(Module):
    """Tokens → latent vector."""

    def __init__(self, in_dim: int = 64, latent_dim: int = 48, seed: int = 0):
        self.net = Sequential(
            Linear(in_dim, 96, seed=seed),
            ReLU(),
            Linear(96, latent_dim, seed=seed + 7),
            Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ShapeField(Module):
    """
    Implicit radius field: (latent, direction) → radius.
    Elke prompt krijgt zo een unieke silhouet.
    """

    def __init__(self, latent_dim: int = 48, seed: int = 0):
        self.net = Sequential(
            Linear(latent_dim + 3, 96, seed=seed),
            ReLU(),
            Linear(96, 96, seed=seed + 11),
            ReLU(),
            Linear(96, 32, seed=seed + 22),
            ReLU(),
            Linear(32, 1, seed=seed + 33),
            Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DetailField(Module):
    """Extra displacement langs normal voor spitse/gladde details."""

    def __init__(self, latent_dim: int = 48, seed: int = 0):
        self.net = Sequential(
            Linear(latent_dim + 3, 64, seed=seed),
            ReLU(),
            Linear(64, 3, seed=seed + 5),
            Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def encode_prompt(plan: PromptPlan, latent_dim: int = 48) -> np.ndarray:
    raw = token_embedding(plan.tokens, dim=64, seed=plan.seed)
    encoder = PromptEncoder(64, latent_dim, seed=plan.seed)
    latent = encoder(Tensor(raw[None, :])).numpy()[0]

    # Modifier-kanalen hard meeschrijven zodat woorden echt meetellen
    boost = np.zeros_like(latent)
    flags = {
        "tall": 0, "short": 1, "big": 2, "small": 3, "wide": 4, "thin": 5,
        "round": 6, "spiky": 7, "smooth": 8, "ornate": 9, "broken": 10,
        "flying": 11, "ancient": 12, "tech": 13, "cute": 14, "scary": 15,
    }
    for mod in plan.modifiers:
        if mod in flags:
            boost[flags[mod] % latent_dim] += 0.55
    for i, obj in enumerate(plan.objects):
        h = (hash(obj) & 0xFFFF) % latent_dim
        boost[h] += 0.4 / (1 + i)
    for i, part in enumerate(sorted(plan.parts)):
        h = (hash(part) & 0xFFFF) % latent_dim
        boost[(h + 3) % latent_dim] += 0.35
    for i, unk in enumerate(plan.unknown[:8]):
        h = (hash(unk) & 0xFFFF) % latent_dim
        boost[(h + 7) % latent_dim] += 0.3 / (1 + 0.2 * i)

    latent = latent + boost
    n = float(np.linalg.norm(latent))
    if n > 1e-8:
        latent = latent / n
    return latent.astype(np.float32)


def neural_blob(
    latent: np.ndarray,
    seed: int,
    segments: int = 28,
    rings: int = 18,
    strength: float = 1.0,
    modifiers: Optional[Set[str]] = None,
) -> Mesh:
    """Uv-sphere waarvan de radius per richting door ShapeField komt."""
    modifiers = modifiers or set()
    field = ShapeField(latent_dim=len(latent), seed=seed)
    detail = DetailField(latent_dim=len(latent), seed=seed + 101)

    # Basisellipsoid uit modifiers
    sx = sy = sz = 1.0
    if "tall" in modifiers:
        sy *= 1.55
        sx *= 0.85
        sz *= 0.85
    if "short" in modifiers:
        sy *= 0.65
        sx *= 1.15
        sz *= 1.15
    if "wide" in modifiers:
        sx *= 1.45
        sz *= 1.25
    if "thin" in modifiers:
        sx *= 0.65
        sz *= 0.65
    if "big" in modifiers:
        sx *= 1.25
        sy *= 1.25
        sz *= 1.25
    if "small" in modifiers:
        sx *= 0.7
        sy *= 0.7
        sz *= 0.7
    if "round" in modifiers:
        strength *= 0.55
    if "spiky" in modifiers:
        strength *= 1.45
    if "smooth" in modifiers:
        strength *= 0.4
    if "scary" in modifiers:
        strength *= 1.2
        sy *= 1.1
    if "cute" in modifiers:
        strength *= 0.7
        sy *= 0.9

    verts = []
    dirs = []
    for i in range(rings + 1):
        v = i / rings
        phi = np.pi * v
        for j in range(segments):
            u = j / segments
            theta = 2.0 * np.pi * u
            d = np.array(
                [
                    np.sin(phi) * np.cos(theta),
                    np.cos(phi),
                    np.sin(phi) * np.sin(theta),
                ],
                dtype=np.float32,
            )
            dirs.append(d)
            # placeholder
            verts.append(d)

    dirs_arr = np.asarray(dirs, dtype=np.float32)
    batch = np.concatenate(
        [np.repeat(latent[None, :], len(dirs_arr), axis=0), dirs_arr],
        axis=1,
    )
    # Zachtere radius-variatie → geen scheuren/gaten
    raw_r = field(Tensor(batch)).numpy().reshape(-1)
    radius = 0.9 + 0.28 * raw_r * min(strength, 1.2)

    freqs = 2 + int(abs(latent[0]) * 3)
    phase = float(latent[1]) * np.pi
    for i, d in enumerate(dirs_arr):
        ang = np.arctan2(d[2], d[0])
        bump = 0.06 * strength * np.sin(freqs * ang + phase) * (1.0 - abs(d[1]))
        if "spiky" in modifiers:
            bump += 0.1 * max(0.0, np.sin(freqs * 2 * ang)) ** 3
        if "ornate" in modifiers:
            bump += 0.04 * np.sin(8 * ang + d[1] * 6)
        if "broken" in modifiers:
            bump -= 0.06 * max(0.0, np.sin(3 * ang + phase))
        radius[i] = max(0.45, radius[i] + bump)

    # Alleen kleine tangentiale detail — niet hard naar binnen duwen
    det = detail(Tensor(batch)).numpy() * (0.04 * strength)
    # Projecteer detail loodrecht op de normal (dirs) zodat oppervlak dicht blijft
    det = det - dirs_arr * np.sum(det * dirs_arr, axis=1, keepdims=True)

    scale = np.array([sx, sy, sz], dtype=np.float64)
    vertices = dirs_arr * radius[:, None] * scale[None, :] + det

    faces = []
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            # Consistente winding (CCW van buitenaf)
            if i == 0:
                faces.append([a, c, d])
            elif i == rings - 1:
                faces.append([a, b, c])
            else:
                faces.append([a, b, c])
                faces.append([a, c, d])

    return Mesh(vertices, np.array(faces, dtype=np.int32), name="neural")


def apply_latent_deform(
    mesh: Mesh,
    latent: np.ndarray,
    seed: int,
    amount: float = 0.08,
    modifiers: Optional[Set[str]] = None,
) -> Mesh:
    if mesh.vertex_count == 0:
        return mesh
    modifiers = modifiers or set()
    if "smooth" in modifiers:
        amount *= 0.4
    if "spiky" in modifiers:
        amount *= 1.15
    if "ornate" in modifiers:
        amount *= 1.05

    detail = DetailField(latent_dim=len(latent), seed=seed + 55)
    verts = mesh.vertices.copy()
    center = verts.mean(axis=0)
    centered = verts - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-9:
        scale = 1.0
    normed = (centered / scale).astype(np.float32)
    batch = np.concatenate(
        [np.repeat(latent[None, :], len(normed), axis=0), normed],
        axis=1,
    )
    # Beperk offset en houd 'm vooral langs de radial (geen gaten)
    raw = detail(Tensor(batch)).numpy()
    radial = normed
    along = np.sum(raw * radial, axis=1, keepdims=True) * radial
    offsets = along * (amount * scale)

    # Stretch volgens modifiers
    stretch = np.ones(3, dtype=np.float64)
    if "tall" in modifiers:
        stretch[1] *= 1.35
    if "short" in modifiers:
        stretch[1] *= 0.75
    if "wide" in modifiers:
        stretch[0] *= 1.3
        stretch[2] *= 1.2
    if "thin" in modifiers:
        stretch[0] *= 0.7
        stretch[2] *= 0.7
    if "big" in modifiers:
        stretch *= 1.2
    if "small" in modifiers:
        stretch *= 0.8

    deformed = centered * stretch + offsets + center
    if "flying" in modifiers:
        deformed[:, 1] += 0.25 * scale
    return Mesh(
        deformed,
        mesh.faces.copy(),
        name=mesh.name,
        colors=None if mesh.colors is None else mesh.colors.copy(),
    )


def attach_parts(
    base: Mesh,
    parts: Set[str],
    latent: np.ndarray,
    seed: int,
    detail: int = 16,
) -> Mesh:
    """Voeg herkenbare onderdelen toe die in de prompt genoemd zijn."""
    if not parts:
        return base
    rng = np.random.default_rng(seed)
    pieces = [base]
    # Schaal t.o.v. huidige mesh
    mins = base.vertices.min(axis=0)
    maxs = base.vertices.max(axis=0)
    size = maxs - mins
    cx = (mins[0] + maxs[0]) * 0.5
    top = maxs[1]
    bottom = mins[1]
    mid_y = (mins[1] + maxs[1]) * 0.5
    depth = size[2] * 0.5

    if "wings" in parts:
        span = size[0] * 0.9 + 0.4
        wing_l = box(span, size[1] * 0.12, 0.08).translate([-(size[0] * 0.5 + span * 0.35), mid_y, 0])
        wing_r = box(span, size[1] * 0.12, 0.08).translate([size[0] * 0.5 + span * 0.35, mid_y, 0])
        pieces += [wing_l, wing_r]
    if "legs" in parts:
        for ox in (-size[0] * 0.25, size[0] * 0.25):
            leg = cylinder(0.08 + abs(latent[2]) * 0.05, size[1] * 0.55, segments=detail)
            pieces.append(leg.translate([ox, bottom - size[1] * 0.15, 0]))
    if "arms" in parts:
        for ox in (-size[0] * 0.55, size[0] * 0.55):
            arm = box(0.12, size[1] * 0.45, 0.12).translate([ox, mid_y, 0])
            pieces.append(arm)
    if "eyes" in parts:
        er = 0.07 + abs(latent[3]) * 0.04
        pieces.append(sphere(er, 10, 8).translate([cx - size[0] * 0.15, top - size[1] * 0.25, depth + 0.05]))
        pieces.append(sphere(er, 10, 8).translate([cx + size[0] * 0.15, top - size[1] * 0.25, depth + 0.05]))
    if "horns" in parts:
        for ox in (-size[0] * 0.15, size[0] * 0.15):
            pieces.append(cone(0.08, size[1] * 0.35, detail).translate([ox, top + size[1] * 0.1, 0]))
    if "tail" in parts:
        pieces.append(cylinder(0.07, size[0] * 0.7, detail).translate([0, bottom + size[1] * 0.2, -depth - 0.2]))
    if "spikes" in parts:
        for i in range(4):
            ox = (i - 1.5) * size[0] * 0.2
            pieces.append(cone(0.06, 0.25 + rng.random() * 0.15, 10).translate([ox, top, 0]))
    if "roof" in parts:
        pieces.append(pyramid(max(size[0], size[2]) * 1.05, size[1] * 0.45).translate([cx, top, 0]))
    if "door" in parts:
        pieces.append(box(size[0] * 0.22, size[1] * 0.4, 0.06).translate([cx, bottom + size[1] * 0.25, depth]))
    if "windows" in parts:
        for ox in (-size[0] * 0.25, size[0] * 0.25):
            pieces.append(box(0.18, 0.18, 0.05).translate([ox, mid_y + size[1] * 0.1, depth]))
    if "wheels" in parts:
        for ox in (-size[0] * 0.35, size[0] * 0.35):
            for oz in (-size[2] * 0.35, size[2] * 0.35):
                pieces.append(cylinder(0.14, 0.08, detail).translate([ox, bottom, oz]))
    if "hat" in parts:
        pieces.append(cylinder(size[0] * 0.35, size[1] * 0.15, detail).translate([cx, top + 0.05, 0]))
        pieces.append(cylinder(size[0] * 0.2, size[1] * 0.35, detail).translate([cx, top + size[1] * 0.25, 0]))

    return Mesh.merge(pieces, name=base.name)
