"""Scene-compositie: grond, props, layout van meerdere acteurs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

import numpy as np

from .colors import RGB, OBJECT_PALETTES, paint_mesh
from .mesh import Mesh
from .primitives import box, cylinder, sphere
from .smooth import ellipsoid, refine, rounded_box, tapered_cylinder


@dataclass
class Actor:
    obj: str
    colors: List[RGB] = field(default_factory=list)
    modifiers: Set[str] = field(default_factory=set)
    label: str = ""  # origineel prompt-woord (voor freeform / slug)


SCENE_WORDS = {
    "scene", "landschap", "landscape", "wereld", "world", "diorama",
    "omgeving", "setting", "plaats", "plaatsje", "tafereel",
}

ENV_MAP = {
    "forest": "forest", "bos": "forest", "woud": "forest", "jungle": "forest",
    "village": "village", "dorp": "village", "town": "village",
    "beach": "beach", "strand": "beach", "kust": "beach",
    "farm": "farm", "boerderij": "farm", "weide": "farm",
    "city": "city", "stad": "city",
    "garden": "garden", "tuin": "garden",
    "desert": "desert", "woestijn": "desert",
    "lake": "lake", "meer": "lake", "vijver": "lake",
    "cave": "cave", "grot": "cave",
    # Weer-woorden zijn géén losse env hier — die gaan via weather.py
}


def detect_environment(tokens: Sequence[str], modifiers: Set[str]) -> str | None:
    for t in tokens:
        if t in ENV_MAP:
            return ENV_MAP[t]
    if "scene" in modifiers or any(t in SCENE_WORDS for t in tokens):
        return "garden"
    return None


def is_scene_prompt(tokens: Sequence[str], n_objects: int, env: str | None) -> bool:
    if env is not None:
        return True
    if n_objects >= 2:
        return True
    if any(t in SCENE_WORDS for t in tokens):
        return True
    return False


def _ground(kind: str, size: float = 4.5, detail: int = 16) -> Mesh:
    if kind == "beach":
        mesh = ellipsoid(size * 0.55, 0.08, size * 0.45, detail, 8)
        mesh = paint_mesh(mesh, OBJECT_PALETTES["ground"][1:2] + [(0.92, 0.85, 0.62)], seed=1, obj="ground")
    elif kind == "water" or kind == "lake":
        mesh = ellipsoid(size * 0.5, 0.05, size * 0.4, detail, 8)
        mesh = paint_mesh(mesh, OBJECT_PALETTES["water"], seed=2, obj="water")
    elif kind == "desert":
        mesh = ellipsoid(size * 0.55, 0.1, size * 0.5, detail, 8)
        mesh = paint_mesh(mesh, [(0.9, 0.78, 0.45), (0.85, 0.65, 0.35), (0.95, 0.88, 0.6)], seed=3, obj="ground")
    else:
        mesh = rounded_box(size, 0.1, size * 0.85, max(10, detail // 2)).translate([0, -0.02, 0])
        mesh = paint_mesh(mesh, OBJECT_PALETTES["ground"], seed=4, obj="ground")
    mesh.name = "ground"
    return mesh


def _prop_tree(rng: np.random.Generator, scale: float = 0.55) -> Mesh:
    d = 12
    trunk = tapered_cylinder(0.06 * scale, 0.04 * scale, 0.7 * scale, d).translate([0, 0.35 * scale, 0])
    leaves = ellipsoid(0.28 * scale, 0.25 * scale, 0.28 * scale, d, 8).translate([0, 0.75 * scale, 0])
    m = Mesh.merge([trunk, leaves], name="prop_tree")
    m = paint_mesh(m, OBJECT_PALETTES["tree"], seed=int(rng.integers(1, 1e6)), obj="tree")
    return m


def _prop_rock(rng: np.random.Generator, scale: float = 0.3) -> Mesh:
    m = ellipsoid(scale, scale * 0.55, scale * 0.8, 10, 6)
    return paint_mesh(m, OBJECT_PALETTES["rock"], seed=int(rng.integers(1, 1e6)), obj="rock")


def _prop_bush(rng: np.random.Generator, scale: float = 0.25) -> Mesh:
    m = ellipsoid(scale, scale * 0.7, scale, 10, 6)
    return paint_mesh(m, [(0.2, 0.55, 0.22), (0.15, 0.4, 0.15), (0.35, 0.65, 0.25)], seed=int(rng.integers(1, 1e6)), obj="tree")


def environment_props(env: str, rng: np.random.Generator, detail: int = 12) -> List[Mesh]:
    props: List[Mesh] = []
    ground = _ground("beach" if env == "beach" else ("lake" if env == "lake" else ("desert" if env == "desert" else "grass")), detail=detail)
    props.append(ground)

    if env == "forest":
        for _ in range(5):
            x, z = rng.uniform(-1.8, 1.8), rng.uniform(-1.5, 1.5)
            if abs(x) < 0.5 and abs(z) < 0.5:
                continue
            props.append(_prop_tree(rng, 0.45 + rng.random() * 0.35).translate([x, 0, z]))
        for _ in range(4):
            props.append(_prop_bush(rng, 0.18 + rng.random() * 0.1).translate([rng.uniform(-2, 2), 0.05, rng.uniform(-1.6, 1.6)]))
    elif env == "village":
        for i, x in enumerate([-1.6, 1.6]):
            house = rounded_box(0.55, 0.4, 0.5, 10).translate([x, 0.2, -1.2])
            roof = box(0.65, 0.08, 0.6).translate([x, 0.45, -1.2])
            h = Mesh.merge([house, roof], name="hut")
            props.append(paint_mesh(h, OBJECT_PALETTES["house"], seed=10 + i, obj="house"))
        for _ in range(3):
            props.append(_prop_tree(rng, 0.4).translate([rng.uniform(-2, 2), 0, rng.uniform(0.8, 1.6)]))
    elif env == "beach":
        water = _ground("lake", size=3.2, detail=detail).translate([0, -0.04, 1.3])
        props.append(water)
        for _ in range(4):
            props.append(_prop_rock(rng, 0.15 + rng.random() * 0.12).translate([rng.uniform(-1.8, 1.8), 0.05, rng.uniform(-1.2, 0.2)]))
    elif env == "farm":
        fence = box(3.5, 0.08, 0.06).translate([0, 0.15, -1.5])
        props.append(paint_mesh(fence, [(0.55, 0.35, 0.15), (0.4, 0.25, 0.1)], seed=20, obj="house"))
        for x in (-1.2, 0, 1.2):
            props.append(paint_mesh(cylinder(0.04, 0.35, 8).translate([x, 0.18, -1.5]), [(0.45, 0.3, 0.12)], seed=21, obj="tree"))
        for _ in range(3):
            props.append(_prop_bush(rng, 0.2).translate([rng.uniform(-1.8, 1.8), 0.05, rng.uniform(0.5, 1.5)]))
    elif env == "garden":
        for _ in range(3):
            props.append(_prop_tree(rng, 0.4).translate([rng.uniform(-1.8, 1.8), 0, rng.uniform(-1.4, 1.4)]))
        for _ in range(5):
            props.append(_prop_bush(rng, 0.15 + rng.random() * 0.08).translate([rng.uniform(-2, 2), 0.04, rng.uniform(-1.5, 1.5)]))
    elif env == "desert":
        for _ in range(5):
            props.append(_prop_rock(rng, 0.2 + rng.random() * 0.2).translate([rng.uniform(-1.8, 1.8), 0.05, rng.uniform(-1.4, 1.4)]))
        cactus = Mesh.merge([
            cylinder(0.08, 0.55, 10).translate([1.2, 0.28, 0.6]),
            cylinder(0.05, 0.25, 8).translate([1.35, 0.4, 0.6]),
        ], name="cactus")
        props.append(paint_mesh(cactus, [(0.25, 0.55, 0.25), (0.15, 0.4, 0.15)], seed=30, obj="tree"))
    elif env == "lake":
        for _ in range(3):
            props.append(_prop_tree(rng, 0.45).translate([rng.uniform(-1.8, 1.8), 0, rng.uniform(-1.6, -0.6)]))
        for _ in range(3):
            props.append(_prop_rock(rng, 0.15).translate([rng.uniform(-1.5, 1.5), 0.04, rng.uniform(0.2, 1.0)]))
    elif env == "city":
        for i, (x, z) in enumerate([(-1.5, -1.0), (1.4, -1.1), (-1.3, 1.0), (1.5, 0.9)]):
            b = rounded_box(0.5 + (i % 2) * 0.2, 0.7 + i * 0.15, 0.45, 10).translate([x, 0.4 + i * 0.05, z])
            props.append(paint_mesh(b, [(0.45, 0.48, 0.55), (0.7, 0.75, 0.85), (0.2, 0.55, 0.85)], seed=40 + i, obj="castle"))
    elif env == "cave":
        arch = ellipsoid(1.8, 1.2, 1.4, 14, 10).scale([1, 0.7, 1]).translate([0, 0.6, -0.8])
        props.append(paint_mesh(arch, OBJECT_PALETTES["rock"], seed=50, obj="rock"))
        for _ in range(4):
            props.append(_prop_rock(rng, 0.2).translate([rng.uniform(-1.5, 1.5), 0.05, rng.uniform(-0.5, 1.2)]))
    else:
        for _ in range(2):
            props.append(_prop_tree(rng, 0.4).translate([rng.uniform(-1.8, 1.8), 0, rng.uniform(-1.4, 1.4)]))

    return props


def layout_positions(n: int, radius: float = 1.15) -> List[Tuple[float, float, float]]:
    """Plaats acteurs in een boog/cirkel zodat ze niet overlappen."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]
    if n == 2:
        return [(-radius * 0.7, 0.0, 0.0), (radius * 0.7, 0.0, 0.0)]
    pos = []
    for i in range(n):
        a = -0.7 + 1.4 * i / max(n - 1, 1)
        # boog naar voren
        x = np.sin(a) * radius * 1.1
        z = -np.cos(a) * radius * 0.35
        pos.append((float(x), 0.0, float(z)))
    return pos


def place_on_ground(mesh: Mesh) -> Mesh:
    """Zet onderkant van mesh op y≈0."""
    ymin = float(mesh.vertices[:, 1].min())
    return mesh.translate([0.0, -ymin, 0.0])


def compose_scene(
    actors: Sequence[Mesh],
    env: str | None,
    seed: int,
    detail: int = 12,
) -> Mesh:
    rng = np.random.default_rng(seed)
    env_name = env or ("garden" if len(actors) >= 2 else None)
    pieces: List[Mesh] = []
    if env_name:
        pieces.extend(environment_props(env_name, rng, detail=detail))

    positions = layout_positions(len(actors), radius=1.2 if env_name else 1.0)
    for mesh, (x, y, z) in zip(actors, positions):
        m = place_on_ground(mesh).translate([x, y, z])
        # Lichte unieke rotatie rond Y
        ang = float(rng.uniform(-0.35, 0.35))
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)
        m = Mesh(m.vertices @ rot.T, m.faces.copy(), name=m.name, colors=None if m.colors is None else m.colors.copy())
        pieces.append(m)

    return Mesh.merge(pieces, name="scene")
