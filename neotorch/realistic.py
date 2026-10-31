"""Realistischere / complexere object-assemblages."""

from __future__ import annotations

from typing import Set

import numpy as np

from .mesh import Mesh
from .neural_mesh import neural_blob
from .primitives import box, cone, cylinder, pyramid, sphere, torus
from .smooth import capsule, ellipsoid, refine, rounded_box, tapered_cylinder


def _scale(mods: Set[str]) -> tuple[float, float, float]:
    sx = sy = sz = 1.0
    if "tall" in mods:
        sy *= 1.4
        sx *= 0.92
        sz *= 0.92
    if "short" in mods:
        sy *= 0.72
    if "wide" in mods:
        sx *= 1.3
        sz *= 1.18
    if "thin" in mods:
        sx *= 0.72
        sz *= 0.72
    if "big" in mods:
        sx *= 1.22
        sy *= 1.22
        sz *= 1.22
    if "small" in mods:
        sx *= 0.78
        sy *= 0.78
        sz *= 0.78
    return sx, sy, sz


def build_house(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    w, depth, h = 1.5 * sx, 1.25 * sz, 1.0 * sy
    body = rounded_box(w, h, depth, segments=d).translate([0, h * 0.5, 0])
    # Overstekend dak
    roof = pyramid(max(w, depth) * 1.18, 0.7 * sy).translate([0, h + 0.02, 0])
    eave = box(max(w, depth) * 1.2, 0.06, max(w, depth) * 1.2).translate([0, h, 0])
    door = rounded_box(0.28 * sx, 0.55 * sy, 0.08, 10).translate([0, 0.3 * sy, depth * 0.5])
    win_l = box(0.22, 0.22, 0.06).translate([-w * 0.28, 0.65 * sy, depth * 0.5])
    win_r = box(0.22, 0.22, 0.06).translate([w * 0.28, 0.65 * sy, depth * 0.5])
    sill_l = box(0.26, 0.04, 0.08).translate([-w * 0.28, 0.52 * sy, depth * 0.5])
    sill_r = box(0.26, 0.04, 0.08).translate([w * 0.28, 0.52 * sy, depth * 0.5])
    chimney = tapered_cylinder(0.1, 0.12, 0.55 * sy, d).translate([w * 0.3, h + 0.35 * sy, -depth * 0.15])
    parts = [body, eave, roof, door, win_l, win_r, sill_l, sill_r, chimney]
    if "ornate" in mods or "ancient" in mods:
        parts.append(box(w * 1.05, 0.08, depth * 1.05).translate([0, 0.08, 0]))  # plint
        parts.append(cylinder(0.05, 0.35 * sy, 10).translate([0, 0.55 * sy, depth * 0.52]))  # deurknop-stijl
    return refine(Mesh.merge(parts, name="house"), levels=1, smooth_iters=2)


def build_tree(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    trunk_h = 1.1 * sy
    trunk = tapered_cylinder(0.16 * sx, 0.09 * sx, trunk_h, d).translate([0, trunk_h * 0.5, 0])
    # Wortels
    roots = []
    for a in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        roots.append(
            capsule(0.045 * sx, 0.28 * sx, 10, 6)
            .scale([1.0, 0.35, 1.0])
            .translate([np.cos(a) * 0.18 * sx, 0.06, np.sin(a) * 0.18 * sz])
        )
    # Bladerdak: meerdere overlappende ellipsoïden
    canopy = []
    base_y = trunk_h * 0.85
    for i, (ox, oy, oz, s) in enumerate([
        (0.0, 0.35, 0.0, 0.55),
        (0.28, 0.5, 0.1, 0.4),
        (-0.25, 0.55, -0.12, 0.38),
        (0.1, 0.75, -0.2, 0.32),
        (-0.05, 0.7, 0.22, 0.34),
        (0.0, 0.95, 0.0, 0.28),
    ]):
        canopy.append(
            ellipsoid(s * sx, s * 0.85 * sy, s * sz, d, max(8, d // 2)).translate(
                [ox * sx, base_y + oy * sy, oz * sz]
            )
        )
    mesh = Mesh.merge([trunk, *roots, *canopy], name="tree")
    return refine(mesh, levels=1, smooth_iters=3)


def build_tower(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(14, detail)
    h = 2.3 * sy
    r = 0.42 * sx
    shaft = tapered_cylinder(r * 1.05, r * 0.9, h, d).translate([0, h * 0.5, 0])
    collar = torus(r * 1.05, 0.06, d, max(8, d // 2)).translate([0, h * 0.72, 0])
    balcony = cylinder(r * 1.25, 0.08, d).translate([0, h * 0.78, 0])
    roof = cone(r * 1.35, 0.85 * sy, d).translate([0, h + 0.25, 0])
    # Kantelen
    merlons = []
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        merlons.append(
            box(0.1, 0.18, 0.1).translate([np.cos(a) * r * 1.05, h * 0.88, np.sin(a) * r * 1.05])
        )
    return refine(Mesh.merge([shaft, collar, balcony, roof, *merlons], name="tower"), 1, 2)


def build_castle(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    keep = rounded_box(1.7 * sx, 1.35 * sy, 1.7 * sz, d).translate([0, 0.7 * sy, 0])
    roof = box(1.8 * sx, 0.12, 1.8 * sz).translate([0, 1.4 * sy, 0])
    parts = [keep, roof]
    for ox, oz in [(-0.95 * sx, -0.95 * sz), (0.95 * sx, -0.95 * sz), (-0.95 * sx, 0.95 * sz), (0.95 * sx, 0.95 * sz)]:
        th = 1.85 * sy
        parts.append(tapered_cylinder(0.3 * sx, 0.26 * sx, th, max(12, d)).translate([ox, th * 0.5, oz]))
        parts.append(cone(0.34 * sx, 0.55 * sy, max(12, d)).translate([ox, th + 0.12, oz]))
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            parts.append(box(0.08, 0.14, 0.08).translate([ox + np.cos(a) * 0.28 * sx, th * 0.92, oz + np.sin(a) * 0.28 * sz]))
    # Poort
    parts.append(rounded_box(0.55 * sx, 0.85 * sy, 0.35, 12).translate([0, 0.42 * sy, 0.9 * sz]))
    parts.append(box(0.35 * sx, 0.55 * sy, 0.08).translate([0, 0.35 * sy, 1.05 * sz]))
    # Muur-ringen
    parts.append(box(2.2 * sx, 0.55 * sy, 0.18).translate([0, 0.3 * sy, -1.0 * sz]))
    parts.append(box(2.2 * sx, 0.55 * sy, 0.18).translate([0, 0.3 * sy, 1.0 * sz]))
    return refine(Mesh.merge(parts, name="castle"), 1, 2)


def build_robot(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    torso = rounded_box(0.75 * sx, 0.95 * sy, 0.45 * sz, d).translate([0, 1.0 * sy, 0])
    chest = rounded_box(0.55 * sx, 0.25 * sy, 0.2 * sz, 10).translate([0, 1.15 * sy, 0.2 * sz])
    pelvis = rounded_box(0.7 * sx, 0.25 * sy, 0.4 * sz, 10).translate([0, 0.55 * sy, 0])
    head = rounded_box(0.42 * sx, 0.4 * sy, 0.4 * sz, d).translate([0, 1.7 * sy, 0])
    neck = cylinder(0.1 * sx, 0.12 * sy, 10).translate([0, 1.48 * sy, 0])
    eye_l = sphere(0.07, 12, 8).translate([-0.1 * sx, 1.72 * sy, 0.2 * sz])
    eye_r = sphere(0.07, 12, 8).translate([0.1 * sx, 1.72 * sy, 0.2 * sz])
    visor = box(0.32 * sx, 0.08, 0.06).translate([0, 1.72 * sy, 0.22 * sz])
    # Armen met gewrichten
    parts = [torso, chest, pelvis, head, neck, eye_l, eye_r, visor]
    for side in (-1, 1):
        parts.append(sphere(0.12 * sx, 12, 8).translate([side * 0.48 * sx, 1.25 * sy, 0]))  # schouder
        parts.append(capsule(0.09 * sx, 0.45 * sy, d, 8).translate([side * 0.58 * sx, 0.95 * sy, 0]))
        parts.append(sphere(0.09 * sx, 10, 8).translate([side * 0.58 * sx, 0.7 * sy, 0]))  # elleboog
        parts.append(capsule(0.08 * sx, 0.4 * sy, d, 8).translate([side * 0.58 * sx, 0.45 * sy, 0]))
        parts.append(rounded_box(0.12 * sx, 0.14 * sy, 0.1 * sz, 8).translate([side * 0.58 * sx, 0.22 * sy, 0]))
        # Benen
        parts.append(sphere(0.12 * sx, 12, 8).translate([side * 0.2 * sx, 0.42 * sy, 0]))
        parts.append(capsule(0.11 * sx, 0.4 * sy, d, 8).translate([side * 0.2 * sx, 0.22 * sy, 0]))
        parts.append(sphere(0.1 * sx, 10, 8).translate([side * 0.2 * sx, 0.05 * sy, 0]))
        parts.append(capsule(0.1 * sx, 0.35 * sy, d, 8).translate([side * 0.2 * sx, -0.12 * sy, 0]))
        parts.append(rounded_box(0.18 * sx, 0.08, 0.28 * sz, 8).translate([side * 0.2 * sx, -0.32 * sy, 0.04]))
    if "tech" in mods or "ornate" in mods:
        parts.append(torus(0.2 * sx, 0.04, 14, 8).translate([0, 1.35 * sy, 0.22 * sz]))
    return refine(Mesh.merge(parts, name="robot"), 1, 2)


def build_humanoid(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    # Anatomischer: hoofd, nek, torso, heupen, ledematen
    head = ellipsoid(0.18 * sx, 0.22 * sy, 0.2 * sz, d, max(8, d // 2)).translate([0, 1.72 * sy, 0])
    neck = capsule(0.06 * sx, 0.12 * sy, 10, 6).translate([0, 1.52 * sy, 0])
    torso = ellipsoid(0.28 * sx, 0.38 * sy, 0.18 * sz, d, max(8, d // 2)).translate([0, 1.2 * sy, 0])
    pelvis = ellipsoid(0.26 * sx, 0.14 * sy, 0.16 * sz, 12, 8).translate([0, 0.85 * sy, 0])
    parts = [head, neck, torso, pelvis]
    # Oren / neus hint
    parts.append(ellipsoid(0.04, 0.05, 0.06, 8, 6).translate([0, 1.7 * sy, 0.18 * sz]))
    for side in (-1.0, 1.0):
        parts.append(sphere(0.1 * sx, 12, 8).translate([side * 0.32 * sx, 1.35 * sy, 0]))  # schouder
        parts.append(capsule(0.07 * sx, 0.42 * sy, d, 8).translate([side * 0.38 * sx, 1.05 * sy, 0]))
        parts.append(sphere(0.065 * sx, 10, 8).translate([side * 0.38 * sx, 0.82 * sy, 0]))
        parts.append(capsule(0.06 * sx, 0.38 * sy, d, 8).translate([side * 0.38 * sx, 0.58 * sy, 0]))
        parts.append(ellipsoid(0.05 * sx, 0.08 * sy, 0.03 * sz, 8, 6).translate([side * 0.38 * sx, 0.38 * sy, 0.02]))
        # Benen
        parts.append(sphere(0.1 * sx, 12, 8).translate([side * 0.12 * sx, 0.72 * sy, 0]))
        parts.append(capsule(0.09 * sx, 0.45 * sy, d, 8).translate([side * 0.12 * sx, 0.45 * sy, 0]))
        parts.append(sphere(0.08 * sx, 10, 8).translate([side * 0.12 * sx, 0.2 * sy, 0]))
        parts.append(capsule(0.08 * sx, 0.42 * sy, d, 8).translate([side * 0.12 * sx, -0.02 * sy, 0]))
        parts.append(ellipsoid(0.08 * sx, 0.05, 0.16 * sz, 10, 6).translate([side * 0.12 * sx, -0.25 * sy, 0.04]))
    if "cute" in mods:
        parts.append(ellipsoid(0.22 * sx, 0.08, 0.22 * sz, 10, 6).translate([0, 1.9 * sy, 0]))  # haar-cap
    if "ornate" in mods:
        parts.append(cylinder(0.22 * sx, 0.12 * sy, 12).translate([0, 1.95 * sy, 0]))
    return refine(Mesh.merge(parts, name="humanoid"), 1, 3)


def build_quadruped(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    body = ellipsoid(0.65 * sx, 0.28 * sy, 0.32 * sz, d, max(8, d // 2)).translate([0, 0.55 * sy, 0])
    chest = ellipsoid(0.35 * sx, 0.3 * sy, 0.3 * sz, 14, 10).translate([0.35 * sx, 0.58 * sy, 0])
    head = ellipsoid(0.22 * sx, 0.2 * sy, 0.18 * sz, d, 10).translate([0.72 * sx, 0.72 * sy, 0])
    snout = ellipsoid(0.14 * sx, 0.1 * sy, 0.1 * sz, 12, 8).translate([0.95 * sx, 0.65 * sy, 0])
    ear_l = cone(0.06, 0.14 * sy, 10).translate([0.65 * sx, 0.92 * sy, -0.1 * sz])
    ear_r = cone(0.06, 0.14 * sy, 10).translate([0.65 * sx, 0.92 * sy, 0.1 * sz])
    tail = capsule(0.05, 0.55 * sx, 12, 6).translate([-0.7 * sx, 0.6 * sy, 0])
    parts = [body, chest, head, snout, ear_l, ear_r, tail]
    for ox in (-0.35 * sx, 0.35 * sx):
        for oz in (-0.18 * sz, 0.18 * sz):
            parts.append(sphere(0.08, 10, 8).translate([ox, 0.4 * sy, oz]))
            parts.append(capsule(0.07, 0.38 * sy, d, 8).translate([ox, 0.2 * sy, oz]))
            parts.append(ellipsoid(0.07, 0.04, 0.12, 8, 6).translate([ox, 0.02, oz + 0.02 * np.sign(oz + 1e-6)]))
    if "cute" in mods:
        parts.append(sphere(0.04, 8, 6).translate([0.88 * sx, 0.7 * sy, 0.08 * sz]))
        parts.append(sphere(0.04, 8, 6).translate([0.88 * sx, 0.7 * sy, -0.08 * sz]))
    return refine(Mesh.merge(parts, name="quadruped"), 1, 3)


def build_dragon(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(14, detail)
    # Romp uit ellipsoïden in een boog
    parts = []
    for i, t in enumerate(np.linspace(-0.7, 0.55, 7)):
        s = 0.28 - abs(t) * 0.08
        parts.append(
            ellipsoid(s * 1.1 * sx, s * 0.85 * sy, s * sz, d, max(8, d // 2)).translate(
                [t * 1.3 * sx, 0.45 * sy + 0.08 * np.sin(t * 3), 0]
            )
        )
    neck = ellipsoid(0.35 * sx, 0.16 * sy, 0.16 * sz, d, 10).translate([0.75 * sx, 0.75 * sy, 0])
    head = ellipsoid(0.28 * sx, 0.18 * sy, 0.2 * sz, d, 10).translate([1.15 * sx, 0.9 * sy, 0])
    jaw = ellipsoid(0.2 * sx, 0.08 * sy, 0.14 * sz, 12, 8).translate([1.25 * sx, 0.78 * sy, 0])
    horn_l = cone(0.05, 0.28 * sy, 10).translate([1.05 * sx, 1.1 * sy, -0.08 * sz])
    horn_r = cone(0.05, 0.28 * sy, 10).translate([1.05 * sx, 1.1 * sy, 0.08 * sz])
    # Vleugels: botten + membraan-platen
    for side in (-1.0, 1.0):
        parts.append(capsule(0.04, 0.9 * sz, 12, 6).translate([0.1 * sx, 0.75 * sy, side * 0.55 * sz]))
        parts.append(capsule(0.035, 0.7 * sz, 10, 6).translate([-0.05 * sx, 0.95 * sy, side * 0.85 * sz]))
        wing = ellipsoid(0.45 * sx, 0.04, 0.7 * sz, 14, 8).translate([0.05 * sx, 0.85 * sy, side * 0.65 * sz])
        parts.append(wing)
    # Poten
    for ox in (-0.25 * sx, 0.35 * sx):
        for oz in (-0.22 * sz, 0.22 * sz):
            parts.append(capsule(0.07, 0.35 * sy, 12, 6).translate([ox, 0.22 * sy, oz]))
            parts.append(cone(0.06, 0.12, 8).translate([ox, 0.02, oz]))
    tail = []
    for i in range(5):
        t = -0.85 - i * 0.18
        s = 0.16 - i * 0.025
        tail.append(ellipsoid(s * sx, s * 0.7 * sy, s * 0.7 * sz, 12, 8).translate([t * sx, 0.4 * sy - i * 0.02, 0]))
    parts.extend([neck, head, jaw, horn_l, horn_r, *tail])
    return refine(Mesh.merge(parts, name="dragon"), 1, 2)


def build_bird(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    body = ellipsoid(0.38 * sx, 0.28 * sy, 0.25 * sz, d, 10)
    head = ellipsoid(0.16 * sx, 0.15 * sy, 0.14 * sz, 14, 10).translate([0.38 * sx, 0.12 * sy, 0])
    beak = cone(0.05, 0.16 * sx, 10).translate([0.55 * sx, 0.08 * sy, 0])
    eye_l = sphere(0.03, 8, 6).translate([0.42 * sx, 0.16 * sy, 0.1 * sz])
    eye_r = sphere(0.03, 8, 6).translate([0.42 * sx, 0.16 * sy, -0.1 * sz])
    wing_l = ellipsoid(0.35 * sx, 0.04, 0.55 * sz, 14, 8).translate([-0.05, 0.08, -0.4 * sz])
    wing_r = ellipsoid(0.35 * sx, 0.04, 0.55 * sz, 14, 8).translate([-0.05, 0.08, 0.4 * sz])
    tail = ellipsoid(0.25 * sx, 0.03, 0.2 * sz, 12, 6).translate([-0.4 * sx, 0.02, 0])
    return refine(Mesh.merge([body, head, beak, eye_l, eye_r, wing_l, wing_r, tail], name="bird"), 1, 3)


def build_vehicle(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    body = rounded_box(1.7 * sx, 0.42 * sy, 0.75 * sz, d).translate([0, 0.38 * sy, 0])
    cabin = rounded_box(0.75 * sx, 0.4 * sy, 0.7 * sz, d).translate([-0.15 * sx, 0.72 * sy, 0])
    hood = rounded_box(0.55 * sx, 0.2 * sy, 0.7 * sz, 12).translate([0.55 * sx, 0.48 * sy, 0])
    bumper = rounded_box(0.15, 0.12, 0.78 * sz, 8).translate([0.9 * sx, 0.28 * sy, 0])
    parts = [body, cabin, hood, bumper]
    for ox in (-0.55 * sx, 0.5 * sx):
        for oz in (-0.42 * sz, 0.42 * sz):
            parts.append(cylinder(0.18, 0.12, d).translate([ox, 0.18, oz]))
            parts.append(cylinder(0.08, 0.13, 10).translate([ox, 0.18, oz]))  # hub
    # Lichten
    parts.append(sphere(0.06, 10, 8).translate([0.88 * sx, 0.4 * sy, 0.28 * sz]))
    parts.append(sphere(0.06, 10, 8).translate([0.88 * sx, 0.4 * sy, -0.28 * sz]))
    return refine(Mesh.merge(parts, name="vehicle"), 1, 2)


def build_crystal(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    parts = []
    n = 8 if "ornate" in mods else 6
    for i in range(n):
        h = (0.7 + rng.random() * 1.1) * sy
        w = (0.12 + rng.random() * 0.18) * sx
        # Dubbele piramide ≈ octaëder
        top = pyramid(w * 2, h * 0.55).translate([0, h * 0.25, 0])
        bot = pyramid(w * 2, h * 0.45).scale([1, -1, 1]).translate([0, -h * 0.1, 0])
        shard = Mesh.merge([top, bot], name=f"shard{i}")
        ox = (rng.random() - 0.5) * 0.55 * sx
        oz = (rng.random() - 0.5) * 0.55 * sz
        oy = rng.random() * 0.15 * sy
        # Lichte rotatie via shear-achtige scale
        ang = (rng.random() - 0.5) * 0.4
        v = shard.vertices.copy()
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        shard = Mesh(v @ rot.T, shard.faces.copy(), name=shard.name).translate([ox, oy + h * 0.2, oz])
        parts.append(shard)
    base = ellipsoid(0.45 * sx, 0.12 * sy, 0.45 * sz, 14, 8)
    return refine(Mesh.merge([base, *parts], name="crystal"), 1, 1)


def build_sword(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    blade = tapered_cylinder(0.02 * sz, 0.06 * sx, 1.5 * sy, d).translate([0, 1.05 * sy, 0])
    # Blade plat maken
    blade = blade.scale([1.0, 1.0, 0.35])
    tip = cone(0.05 * sx, 0.25 * sy, d).scale([1.0, 1.0, 0.35]).translate([0, 1.9 * sy, 0])
    guard = rounded_box(0.55 * sx, 0.08, 0.14 * sz, 10).translate([0, 0.32 * sy, 0])
    handle = tapered_cylinder(0.05, 0.045, 0.4 * sy, d).translate([0, 0.08, 0])
    wrap = torus(0.055, 0.015, 12, 8).translate([0, 0.15, 0])
    pommel = ellipsoid(0.09, 0.09, 0.09, 12, 8).translate([0, -0.18, 0])
    return refine(Mesh.merge([blade, tip, guard, handle, wrap, pommel], name="sword"), 1, 2)


def build_rocket(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(14, detail)
    body = tapered_cylinder(0.32 * sx, 0.28 * sx, 1.5 * sy, d).translate([0, 0.9 * sy, 0])
    tip = cone(0.28 * sx, 0.55 * sy, d).translate([0, 1.9 * sy, 0])
    window = sphere(0.1, 12, 8).translate([0.22 * sx, 1.2 * sy, 0])
    parts = [body, tip, window]
    for a in np.linspace(0, 2 * np.pi, 3, endpoint=False):
        parts.append(
            box(0.08, 0.4 * sy, 0.35 * sz)
            .translate([np.cos(a) * 0.32 * sx, 0.25 * sy, np.sin(a) * 0.32 * sz])
        )
    exhaust = cylinder(0.2 * sx, 0.15, d).translate([0, 0.08, 0])
    flame = cone(0.18 * sx, 0.35 * sy, d).scale([1, -1, 1]).translate([0, -0.15, 0])
    parts.extend([exhaust, flame])
    return refine(Mesh.merge(parts, name="rocket"), 1, 2)


def build_mushroom(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    stem = tapered_cylinder(0.18 * sx, 0.12 * sx, 0.7 * sy, d).translate([0, 0.35 * sy, 0])
    cap = ellipsoid(0.55 * sx, 0.28 * sy, 0.55 * sz, d, max(8, d // 2)).translate([0, 0.85 * sy, 0])
    spots = [
        sphere(0.08, 8, 6).translate([0.2 * sx, 0.95 * sy, 0.15 * sz]),
        sphere(0.06, 8, 6).translate([-0.22 * sx, 0.9 * sy, -0.1 * sz]),
        sphere(0.07, 8, 6).translate([0.05 * sx, 1.0 * sy, -0.2 * sz]),
    ]
    return refine(Mesh.merge([stem, cap, *spots], name="mushroom"), 1, 3)


def build_boat(rng: np.random.Generator, mods: Set[str], detail: int) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(12, detail)
    hull = ellipsoid(0.95 * sx, 0.22 * sy, 0.4 * sz, d, 10).translate([0, 0.15 * sy, 0])
    # Plat afsnijden onderkant visueel door tweede hull
    deck = rounded_box(1.5 * sx, 0.08, 0.55 * sz, 12).translate([0, 0.28 * sy, 0])
    cabin = rounded_box(0.45 * sx, 0.3 * sy, 0.4 * sz, 12).translate([-0.2 * sx, 0.48 * sy, 0])
    mast = tapered_cylinder(0.04, 0.03, 1.1 * sy, 10).translate([0.15 * sx, 0.85 * sy, 0])
    sail = box(0.02, 0.7 * sy, 0.45 * sz).translate([0.15 * sx, 0.85 * sy, 0.05])
    return refine(Mesh.merge([hull, deck, cabin, mast, sail], name="boat"), 1, 2)


def build_mountain(rng: np.random.Generator, mods: Set[str], detail: int, latent: np.ndarray) -> Mesh:
    sx, sy, sz = _scale(mods)
    d = max(16, detail)
    # Organische berg via neural blob + pieken
    base = neural_blob(latent, int(rng.integers(1, 1e9)), d, max(10, d // 2), 0.7, mods | {"broken"})
    base = base.scale([1.6 * sx, 0.9 * sy, 1.6 * sz])
    peaks = [
        cone(0.55 * sx, 1.2 * sy, d).translate([0.2, 0.5 * sy, 0.1]),
        cone(0.35 * sx, 0.9 * sy, d).translate([-0.45 * sx, 0.35 * sy, -0.2]),
        cone(0.28 * sx, 0.7 * sy, d).translate([0.5 * sx, 0.3 * sy, -0.35 * sz]),
    ]
    return refine(Mesh.merge([base, *peaks], name="mountain"), 1, 2)


def build_creature(rng: np.random.Generator, mods: Set[str], detail: int, latent: np.ndarray) -> Mesh:
    d = max(14, detail)
    body = neural_blob(latent, int(rng.integers(1, 1e9)), d, max(10, d // 2), 0.95, mods)
    body = body.scale([1.3, 0.75, 0.95])
    head = ellipsoid(0.35, 0.3, 0.3, d, 10).translate([0.7, 0.25, 0])
    jaw = ellipsoid(0.22, 0.12, 0.2, 12, 8).translate([0.85, 0.1, 0])
    parts = [body, head, jaw]
    for side in (-1, 1):
        parts.append(capsule(0.08, 0.45, 12, 6).translate([0.2, -0.15, side * 0.35]))
        parts.append(capsule(0.07, 0.4, 12, 6).translate([-0.25, -0.1, side * 0.3]))
    return refine(Mesh.merge(parts, name="creature"), 1, 2)
