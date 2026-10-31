"""Weer- en sfeer-scenes: storm, regen, vuur, sneeuw, … (geen regenboogbol)."""

from __future__ import annotations

from typing import List

import numpy as np

from .colors import paint_mesh
from .mesh import Mesh
from .primitives import box, cone, cylinder, sphere
from .smooth import ellipsoid, refine, tapered_cylinder


STORM_PALETTE = [
    (0.18, 0.20, 0.24),
    (0.28, 0.30, 0.34),
    (0.45, 0.48, 0.52),
    (0.12, 0.13, 0.15),
    (0.85, 0.88, 0.55),  # bliksem accent
]
RAIN_PALETTE = [
    (0.22, 0.28, 0.35),
    (0.35, 0.42, 0.50),
    (0.55, 0.62, 0.68),
    (0.15, 0.35, 0.55),
]
SNOW_PALETTE = [
    (0.88, 0.91, 0.95),
    (0.75, 0.80, 0.88),
    (0.55, 0.62, 0.72),
    (0.95, 0.97, 1.0),
]
FIRE_PALETTE = [
    (0.95, 0.35, 0.08),
    (0.95, 0.75, 0.15),
    (0.55, 0.12, 0.05),
    (0.25, 0.08, 0.05),
]
VOLCANO_PALETTE = [
    (0.25, 0.18, 0.15),
    (0.45, 0.28, 0.18),
    (0.9, 0.35, 0.1),
    (0.15, 0.12, 0.1),
]
OCEAN_PALETTE = [
    (0.08, 0.28, 0.45),
    (0.12, 0.45, 0.62),
    (0.25, 0.65, 0.75),
    (0.85, 0.9, 0.92),
]
NIGHT_PALETTE = [
    (0.08, 0.09, 0.16),
    (0.15, 0.16, 0.28),
    (0.35, 0.38, 0.55),
    (0.9, 0.85, 0.55),
]


def build_storm(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(12, detail)
    parts: List[Mesh] = []
    # Donkere grond
    ground = ellipsoid(2.4, 0.08, 2.0, d, 8).translate([0, 0.0, 0])
    parts.append(paint_mesh(ground, STORM_PALETTE[1:4], seed=1, obj="ground"))
    # Wolkenmassa
    for i in range(7):
        cloud = ellipsoid(
            0.55 + rng.random() * 0.45,
            0.22 + rng.random() * 0.12,
            0.4 + rng.random() * 0.35,
            d, 8,
        ).translate([
            (rng.random() - 0.5) * 2.2,
            1.35 + rng.random() * 0.45,
            (rng.random() - 0.5) * 1.6,
        ])
        parts.append(paint_mesh(cloud, STORM_PALETTE[:4], seed=10 + i, obj="neural"))
    # Regenstrepen
    for i in range(28):
        drop = box(0.015, 0.35 + rng.random() * 0.25, 0.015).translate([
            (rng.random() - 0.5) * 2.4,
            0.55 + rng.random() * 0.7,
            (rng.random() - 0.5) * 2.0,
        ])
        parts.append(paint_mesh(drop, [(0.55, 0.65, 0.75)], seed=50 + i, obj="water"))
    # Bliksem
    bolt_pts = [(0.3, 1.5, 0.2), (0.15, 1.1, 0.15), (0.35, 0.85, 0.1), (0.1, 0.45, 0.05)]
    for i in range(len(bolt_pts) - 1):
        a = np.array(bolt_pts[i])
        b = np.array(bolt_pts[i + 1])
        mid = (a + b) * 0.5
        length = float(np.linalg.norm(b - a))
        seg = box(0.04, length, 0.04).translate(mid.tolist())
        parts.append(paint_mesh(seg, [STORM_PALETTE[4], (1.0, 1.0, 0.75)], seed=80 + i, obj="crystal"))
    # Windgebogen boom
    trunk = tapered_cylinder(0.08, 0.05, 0.9, d).translate([-1.1, 0.45, -0.6])
    canopy = ellipsoid(0.35, 0.28, 0.32, d, 8).translate([-0.85, 1.05, -0.55])
    parts.append(paint_mesh(trunk, [(0.28, 0.18, 0.1)], seed=90, obj="tree"))
    parts.append(paint_mesh(canopy, [(0.12, 0.22, 0.12), (0.18, 0.28, 0.14)], seed=91, obj="tree"))
    return Mesh.merge(parts, name="storm")


def build_rain(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(12, detail)
    parts: List[Mesh] = []
    parts.append(paint_mesh(ellipsoid(2.2, 0.07, 1.9, d, 8), RAIN_PALETTE[1:3], seed=1, obj="ground"))
    for i in range(5):
        cloud = ellipsoid(0.7, 0.2, 0.5, d, 8).translate([(i - 2) * 0.55, 1.5, (rng.random() - 0.5) * 0.4])
        parts.append(paint_mesh(cloud, RAIN_PALETTE[:3], seed=5 + i, obj="neural"))
    for i in range(40):
        drop = box(0.012, 0.28, 0.012).translate([
            (rng.random() - 0.5) * 2.3, 0.4 + rng.random() * 0.9, (rng.random() - 0.5) * 1.9,
        ])
        parts.append(paint_mesh(drop, [(0.6, 0.7, 0.8)], seed=20 + i, obj="water"))
    puddle = ellipsoid(0.55, 0.03, 0.4, 12, 6).translate([0.4, 0.04, 0.3])
    parts.append(paint_mesh(puddle, RAIN_PALETTE[2:4], seed=99, obj="water"))
    return Mesh.merge(parts, name="rain")


def build_snow(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(12, detail)
    parts: List[Mesh] = []
    parts.append(paint_mesh(ellipsoid(2.3, 0.12, 2.0, d, 8), SNOW_PALETTE[:3], seed=1, obj="ground"))
    for i in range(6):
        cloud = ellipsoid(0.6, 0.22, 0.45, d, 8).translate([(i - 2.5) * 0.5, 1.55, (rng.random() - 0.5) * 0.5])
        parts.append(paint_mesh(cloud, SNOW_PALETTE[:2], seed=5 + i, obj="neural"))
    for i in range(35):
        flake = sphere(0.03 + rng.random() * 0.025, 8, 6).translate([
            (rng.random() - 0.5) * 2.2, 0.3 + rng.random() * 1.2, (rng.random() - 0.5) * 1.8,
        ])
        parts.append(paint_mesh(flake, [SNOW_PALETTE[3]], seed=30 + i, obj="crystal"))
    tree = Mesh.merge([
        tapered_cylinder(0.07, 0.04, 0.7, 10).translate([-0.9, 0.35, -0.5]),
        cone(0.35, 0.55, 12).translate([-0.9, 0.85, -0.5]),
        cone(0.28, 0.4, 12).translate([-0.9, 1.15, -0.5]),
    ], name="pine")
    parts.append(paint_mesh(tree, [(0.2, 0.32, 0.18), (0.55, 0.45, 0.3), SNOW_PALETTE[0]], seed=70, obj="tree"))
    return Mesh.merge(parts, name="snow")


def build_fire(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(12, detail)
    parts: List[Mesh] = []
    parts.append(paint_mesh(ellipsoid(1.6, 0.06, 1.4, d, 8), [(0.2, 0.14, 0.1), (0.3, 0.2, 0.12)], seed=1, obj="ground"))
    # Vlammen
    for i in range(8):
        flame = cone(0.12 + rng.random() * 0.1, 0.45 + rng.random() * 0.55, d).translate([
            (rng.random() - 0.5) * 0.5, 0.25 + rng.random() * 0.15, (rng.random() - 0.5) * 0.4,
        ])
        parts.append(paint_mesh(flame, FIRE_PALETTE, seed=10 + i, obj="crystal"))
    # Hout
    for i, ox in enumerate((-0.2, 0.0, 0.2)):
        log = cylinder(0.06, 0.55, 10).translate([ox, 0.08, 0.05 * i])
        parts.append(paint_mesh(log, [(0.35, 0.22, 0.1)], seed=40 + i, obj="tree"))
    # Rook
    for i in range(4):
        smoke = ellipsoid(0.2 + i * 0.05, 0.15, 0.2, 10, 6).translate([0.05 * i, 0.9 + i * 0.25, 0])
        parts.append(paint_mesh(smoke, [(0.35, 0.35, 0.36), (0.25, 0.25, 0.26)], seed=50 + i, obj="neural"))
    return Mesh.merge(parts, name="fire")


def build_volcano(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(14, detail)
    mountain = cone(1.4, 1.6, d).translate([0, 0.8, 0])
    crater = cylinder(0.35, 0.15, d).translate([0, 1.55, 0])
    lava = cone(0.25, 0.45, d).translate([0, 1.75, 0])
    flow = ellipsoid(0.35, 0.08, 0.9, 12, 6).translate([0.4, 0.55, 0.5])
    ground = ellipsoid(2.2, 0.08, 2.0, d, 8)
    parts = [
        paint_mesh(ground, VOLCANO_PALETTE[1:3], seed=1, obj="ground"),
        paint_mesh(mountain, VOLCANO_PALETTE[:3], seed=2, obj="mountain"),
        paint_mesh(crater, VOLCANO_PALETTE[3:4], seed=3, obj="rock"),
        paint_mesh(lava, FIRE_PALETTE, seed=4, obj="crystal"),
        paint_mesh(flow, FIRE_PALETTE[:3], seed=5, obj="crystal"),
    ]
    for i in range(5):
        ash = sphere(0.05, 8, 6).translate([(rng.random() - 0.5) * 1.2, 1.9 + rng.random() * 0.5, (rng.random() - 0.5)])
        parts.append(paint_mesh(ash, [(0.3, 0.28, 0.26)], seed=20 + i, obj="rock"))
    return Mesh.merge(parts, name="volcano")


def build_ocean(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(14, detail)
    water = ellipsoid(2.5, 0.15, 2.1, d, 10)
    parts = [paint_mesh(water, OCEAN_PALETTE, seed=1, obj="water")]
    for i in range(6):
        wave = ellipsoid(0.7, 0.08, 0.25, 12, 6).translate([(i - 2.5) * 0.55, 0.12, (rng.random() - 0.5) * 0.8])
        parts.append(paint_mesh(wave, OCEAN_PALETTE[1:3] + [OCEAN_PALETTE[3]], seed=5 + i, obj="water"))
    # Eilandje
    island = ellipsoid(0.55, 0.18, 0.45, 12, 8).translate([0.9, 0.12, -0.6])
    palm = Mesh.merge([
        tapered_cylinder(0.05, 0.03, 0.55, 10).translate([0.9, 0.4, -0.6]),
        ellipsoid(0.25, 0.08, 0.25, 10, 6).translate([0.9, 0.7, -0.6]),
    ], name="palm")
    parts.append(paint_mesh(island, [(0.75, 0.68, 0.45), (0.35, 0.55, 0.25)], seed=20, obj="ground"))
    parts.append(paint_mesh(palm, [(0.4, 0.25, 0.12), (0.2, 0.55, 0.2)], seed=21, obj="tree"))
    return Mesh.merge(parts, name="ocean")


def build_night(rng: np.random.Generator, detail: int = 14) -> Mesh:
    d = max(12, detail)
    ground = ellipsoid(2.2, 0.08, 1.9, d, 8)
    parts = [paint_mesh(ground, NIGHT_PALETTE[:3], seed=1, obj="ground")]
    # Maan
    moon = sphere(0.28, d, max(8, d // 2)).translate([-1.0, 1.7, -0.8])
    parts.append(paint_mesh(moon, [(0.92, 0.9, 0.8), (0.75, 0.75, 0.7)], seed=2, obj="crystal"))
    # Sterren
    for i in range(18):
        star = sphere(0.02, 6, 4).translate([
            (rng.random() - 0.5) * 2.4, 1.2 + rng.random() * 0.9, (rng.random() - 0.5) * 2.0,
        ])
        parts.append(paint_mesh(star, [NIGHT_PALETTE[3]], seed=10 + i, obj="crystal"))
    # Huisje met licht
    house = box(0.55, 0.4, 0.45).translate([0.7, 0.25, 0.4])
    roof = cone(0.4, 0.3, 12).translate([0.7, 0.55, 0.4])
    window = box(0.12, 0.12, 0.04).translate([0.7, 0.28, 0.62])
    parts.append(paint_mesh(Mesh.merge([house, roof], name="hut"), [(0.25, 0.2, 0.18), (0.35, 0.15, 0.12)], seed=40, obj="house"))
    parts.append(paint_mesh(window, [(0.95, 0.85, 0.4)], seed=41, obj="crystal"))
    return Mesh.merge(parts, name="night")


BUILDERS = {
    "storm": build_storm,
    "rain": build_rain,
    "snow": build_snow,
    "fire": build_fire,
    "volcano": build_volcano,
    "ocean": build_ocean,
    "night": build_night,
}

# Woorden → weer/sfeer concept
WEATHER_MAP = {
    "storm": "storm", "onweer": "storm", "bliksem": "storm", "thunder": "storm",
    "thunderstorm": "storm", "orkaan": "storm", "hurricane": "storm", "tyfoon": "storm",
    "wind": "storm", "winderig": "storm", "stormachtig": "storm",
    "rain": "rain", "regen": "rain", "regenen": "rain", "rainy": "rain", "bui": "rain",
    "snow": "snow", "sneeuw": "snow", "sneeuwen": "snow", "blizzard": "snow", "winter": "snow",
    "fire": "fire", "vuur": "fire", "vlam": "fire", "vlammen": "fire", "kampvuur": "fire", "campfire": "fire",
    "volcano": "volcano", "vulkaan": "volcano", "lava": "volcano",
    "ocean": "ocean", "oceaan": "ocean", "zee": "ocean", "sea": "ocean", "golf": "ocean",
    "night": "night", "nacht": "night", "avond": "night", "duisternis": "night",
    "sunset": "night", "zonsondergang": "night",
}
