"""Realistische material-kleuren (geen regenboogbol tenzij gevraagd)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mesh import Mesh

RGB = Tuple[float, float, float]

COLOR_MAP: Dict[str, RGB] = {
    "red": (0.72, 0.14, 0.12), "rood": (0.72, 0.14, 0.12), "rode": (0.72, 0.14, 0.12),
    "blue": (0.18, 0.35, 0.72), "blauw": (0.18, 0.35, 0.72), "blauwe": (0.18, 0.35, 0.72),
    "green": (0.18, 0.48, 0.28), "groen": (0.18, 0.48, 0.28), "groene": (0.18, 0.48, 0.28),
    "yellow": (0.85, 0.72, 0.2), "geel": (0.85, 0.72, 0.2), "gele": (0.85, 0.72, 0.2),
    "orange": (0.85, 0.42, 0.12), "oranje": (0.85, 0.42, 0.12),
    "purple": (0.42, 0.22, 0.55), "paars": (0.42, 0.22, 0.55), "paarse": (0.42, 0.22, 0.55),
    "pink": (0.78, 0.4, 0.55), "roze": (0.78, 0.4, 0.55),
    "cyan": (0.2, 0.55, 0.58), "turquoise": (0.2, 0.55, 0.58), "turkoois": (0.2, 0.55, 0.58),
    "white": (0.88, 0.88, 0.9), "wit": (0.88, 0.88, 0.9), "witte": (0.88, 0.88, 0.9),
    "black": (0.1, 0.1, 0.11), "zwart": (0.1, 0.1, 0.11), "zwarte": (0.1, 0.1, 0.11),
    "gray": (0.45, 0.46, 0.48), "grey": (0.45, 0.46, 0.48), "grijs": (0.45, 0.46, 0.48), "grijze": (0.45, 0.46, 0.48),
    "brown": (0.42, 0.28, 0.16), "bruin": (0.42, 0.28, 0.16), "bruine": (0.42, 0.28, 0.16),
    "gold": (0.78, 0.62, 0.22), "goud": (0.78, 0.62, 0.22), "gouden": (0.78, 0.62, 0.22),
    "silver": (0.65, 0.68, 0.72), "zilver": (0.65, 0.68, 0.72), "zilveren": (0.65, 0.68, 0.72),
    "bronze": (0.58, 0.38, 0.2), "brons": (0.58, 0.38, 0.2),
    "teal": (0.15, 0.45, 0.45), "navy": (0.12, 0.18, 0.35), "lime": (0.45, 0.65, 0.22),
    "magenta": (0.65, 0.18, 0.45), "crimson": (0.55, 0.1, 0.12),
    "ivory": (0.9, 0.87, 0.8), "amber": (0.8, 0.55, 0.18),
    "beige": (0.78, 0.7, 0.58), "cream": (0.9, 0.87, 0.8),
}

# Sobere, realistische paletten (tonale familie, geen disco)
OBJECT_PALETTES: Dict[str, List[RGB]] = {
    "dragon": [(0.55, 0.12, 0.1), (0.35, 0.1, 0.08), (0.65, 0.45, 0.15), (0.15, 0.12, 0.12)],
    "robot": [(0.42, 0.45, 0.48), (0.22, 0.24, 0.26), (0.55, 0.58, 0.62), (0.15, 0.55, 0.65)],
    "humanoid": [(0.72, 0.55, 0.42), (0.22, 0.28, 0.45), (0.18, 0.16, 0.15), (0.55, 0.4, 0.28)],
    "quadruped": [(0.42, 0.3, 0.18), (0.28, 0.2, 0.12), (0.7, 0.65, 0.55), (0.12, 0.1, 0.1)],
    "bird": [(0.25, 0.35, 0.55), (0.65, 0.55, 0.2), (0.55, 0.25, 0.15), (0.9, 0.9, 0.88)],
    "tree": [(0.32, 0.22, 0.12), (0.18, 0.38, 0.18), (0.12, 0.28, 0.12), (0.4, 0.28, 0.14)],
    "house": [(0.65, 0.55, 0.42), (0.45, 0.18, 0.14), (0.35, 0.45, 0.55), (0.35, 0.25, 0.15)],
    "castle": [(0.48, 0.48, 0.5), (0.35, 0.32, 0.3), (0.55, 0.2, 0.15), (0.65, 0.65, 0.68)],
    "tower": [(0.5, 0.48, 0.45), (0.4, 0.2, 0.15), (0.65, 0.55, 0.3)],
    "crystal": [(0.45, 0.65, 0.75), (0.55, 0.45, 0.7), (0.85, 0.9, 0.95)],
    "vehicle": [(0.55, 0.15, 0.12), (0.15, 0.15, 0.16), (0.7, 0.7, 0.72), (0.75, 0.65, 0.2)],
    "boat": [(0.3, 0.2, 0.12), (0.75, 0.75, 0.78), (0.2, 0.35, 0.5)],
    "sword": [(0.6, 0.62, 0.66), (0.7, 0.55, 0.2), (0.3, 0.2, 0.12)],
    "rocket": [(0.75, 0.75, 0.78), (0.6, 0.18, 0.12), (0.75, 0.45, 0.12)],
    "mushroom": [(0.65, 0.2, 0.16), (0.85, 0.8, 0.7), (0.55, 0.45, 0.3)],
    "mountain": [(0.4, 0.38, 0.36), (0.75, 0.78, 0.82), (0.3, 0.38, 0.25)],
    "creature": [(0.3, 0.4, 0.22), (0.5, 0.28, 0.15), (0.15, 0.14, 0.12)],
    "neural": [(0.35, 0.36, 0.38), (0.25, 0.26, 0.28), (0.48, 0.5, 0.52), (0.18, 0.2, 0.24)],
    "fruit": [(0.65, 0.2, 0.14), (0.25, 0.45, 0.18), (0.75, 0.65, 0.2)],
    "rock": [(0.42, 0.4, 0.38), (0.3, 0.3, 0.28), (0.55, 0.52, 0.5)],
    "ground": [(0.28, 0.35, 0.18), (0.35, 0.28, 0.16), (0.22, 0.3, 0.15)],
    "water": [(0.12, 0.32, 0.48), (0.2, 0.48, 0.58), (0.08, 0.22, 0.35)],
    "storm": [(0.18, 0.2, 0.24), (0.28, 0.3, 0.34), (0.12, 0.13, 0.15), (0.75, 0.78, 0.5)],
    "rain": [(0.25, 0.32, 0.4), (0.4, 0.48, 0.55), (0.15, 0.35, 0.5)],
    "snow": [(0.85, 0.88, 0.92), (0.7, 0.75, 0.82), (0.55, 0.6, 0.7)],
    "fire": [(0.75, 0.28, 0.08), (0.85, 0.55, 0.12), (0.35, 0.1, 0.05)],
    "volcano": [(0.25, 0.18, 0.15), (0.7, 0.3, 0.1), (0.15, 0.12, 0.1)],
    "ocean": [(0.08, 0.28, 0.42), (0.15, 0.45, 0.55), (0.8, 0.85, 0.9)],
    "night": [(0.08, 0.09, 0.15), (0.18, 0.2, 0.3), (0.75, 0.7, 0.45)],
}


def resolve_colors(tokens: Sequence[str]) -> List[RGB]:
    found: List[RGB] = []
    for tok in tokens:
        if tok in COLOR_MAP:
            c = COLOR_MAP[tok]
            if c not in found:
                found.append(c)
            continue
        for suf in ("e", "en", "er", "s"):
            if len(tok) > len(suf) + 2 and tok.endswith(suf):
                stem = tok[: -len(suf)]
                if stem in COLOR_MAP:
                    c = COLOR_MAP[stem]
                    if c not in found:
                        found.append(c)
                    break
    return found


def latent_fallback_color(latent: np.ndarray) -> RGB:
    # Neutraal grijsblauw i.p.v. felle random kleur
    g = 0.28 + 0.2 * abs(float(latent[0 % len(latent)]))
    return (float(g), float(g * 1.02), float(min(0.55, g * 1.12)))


def _shift(c: RGB, dr: float, dg: float, db: float) -> RGB:
    return (
        float(np.clip(c[0] + dr, 0.05, 1.0)),
        float(np.clip(c[1] + dg, 0.05, 1.0)),
        float(np.clip(c[2] + db, 0.05, 1.0)),
    )


def expand_palette(
    colors: Sequence[RGB],
    obj: str = "neural",
    min_colors: int = 3,
) -> List[RGB]:
    """Tonale familie: schaduwen/midtones/highlights van dezelfde tint."""
    defaults = OBJECT_PALETTES.get(obj, OBJECT_PALETTES["neural"])
    if colors:
        base = colors[0]
        palette = [base]
        for c in colors[1:]:
            if c not in palette:
                palette.append(c)
        # Alleen tonale varianten van base — geen vreemde accenten
        for e in (
            _shift(base, -0.18, -0.16, -0.14),
            _shift(base, 0.12, 0.1, 0.08),
            _shift(base, -0.28, -0.26, -0.24),
            defaults[min(1, len(defaults) - 1)],
        ):
            if len(palette) >= max(min_colors, 4):
                break
            palette.append(e)
        return palette[:5]
    return list(defaults)[:5]


def paint_mesh(
    mesh: Mesh,
    colors: Sequence[RGB],
    seed: int = 0,
    rainbow: bool = False,
    obj: str = "neural",
) -> Mesh:
    """Realistische zone-kleuren: zachte hoogte-blend, geen regenboogstrepen."""
    n = mesh.vertex_count
    if n == 0:
        return mesh
    rng = np.random.default_rng(seed)

    if rainbow:
        palette = [
            (0.75, 0.2, 0.18), (0.8, 0.5, 0.12), (0.8, 0.75, 0.2),
            (0.2, 0.55, 0.28), (0.2, 0.4, 0.7), (0.45, 0.22, 0.6),
        ]
    else:
        palette = expand_palette(colors, obj=obj, min_colors=3)

    v = mesh.vertices
    y = v[:, 1]
    t = (y - y.min()) / max(1e-9, float(y.max() - y.min()))
    center = v.mean(axis=0)
    radial = np.linalg.norm(v - center, axis=1)
    ao = 0.55 + 0.45 * (radial / max(1e-9, float(radial.max())))

    # Soft blend tussen 2–3 tonale stops op hoogte (fotorealistischer)
    stops = np.asarray(palette[:3], dtype=np.float64)
    if len(stops) == 1:
        stops = np.vstack([stops, _shift(tuple(stops[0]), -0.15, -0.12, -0.1), _shift(tuple(stops[0]), 0.1, 0.08, 0.06)])
    elif len(stops) == 2:
        mid = (stops[0] + stops[1]) * 0.5
        stops = np.vstack([stops[0], mid, stops[1]])

    # t=0 bottom, t=1 top → blend
    rgb = np.zeros((n, 3), dtype=np.float64)
    # bottom→mid
    m0 = t < 0.5
    f0 = (t[m0] / 0.5)[:, None]
    rgb[m0] = stops[0] * (1 - f0) + stops[1] * f0
    # mid→top
    m1 = ~m0
    f1 = ((t[m1] - 0.5) / 0.5)[:, None]
    rgb[m1] = stops[1] * (1 - f1) + stops[2] * f1

    # Subtiele variation (geen harde kleurstrepen)
    if len(palette) > 3:
        accent = np.asarray(palette[3], dtype=np.float64)
        mask = (np.sin(v[:, 0] * 3.5 + v[:, 2] * 2.8) > 0.75) & (t > 0.55)
        rgb[mask] = rgb[mask] * 0.7 + accent * 0.3

    noise = rng.normal(0, 0.012, size=rgb.shape)
    rgb = np.clip(rgb * ao[:, None] + noise, 0.03, 1.0)
    return mesh.paint(rgb)
