"""Smoothing + ronde hulpvormen voor realistischer meshes."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .primitives import cylinder, sphere
from .subdivide import subdivide_once


def laplacian_smooth(mesh: Mesh, iterations: int = 2, lambda_: float = 0.45) -> Mesh:
    """Snelle vectorized Laplacian smooth."""
    if mesh.vertex_count == 0 or mesh.face_count == 0:
        return mesh
    verts = mesh.vertices.copy()
    faces = mesh.faces
    n = len(verts)

    # Bouw sparse adjacency via edge list (uniek)
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    a = edges[:, 0]
    b = edges[:, 1]

    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, a, 1)
    np.add.at(degree, b, 1)
    degree = np.maximum(degree, 1.0)

    for _ in range(iterations):
        acc = np.zeros_like(verts)
        np.add.at(acc, a, verts[b])
        np.add.at(acc, b, verts[a])
        avg = acc / degree[:, None]
        verts = verts * (1.0 - lambda_) + avg * lambda_

    cols = None if mesh.colors is None else mesh.colors.copy()
    return Mesh(verts, faces.copy(), name=mesh.name, colors=cols)


def refine(mesh: Mesh, levels: int = 1, smooth_iters: int = 2) -> Mesh:
    """Subdivide + smooth voor gladde, complexere oppervlakken."""
    out = mesh
    for _ in range(max(0, levels)):
        out = subdivide_once(out)
        out = laplacian_smooth(out, iterations=smooth_iters, lambda_=0.42)
    return out


def capsule(
    radius: float = 0.15,
    height: float = 0.8,
    segments: int = 14,
    rings: int = 8,
    name: str = "capsule",
) -> Mesh:
    h = max(height, radius * 2.2)
    shaft_h = h - 2 * radius
    parts = [
        cylinder(radius, max(shaft_h, 0.05), segments=segments, name="shaft"),
        sphere(radius, segments=segments, rings=rings).translate([0, shaft_h * 0.5 + radius * 0.85, 0]),
        sphere(radius, segments=segments, rings=rings).translate([0, -shaft_h * 0.5 - radius * 0.85, 0]),
    ]
    return Mesh.merge(parts, name=name)


def ellipsoid(
    rx: float,
    ry: float,
    rz: float,
    segments: int = 18,
    rings: int = 12,
    name: str = "ellipsoid",
) -> Mesh:
    return sphere(1.0, segments=segments, rings=rings, name=name).scale([rx, ry, rz])


def tapered_cylinder(
    r_bottom: float,
    r_top: float,
    height: float,
    segments: int = 16,
    name: str = "taper",
) -> Mesh:
    segments = max(8, segments)
    verts = []
    for i in range(segments):
        ang = 2 * np.pi * i / segments
        c, s = np.cos(ang), np.sin(ang)
        verts.append([r_bottom * c, -height * 0.5, r_bottom * s])
        verts.append([r_top * c, height * 0.5, r_top * s])
    bot_c = len(verts)
    verts.append([0.0, -height * 0.5, 0.0])
    top_c = len(verts)
    verts.append([0.0, height * 0.5, 0.0])
    faces = []
    for i in range(segments):
        b0 = 2 * i
        t0 = b0 + 1
        b1 = 2 * ((i + 1) % segments)
        t1 = b1 + 1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        faces.append([bot_c, b1, b0])
        faces.append([top_c, t0, t1])
    return Mesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32), name=name)


def rounded_box(sx: float, sy: float, sz: float, segments: int = 10, name: str = "rbox") -> Mesh:
    mesh = ellipsoid(sx * 0.55, sy * 0.55, sz * 0.55, segments=max(12, segments), rings=max(8, segments // 2), name=name)
    v = mesh.vertices.copy()
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    boxy = np.clip(v, [-hx, -hy, -hz], [hx, hy, hz])
    v = 0.35 * v + 0.65 * boxy
    return Mesh(v, mesh.faces.copy(), name=name)
