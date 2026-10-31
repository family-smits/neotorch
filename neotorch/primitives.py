"""Geometrische primitieven die NeoTorch kan samenstellen."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def box(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, name: str = "box") -> Mesh:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, faces, name=name)


def sphere(radius: float = 1.0, segments: int = 24, rings: int = 16, name: str = "sphere") -> Mesh:
    """UV-sphere met gedeelde polen (geen spleten) en consistente winding."""
    segments = max(8, segments)
    rings = max(4, rings)
    verts = [[0.0, radius, 0.0]]  # north pole
    for i in range(1, rings):
        phi = np.pi * i / rings
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])
    verts.append([0.0, -radius, 0.0])  # south pole
    vertices = np.array(verts, dtype=np.float64)

    faces = []
    # Top cap
    for j in range(segments):
        a = 1 + j
        b = 1 + (j + 1) % segments
        faces.append([0, a, b])
    # Rings
    for i in range(rings - 2):
        row = 1 + i * segments
        next_row = 1 + (i + 1) * segments
        for j in range(segments):
            a = row + j
            b = row + (j + 1) % segments
            c = next_row + (j + 1) % segments
            d = next_row + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    # Bottom cap
    south = len(verts) - 1
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        a = last + j
        b = last + (j + 1) % segments
        faces.append([a, b, south])
    return Mesh(vertices, np.array(faces, dtype=np.int32), name=name)


def cylinder(
    radius: float = 0.5,
    height: float = 1.5,
    segments: int = 24,
    name: str = "cylinder",
) -> Mesh:
    verts = [[0.0, -height * 0.5, 0.0], [0.0, height * 0.5, 0.0]]
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        verts.append([x, -height * 0.5, z])
        verts.append([x, height * 0.5, z])
    vertices = np.array(verts, dtype=np.float64)

    faces = []
    for i in range(segments):
        i0 = 2 + 2 * i
        i1 = 2 + 2 * ((i + 1) % segments)
        b0, t0 = i0, i0 + 1
        b1, t1 = i1, i1 + 1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        faces.append([0, b1, b0])
        faces.append([1, t0, t1])
    return Mesh(vertices, np.array(faces, dtype=np.int32), name=name)


def cone(radius: float = 0.6, height: float = 1.2, segments: int = 24, name: str = "cone") -> Mesh:
    verts = [[0.0, height * 0.5, 0.0], [0.0, -height * 0.5, 0.0]]
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        verts.append([radius * np.cos(angle), -height * 0.5, radius * np.sin(angle)])
    vertices = np.array(verts, dtype=np.float64)
    faces = []
    for i in range(segments):
        a = 2 + i
        b = 2 + (i + 1) % segments
        faces.append([0, a, b])
        faces.append([1, b, a])
    return Mesh(vertices, np.array(faces, dtype=np.int32), name=name)


def torus(
    major: float = 1.0,
    minor: float = 0.35,
    major_seg: int = 32,
    minor_seg: int = 16,
    name: str = "torus",
) -> Mesh:
    verts = []
    for i in range(major_seg):
        u = 2.0 * np.pi * i / major_seg
        for j in range(minor_seg):
            v = 2.0 * np.pi * j / minor_seg
            x = (major + minor * np.cos(v)) * np.cos(u)
            y = minor * np.sin(v)
            z = (major + minor * np.cos(v)) * np.sin(u)
            verts.append([x, y, z])
    vertices = np.array(verts, dtype=np.float64)
    faces = []
    for i in range(major_seg):
        for j in range(minor_seg):
            a = i * minor_seg + j
            b = i * minor_seg + (j + 1) % minor_seg
            c = ((i + 1) % major_seg) * minor_seg + (j + 1) % minor_seg
            d = ((i + 1) % major_seg) * minor_seg + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return Mesh(vertices, np.array(faces, dtype=np.int32), name=name)


def pyramid(base: float = 1.2, height: float = 1.0, name: str = "pyramid") -> Mesh:
    h = base * 0.5
    vertices = np.array(
        [
            [-h, 0.0, -h],
            [h, 0.0, -h],
            [h, 0.0, h],
            [-h, 0.0, h],
            [0.0, height, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, faces, name=name)
