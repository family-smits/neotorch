"""Mesh-subdivisie om complexiteit / vertex-budget te verhogen."""

from __future__ import annotations

import numpy as np

from .mesh import PRACTICAL_MAX_VERTS, Mesh


def subdivide_once(mesh: Mesh) -> Mesh:
    """Eén mid-point subdivision: elke driehoek → 4 driehoeken."""
    if mesh.face_count == 0:
        return mesh

    verts = [v for v in mesh.vertices]
    cols = [c for c in (mesh.colors if mesh.colors is not None else np.full((mesh.vertex_count, 3), 0.75))]
    edge_mid: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in edge_mid:
            return edge_mid[key]
        idx = len(verts)
        verts.append(0.5 * (mesh.vertices[i] + mesh.vertices[j]))
        ci = cols[i] if i < len(cols) else np.array([0.75, 0.75, 0.75])
        cj = cols[j] if j < len(cols) else np.array([0.75, 0.75, 0.75])
        cols.append(0.5 * (ci + cj))
        edge_mid[key] = idx
        return idx

    new_faces = []
    for face in mesh.faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])

    return Mesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int32),
        name=mesh.name,
        colors=np.asarray(cols, dtype=np.float64),
    )


def enrich_to_budget(mesh: Mesh, target_verts: int, hard_cap: int = PRACTICAL_MAX_VERTS) -> Mesh:
    """
    Subdivide tot we in de buurt van target_verts zitten.
    Cap altijd op hard_cap (praktisch maximum).
    """
    target = int(min(max(mesh.vertex_count, target_verts), hard_cap))
    out = mesh
    # Genoeg rondes voor low-poly starts (box → ultra)
    for _ in range(10):
        if out.vertex_count >= target * 0.92:
            break
        if out.vertex_count * 2.3 > hard_cap and out.vertex_count >= hard_cap * 0.55:
            break
        nxt = subdivide_once(out)
        if nxt.vertex_count > hard_cap:
            break
        out = nxt
    return out
