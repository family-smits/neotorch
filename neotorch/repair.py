"""Mesh-reparatie: normalen naar buiten, preview-decimate zonder gaten."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def fix_face_winding(mesh: Mesh) -> Mesh:
    """Draai faces om zodat normalen van het zwaartepunt af wijzen (dicht gaten-achtige cull)."""
    if mesh.face_count == 0:
        return mesh
    center = mesh.vertices.mean(axis=0)
    faces = mesh.faces.copy()
    verts = mesh.vertices
    for i, face in enumerate(faces):
        a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        if np.dot(normal, face_center - center) < 0:
            faces[i] = face[[0, 2, 1]]
    return Mesh(verts.copy(), faces, name=mesh.name, colors=None if mesh.colors is None else mesh.colors.copy())


def preview_decimate(mesh: Mesh, max_faces: int = 3500) -> Mesh:
    """
    Maak een lichtere mesh voor preview door vertices op een grid te snappen.
    Behoudt gesloten oppervlakken beter dan faces overslaan (dat gaf gaten).
    """
    if mesh.face_count <= max_faces:
        return mesh

    verts = mesh.vertices
    # Fijnere grid → mooiere preview, nog steeds onder max_faces
    target_verts = max(400, int(max_faces * 0.7))
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    cells = max(10, int(np.ceil(target_verts ** (1 / 3) * 1.35)))
    cell = extent / cells

    keys = np.floor((verts - mins) / cell).astype(np.int32)
    keys = np.clip(keys, 0, cells - 1)
    # Pack key
    packed = keys[:, 0] + keys[:, 1] * (cells + 1) + keys[:, 2] * (cells + 1) ** 2

    # Unieke cells → nieuwe verts (gemiddelde van originals)
    uniq, inv = np.unique(packed, return_inverse=True)
    new_count = len(uniq)
    new_verts = np.zeros((new_count, 3), dtype=np.float64)
    counts = np.zeros(new_count, dtype=np.int32)
    np.add.at(new_verts, inv, verts)
    np.add.at(counts, inv, 1)
    new_verts /= counts[:, None]

    new_colors = None
    if mesh.colors is not None:
        new_colors = np.zeros((new_count, 3), dtype=np.float64)
        np.add.at(new_colors, inv, mesh.colors)
        new_colors /= counts[:, None]

    # Remap faces, drop degenerate
    mapped = inv[mesh.faces]
    keep = (
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 1] != mapped[:, 2])
        & (mapped[:, 0] != mapped[:, 2])
    )
    new_faces = mapped[keep]
    # Unieke faces
    if len(new_faces) > 0:
        # sort indices within face for uniqueness only for dedupe key — keep winding from first
        # Dedup exact identical index triples
        _, uniq_idx = np.unique(new_faces, axis=0, return_index=True)
        new_faces = new_faces[np.sort(uniq_idx)]

    out = Mesh(new_verts, new_faces.astype(np.int32), name=mesh.name, colors=new_colors)
    out = fix_face_winding(out)

    # Als nog te veel: uniforme face-sample is NO-GO; tweede pass grover grid
    if out.face_count > max_faces * 1.2:
        return preview_decimate(out, max_faces=max_faces)
    return out
