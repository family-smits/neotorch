"""Export van NeoTorch-meshes naar OBJ (met kleuren) en STL."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh import Mesh


def export_obj(mesh: Mesh, path: str | Path) -> Path:
    """OBJ met vertexkleuren: `v x y z r g b` (breed ondersteund)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = mesh.colors
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# NeoTorch export: {mesh.name}\n")
        f.write(f"# verts={mesh.vertex_count} faces={mesh.face_count}\n")
        f.write(f"o {mesh.name}\n")
        if colors is not None:
            for v, c in zip(mesh.vertices, colors):
                f.write(
                    f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n"
                )
        else:
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return path


def export_stl(mesh: Mesh, path: str | Path) -> Path:
    """ASCII STL (geen kleuren — gebruik OBJ voor kleur)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.vertices
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {mesh.name}\n")
        for face in mesh.faces:
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(b - a, c - a)
            nlen = np.linalg.norm(normal)
            if nlen > 1e-12:
                normal = normal / nlen
            else:
                normal = np.array([0.0, 1.0, 0.0])
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.6e} {a[1]:.6e} {a[2]:.6e}\n")
            f.write(f"      vertex {b[0]:.6e} {b[1]:.6e} {b[2]:.6e}\n")
            f.write(f"      vertex {c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {mesh.name}\n")
    return path
