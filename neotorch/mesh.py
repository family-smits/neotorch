"""Mesh-datastructuur voor NeoTorch 3D-modellen (met vertexkleuren)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# Praktisch hard plafond (geheugen/CPU) voor soepele generatie.
PRACTICAL_MAX_VERTS = 120_000
THEORETICAL_MAX_VERTS = 5_000_000_000_000_000_000_000


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3)
    name: str = "model"
    colors: Optional[np.ndarray] = None  # (N, 3) RGB in 0..1

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be (M, 3)")
        if self.colors is None:
            self.colors = np.full((self.vertex_count, 3), 0.75, dtype=np.float64)
        else:
            self.colors = np.asarray(self.colors, dtype=np.float64)
            if self.colors.shape != (self.vertex_count, 3):
                # Herstel bij mismatch
                self.colors = np.full((self.vertex_count, 3), 0.75, dtype=np.float64)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def with_color(self, rgb: Sequence[float]) -> "Mesh":
        c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
        colors = np.repeat(c.reshape(1, 3), self.vertex_count, axis=0)
        return Mesh(self.vertices.copy(), self.faces.copy(), name=self.name, colors=colors)

    def normalize(self, target_size: float = 2.0) -> "Mesh":
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        if extent < 1e-9:
            extent = 1.0
        scale = target_size / extent
        verts = (self.vertices - center) * scale
        return Mesh(verts, self.faces.copy(), name=self.name, colors=None if self.colors is None else self.colors.copy())

    def translate(self, offset: Sequence[float]) -> "Mesh":
        return Mesh(
            self.vertices + np.asarray(offset, dtype=np.float64),
            self.faces.copy(),
            name=self.name,
            colors=None if self.colors is None else self.colors.copy(),
        )

    def scale(self, factor: float | Sequence[float]) -> "Mesh":
        return Mesh(
            self.vertices * np.asarray(factor, dtype=np.float64),
            self.faces.copy(),
            name=self.name,
            colors=None if self.colors is None else self.colors.copy(),
        )

    def paint(self, colors: np.ndarray) -> "Mesh":
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape != (self.vertex_count, 3):
            raise ValueError("colors must match vertex count")
        return Mesh(self.vertices.copy(), self.faces.copy(), name=self.name, colors=np.clip(colors, 0, 1))

    @staticmethod
    def merge(meshes: Sequence["Mesh"], name: str = "merged") -> "Mesh":
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), name=name)
        verts: List[np.ndarray] = []
        faces: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        offset = 0
        for mesh in meshes:
            verts.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            if mesh.colors is None:
                cols.append(np.full((mesh.vertex_count, 3), 0.75))
            else:
                cols.append(mesh.colors)
            offset += mesh.vertex_count
        return Mesh(np.vstack(verts), np.vstack(faces), name=name, colors=np.vstack(cols))
