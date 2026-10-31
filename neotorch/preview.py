"""Levende fotorealistische preview: adem, idle-sway, studio light."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pygame

from .mesh import Mesh
from .repair import preview_decimate
from .shading import compute_vertex_normals

PREVIEW_FACE_LIMIT = 5500


class MeshPreview:
    def __init__(self):
        self.full_mesh: Optional[Mesh] = None
        self.mesh: Optional[Mesh] = None
        self.base_verts: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.yaw = 0.45
        self.pitch = 0.32
        self.zoom = 2.55
        self._dragging = False
        self._last_mouse = (0, 0)
        self.auto_rotate = True
        self._bg: Optional[pygame.Surface] = None
        self._bg_size = (0, 0)
        self.photoreal = True
        self.life_t = 0.0
        self.alive = True

    def set_mesh(self, mesh: Optional[Mesh], photoreal: bool = True) -> None:
        self.full_mesh = mesh
        self.photoreal = photoreal
        self.yaw = 0.45
        self.pitch = 0.32
        self.life_t = 0.0
        if mesh is None:
            self.mesh = None
            self.base_verts = None
            self.normals = None
        else:
            limit = 6500 if photoreal else PREVIEW_FACE_LIMIT
            self.mesh = preview_decimate(mesh, max_faces=limit)
            self.base_verts = self.mesh.vertices.copy()
            self.normals = compute_vertex_normals(self.mesh)

    def handle_event(self, event: pygame.event.Event, rect: pygame.Rect) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if rect.collidepoint(event.pos):
                self._dragging = True
                self._last_mouse = event.pos
                self.auto_rotate = False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            dx = event.pos[0] - self._last_mouse[0]
            dy = event.pos[1] - self._last_mouse[1]
            self.yaw += dx * 0.01
            self.pitch = float(np.clip(self.pitch + dy * 0.01, -1.2, 1.2))
            self._last_mouse = event.pos
        elif event.type == pygame.MOUSEWHEEL and rect.collidepoint(pygame.mouse.get_pos()):
            self.zoom = float(np.clip(self.zoom - event.y * 0.15, 1.2, 5.0))

    def update(self, dt: float) -> None:
        self.life_t += dt
        if self.auto_rotate and self.mesh is not None:
            self.yaw += dt * 0.28

    def _living_vertices(self) -> np.ndarray:
        """Ademhaling + subtiele idle-pulse op rest-pose."""
        if self.base_verts is None:
            return self.mesh.vertices if self.mesh is not None else np.zeros((0, 3))
        v = self.base_verts.copy()
        if not self.alive or len(v) == 0:
            return v
        y = v[:, 1]
        y0, y1 = float(y.min()), float(y.max())
        yn = (y - y0) / max(1e-9, y1 - y0)
        # Adem: borst/midden zet uit
        breath = math.sin(self.life_t * 2.15)
        chest = np.exp(-((yn - 0.55) ** 2) / 0.07)
        expand = 1.0 + 0.045 * breath * chest
        v[:, 0] *= expand
        v[:, 2] *= expand
        v[:, 1] = y0 + (y - y0) * (1.0 + 0.018 * breath * (0.3 + 0.7 * yn))
        # Hartslag-micro pulse
        beat = max(0.0, math.sin(self.life_t * 6.5)) ** 8
        v *= 1.0 + 0.008 * beat
        # Idle weight-shift
        sway = 0.03 * math.sin(self.life_t * 1.05)
        v[:, 0] += sway * yn
        v[:, 1] += 0.012 * math.sin(self.life_t * 1.05 + 1.2) * yn
        return v

    def _ensure_bg(self, size: tuple[int, int]) -> pygame.Surface:
        if self._bg is None or self._bg_size != size:
            self._bg = pygame.Surface(size)
            for i in range(size[1]):
                t = i / max(size[1] - 1, 1)
                r = int(48 + (18 - 48) * t + 40 * (1 - t) ** 2)
                g = int(52 + (22 - 52) * t + 35 * (1 - t) ** 2)
                b = int(62 + (32 - 62) * t + 30 * (1 - t) ** 2)
                pygame.draw.line(self._bg, (r, g, b), (0, i), (size[0] - 1, i))
            cx, cy = size[0] // 2, int(size[1] * 0.42)
            for rad in range(max(size), 40, -18):
                a = int(18 * (rad / max(size)))
                pygame.draw.circle(self._bg, (max(0, 20 - a), max(0, 22 - a), max(0, 28 - a)), (cx, cy), rad, 2)
            self._bg_size = size
        return self._bg

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        bg = self._ensure_bg((rect.width, rect.height))
        surface.blit(bg, rect.topleft)

        if self.mesh is None or self.mesh.vertex_count == 0:
            font = pygame.font.SysFont("segoeui", 18)
            tip = font.render("Genereer iets levends", True, (160, 170, 185))
            surface.blit(tip, tip.get_rect(center=rect.center))
            return

        # Contact shadow pulseert mee met adem
        breath = 0.5 + 0.5 * math.sin(self.life_t * 2.15)
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        sw = rect.width * (0.42 + 0.04 * breath)
        pygame.draw.ellipse(
            shadow,
            (0, 0, 0, int(55 + 20 * breath)),
            pygame.Rect(rect.width * 0.5 - sw * 0.5, rect.height * 0.72, sw, rect.height * 0.08),
        )
        surface.blit(shadow, rect.topleft)

        live = self._living_vertices()
        R = self._rotation_matrix()
        rotated = live @ R.T
        # Recompute light normals roughly from rest normals rotated
        nrm = self.normals @ R.T if self.normals is not None else None
        faces = self.mesh.faces
        colors = self.mesh.colors

        focal = min(rect.width, rect.height) * 0.44 * (3.15 / self.zoom)
        cam_z = self.zoom + 2.15
        z = rotated[:, 2] + cam_z
        z_safe = np.maximum(z, 0.25)
        xs = rect.centerx + (rotated[:, 0] * focal) / z_safe
        ys = rect.centery - (rotated[:, 1] * focal) / z_safe + 18
        proj = np.stack([xs, ys], axis=1)

        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        a, b, c = rotated[i0], rotated[i1], rotated[i2]
        face_n = np.cross(b - a, c - a)
        visible = face_n[:, 2] > 1e-8
        depth = (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0
        order = np.argsort(depth)

        key = np.array([0.55, 0.75, 0.35], dtype=np.float64)
        key /= np.linalg.norm(key)
        fill = np.array([-0.55, 0.35, 0.45], dtype=np.float64)
        fill /= np.linalg.norm(fill)
        rim = np.array([-0.2, 0.15, -0.85], dtype=np.float64)
        rim /= np.linalg.norm(rim)
        view = np.array([0.0, 0.12, 1.0], dtype=np.float64)
        view /= np.linalg.norm(view)

        if nrm is not None:
            nd_key = np.clip(nrm @ key, 0.0, 1.0)
            nd_fill = np.clip(nrm @ fill, 0.0, 1.0)
            nd_rim = np.clip(nrm @ rim, 0.0, 1.0)
            half = key + view
            half /= np.linalg.norm(half)
            spec_v = np.clip(nrm @ half, 0.0, 1.0) ** 28
            fres = (1.0 - np.clip(nrm @ view, 0.0, 1.0)) ** 2
            # Leven: warme fill pulse
            warm_pulse = 0.04 * (0.5 + 0.5 * math.sin(self.life_t * 2.15))
            ambient = 0.14 + warm_pulse
            vert_light = ambient + 0.62 * nd_key + 0.30 * nd_fill + 0.22 * nd_rim
            vert_spec = 0.5 * spec_v + 0.22 * fres * nd_rim
        else:
            vert_light = np.ones(len(rotated)) * 0.7
            vert_spec = np.zeros(len(rotated))

        albedo = colors if colors is not None else np.full((len(rotated), 3), 0.7)

        for fi in order:
            if not visible[fi]:
                continue
            ia, ib, ic = int(i0[fi]), int(i1[fi]), int(i2[fi])
            if z[ia] < 0.25 or z[ib] < 0.25 or z[ic] < 0.25:
                continue
            pts = [
                (int(proj[ia, 0]), int(proj[ia, 1])),
                (int(proj[ib, 0]), int(proj[ib, 1])),
                (int(proj[ic, 0]), int(proj[ic, 1])),
            ]
            lit = (
                albedo[ia] * vert_light[ia]
                + albedo[ib] * vert_light[ib]
                + albedo[ic] * vert_light[ic]
            ) / 3.0
            sp = (vert_spec[ia] + vert_spec[ib] + vert_spec[ic]) / 3.0
            col = lit + sp * np.array([1.0, 0.96, 0.9])
            # Lichte warme glow (leven)
            col = col * np.array([1.02, 0.99, 0.97])
            col = col / (1.0 + np.maximum(col - 0.85, 0) * 0.65)
            color = (
                int(np.clip(col[0] * 255, 0, 255)),
                int(np.clip(col[1] * 255, 0, 255)),
                int(np.clip(col[2] * 255, 0, 255)),
            )
            pygame.draw.polygon(surface, color, pts)

        font = pygame.font.SysFont("segoeui", 13)
        full_n = self.full_mesh.face_count if self.full_mesh is not None else 0
        mode = "LEVEN" if self.alive else ("PHOTOREAL" if self.photoreal else "preview")
        lod = f" · {mode} {self.mesh.face_count}/{full_n}" if full_n else f" · {mode}"
        hint = font.render("Sleep · scroll" + lod, True, (150, 160, 175))
        surface.blit(hint, (rect.left + 14, rect.bottom - 28))

    def _rotation_matrix(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cx, sx = math.cos(self.pitch), math.sin(self.pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        return rx @ ry
