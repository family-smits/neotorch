"""Prompt → realistischere 3D mesh via NeoTorch."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import realistic as R
from .colors import expand_palette, latent_fallback_color, paint_mesh
from .mesh import PRACTICAL_MAX_VERTS, Mesh
from .concepts import word_hash
from .neural_mesh import apply_latent_deform, attach_parts, encode_prompt, neural_blob
from .primitives import box, cone, cylinder, pyramid, sphere, torus
from .prompt import PromptPlan, parse_prompt, tokenize
from .repair import fix_face_winding
from .photoreal import apply_photoreal_materials, photoreal_refine, wants_photoreal
from .scene import Actor, compose_scene
from .smooth import refine, rounded_box
from .subdivide import enrich_to_budget
from .tokens import difficulty_for_cost
from .weather import BUILDERS as WEATHER_BUILDERS

_WEATHER = set(WEATHER_BUILDERS.keys())


class Model3DGenerator:
    """
    Realistischere meshes:
    gedetailleerde assemblages → refine (subdivide+smooth) → lichte deform → kleuren.
    """

    TIER_BUDGET = {
        "normal": 6_000,
        "detailed": 16_000,
        "complex": 40_000,
        "ultra": 100_000,
    }
    TIER_SEGMENTS = {
        "normal": 18,
        "detailed": 28,
        "complex": 38,
        "ultra": 52,
    }

    def __init__(self, detail: int = 20, deform_strength: float = 0.05, max_verts: int = 100_000):
        self.detail = max(12, detail)
        self.deform_strength = deform_strength
        self.max_verts = min(max(500, max_verts), PRACTICAL_MAX_VERTS)

    def parse(self, prompt: str) -> PromptPlan:
        return parse_prompt(prompt)

    def resolve_shape(self, prompt: str) -> str:
        return parse_prompt(prompt).label

    def estimate_cost(self, prompt: str) -> Tuple[int, str, str]:
        plan = parse_prompt(prompt)
        cost = plan.complexity
        if plan.detail_tier == "ultra":
            cost = min(25, cost + 5)
        elif plan.detail_tier == "complex":
            cost = min(25, cost + 3)
        if plan.is_scene:
            cost = min(25, cost + 2)
        if self.detail >= 32:
            cost = min(25, cost + 2)
        return cost, plan.label, difficulty_for_cost(cost)

    def estimate_verts(self, prompt: str) -> int:
        plan = parse_prompt(prompt)
        base = self.TIER_BUDGET.get(plan.detail_tier, 6_000)
        if plan.is_scene:
            base = int(base * 1.35)
        return min(base, self.max_verts)

    def generate(self, prompt: str) -> Mesh:
        plan = parse_prompt(prompt)
        latent = encode_prompt(plan)
        tier = plan.detail_tier
        segs = max(self.detail, self.TIER_SEGMENTS.get(tier, 18))
        old_detail = self.detail
        self.detail = segs

        try:
            actor_meshes: List[Mesh] = []
            for i, actor in enumerate(plan.actors):
                mesh = self._build_object(actor.obj, plan, latent, index=i, actor=actor)
                # Weer-scenes zijn al geschilderd; andere acteurs krijgen tonale materials
                if actor.obj not in _WEATHER:
                    cols = list(actor.colors) if actor.colors else list(plan.colors)
                    if not cols:
                        cols = [latent_fallback_color(latent)]
                    cols = expand_palette(cols, obj=actor.obj, min_colors=3)
                    rainbow = "rainbow" in plan.modifiers or "rainbow" in actor.modifiers
                    mesh = paint_mesh(
                        mesh, cols, seed=plan.seed + i * 17,
                        rainbow=rainbow, obj=actor.obj,
                    )
                if plan.photoreal and actor.obj not in _WEATHER:
                    mesh = photoreal_refine(
                        mesh,
                        strength="ultra" if "photoreal" in plan.modifiers else "high",
                    )
                mesh = apply_photoreal_materials(mesh, obj=actor.obj, seed=plan.seed + i * 31)
                actor_meshes.append(mesh)

            part_detail = max(10, min(22, self.detail // 2))
            if plan.parts and actor_meshes and plan.actors[0].obj not in _WEATHER:
                actor_meshes[0] = attach_parts(
                    actor_meshes[0], plan.parts, latent, plan.seed, detail=part_detail,
                )
                actor_meshes[0] = paint_mesh(
                    actor_meshes[0],
                    expand_palette(plan.actors[0].colors or plan.colors or [latent_fallback_color(latent)], plan.actors[0].obj),
                    seed=plan.seed + 99,
                    obj=plan.actors[0].obj,
                )

            only_weather = len(actor_meshes) == 1 and plan.actors and plan.actors[0].obj in _WEATHER
            if only_weather:
                # Storm/regen/… is al een complete scene
                combined = actor_meshes[0]
                combined.name = self._slug(prompt) or plan.actors[0].obj
            elif plan.is_scene or len(actor_meshes) >= 2:
                combined = compose_scene(
                    actor_meshes,
                    env=plan.environment,
                    seed=plan.seed,
                    detail=max(10, self.detail // 2),
                )
                combined.name = self._slug(prompt) or "scene"
            else:
                combined = actor_meshes[0] if actor_meshes else Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))
                combined = apply_latent_deform(
                    combined, latent, plan.seed,
                    amount=min(self.deform_strength, 0.08),
                    modifiers=plan.modifiers,
                )
                a0 = plan.actors[0]
                cols = expand_palette(a0.colors or plan.colors or [latent_fallback_color(latent)], a0.obj)
                combined = paint_mesh(
                    combined, cols, seed=plan.seed,
                    rainbow="rainbow" in plan.modifiers,
                    obj=a0.obj,
                )
                combined.name = self._slug(prompt)

            budget = min(self.TIER_BUDGET.get(tier, 6_000), self.max_verts)
            if tier == "detailed":
                combined = refine(combined, levels=1, smooth_iters=1)
            elif tier in ("complex", "ultra"):
                combined = refine(combined, levels=1 if tier == "complex" else 2, smooth_iters=1)
                if combined.vertex_count < budget:
                    combined = enrich_to_budget(combined, budget, hard_cap=self.max_verts)

            combined = fix_face_winding(combined)
            primary = plan.actors[0].obj if plan.actors else "neural"
            combined = apply_photoreal_materials(combined, obj=primary, seed=plan.seed + 7)
            return combined.normalize(target_size=2.4 if plan.is_scene else 2.0)
        finally:
            self.detail = old_detail

    def _build_object(
        self,
        obj: str,
        plan: PromptPlan,
        latent: np.ndarray,
        index: int,
        actor: Actor | None = None,
    ) -> Mesh:
        seed = plan.seed + index * 97
        rng = np.random.default_rng(seed)
        d = self.detail
        mods = set(plan.modifiers)
        if actor is not None:
            mods |= set(actor.modifiers)

        # Weer / sfeer → complete scene-meshes
        if obj in WEATHER_BUILDERS:
            mesh = WEATHER_BUILDERS[obj](rng, detail=d)
            mesh.name = obj
            return mesh

        if obj in ("neural", "freeform"):
            label = (actor.label if actor else "") or "freeform"
            mesh = self._build_freeform(rng, mods, d, latent, seed, label)
            mesh.name = label
            return mesh

        builders = {
            "box": lambda: refine(rounded_box(1.0 + abs(latent[0]) * 0.4, 1.0 + abs(latent[1]) * 0.4, 1.0 + abs(latent[2]) * 0.4, d), 1, 2),
            "sphere": lambda: refine(sphere(0.8 + abs(latent[3]) * 0.25, max(16, d), max(12, d // 2)), 1, 3),
            "cylinder": lambda: refine(cylinder(0.35 + abs(latent[4]) * 0.2, 1.4 + abs(latent[5]) * 0.4, max(16, d)), 1, 2),
            "cone": lambda: refine(cone(0.5 + abs(latent[6]) * 0.2, 1.2 + abs(latent[7]) * 0.3, max(16, d)), 1, 2),
            "torus": lambda: refine(torus(0.85, 0.28 + abs(latent[8]) * 0.12, max(20, d), max(12, d // 2)), 1, 2),
            "pyramid": lambda: refine(pyramid(1.1 + abs(latent[9]) * 0.3, 1.0 + abs(latent[10]) * 0.3), 1, 1),
            "house": lambda: R.build_house(rng, mods, d),
            "tree": lambda: R.build_tree(rng, mods, d),
            "tower": lambda: R.build_tower(rng, mods, d),
            "castle": lambda: R.build_castle(rng, mods, d),
            "crystal": lambda: R.build_crystal(rng, mods, d),
            "robot": lambda: R.build_robot(rng, mods, d),
            "sword": lambda: R.build_sword(rng, mods, d),
            "rock": lambda: refine(neural_blob(latent, seed, d, max(12, d // 2), 0.75, mods | {"broken"}), 1, 2),
            "mountain": lambda: R.build_mountain(rng, mods, d, latent),
            "humanoid": lambda: R.build_humanoid(rng, mods, d),
            "creature": lambda: R.build_creature(rng, mods, d, latent),
            "dragon": lambda: R.build_dragon(rng, mods, d),
            "bird": lambda: R.build_bird(rng, mods, d),
            "quadruped": lambda: R.build_quadruped(rng, mods, d),
            "vehicle": lambda: R.build_vehicle(rng, mods, d),
            "boat": lambda: R.build_boat(rng, mods, d),
            "chair": lambda: refine(Mesh.merge([
                rounded_box(0.7, 0.1, 0.7, 12).translate([0, 0.45, 0]),
                rounded_box(0.7, 0.7, 0.1, 12).translate([0, 0.85, -0.3]),
                *[box(0.07, 0.45, 0.07).translate([ox, 0.22, oz]) for ox in (-0.28, 0.28) for oz in (-0.28, 0.28)],
            ], name="chair"), 1, 2),
            "table": lambda: refine(Mesh.merge([
                rounded_box(1.4, 0.08, 0.85, 12).translate([0, 0.72, 0]),
                *[box(0.07, 0.72, 0.07).translate([ox, 0.36, oz]) for ox in (-0.55, 0.55) for oz in (-0.3, 0.3)],
            ], name="table"), 1, 2),
            "hat": lambda: refine(Mesh.merge([
                cylinder(0.55, 0.05, d),
                cylinder(0.28, 0.35, d).translate([0, 0.2, 0]),
            ], name="hat"), 1, 2),
            "crown": lambda: refine(Mesh.merge([
                cylinder(0.4, 0.14, 16),
                *[cone(0.055, 0.28, 8).translate([np.cos(a) * 0.35, 0.22, np.sin(a) * 0.35])
                  for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)],
            ], name="crown"), 1, 2),
            "star": lambda: refine(Mesh.merge([
                pyramid(0.35, 0.9).translate([0, 0.2, 0]),
                *[pyramid(0.22, 0.65).translate([np.cos(a) * 0.18, 0.12, np.sin(a) * 0.18])
                  for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)],
            ], name="star"), 1, 1),
            "mushroom": lambda: R.build_mushroom(rng, mods, d),
            "rocket": lambda: R.build_rocket(rng, mods, d),
            "wand": lambda: refine(Mesh.merge([
                cylinder(0.035, 1.45, d).translate([0, 0.72, 0]),
                sphere(0.09, 12, 8).translate([0, 1.5, 0]),
            ], name="wand"), 1, 2),
            "book": lambda: refine(rounded_box(0.75, 0.14, 1.0, 12), 1, 2),
            "fruit": lambda: refine(sphere(0.55 + abs(latent[11 % len(latent)]) * 0.2, max(16, d), max(12, d // 2)), 1, 3),
        }

        builder = builders.get(obj)
        if builder is None:
            label = (actor.label if actor else "") or obj
            mesh = self._build_freeform(rng, mods, d, latent, seed, label)
            mesh.name = label
        else:
            mesh = builder()
            mesh = apply_latent_deform(mesh, latent, seed, amount=self.deform_strength * 0.45, modifiers=mods)
            mesh.name = (actor.label if actor and actor.label else obj)
        return mesh

    def _build_freeform(
        self,
        rng: np.random.Generator,
        mods: set,
        d: int,
        latent: np.ndarray,
        seed: int,
        label: str,
    ) -> Mesh:
        """Unieke assemblage per willekeurig woord (altijd iets anders)."""
        h = word_hash(label)
        recipe = h % 8
        segs = max(14, d)
        parts: List[Mesh] = []

        if recipe == 0:
            # gestapelde bollen
            n = 3 + (h % 4)
            y = 0.0
            for i in range(n):
                r = 0.35 + 0.12 * abs(latent[i % len(latent)]) - i * 0.03
                parts.append(sphere(max(0.18, r), segs, max(10, segs // 2)).translate([0, y + r, 0]))
                y += r * 1.55
        elif recipe == 1:
            # bloktoren
            n = 2 + (h % 4)
            y = 0.0
            for i in range(n):
                w = 0.9 - i * 0.12
                hh = 0.35 + abs(latent[(i + 3) % len(latent)]) * 0.25
                parts.append(rounded_box(w, hh, w * 0.85, max(10, segs // 2)).translate([0, y + hh * 0.5, 0]))
                y += hh
        elif recipe == 2:
            # kern + stekels
            parts.append(sphere(0.55, segs, max(10, segs // 2)))
            for i, a in enumerate(np.linspace(0, 2 * np.pi, 6 + (h % 5), endpoint=False)):
                parts.append(
                    cone(0.12, 0.45 + 0.1 * (i % 3), max(8, segs // 2)).translate(
                        [np.cos(a) * 0.5, 0.1 * (i % 2), np.sin(a) * 0.5]
                    )
                )
        elif recipe == 3:
            # langwerpig lichaam + kop
            parts.append(cylinder(0.28, 1.1, segs).translate([0, 0.55, 0]))
            parts.append(sphere(0.32, segs, max(10, segs // 2)).translate([0, 1.25, 0]))
            parts.append(box(0.7, 0.12, 0.35).translate([0, 0.08, 0]))
        elif recipe == 4:
            # schijf + ring
            parts.append(cylinder(0.75, 0.18, segs))
            parts.append(torus(0.55, 0.12, max(16, segs), max(10, segs // 2)).translate([0, 0.2, 0]))
            parts.append(sphere(0.2, 12, 8).translate([0, 0.45, 0]))
        elif recipe == 5:
            # kruisvorm / artefact
            parts.append(rounded_box(1.2, 0.25, 0.25, 12))
            parts.append(rounded_box(0.25, 1.0, 0.25, 12).translate([0, 0.2, 0]))
            parts.append(sphere(0.18, 12, 8).translate([0, 0.75, 0]))
        elif recipe == 6:
            # plateau + zuilen
            parts.append(rounded_box(1.3, 0.12, 1.0, 12).translate([0, 0.06, 0]))
            for ox, oz in ((-0.45, -0.3), (0.45, -0.3), (-0.45, 0.3), (0.45, 0.3)):
                parts.append(cylinder(0.08, 0.7, max(10, segs // 2)).translate([ox, 0.45, oz]))
            parts.append(sphere(0.25, 12, 8).translate([0, 0.95, 0]))
        else:
            # asymmetrisch cluster
            for i in range(5 + (h % 4)):
                p = rng.normal(0, 0.35, size=3)
                p[1] = abs(p[1]) + 0.2
                r = 0.18 + 0.15 * abs(latent[i % len(latent)])
                if i % 2 == 0:
                    parts.append(sphere(r, max(12, segs // 2), max(8, segs // 3)).translate(p))
                else:
                    parts.append(rounded_box(r * 1.4, r, r * 1.1, 10).translate(p))

        mesh = Mesh.merge(parts, name=label)
        mesh = refine(mesh, levels=1, smooth_iters=2)
        mesh = apply_latent_deform(
            mesh, latent, seed,
            amount=min(0.14, self.deform_strength + 0.06),
            modifiers=mods | ({"spiky"} if "spiky" in mods else {"smooth"}),
        )
        return mesh

    @staticmethod
    def _slug(prompt: str) -> str:
        tokens = tokenize(prompt)
        return "_".join(tokens[:6]) if tokens else "model"
