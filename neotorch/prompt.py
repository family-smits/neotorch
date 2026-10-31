"""Prompt-analyse: elk woord mag — geen whitelist-limiet meer."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .colors import COLOR_MAP, RGB, resolve_colors
from .concepts import OBJECT_MAP, concept_from_word
from .scene import ENV_MAP, SCENE_WORDS, Actor, detect_environment, is_scene_prompt
from .weather import WEATHER_MAP


def tokenize(prompt: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", prompt.lower())


def prompt_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


PART_MAP: Dict[str, str] = {
    "wing": "wings", "wings": "wings", "vleugel": "wings", "vleugels": "wings",
    "leg": "legs", "legs": "legs", "poot": "legs", "poten": "legs", "benen": "legs",
    "arm": "arms", "arms": "arms", "armen": "arms",
    "eye": "eyes", "eyes": "eyes", "oog": "eyes", "ogen": "eyes",
    "horn": "horns", "horns": "horns", "hoorn": "horns", "hoorns": "horns",
    "tail": "tail", "staart": "tail",
    "roof": "roof", "dak": "roof",
    "door": "door", "deur": "door",
    "window": "windows", "ramen": "windows", "raam": "windows",
    "wheel": "wheels", "wheels": "wheels", "wiel": "wheels", "wielen": "wheels",
    "spike": "spikes", "spikes": "spikes", "stekel": "spikes", "stekels": "spikes",
    "hat": "hat", "hoed": "hat",
    "head": "hat", "hoofd": "hat",
    "hand": "arms", "hands": "arms", "voet": "legs", "feet": "legs", "voeten": "legs",
}

MODIFIERS: Dict[str, str] = {
    "tall": "tall", "hoog": "tall", "hoge": "tall", "lang": "tall", "lange": "tall",
    "short": "short", "kort": "short", "korte": "short", "laag": "short",
    "big": "big", "groot": "big", "grote": "big", "huge": "big", "reusachtig": "big",
    "enormous": "big", "gigantic": "big", "massive": "big",
    "small": "small", "klein": "small", "kleine": "small", "tiny": "small", "mini": "small",
    "wide": "wide", "breed": "wide", "brede": "wide", "dik": "wide",
    "thin": "thin", "dun": "thin", "smal": "thin", "slank": "thin",
    "round": "round", "rond": "round", "ronde": "round", "bolvormig": "round",
    "spiky": "spiky", "spits": "spiky", "scherp": "spiky", "sharp": "spiky", "stekelig": "spiky",
    "smooth": "smooth", "glad": "smooth", "zacht": "smooth",
    "fancy": "ornate", "sierlijk": "ornate", "ornate": "ornate", "magisch": "ornate",
    "magic": "ornate", "magical": "ornate", "fantasy": "ornate",
    "broken": "broken", "kapot": "broken", "ruïne": "broken", "ruin": "broken",
    "flying": "flying", "vliegend": "flying", "zwevend": "flying", "floating": "flying",
    "ancient": "ancient", "oud": "ancient", "oude": "ancient",
    "futuristic": "tech", "toekomstig": "tech", "sci": "tech", "fi": "tech", "cyber": "tech",
    "cute": "cute", "schattig": "cute", "lief": "cute",
    "scary": "scary", "eng": "scary", "duister": "scary", "dark": "scary", "evil": "scary",
    "spitse": "spiky", "scherpe": "spiky", "magische": "ornate",
    "schattige": "cute", "kleine": "small", "grote": "big", "lange": "tall",
    "complex": "complex", "complexe": "complex", "ingewikkeld": "complex",
    "detailed": "detailed", "detail": "detailed", "gedetailleerd": "detailed",
    "intricate": "detailed", "highpoly": "ultra", "hd": "ultra", "ultra": "ultra",
    "insane": "ultra", "epic": "ultra", "mega": "ultra", "4k": "ultra",
    "hyper": "ultra", "max": "ultra", "extreme": "ultra", "extreem": "ultra",
    "rainbow": "rainbow", "regenboog": "rainbow", "multicolor": "rainbow",
    "kleurrijk": "rainbow", "colorful": "rainbow",
    "scene": "scene", "landschap": "scene", "diorama": "scene",
    "photoreal": "photoreal", "photorealistic": "photoreal",
    "fotorealistisch": "photoreal", "fotorealistische": "photoreal",
    "foto": "photoreal", "realistic": "realistic", "realistisch": "realistic",
    "realistische": "realistic", "cinematic": "cinematic", "cinematisch": "cinematic",
    "mooi": "ornate", "beautiful": "ornate", "cool": "tech",
}

CONNECTIVES = {
    "en", "and", "met", "with", "plus", "naast", "bij", "of", "a", "an",
    "de", "het", "een", "the", "in", "op", "near", "tegen", "voor", "achter",
    "van", "from", "to", "naar", "als", "like", "als", "very", "heel", "erg",
    "my", "mijn", "your", "jouw", "zijn", "haar", "ons", "onze", "their",
    "this", "that", "deze", "die", "dat", "some", "any", "also", "ook",
    "into", "onto", "over", "under", "onder", "boven", "tussen", "between",
    "make", "maak", "create", "creeer", "genereer", "generate", "bouw", "build",
    "please", "alsjeblieft", "wil", "want", "kan", "kun", "graag",
}

STOP = CONNECTIVES | SCENE_WORDS | {
    "model", "mesh", "object", "ding", "thing", "iets", "something",
    "3d", "render", "image", "beeld", "plaatje",
}


@dataclass
class PromptPlan:
    raw: str
    tokens: List[str]
    seed: int
    objects: List[str] = field(default_factory=list)
    actors: List[Actor] = field(default_factory=list)
    parts: Set[str] = field(default_factory=set)
    modifiers: Set[str] = field(default_factory=set)
    colors: List[RGB] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    environment: str | None = None
    is_scene: bool = False

    @property
    def complexity(self) -> int:
        score = max(1, len(self.actors) or len(self.objects)) * 4
        score += len(self.parts) * 3
        score += len(self.modifiers) * 2
        score += sum(max(1, len(a.colors)) for a in self.actors)
        score += min(6, len(self.unknown))
        score += min(4, len(self.tokens) // 3)
        if self.is_scene:
            score += 5
        if self.environment:
            score += 3
        if "detailed" in self.modifiers:
            score += 4
        if "complex" in self.modifiers:
            score += 6
        if "ultra" in self.modifiers:
            score += 10
        return int(min(25, max(3, score)))

    @property
    def detail_tier(self) -> str:
        if "ultra" in self.modifiers or "photoreal" in self.modifiers:
            return "ultra" if "ultra" in self.modifiers else "complex"
        if "complex" in self.modifiers or "cinematic" in self.modifiers:
            return "complex"
        if "detailed" in self.modifiers or "realistic" in self.modifiers:
            return "detailed"
        return "normal"

    @property
    def photoreal(self) -> bool:
        return bool(self.modifiers & {"photoreal", "realistic", "cinematic"})

    @property
    def label(self) -> str:
        bits = []
        if self.is_scene:
            bits.append("scene")
            if self.environment:
                bits.append(self.environment)
        if self.actors:
            names = [(a.label or a.obj) for a in self.actors[:3]]
            bits.append("+".join(names))
        elif self.objects:
            bits.append("+".join(self.objects[:3]))
        else:
            bits.append("freeform")
        n_col = len(self.colors) + sum(len(a.colors) for a in self.actors)
        if n_col:
            bits.append(f"{n_col}kleur")
        if self.detail_tier != "normal":
            bits.append(self.detail_tier)
        return "+".join(bits)


def _lookup(tok: str, table: Dict[str, str]) -> str | None:
    if tok in table:
        return table[tok]
    for suf in ("ene", "ste", "ere", "en", "es", "er", "e", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            stem = tok[: -len(suf)]
            if stem in table:
                return table[stem]
    return None


def _is_color_token(tok: str) -> bool:
    if tok in COLOR_MAP:
        return True
    for suf in ("e", "en", "er", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf) and tok[: -len(suf)] in COLOR_MAP:
            return True
    return False


def _token_color(tok: str) -> RGB | None:
    if tok in COLOR_MAP:
        return COLOR_MAP[tok]
    for suf in ("e", "en", "er", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            stem = tok[: -len(suf)]
            if stem in COLOR_MAP:
                return COLOR_MAP[stem]
    return None


def _weather_of(tok: str) -> str | None:
    if tok in WEATHER_MAP:
        return WEATHER_MAP[tok]
    for suf in ("achtig", "ige", "ig", "en", "e", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf) and tok[: -len(suf)] in WEATHER_MAP:
            return WEATHER_MAP[tok[: -len(suf)]]
    return None


def parse_prompt(prompt: str) -> PromptPlan:
    tokens = tokenize(prompt)
    seed = prompt_seed(prompt)
    objects: List[str] = []
    actors: List[Actor] = []
    parts: Set[str] = set()
    modifiers: Set[str] = set()
    unknown: List[str] = []
    colors = resolve_colors(tokens)

    pending_colors: List[RGB] = []
    pending_mods: Set[str] = set()

    def add_actor(obj: str, label: str = "") -> None:
        actors.append(
            Actor(
                obj=obj,
                colors=list(pending_colors),
                modifiers=set(pending_mods) | set(modifiers),
                label=label or obj,
            )
        )
        if obj not in objects:
            objects.append(obj)
        pending_colors.clear()
        pending_mods.clear()

    for tok in tokens:
        if tok in STOP or tok in SCENE_WORDS:
            continue
        if _is_color_token(tok):
            c = _token_color(tok)
            if c is not None and c not in pending_colors:
                pending_colors.append(c)
            continue
        if tok in ENV_MAP:
            continue

        weather = _weather_of(tok)
        if weather is not None:
            add_actor(weather, tok)
            continue

        part = _lookup(tok, PART_MAP)
        if part is not None:
            parts.add(part)
            continue

        mod = _lookup(tok, MODIFIERS)
        if mod is not None:
            modifiers.add(mod)
            pending_mods.add(mod)
            continue

        # Bekend object OF willekeurig woord → altijd een acteur
        if tok in OBJECT_MAP or _lookup(tok, OBJECT_MAP) or len(tok) > 2:
            obj = concept_from_word(tok)
            add_actor(obj, tok)
            if tok not in OBJECT_MAP and _lookup(tok, OBJECT_MAP) is None:
                unknown.append(tok)
            continue

    # Helemaal niets? → freeform uit hele prompt
    if not actors:
        slug = "_".join(tokens[:4]) if tokens else "idee"
        obj = concept_from_word(slug if tokens else "idea")
        actors = [Actor(obj=obj, colors=list(colors), modifiers=set(modifiers), label=slug)]
        objects = [obj]
        if tokens:
            unknown = [t for t in tokens if t not in STOP][:8]

    weather_objs = {"storm", "rain", "snow", "fire", "volcano", "ocean", "night"}

    for a in actors:
        if not a.colors and colors:
            a.colors = list(colors)

    env = detect_environment(tokens, modifiers)
    if any(a.obj in weather_objs for a in actors):
        env = env or ("storm" if any(a.obj == "storm" for a in actors) else env)
        scene = True
    else:
        scene = is_scene_prompt(tokens, len(actors), env)

    return PromptPlan(
        raw=prompt,
        tokens=tokens,
        seed=seed,
        objects=objects,
        actors=actors,
        parts=parts,
        modifiers=modifiers,
        colors=colors,
        unknown=unknown,
        environment=env,
        is_scene=scene,
    )


def token_embedding(tokens: List[str], dim: int = 64, seed: int = 0) -> "np.ndarray":
    import numpy as np

    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((256, dim)).astype(np.float32) * 0.15
    emb = np.zeros(dim, dtype=np.float32)
    if not tokens:
        emb[0] = 1.0
        return emb
    for i, tok in enumerate(tokens):
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec = proj[h % 256]
        weight = 1.0 / (1.0 + 0.15 * i)
        emb += vec * weight
        for j in range(len(tok) - 1):
            nh = int(hashlib.md5(tok[j : j + 2].encode()).hexdigest()[:8], 16)
            emb += proj[nh % 256] * 0.25 * weight
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb /= norm
    return emb
