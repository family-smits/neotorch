"""Prompt-analyse: woorden → semantische bouwstenen voor 3D-generatie."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .colors import COLOR_MAP, RGB, resolve_colors
from .scene import ENV_MAP, SCENE_WORDS, Actor, detect_environment, is_scene_prompt
from .weather import WEATHER_MAP


def tokenize(prompt: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", prompt.lower())


def prompt_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# Object-synoniemen → canonieke concept-id
OBJECT_MAP: Dict[str, str] = {
    # basis
    "cube": "box", "box": "box", "blok": "box", "kubus": "box", "vierkant": "box",
    "sphere": "sphere", "ball": "sphere", "bol": "sphere", "bal": "sphere",
    "cylinder": "cylinder", "cilinder": "cylinder", "buis": "cylinder", "pijp": "cylinder",
    "cone": "cone", "kegel": "cone",
    "torus": "torus", "donut": "torus", "ring": "torus",
    "pyramid": "pyramid", "piramide": "pyramid",
    # gebouwen / natuur
    "house": "house", "huis": "house", "home": "house", "cottage": "house", "woning": "house",
    "tree": "tree", "boom": "tree", "plant": "tree",
    "tower": "tower", "toren": "tower", "spire": "tower",
    "castle": "castle", "kasteel": "castle", "fort": "castle", "burcht": "castle",
    "crystal": "crystal", "kristal": "crystal", "edelsteen": "crystal", "diamant": "crystal",
    "rock": "rock", "steen": "rock", "rots": "rock", "boulder": "rock",
    "mountain": "mountain", "berg": "mountain", "heuvel": "mountain",
    # wezens / figuren
    "robot": "robot", "android": "robot", "mecha": "robot",
    "human": "humanoid", "mens": "humanoid", "man": "humanoid", "vrouw": "humanoid",
    "person": "humanoid", "persoon": "humanoid", "figuur": "humanoid", "character": "humanoid",
    "animal": "creature", "dier": "creature", "monster": "creature", "beest": "creature",
    "dragon": "dragon", "draak": "dragon",
    "bird": "bird", "vogel": "bird",
    "cat": "quadruped", "kat": "quadruped", "dog": "quadruped", "hond": "quadruped",
    "horse": "quadruped", "paard": "quadruped",
    # objecten
    "sword": "sword", "zwaard": "sword", "blade": "sword",
    "car": "vehicle", "auto": "vehicle", "wagen": "vehicle", "truck": "vehicle",
    "boat": "boat", "boot": "boat", "ship": "boat", "schip": "boat",
    "chair": "chair", "stoel": "chair", "table": "table", "tafel": "table",
    "hat": "hat", "hoed": "hat", "crown": "crown", "kroon": "crown",
    "star": "star", "ster": "star",
    "mushroom": "mushroom", "paddenstoel": "mushroom",
    "rocket": "rocket", "raket": "rocket",
    "wand": "wand", "staf": "wand", "staff": "wand",
    "book": "book", "boek": "book",
    "apple": "fruit", "appel": "fruit", "fruit": "fruit",
}

# Onderdelen die als add-ons op een basisvorm komen
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
}

MODIFIERS: Dict[str, str] = {
    "tall": "tall", "hoog": "tall", "hoge": "tall", "lang": "tall", "lange": "tall",
    "short": "short", "kort": "short", "korte": "short", "laag": "short",
    "big": "big", "groot": "big", "grote": "big", "huge": "big", "reusachtig": "big",
    "small": "small", "klein": "small", "kleine": "small", "tiny": "small",
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
    "futuristic": "tech", "toekomstig": "tech", "sci": "tech", "fi": "tech",
    "cute": "cute", "schattig": "cute", "lief": "cute",
    "scary": "scary", "eng": "scary", "duister": "scary", "dark": "scary",
    "spitse": "spiky", "scherpe": "spiky", "magische": "ornate",
    "schattige": "cute", "kleine": "small", "grote": "big", "lange": "tall",
    # complexiteit / detail
    "complex": "complex", "complexe": "complex", "ingewikkeld": "complex",
    "detailed": "detailed", "detail": "detailed", "gedetailleerd": "detailed",
    "intricate": "detailed", "highpoly": "ultra", "hd": "ultra", "ultra": "ultra",
    "insane": "ultra", "epic": "ultra", "mega": "ultra", "4k": "ultra",
    "hyper": "ultra", "max": "ultra", "extreme": "ultra", "extreem": "ultra",
    "rainbow": "rainbow", "regenboog": "rainbow", "multicolor": "rainbow",
    "kleurrijk": "rainbow", "colorful": "rainbow",
    "scene": "scene", "landschap": "scene", "diorama": "scene",
    # fotorealisme
    "photoreal": "photoreal", "photorealistic": "photoreal",
    "fotorealistisch": "photoreal", "fotorealistische": "photoreal",
    "foto": "photoreal", "realistic": "realistic", "realistisch": "realistic",
    "realistische": "realistic", "cinematic": "cinematic", "cinematisch": "cinematic",
}

CONNECTIVES = {
    "en", "and", "met", "with", "plus", "naast", "bij", "of", "a", "an",
    "de", "het", "een", "the", "in", "op", "near", "tegen", "voor", "achter",
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
            bits.append("+".join(a.obj for a in self.actors[:3]))
        elif self.objects:
            bits.append("+".join(self.objects[:3]))
        else:
            bits.append("neural")
        n_col = len(self.colors) + sum(len(a.colors) for a in self.actors)
        if n_col:
            bits.append(f"{n_col}kleur")
        if self.detail_tier != "normal":
            bits.append(self.detail_tier)
        return "+".join(bits)


def _lookup(tok: str, table: Dict[str, str]) -> str | None:
    """Match woord of Nederlandse/Engelse uitgang (spitse→spits, wings→wing)."""
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


def parse_prompt(prompt: str) -> PromptPlan:
    tokens = tokenize(prompt)
    seed = prompt_seed(prompt)
    objects: List[str] = []
    actors: List[Actor] = []
    parts: Set[str] = set()
    modifiers: Set[str] = set()
    unknown: List[str] = []
    colors = resolve_colors(tokens)

    # Per-acteur kleuren: "rode robot en blauwe kat"
    pending_colors: List[RGB] = []
    pending_mods: Set[str] = set()

    for tok in tokens:
        if tok in CONNECTIVES or tok in SCENE_WORDS:
            continue
        if _is_color_token(tok):
            c = _token_color(tok)
            if c is not None and c not in pending_colors:
                pending_colors.append(c)
            continue
        if tok in ENV_MAP:
            continue
        # Weer/sfeer eerst (storm → echte scene, geen neural-bol)
        weather = WEATHER_MAP.get(tok)
        if weather is None:
            for suf in ("achtig", "ige", "ig", "en", "e", "s"):
                if len(tok) > len(suf) + 2 and tok.endswith(suf) and tok[: -len(suf)] in WEATHER_MAP:
                    weather = WEATHER_MAP[tok[: -len(suf)]]
                    break
        if weather is not None:
            actors.append(Actor(obj=weather, colors=list(pending_colors), modifiers=set(pending_mods) | set(modifiers)))
            if weather not in objects:
                objects.append(weather)
            pending_colors = []
            pending_mods = set()
            continue
        obj = _lookup(tok, OBJECT_MAP)
        if obj is not None:
            actors.append(Actor(obj=obj, colors=list(pending_colors), modifiers=set(pending_mods) | set(modifiers)))
            if obj not in objects:
                objects.append(obj)
            pending_colors = []
            pending_mods = set()
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
        if len(tok) > 2:
            unknown.append(tok)

    # Onbekende woorden: fuzzy match op weather/object i.p.v. disco-bol
    if not actors:
        inferred = None
        for u in unknown:
            if u in WEATHER_MAP:
                inferred = WEATHER_MAP[u]
                break
            for key, val in WEATHER_MAP.items():
                if key in u or u in key:
                    inferred = val
                    break
            if inferred:
                break
            for key, val in OBJECT_MAP.items():
                if key in u or u.startswith(key[:4]):
                    inferred = val
                    break
            if inferred:
                break
        if inferred is None:
            # Laatste redmiddel: neutraal "rock"/landschap i.p.v. felle neural blob
            inferred = "mountain" if any(x in unknown for x in ("berg", "land", "wereld")) else "rock"
        objects.append(inferred)
        actors = [Actor(obj=inferred, colors=list(colors), modifiers=set(modifiers))]

    weather_objs = {"storm", "rain", "snow", "fire", "volcano", "ocean", "night"}

    # Globale kleuren doorgeven aan acteurs zonder eigen kleur
    for a in actors:
        if not a.colors and colors:
            # Roteer globale kleuren per acteur
            a.colors = list(colors)

    env = detect_environment(tokens, modifiers)
    # Weer-concepten zijn altijd scenes
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
    # Vaste projectiematrix per "vocab bucket"
    proj = rng.standard_normal((256, dim)).astype(np.float32) * 0.15
    emb = np.zeros(dim, dtype=np.float32)
    if not tokens:
        emb[0] = 1.0
        return emb
    for i, tok in enumerate(tokens):
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec = proj[h % 256]
        # Positie-gewogen: vroege woorden iets zwaarder
        weight = 1.0 / (1.0 + 0.15 * i)
        emb += vec * weight
        # Karakter-n-grams
        for j in range(len(tok) - 1):
            nh = int(hashlib.md5(tok[j : j + 2].encode()).hexdigest()[:8], 16)
            emb += proj[nh % 256] * 0.25 * weight
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb /= norm
    return emb
