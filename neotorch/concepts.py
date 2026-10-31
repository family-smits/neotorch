"""Elk willekeurig woord → bouwbaar 3D-concept (geen vaste whitelist)."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

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
    "hut": "house", "cabin": "house", "chalet": "house", "villa": "house", "iglo": "house",
    "barn": "house", "schuur": "house", "tent": "house", "camp": "house", "kamp": "house",
    "tree": "tree", "boom": "tree", "plant": "tree", "palm": "tree", "oak": "tree", "eik": "tree",
    "pine": "tree", "willow": "tree", "wilg": "tree", "cactus": "tree", "bamboo": "tree", "bamboe": "tree",
    "tower": "tower", "toren": "tower", "spire": "tower", "lighthouse": "tower", "vuurtoren": "tower",
    "skyscraper": "tower", "wolkenkrabber": "tower", "mill": "tower", "molen": "tower",
    "lamp": "tower", "lantern": "tower", "lantaarn": "tower", "candle": "tower", "kaars": "tower",
    "fountain": "tower", "fontein": "tower", "totem": "tower",
    "castle": "castle", "kasteel": "castle", "fort": "castle", "burcht": "castle",
    "palace": "castle", "paleis": "castle", "temple": "castle", "tempel": "castle",
    "church": "castle", "kerk": "castle", "cathedral": "castle", "kathedraal": "castle",
    "bridge": "castle", "brug": "castle",
    "crystal": "crystal", "kristal": "crystal", "edelsteen": "crystal", "diamant": "crystal",
    "gem": "crystal", "jewel": "crystal", "juweel": "crystal", "ice": "crystal", "ijs": "crystal",
    "moon": "crystal", "maan": "crystal", "flame": "crystal", "vlam": "crystal",
    "rock": "rock", "steen": "rock", "rots": "rock", "boulder": "rock", "pebble": "rock",
    "kei": "rock", "meteor": "rock", "meteoor": "rock", "asteroid": "rock",
    "mountain": "mountain", "berg": "mountain", "heuvel": "mountain", "hill": "mountain",
    "cliff": "mountain", "klif": "mountain",
    # wezens / figuren
    "robot": "robot", "android": "robot", "mecha": "robot", "cyborg": "robot",
    "droid": "robot", "bot": "robot", "drone": "robot", "armor": "robot", "harnas": "robot",
    "human": "humanoid", "mens": "humanoid", "man": "humanoid", "vrouw": "humanoid",
    "woman": "humanoid", "boy": "humanoid", "girl": "humanoid", "jongen": "humanoid",
    "meisje": "humanoid", "person": "humanoid", "persoon": "humanoid",
    "figuur": "humanoid", "character": "humanoid", "karakter": "humanoid",
    "wizard": "humanoid", "tovenaar": "humanoid", "witch": "humanoid", "heks": "humanoid",
    "knight": "humanoid", "ridder": "humanoid", "ninja": "humanoid", "samurai": "humanoid",
    "pirate": "humanoid", "piraat": "humanoid", "soldier": "humanoid", "soldaat": "humanoid",
    "king": "humanoid", "koning": "humanoid", "queen": "humanoid", "koningin": "humanoid",
    "hero": "humanoid", "held": "humanoid", "villain": "humanoid", "schurk": "humanoid",
    "zombie": "humanoid", "skeleton": "humanoid", "skelet": "humanoid",
    "ghost": "humanoid", "geest": "humanoid", "alien": "humanoid", "astronaut": "humanoid",
    "elf": "humanoid", "dwarf": "humanoid", "dwerg": "humanoid",
    "snowman": "humanoid", "sneeuwpop": "humanoid", "statue": "humanoid", "beeld": "humanoid",
    "skull": "humanoid", "schedel": "humanoid",
    "animal": "creature", "dier": "creature", "monster": "creature", "beest": "creature",
    "creature": "creature", "wezen": "creature", "beast": "creature", "demon": "creature",
    "goblin": "creature", "troll": "creature", "orc": "creature", "fairy": "creature", "fee": "creature",
    "snake": "creature", "slang": "creature", "fish": "creature", "vis": "creature",
    "shark": "creature", "haai": "creature", "whale": "creature", "walvis": "creature",
    "octopus": "creature", "inktvis": "creature", "spider": "creature", "spin": "creature",
    "insect": "creature", "bug": "creature", "bee": "creature", "bij": "creature",
    "dragon": "dragon", "draak": "dragon", "wyvern": "dragon",
    "bird": "bird", "vogel": "bird", "eagle": "bird", "adelaar": "bird", "owl": "bird", "uil": "bird",
    "crow": "bird", "kraai": "bird", "duck": "bird", "eend": "bird", "chicken": "bird", "kip": "bird",
    "penguin": "bird", "pinguïn": "bird", "phoenix": "bird", "butterfly": "bird", "vlinder": "bird",
    "cat": "quadruped", "kat": "quadruped", "kitten": "quadruped", "poes": "quadruped",
    "dog": "quadruped", "hond": "quadruped", "puppy": "quadruped", "wolf": "quadruped",
    "fox": "quadruped", "vos": "quadruped", "bear": "quadruped", "beer": "quadruped",
    "lion": "quadruped", "leeuw": "quadruped", "tiger": "quadruped", "tijger": "quadruped",
    "horse": "quadruped", "paard": "quadruped", "cow": "quadruped", "koe": "quadruped",
    "pig": "quadruped", "varken": "quadruped", "sheep": "quadruped", "schaap": "quadruped",
    "deer": "quadruped", "hert": "quadruped", "rabbit": "quadruped", "konijn": "quadruped",
    "mouse": "quadruped", "muis": "quadruped", "rat": "quadruped",
    "elephant": "quadruped", "olifant": "quadruped", "giraffe": "quadruped", "zebra": "quadruped",
    # wapens / tools / meubels
    "sword": "sword", "zwaard": "sword", "blade": "sword", "mes": "sword", "knife": "sword",
    "axe": "sword", "bijl": "sword", "spear": "sword", "speer": "sword",
    "car": "vehicle", "auto": "vehicle", "wagen": "vehicle", "truck": "vehicle",
    "bus": "vehicle", "taxi": "vehicle", "jeep": "vehicle", "tank": "vehicle",
    "bike": "vehicle", "fiets": "vehicle", "motorcycle": "vehicle", "motor": "vehicle",
    "train": "vehicle", "trein": "vehicle", "tram": "vehicle",
    "plane": "vehicle", "vliegtuig": "vehicle", "airplane": "vehicle", "jet": "vehicle",
    "helicopter": "vehicle", "helikopter": "vehicle",
    "boat": "boat", "boot": "boat", "ship": "boat", "schip": "boat", "yacht": "boat",
    "submarine": "boat", "onderzeeer": "boat", "canoe": "boat", "kano": "boat",
    "chair": "chair", "stoel": "chair", "throne": "chair", "troon": "chair",
    "sofa": "chair", "bankstel": "chair",
    "table": "table", "tafel": "table", "desk": "table", "bureau": "table",
    "hat": "hat", "hoed": "hat", "cap": "hat", "pet": "hat", "helmet": "hat", "helm": "hat",
    "crown": "crown", "kroon": "crown", "tiara": "crown",
    "star": "star", "ster": "star", "sun": "star", "zon": "star",
    "mushroom": "mushroom", "paddenstoel": "mushroom", "fungus": "mushroom",
    "flower": "mushroom", "bloem": "mushroom", "rose": "mushroom", "roos": "mushroom",
    "rocket": "rocket", "raket": "rocket", "spaceship": "rocket", "ruimteschip": "rocket",
    "wand": "wand", "staf": "wand", "staff": "wand", "cane": "wand", "scepter": "wand",
    "key": "wand", "sleutel": "wand", "guitar": "wand", "gitaar": "wand",
    "book": "book", "boek": "book", "tome": "book", "scroll": "book",
    "apple": "fruit", "appel": "fruit", "fruit": "fruit", "banana": "fruit", "banaan": "fruit",
    "pear": "fruit", "peer": "fruit", "strawberry": "fruit", "aardbei": "fruit",
    "heart": "fruit", "hart": "fruit", "pumpkin": "fruit", "pompoen": "fruit",
    "pizza": "fruit", "burger": "fruit", "cake": "fruit", "taart": "fruit",
    "bread": "box", "brood": "box", "cheese": "box", "kaas": "box",
    "clock": "box", "klok": "box", "watch": "box", "horloge": "box",
    "phone": "box", "telefoon": "box", "laptop": "box", "computer": "box", "pc": "box",
    "camera": "box", "radio": "box", "tv": "box", "televisie": "box",
    "bed": "box", "closet": "box", "kast": "box", "door": "box", "deur": "box",
    "window": "box", "raam": "box", "fence": "box", "hek": "box", "gate": "box", "poort": "box",
    "ladder": "box", "stairs": "box", "trap": "box", "shield": "box", "schild": "box",
    "piano": "box", "cup": "cylinder", "mok": "cylinder", "mug": "cylinder",
    "glass": "cylinder", "glas": "cylinder", "bottle": "cylinder", "fles": "cylinder",
    "vase": "cylinder", "vaas": "cylinder", "drum": "cylinder", "trommel": "cylinder",
    "well": "cylinder", "put": "cylinder", "coin": "torus", "munt": "torus",
    "carrot": "cone", "wortel": "cone",
    "cloud": "sphere", "wolk": "sphere",
    "football": "sphere", "voetbal": "sphere", "basketball": "sphere",
}

ARCHETYPE_BUCKETS: List[str] = [
    "freeform", "humanoid", "creature", "quadruped", "bird", "robot",
    "vehicle", "boat", "house", "tower", "tree", "crystal", "rock",
    "mountain", "sword", "fruit", "mushroom", "rocket", "hat", "chair",
]

HINTS: List[Tuple[str, str]] = [
    ("bot", "robot"), ("droid", "robot"), ("mech", "robot"),
    ("man", "humanoid"), ("woman", "humanoid"), ("person", "humanoid"),
    ("boy", "humanoid"), ("girl", "humanoid"), ("king", "humanoid"), ("queen", "humanoid"),
    ("bird", "bird"), ("vogel", "bird"), ("wing", "bird"),
    ("cat", "quadruped"), ("dog", "quadruped"), ("wolf", "quadruped"), ("fox", "quadruped"),
    ("bear", "quadruped"), ("lion", "quadruped"), ("horse", "quadruped"), ("paard", "quadruped"),
    ("car", "vehicle"), ("auto", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"),
    ("plane", "vehicle"), ("jet", "vehicle"), ("train", "vehicle"),
    ("boat", "boat"), ("ship", "boat"), ("boot", "boat"), ("schip", "boat"),
    ("house", "house"), ("huis", "house"), ("home", "house"), ("hut", "house"),
    ("tower", "tower"), ("toren", "tower"), ("castle", "castle"), ("kasteel", "castle"),
    ("tree", "tree"), ("boom", "tree"), ("plant", "tree"), ("flower", "mushroom"), ("bloem", "mushroom"),
    ("crystal", "crystal"), ("kristal", "crystal"),
    ("rock", "rock"), ("steen", "rock"), ("berg", "mountain"), ("mount", "mountain"),
    ("sword", "sword"), ("zwaard", "sword"), ("gun", "wand"), ("wapen", "sword"),
    ("fruit", "fruit"), ("food", "fruit"), ("eten", "fruit"), ("cake", "fruit"),
    ("rocket", "rocket"), ("raket", "rocket"), ("space", "rocket"),
    ("hat", "hat"), ("hoed", "hat"), ("crown", "crown"), ("kroon", "crown"),
    ("monster", "creature"), ("beast", "creature"), ("dragon", "dragon"), ("draak", "dragon"),
    ("fish", "creature"), ("vis", "creature"), ("snake", "creature"), ("slang", "creature"),
]


def word_hash(word: str) -> int:
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)


def concept_from_word(tok: str) -> str:
    """Map elk woord naar een builder-concept — altijd een geldig resultaat."""
    if tok in OBJECT_MAP:
        return OBJECT_MAP[tok]
    for suf in ("ene", "ste", "ere", "jes", "je", "en", "es", "er", "e", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            stem = tok[: -len(suf)]
            if stem in OBJECT_MAP:
                return OBJECT_MAP[stem]
    for key, val in OBJECT_MAP.items():
        if len(key) >= 4 and (key in tok or tok.startswith(key[: max(4, len(key) - 1)])):
            return val
    for needle, arch in HINTS:
        if needle in tok:
            return arch
    return ARCHETYPE_BUCKETS[word_hash(tok) % len(ARCHETYPE_BUCKETS)]
