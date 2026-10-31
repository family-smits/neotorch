# NeoTorch 3D Model Generator

Lokale 3D-modelgenerator in Python. Gebruikt **NeoTorch** — een eigen minimale tensor/neural module. Geen PyTorch API-keys, geen Ollama, geen cloud.

## Installatie

```bash
pip install -r requirements.txt
```

## Één EXE (alles erin)

```bash
build_exe.bat
```

Output: `dist\NeoTorch.exe` (logo = `ico.ico`). Bevat GUI + C/C++/Rust tools + Java classes + JS viewer.

```bash
# of vanuit source
pip install -r requirements.txt
python gui.py
```

Links: **token-teller** (50/dag) · midden: **AI-bar** · rechts: **3D-preview**.  
Lastigere vormen (castle, robot) kosten meer tokens dan eenvoudige (box, sphere).

## CLI

```bash
python generate_3d.py "medieval castle"
python generate_3d.py "robot" -o models/robot --format obj
python generate_3d.py "kristal toren" --detail 32 --deform 0.2
```

Of in code:

```python
from neotorch import Model3DGenerator, export_obj

gen = Model3DGenerator(detail=24)
mesh = gen.generate("een boom naast een huis")
export_obj(mesh, "output/boom_huis.obj")
```

## Herkende vormen

| Prompt-woorden | Model |
|---|---|
| cube, box, kubus, blok | doos |
| sphere, bol, bal | bol |
| cylinder, cilinder, zuil | cilinder |
| cone, kegel | kegel |
| torus, donut, ring | torus |
| pyramid, piramide | piramide |
| house, huis | huis |
| tree, boom | boom |
| tower, toren | toren |
| castle, kasteel | kasteel |
| crystal, kristal | kristal |
| robot | robot |
| sword, zwaard | zwaard |

De generator **parst je prompt**: objecten, onderdelen, modifiers **én kleuren**.  
**Alles mag** — onbekende woorden worden freeform/archetype-meshes (geen vaste whitelist).

### Kleuren (multi-color)
Elk model krijgt **meerdere** kleurzones (object-palette + promptkleuren).  
Per acteur: `rode robot en blauwe kat` → robot rood tinten, kat blauw tinten.

### Scenes
Meerdere objecten of omgeving → volle scene met grond + props:
```text
rode robot en blauwe kat in een bos
scene dorp met huis en boom
strand met boot
kasteel en draak in een landschap
```
Omgevingen: `bos`, `dorp`, `strand`, `boerderij`, `tuin`, `woestijn`, `meer`, `stad`, `grot`

### Weer & sfeer (geen regenboogbol)
```text
storm
regen / sneeuw / vuur
vulkaan / oceaan / nacht
```

### Materials
Kleuren zijn nu **tonale families** (schaduw→licht van dezelfde tint), geen regenboogzones — behalve bij `regenboog`.

### Ultra Photoreal (JS · Java · Python · Java · C++ · Rust · C)
Volledige pipeline:

1. **Python** — mesh genereren / orkestratie  
2. **C** — `meshpack` OBJ → binary `.ntmesh`  
3. **C++** — GGX PBR path tracer → HDR float + BMP  
4. **Rust** — bilateral denoise + ACES tonemap  
5. **Java** — `PhotorealViewer`  
6. **Java** — `PhotoStudio` metadata  
7. **JavaScript** — browser viewer (`js/serve.js`)

```bash
# tools bouwen (eenmalig)
native\c\build.bat
native\pathtracer\build.bat
native\rust\build.bat
javac java\PhotorealViewer.java java\PhotoStudio.java

# GUI: knop **ULTRA**
python gui.py

# CLI
python -m neotorch.ultraphoto "rode robot" 800 600 96
```

### Complexiteit
| Woord | ~vertices |
|---|---|
| (normaal) | ~6k |
| `detailed` / `realistic` | ~16k |
| `complex` / `fotorealistisch` | ~40k |
| `ultra` | tot ~100k |

## NeoTorch

`neotorch/` bevat:

- `tensor` — Tensor + basisops
- `nn` — Linear, ReLU, Tanh, Sequential
- `generate` — prompt → mesh via DeformNet
- `primitives` / `mesh` / `export` — geometrie en OBJ/STL

Open `.obj`-bestanden in Blender, MeshLab of een online OBJ-viewer.
