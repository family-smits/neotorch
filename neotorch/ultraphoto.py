"""
Ultra Photoreal pipeline:
  Python (orchestrate + mesh)
  -> C (meshpack OBJ->ntmesh)
  -> C++ (GGX path tracer -> HDRF + BMP)
  -> Rust (bilateral denoise + ACES)
  -> Java PhotorealViewer
  -> Java PhotoStudio
  -> JavaScript browser viewer
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .export import export_obj
from .mesh import Mesh
from .paths import app_dir, bin_dir, frozen, java_dir, js_dir, output_dir, resource_root
from .repair import preview_decimate

MAX_TRACE_FACES = 32_000


def _tools():
    b = bin_dir()
    return b / "meshpack.exe", b / "pathtracer.exe", b / "denoise.exe"


def _run(cmd: list, cwd: Optional[Path] = None, label: str = "") -> None:
    print(f"[{label or 'run'}] {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd or app_dir()))
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed with code {r.returncode}")


def ensure_tools() -> None:
    meshpack, tracer, denoise = _tools()
    root = resource_root()

    if frozen():
        missing = [p.name for p in (meshpack, tracer, denoise) if not p.exists()]
        if missing:
            raise RuntimeError(f"Gebundelde tools ontbreken in exe: {', '.join(missing)}")
        for name in ("PhotorealViewer", "PhotoStudio"):
            if not (java_dir() / f"{name}.class").exists():
                raise RuntimeError(f"Java class ontbreekt in exe: {name}")
        return

    if not meshpack.exists():
        _run(["cmd", "/c", str(root / "native" / "c" / "build.bat")], label="C-build")
    if not tracer.exists():
        _run(["cmd", "/c", str(root / "native" / "pathtracer" / "build.bat")], label="C++-build")
    else:
        src = root / "native" / "pathtracer" / "tracer.cpp"
        if src.exists() and src.stat().st_mtime > tracer.stat().st_mtime:
            _run(["cmd", "/c", str(root / "native" / "pathtracer" / "build.bat")], label="C++-rebuild")
    if not denoise.exists():
        _run(["cmd", "/c", str(root / "native" / "rust" / "build.bat")], label="Rust-build")

    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("javac missing")
    jdir = java_dir()
    for name in ("PhotorealViewer", "PhotoStudio"):
        src = jdir / f"{name}.java"
        cls = jdir / f"{name}.class"
        if not cls.exists() or src.stat().st_mtime > cls.stat().st_mtime:
            _run([javac, str(src)], cwd=jdir, label=f"Java-{name}")


def _to_png(bmp: Path, png: Path) -> Path:
    try:
        from PIL import Image

        Image.open(bmp).save(png)
        return png
    except Exception:
        try:
            import pygame

            pygame.init()
            s = pygame.image.load(str(bmp))
            pygame.image.save(s, str(png))
            return png
        except Exception as e:
            print(f"[Python] PNG convert skipped: {e}")
            return bmp


def render_ultra(
    mesh: Mesh,
    out_dir: Optional[Path] = None,
    width: int = 800,
    height: int = 600,
    spp: int = 96,
    open_viewers: bool = True,
) -> Tuple[Path, Path]:
    """Full multi-language ultra photoreal. Returns (final_image, obj_path)."""
    ensure_tools()
    meshpack, tracer, denoise = _tools()
    out_dir = Path(out_dir or output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in mesh.name)[:40] or "model"

    render_mesh = mesh
    if mesh.face_count > MAX_TRACE_FACES:
        print(f"[Python] Decimate {mesh.face_count} -> <={MAX_TRACE_FACES}")
        render_mesh = preview_decimate(mesh, max_faces=MAX_TRACE_FACES)
        render_mesh.name = mesh.name

    obj_path = out_dir / f"{safe}_ultra.obj"
    ntmesh = out_dir / f"{safe}_ultra.ntmesh"
    raw_bmp = out_dir / f"{safe}_ultra_raw.bmp"
    hdrf = out_dir / f"{safe}_ultra.hdrf"
    final_bmp = out_dir / f"{safe}_ultra.bmp"
    final_png = out_dir / f"{safe}_ultra.png"

    print("[Python] Export OBJ...")
    export_obj(render_mesh, obj_path)
    _run([meshpack, obj_path, ntmesh], label="C")
    _run([tracer, ntmesh, raw_bmp, str(width), str(height), str(spp), hdrf], label="C++")
    _run([denoise, hdrf, final_bmp, "2.4", "0.10"], label="Rust")

    img = _to_png(final_bmp, final_png)
    canon = out_dir / "ultra_photo.png"
    if img.suffix.lower() == ".png":
        shutil.copy2(img, canon)
    else:
        _to_png(final_bmp, canon)

    if open_viewers:
        open_all_viewers(img if img.exists() else final_bmp, title=mesh.name)

    return (canon if canon.exists() else final_bmp), obj_path


def open_all_viewers(image: Path, title: str = "Ultra Photoreal") -> None:
    image = image.resolve()
    jdir = java_dir()
    java = shutil.which("java")
    if java:
        subprocess.Popen([java, "-cp", str(jdir), "PhotorealViewer", str(image)], cwd=str(jdir))
        subprocess.Popen(
            [java, "-cp", str(jdir), "PhotoStudio", str(image), title],
            cwd=str(jdir),
        )
    else:
        print("[Java] java niet gevonden — skip desktop viewers")

    node = shutil.which("node")
    if node:
        env = os.environ.copy()
        # Serve from app_dir so output/ naast exe bereikbaar is; assets uit resource_root
        env["NEOTORCH_ROOT"] = str(app_dir())
        env["NEOTORCH_ASSETS"] = str(resource_root())
        subprocess.Popen(
            [node, str(js_dir() / "serve.js"), str(image)],
            cwd=str(app_dir()),
            env=env,
        )
    else:
        print("[JS] node not found — skip browser viewer")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print("Usage: python -m neotorch.ultraphoto <prompt|mesh.obj> [w] [h] [spp]")
        return 1
    ensure_tools()
    meshpack, tracer, denoise = _tools()
    target = argv[0]
    w = int(argv[1]) if len(argv) > 1 else 800
    h = int(argv[2]) if len(argv) > 2 else 600
    spp = int(argv[3]) if len(argv) > 3 else 96

    if target.lower().endswith(".obj"):
        out = output_dir()
        stem = Path(target).stem[:40]
        ntmesh = out / f"{stem}_ultra.ntmesh"
        raw_bmp = out / f"{stem}_ultra_raw.bmp"
        hdrf = out / f"{stem}_ultra.hdrf"
        final_bmp = out / f"{stem}_ultra.bmp"
        _run([meshpack, target, ntmesh], label="C")
        _run([tracer, ntmesh, raw_bmp, str(w), str(h), str(spp), hdrf], label="C++")
        _run([denoise, hdrf, final_bmp, "2.4", "0.10"], label="Rust")
        png = _to_png(final_bmp, out / f"{stem}_ultra.png")
        shutil.copy2(png if png.suffix == ".png" else final_bmp, out / "ultra_photo.png")
        open_all_viewers(out / "ultra_photo.png", title=stem)
        return 0

    from .generate import Model3DGenerator

    gen = Model3DGenerator(detail=22, max_verts=40_000)
    print(f"[Python] Generating mesh for: {target}")
    mesh = gen.generate(target)
    render_ultra(mesh, width=w, height=h, spp=spp, open_viewers=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
