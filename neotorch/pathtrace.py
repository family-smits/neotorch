"""Python bridge: NeoTorch mesh → C++ path tracer → Java viewer."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .export import export_obj
from .mesh import Mesh
from .paths import bin_dir, frozen, java_dir, output_dir, resource_root
from .repair import preview_decimate

TRACER_EXE = None  # resolved at call time
MAX_TRACE_FACES = 28_000


def ensure_tracer() -> Path:
    exe = bin_dir() / "pathtracer.exe"
    if exe.exists():
        return exe
    if frozen():
        raise FileNotFoundError("pathtracer.exe ontbreekt in gebundelde exe")
    bat = resource_root() / "native" / "pathtracer" / "build.bat"
    print("[pathtrace] Building C++ path tracer...")
    r = subprocess.run(["cmd", "/c", str(bat)], cwd=str(bat.parent), capture_output=True, text=True)
    if r.returncode != 0 or not exe.exists():
        raise RuntimeError(f"C++ build failed:\n{(r.stdout or '')}\n{(r.stderr or '')}")
    return exe


def ensure_java_viewer() -> Path:
    src = java_dir() / "PhotorealViewer.java"
    cls = java_dir() / "PhotorealViewer.class"
    if cls.exists():
        return cls
    if frozen():
        raise FileNotFoundError("PhotorealViewer.class ontbreekt in exe")
    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("javac niet gevonden")
    r = subprocess.run([javac, str(src)], cwd=str(java_dir()), capture_output=True, text=True)
    if r.returncode != 0 or not cls.exists():
        raise RuntimeError(f"javac failed:\n{r.stderr or r.stdout}")
    return cls


def render_photoreal(
    mesh: Mesh,
    out_dir: Optional[Path] = None,
    width: int = 640,
    height: int = 480,
    spp: int = 24,
    open_viewer: bool = True,
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir or output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in mesh.name)[:48] or "model"
    obj_path = out_dir / f"{safe}_photo.obj"
    bmp_path = out_dir / f"{safe}_photo.bmp"

    render_mesh = mesh
    if mesh.face_count > MAX_TRACE_FACES:
        print(f"[pathtrace] Decimating {mesh.face_count} -> <={MAX_TRACE_FACES} faces...")
        render_mesh = preview_decimate(mesh, max_faces=MAX_TRACE_FACES)
        render_mesh.name = mesh.name
    export_obj(render_mesh, obj_path)
    exe = ensure_tracer()
    print(f"[pathtrace] Rendering {width}x{height} @ {spp} spp ({render_mesh.face_count} tris)...")
    r = subprocess.run(
        [str(exe), str(obj_path), str(bmp_path), str(width), str(height), str(spp)],
        cwd=str(out_dir),
    )
    if r.returncode != 0 or not bmp_path.exists():
        raise RuntimeError(f"Path tracer failed (code {r.returncode})")
    if open_viewer:
        show_in_java(bmp_path)
    return obj_path, bmp_path


def show_in_java(image_path: Path) -> None:
    ensure_java_viewer()
    java = shutil.which("java")
    if not java:
        raise RuntimeError("java niet gevonden")
    subprocess.Popen(
        [java, "-cp", str(java_dir()), "PhotorealViewer", str(image_path.resolve())],
        cwd=str(java_dir()),
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print("Usage: python -m neotorch.pathtrace <mesh.obj> [out.bmp] [w] [h] [spp]")
        return 1
    obj = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else obj.with_suffix(".bmp")
    w = int(argv[2]) if len(argv) > 2 else 640
    h = int(argv[3]) if len(argv) > 3 else 480
    spp = int(argv[4]) if len(argv) > 4 else 24
    exe = ensure_tracer()
    r = subprocess.run([str(exe), str(obj), str(out), str(w), str(h), str(spp)])
    if r.returncode == 0 and out.exists():
        try:
            show_in_java(out)
        except Exception as e:
            print(f"[pathtrace] Java viewer: {e}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
