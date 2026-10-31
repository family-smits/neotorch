#!/usr/bin/env python3
"""CLI: genereer een 3D-model met NeoTorch (lokaal, geen API-keys)."""

from __future__ import annotations

import argparse
from pathlib import Path

from neotorch import Model3DGenerator, export_obj, export_stl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeoTorch 3D Model Generator — lokaal, zonder Torch/Ollama API-keys",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="a medieval castle",
        help='Tekstprompt, bijv. "huis", "robot", "crystal tower"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Outputmap of bestand zonder extensie (default: output)",
    )
    parser.add_argument(
        "--format",
        choices=("obj", "stl", "both"),
        default="both",
        help="Exportformaat (default: both)",
    )
    parser.add_argument(
        "--detail",
        type=int,
        default=24,
        help="Mesh-detailniveau (default: 24)",
    )
    parser.add_argument(
        "--deform",
        type=float,
        default=0.12,
        help="Sterkte van NeoTorch-vervorming 0–1 (default: 0.12)",
    )
    args = parser.parse_args()

    generator = Model3DGenerator(detail=args.detail, deform_strength=args.deform)
    mesh = generator.generate(args.prompt)

    out = Path(args.output)
    if out.suffix.lower() in {".obj", ".stl"}:
        stem = out.with_suffix("")
        fmt = out.suffix.lower().lstrip(".")
    else:
        stem = out
        fmt = args.format

    written: list[Path] = []
    if fmt in ("obj", "both"):
        written.append(export_obj(mesh, stem.with_suffix(".obj")))
    if fmt in ("stl", "both"):
        written.append(export_stl(mesh, stem.with_suffix(".stl")))

    print(f"Prompt : {args.prompt}")
    print(f"Model  : {mesh.name}")
    print(f"Verts  : {mesh.vertex_count}")
    print(f"Faces  : {mesh.face_count}")
    for path in written:
        print(f"Saved  : {path.resolve()}")


if __name__ == "__main__":
    main()
