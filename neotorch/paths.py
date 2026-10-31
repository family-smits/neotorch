"""Pad-resolutie voor dev én PyInstaller onefile (.exe)."""

from __future__ import annotations

import sys
from pathlib import Path


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def resource_root() -> Path:
    """Leesbare assets (bin, java, js, ico) — in exe = _MEIPASS."""
    if frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """Schrijfbare map naast de .exe (of projectroot in dev)."""
    if frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    d = app_dir() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def icon_path() -> Path:
    return resource_root() / "ico.ico"


def bin_dir() -> Path:
    return resource_root() / "bin"


def java_dir() -> Path:
    return resource_root() / "java"


def js_dir() -> Path:
    return resource_root() / "js"
