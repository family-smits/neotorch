#!/usr/bin/env python3
"""NeoTorch Studio — single entrypoint for PyInstaller onefile exe."""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Zorg dat subprocessen dezelfde map zien
    os.environ.setdefault("NEOTORCH_APP", "1")
    from gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
