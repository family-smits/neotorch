# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile — NeoTorch Studio + native photoreal tools."""

import glob
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "ico.ico"), "."),
    (str(ROOT / "bin" / "meshpack.exe"), "bin"),
    (str(ROOT / "bin" / "pathtracer.exe"), "bin"),
    (str(ROOT / "bin" / "denoise.exe"), "bin"),
]

for cls in (ROOT / "java").glob("*.class"):
    datas.append((str(cls), "java"))
for src in (ROOT / "java").glob("*.java"):
    datas.append((str(src), "java"))

for js_file in (ROOT / "js").glob("*"):
    if js_file.is_file():
        datas.append((str(js_file), "js"))

hiddenimports = collect_submodules("neotorch")
hiddenimports += [
    "numpy",
    "pygame",
    "PIL",
    "PIL.Image",
]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="NeoTorch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "ico.ico"),
)
