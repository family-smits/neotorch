@echo off
setlocal
cd /d "%~dp0denoise"
set OUTDIR=%~dp0..\..\bin
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set "CARGO_TARGET_DIR=%cd%\target"
cargo build --release
if errorlevel 1 exit /b 1
if not exist "target\release\denoise.exe" (
  echo ERROR: denoise.exe not found in target\release
  exit /b 1
)
copy /Y "target\release\denoise.exe" "%OUTDIR%\denoise.exe" >nul
echo Built: %OUTDIR%\denoise.exe
exit /b 0
