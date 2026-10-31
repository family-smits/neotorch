@echo off
setlocal
cd /d "%~dp0"
set OUTDIR=%~dp0..\..\bin
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set VCVARS=
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if defined VCVARS (
  call "%VCVARS%" >nul
  cl /nologo /O2 /EHsc /std:c++17 /Fe:"%OUTDIR%\pathtracer.exe" tracer.cpp /link /SUBSYSTEM:CONSOLE
  if errorlevel 1 exit /b 1
  del /q tracer.obj 2>nul
  echo Built: %OUTDIR%\pathtracer.exe
  exit /b 0
)

where clang++ >nul 2>&1
if %errorlevel%==0 (
  clang++ -O2 -std=c++17 -o "%OUTDIR%\pathtracer.exe" tracer.cpp
  if errorlevel 1 exit /b 1
  echo Built: %OUTDIR%\pathtracer.exe
  exit /b 0
)

echo ERROR: geen C++ compiler gevonden (MSVC of clang++)
exit /b 1
