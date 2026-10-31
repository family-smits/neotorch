@echo off
setlocal
cd /d "%~dp0"
set OUTDIR=%~dp0..\..\bin
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set VCVARS=
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if defined VCVARS (
  call "%VCVARS%" >nul
  cl /nologo /O2 /Fe:"%OUTDIR%\meshpack.exe" meshpack.c /link /SUBSYSTEM:CONSOLE
  if errorlevel 1 exit /b 1
  del /q meshpack.obj 2>nul
  echo Built: %OUTDIR%\meshpack.exe
  exit /b 0
)

where clang >nul 2>&1
if %errorlevel%==0 (
  clang -O2 -o "%OUTDIR%\meshpack.exe" meshpack.c
  if errorlevel 1 exit /b 1
  echo Built: %OUTDIR%\meshpack.exe
  exit /b 0
)

echo ERROR: no C compiler
exit /b 1
