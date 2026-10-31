@echo off
setlocal
cd /d "%~dp0"

echo === NeoTorch one-exe build ===

if not exist "bin\meshpack.exe" call native\c\build.bat
if not exist "bin\pathtracer.exe" call native\pathtracer\build.bat
if not exist "bin\denoise.exe" call native\rust\build.bat

where javac >nul 2>&1
if %errorlevel%==0 (
  javac java\PhotorealViewer.java java\PhotoStudio.java
)

python -m pip install -q -r requirements.txt pyinstaller pillow
if errorlevel 1 exit /b 1

echo Building NeoTorch.exe ...
python -m PyInstaller --noconfirm --clean NeoTorch.spec
if errorlevel 1 exit /b 1

echo.
echo Klaar: dist\NeoTorch.exe
echo Icon: ico.ico
exit /b 0
