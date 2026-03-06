@echo off
:: ============================================================================
:: LLM Query System — Windows EXE Build Script
:: ============================================================================
::
:: Prerequisites
::   * Python 3.9+ installed and on PATH
::   * pip available
::
:: Usage
::   Double-click build.bat  OR  run it from a Command Prompt / PowerShell
::
:: Output
::   dist\LLMQuerySystem.exe   — standalone single-file executable
::
:: ============================================================================

setlocal

echo.
echo ============================================================
echo  LLM Query System — Building Windows EXE
echo ============================================================
echo.

:: ── 1. Install / upgrade dependencies ──────────────────────────────────────
echo [1/3] Installing dependencies...
pip install --upgrade requests pyinstaller
if ERRORLEVEL 1 (
    echo ERROR: pip install failed. Make sure Python and pip are available.
    pause
    exit /b 1
)
echo.

:: ── 2. Run PyInstaller ──────────────────────────────────────────────────────
echo [2/3] Building executable with PyInstaller...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "LLMQuerySystem" ^
    --icon NONE ^
    --add-data "endpoint_config.ini;." ^
    --add-data "example_input.txt;." ^
    --add-data "example_input.md;." ^
    llm_query_system.py

if ERRORLEVEL 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)
echo.

:: ── 3. Report ───────────────────────────────────────────────────────────────
echo [3/3] Build complete!
echo.
echo   Executable : dist\LLMQuerySystem.exe
echo   Size       : (see above)
echo.
echo   The exe is fully self-contained — copy it anywhere and run it.
echo   Place your endpoint_config.ini and input .txt/.md files alongside
echo   the exe, or browse to them from the GUI.
echo.

pause
endlocal
