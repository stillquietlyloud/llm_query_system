#!/usr/bin/env bash
# =============================================================================
# LLM Query System — Unix/macOS Build Script
# =============================================================================
#
# Prerequisites
#   * Python 3.9+ on PATH
#   * pip available
#
# Usage
#   chmod +x build.sh && ./build.sh
#
# Output
#   dist/LLMQuerySystem   — standalone executable (Linux / macOS)
#
# =============================================================================

set -e

echo ""
echo "============================================================"
echo " LLM Query System — Building executable"
echo "============================================================"
echo ""

# ── 1. Install / upgrade dependencies ────────────────────────────────────────
echo "[1/3] Installing dependencies..."
pip install --upgrade requests pyinstaller
echo ""

# ── 2. Run PyInstaller ───────────────────────────────────────────────────────
echo "[2/3] Building executable with PyInstaller..."
pyinstaller \
    --onefile \
    --windowed \
    --name "LLMQuerySystem" \
    --add-data "endpoint_config.ini:." \
    --add-data "example_input.txt:." \
    --add-data "example_input.md:." \
    llm_query_system.py

echo ""

# ── 3. Report ─────────────────────────────────────────────────────────────────
echo "[3/3] Build complete!"
echo ""
echo "  Executable : dist/LLMQuerySystem"
echo ""
echo "  The executable is self-contained — distribute it freely."
echo "  Place your endpoint_config.ini and input files alongside it,"
echo "  or browse to them in the GUI."
echo ""
