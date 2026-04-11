#!/usr/bin/env bash
# ============================================================================
# VideoSynth -- Linux / macOS build script
# ============================================================================
# Prerequisites:
#   - Python 3.11 installed (python3 on PATH)
#   - Node.js 18+ installed (for web UI build)
#   - Activated virtual environment (recommended):
#       python3 -m venv .venv && source .venv/bin/activate
#
# Usage: bash build.sh
# Output: dist/VideoSynth/VideoSynth  (or VideoSynth.app on macOS)
# ============================================================================

set -euo pipefail

echo ""
echo "============================================================"
echo " VideoSynth -- Linux/macOS Build"
echo "============================================================"
echo ""

# ---- 1. Install / upgrade PyInstaller ----
echo "[1/4] Installing PyInstaller..."
pip install pyinstaller --upgrade --quiet

# ---- 2. Install Python dependencies ----
echo "[2/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# ---- 3. Build the React web UI ----
echo "[3/4] Building web UI..."
if [ -f "web/dist/index.html" ]; then
    echo "      web/dist already exists -- skipping web build."
elif [ -f "web/package.json" ]; then
    pushd web > /dev/null
    echo "      Running npm install..."
    npm install --silent || { echo "WARNING: npm install failed. Web UI will not be available."; }
    echo "      Running npm run build..."
    npm run build || { echo "WARNING: npm build failed. Web UI will not be available."; }
    popd > /dev/null
else
    echo "      web/package.json not found -- skipping web build."
fi

# ---- 4. Run PyInstaller ----
echo "[4/4] Running PyInstaller..."
pyinstaller build/video_synth.spec --clean --noconfirm

echo ""
echo "============================================================"
echo " Build complete!"
echo " Output : dist/VideoSynth/"
echo " Run    : dist/VideoSynth/VideoSynth"
echo "============================================================"
echo ""
