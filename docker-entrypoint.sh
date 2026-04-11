#!/bin/bash
set -e

# Start a virtual framebuffer so Qt6 libs don't complain at import time,
# even though we're running in headless mode (no actual QApplication created).
Xvfb :99 -screen 0 1280x1024x24 +extension GLX -ac &
XVFB_PID=$!

cleanup() {
    echo "Shutting down..."
    kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Brief pause for Xvfb to be ready
sleep 1

echo "Starting video synthesizer in headless API mode..."
# PYTHONPATH set in Dockerfile: /app/src (package) + /app/src/video_synth (bare imports)
exec python -m video_synth \
    --headless \
    --api \
    --api-host 0.0.0.0 \
    --api-port 8000 \
    --no-virtualcam
