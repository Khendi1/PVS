# Getting Started

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Python 3.12 recommended for full Numba support |
| **Node.js 18+** | Required once to build the React web UI |
| **Git** | To clone the repository |
| **Webcam** | Optional — animations work without any video input |
| **MIDI controller** | Optional — any class-compliant USB MIDI device works |
| **FFmpeg** | Optional — required only for video recording/streaming output |
| **Docker & Docker Compose** | Optional — for the containerized stack with AI agent |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/video_synth.git
cd video_synth
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the Web UI

The React control panel is served by the Python API server. Build it once:

```bash
cd web
npm install
npm run build
cd ..
```

The compiled output lands in `web/dist/` and is automatically served at `/ui/` by the API.

---

## Running Modes

### Desktop GUI (default)

Launches the PyQt6 control panel window plus a video output window:

```bash
python -m video_synth
```

Add `--api` to also start the REST API and web UI while the desktop GUI is open:

```bash
python -m video_synth --api --api-host 0.0.0.0
```

### Headless API Server

No Qt window — pure API and web UI, suitable for servers and Docker:

```bash
python -m video_synth --headless --api --api-host 0.0.0.0 --no-virtualcam
```

Open `http://localhost:8000/ui/` to control the synth from any browser on the network.

### Headless with FFmpeg Recording

```bash
python -m video_synth --headless --api --ffmpeg --ffmpeg-output recording.mp4
```

### Full Docker Stack

Spins up the video synth, Ollama inference server, and AI agent chat UI:

```bash
docker compose up --build
```

First run downloads the `llama3.2:3b` model (~2 GB). Subsequent starts use the cached weights in the `ollama_data` volume.

See [Docker & Agent](docker.md) for full details.

---

## Command-Line Reference

```
python -m video_synth [options]

Core:
  -l, --log-level       Logging level: DEBUG, INFO, WARNING, ERROR
  -nd, --devices        Number of USB capture devices to scan (default: 5)
  -pn, --patch          Load saved patch by index (default: 0)
  -f, --file            Alternate save file path
  -c, --control-layout  GUI layout: SPLIT, QUAD_PREVIEW, QUAD_FULL
  -o, --output-mode     External window: NONE, WINDOW, FULLSCREEN
  -d, --diagnose        Log performance every N frames

API:
  --api                 Enable REST API server
  --api-host HOST       Bind host (default: 127.0.0.1; use 0.0.0.0 for LAN)
  --api-port PORT       Port (default: 8000)
  --headless            Run without GUI (requires --api or --ffmpeg)

FFmpeg:
  --ffmpeg              Enable FFmpeg video output
  --ffmpeg-output PATH  File path or stream URL (udp://, srt://, rtmp://)
  --ffmpeg-preset PRE   Encoding preset: ultrafast..veryslow (default: medium)
  --ffmpeg-crf CRF      Quality 0–51, lower = better (default: 23)

Hardware:
  --no-virtualcam       Disable virtual camera (use in Docker/CI)
  --obs                 Enable OBS WebSocket connection
  --obs-host, --obs-port, --obs-password
  --osc                 Enable OSC server
  --osc-host HOST       OSC bind host (default: 0.0.0.0)
  --osc-port PORT       OSC port (default: 9000)
```

---

## First Steps: Web UI Walkthrough

Once the API is running, open `http://localhost:8000/ui/` in a browser.

### Layout Overview

- **Left panel** — Source 1 effect chain controls (animations, effects, LFOs)
- **Center** — Live video preview and mixer controls (blend mode, mix amount)
- **Right panel** — Source 2 effect chain controls
- **Bottom bar** — Post-mix effects, patch browser, performance indicators

### Selecting an Animation

1. In the Source 1 panel, find the **Animation** dropdown at the top.
2. Select any animation (e.g., **Plasma**, **Metaballs**, **Strange Attractor**).
3. Parameters for the selected animation appear immediately below.

### Adjusting Parameters

- Drag any slider left/right to change its value.
- Double-click a slider to type an exact value.
- Click the **LFO** button next to a slider to open the modulation dialog.
- Click the **AUD** button to link a parameter to an audio frequency band.

### Loading a Patch

1. Click the **Patches** tab in the bottom bar.
2. Click any saved patch name to load it instantly — all parameters update smoothly.
3. Use **Save** to store the current state as a new patch.

### Using LFOs

LFOs (Low Frequency Oscillators) animate any parameter automatically:

1. Click the **LFO** button next to a parameter slider.
2. Choose a **waveform**: Sine, Square, Triangle, Sawtooth, or Perlin Noise.
3. Set **Rate** (Hz), **Min**, and **Max** to define the modulation range.
4. The LFO runs continuously and can itself be modulated by another LFO (nested modulation).

---

## Connecting MIDI

1. Plug in any class-compliant USB MIDI controller before launching.
2. Start the synth: `python -m video_synth`
3. In the GUI, right-click any slider and choose **MIDI Learn**.
4. Move the knob or fader on your controller — the mapping is saved automatically to `save/midi_mappings.yaml`.

MIDI mappings persist across sessions. To share your mappings with another machine, copy `save/midi_mappings.yaml`.

---

## Connecting OSC

Start the OSC server:

```bash
python -m video_synth --osc --osc-host 0.0.0.0 --osc-port 9000
```

Send OSC messages with the address pattern `/params/<param_name>` and a float value:

```
/params/plasma_speed  2.5
/params/glitch_intensity_max  75
```

Compatible with TouchOSC, Max/MSP, SuperCollider, and any OSC-capable application.

---

## Saving and Loading Patches

Patches are stored in `save/saved_values.yaml` as YAML entries. To generate patches programmatically:

```bash
python write_patches.py
```

On exit, the current state is written to `save/autosave.yaml` and reloaded automatically on the next start.

---

## Performance Tips

- **Render scale** — reduce `render_scale` if frame rate drops; frames are upscaled to display size.
- **Diagnose mode** — run with `--diagnose 100` to log timing per pipeline stage every 100 frames.
- **Effect count** — disable effects you are not using; each active effect adds processing time.
- **Animation choice** — Reaction Diffusion and Physarum are the most CPU-intensive animations; Plasma and Moire are the lightest.
