# Python Video Synthesizer

A real-time video synthesizer for creating live visual effects and generative animations. Control video processing pipelines with MIDI controllers or GUI, featuring modular effects, LFO modulation, and procedural animations.

**No expensive hardware required** - works with just a laptop webcam, though it integrates seamlessly with capture cards, MIDI controllers, and external displays.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Audio Reactive](#audio-reactive)
- [API & Remote Control](#api--remote-control)
- [FFmpeg & OBS Integration](#ffmpeg--obs-integration)
- [Performance Profiling](#performance-profiling)
- [Architecture Overview](#architecture-overview)
- [Hardware Integration](#hardware-integration)

---

## Quick Start

### Requirements
- **Python 3.12** (recommended) or 3.13
- Webcam (optional - works with animations only)
- MIDI controller (optional)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd video_synth

# Create virtual environment (Python 3.12 recommended for Numba support)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scriptsctivate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the program
python ./video_synth
```

### Command Line Options

```bash
python ./video_synth --h

Options:
  -l, --log-level       Set logging level (DEBUG, INFO, WARNING, ERROR)
  -nd, --devices        Number of USB capture devices to search for (default: 5)
  -pn, --patch          Load saved patch by index (default: 0)
  -f, --file            Use alternate save file
  -c, --control-layout  GUI layout: SPLIT, QUAD_PREVIEW, QUAD_FULL
  -o, --output-mode     External window: NONE, WINDOW, FULLSCREEN
  -d, --diagnose        Enable performance logging every N frames

  --api                 Enable REST API server for remote control
  --api-host HOST       API server host (default: 127.0.0.1)
  --api-port PORT       API server port (default: 8000)
  --ffmpeg              Enable FFmpeg output to file or stream
  --ffmpeg-output PATH  Output path or stream URL: udp://, srt://, rtmp:// (default: output.mp4)
  --ffmpeg-preset PRE   Encoding preset (ultrafast..veryslow, default: medium)
  --ffmpeg-crf CRF      Quality 0-51, lower=better (default: 23)
  --no-virtualcam       Disable virtual camera output (enabled by default)
  --headless            Run without GUI (requires --api or --ffmpeg)
```

**Examples:**
```bash
# Standard usage with performance monitoring
python video_synth --diagnose 100 --output-mode FULLSCREEN

# API-controlled with FFmpeg recording
python video_synth --api --ffmpeg --ffmpeg-output recording.mp4

# Headless server mode, streaming over UDP (near-zero latency)
python video_synth --headless --api --ffmpeg \
  --ffmpeg-output udp://127.0.0.1:1234 --ffmpeg-preset veryfast
```

---

## Features

### Video Processing Pipeline
- **Dual-Source Mixer**
  - Alpha blending
  - Luma keying (key on bright or dark areas)
  - Chroma keying (green screen)
  - Mix camera feeds, video files, images, or animations

- **Effect Sequencer**
  - Drag-and-drop effect ordering
  - Three independent effect chains (Source 1, Source 2, Post-Mix)
  - Enable/disable effects dynamically

- **LFO Modulation System**
  - Link oscillators to any parameter
  - 5 waveforms: Sine, Square, Triangle, Sawtooth, Perlin Noise
  - Nested LFOs (modulate LFO parameters with other LFOs)
  - Per-parameter cutoff ranges

- **Audio Reactive Modulation**
  - Link any parameter to audio frequency bands
  - 5 FFT bands: Bass, Low Mid, Mid, High Mid, Treble
  - Per-band sensitivity, attack/decay envelope, cutoff range
  - Microphone or line-in input via sounddevice

### Effects Categories

**Color**
- HSV manipulation (hue rotation, saturation, value)
- Brightness/contrast adjustment
- Solarization and posterization
- Color polarization

**Pixels**
- Noise addition (Gaussian, salt & pepper)
- Blur and sharpen
- Edge detection

**Transform**
- Pan, tilt, zoom (PTZ)
- Reflection (horizontal, vertical, kaleidoscope)
- Polar coordinate transform
- Warp effects (sine, radial, fractal, perlin, feedback warp)
- Feedback warp: self-referential displacement maps for chaotic patterns

**Temporal**
- Feedback blending (alpha)
- Frame averaging and temporal filtering
- Luma feedback
- Pattern feedback modulation (self-warping pattern accumulation)

**Glitch**
- Scanline displacement, color channel splitting
- Block corruption, random rectangles
- Slitscan (directional, reversible, wobble, blending modes)
- Echo/stutter (probability-based frame freeze)
- Sync modulation emulation

### Procedural Animations

All animations are fully parametric with real-time adjustment:

- **Metaballs** - Organic blob simulation with configurable physics
- **Moire Patterns** - 7 pattern types (line, radial, grid, spiral, diamond, checkerboard, hexagonal)
- **Reaction Diffusion** - Gray-Scott chemical simulation
- **Plasma** - Classic demoscene plasma effect
- **Strange Attractors** - Chaotic systems (Lorenz, Clifford, De Jong, Aizawa, Thomas)
- **Physarum** - Slime mold simulation
- **Shaders** - 11 procedural shaders (fractal, plasma, galaxy, etc.)
- **DLA** - Diffusion-limited aggregation
- **Chladni** - Vibration pattern simulation
- **Voronoi** - Cellular tessellation

### Patch System
- Save/recall parameter configurations
- YAML-based patch storage
- Multiple patch slots per save file
- GUI patch browser

---

## Audio Reactive

Drive any parameter from live audio input. The audio reactive system analyzes microphone or line-in audio via FFT and maps frequency band energy to parameter values in real time.

### How It Works
1. Audio is captured via `sounddevice` (PortAudio) in a background thread
2. Each frame, the audio buffer is analyzed with a windowed FFT
3. Magnitude spectrum is split into 5 frequency bands:
   - **Bass** (20-250 Hz) - kick drums, bass guitar
   - **Low Mid** (250-500 Hz) - warmth, body
   - **Mid** (500-2000 Hz) - vocals, snare
   - **High Mid** (2000-6000 Hz) - presence, cymbals
   - **Treble** (6000-20000 Hz) - air, brightness
4. Band energy is smoothed with per-band attack/decay envelope followers
5. Smoothed energy is mapped to the linked parameter's range

### Linking Parameters
Each slider parameter has an **AUD** button (orange when active) next to the existing LFO button. Click it to open the audio link dialog:

- **Band** - Which frequency band to follow
- **Sensitivity** (0.0-5.0) - Gain multiplier on the band energy
- **Attack** (0.0-1.0) - How fast the value rises with the audio
- **Decay** (0.0-1.0) - How fast the value falls when audio drops
- **Cutoff Min/Max** - Clamp the output range (same as LFO cutoffs)

A parameter can have both an LFO and an audio band linked simultaneously.

---

## API & Remote Control

Enable the REST API with `--api` to control the synthesizer programmatically. The API runs on a background thread and provides full parameter access.

### Starting the API
```bash
python ./video_synth --api                          # Default: 127.0.0.1:8000
python ./video_synth --api --api-host 0.0.0.0       # Expose to network
python ./video_synth --api --api-port 9000           # Custom port
```

Interactive docs available at `http://127.0.0.1:8000/docs` (Swagger UI).

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/params` | List all parameters with current values |
| GET | `/params/{name}` | Get a single parameter |
| PUT | `/params/{name}` | Set parameter value (`{"value": 42}`) |
| POST | `/params/reset/{name}` | Reset parameter to default |
| GET | `/snapshot` | Current frame as JPEG image |

### Python Example
```python
import requests

API = 'http://127.0.0.1:8000'

# Get all parameters
params = requests.get(f'{API}/params').json()

# Set a parameter
requests.put(f'{API}/params/hue', json={'value': 90})

# Save a snapshot
img = requests.get(f'{API}/snapshot')
with open('frame.jpg', 'wb') as f:
    f.write(img.content)
```

### Headless Mode
Run without any GUI for server/automation use cases:
```bash
python ./video_synth --headless --api --ffmpeg --ffmpeg-output output.mp4
```
Headless mode requires at least `--api` or `--ffmpeg` to be useful.

---

## FFmpeg & OBS Integration

### Recording to File
```bash
python ./video_synth --ffmpeg --ffmpeg-output recording.mp4
python ./video_synth --ffmpeg --ffmpeg-output recording.mp4 --ffmpeg-preset slow --ffmpeg-crf 18
```

### Live Streaming

**UDP (Recommended - lowest latency, no server needed):**
```bash
python ./video_synth --api --ffmpeg \
  --ffmpeg-output udp://127.0.0.1:1234 \
  --ffmpeg-preset veryfast
```

**SRT (low latency with error recovery):**
```bash
python ./video_synth --api --ffmpeg \
  --ffmpeg-output "srt://127.0.0.1:1234?pkt_size=1316" \
  --ffmpeg-preset veryfast
```

**RTMP (legacy, higher latency, requires RTMP server):**
```bash
docker run -d -p 1935:1935 --name rtmp-server tiangolo/nginx-rtmp
python ./video_synth --api --ffmpeg \
  --ffmpeg-output rtmp://localhost/live/stream \
  --ffmpeg-preset veryfast
```

### OBS Integration
The synthesizer can feed into OBS Studio via virtual camera, UDP/SRT stream, or be controlled alongside OBS via WebSocket.

**Method 1: Virtual Camera (enabled by default - zero latency)**

The virtual camera starts automatically. No extra flags needed.
1. In OBS, add a **Video Capture Device** source
2. Select the virtual camera from the device dropdown
3. No encoding/decoding - raw pixel transfer with zero latency

**Method 2: UDP Media Source**
1. Start the synth with `--ffmpeg --ffmpeg-output udp://127.0.0.1:1234`
2. In OBS, add a Media Source:
   - Uncheck "Local File"
   - Input: `udp://127.0.0.1:1234`
   - Check "Restart playback when source becomes active"

**Method 3: OBS WebSocket Control**
Use `obs_controller.py` for programmatic OBS control (recording, streaming, scene switching):
```python
from obs_controller import OBSController

obs = OBSController(password="your_password")
obs.connect()
obs.start_recording()
# ... run sequences via the API ...
obs.stop_recording()
obs.disconnect()
```

See [examples/](examples/) for complete automation scripts.

---

## Performance Profiling

Enable diagnostics to monitor frame timing and identify bottlenecks:

```bash
python ./video_synth --diagnose 100    # Log every 100 frames
```

### Tracked Stages

| Stage | Description |
|-------|-------------|
| `capture` | Camera/source frame acquisition |
| `lfo` | LFO oscillator updates |
| `audio` | Audio reactive analysis + parameter updates |
| `effects_1` / `effects_2` | Per-source effect chain processing |
| `mix` | Source blending (alpha, luma key, chroma key) |
| `post_fx` | Post-mix effect chain |
| `api_copy` | Frame copy for API snapshot endpoint |
| `ffmpeg` | FFmpeg frame encoding |
| `gui_emit` | GUI frame signal emission |

### Effects Breakdown
Within each effect chain, individual effect timings are tracked. The diagnostic log highlights the top 3 slowest effects per chain. Effects exceeding 20ms per frame are logged as warnings in real time.

---

## Architecture Overview

```
video_synth/
  __main__.py          # Entry point, CLI args, video loop, thread management
  settings.py          # Global settings from CLI args
  common.py            # Shared enums (Widget, Groups, Toggle, etc.)
  param.py             # ParamTable / Param system (central parameter management)
  lfo.py               # LFO oscillator bank
  audio_reactive.py    # Audio input analysis + parameter modulation
  mixer.py             # Dual-source mixer (alpha, luma key, chroma key)
  effects_manager.py   # Effect chain sequencing + performance tracking
  patterns3.py         # Procedural pattern generation + pattern feedback
  luma.py              # Luma keying utilities
  pyqt_gui.py          # PyQt6 GUI (tabs, sliders, LFO/audio dialogs)
  api.py               # FastAPI REST server
  ffmpeg_output.py     # FFmpeg subprocess pipe (file + UDP/SRT/RTMP streaming)
  obs_controller.py    # OBS WebSocket controller

  effects/             # Modular effect classes
    color.py, pixels.py, warp.py, feedback.py,
    glitch.py, shapes.py, reflector.py, ptz.py, sync.py

  animations/          # Procedural animation generators
    metaballs.py, moire.py, reaction_diffusion.py, plasma.py,
    attractors.py, physarum.py, shaders.py, dla.py, chladni.py, voronoi.py
```

### Data Flow
```
Camera/Source --> EffectManager (per-source effect chain)
                     |
              Mixer (blend sources)
                     |
              EffectManager (post-mix effects)
                     |
           +----+----+----+
           |    |         |
          GUI  FFmpeg    API
```

LFOs and Audio Reactive modules update parameter values each frame before effects are applied.

---

## Hardware Integration

### MIDI Controllers
Any MIDI controller with knobs/sliders works. MIDI CC messages are mapped to parameters via the GUI. Connect your controller before launching the application.

### Capture Devices
Use `-nd` to set how many USB devices to scan. Works with:
- USB webcams
- HDMI/SDI capture cards (e.g., Elgato, Blackmagic)
- Virtual cameras

### External Display
```bash
python ./video_synth --output-mode WINDOW       # Separate output window
python ./video_synth --output-mode FULLSCREEN    # Fullscreen on secondary display
```
