# Video Synth

**Real-time collaborative visual art synthesizer for live performance.**

Video Synth is a dual-source video mixer with a GPU-accelerated effect chain, LFO modulation, audio reactivity, MIDI/OSC control, and a REST API with a React web UI. Designed for VJs, live coders, and audiovisual artists who want expressive, programmable visuals without expensive hardware.

---

## Feature Highlights

### Procedural Animations

Ten fully parametric animation engines render at real-time frame rates:

| Animation | Description |
|---|---|
| **Metaballs** | Organic blob simulation with configurable physics and color |
| **Plasma** | Classic demoscene plasma via layered sine functions |
| **Reaction Diffusion** | Gray-Scott chemical simulation producing organic textures |
| **Strange Attractors** | Lorenz, Clifford, De Jong, Aizawa, Thomas chaotic systems |
| **Physarum** | Slime mold trail simulation |
| **Moire Patterns** | 7 pattern types: line, radial, grid, spiral, diamond, checkerboard, hexagonal |
| **DLA** | Diffusion-limited aggregation crystal growth |
| **Chladni** | Vibration pattern simulation |
| **Voronoi** | Cellular tessellation with animated seeds |
| **Shaders** | 11 procedural GLSL-style shaders (fractal, galaxy, plasma, and more) |

### Effect Categories

Effects are organized into three independent chains (Source 1, Source 2, Post-Mix) and can be reordered and toggled live:

- **Color** — HSV manipulation, brightness/contrast, solarization, posterization, color polarization
- **Pixels** — Gaussian/salt-and-pepper noise, blur, sharpen, edge detection
- **Transform** — Pan/tilt/zoom, reflection, polar coordinates, warp (sine, radial, fractal, Perlin, feedback)
- **Temporal** — Feedback blending, frame averaging, luma feedback, pattern feedback
- **Glitch** — Scanline displacement, color channel splitting, block corruption, slitscan, echo/stutter, sync modulation

### Control Methods

- **Web UI** — React-based control panel at `http://localhost:8000/ui/` — sliders, dropdowns, LFO editor, patch browser
- **REST API** — Full parameter control over HTTP; integrates with Python scripts, LLM agents, and automation tools
- **MIDI** — Generic MIDI learn: click any slider, twist a knob, mapping saved automatically
- **OSC** — Address pattern maps directly to parameter names; works with TouchOSC, Max/MSP, SuperCollider
- **LFO Modulation** — 5 waveforms (Sine, Square, Triangle, Sawtooth, Perlin Noise) linkable to any parameter
- **Audio Reactivity** — FFT bands (Bass, Low Mid, Mid, High Mid, Treble) drive parameters in real time

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-org/video_synth.git
cd video_synth

# Create and activate a virtual environment (Python 3.11+ required)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the web UI (one-time step)
cd web && npm install && npm run build && cd ..

# Launch — desktop GUI + web UI + API
python -m video_synth --api --api-host 0.0.0.0
```

Then open [http://localhost:8000/ui/](http://localhost:8000/ui/) in your browser.

For the full Docker stack (video synth + Ollama AI agent):

```bash
docker compose up --build
```

See [Getting Started](getting-started.md) for step-by-step instructions and all launch modes.

---

## Service URLs

When running with `--api` (or via Docker):

| URL | What it is |
|---|---|
| `http://localhost:8000/ui/` | React web control panel |
| `http://localhost:8000/docs` | Interactive API docs (Swagger UI) |
| `http://localhost:8000/stream` | MJPEG live video stream |
| `http://localhost:8000/snapshot` | Current frame as JPEG |
| `ws://localhost:8000/ws/stream` | WebSocket binary frame stream |
| `http://localhost:8001/` | AI agent chat UI (Docker only) |

---

## What's New

**Docker + AI Agent Stack** — The full stack now runs in three Docker containers with a single command:

- `video_synth` — headless API mode with Mesa software GL and Xvfb virtual display
- `ollama` — local LLM inference server (no API key required, fully offline)
- `agent` — FastAPI web chat UI that controls the synthesizer via natural language

The agent reads `docs/parameters-reference.md` at startup to understand every parameter's purpose and range, then uses tool calls to drive the REST API in response to natural-language prompts like "make it more chaotic" or "create a slow ambient blue wash."

See [Docker & Agent](docker.md) for setup details, model selection, and customizing the agent's behavior.
