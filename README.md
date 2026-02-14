# Python Video Synthesizer

A real-time video synthesizer for creating live visual effects and generative animations. Control video processing pipelines with MIDI controllers or GUI, featuring modular effects, LFO modulation, and procedural animations.

**No expensive hardware required** - works with just a laptop webcam, though it integrates seamlessly with capture cards, MIDI controllers, and external displays.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Usage Guide](#usage-guide)
- [Developer Guide](#developer-guide)
- [Performance Profiling](#performance-profiling)
- [Hardware Integration](#hardware-integration)
- [Background](#background-and-inspiration)

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
python video_synth
```

### Command Line Options

```bash
python video_synth --help

Options:
  -l, --log-level     Set logging level (DEBUG, INFO, WARNING, ERROR)
  -nd, --devices      Number of USB capture devices to search for (default: 5)
  -pn, --patch        Load saved patch by index (default: 0)
  -f, --file          Use alternate save file
  -c, --control-layout  GUI layout: SPLIT, QUAD_PREVIEW, QUAD_FULL
  -o, --output-mode   External window: NONE, WINDOW, FULLSCREEN
  -d, --diagnose      Enable performance logging every N frames
```

**Example:**
```bash
python video_synth --diagnose 100 --output-mode FULLSCREEN
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
- Warp effects

**Temporal**
- Feedback blending
- Frame averaging
- Temporal filtering
- Luma feedback

**Glitch**
- Scanline displacement
- Color channel shifting
- Block corruption
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
