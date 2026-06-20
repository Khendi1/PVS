# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (nothing yet)

## [0.1.0]

Initial release of the Python Video Synthesizer.

### Added

- **Dual-source video mixer** with alpha blending, luma keying, and chroma
  keying across camera feeds, video files, images, and procedural animations.
- **GPU-accelerated effect chains** — three independent chains (Source 1,
  Source 2, Post-Mix) covering color, pixels, transform/warp, temporal/feedback,
  and glitch effects, with drag-and-drop ordering.
- **Procedural animations** — Metaballs, Moiré, Reaction Diffusion, Plasma,
  Strange Attractors, Physarum, Shaders, DLA, Chladni, Voronoi, Lenia,
  Harmonic Interference, Fractal Zoom, Oscillator Grid, and Drift Field.
- **LFO modulation system** — link oscillators (sine, square, triangle,
  sawtooth, Perlin) to any parameter, including nested LFOs.
- **Audio-reactive modulation** — map five FFT frequency bands to any parameter
  with per-band sensitivity and attack/decay envelopes.
- **MIDI control** — generic MIDI learn/mapping system with YAML persistence.
- **OSC control** — UDP-based Open Sound Control server for TouchOSC, Max/MSP,
  SuperCollider, and similar.
- **OBS integration** — virtual camera output and OBS WebSocket control.
- **REST API** — FastAPI server exposing all parameters, LFO control, patch
  management, MJPEG `/stream`, and `/snapshot` endpoints, with Swagger docs.
- **React web UI** — browser-based control surface served at `/ui/`.
- **AI agent** — optional Ollama-powered local LLM agent for natural-language
  control (separate service).
- **Patch system** — YAML-based save/recall of parameter configurations.
- **FFmpeg output** — file recording and low-latency UDP/SRT/RTMP streaming.
- **Docker stack** — Compose orchestration of the synth, Ollama, and agent
  services for headless deployment.
- **PyInstaller packaging** — standalone executable build via `build/`.

[Unreleased]: https://github.com/Khendi1/PVS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Khendi1/PVS/releases/tag/v0.1.0
