# Video Synth — Development Roadmap

> Collaborative real-time visual art synthesizer for live performance.
> Items are grouped by domain and ordered by impact vs. effort within each group.
> Scope tags: `[S]` = days · `[M]` = 1–3 weeks · `[L]` = month+

---

## Status Key

| Symbol | Meaning |
| --- | --- |
| ✅ | Complete — shipped and working |
| 🔄 | In Progress — active development |
| 📋 | Planned — not yet started |

---

## Phase Ordering (TL;DR)

| Phase | Theme | Horizon |
| --- | --- | --- |
| **1** | Polish & Ship | 1–2 months |
| **2** | Performance Power | 2–3 months |
| **3** | Collaborative & AI | 3–6 months |
| **4** | Product | 6+ months |

---

## I. Core Engine

### 🔄 1. GPU-Native Rendering via GLSL `[L]` — Phase 2

The biggest architectural unlock. All animations currently run on CPU NumPy, which caps throughput with a single animation active. Moving field-based generators (Reaction Diffusion, Plasma, DLA, Physarum) to GLSL fragment shaders via the existing **ModernGL** dependency would give 4–10× speedup and enable 1080p/4K output.

Pattern: each animation optionally provides a `.glsl` file; the base class detects it and routes through the GPU path. CPU NumPy fallback stays for compatibility. New shaders dropped into `shaders/` auto-appear as animation sources.

*Status: `GLSLAnimation` base class added to `animations/base.py` — provides standalone ModernGL context, FBO, fullscreen quad VAO, and `_compile_program` / `_render` helpers. Graceful CPU fallback when GPU unavailable. Individual animation ports (Plasma, etc.) are the remaining work.*

### 📋 2. Modular Effect Graph (Node-Based Pipeline) `[L]` — Phase 3

Currently the effect chain is linear: `src_1 → effects → mix → post`. A node graph would allow routing src_1 through feedback into src_2, or applying a warp only to the luma channel.

Internally: a directed acyclic graph of `EffectNode` objects. Externally: a visual canvas in the web UI (React Flow). This is the feature that would most differentiate the tool from existing VJing software.

### ✅ 3. Resolution & Output Profiles `[S]` — Phase 1

Add `--resolution WxH` and `--fps N` CLI flags. Currently hardcoded at 640×480 in `common.py`. Quick win — the internals already pass `width`/`height` everywhere.

*Status: `--resolution WxH` and `--fps N` added to `parse_args()` in `__main__.py`. Resolution overrides `common.WIDTH`/`common.HEIGHT` before `main()` runs.*

### ✅ 4. MIDI Clock & BPM Sync `[M]` — Phase 2

A global BPM clock that LFOs can quantize to (1/4, 1/8, 1/16 note). The `mido` library already handles MIDI clock messages. When a DAW or drum machine sends MIDI clock, every LFO set to "tempo sync" snaps to musical time. Essential for live AV performance — visual events hitting on the beat.

*Status: Shipped — `src/video_synth/bpm_clock.py` (`BPMClock`). Listens for MIDI clock (24 ticks/beat) on the first available MIDI port; estimates BPM from rolling inter-tick intervals; supports MIDI Start/Stop/Song Position messages. LFO gains `tempo_sync` (FREE/SYNC) and `note_division` (1/1 through 1/32) params — when SYNC, the oscillator uses `bpm_clock.frequency_for_division()` instead of its own Hz param. Tap-tempo via `POST /bpm/tap`. REST API: `GET /bpm`, `PUT /bpm {"bpm": 140}`, `POST /bpm/tap`. BPMClock auto-connects at startup; propagated to all OscBank LFOs in `main()`.*

---

## II. Live Performance & Collaboration

### 📋 5. Multi-User Web UI with Roles `[M]` — Phase 3

Extend the web UI for collaborative performance:

- **Performer** — full parameter control
- **Audience** — stream view only
- **Guest** — assigned parameter group only

WebSocket-based state sync (already have `/ws/stream`) broadcasts parameter changes to all connected clients in real time. Each tab could show live cursors from other performers. Think "Google Docs for the visuals."

### 📋 6. Performance Mode / Setlist `[M]` — Phase 2

A sequenced list of patches with configurable crossfade times and trigger conditions (manual, timer, beat-sync, audio threshold). The performer sees a setlist view in the web UI — hit Next to glide to the next patch over N seconds. Internally: a background thread lerps all param values between current and target patch state.

### ✅ 7. Eurorack / CV-Gate Input `[M]` — Phase 4

A `cv_controller.py` using a USB audio interface's line input to read control voltages (0–5V mapped to param ranges). Direct modular synth → video synth integration with zero latency. The dream hardware pairing for the intersection of visual art and electronic music.

*Status: Shipped — `src/video_synth/cv_controller.py` (`CVController`). Reads audio-rate CV from any sounddevice input; each channel independently maps to a parameter with configurable volt range, IIR smoothing, and YAML-persisted mappings (`save/cv_mappings.yaml`). CLI: `--cv`, `--cv-device`, `--cv-channels`. REST API: `GET /cv/devices`, `GET /cv/mappings`, `GET /cv/values`, `POST /cv/map`, `DELETE /cv/map/{channel}`. Learn mode: `start_learn(channel, qualified_key)`. Graceful no-op when sounddevice is unavailable.*

---

## III. AI & Automation

### ✅ 8. AI Agent / Ollama Integration `[M]` — Phase 3

The Ollama agent (`agent/agent.py`) is fully operational. It reads `PARAMETERS.md` at startup to understand every parameter's purpose, uses tool calls to drive the REST API, and exposes a web chat UI at `http://localhost:8001/`. Deploy with `docker compose up --build`. Model: `llama3.2:3b` by default; configurable via `OLLAMA_MODEL` env var. Optional vision support via `VISION_MODEL`.

### 📋 9. Generative Patch Composer `[M]` — Phase 3

Extend the existing Ollama agent to generate new patch YAML entries from a mood description. The agent already has all the API tools — add a `write_patch` tool that calls `write_patches.py` logic and appends to the save file. The model can explore the parameter space creatively in ways a human wouldn't think to try.

### 📋 10. Audio-Driven Auto-Pilot `[M]` — Phase 3

A background agent that listens to audio analysis bands and automatically morphs parameters in musically meaningful ways: bass hits → glitch intensity spike, energy builds → zoom in, breakdown → hue shift cycle.

Different from the existing audio-reactive module in that it does macro-level composition decisions, not just 1:1 band→param mapping.

### 📋 11. Computer Vision Input Sources `[L]` — Phase 3

Add MediaPipe (pose estimation, hand tracking, face mesh) as a parameter source. Hands raised = alpha up; body leaning = warp direction. Turns the performer's body into a zero-hardware control interface.

Also: optical flow from webcam → warp direction; depth camera (Intel RealSense) → parallax animations.

---

## IV. New Animations & Effects

### 📋 12. GLSL Shader Bank Expansion `[S each]` — Phase 1 / ongoing

Each new shader is a weekend project. Strong candidates for live visual performance:

- **Tunnel zoom** with twist
- **Kaleidoscope** (N-fold symmetry; `reflector.py` already exists as an effect)
- **Truchet tiles** / aperiodic tiling
- **Flame fractals** (IFS-based)
- **Fluid simulation** (Jos Stam stable fluids, GPU-friendly)
- **Conway's Game of Life** with colored generations
- **Lissajous 3D** curves projected to 2D

### 🔄 13. Text & Typography Engine `[M]` — Phase 2

Render animated text as an animation source — scrolling lyrics, live-coded messages from the web UI, random poetry generators. Pillow supports custom fonts at quality. Parameters: font, size, scroll speed, color, blend mode. For live performance: a "ticker" text input in the web UI that the performer types into and it flows onto the output.

*Status: Core implementation shipped — `animations/text_engine.py` (`TextEngine` class). Registered as `TEXT_ENGINE = 19` in `AnimSource` enum; available in both src_1 and src_2. Parameters: `text_scroll_speed`, `text_scroll_axis` (H/V), `text_font_size`, `text_r/g/b`, `text_brightness`, `text_pulse_speed/depth`, `text_bg_alpha`, `text_letter_spacing`, `text_line_spacing`. Auto-cycles 4 built-in messages when no custom text is set. REST endpoints: `GET /text?source=1`, `PUT /text` `{"message": "...", "source": 1}`. Remaining: web UI ticker input component.*

### 📋 14. 3D Scene Renderer `[L]` — Phase 4

Use ModernGL (already installed) to render simple 3D geometry as an animation source: rotating wireframes, 3D particle systems, 3D Lissajous curves, point clouds. The existing ModernGL context setup extends naturally.

---

## V. Web UI & UX

### ✅ 15. XY Pad Control Surface `[S]` — Phase 1

A 2D touch/mouse pad that controls two parameters simultaneously — X axis and Y axis independently assignable. Essential for live performance: expressive two-dimensional gestures in a single interaction. Per-session assignment stored in local state.

*Status: Shipped — `web/src/components/XYPad.jsx`. Mouse and touch drag supported; X/Y params independently assignable from dropdowns; `-webkit-user-select: none` for Safari/iOS touch. Mounted as a tab in `App.jsx`.*

### 📋 16. Macro Knobs `[M]` — Phase 2

User-defined "macro" controls that drive multiple parameters with a single slider, each with its own scaling curve. E.g., a "Chaos" macro that simultaneously increases glitch intensity, warp amplitude, and LFO rates. Macros are defined in the patch YAML and exposed as top-level controls in the web UI.

### ✅ 17. Visual Waveform / Spectrum Display `[S]` — Phase 1

Show audio bands and beat detection state in the web UI header — a small canvas with FFT bars and a beat flash. The data is already computed by `AudioReactiveModule`; just needs an API endpoint and a React component. Immediate feedback for performers calibrating audio reactivity.

*Status: Shipped — `web/src/components/SpectrumDisplay.jsx` polls `GET /audio/bands` and renders a 5-bar canvas with beat flash in the app header. Backend: `GET /audio/bands` → `{"bands": [5 floats 0–1], "beat": bool}`; `audio_module` wired into `APIServer`.*

### 📋 18. Patch Browser with Thumbnails `[M]` — Phase 1

A visual grid of patch thumbnails (pre-rendered JPEG snapshots) with search and tagging. Clicking a patch previews it before committing. The snapshot is captured via `/snapshot` right after each patch is saved.

### 📋 19. Mobile-Optimized Performer UI `[M]` — Phase 2

A simplified touch-friendly layout for phone/tablet — big sliders, swipe to change patches, shake-to-randomize. Performers often want to walk around with their phone while the visuals run on desktop. React responsive mobile breakpoint with a reduced, performance-focused control set.

---

## VI. Distribution & Packaging

### ✅ 20. Windows Executable (PyInstaller) `[M]` — Phase 1

Package into a single `.exe` with PyInstaller, bundling Python, all deps, the pre-built React UI, and the shaders directory. Target UX: download, double-click, browser opens. Key challenges: ModernGL DLL bundling, PyQt6 plugin dirs, `web/dist/` static files. GitHub Actions builds on every tag.

*Status: Shipped — `build/video_synth.spec` + `.github/workflows/build.yml`. Full CI pipeline: installs deps, builds React UI, runs PyInstaller, zips output, uploads artifact, creates GitHub Release on version tags. Permissions fixed (`contents: write`). Triggered on `v*.*.*` tags or manual dispatch.*

### 📋 21. Electron Wrapper `[L]` — Phase 4

Wrap the whole stack (Python subprocess + web UI) in an Electron shell for a proper cross-platform desktop app with native menu, system tray, auto-update, and file dialogs. Python backend runs as a child process; Electron manages its lifecycle. The "ship it as a product" path.

### 📋 22. Installer & Auto-Updater `[M]` — Phase 4

Windows: NSIS or WiX installer generated by GitHub Actions. Mac: `.dmg` with notarization. Bundles Python runtime — users install nothing. Auto-updater (Squirrel/Sparkle) delivers new animations silently.

---

## VII. Developer Experience

### ✅ 23. Docker Stack `[M]` — Phase 1

Three-service Docker Compose stack (`docker-compose.yml`) with `video_synth`, `ollama`, and `agent` containers. Headless API mode with Mesa software GL and Xvfb virtual display. Model weights persisted in `ollama_data` volume. Single-command start: `docker compose up --build`.

### ✅ 24. Documentation Site `[M]` — Phase 1

A **MkDocs Material** site covering API reference, parameter tables, getting started guide, Docker setup, and roadmap. Deployed to GitHub Pages via GitHub Actions on every push to main. Interactive Swagger API explorer available at `/docs` on the running server.

*Status: Complete — MkDocs Material site with full nav (`index`, `getting-started`, `api`, `parameters-reference`, `docker`, `build`, `roadmap`). GitHub Actions deploys to GitHub Pages on every push to main (`docs.yml`). API reference updated with all endpoints including `/audio/bands`, `/params/bulk`, LFO, MIDI, and patch routes. `.markdownlint.json` added to suppress false positives from MkDocs tabbed block syntax.*

### 📋 25. Patch Sharing Hub `[L]` — Phase 3

A lightweight web service where users upload `.yaml` patch files, tag them (ambient, glitch, psychedelic, etc.), and browse/download others' work. Could be as simple as a GitHub Discussions thread with a bot that validates and indexes uploaded YAML, or a proper FastAPI + SQLite backend. Patches from the community become the social content layer.

### 📋 26. Hot-Reload Dev Mode `[M]` — Phase 2

Watch `video_synth/animations/` and `video_synth/effects/` for file changes and reload the affected module without restarting. Python `importlib.reload()` can do this; the tricky part is re-instantiating only the changed class in the running EffectManager while preserving param state. Massive DX improvement when iterating on a new shader.

### ✅ 27. Test Suite & CI `[M]` — Phase 1

A pytest suite covering core subsystems, with GitHub Actions running on every push to main and every PR.

- **Smoke tests** — each animation and effect instantiates and produces a frame without crashing
- **Param bounds** — every param's min/max is honored; value clamping, reset, randomize verified
- **API contract** — REST endpoints return expected schema
- **LFO coverage** — all waveform shapes, link/unlink, OscBank, consecutive advancement

*Status: Shipped — `tests/test_params.py`, `test_lfo.py`, `test_effects.py`, `test_animations.py`, `test_api.py` with `conftest.py`. CI in `.github/workflows/test.yml` runs on push/PR to main. Mesa software GL + offscreen Qt platform configured for headless runner.*

### ✅ 28. CLAUDE.md & Project Context `[S]` — Phase 1

Architecture overview, key conventions, file map, dev workflow, and Docker stack notes documented in `CLAUDE.md` for AI-assisted development. Skills and hooks configured for the Claude Code harness.

---

## VIII. Backlog / Fast Wins

| Status | Item | Size | Phase | Description |
| --- | --- | --- | --- | --- |
| ✅ | `--resolution` flag | `[S]` | 1 | `--resolution WxH` and `--fps N` CLI args added |
| ✅ | Param search in web UI | `[S]` | 1 | Filter input on the param list shipped in web UI |
| ✅ | `/params/bulk` PUT endpoint | `[S]` | 1 | Set multiple params in one API call (`PUT /params/bulk`) |
| ✅ | LFO tap-tempo | `[S]` | 1 | `POST /bpm/tap` + BPMClock; LFO tempo-sync to note divisions |
| ✅ | XY pad control | `[S]` | 1 | `XYPad.jsx` — see V.15 |
| ✅ | Audio spectrum display | `[S]` | 1 | `SpectrumDisplay.jsx` + `/audio/bands` — see V.17 |
| ✅ | Video source playback controls | `[S]` | 1 | Pause + scrub for VIDEO file sources: `video_pause_src1/2` toggle + `video_scrub_src1/2` (0–100%) params in `mixer.py`; seek-on-change + paused freeze-frame in `_process_single_source`. Web `ParamToggle` switch renders TOGGLE params |
| 📋 | Patch interpolation API | `[S]` | 1 | `POST /patch/morph?target=3&duration=5` — lerp to patch over N seconds |
| 📋 | MIDI mapping export/import | `[S]` | 1 | Share `midi_mappings.yaml` between machines via UI |
| 📋 | Param history / undo | `[M]` | 2 | Ring buffer of last N param states, undo on Ctrl+Z |
| 📋 | Preset randomizer with constraints | `[S]` | 1 | `POST /patch/random` with optional group exclusion mask |
| 📋 | WebRTC stream output | `[M]` | 2 | Browser-native peer-to-peer stream instead of MJPEG |
| 📋 | OSC learn mode | `[M]` | 2 | Like MIDI learn but for OSC addresses |
| 📋 | Pixel shader import | `[M]` | 2 | Drop a `.glsl` into `shaders/` and it auto-appears as animation source |
