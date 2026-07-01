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

---

## III. AI & Automation

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

### 📋 16. Macro Knobs `[M]` — Phase 2

User-defined "macro" controls that drive multiple parameters with a single slider, each with its own scaling curve. E.g., a "Chaos" macro that simultaneously increases glitch intensity, warp amplitude, and LFO rates. Macros are defined in the patch YAML and exposed as top-level controls in the web UI.

### 📋 18. Patch Browser with Thumbnails `[M]` — Phase 1

A visual grid of patch thumbnails (pre-rendered JPEG snapshots) with search and tagging. Clicking a patch previews it before committing. The snapshot is captured via `/snapshot` right after each patch is saved.

### 📋 19. Mobile-Optimized Performer UI `[M]` — Phase 2

A simplified touch-friendly layout for phone/tablet — big sliders, swipe to change patches, shake-to-randomize. Performers often want to walk around with their phone while the visuals run on desktop. React responsive mobile breakpoint with a reduced, performance-focused control set.

---

## VI. Distribution & Packaging

### 📋 21. Electron Wrapper `[L]` — Phase 4

Wrap the whole stack (Python subprocess + web UI) in an Electron shell for a proper cross-platform desktop app with native menu, system tray, auto-update, and file dialogs. Python backend runs as a child process; Electron manages its lifecycle. The "ship it as a product" path.

### 📋 22. Installer & Auto-Updater `[M]` — Phase 4

Windows: NSIS or WiX installer generated by GitHub Actions. Mac: `.dmg` with notarization. Bundles Python runtime — users install nothing. Auto-updater (Squirrel/Sparkle) delivers new animations silently.

---

## VII. Developer Experience

### 📋 25. Patch Sharing Hub `[L]` — Phase 3

A lightweight web service where users upload `.yaml` patch files, tag them (ambient, glitch, psychedelic, etc.), and browse/download others' work. Could be as simple as a GitHub Discussions thread with a bot that validates and indexes uploaded YAML, or a proper FastAPI + SQLite backend. Patches from the community become the social content layer.

### 📋 26. Hot-Reload Dev Mode `[M]` — Phase 2

Watch `video_synth/animations/` and `video_synth/effects/` for file changes and reload the affected module without restarting. Python `importlib.reload()` can do this; the tricky part is re-instantiating only the changed class in the running EffectManager while preserving param state. Massive DX improvement when iterating on a new shader.

---

## VIII. Backlog / Fast Wins

| Status | Item | Size | Phase | Description |
| --- | --- | --- | --- | --- |
| ✅ | Patch interpolation API | `[S]` | 1 | `POST /patch/morph?target=&duration=` — `SaveController.morph_to()` lerps numeric params to a patch over N seconds in a daemon thread (`save.py`, `api.py`) |
| ✅ | MIDI mapping export/import | `[S]` | 1 | `GET /midi/export` + `POST /midi/import` + MidiTab Export/Import buttons; `MidiMapper.reload_mappings()`/`import_mappings()`/`export_mappings_yaml()` (`midi_mapper.py`, `api.py`, `web/`) |
| ✅ | Param history / undo | `[M]` | 2 | `ParamHistory` ring buffer + `/undo` `/redo` `/history`; Ctrl+Z / Ctrl+Shift+Z in web UI (`param_history.py`, `api.py`, `web/src/App.jsx`) |
| 📋 | Preset randomizer with constraints | `[S]` | 1 | `POST /patch/random` with optional group exclusion mask |
| 📋 | WebRTC stream output | `[M]` | 2 | Browser-native peer-to-peer stream instead of MJPEG |
| 📋 | OSC learn mode | `[M]` | 2 | Like MIDI learn but for OSC addresses |
| 📋 | Pixel shader import | `[M]` | 2 | Drop a `.glsl` into `shaders/` and it auto-appears as animation source |
