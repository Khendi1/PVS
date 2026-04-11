# PVS Code Review

**Date:** 2026-03-27
**Scope:** Full audit of `video_synth/` — bugs, stubs, performance, architecture, and future direction
**Reviewer:** Claude (Sonnet 4.6)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Bug & Issue Registry](#bug--issue-registry)
3. [Performance Concerns](#performance-concerns)
4. [Architectural Debt](#architectural-debt)
5. [Roadmap](#roadmap)
6. [Future User & Functionality Improvements](#future-user--functionality-improvements)

---

## Architecture Overview

PVS is a GPU-accelerated real-time video synthesizer combining live capture, procedural animation, effects processing, and multi-output rendering. It is structured as a pipeline:

```
[Sources: Device / Animation / File]
        ↓  (ThreadPoolExecutor, 50ms timeout)
[EffectManager: per-source effects chain]
        ↓
[Mixer: alpha blend SRC_1 + SRC_2]
        ↓
[EffectManager: post-mix effects]
        ↓
[Outputs: GUI / VirtualCam / FFmpeg / OBS / WebSocket / OSC]
```

**Key subsystems:**

| Component | File | Role |
|---|---|---|
| `Mixer` | mixer.py | Owns both source banks, blends frames, frame caching |
| `Animation` | animations/base.py + 20 impls | Abstract procedural content (ModernGL shaders, numpy) |
| `EffectManager` | effects_manager.py | Configurable effects chain per group, LFO orchestration |
| `Param / ParamTable` | param.py | Unified control layer — GUI, MIDI, OSC, API, LFO, audio |
| `AudioReactiveModule` | audio_reactive.py | sounddevice FFT → per-band energy → param modulation |
| `REST API` | api.py | FastAPI — GET/PUT /params, GET /snapshot, WebSocket stream |
| `LFO / OscBank` | lfo.py | Oscillator bank for parameter automation |

---

## Bug & Issue Registry

Severity: **CRITICAL** → breaks functionality / data loss · **HIGH** → wrong output or crash under normal use · **MEDIUM** → degraded behavior, correctness risk · **LOW** → cleanup, style, documentation

### CRITICAL

None identified.

---

### HIGH

#### H1 — `apply_perlin_noise()` is a stub (feedback.py:271)
Pixel-by-pixel nested loop implementation is present but functionally incomplete. At 1080p this would take seconds per frame.
**Fix:** Vectorize with `numpy` + a pre-generated perlin field (same pattern used in `warp.py`), or remove if superseded by the warp effect.

#### H2 — Error recovery frame can be `None` mid-chain (effects_manager.py:177)
`original_frame_for_recovery` is initialized to `None` and only assigned after the first effect processes successfully. If the first effect raises an exception, recovery falls back to `None` → likely crash or black frame propagation.
**Fix:** Assign `original_frame_for_recovery = frame` before entering the effect loop.

#### H3 — Per-pixel noise generation in `warp.py` (warp.py:141, 152)
`_generate_perlin_flow()` and `_generate_fractal_flow()` use nested Python loops calling `pnoise2()` per pixel. At 1080p (~2M calls per frame) this is unusable. Currently masked by the 50ms frame timeout falling back to cache.
**Fix:** Pre-generate noise fields using `numpy` vectorization and `noise.pnoise2` with array inputs, or cache and interpolate across frames (only regenerate every N frames).

#### H4 — Oscillator linking duplication in `patterns3.py` (patterns3.py:283–348)
`_set_osc_params()` has 65+ lines of near-identical repetitive blocks, one per oscillator parameter. Any change to oscillator linking logic requires updating every block manually.
**Fix:** Refactor into a data-driven loop over a list of `(param_name, osc_attr)` pairs.

---

### MEDIUM

#### M1 — Duplicate alpha-blend logic (feedback.py:232, mixer.py)
`apply_luma_feedback2()` in `feedback.py` duplicates alpha-blend logic already in `Mixer._alpha_blend()`. Bug fixes to one will not propagate.
**Fix:** Extract to a shared utility function in `common.py`.

#### M2 — Color parameter mapping documented as wrong (patterns3.py:101)
`# TODO: mapping seems off` — color parameters for patterns may produce incorrect visuals. Needs visual verification and correction.

#### M3 — LFO PERLIN shape value mapping incomplete (lfo.py:178)
`# TODO: handle perlin noise mapping using the map_value method` — the PERLIN oscillator shape does not apply the same output scaling as other shapes, producing inconsistent modulation depth.
**Fix:** Apply `map_value()` call consistently with other shape branches.

#### M4 — Feedback class excluded from sequencer via string filtering (effects_manager.py:103)
Feedback effect methods are removed from the effect sequence by filtering on method names (`"feedback"` substring check). This is fragile — a rename silently breaks the exclusion.
**Fix:** Add an explicit class-level flag (e.g. `SEQUENCEABLE = False`) and filter on that.

#### M5 — `scale_frame()` padding logic may crop valid data (feedback.py:124)
Frame is padded then immediately modified with slicing that may discard the padding. Edge case: if target dimensions are smaller than source, the crop may not be applied to the right region.
**Fix:** Add unit tests for upscale, downscale, and equal-size cases.

#### M6 — Off-by-one risk in `avg_frame_buffer()` (feedback.py:188)
The oldest-frame tracking in the running-sum buffer uses a manual index that is incremented before use in one path and after in another. Needs a review pass with explicit test for buffer-full wraparound.

#### M7 — `warp.py` time accumulator coupled to frame rate (warp.py:104)
`self.t += 0.1` is called once per `get_frame()` call. At 60 FPS this runs twice as fast as at 30 FPS, making all time-based animations frame-rate dependent.
**Fix:** Pass `dt` (elapsed seconds) into `get_frame()` and use `self.t += dt * speed_scale`.

#### M8 — `warp_use_fractal` uses float comparison instead of toggle (warp.py:430)
Parameter is a SLIDER being compared as a boolean. Any non-zero value enables fractal mode, but the widget type is misleading and inconsistent with other boolean params.
**Fix:** Change to `Widget.TOGGLE`.

#### M9 — Glitch state management scattered across instance variables (glitch.py:20–31)
Multiple frame-counter variables (`current_fixed_y_end`, `last_scroll_glitch_reset_frame`, etc.) are declared at the top level with no grouping or reset mechanism. Hard to track and test.
**Fix:** Group into a `@dataclass GlitchState` and expose a `reset()` method.

#### M10 — `EffectBase` singleton declared but not implemented (effects/base.py)
The class has `_instance = None` but no `__init__`, `get_instance()`, or `__new__` override. Either complete the pattern or remove it — currently it adds confusion with no benefit.

---

### LOW

#### L1 — Ambiguous `seed` parameter name in LFO (lfo.py:75)
The variable name is unclear. Rename to reflect its actual role (e.g. `phase_offset` or `noise_seed`).

#### L2 — `posc_bank` global is unused (patterns3.py:10)
Declared at module level, never referenced. Dead code — remove.

#### L3 — Magic number `PERLIN_SCALE = 1700` undocumented (warp.py:9)
No comment explaining what this controls or how it was derived. Add a brief explanation.

#### L4 — Typo in `Animation` base class error message (animations/base.py:19)
`"subgroupes"` → `"subclasses"`

#### L5 — LFO `OscBank.__init__` uses list comprehension for side effects (lfo.py:234)
`temp = [... for i in ...]` discards the result. Use a plain `for` loop.

#### L6 — Commented-out MIDI code in `__main__.py` (\_\_main\_\_.py:39)
Stale commented code with no removal plan. Either document why it's kept or delete.

#### L7 — Hardcoded 20ms slow-effect threshold (effects_manager.py:165)
Not a bug but limits flexibility. Could be a configurable setting.

#### L8 — Duplicate identity maps in warp (warp.py:107–112)
`_fb_base_x/y` and `_fb_map_x/y` are both stored when only one set is needed at a time. Small memory savings available.

#### L9 — Excess per-frame param recalculation in LFO (lfo.py:127)
`create_oscillator()` re-reads all param values on every `yield`. Values that don't change between frames could be cached with dirty-flag invalidation.

---

## Performance Concerns

| Area | File | Issue | Impact |
|---|---|---|---|
| Warp perlin/fractal flow | warp.py:141, 152 | Per-pixel Python loops | Catastrophic at HD — currently masked by frame cache timeout |
| Patterns perlin blobs | patterns3.py:589 | Multiple per-pixel `pnoise3()` loops | High CPU, limits usable resolution |
| Feedback perlin | feedback.py:271 | Stub — per-pixel loop skeleton | Unrunnable at any practical resolution |
| Animation startup | mixer.py | All 40 animation objects instantiated at init | ~50–200MB memory for unused animations |
| LFO oscillator | lfo.py:127 | Full param re-read every yield | Minor per-frame overhead |
| Effects chain | effects_manager.py | No dynamic culling — all effects run every frame | Minor; instrumentation already tracks slow effects |

**Recommended priority:** Fix H3 (warp perlin loops) first — it's the most impactful performance issue and likely the cause of any frame drops under warp effects.

---

## Architectural Debt

### AD1 — No lazy animation loading
`Mixer.__init__` instantiates every animation class for both SRC_1 and SRC_2 (currently ~40 objects including the new `Shaders3`). Many of these allocate GPU buffers or large numpy arrays on init.

**Impact:** ~50–200MB overhead at startup. ModernGL contexts per animation may also compete.
**Solution:** Implement an animation factory that instantiates on first selection and optionally caches.

### AD2 — Duplicated alpha-blend logic
Present in at least `Mixer._alpha_blend()` and `feedback.py:apply_luma_feedback2()`. Likely also pattern-matched elsewhere.
**Solution:** Single `alpha_blend(src, overlay, alpha)` function in `common.py`.

### AD3 — `EffectBase` non-pattern
The base class for effects provides no interface contract. Each effect defines its own method names and param registration without enforcement.
**Solution:** Define abstract methods (`apply()`, `register_params()`) in `EffectBase` and enforce via `ABCMeta`.

### AD4 — Frame-rate-dependent animation time
Multiple animations/effects use frame counters or fixed-increment `self.t` instead of wall-clock time. Behavior changes with FPS settings.
**Solution:** Pass `dt` through the rendering pipeline from `Mixer.get_frame()` downward.

### AD5 — Feedback excluded by string matching
See M4. The architecture places Feedback in `_all_services` then removes it, which is the wrong direction.

---

## Roadmap

Organized by effort and impact. Each phase is independently deliverable.

---

### Phase 1 — Critical Fixes (1–2 sessions)
*Zero user-visible regressions. Fix things that are broken or will break.*

| # | Task | File(s) | Issue |
|---|---|---|---|
| 1.1 | Fix error recovery frame assignment | effects_manager.py:177 | H2 |
| 1.2 | Fix `warp_use_fractal` to use `Widget.TOGGLE` | warp.py:430 | M8 |
| 1.3 | Fix `original_frame_for_recovery = frame` before loop | effects_manager.py | H2 |
| 1.4 | Fix LFO PERLIN shape — apply `map_value()` | lfo.py:178 | M3 |
| 1.5 | Fix `EffectBase` — add `ABCMeta` + abstract `apply()` | effects/base.py | AD3 |
| 1.6 | Remove or vectorize `apply_perlin_noise()` stub | feedback.py:271 | H1 |

---

### Phase 2 — Performance (2–4 sessions)
*Make warp and pattern effects usable at full resolution.*

| # | Task | File(s) | Issue |
|---|---|---|---|
| 2.1 | Vectorize `_generate_perlin_flow()` and `_generate_fractal_flow()` | warp.py:141, 152 | H3 |
| 2.2 | Vectorize `_generate_perlin_blobs()` in patterns3 | patterns3.py:589 | Perf |
| 2.3 | Add frame-rate-independent `dt` to rendering pipeline | mixer.py, warp.py, animations | M7, AD4 |
| 2.4 | Implement lazy animation factory | mixer.py | AD1 |

---

### Phase 3 — Code Quality (1–2 sessions)
*Reduce duplication, fix fragile patterns, clean dead code.*

| # | Task | File(s) | Issue |
|---|---|---|---|
| 3.1 | Extract `alpha_blend()` to `common.py` | feedback.py, mixer.py | M1, AD2 |
| 3.2 | Refactor `_set_osc_params()` into data-driven loop | patterns3.py:283 | H4 |
| 3.3 | Replace string-based Feedback exclusion with class flag | effects_manager.py:103 | M4 |
| 3.4 | Group glitch state into `GlitchState` dataclass | glitch.py:20 | M9 |
| 3.5 | Verify and fix color parameter mapping | patterns3.py:101 | M2 |
| 3.6 | Remove `posc_bank` dead code | patterns3.py:10 | L2 |
| 3.7 | Rename `seed` → `phase_offset` in LFO | lfo.py:75 | L1 |
| 3.8 | Fix typo in `Animation` docstring | animations/base.py:19 | L4 |
| 3.9 | Document `PERLIN_SCALE` magic number | warp.py:9 | L3 |
| 3.10 | Fix `OscBank` list comprehension side effect | lfo.py:234 | L5 |

---

### Phase 4 — Architecture (4+ sessions)
*Larger structural changes. May require refactor branches.*

| # | Task | File(s) | Notes |
|---|---|---|---|
| 4.1 | Implement `EffectBase` with `ABCMeta` and enforce interface across all effects | effects/ | Enables safe sequencer, better tooling |
| 4.2 | Animation lazy factory | mixer.py, animations/ | Cuts startup memory; enables hot-swap |
| 4.3 | Full `dt`-based time in all animations and effects | All | Enables frame-rate independence |
| 4.4 | Audit `avg_frame_buffer()` and `scale_frame()` with unit tests | feedback.py | Catch M5, M6 definitively |
| 4.5 | Effects chain validation / integration test suite | effects_manager.py | Covers TODO at line 140 |

---

## Future User & Functionality Improvements

### UX / Control

- **Preset system with named snapshots**: Save/load full parameter state (beyond current save patches). Support banks of presets browsable from the web UI and keyboard shortcut cycling.
- **Macro knobs**: Map a single MIDI knob to a named group of parameters with configurable scaling per-param. Useful for "one knob does the vibe" performance control.
- **Undo/redo for parameter changes**: Ring buffer of param states, reversible via keyboard or MIDI.
- **Parameter lock**: Prevent a param from being modified by audio-reactive, LFO, or API during performance.
- **MIDI scene recall**: Store full parameter snapshots to MIDI program change messages.

### Animations & Visuals

- **Shader animation authoring in the web UI**: Live-editable GLSL fragment shader input field with hot-reload. `Shaders3` already proves the ModernGL infrastructure works.
- **Video file looping with beat-sync**: Sync video file loop points to BPM detected from audio or manual tap-tempo.
- **Image/texture injection**: Load static images or sprite sheets as texture inputs to animations.
- **Animation crossfade**: Smooth interpolated transition between two animation types on the same source (blend parameters over N frames).
- **More `Shaders3` presets**: Voronoi flow, curl noise, domain-warped fractals, SDF shapes.

### Effects

- **Spectral blur / frequency-domain effects**: Apply effects in FFT space (blur specific frequencies, smear, etc.).
- **Chromatic aberration as a proper effect**: Currently ad-hoc in some animations; should be a first-class effect with configurable split distance and angle.
- **AI upscaling integration**: Optional Real-ESRGAN or EDSR pass on the output frame for higher-quality recording at lower render cost.
- **3D LUT support**: Apply color grading via .cube LUT files as a post-processing effect.

### Audio Reactivity

- **Onset / transient detection**: Separate from beat detection — trigger one-shot parameter jumps on percussive hits (snare, kick individually).
- **Pitch tracking**: Map fundamental frequency of audio input to a parameter (e.g. drive animation speed or hue with melody).
- **MIDI clock sync**: Sync LFO and beat detection to incoming MIDI clock for tight DAW integration.
- **Audio stem separation** (offline/background): Route kick, snare, melody stems to independent parameter groups.

### LLM / AI Control

- **Local LLM skill set (Qwen / Ollama)**: A directory of Python tool functions wrapping the REST API, with a JSON schema for Ollama tool-use mode and a system prompt describing the parameter space. Enables Qwen to autonomously explore and drive the engine. *(In progress — see separate skills work)*
- **Vision-in-the-loop**: Expose `GET /snapshot` to the LLM so it can evaluate what it sees and adjust parameters toward a described aesthetic.
- **Natural language preset authoring**: Describe a mood ("slow hypnotic blue waveforms") and have the LLM translate it to a set of parameter values.
- **Autonomous performance mode**: LLM-driven session that evolves visuals over a set duration, avoiding repetition.

### Infrastructure & Developer Experience

- **Configurable slow-effect threshold**: Currently hardcoded at 20ms (effects_manager.py:165). Expose via settings.
- **Web UI parameter search**: Filter parameter list by name in the web control panel.
- **Hot-reload for shader files**: Watch `shaders3.py` / GLSL source files and reload without restarting the engine.
- **Headless test mode**: CLI flag that runs one frame through the full pipeline and saves to disk — enables CI visual regression testing.
- **OSC namespace documentation**: Auto-generate OSC address list from the Param system (similar to how PARAMETERS.md is generated).
- **Plugin/extension architecture**: Allow external Python files in a `plugins/` directory to register new animation types or effects at startup without modifying core files.
