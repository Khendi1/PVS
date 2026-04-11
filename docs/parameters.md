# Parameter Reference

This page documents every controllable parameter in Video Synth — animations, effects, LFOs, audio reactivity, and the mixer. Each parameter has a unique string key used in API calls, MIDI/OSC mappings, and patch YAML files.

## How Parameters Work

All parameters are registered at startup through a central `ParamTable`. Every parameter exposes:

- **Key** (`name`) — the string identifier used in `PUT /params/{name}` and patch YAML
- **Value** — current runtime value; changes take effect on the next frame
- **Min / Max** — enforced by the API; the web UI uses these for slider range
- **Default** — value loaded when the synthesizer starts fresh or when you call `POST /params/reset/{name}`
- **Group / Subgroup** — determines which panel the slider appears in within the web UI

Parameters can be modulated by:

- **LFOs** — continuous oscillation via `PUT /lfo/{param_name}`
- **Audio bands** — FFT-driven modulation linked per parameter
- **API / scripts** — direct value writes at any time; changes are applied on the next render frame
- **MIDI CC** — learned via right-click → MIDI Learn in the GUI; persisted in `save/midi_mappings.yaml`
- **OSC** — address `/params/{param_name}` with a float value

## Widget Types

| Type | Description |
|---|---|
| `Widget.SLIDER` | Numeric range slider; min/max enforced |
| `Widget.DROPDOWN` | Enum selector (e.g., animation type, warp mode) |
| `Widget.TOGGLE` | Boolean enable/disable (0 = off, 1 = on) |

---

## Full Parameter Table

The complete parameter reference is maintained in [`documentation/PARAMETERS.md`](../documentation/PARAMETERS.md) alongside the source code. It is organized by module and includes the parameter key, min, max, default value, and a plain-English description for every controllable value in the system.

### Quick Navigation

- [Plasma](../documentation/PARAMETERS.md#plasma)
- [Reaction Diffusion](../documentation/PARAMETERS.md#reaction-diffusion)
- [Drift Field](../documentation/PARAMETERS.md#drift-field)
- [Shaders](../documentation/PARAMETERS.md#shaders-s1)
- [Voronoi](../documentation/PARAMETERS.md#voronoi)
- [Metaballs](../documentation/PARAMETERS.md#metaballs)
- [Moire](../documentation/PARAMETERS.md#moire)
- [Chladni](../documentation/PARAMETERS.md#chladni)
- [DLA](../documentation/PARAMETERS.md#dla-diffusion-limited-aggregation)
- [Physarum](../documentation/PARAMETERS.md#physarum)
- [Strange Attractor](../documentation/PARAMETERS.md#strange-attractor)
- [Color Effects](../documentation/PARAMETERS.md#color-effects)
- [Warp](../documentation/PARAMETERS.md#warp)
- [Feedback](../documentation/PARAMETERS.md#feedback)
- [Glitch](../documentation/PARAMETERS.md#glitch)
- [Shapes](../documentation/PARAMETERS.md#shapes)
- [Pixels](../documentation/PARAMETERS.md#pixels)
- [PTZ (Pan/Tilt/Zoom)](../documentation/PARAMETERS.md#ptz-pantiltzoom)
- [Reflector](../documentation/PARAMETERS.md#reflector)
- [Mixer](../documentation/PARAMETERS.md#mixer)
- [LFO](../documentation/PARAMETERS.md#lfo-low-frequency-oscillator)
- [Audio Reactive](../documentation/PARAMETERS.md#audio-reactive)
- [Beat Detector](../documentation/PARAMETERS.md#beat-detector)

---

## Using Parameters in Scripts

### Read all parameters and filter by group

```python
import requests

params = requests.get('http://localhost:8000/params').json()

# Find all glitch parameters
glitch = [p for p in params if 'glitch' in p['name']]
for p in glitch:
    print(f"{p['name']:40s} = {p['value']:6.1f}  [{p['min']}–{p['max']}]")
```

### Reset a whole subgroup to defaults

```python
import requests

params = requests.get('http://localhost:8000/params').json()

# Reset all warp parameters
warp_params = [p for p in params if p['subgroup'] == 'Warp']
for p in warp_params:
    requests.post(f"http://localhost:8000/params/reset/{p['name']}")
```

### Build a patch dictionary from current values

```python
import requests
import yaml

params = requests.get('http://localhost:8000/params').json()
patch = {p['name']: p['value'] for p in params}

with open('my_patch.yaml', 'w') as f:
    yaml.dump({'entries': [patch]}, f)
```

---

## LFO Constants

When setting LFO shape via the API or patch YAML, use these integer values:

| Constant | Value | Shape |
|---|---|---|
| NONE | 0 | No modulation (LFO disabled) |
| SINE | 1 | Smooth sine wave |
| SQUARE | 2 | Hard square wave |
| TRIANGLE | 3 | Linear triangle wave |
| SAWTOOTH | 4 | Rising sawtooth |
| PERLIN | 5 | Smooth random (Perlin noise) |

Example — attach a slow sine LFO to plasma speed via the API:

```bash
curl -X PUT http://localhost:8000/lfo/plasma_speed \
  -H "Content-Type: application/json" \
  -d '{"shape": 1, "rate": 0.1, "min": 0.5, "max": 3.0}'
```
