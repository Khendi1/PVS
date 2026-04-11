# API Reference

The Video Synth exposes a REST API for full remote control of all parameters, patch management, and video output. The API is built on [FastAPI](https://fastapi.tiangolo.com/) and serves interactive Swagger docs at `/docs`.

## Starting the API

```bash
# Accessible from localhost only
python -m video_synth --api

# Expose to all devices on the local network
python -m video_synth --api --api-host 0.0.0.0

# Custom port
python -m video_synth --api --api-host 0.0.0.0 --api-port 9000

# Headless (no desktop window) — ideal for servers and Docker
python -m video_synth --headless --api --api-host 0.0.0.0 --no-virtualcam
```

---

## Endpoint Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check, returns service info |
| `GET` | `/params` | List all parameters with current values, bounds, and metadata |
| `GET` | `/params/{name}` | Get a single parameter by name |
| `PUT` | `/params/{name}` | Set a parameter value |
| `POST` | `/params/reset/{name}` | Reset a parameter to its default value |
| `GET` | `/snapshot` | Current frame as a JPEG image |
| `GET` | `/stream` | Continuous MJPEG stream (~30 fps) |
| `WS` | `/ws/stream` | WebSocket binary JPEG frame stream |
| `GET` | `/ui` | Redirect to the React web control panel |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Parameter Object Schema

Every parameter returned by `/params` or `/params/{name}` has this shape:

```json
{
  "name": "glitch_intensity_max",
  "value": 50,
  "min": 0,
  "max": 100,
  "default": 50,
  "group": "Groups.SRC_1_EFFECTS",
  "subgroup": "Glitch_General",
  "type": "Widget.SLIDER"
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique parameter key used in all API calls |
| `value` | number | Current value |
| `min` | number | Minimum allowed value |
| `max` | number | Maximum allowed value |
| `default` | number | Factory default value |
| `group` | string | Top-level grouping (SRC_1, SRC_2, POST, MIXER) |
| `subgroup` | string | Effect or animation class name |
| `type` | string | Widget type: SLIDER, DROPDOWN, TOGGLE |

---

## Examples

### List All Parameters

=== "curl"

    ```bash
    curl http://127.0.0.1:8000/params
    ```

=== "Python"

    ```python
    import requests

    params = requests.get('http://127.0.0.1:8000/params').json()
    print(f"Total parameters: {len(params)}")
    for p in params[:5]:
        print(f"  {p['name']}: {p['value']} ({p['min']}–{p['max']})")
    ```

### Get a Single Parameter

=== "curl"

    ```bash
    curl http://127.0.0.1:8000/params/glitch_intensity_max
    ```

=== "Python"

    ```python
    import requests

    p = requests.get('http://127.0.0.1:8000/params/glitch_intensity_max').json()
    print(p['value'])
    ```

### Set a Parameter

=== "curl"

    ```bash
    curl -X PUT http://127.0.0.1:8000/params/glitch_intensity_max \
      -H "Content-Type: application/json" \
      -d '{"value": 75}'
    ```

=== "Python"

    ```python
    import requests

    requests.put(
        'http://127.0.0.1:8000/params/glitch_intensity_max',
        json={'value': 75}
    )
    ```

### Reset a Parameter

=== "curl"

    ```bash
    curl -X POST http://127.0.0.1:8000/params/reset/glitch_intensity_max
    ```

=== "Python"

    ```python
    import requests

    requests.post('http://127.0.0.1:8000/params/reset/glitch_intensity_max')
    ```

### Capture a Snapshot

=== "curl"

    ```bash
    curl http://127.0.0.1:8000/snapshot -o frame.jpg
    ```

=== "Python"

    ```python
    import requests
    from PIL import Image
    import io

    response = requests.get('http://127.0.0.1:8000/snapshot')
    image = Image.open(io.BytesIO(response.content))
    image.save('frame.jpg')
    ```

---

## WebSocket Streaming

The `/ws/stream` endpoint pushes raw JPEG binary frames at the render rate (~30 fps). This is how the web UI's live preview panel receives video.

```python
import asyncio
import websockets

async def stream_frames():
    uri = "ws://127.0.0.1:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        while True:
            jpeg_bytes = await ws.recv()
            # jpeg_bytes is a raw JPEG — decode with PIL, OpenCV, etc.
            print(f"Received frame: {len(jpeg_bytes)} bytes")

asyncio.run(stream_frames())
```

For display in a browser, the web UI uses:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');
ws.binaryType = 'blob';
ws.onmessage = (event) => {
    const url = URL.createObjectURL(event.data);
    videoElement.src = url;
};
```

---

## Python Agent Example

A reusable client class for scripting parameter sequences:

```python
import requests
import time
from PIL import Image
import io

API_BASE = 'http://127.0.0.1:8000'

class VideoSynthClient:
    def __init__(self, base_url=API_BASE):
        self.base_url = base_url

    def get_params(self) -> list[dict]:
        """Return all parameters."""
        return requests.get(f'{self.base_url}/params').json()

    def get_param(self, name: str) -> dict:
        """Return a single parameter."""
        return requests.get(f'{self.base_url}/params/{name}').json()

    def set_param(self, name: str, value: float) -> dict:
        """Set a parameter value."""
        return requests.put(
            f'{self.base_url}/params/{name}',
            json={'value': value}
        ).json()

    def reset_param(self, name: str):
        """Reset a parameter to its default."""
        requests.post(f'{self.base_url}/params/reset/{name}')

    def snapshot(self) -> Image.Image:
        """Capture the current frame as a PIL Image."""
        data = requests.get(f'{self.base_url}/snapshot').content
        return Image.open(io.BytesIO(data))

    def animate(self, name: str, start: float, end: float,
                duration: float = 5.0, steps: int = 100):
        """Smoothly interpolate a parameter from start to end over duration seconds."""
        for i in range(steps):
            t = i / (steps - 1)
            self.set_param(name, start + (end - start) * t)
            time.sleep(duration / steps)


# Example usage
synth = VideoSynthClient()

# Ramp up glitch intensity over 10 seconds
synth.animate('glitch_intensity_max', 0, 100, duration=10.0)

# Set multiple parameters at once
for name, value in [('pattern_alpha', 0.8), ('pattern_speed', 2.5)]:
    synth.set_param(name, value)

# Save a snapshot
synth.snapshot().save('snapshot.jpg')
```

---

## AI Agent Integration

The REST API is designed for LLM tool use. The Docker AI agent uses function-calling to drive the synthesizer from natural-language prompts. Here is a minimal example using the OpenAI SDK against a local Ollama endpoint:

```python
from openai import OpenAI
import requests

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # required by the SDK but unused by Ollama
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "set_param",
            "description": "Set a video synth parameter value",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Parameter name"},
                    "value": {"type": "number", "description": "New value"}
                },
                "required": ["name", "value"]
            }
        }
    }
]

response = client.chat.completions.create(
    model='llama3.2:3b',
    messages=[
        {"role": "system", "content": "You control a video synthesizer via REST API."},
        {"role": "user", "content": "Make the glitch more intense"}
    ],
    tools=tools
)

# Execute the tool call
for call in response.choices[0].message.tool_calls or []:
    args = call.function.arguments
    requests.put(
        f'http://localhost:8000/params/{args["name"]}',
        json={'value': args['value']}
    )
```

For the full production agent implementation, see `agent/agent.py` and the [Docker & Agent](docker.md) documentation.

---

## OBS Integration

### UDP Stream (Recommended)

```bash
# Start Video Synth with UDP output
python -m video_synth --api --ffmpeg \
  --ffmpeg-output udp://127.0.0.1:1234 \
  --ffmpeg-preset veryfast
```

In OBS: Add **Media Source** → uncheck "Local File" → input `udp://127.0.0.1:1234`.

### SRT (Low Latency with Error Recovery)

```bash
python -m video_synth --api --ffmpeg \
  --ffmpeg-output "srt://127.0.0.1:9999?pkt_size=1316" \
  --ffmpeg-preset veryfast
```

In OBS: Media Source → input `srt://127.0.0.1:9999`.

### Virtual Camera (Zero Latency)

Virtual camera output is enabled by default. In OBS: Add **Video Capture Device** and select the virtual camera from the device list.

---

## Performance Notes

- API calls are thread-safe and non-blocking — parameter changes take effect on the next rendered frame.
- Snapshot responses may lag by up to one frame interval depending on render rate.
- For real-time streaming via FFmpeg, use `ultrafast` or `veryfast` preset.
- The WebSocket stream (`/ws/stream`) has lower overhead than polling `/snapshot` repeatedly.
