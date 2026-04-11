# Docker & AI Agent

The Docker stack runs the full Video Synth environment — including a local LLM — with a single command. No GPU required; the LLM runs on CPU via Ollama and the video synth uses Mesa software OpenGL rendering.

---

## Stack Architecture

Three services defined in `docker-compose.yml`:

| Service | Container | Port | Description |
|---|---|---|---|
| `ollama` | `video_synth_ollama` | 11434 | Ollama local LLM inference server |
| `video_synth` | `video_synth_app` | 8000 | Video synth in headless API mode |
| `agent` | `video_synth_agent` | 8001 | AI agent web chat UI |

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                │
│    http://localhost:8000/ui/    ←  Web control panel    │
│    http://localhost:8001/       ←  AI agent chat        │
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐
    │   video_synth:8000  │  │    agent:8001        │
    │   FastAPI + Mesa GL │  │   FastAPI + OpenAI   │
    │   Xvfb virtual disp │  │   SDK → Ollama       │
    └─────────────────────┘  └──────────┬───────────┘
                                        │
                             ┌──────────▼──────────┐
                             │   ollama:11434       │
                             │   llama3.2:3b (CPU)  │
                             └─────────────────────┘
```

The `agent` service waits for both `ollama` and `video_synth` to pass their health checks before starting.

---

## Quick Start

```bash
# Build and start all three services
docker compose up --build

# Run in the background
docker compose up --build -d

# View logs
docker compose logs -f

# Tear down (keeps ollama_data volume with downloaded model weights)
docker compose down
```

### First-Run Note

On the first start, Ollama downloads the `llama3.2:3b` model (~2 GB). This happens once; subsequent starts use the `ollama_data` Docker volume as a cache. You can track progress with:

```bash
docker compose logs -f ollama
```

---

## Service URLs

Once all services are healthy:

| URL | Description |
|---|---|
| `http://localhost:8000/ui/` | React web control panel |
| `http://localhost:8000/docs` | Swagger interactive API explorer |
| `http://localhost:8000/stream` | MJPEG live video stream |
| `http://localhost:8000/snapshot` | Current frame as JPEG |
| `http://localhost:8001/` | AI agent web chat UI |
| `http://localhost:8001/chat` | Agent POST endpoint (`{"message": "..."}`) |
| `http://localhost:8001/docs` | Agent API docs |
| `http://localhost:11434` | Ollama API (for direct model queries) |

---

## Changing the LLM Model

The default model is `llama3.2:3b` (~2 GB, fast on CPU). To use a larger or different model, edit the `OLLAMA_MODEL` environment variable in `docker-compose.yml`:

```yaml
agent:
  environment:
    - OLLAMA_MODEL=llama3.1:8b   # ~5 GB, better reasoning
```

Recommended models for this use case:

| Model | Size | Notes |
|---|---|---|
| `llama3.2:3b` | ~2 GB | Default; fast, adequate for parameter control |
| `llama3.1:8b` | ~5 GB | Better at multi-step reasoning |
| `qwen2.5:7b` | ~5 GB | Strong tool-calling performance |
| `minicpm-v` | ~5 GB | Multimodal — can analyze `/snapshot` frames visually |

After changing the model, rebuild:

```bash
docker compose down
docker compose up --build
```

### Enabling Vision (Multimodal)

Set `VISION_MODEL` to a multimodal model to let the agent analyze the current visual output before deciding what to change:

```yaml
agent:
  environment:
    - OLLAMA_MODEL=llama3.2:3b
    - VISION_MODEL=minicpm-v
```

When `VISION_MODEL` is set, the agent fetches `/snapshot` from the video synth and includes the frame in its context before responding.

---

## AI Agent Chat UI

Open `http://localhost:8001/` in a browser. Type natural-language commands:

> "Make the plasma animation pulse with a slow sine wave on the speed parameter"

> "Switch to Metaballs and crank up the glitch intensity"

> "Reduce all warp parameters and create a smooth, calming blue wash"

The agent has access to two tools:

- **`list_params`** — fetches the full parameter list from `/params`
- **`set_param`** — calls `PUT /params/{name}` to update a value

At startup, the agent reads `documentation/PARAMETERS.md` to learn what every parameter does, so it can make informed decisions without fetching docs on every request.

### Chat API

You can also POST to the agent programmatically:

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Set a random combination of glitch effects"}'
```

Response:

```json
{
  "response": "I've enabled pixel shift, color splitting, and set glitch intensity to 70...",
  "tool_calls": [
    {"name": "set_param", "args": {"name": "enable_pixel_shift", "value": 1}},
    {"name": "set_param", "args": {"name": "enable_color_split", "value": 1}},
    {"name": "set_param", "args": {"name": "glitch_intensity_max", "value": 70}}
  ]
}
```

---

## Customizing the Agent System Prompt

The agent's personality, constraints, and context are defined in the system prompt inside `agent/agent.py`. To change how the agent behaves — for example to restrict it to a specific animation style or give it a performance persona — edit the `SYSTEM_PROMPT` string in that file and rebuild:

```bash
docker compose up --build agent
```

You can also inject additional context via a mounted file. For example, add a `agent/context.md` with patch descriptions and mount it:

```yaml
agent:
  volumes:
    - ./agent/context.md:/app/context.md
```

Then reference it in the system prompt: `Path('/app/context.md').read_text()`.

---

## Persisting Patches

Patches saved through the web UI or API are written to `save/saved_values.yaml`. This directory is mounted as a Docker volume so your patches survive container restarts:

```yaml
video_synth:
  volumes:
    - ./save:/app/save
```

To back up your patches, copy the `save/` directory to a safe location.

---

## Troubleshooting

### Agent never becomes ready

The agent waits for Ollama's health check to pass and for the model pull to complete. Check:

```bash
docker compose logs ollama   # Is the model still downloading?
docker compose logs agent    # Any startup errors?
```

### Video synth shows black frames

Mesa software rendering requires the `LIBGL_ALWAYS_SOFTWARE=1` and `DISPLAY=:99` environment variables (set in `docker-compose.yml`). If you see GL errors, confirm these are present:

```bash
docker compose exec video_synth env | grep -E "DISPLAY|LIBGL"
```

### Port conflicts

If ports 8000, 8001, or 11434 are already in use, change the host-side port mappings in `docker-compose.yml`:

```yaml
ports:
  - "9000:8000"   # Access video synth at localhost:9000
```

### Running on a machine with a GPU (Nvidia)

Add the `deploy` section to the `ollama` service for GPU acceleration:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```
