# Video Synth AI Agent

A local LLM agent that controls the video synthesizer through natural language. Describe the visuals you want — the agent figures out which parameters to change and does it.

## How it works

```
You (browser chat)
      │
      ▼
agent.py  :8001          ← FastAPI + Ollama tool-calling loop
      │
      ├── Ollama  :11434  ← local LLM (qwen3-coder, llama3.2, etc.)
      │
      └── PVS API :8000   ← video synth REST API
```

The agent exposes a minimal chat UI at `http://localhost:8001`. When you send a message, it:

1. Calls `get_params` to discover all available parameters
2. Decides which tools to call (`set_param`, `create_lfo`, `next_patch`, etc.)
3. Executes them against the synth API in a loop until done
4. Returns a text explanation of what it changed

## Prerequisites

- Ollama running locally: `ollama serve`
- The model pulled: `ollama pull qwen3-coder`
- PVS video synth running: `python -m video_synth` (port 8000)
- Python venv active with agent dependencies installed

## Running locally (no Docker)

```bash
# Install dependencies into the active venv
pip install -r agent/requirements.txt

# Run the agent (from repo root)
OLLAMA_URL=http://localhost:11434 \
OLLAMA_MODEL=qwen3-coder \
SYNTH_URL=http://localhost:8000 \
python agent/agent.py
```

On Windows (PowerShell):

```powershell
$env:OLLAMA_URL   = "http://localhost:11434"
$env:OLLAMA_MODEL = "qwen3-coder"
$env:SYNTH_URL    = "http://localhost:8000"
python agent/agent.py
```

Then open `http://localhost:8001` in a browser.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model name as shown in `ollama list` |
| `SYNTH_URL` | `http://video_synth:8000` | PVS REST API base URL |

The defaults use Docker Compose hostnames. Override them for local use as shown above.

## Running with Docker

A `Dockerfile` is included if you want to containerise the agent. You'll need Ollama and the synth API reachable from the container — easiest via Docker Compose or by pointing `OLLAMA_URL` and `SYNTH_URL` at the host IP.

```bash
docker build -t pvs-agent agent/
docker run -p 8001:8001 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen3-coder \
  -e SYNTH_URL=http://host.docker.internal:8000 \
  pvs-agent
```

## Available tools

| Tool | What it does |
|---|---|
| `get_params` | Fetch all parameters with values, ranges, and descriptions |
| `set_param` | Set a parameter by full path, e.g. `SRC_1_EFFECTS.hue_shift` |
| `reset_param` | Reset a parameter to its default |
| `next_patch` | Load next preset |
| `prev_patch` | Load previous preset |
| `random_patch` | Load a random preset |
| `create_lfo` | Attach an oscillator to a param (rate Hz, amplitude, shape) |
| `delete_lfo` | Remove an oscillator from a param |
| `list_lfos` | Show all active oscillators |

## Example prompts

```
slow hypnotic blue pulse on both sources
make it chaotic — glitch everything
calm down and fade to black
add a breathing effect to the brightness
load a random patch and make the colors warmer
```

## Conversation history

History is kept in memory for the duration of the process. Use the **Clear** button in the UI or:

```bash
curl -X DELETE http://localhost:8001/chat/history
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send `{"message": "..."}`, get `{"response": "..."}` |
| `DELETE` | `/chat/history` | Clear conversation history |
| `GET` | `/health` | Returns model name and synth URL |
| `GET` | `/` | Chat UI (HTML) |
