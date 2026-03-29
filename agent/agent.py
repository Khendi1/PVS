"""
Video Synth AI Agent
====================
A local LLM agent (via Ollama) that controls the video synthesizer over its REST API.
Exposes a web chat UI at http://localhost:8001 and a POST /chat endpoint.

Environment variables:
    SYNTH_URL      URL of the video synth API  (default: http://video_synth:8000)
    OLLAMA_URL     URL of the Ollama server     (default: http://ollama:11434)
    OLLAMA_MODEL   Model name to use            (default: llama3.2:3b)
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import ollama
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYNTH_URL   = os.getenv("SYNTH_URL",    "http://video_synth:8000")
OLLAMA_URL  = os.getenv("OLLAMA_URL",   "http://ollama:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ---------------------------------------------------------------------------
# Tool implementations — each calls the synth REST API
# ---------------------------------------------------------------------------

def _synth(method: str, path: str, **kwargs):
    """Helper: call the synth API, return parsed JSON or error dict."""
    try:
        r = httpx.request(method, f"{SYNTH_URL}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def tool_get_params(_args: dict) -> dict:
    return _synth("GET", "/params")


def tool_set_param(args: dict) -> dict:
    return _synth("PUT", f"/params/{args['param_name']}", json={"value": args["value"]})


def tool_reset_param(args: dict) -> dict:
    return _synth("POST", f"/params/reset/{args['param_name']}")


def tool_next_patch(_args: dict) -> dict:
    return _synth("POST", "/patch/next")


def tool_prev_patch(_args: dict) -> dict:
    return _synth("POST", "/patch/prev")


def tool_random_patch(_args: dict) -> dict:
    return _synth("POST", "/patch/random")


def tool_create_lfo(args: dict) -> dict:
    payload = {
        "rate":      args.get("rate",      1.0),
        "amplitude": args.get("amplitude", 0.5),
        "shape":     args.get("shape",     "sine"),
    }
    return _synth("POST", f"/lfo/{args['param_name']}", json=payload)


def tool_delete_lfo(args: dict) -> dict:
    return _synth("DELETE", f"/lfo/{args['param_name']}")


def tool_list_lfos(_args: dict) -> dict:
    return _synth("GET", "/lfo")


TOOL_MAP = {
    "get_params":   tool_get_params,
    "set_param":    tool_set_param,
    "reset_param":  tool_reset_param,
    "next_patch":   tool_next_patch,
    "prev_patch":   tool_prev_patch,
    "random_patch": tool_random_patch,
    "create_lfo":   tool_create_lfo,
    "delete_lfo":   tool_delete_lfo,
    "list_lfos":    tool_list_lfos,
}

# ---------------------------------------------------------------------------
# Tool schema passed to Ollama
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_params",
            "description": (
                "Get all parameters from the video synthesizer with their current values, "
                "min/max ranges, and descriptions. Call this first to discover what you can control."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_param",
            "description": "Set a parameter value on the video synthesizer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "Full parameter path, e.g. 'SRC_1_EFFECTS.hue_shift'",
                    },
                    "value": {
                        "type": "number",
                        "description": "The numeric value to set",
                    },
                },
                "required": ["param_name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_param",
            "description": "Reset a single parameter to its default value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "Full parameter path"},
                },
                "required": ["param_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_patch",
            "description": "Load the next preset/patch in sequence.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prev_patch",
            "description": "Load the previous preset/patch in sequence.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_patch",
            "description": "Load a random preset/patch for creative exploration.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_lfo",
            "description": (
                "Attach a Low-Frequency Oscillator to a parameter so it animates over time. "
                "Good for pulsing, sweeping, or breathing effects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "Full parameter path to animate"},
                    "rate":       {"type": "number", "description": "Oscillation rate in Hz (0.05–10)"},
                    "amplitude":  {"type": "number", "description": "Oscillation amplitude 0.0–1.0"},
                    "shape":      {
                        "type": "string",
                        "description": "Wave shape",
                        "enum": ["sine", "square", "triangle", "sawtooth"],
                    },
                },
                "required": ["param_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_lfo",
            "description": "Remove an LFO from a parameter so it stops animating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "Full parameter path"},
                },
                "required": ["param_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_lfos",
            "description": "List all currently active LFOs and their settings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """You are a creative AI assistant that controls a real-time video synthesizer.

You have tools to inspect and modify every parameter of the synthesizer — effects, animations,
mixer controls, color shifts, glitch intensity, warp distortion, etc. You can also attach LFOs
to make parameters animate over time, and switch between saved presets.

When the user describes a visual or emotional intent (e.g. "make it hypnotic", "calm blue waves",
"chaotic glitch storm"), use get_params first to see what's available, then manipulate relevant
parameters to achieve that look. Explain what you changed and why.

Be creative and precise. Always prefer small, deliberate changes over resetting everything."""

# ---------------------------------------------------------------------------
# Conversation history (in-memory, per-process)
# ---------------------------------------------------------------------------

conversation_history: list[dict] = []


# ---------------------------------------------------------------------------
# Model initialisation — pull the model at startup if not already present
# ---------------------------------------------------------------------------

def ensure_model_ready():
    client = ollama.Client(host=OLLAMA_URL)
    for attempt in range(30):
        try:
            available = [m.model for m in client.list().models]
            if any(MODEL in m for m in available):
                log.info("Model '%s' is ready.", MODEL)
                return
            log.info("Pulling model '%s' (this may take a while)…", MODEL)
            for progress in client.pull(MODEL, stream=True):
                status = getattr(progress, "status", "")
                if "pulling" in status or "success" in status:
                    log.info("  %s", status)
            return
        except Exception as exc:
            log.warning("Ollama not reachable yet (%s), retrying in 5 s…", exc)
            time.sleep(5)
    raise RuntimeError("Could not reach Ollama after 30 attempts.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_model_ready()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Synth AI Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Send a message to the agent; returns the final text response."""
    conversation_history.append({"role": "user", "content": req.message})

    client = ollama.Client(host=OLLAMA_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    # Agentic tool-calling loop — runs until the model stops requesting tools
    for iteration in range(15):
        response = client.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.message

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = msg.tool_calls
        messages.append(assistant_entry)

        if not msg.tool_calls:
            # Model produced a final text response
            conversation_history.append({"role": "assistant", "content": msg.content})
            return {"response": msg.content}

        # Execute each requested tool and feed results back
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            log.info("Tool call: %s(%s)", fn_name, fn_args)

            if fn_name in TOOL_MAP:
                result = TOOL_MAP[fn_name](fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            log.info("Tool result: %s", result)
            messages.append({"role": "tool", "content": json.dumps(result)})

    raise HTTPException(status_code=500, detail="Agent exceeded max iterations without a final response.")


@app.delete("/chat/history")
async def clear_history():
    """Reset the conversation history."""
    conversation_history.clear()
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "synth_url": SYNTH_URL}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Video Synth AI Agent</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Courier New', monospace;
      background: #080810;
      color: #c0ffc0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      padding: 16px;
      gap: 12px;
    }
    h1 { font-size: 1.1rem; color: #00ff88; letter-spacing: 0.1em; }
    #chat {
      flex: 1;
      border: 1px solid #00ff4433;
      border-radius: 4px;
      padding: 12px;
      overflow-y: auto;
      background: #0a0a14;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg { line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
    .user    { color: #88aaff; }
    .user::before    { content: "you › "; color: #4466cc; }
    .assistant { color: #00ff88; }
    .assistant::before { content: "agent › "; color: #008844; }
    .thinking { color: #888; font-style: italic; }
    #input-row { display: flex; gap: 8px; }
    #input {
      flex: 1;
      background: #0d0d1a;
      color: #c0ffc0;
      border: 1px solid #00ff4466;
      border-radius: 4px;
      padding: 10px;
      font-family: inherit;
      font-size: 0.95rem;
    }
    #input:focus { outline: none; border-color: #00ff88; }
    button {
      background: #00ff88;
      color: #080810;
      border: none;
      border-radius: 4px;
      padding: 10px 20px;
      font-family: inherit;
      font-weight: bold;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    button:disabled { opacity: 0.4; cursor: default; }
    #clear-btn {
      background: transparent;
      color: #666;
      border: 1px solid #333;
    }
    #clear-btn:hover { color: #aaa; border-color: #666; }
  </style>
</head>
<body>
  <h1>⬡ VIDEO SYNTH AI AGENT</h1>
  <div id="chat"></div>
  <div id="input-row">
    <input id="input" type="text"
           placeholder="Describe the visuals you want… (e.g. 'slow hypnotic blue pulse')"
           autofocus>
    <button id="send-btn" onclick="send()">Send</button>
    <button id="clear-btn" onclick="clearHistory()">Clear</button>
  </div>
  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');

    input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

    function addMsg(role, text) {
      const p = document.createElement('p');
      p.className = `msg ${role}`;
      p.textContent = text;
      chat.appendChild(p);
      chat.scrollTop = chat.scrollHeight;
      return p;
    }

    async function send() {
      const msg = input.value.trim();
      if (!msg) return;
      input.value = '';
      sendBtn.disabled = true;

      addMsg('user', msg);
      const thinking = addMsg('thinking', 'thinking…');

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg }),
        });
        const data = await res.json();
        thinking.remove();
        addMsg('assistant', data.response ?? data.detail ?? 'No response.');
      } catch (err) {
        thinking.remove();
        addMsg('thinking', `Error: ${err}`);
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    async function clearHistory() {
      await fetch('/chat/history', { method: 'DELETE' });
      chat.innerHTML = '';
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
