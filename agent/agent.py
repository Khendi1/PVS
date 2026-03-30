"""
Video Synth AI Agent
====================
A local LLM agent (via Ollama's OpenAI-compatible API) that controls the video
synthesizer over its REST API.  Exposes a web chat UI at http://localhost:8001
and a POST /chat endpoint.

Environment variables:
    SYNTH_URL       URL of the video synth API  (default: http://localhost:8000)
    OLLAMA_URL      Base URL of the Ollama server (default: http://localhost:11434)
    OLLAMA_MODEL    Chat/tool model name          (default: qwen3-coder)
    VISION_MODEL    Vision model for frame review (default: same as OLLAMA_MODEL)
                    Set to a multimodal model like "minicpm-v" or "llava" to enable
                    visual feedback.  Leave unset to disable vision.
"""

import base64
import json
import logging
import os
import re
import uvicorn
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load parameter descriptions from PARAMETERS.md at startup
# Parses table rows: | Label | `key` | min | max | default | Description |
# Builds a dict: short_key -> description
# ---------------------------------------------------------------------------

_PARAM_DOC_RE = re.compile(r"\|\s*[^|]+\|\s*`([^`]+)`\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|")

def _load_param_docs() -> dict[str, str]:
    candidates = [
        Path(__file__).parent.parent / "documentation" / "PARAMETERS.md",
        Path(__file__).parent / "PARAMETERS.md",
    ]
    for path in candidates:
        if path.exists():
            docs = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                m = _PARAM_DOC_RE.match(line)
                if m:
                    docs[m.group(1).strip()] = m.group(2).strip()
            log.info("Loaded %d param descriptions from %s", len(docs), path)
            return docs
    log.warning("PARAMETERS.md not found — agent will have no param descriptions")
    return {}

_PARAM_DOCS: dict[str, str] = _load_param_docs()

SYNTH_URL    = os.getenv("SYNTH_URL",    "http://localhost:8000")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
MODEL        = os.getenv("OLLAMA_MODEL", "qwen3-coder")
VISION_MODEL = os.getenv("VISION_MODEL", "")   # empty = vision disabled

# OpenAI client pointing at Ollama's OpenAI-compatible API
client = OpenAI(
    base_url=f"{OLLAMA_URL}/v1",
    api_key="ollama",          # Ollama ignores the key but the client requires one
)

# ---------------------------------------------------------------------------
# Param cache — refreshed each request, used for validation and system prompt
# ---------------------------------------------------------------------------

_param_cache: dict[str, dict] = {}  # name -> {value, min, max}


def _refresh_param_cache():
    raw = _synth("GET", "/params")
    if isinstance(raw, list):
        _param_cache.clear()
        for p in raw:
            _param_cache[p["name"]] = {"value": p["value"], "min": p.get("min"), "max": p.get("max")}


def _coerce_value(value):
    """Coerce a string number to float/int so the API doesn't get type errors."""
    if isinstance(value, str):
        try:
            f = float(value)
            return int(f) if f == int(f) else f
        except (ValueError, OverflowError):
            pass
    return value


def _suggest_params(name: str) -> list[str]:
    """Return up to 5 real param names that share any word with the requested name."""
    short = name.split(".")[-1].lower()
    words = [w for w in re.split(r"[_\-.]", short) if w]
    scored = []
    for k in _param_cache:
        k_low = k.lower()
        score = sum(1 for w in words if w in k_low)
        if score:
            scored.append((score, k))
    scored.sort(key=lambda x: -x[0])
    return [k for _, k in scored[:5]]


def _validate_and_set(name: str, value) -> dict:
    """Validate param name against cache, coerce value, then call API."""
    value = _coerce_value(value)
    if _param_cache and name not in _param_cache:
        candidates = _suggest_params(name)
        hint = f" Use one of these instead: {candidates}" if candidates else " Use exact names from the parameter list."
        return {"error": f"Parameter '{name}' not found.{hint}"}
    return _synth("PUT", f"/params/{name}", json={"value": value})


# ---------------------------------------------------------------------------
# Tool implementations — each calls the synth REST API
# ---------------------------------------------------------------------------

def _synth(method: str, path: str, **kwargs):
    """Call the synth API, return parsed JSON or error dict."""
    try:
        r = httpx.request(method, f"{SYNTH_URL}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def tool_get_params(_args: dict) -> dict:
    """Return a condensed list: just name, value, min, max — no metadata bloat."""
    raw = _synth("GET", "/params")
    if isinstance(raw, list):
        return [{"name": p["name"], "value": p["value"], "min": p.get("min"), "max": p.get("max")} for p in raw]
    return raw


def tool_set_param(args: dict) -> dict:
    return _validate_and_set(args["param_name"], args["value"])


def tool_set_params(args: dict) -> dict:
    """Set multiple parameters at once."""
    results = {}
    for item in args.get("changes", []):
        results[item["param_name"]] = _validate_and_set(item["param_name"], item["value"])
    return results


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


def tool_get_frame(_args: dict) -> dict:
    """Capture a JPEG frame from the synth for visual inspection."""
    try:
        r = httpx.get(f"{SYNTH_URL}/snapshot", timeout=5)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode()
        return {"frame_base64": b64, "content_type": "image/jpeg"}
    except Exception as exc:
        return {"error": str(exc)}


TOOL_MAP = {
    "get_params":   tool_get_params,
    "set_param":    tool_set_param,
    "set_params":   tool_set_params,
    "reset_param":  tool_reset_param,
    "next_patch":   tool_next_patch,
    "prev_patch":   tool_prev_patch,
    "random_patch": tool_random_patch,
    "create_lfo":   tool_create_lfo,
    "delete_lfo":   tool_delete_lfo,
    "list_lfos":    tool_list_lfos,
    "get_frame":    tool_get_frame,
}

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_param",
            "description": "Set a single parameter value on the video synthesizer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "Full parameter path, e.g. 'SRC_1_EFFECTS.hue_shift'",
                    },
                    "value": {
                        "description": "The value to set (number or string for dropdowns)",
                    },
                },
                "required": ["param_name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_params",
            "description": (
                "Set MULTIPLE parameters at once. Use this instead of calling set_param "
                "repeatedly — it is faster and preferred for any change involving more than "
                "one parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "description": "List of {param_name, value} pairs",
                        "items": {
                            "type": "object",
                            "properties": {
                                "param_name": {"type": "string"},
                                "value": {"description": "number or string"},
                            },
                            "required": ["param_name", "value"],
                        },
                    }
                },
                "required": ["changes"],
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

# Only expose get_frame if a vision-capable model is configured
_VISION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_frame",
        "description": (
            "Capture the current video output as a JPEG image so you can see what the "
            "synthesizer is rendering right now. Use this to evaluate your changes visually."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _get_tools() -> list:
    return TOOLS + [_VISION_TOOL] if VISION_MODEL else TOOLS

SYSTEM_PROMPT_BASE = """You are a creative AI assistant that controls a real-time video synthesizer.
Your ONLY job is to call tools to change the synthesizer. The current parameter list is provided below — use it immediately.

RULES:
- NEVER describe, summarize, or comment on the parameter list. It is context for you to act on.
- Your FIRST action must always be a set_params or create_lfo call. Never respond with only text.
- Use set_params (plural) for ALL multi-parameter changes — pass all changes in one call.
- For "make it glitchy": max out params with names like glitch, distort, chromatic, pixel, scan, aberration.
- For "pulse" or "breathing": use create_lfo on alpha, brightness, or hue_shift.
- For "freestyle": change 8-15 parameters boldly.
- Parameter names follow the pattern GROUP.param_name — use exact names from the list below.
- After calling tools, respond with 1-3 sentences describing what was done. Nothing more.

CURRENT PARAMETERS:
{param_list}
"""


def _build_system_prompt() -> str:
    """Refresh param cache and inject current params (with descriptions) into the system prompt."""
    _refresh_param_cache()
    if _param_cache:
        lines = []
        for name, p in _param_cache.items():
            short_key = name.split(".")[-1]
            desc = _PARAM_DOCS.get(short_key, "")
            suffix = f"  # {desc}" if desc else ""
            lines.append(f"  {name}: {p['value']} (min={p['min']}, max={p['max']}){suffix}")
        param_str = "\n".join(lines)
    else:
        param_str = "(could not fetch params)"
    return SYSTEM_PROMPT_BASE.format(param_list=param_str)

# ---------------------------------------------------------------------------
# Fallback parser for text-format tool calls
# Some models (e.g. qwen3-coder) embed tool calls in response content as:
#   <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
# instead of using the structured tool_calls field.
# ---------------------------------------------------------------------------

_FN_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_text_tool_calls(text: str) -> list[dict]:
    """Extract tool calls embedded as XML-ish text. Returns list of {name, arguments}."""
    calls = []
    for fn_match in _FN_RE.finditer(text):
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)
        args = {}
        for p_match in _PARAM_RE.finditer(fn_body):
            key = p_match.group(1)
            raw = p_match.group(2).strip()
            try:
                args[key] = json.loads(raw)
            except json.JSONDecodeError:
                args[key] = raw
        calls.append({"name": fn_name, "arguments": args})
    return calls


def _strip_tool_calls(text: str) -> str:
    """Remove embedded <function=...> blocks from response text."""
    return _FN_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Conversation history (in-memory, per-process)
# ---------------------------------------------------------------------------

conversation_history: list[dict] = []

# ---------------------------------------------------------------------------
# Agentic chat loop
# ---------------------------------------------------------------------------

def _execute_tool(fn_name: str, fn_args: dict) -> dict:
    """Run a tool and return the result."""
    log.info("Tool call: %s(%s)", fn_name, fn_args)
    if fn_name in TOOL_MAP:
        result = TOOL_MAP[fn_name](fn_args)
    else:
        result = {"error": f"Unknown tool: {fn_name}"}
    return result


def _append_tool_result(messages: list, tool_call_id: str, fn_name: str, result: dict):
    """Append a tool result to the message list, handling vision frames specially."""
    if fn_name == "get_frame" and VISION_MODEL and "frame_base64" in result:
        tool_content = [
            {"type": "text", "text": "Current video frame:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{result['frame_base64']}"
            }},
        ]
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_content})
    else:
        log.info("Tool result: %s", result)
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})


def run_agent(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": _build_system_prompt()}] + conversation_history

    for _ in range(20):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_get_tools(),
            tool_choice="auto",
        )
        msg = response.choices[0].message
        content = msg.content or ""

        # --- Structured tool calls (OpenAI format) ---
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}
                result = _execute_tool(fn_name, fn_args)
                _append_tool_result(messages, tc.id, fn_name, result)
            continue

        # --- Fallback: text-embedded tool calls (qwen3-coder style) ---
        text_calls = _parse_text_tool_calls(content)
        if text_calls:
            messages.append({"role": "assistant", "content": content})
            for i, tc in enumerate(text_calls):
                fn_name = tc["name"]
                fn_args = tc["arguments"]
                result = _execute_tool(fn_name, fn_args)
                # Text-format tool calls have no real tool_call_id; use a synthetic one
                _append_tool_result(messages, f"text_{i}", fn_name, result)
            continue

        # --- Final text response ---
        text = _strip_tool_calls(content)
        conversation_history.append({"role": "assistant", "content": text})
        return text

    raise RuntimeError("Agent exceeded max iterations without a final response.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Synth AI Agent")


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = run_agent(req.message)
        return {"response": response}
    except Exception as exc:
        log.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/chat/history")
async def clear_history():
    conversation_history.clear()
    return {"status": "cleared"}


@app.post("/shutdown")
async def shutdown():
    """Gracefully stop the agent process."""
    import signal
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "vision_model": VISION_MODEL or None,
        "synth_url": SYNTH_URL,
        "ollama_url": OLLAMA_URL,
    }


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
    .user       { color: #88aaff; }
    .user::before       { content: "you › "; color: #4466cc; }
    .assistant  { color: #00ff88; }
    .assistant::before  { content: "agent › "; color: #008844; }
    .thinking   { color: #888; font-style: italic; }
    .error      { color: #ff4444; }
    #input-row  { display: flex; gap: 8px; }
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
    }
    button:disabled { opacity: 0.4; cursor: default; }
    #clear-btn {
      background: transparent;
      color: #666;
      border: 1px solid #333;
      padding: 10px 12px;
    }
    #clear-btn:hover { color: #aaa; border-color: #666; }
    #shutdown-btn {
      background: transparent;
      color: #663333;
      border: 1px solid #441111;
      padding: 10px 12px;
    }
    #shutdown-btn:hover { color: #ff4444; border-color: #ff4444; }
  </style>
</head>
<body>
  <h1>⬡ VIDEO SYNTH AI AGENT</h1>
  <div id="chat"></div>
  <div id="input-row">
    <input id="input" type="text"
           placeholder="Describe the visuals you want…"
           autofocus>
    <button id="send-btn" onclick="send()">Send</button>
    <button id="clear-btn" onclick="clearHistory()" title="Clear history">✕</button>
    <button id="shutdown-btn" onclick="shutdown()" title="Kill agent process">⏻</button>
  </div>
  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

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
        addMsg(res.ok ? 'assistant' : 'error', data.response ?? data.detail ?? 'No response.');
      } catch (err) {
        thinking.remove();
        addMsg('error', `Network error: ${err}`);
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    async function clearHistory() {
      await fetch('/chat/history', { method: 'DELETE' });
      chat.innerHTML = '';
    }

    async function shutdown() {
      if (!confirm('Kill the agent process?')) return;
      try { await fetch('/shutdown', { method: 'POST' }); } catch (_) {}
      document.body.innerHTML = '<p style="color:#ff4444;font-family:monospace;padding:2rem">Agent stopped.</p>';
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
