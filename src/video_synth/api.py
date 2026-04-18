"""
FastAPI server for remote control of the video synthesizer.
Allows an agent or external application to control parameters via HTTP REST API.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
import uvicorn
import io
import cv2
from lfo import LFO, LFOShape

import sys as _sys
if getattr(_sys, 'frozen', False):
    # Running inside a PyInstaller bundle: _MEIPASS is the bundle root
    _BUNDLE_ROOT = _sys._MEIPASS
else:
    # Running from source: go up one level from video_synth/ to the project root
    _BUNDLE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_WEB_DIST = os.path.join(_BUNDLE_ROOT, 'web', 'dist')

log = logging.getLogger(__name__)


class ParamValue(BaseModel):
    """Request model for setting a parameter value."""
    value: Any


class ParamInfo(BaseModel):
    """Response model for parameter information."""
    name: str
    value: Any
    min: Any
    max: Any
    default: Any
    group: str
    subgroup: str
    type: str
    options: Optional[List[str]] = None


class LfoInfo(BaseModel):
    param_name: str
    osc_name: str
    shape: str
    frequency: float
    amplitude: float
    phase: float
    seed: float

class LfoRequest(BaseModel):
    shape: str = "SINE"
    frequency: float = 0.5
    amplitude: Optional[float] = None
    phase: float = 0.0
    seed: float = 0.0


class BulkParamUpdate(BaseModel):
    """Request model for bulk parameter update."""
    params: Dict[str, Any]


class APIServer:
    """FastAPI server for controlling video synthesizer parameters."""

    def __init__(self, params, mixer=None, save_controller=None, midi_mapper=None, osc_banks=None, audio_module=None, host="127.0.0.1", port=8000):
        self.params = params
        self.mixer = mixer
        self.save_controller = save_controller
        self.midi_mapper = midi_mapper
        self.osc_banks = osc_banks or {}  # group_prefix → OscBank
        self.audio_module = audio_module
        self._lfo_map = {}  # param_name → LFO
        self.host = host
        self.port = port
        self._ws_clients: set[WebSocket] = set()
        self._loop = None

        @asynccontextmanager
        async def lifespan(app):
            self._loop = asyncio.get_running_loop()
            yield

        self.app = FastAPI(
            title="Video Synth API",
            description="API for controlling video synthesizer parameters",
            version="1.0.0",
            lifespan=lifespan,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()
        if os.path.isdir(_WEB_DIST):
            self.app.mount("/ui", StaticFiles(directory=_WEB_DIST, html=True), name="static")
            self._web_ui = True
        else:
            log.warning("web/dist not found - run 'npm run build' in web/ to enable the web UI")
            self._web_ui = False
        self._server_thread = None
        self._should_stop = False

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """API root - returns basic info."""
            return {
                "name": "Video Synth API",
                "version": "1.0.0",
                "endpoints": {
                    "GET /params": "List all parameters",
                    "GET /params/{name}": "Get specific parameter",
                    "PUT /params/{name}": "Set parameter value",
                    "POST /params/reset/{name}": "Reset parameter to default",
                    "GET /snapshot": "Get current frame as JPEG"
                }
            }

        def _serialize_options(param) -> Optional[List[str]]:
            """Convert a Param's options to a flat list of strings for the API."""
            opts = param.options
            if opts is None:
                return None
            if isinstance(opts, dict):
                return list(opts.keys())
            if isinstance(opts, list):
                return [str(o) for o in opts]
            # Enum class
            try:
                return [e.name for e in opts]
            except TypeError:
                return [str(opts)]

        def _enum_name(v) -> str:
            """Return the enum member name if v is an enum, else str(v)."""
            return v.name if hasattr(v, 'name') else str(v)

        def _make_param_info(name, param) -> ParamInfo:
            return ParamInfo(
                name=name,
                value=param.value,
                min=param.min,
                max=param.max,
                default=param.default,
                group=_enum_name(param.group),
                subgroup=_enum_name(param.subgroup),
                type=_enum_name(param.type),
                options=_serialize_options(param),
            )

        @self.app.get("/params", response_model=List[ParamInfo])
        async def list_params():
            """Get list of all parameters with their current values."""
            return [_make_param_info(n, p) for n, p in self.params.params.items()]

        @self.app.get("/params/{param_name}", response_model=ParamInfo)
        async def get_param(param_name: str):
            """Get specific parameter info and current value."""
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")
            return _make_param_info(param_name, self.params.params[param_name])

        @self.app.put("/params/{param_name}")
        async def set_param(param_name: str, param_value: ParamValue):
            """Set a parameter value."""
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.params[param_name]

            # Validate numeric bounds (skip for dropdown string values)
            if isinstance(param_value.value, (int, float)):
                if param_value.value < param.min or param_value.value > param.max:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Value {param_value.value} out of range [{param.min}, {param.max}]"
                    )

            # Set the value
            param.value = param_value.value

            log.info(f"API: Set {param_name} = {param_value.value}")

            return {
                "success": True,
                "param": param_name,
                "value": param.value
            }

        @self.app.post("/params/reset/{param_name}")
        async def reset_param(param_name: str):
            """Reset parameter to its default value."""
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.params[param_name]
            param.reset()

            log.info(f"API: Reset {param_name} to {param.value}")

            return {
                "success": True,
                "param": param_name,
                "value": param.value
            }

        @self.app.get("/snapshot")
        async def get_snapshot():
            """Get current frame as JPEG image."""
            if self.mixer is None or self.mixer.current_frame is None:
                raise HTTPException(status_code=503, detail="No frame available")

            success, buffer = cv2.imencode('.jpg', self.mixer.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame")

            return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

        @self.app.get("/stream")
        async def mjpeg_stream():
            """MJPEG stream of the current output frame (~30 fps)."""
            def frame_generator():
                while not self._should_stop:
                    if self.mixer is None or self.mixer.current_frame is None:
                        time.sleep(0.05)
                        continue
                    frame = self.mixer.current_frame.copy()
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if success:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                               + buffer.tobytes() + b'\r\n')
                    time.sleep(1 / 30)

            return StreamingResponse(
                frame_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        # --- Patch endpoints ---

        @self.app.post("/patch/save")
        async def patch_save():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            self.save_controller.save_patch()
            return {"success": True}

        @self.app.post("/patch/next")
        async def patch_next():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            self.save_controller.load_next_patch()
            return {"success": True}

        @self.app.post("/patch/prev")
        async def patch_prev():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            self.save_controller.load_prev_patch()
            return {"success": True}

        @self.app.post("/patch/random")
        async def patch_random():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            self.save_controller.load_random_patch()
            return {"success": True}

        # --- MIDI learn endpoints ---

        class MidiLearnRequest(BaseModel):
            param: str  # qualified key "group/param_name"

        @self.app.post("/midi/learn")
        async def midi_learn(req: MidiLearnRequest):
            if self.midi_mapper is None:
                raise HTTPException(status_code=503, detail="MidiMapper not available")
            ok = self.midi_mapper.start_learn(req.param)
            if not ok:
                raise HTTPException(status_code=404, detail=f"Param '{req.param}' not found")
            return {"success": True, "target": req.param}

        @self.app.post("/midi/learn/cancel")
        async def midi_learn_cancel():
            if self.midi_mapper is None:
                raise HTTPException(status_code=503, detail="MidiMapper not available")
            self.midi_mapper.cancel_learn()
            return {"success": True}

        @self.app.get("/midi/learn/status")
        async def midi_learn_status():
            if self.midi_mapper is None:
                raise HTTPException(status_code=503, detail="MidiMapper not available")
            return self.midi_mapper.get_learn_state()

        # --- LFO endpoints ---

        def _get_osc_bank(param_name: str):
            prefix = param_name.split('.')[0] if '.' in param_name else None
            return self.osc_banks.get(prefix)

        @self.app.get("/lfo", response_model=List[LfoInfo])
        async def list_lfos():
            result = []
            for pname, osc in self._lfo_map.items():
                try:
                    shape_name = LFOShape(int(osc.shape.value)).name
                except (ValueError, KeyError):
                    shape_name = "SINE"
                result.append(LfoInfo(
                    param_name=pname,
                    osc_name=osc.name,
                    shape=shape_name,
                    frequency=float(osc.frequency.value),
                    amplitude=float(osc.amplitude.value),
                    phase=float(osc.phase.value),
                    seed=float(osc.seed.value),
                ))
            return result

        @self.app.get("/lfo/{param_name:path}", response_model=LfoInfo)
        async def get_lfo(param_name: str):
            if param_name not in self._lfo_map:
                raise HTTPException(status_code=404, detail=f"No LFO for param '{param_name}'")
            osc = self._lfo_map[param_name]
            try:
                shape_name = LFOShape(int(osc.shape.value)).name
            except (ValueError, KeyError):
                shape_name = "SINE"
            return LfoInfo(
                param_name=param_name,
                osc_name=osc.name,
                shape=shape_name,
                frequency=float(osc.frequency.value),
                amplitude=float(osc.amplitude.value),
                phase=float(osc.phase.value),
                seed=float(osc.seed.value),
            )

        @self.app.post("/lfo/{param_name:path}")
        async def create_lfo(param_name: str, req: LfoRequest):
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Param '{param_name}' not found")
            osc_bank = _get_osc_bank(param_name)
            if osc_bank is None:
                raise HTTPException(status_code=400, detail=f"No oscillator bank for param '{param_name}'")
            param = self.params.params[param_name]
            # Remove existing LFO for this param if any
            if param_name in self._lfo_map:
                old_osc = self._lfo_map.pop(param_name)
                old_osc.unlink_param()
                osc_bank.remove_oscillator(old_osc)
            shape_val = LFOShape[req.shape].value if req.shape in LFOShape.__members__ else LFOShape.SINE.value
            amplitude = req.amplitude if req.amplitude is not None else (param.max - param.min) / 2
            osc_name = f"web_lfo_{id(param)}"
            osc = osc_bank.add_oscillator(osc_name, frequency=req.frequency, amplitude=amplitude, phase=req.phase, shape=shape_val)
            osc.link_param(param)
            osc.seed.value = req.seed
            self._lfo_map[param_name] = osc
            log.info(f"LFO created for '{param_name}' shape={req.shape} freq={req.frequency}")
            return {"success": True, "osc_name": osc_name}

        @self.app.put("/lfo/{param_name:path}")
        async def update_lfo(param_name: str, req: LfoRequest):
            if param_name not in self._lfo_map:
                raise HTTPException(status_code=404, detail=f"No LFO for param '{param_name}'")
            osc = self._lfo_map[param_name]
            if req.shape in LFOShape.__members__:
                osc.shape.value = LFOShape[req.shape].value
            osc.frequency.value = req.frequency
            if req.amplitude is not None:
                osc.amplitude.value = req.amplitude
            osc.phase.value = req.phase
            osc.seed.value = req.seed
            return {"success": True}

        @self.app.delete("/lfo/{param_name:path}")
        async def delete_lfo(param_name: str):
            if param_name not in self._lfo_map:
                raise HTTPException(status_code=404, detail=f"No LFO for param '{param_name}'")
            osc = self._lfo_map.pop(param_name)
            osc.unlink_param()
            osc_bank = _get_osc_bank(param_name)
            if osc_bank:
                osc_bank.remove_oscillator(osc)
            log.info(f"LFO removed for '{param_name}'")
            return {"success": True}

        # --- Audio endpoints ---

        @self.app.get("/audio/bands")
        async def get_audio_bands():
            """Get current FFT band energies and beat detection state."""
            if self.audio_module is None:
                raise HTTPException(status_code=503, detail="Audio module not available")
            return {
                "bands": list(self.audio_module.band_energies),
                "beat": bool(self.audio_module.beat_detector.is_beat),
            }

        # --- Bulk param update ---

        @self.app.put("/params/bulk")
        async def set_params_bulk(req: BulkParamUpdate):
            """Set multiple parameters at once. Returns per-param result."""
            results = {}
            for name, value in req.params.items():
                if name not in self.params.params:
                    results[name] = {"success": False, "error": "not found"}
                    continue
                param = self.params.params[name]
                param.value = value
                results[name] = {"success": True, "value": param.value}
            return results

        @self.app.websocket("/ws/stream")
        async def ws_stream(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except (WebSocketDisconnect, Exception):
                self._ws_clients.discard(websocket)

    def push_frame(self, frame):
        """Push a video frame to all connected WebSocket clients (call from main loop)."""
        if not self._ws_clients or self._loop is None:
            return
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(buffer.tobytes()), self._loop)

    async def _broadcast(self, data: bytes):
        disconnected = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_bytes(data)
            except Exception:
                disconnected.add(ws)
        self._ws_clients -= disconnected

    def start(self):
        """Start the API server in a background thread."""
        if self._server_thread is not None:
            log.warning("API server already running")
            return

        def run_server():
            log.info(f"Starting API server on {self.host}:{self.port}")
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        log.info(f"API server started at http://{self.host}:{self.port}")
        log.info(f"API docs available at http://{self.host}:{self.port}/docs")
        if self._web_ui:
            log.info(f"Web UI available at http://{self.host}:{self.port}/ui")

    def stop(self):
        """Stop the API server."""
        self._should_stop = True
        log.info("API server stopped")
