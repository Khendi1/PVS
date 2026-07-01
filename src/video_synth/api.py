# Video Synth — real-time collaborative visual art synthesizer.
# Copyright (C) 2026 Kyle Henderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
import uvicorn
import io
import cv2
import yaml
from lfo import LFO, LFOShape
from param_history import ParamHistory

import sys as _sys
if getattr(_sys, 'frozen', False):
    # Running inside a PyInstaller bundle: _MEIPASS is the bundle root
    _BUNDLE_ROOT = _sys._MEIPASS
else:
    # Running from source: api.py lives at <root>/src/video_synth/, so the
    # project root (which holds web/dist) is two levels up.
    _BUNDLE_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
_WEB_DIST = os.path.join(_BUNDLE_ROOT, 'web', 'dist')

log = logging.getLogger(__name__)

# --- AGPL §13: where network users can obtain the Corresponding Source ---
SOURCE_URL = "https://github.com/Khendi1/PVS"
try:
    from video_synth import __version__ as _VERSION
except Exception:  # pragma: no cover - version lookup is best-effort
    _VERSION = "unknown"


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
    info: str = ""


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
        self.history = ParamHistory(params)  # undo/redo ring buffer over all params
        self.cv_controller = None  # set after construction if --cv is active
        self.bpm_clock = None      # set after construction
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
        self._uvicorn_server = None
        self._should_stop = False

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """API root - returns basic info."""
            return {
                "name": "Video Synth API",
                "version": "1.0.0",
                "license": "AGPL-3.0-or-later",
                "source": SOURCE_URL,
                "endpoints": {
                    "GET /params": "List all parameters",
                    "GET /params/{name}": "Get specific parameter",
                    "PUT /params/{name}": "Set parameter value",
                    "POST /params/reset/{name}": "Reset parameter to default",
                    "GET /snapshot": "Get current frame as JPEG",
                    "POST /patch/morph": "Lerp params to a patch over N seconds",
                    "GET /source": "Corresponding source location (AGPL §13)"
                }
            }

        @self.app.get("/source")
        async def source():
            """AGPL §13 source offer: where to obtain this instance's source.

            The AGPL requires that users who interact with a modified version
            over a network be offered the Corresponding Source. This endpoint
            (and the link in the web UI) provides that offer.
            """
            return {
                "program": "Video Synth",
                "version": _VERSION,
                "license": "AGPL-3.0-or-later",
                "source_url": SOURCE_URL,
                "notice": (
                    "This program is free software under the GNU Affero General "
                    "Public License v3 or later. As required by AGPL section 13, "
                    "the Corresponding Source for this running instance is "
                    "available at source_url."
                ),
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
                info=param.info or "",
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

            # Record the previous state so undo returns to it, then set the value
            self.history.snapshot()
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
            # Record the previous state so undo returns to it before resetting
            self.history.snapshot()
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
            # Record the pre-load state so a patch load can be undone
            self.history.snapshot()
            self.save_controller.load_next_patch()
            return {"success": True}

        @self.app.post("/patch/prev")
        async def patch_prev():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            # Record the pre-load state so a patch load can be undone
            self.history.snapshot()
            self.save_controller.load_prev_patch()
            return {"success": True}

        @self.app.post("/patch/random")
        async def patch_random():
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            # Record the pre-load state so a patch load can be undone
            self.history.snapshot()
            self.save_controller.load_random_patch()
            return {"success": True}

        # --- Patch morph (interpolation) endpoint ---

        @self.app.post("/patch/morph")
        async def patch_morph(target: int, duration: float = 5.0):
            """Lerp all numeric params from their current values to patch
            ``target`` over ``duration`` seconds in a background thread."""
            if self.save_controller is None:
                raise HTTPException(status_code=503, detail="SaveController not available")
            if self.save_controller.get_patch_entry(target) is None:
                raise HTTPException(status_code=404, detail=f"Patch index {target} not found")
            self.save_controller.morph_to(target, duration)
            return {"success": True, "target": target, "duration": duration}

        # --- History / undo endpoints ---

        @self.app.post("/undo")
        async def undo():
            """Revert params to the previous recorded state."""
            applied = self.history.undo()
            return {"success": True, "applied": applied}

        @self.app.post("/redo")
        async def redo():
            """Re-apply the most recently undone param state."""
            applied = self.history.redo()
            return {"success": True, "applied": applied}

        @self.app.get("/history")
        async def history():
            """Return the number of available undo and redo states."""
            return self.history.counts()

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

        # --- MIDI mapping export / import (share midi_mappings.yaml) ---

        class MidiImportRequest(BaseModel):
            mappings: str  # YAML text of the mappings document

        @self.app.get("/midi/export")
        async def midi_export():
            """Download the current MIDI mappings as a YAML file."""
            if self.midi_mapper is None:
                raise HTTPException(status_code=503, detail="MidiMapper not available")
            # Prefer the on-disk file so exports match what will be reloaded;
            # fall back to serializing the live mappings if the file is absent.
            yaml_path = self.midi_mapper._yaml_path
            try:
                if yaml_path.exists():
                    yaml_text = yaml_path.read_text()
                else:
                    yaml_text = self.midi_mapper.export_mappings_yaml()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read mappings: {e}")
            return PlainTextResponse(
                yaml_text,
                media_type="application/x-yaml",
                headers={
                    "Content-Disposition": 'attachment; filename="midi_mappings.yaml"'
                },
            )

        @self.app.post("/midi/import")
        async def midi_import(req: MidiImportRequest):
            """Import MIDI mappings from YAML text and apply them live."""
            if self.midi_mapper is None:
                raise HTTPException(status_code=503, detail="MidiMapper not available")
            try:
                loaded = self.midi_mapper.import_mappings(req.mappings)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            port_count = len(loaded)
            mapping_count = sum(len(m) for m in loaded.values())
            return {
                "success": True,
                "ports": port_count,
                "mappings": mapping_count,
            }

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

        # --- BPM Clock endpoints ---

        class BPMRequest(BaseModel):
            bpm: float

        @self.app.get("/bpm")
        async def get_bpm():
            """Get the current BPM clock status (BPM, beat phase, MIDI active)."""
            if self.bpm_clock is None:
                raise HTTPException(status_code=503, detail="BPM clock not available")
            return self.bpm_clock.get_status()

        @self.app.put("/bpm")
        async def set_bpm(req: BPMRequest):
            """Manually set BPM (overrides MIDI clock until next MIDI clock message)."""
            if self.bpm_clock is None:
                raise HTTPException(status_code=503, detail="BPM clock not available")
            self.bpm_clock.bpm = req.bpm
            return {"success": True, "bpm": self.bpm_clock.bpm}

        @self.app.post("/bpm/tap")
        async def tap_tempo():
            """Record a tap for tap-tempo BPM detection."""
            if self.bpm_clock is None:
                raise HTTPException(status_code=503, detail="BPM clock not available")
            self.bpm_clock.record_tap()
            return {"success": True, "bpm": self.bpm_clock.bpm}

        # --- CV-Gate endpoints ---

        class CVMapRequest(BaseModel):
            channel: int
            param: str          # qualified key "GroupLabel/param_name"
            volt_min: float = -5.0
            volt_max: float = 5.0
            smoothing: float = 0.05

        @self.app.get("/cv/devices")
        async def cv_list_devices():
            """List audio input devices available for CV input."""
            from cv_controller import CVController
            return {"devices": CVController.list_devices()}

        @self.app.get("/cv/mappings")
        async def cv_get_mappings():
            """Get current CV channel → parameter mappings."""
            if self.cv_controller is None:
                raise HTTPException(status_code=503, detail="CV controller not active (start with --cv)")
            return {"mappings": self.cv_controller.get_mappings()}

        @self.app.get("/cv/values")
        async def cv_get_values():
            """Get the current smoothed voltage reading for each mapped channel."""
            if self.cv_controller is None:
                raise HTTPException(status_code=503, detail="CV controller not active")
            return {"values": self.cv_controller.get_channel_values()}

        @self.app.post("/cv/map")
        async def cv_map_channel(req: CVMapRequest):
            """Map an audio channel to a parameter."""
            if self.cv_controller is None:
                raise HTTPException(status_code=503, detail="CV controller not active (start with --cv)")
            self.cv_controller.map_channel(
                channel=req.channel,
                qualified_key=req.param,
                volt_min=req.volt_min,
                volt_max=req.volt_max,
                smoothing=req.smoothing,
            )
            return {"success": True, "channel": req.channel, "param": req.param}

        @self.app.delete("/cv/map/{channel}")
        async def cv_unmap_channel(channel: int):
            """Remove the CV mapping for a channel."""
            if self.cv_controller is None:
                raise HTTPException(status_code=503, detail="CV controller not active")
            self.cv_controller.unmap_channel(channel)
            return {"success": True, "channel": channel}

        # --- Text engine endpoints ---

        class TextRequest(BaseModel):
            message: Optional[str] = None  # None clears custom message, restores default rotation
            source: int = 1                 # 1 = src_1, 2 = src_2

        def _get_text_engine(source: int):
            from animations.text_engine import TextEngine
            from animations.enums import AnimSource
            anim_map = (self.mixer.src_1_animations if source == 1
                        else self.mixer.src_2_animations)
            return anim_map.get(AnimSource.TEXT_ENGINE.name)

        @self.app.put("/text")
        async def set_text(req: TextRequest):
            """Set or clear the ticker message on a TextEngine animation source."""
            if self.mixer is None:
                raise HTTPException(status_code=503, detail="Mixer not available")
            engine = _get_text_engine(req.source)
            if engine is None:
                raise HTTPException(status_code=503, detail="TextEngine not found — is TEXT_ENGINE the active source?")
            engine.set_message(req.message)
            return {"success": True, "message": req.message, "source": req.source}

        @self.app.get("/text")
        async def get_text(source: int = 1):
            """Get the current message displayed by a TextEngine animation source."""
            if self.mixer is None:
                raise HTTPException(status_code=503, detail="Mixer not available")
            engine = _get_text_engine(source)
            if engine is None:
                raise HTTPException(status_code=503, detail="TextEngine not found")
            return {"message": engine.get_message(), "source": source}

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
        if self._server_thread is not None and self._server_thread.is_alive():
            log.warning("API server already running")
            return

        self._should_stop = False
        # Use a controllable uvicorn.Server (rather than uvicorn.run) so the
        # server can be shut down cleanly and later restarted / rebound to a
        # different host — required by the GUI Start/Stop and LAN controls.
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)

        def run_server():
            log.info(f"Starting API server on {self.host}:{self.port}")
            self._uvicorn_server.run()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        log.info(f"API server started at http://{self.host}:{self.port}")
        log.info(f"API docs available at http://{self.host}:{self.port}/docs")
        if self._web_ui:
            log.info(f"Web UI available at http://{self.host}:{self.port}/ui")

    def stop(self):
        """Stop the API server and release the port (blocks until shut down)."""
        self._should_stop = True
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._server_thread = None
        self._uvicorn_server = None
        log.info("API server stopped")
