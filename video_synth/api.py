"""
FastAPI server for remote control of the video synthesizer.
Allows an agent or external application to control parameters via HTTP REST API.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import uvicorn
import io
import cv2

log = logging.getLogger(__name__)


class ParamValue(BaseModel):
    """Request model for setting a parameter value."""
    value: float | int


class ParamInfo(BaseModel):
    """Response model for parameter information."""
    name: str
    value: float | int
    min: float | int
    max: float | int
    default: float | int
    group: str
    subgroup: str
    type: str


class APIServer:
    """FastAPI server for controlling video synthesizer parameters."""

    def __init__(self, params, mixer=None, save_controller=None, midi_mapper=None, host="127.0.0.1", port=8000):
        self.params = params
        self.mixer = mixer
        self.save_controller = save_controller
        self.midi_mapper = midi_mapper
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Video Synth API",
            description="API for controlling video synthesizer parameters",
            version="1.0.0"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()
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

        @self.app.get("/params", response_model=List[ParamInfo])
        async def list_params():
            """Get list of all parameters with their current values."""
            result = []
            for name, param in self.params.params.items():
                result.append(ParamInfo(
                    name=name,
                    value=param.value,
                    min=param.min,
                    max=param.max,
                    default=param.default,
                    group=str(param.group),
                    subgroup=str(param.subgroup),
                    type=str(param.type)
                ))
            return result

        @self.app.get("/params/{param_name}", response_model=ParamInfo)
        async def get_param(param_name: str):
            """Get specific parameter info and current value."""
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.params[param_name]
            return ParamInfo(
                name=param_name,
                value=param.value,
                min=param.min,
                max=param.max,
                default=param.default,
                group=str(param.group),
                subgroup=str(param.subgroup),
                type=str(param.type)
            )

        @self.app.put("/params/{param_name}")
        async def set_param(param_name: str, param_value: ParamValue):
            """Set a parameter value."""
            if param_name not in self.params.params:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.params[param_name]

            # Validate value is within bounds
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

    def stop(self):
        """Stop the API server."""
        self._should_stop = True
        log.info("API server stopped")
