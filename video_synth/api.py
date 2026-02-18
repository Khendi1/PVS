"""
FastAPI server for remote control of the video synthesizer.
Allows an agent or external application to control parameters via HTTP REST API.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

    def __init__(self, params, mixer=None, host="127.0.0.1", port=8000):
        """
        Initialize API server.

        Args:
            params: ParamTable instance containing all parameters
            mixer: Mixer instance (optional, for mixer-specific controls)
            host: Host to bind to (default: localhost)
            port: Port to bind to (default: 8000)
        """
        self.params = params
        self.mixer = mixer
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Video Synth API",
            description="API for controlling video synthesizer parameters",
            version="1.0.0"
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
            for name, param in self.params.table.items():
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
            if param_name not in self.params.table:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.table[param_name]
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
            if param_name not in self.params.table:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.table[param_name]

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
            if param_name not in self.params.table:
                raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found")

            param = self.params.table[param_name]
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

            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', self.mixer.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame")

            return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

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
