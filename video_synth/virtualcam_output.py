"""
Virtual camera output module for video synthesizer.
Outputs frames directly to a virtual webcam device via pyvirtualcam.
Any application (OBS, Zoom, etc.) can capture this as a regular camera source.
"""

import logging
import cv2
import numpy as np

log = logging.getLogger(__name__)


class VirtualCamOutput:
    """Outputs frames to a virtual camera device using pyvirtualcam."""

    def __init__(self, width: int, height: int, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.frame_count = 0

    def start(self):
        """Start the virtual camera device."""
        if self.cam is not None:
            log.warning("Virtual camera already running")
            return

        try:
            import pyvirtualcam
            self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
            log.info(f"Virtual camera started: {self.cam.device} ({self.width}x{self.height} @ {self.fps}fps)")
        except ImportError:
            log.error("pyvirtualcam not installed. Install with: pip install pyvirtualcam")
            self.cam = None
        except Exception as e:
            log.error(f"Failed to start virtual camera: {e}")
            self.cam = None

    def write_frame(self, frame: np.ndarray):
        """
        Send a frame to the virtual camera.

        Args:
            frame: numpy array (height, width, 3) in BGR format (OpenCV format)
        """
        if self.cam is None:
            return

        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # pyvirtualcam expects RGB, OpenCV uses BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cam.send(rgb_frame)
            self.cam.sleep_until_next_frame()
            self.frame_count += 1

            if self.frame_count % 300 == 0:
                log.info(f"VirtualCam: {self.frame_count} frames sent")

        except Exception as e:
            log.error(f"Error sending frame to virtual camera: {e}")

    def stop(self):
        """Stop the virtual camera device."""
        if self.cam is None:
            return

        log.info(f"Stopping virtual camera after {self.frame_count} frames")
        try:
            self.cam.close()
        except Exception as e:
            log.error(f"Error stopping virtual camera: {e}")
        finally:
            self.cam = None
            log.info("Virtual camera stopped")
