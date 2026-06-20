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

import cv2
import numpy as np
import logging

from animations.base import Animation
from common import Widget

log = logging.getLogger(__name__)

# mss is imported lazily so startup doesn't fail if it's not installed
try:
    import mss
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False
    log.warning("mss not installed — ScreenCapture source unavailable. Run: pip install mss")


class ScreenCapture(Animation):
    """
    Captures the local desktop (or a sub-region) as a live video source.

    Uses mss for low-overhead screen grabbing (~2–8 ms/frame depending on
    capture area size). Output is resized to the mixer resolution.

    Target render budget: ≤ 8 ms at 640×480 with default full-monitor capture.
    """

    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.monitor_index = params.new(
            "sc_monitor",
            min=0, max=8, default=1,
            subgroup=subgroup, group=group,
            info="Monitor to capture (1 = primary; 0 = virtual full-desktop across all monitors)",
        )
        # Region offset and size as fractions of the monitor dimensions (0.0–1.0),
        # so they stay valid across different screen resolutions.
        self.region_x = params.new(
            "sc_region_x",
            min=0.0, max=1.0, default=0.0,
            subgroup=subgroup, group=group,
            info="Left edge of capture region as a fraction of monitor width",
        )
        self.region_y = params.new(
            "sc_region_y",
            min=0.0, max=1.0, default=0.0,
            subgroup=subgroup, group=group,
            info="Top edge of capture region as a fraction of monitor height",
        )
        self.region_w = params.new(
            "sc_region_w",
            min=0.05, max=1.0, default=1.0,
            subgroup=subgroup, group=group,
            info="Width of capture region as a fraction of monitor width",
        )
        self.region_h = params.new(
            "sc_region_h",
            min=0.05, max=1.0, default=1.0,
            subgroup=subgroup, group=group,
            info="Height of capture region as a fraction of monitor height",
        )
        self.zoom = params.new(
            "sc_zoom",
            min=0.1, max=4.0, default=1.0,
            subgroup=subgroup, group=group,
            info="Scale factor applied after capture; >1 zooms in (crops centre), <1 zooms out (adds black border)",
        )
        self.flip_h = params.new(
            "sc_flip_h",
            min=0, max=1, default=0,
            subgroup=subgroup, group=group,
            type=Widget.TOGGLE,
            info="Flip frame horizontally",
        )
        self.flip_v = params.new(
            "sc_flip_v",
            min=0, max=1, default=0,
            subgroup=subgroup, group=group,
            type=Widget.TOGGLE,
            info="Flip frame vertically",
        )

        self._sct = None
        self._monitors = []
        self._black = np.zeros((height, width, 3), dtype=np.uint8)
        self._init_mss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_mss(self):
        if not _MSS_AVAILABLE:
            return
        try:
            self._sct = mss.mss()
            self._monitors = self._sct.monitors  # index 0 = all, 1+ = individual
        except Exception as exc:
            log.error("ScreenCapture: failed to initialise mss: %s", exc)
            self._sct = None

    def _get_monitor_rect(self):
        """Return the mss monitor dict for the current monitor_index param."""
        idx = int(self.monitor_index.value)
        if not self._monitors:
            return None
        idx = max(0, min(idx, len(self._monitors) - 1))
        mon = self._monitors[idx]

        # Apply fractional region
        mon_w = mon["width"]
        mon_h = mon["height"]
        rx = self.region_x.value
        ry = self.region_y.value
        rw = max(0.05, self.region_w.value)
        rh = max(0.05, self.region_h.value)

        left   = mon["left"] + int(rx * mon_w)
        top    = mon["top"]  + int(ry * mon_h)
        width  = max(1, int(rw * mon_w))
        height = max(1, int(rh * mon_h))

        # clamp to monitor bounds
        width  = min(width,  mon["left"] + mon_w - left)
        height = min(height, mon["top"]  + mon_h - top)

        return {"left": left, "top": top, "width": width, "height": height}

    # ------------------------------------------------------------------
    # Main frame method
    # ------------------------------------------------------------------

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        if self._sct is None:
            if _MSS_AVAILABLE:
                self._init_mss()
            return self._black.copy()

        rect = self._get_monitor_rect()
        if rect is None:
            return self._black.copy()

        try:
            shot = self._sct.grab(rect)
        except Exception as exc:
            log.warning("ScreenCapture: grab failed: %s", exc)
            return self._black.copy()

        # mss returns BGRA — drop alpha channel
        img = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        img = img[:, :, :3]  # BGRA → BGR

        # Apply zoom by cropping the centre
        zoom = float(self.zoom.value)
        if zoom != 1.0:
            h, w = img.shape[:2]
            if zoom > 1.0:
                crop_w = max(1, int(w / zoom))
                crop_h = max(1, int(h / zoom))
                x0 = (w - crop_w) // 2
                y0 = (h - crop_h) // 2
                img = img[y0:y0 + crop_h, x0:x0 + crop_w]
            else:
                # zoom out: embed in black canvas
                embed_w = int(w * zoom)
                embed_h = int(h * zoom)
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                x0 = (w - embed_w) // 2
                y0 = (h - embed_h) // 2
                small = cv2.resize(img, (embed_w, embed_h), interpolation=cv2.INTER_LINEAR)
                canvas[y0:y0 + embed_h, x0:x0 + embed_w] = small
                img = canvas

        # Resize to mixer output dimensions
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Optional flips
        if int(self.flip_h.value):
            img = cv2.flip(img, 1)
        if int(self.flip_v.value):
            img = cv2.flip(img, 0)

        return img
