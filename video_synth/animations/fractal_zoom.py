"""
Fractal Zoom - Continuous slow zoom into a Julia set with the Julia constant
slowly orbiting. The deeper you go, the more structures emerge. The parameter
drift means you never see the same thing twice.
"""

import cv2
import numpy as np
import time

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class FractalZoom(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.fractal_zoom_speed = params.new("fractal_zoom_speed",
                                              min=0.0, max=1.0, default=0.1,
                                              subgroup=subgroup, group=group)
        self.fractal_drift_speed = params.new("fractal_drift_speed",
                                               min=0.0, max=1.0, default=0.2,
                                               subgroup=subgroup, group=group)
        self.fractal_max_iter = params.new("fractal_max_iter",
                                            min=20, max=200, default=64,
                                            subgroup=subgroup, group=group)
        self.fractal_color_speed = params.new("fractal_color_speed",
                                               min=0.0, max=2.0, default=0.5,
                                               subgroup=subgroup, group=group)
        self.fractal_colormap = params.new("fractal_colormap",
                                            min=0, max=len(COLORMAP_OPTIONS)-1,
                                            default=int(Colormap.TWILIGHT_SHIFTED),
                                            subgroup=subgroup, group=group,
                                            type=Widget.DROPDOWN, options=Colormap)

        # Work at reduced resolution
        self.rw = width // 2
        self.rh = height // 2

        self._time = 0.0
        self._zoom_level = 1.0
        self._center_x = -0.5
        self._center_y = 0.0

    def _compute_julia(self, cx, cy, max_iter):
        """Compute Julia set at current zoom/center."""
        # Build coordinate grid in complex plane
        scale = 3.0 / self._zoom_level
        x_min = self._center_x - scale * 0.5
        x_max = self._center_x + scale * 0.5
        y_min = self._center_y - scale * 0.5 * (self.rh / self.rw)
        y_max = self._center_y + scale * 0.5 * (self.rh / self.rw)

        x = np.linspace(x_min, x_max, self.rw, dtype=np.float64)
        y = np.linspace(y_min, y_max, self.rh, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        C = complex(cx, cy)

        iterations = np.zeros(Z.shape, dtype=np.float32)
        mask = np.ones(Z.shape, dtype=bool)

        for i in range(int(max_iter)):
            Z[mask] = Z[mask] * Z[mask] + C
            diverged = np.abs(Z) > 4.0
            newly_diverged = diverged & mask
            # Smooth iteration count
            iterations[newly_diverged] = i + 1 - np.log2(np.log2(np.abs(Z[newly_diverged]) + 1e-10))
            mask &= ~diverged

        return iterations

    def get_frame(self, frame=None):
        self._time += 0.016

        zoom_speed = self.fractal_zoom_speed.value
        drift_speed = self.fractal_drift_speed.value
        max_iter = int(self.fractal_max_iter.value)

        # Slowly zoom in
        self._zoom_level *= (1.0 + zoom_speed * 0.01)

        # Reset zoom if it gets too deep (precision loss)
        if self._zoom_level > 1e8:
            self._zoom_level = 1.0

        # Slowly drift the center toward interesting regions
        t = self._time * drift_speed * 0.3
        self._center_x = -0.5 + 0.3 * np.sin(t * 0.7)
        self._center_y = 0.3 * np.cos(t * 0.5)

        # Julia constant orbits slowly
        jt = self._time * drift_speed * 0.5
        cx = 0.355 + 0.1 * np.sin(jt * 0.3)
        cy = 0.355 + 0.1 * np.cos(jt * 0.4)

        iterations = self._compute_julia(cx, cy, max_iter)

        # Color cycling
        color_offset = self._time * self.fractal_color_speed.value * 10
        display = ((iterations + color_offset) % max_iter) / max_iter * 255
        display = display.astype(np.uint8)

        cmap_idx = int(self.fractal_colormap.value)
        colored = cv2.applyColorMap(display, COLORMAP_OPTIONS[cmap_idx])

        # Upscale
        if colored.shape[:2] != (self.height, self.width):
            colored = cv2.resize(colored, (self.width, self.height),
                                 interpolation=cv2.INTER_LINEAR)

        return colored
