"""
Drift Fields - A 2D vector field that slowly advects color through it.
Like ink dropped into slowly stirring water. The field is driven by
layered Perlin noise with very low frequencies, creating currents
that shift over minutes rather than seconds.
"""

import cv2
import numpy as np
import time

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class DriftField(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.drift_speed = params.new("drift_speed",
                                       min=0.01, max=2.0, default=0.15,
                                       subgroup=subgroup, group=group)
        self.drift_complexity = params.new("drift_complexity",
                                            min=1, max=8, default=3,
                                            subgroup=subgroup, group=group)
        self.drift_scale = params.new("drift_scale",
                                       min=0.5, max=10.0, default=3.0,
                                       subgroup=subgroup, group=group)
        self.drift_viscosity = params.new("drift_viscosity",
                                           min=0.9, max=1.0, default=0.995,
                                           subgroup=subgroup, group=group)
        self.drift_injection = params.new("drift_injection",
                                           min=0.0, max=1.0, default=0.02,
                                           subgroup=subgroup, group=group)
        self.drift_colormap = params.new("drift_colormap",
                                          min=0, max=len(COLORMAP_OPTIONS)-1, default=int(Colormap.TWILIGHT),
                                          subgroup=subgroup, group=group,
                                          type=Widget.DROPDOWN, options=Colormap)
        self.drift_color_speed = params.new("drift_color_speed",
                                             min=0.0, max=2.0, default=0.3,
                                             subgroup=subgroup, group=group)

        # Work at half resolution for performance
        self.rw = width // 2
        self.rh = height // 2

        # Scalar field that gets advected (represents "dye concentration")
        self.field = np.random.uniform(0, 1, (self.rh, self.rw)).astype(np.float32)

        # Precompute coordinate grids
        y = np.linspace(0, 1, self.rh, dtype=np.float32)
        x = np.linspace(0, 1, self.rw, dtype=np.float32)
        self._X, self._Y = np.meshgrid(x, y)

        self._time = 0.0

    def _vector_field(self, t):
        """Generate a time-varying 2D vector field from layered sine waves."""
        X, Y = self._X, self._Y
        scale = self.drift_scale.value
        layers = int(self.drift_complexity.value)

        vx = np.zeros_like(X)
        vy = np.zeros_like(Y)

        for i in range(1, layers + 1):
            freq = i * scale
            phase = t * (0.3 + i * 0.1)
            amp = 1.0 / i

            # Curl noise pattern: take derivatives of a scalar potential
            # to get a divergence-free (incompressible) flow
            vx += amp * np.cos(freq * Y + phase) * np.sin(freq * X * 0.7 + phase * 1.3)
            vy += amp * -np.sin(freq * X + phase * 0.8) * np.cos(freq * Y * 0.7 + phase * 0.6)

        return vx, vy

    def get_frame(self, frame=None):
        dt = 0.016  # ~60fps timestep
        speed = self.drift_speed.value
        self._time += dt * speed

        # Get vector field
        vx, vy = self._vector_field(self._time)

        # Advect the scalar field using the vector field
        # Semi-Lagrangian advection: trace back where each pixel came from
        displacement = speed * 2.0
        src_x = self._X - vx * dt * displacement
        src_y = self._Y - vy * dt * displacement

        # Convert to pixel coordinates for remapping
        map_x = (src_x * (self.rw - 1)).astype(np.float32)
        map_y = (src_y * (self.rh - 1)).astype(np.float32)

        # Remap (advect)
        self.field = cv2.remap(self.field, map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        # Viscosity (slow decay toward mean)
        viscosity = self.drift_viscosity.value
        self.field *= viscosity

        # Inject new dye at random spots
        injection = self.drift_injection.value
        if injection > 0 and np.random.random() < injection * 5:
            cx = np.random.randint(self.rw // 4, 3 * self.rw // 4)
            cy = np.random.randint(self.rh // 4, 3 * self.rh // 4)
            radius = np.random.randint(5, 20)
            cv2.circle(self.field, (cx, cy), radius, 1.0, -1)

        # Color cycling offset - use modulo so it wraps instead of saturating
        color_offset = self._time * self.drift_color_speed.value * 50

        # Map field to [0,255] then add cyclic offset, wrapping with modulo
        display = ((self.field * 255.0 + color_offset) % 256).astype(np.uint8)

        cmap_idx = int(self.drift_colormap.value)
        colored = cv2.applyColorMap(display, COLORMAP_OPTIONS[cmap_idx])

        # Upscale to full resolution
        if colored.shape[:2] != (self.height, self.width):
            colored = cv2.resize(colored, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return colored
