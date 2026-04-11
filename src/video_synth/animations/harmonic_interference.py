"""
Harmonic Interference Textures - Layer multiple sine gratings at slightly
different frequencies and orientations, all drifting at different speeds.
The interference patterns create slowly morphing textures that never repeat.
"""

import cv2
import numpy as np
import time

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class HarmonicInterference(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.hi_num_layers = params.new("hi_num_layers",
                                         min=2, max=8, default=5,
                                         subgroup=subgroup, group=group)
        self.hi_base_freq = params.new("hi_base_freq",
                                        min=0.5, max=20.0, default=4.0,
                                        subgroup=subgroup, group=group)
        self.hi_freq_spread = params.new("hi_freq_spread",
                                          min=0.0, max=2.0, default=0.5,
                                          subgroup=subgroup, group=group)
        self.hi_drift_speed = params.new("hi_drift_speed",
                                          min=0.0, max=2.0, default=0.3,
                                          subgroup=subgroup, group=group)
        self.hi_rotation_speed = params.new("hi_rotation_speed",
                                             min=0.0, max=1.0, default=0.1,
                                             subgroup=subgroup, group=group)
        self.hi_color_speed = params.new("hi_color_speed",
                                          min=0.0, max=2.0, default=0.2,
                                          subgroup=subgroup, group=group)
        self.hi_colormap = params.new("hi_colormap",
                                       min=0, max=len(COLORMAP_OPTIONS)-1,
                                       default=int(Colormap.TWILIGHT),
                                       subgroup=subgroup, group=group,
                                       type=Widget.DROPDOWN, options=Colormap)

        # Precompute normalized coordinate grids [-1, 1]
        x = np.linspace(-1, 1, width, dtype=np.float32)
        y = np.linspace(-1, 1, height, dtype=np.float32)
        self._X, self._Y = np.meshgrid(x, y)

        # Random per-layer offsets for variety (fixed at init)
        self._layer_angles = np.random.uniform(0, np.pi, 8).astype(np.float32)
        self._layer_phases = np.random.uniform(0, 2 * np.pi, 8).astype(np.float32)

        self._time = 0.0

    def get_frame(self, frame=None):
        self._time += 0.016

        num_layers = int(self.hi_num_layers.value)
        base_freq = self.hi_base_freq.value
        freq_spread = self.hi_freq_spread.value
        drift_speed = self.hi_drift_speed.value
        rot_speed = self.hi_rotation_speed.value
        color_speed = self.hi_color_speed.value

        t = self._time
        X, Y = self._X, self._Y

        # Accumulate interference from all layers
        result = np.zeros_like(X)

        for i in range(num_layers):
            # Each layer has slightly different frequency
            freq = base_freq * (1.0 + freq_spread * (i - num_layers / 2) / num_layers)

            # Slowly rotating angle for this layer
            angle = self._layer_angles[i] + t * rot_speed * (0.3 + i * 0.15)

            # Phase drifts over time
            phase = self._layer_phases[i] + t * drift_speed * (0.5 + i * 0.2)

            # Rotated coordinates
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = X * cos_a + Y * sin_a

            # Add sine grating
            result += np.sin(rotated * freq * np.pi + phase)

        # Normalize to [0, 1]
        result = (result / num_layers + 1.0) * 0.5

        # Color cycling
        color_offset = t * color_speed * 50
        display = ((result * 200 + color_offset) % 256).astype(np.uint8)

        cmap_idx = int(self.hi_colormap.value)
        colored = cv2.applyColorMap(display, COLORMAP_OPTIONS[cmap_idx])

        return colored
