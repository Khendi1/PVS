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
Phase-Coupled Oscillators (Kuramoto model) - A grid of oscillators that
slowly synchronize and desynchronize in clusters. Visualized as color phase,
you see waves of synchronization ripple across the screen.
"""

import cv2
import numpy as np

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class OscillatorGrid(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.osc_coupling = params.new("osc_coupling",
                                        min=0.0, max=2.0, default=0.5,
                                        subgroup=subgroup, group=group,
                                        info="Strength of phase coupling between neighboring oscillators")
        self.osc_noise = params.new("osc_noise",
                                     min=0.0, max=0.5, default=0.05,
                                     subgroup=subgroup, group=group,
                                     info="Random perturbation added to each oscillator per frame")
        self.osc_freq_spread = params.new("osc_freq_spread",
                                           min=0.0, max=2.0, default=0.5,
                                           subgroup=subgroup, group=group,
                                           info="Variance in natural frequencies across the grid")
        self.osc_speed = params.new("osc_speed",
                                     min=0.1, max=5.0, default=1.0,
                                     subgroup=subgroup, group=group,
                                     info="Overall simulation speed multiplier")
        self.osc_colormap = params.new("osc_colormap",
                                        min=0, max=len(COLORMAP_OPTIONS)-1,
                                        default=int(Colormap.HSV),
                                        subgroup=subgroup, group=group,
                                        type=Widget.DROPDOWN, options=Colormap,
                                        info="Color palette for phase visualization")
        self.osc_grid_size = params.new("osc_grid_size",
                                         min=32, max=256, default=80,
                                         subgroup=subgroup, group=group,
                                         info="Number of oscillators per side (N×N grid)")

        self._init_grid()

    def _init_grid(self):
        """Initialize oscillator phases and natural frequencies."""
        n = int(self.osc_grid_size.value)
        self._n = n
        # Random initial phases [0, 2pi)
        self.phases = np.random.uniform(0, 2 * np.pi, (n, n)).astype(np.float32)
        # Natural frequencies with some spread
        self.nat_freq = np.random.normal(1.0, 0.3, (n, n)).astype(np.float32)

    def get_frame(self, frame=None):
        n = int(self.osc_grid_size.value)
        if n != self._n:
            self._init_grid()

        dt = 0.016 * self.osc_speed.value
        coupling = self.osc_coupling.value
        noise_amp = self.osc_noise.value

        # Natural frequency with spread
        freq = self.nat_freq * (1.0 + self.osc_freq_spread.value)

        # Compute mean phase of 4-connected neighbors (with wrapping)
        sin_phases = np.sin(self.phases)
        cos_phases = np.cos(self.phases)

        # Sum of neighbor sin/cos (toroidal boundary)
        sin_sum = (np.roll(sin_phases, 1, 0) + np.roll(sin_phases, -1, 0) +
                   np.roll(sin_phases, 1, 1) + np.roll(sin_phases, -1, 1))
        cos_sum = (np.roll(cos_phases, 1, 0) + np.roll(cos_phases, -1, 0) +
                   np.roll(cos_phases, 1, 1) + np.roll(cos_phases, -1, 1))

        # Kuramoto coupling: K/N * sum(sin(theta_j - theta_i))
        # = K/4 * (cos_i * sin_sum - sin_i * cos_sum)  [from trig identity]
        coupling_term = (coupling / 4.0) * (
            cos_phases * sin_sum - sin_phases * cos_sum
        )

        # Noise
        noise = noise_amp * np.random.normal(0, 1, self.phases.shape).astype(np.float32)

        # Update phases
        self.phases += dt * (freq + coupling_term + noise)
        self.phases = self.phases % (2 * np.pi)

        # Visualize: map phase to color via colormap
        display = ((self.phases / (2 * np.pi)) * 255).astype(np.uint8)

        cmap_idx = int(self.osc_colormap.value)
        colored = cv2.applyColorMap(display, COLORMAP_OPTIONS[cmap_idx])

        # Upscale
        if colored.shape[:2] != (self.height, self.width):
            colored = cv2.resize(colored, (self.width, self.height),
                                 interpolation=cv2.INTER_LINEAR)

        return colored
