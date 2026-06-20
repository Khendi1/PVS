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
import math
import logging

from animations.base import Animation
from animations.enums import MoirePattern, MoireBlend
from common import Widget

log = logging.getLogger(__name__)

class Moire(Animation):
    def __init__(self, params, width=800, height=600, group=None, oscs=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        self._oscs = oscs
        
        self.blend_mode = params.new("moire_blend",
                                      min=0, max=len(MoireBlend)-1, default=0,
                                      group=group, subgroup=subgroup,
                                      type=Widget.DROPDOWN, options=MoireBlend,
                                      info="How the two layers are composited (e.g. multiply, XOR)")

        center_x, center_y = self.width//2, self.height//2

        self.pattern_1 = params.new("moire_type_1",
                                    min=0, max=len(MoirePattern)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=MoirePattern,
                                    info="Shape of layer 1 (lines, circles, dots, etc.)")
        self.freq_1 = params.new("spatial_freq_1",
                                 min=0.01, max=25, default=10.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER,
                                 info="Spatial frequency (line density) of layer 1")
        self.angle_1 = params.new("angle_1",
                                  min=0, max=360, default=90.0,
                                  group=group, subgroup=subgroup,
                                  type=Widget.SLIDER,
                                  info="Rotation angle of layer 1 (degrees)")
        self.zoom_1 = params.new("zoom_1",
                                 min=0.05, max=1.5, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER,
                                 info="Zoom level of layer 1")
        self.center_x_1 = params.new("moire_center_x_1",
                                     min=0, max=self.width, default=center_x,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER,
                                     info="Horizontal center point of layer 1")
        self.center_y_1 = params.new("moire_center_y_1",
                                     min=0, max=self.height, default=center_y,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER,
                                     info="Vertical center point of layer 1")

        self.pattern_2 = params.new("moire_type_2",
                                    min=0, max=len(MoirePattern)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=MoirePattern,
                                    info="Shape of layer 2")
        self.freq_2 = params.new("spatial_freq_2",
                                 min=0.01, max=25, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER,
                                 info="Spatial frequency of layer 2")
        self.angle_2 = params.new("angle_2",
                                  min=0, max=360, default=0.0,
                                  group=group, subgroup=subgroup,
                                  type=Widget.SLIDER,
                                  info="Rotation angle of layer 2 (degrees)")
        self.zoom_2 = params.new("zoom_2",
                                 min=0.05, max=1.5, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER,
                                 info="Zoom level of layer 2")
        self.center_x_2 = params.new("moire_center_x_2",
                                     min=0, max=self.width, default=center_x,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER,
                                     info="Horizontal center point of layer 2")
        self.center_y_2 = params.new("moire_center_y_2",
                                     min=0, max=self.height, default=center_y,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER,
                                     info="Vertical center point of layer 2")

        # Performance optimization: cache meshgrid (created once, reused every frame)
        self._X, self._Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Auto-link LFOs to rotation params
        if self._oscs is not None:
            self._link_rotation_lfos(p_name)

    def _link_rotation_lfos(self, prefix):
        """Create two LFO oscillators and link them to the angle params."""
        try:
            osc1 = self._oscs.add_oscillator(
                name=f"{prefix}_moire_rot1",
                frequency=0.1, amplitude=1.0, phase=0.0, shape=1
            )
            osc1.link_param(self.angle_1)
            self.angle_1.linked_oscillator = osc1

            osc2 = self._oscs.add_oscillator(
                name=f"{prefix}_moire_rot2",
                frequency=0.07, amplitude=1.0, phase=90.0, shape=1
            )
            osc2.link_param(self.angle_2)
            self.angle_2.linked_oscillator = osc2

            log.info(f"Auto-linked LFOs to moire rotation params ({prefix})")
        except Exception as e:
            log.warning(f"Could not auto-link LFOs to moire rotation: {e}")

    def _generate_single_pattern(self, X_shifted, Y_shifted, frequency, angle_rad, zoom, pattern_type):
        angle_rad = math.radians(angle_rad)
        X_z, Y_z = X_shifted * zoom, Y_shifted * zoom
        SCALE_FACTOR = 127.5
        needs_half_scale = False

        if pattern_type == MoirePattern.LINE.value:
            P = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            pattern = np.sin(P * frequency / 2)
        elif pattern_type == MoirePattern.RADIAL.value:
            R = np.sqrt(X_z**2 + Y_z**2)
            pattern = np.sin(R * frequency / 2)
        elif pattern_type == MoirePattern.GRID.value:
            freq_x = frequency * (1.0 + np.sin(angle_rad) * .01)
            freq_y = frequency * (1.0 + np.cos(angle_rad) * .01)
            pattern = np.sin(X_z * freq_x) + np.sin(Y_z * freq_y)
            needs_half_scale = True
        elif pattern_type == MoirePattern.SPIRAL.value:
            R = np.sqrt(X_z**2 + Y_z**2)
            theta = np.arctan2(Y_z, X_z)
            pattern = np.sin(R * frequency / 2 + theta * frequency)
        elif pattern_type == MoirePattern.DIAMOND.value:
            X_r = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            Y_r = -X_z * np.sin(angle_rad) + Y_z * np.cos(angle_rad)
            D = np.abs(X_r) + np.abs(Y_r)
            pattern = np.sin(D * frequency / 2)
        elif pattern_type == MoirePattern.CHECKERBOARD.value:
            X_r = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            Y_r = -X_z * np.sin(angle_rad) + Y_z * np.cos(angle_rad)
            pattern = np.sign(np.sin(X_r * frequency / 2)) * np.sign(np.sin(Y_r * frequency / 2))
        elif pattern_type == MoirePattern.HEXAGONAL.value:
            X_r = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            Y_r = -X_z * np.sin(angle_rad) + Y_z * np.cos(angle_rad)
            pattern = (np.sin(X_r * frequency / 2)
                       + np.sin((X_r * 0.5 + Y_r * 0.866) * frequency / 2)
                       + np.sin((-X_r * 0.5 + Y_r * 0.866) * frequency / 2))
            needs_half_scale = True
        else:
            pattern = np.zeros_like(X_shifted)

        scale = SCALE_FACTOR / (2 if needs_half_scale else 1)
        return (pattern * scale) + SCALE_FACTOR

    def get_frame(self, frame):
        # Use cached meshgrid instead of creating new one every frame
        x1, y1 = self._X - self.center_x_1.value, self._Y - self.center_y_1.value
        x2, y2 = self._X - self.center_x_2.value, self._Y - self.center_y_2.value

        pattern1_float = self._generate_single_pattern(
            x1, y1, self.freq_1.value, self.angle_1.value, self.zoom_1.value, self.pattern_1.value
        )
        pattern2_float = self._generate_single_pattern(
            x2, y2, self.freq_2.value, self.angle_2.value, self.zoom_2.value, self.pattern_2.value
        )

        blend_mode = self.blend_mode.value
        if blend_mode == MoireBlend.MULTIPLY.value:
            combined_pattern = (pattern1_float / 255.0) * (pattern2_float / 255.0)
            moire_image = (combined_pattern * 255).astype(np.uint8)
        elif blend_mode == MoireBlend.ADD.value:
            combined_pattern = pattern1_float + pattern2_float
            moire_image = np.clip(combined_pattern, 0, 255).astype(np.uint8)
        elif blend_mode == MoireBlend.SUB.value:
            combined_pattern = pattern1_float - pattern2_float
            moire_image = np.clip(combined_pattern, 0, 255).astype(np.uint8)
        else:
            moire_image = pattern1_float.astype(np.uint8)

        moire_image = cv2.equalizeHist(moire_image)
        return cv2.cvtColor(moire_image, cv2.COLOR_GRAY2BGR)
