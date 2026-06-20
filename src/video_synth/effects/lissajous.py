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

from effects.base import EffectBase

class Lissajous(EffectBase):
    def __init__(self, params, width=640, height=480, group=None):
        self.params = params
        self.width = width
        self.height = height
        self.t = 0
        subgroup = self.__class__.__name__

        # Amplitude controls (as percentage of frame size)
        self.amplitude_x = params.new("lissajous_amp_x",
                                      min=0.0, max=1.0, default=0.4,
                                      subgroup=subgroup, group=group,
                                      info="Horizontal amplitude as a fraction of canvas size")
        self.amplitude_y = params.new("lissajous_amp_y",
                                      min=0.0, max=1.0, default=0.4,
                                      subgroup=subgroup, group=group,
                                      info="Vertical amplitude as a fraction of canvas size")

        # Frequency ratio controls (these create the pattern shape)
        self.freq_x = params.new("lissajous_freq_x",
                                 min=1, max=12, default=3,
                                 subgroup=subgroup, group=group,
                                 info="Horizontal frequency ratio")
        self.freq_y = params.new("lissajous_freq_y",
                                 min=1, max=12, default=2,
                                 subgroup=subgroup, group=group,
                                 info="Vertical frequency ratio")

        # Phase offset (controls pattern rotation/shape)
        self.phase = params.new("lissajous_phase",
                                min=0.0, max=1.0, default=0.25,
                                subgroup=subgroup, group=group,
                                info="Phase offset between X and Y oscillations (as fraction of 2π)")

        # Animation speed
        self.speed = params.new("lissajous_speed",
                                min=0.0, max=2.0, default=0.5,
                                subgroup=subgroup, group=group,
                                info="Rate at which the phase evolves")

        # Visual parameters
        self.num_points = params.new("lissajous_points",
                                     min=100, max=5000, default=1000,
                                     subgroup=subgroup, group=group,
                                     info="Number of points sampled along the curve")
        self.line_mode = params.new("lissajous_line_mode",
                                    min=0, max=1, default=1,
                                    subgroup=subgroup, group=group,
                                    info="Whether to draw continuous lines or individual dots")
        self.thickness = params.new("lissajous_thickness",
                                    min=1, max=10, default=2,
                                    subgroup=subgroup, group=group,
                                    info="Stroke width of the curve")

        # Color controls
        self.hue_start = params.new("lissajous_hue_start",
                                    min=0, max=180, default=0,
                                    subgroup=subgroup, group=group,
                                    info="Starting hue when not in rainbow mode")
        self.hue_range = params.new("lissajous_hue_range",
                                    min=0, max=180, default=60,
                                    subgroup=subgroup, group=group,
                                    info="Range of hues spanned along the curve")
        self.saturation = params.new("lissajous_saturation",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group,
                                     info="Saturation of the curve color")
        self.brightness = params.new("lissajous_brightness",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group,
                                     info="Brightness of the curve color")
        self.rainbow_mode = params.new("lissajous_rainbow",
                                       min=0, max=1, default=1,
                                       subgroup=subgroup, group=group,
                                       info="Colors cycle through the full hue wheel along the curve")

        # Second harmonic for more complex patterns
        self.harmonic_strength = params.new("lissajous_harmonic",
                                            min=0.0, max=1.0, default=0.0,
                                            subgroup=subgroup, group=group,
                                            info="Adds a secondary harmonic frequency to the curve")
        self.harmonic_freq = params.new("lissajous_harm_freq",
                                        min=2, max=8, default=3,
                                        subgroup=subgroup, group=group,
                                        info="Frequency multiplier for the harmonic component")

    def lissajous_pattern(self, frame):
        self.t += 0.02 * self.speed.value

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        amp_x = self.amplitude_x.value * width / 2
        amp_y = self.amplitude_y.value * height / 2

        num_pts = self.num_points.value
        phase_offset = self.phase.value * 2 * math.pi

        points = []
        colors = []

        for i in range(num_pts):
            # Parameter along the curve
            s = (i / num_pts) * 2 * math.pi

            # Primary Lissajous calculation
            x = amp_x * math.sin(self.freq_x.value * s + self.t)
            y = amp_y * math.sin(self.freq_y.value * s + self.t + phase_offset)

            # Add harmonic for more complex shapes
            if self.harmonic_strength.value > 0:
                harm_freq = self.harmonic_freq.value
                x += amp_x * self.harmonic_strength.value * 0.3 * math.sin(harm_freq * self.freq_x.value * s + self.t * 1.5)
                y += amp_y * self.harmonic_strength.value * 0.3 * math.sin(harm_freq * self.freq_y.value * s + self.t * 1.5 + phase_offset)

            px = int(center_x + x)
            py = int(center_y + y)
            points.append((px, py))

            # Calculate color
            if self.rainbow_mode.value > 0:
                hue = int((self.hue_start.value + (i / num_pts) * self.hue_range.value) % 180)
            else:
                hue = self.hue_start.value

            # Convert HSV to BGR
            hsv_color = np.array([[[hue, self.saturation.value, self.brightness.value]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))

        # Draw the pattern
        if self.line_mode.value > 0 and len(points) > 1:
            # Draw connected lines with varying colors
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], colors[i], self.thickness.value, cv2.LINE_AA)
            # Connect last to first for closed curve
            cv2.line(frame, points[-1], points[0], colors[-1], self.thickness.value, cv2.LINE_AA)
        else:
            # Draw individual points
            for i, pt in enumerate(points):
                cv2.circle(frame, pt, self.thickness.value, colors[i], -1, cv2.LINE_AA)

        return frame
