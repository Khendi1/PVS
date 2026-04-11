"""
Layered Noise Erosion - Stacks animated Perlin noise as a height map and
applies simulated erosion. Over time, ridges and valleys form and shift.
Rendered as luminance displacement on the input frame.
"""

import cv2
import numpy as np
import time

from effects.base import EffectBase


class Erosion(EffectBase):

    def __init__(self, params, width, height, group=None):
        subgroup = self.__class__.__name__
        self.width = width
        self.height = height

        self.erosion_strength = params.new("erosion_strength",
                                            min=0.0, max=1.0, default=0.0,
                                            subgroup=subgroup, group=group)
        self.erosion_scale = params.new("erosion_scale",
                                         min=1.0, max=10.0, default=3.0,
                                         subgroup=subgroup, group=group)
        self.erosion_speed = params.new("erosion_speed",
                                         min=0.0, max=2.0, default=0.2,
                                         subgroup=subgroup, group=group)
        self.erosion_octaves = params.new("erosion_octaves",
                                           min=1, max=6, default=4,
                                           subgroup=subgroup, group=group)
        self.erosion_sharpness = params.new("erosion_sharpness",
                                             min=0.0, max=1.0, default=0.3,
                                             subgroup=subgroup, group=group)

        # Work at half res
        self.rw = width // 2
        self.rh = height // 2

        # Precompute coordinate grids
        x = np.linspace(0, 1, self.rw, dtype=np.float32)
        y = np.linspace(0, 1, self.rh, dtype=np.float32)
        self._X, self._Y = np.meshgrid(x, y)

        self._time = 0.0

    def _layered_noise(self):
        """Generate animated layered noise height map using sine approximation."""
        X, Y = self._X, self._Y
        t = self._time
        scale = self.erosion_scale.value
        octaves = int(self.erosion_octaves.value)

        height_map = np.zeros_like(X)
        amplitude = 1.0
        frequency = scale

        for i in range(octaves):
            phase = t * (0.5 + i * 0.2)
            # Approximate noise with rotated sine waves
            angle = i * 2.399  # golden angle for varied directions
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            coord = X * cos_a + Y * sin_a

            height_map += amplitude * np.sin(coord * frequency * np.pi + phase)
            height_map += amplitude * 0.5 * np.cos(
                (X * sin_a - Y * cos_a) * frequency * 0.7 * np.pi + phase * 1.3
            )

            amplitude *= 0.5
            frequency *= 2.0

        return height_map

    def _erode(self, height_map):
        """Simulate simple erosion: blur valleys, sharpen ridges."""
        sharpness = self.erosion_sharpness.value

        # Blur pass (water flow simulation approximation)
        blurred = cv2.GaussianBlur(height_map, (5, 5), 0)

        # Gradient magnitude (steepness)
        gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)

        # Erode more where gradient is high (steep areas lose material)
        eroded = height_map - gradient * sharpness * 0.1

        # Deposit in flat areas (blend with blurred version)
        result = eroded * (1 - sharpness * 0.3) + blurred * sharpness * 0.3

        return result

    def apply_erosion(self, frame):
        """Apply noise erosion as a luminance displacement on the input frame."""
        strength = self.erosion_strength.value
        if strength == 0:
            return frame

        self._time += 0.016 * self.erosion_speed.value

        # Generate and erode height map
        height_map = self._layered_noise()
        height_map = self._erode(height_map)

        # Normalize to [-1, 1]
        h_min, h_max = height_map.min(), height_map.max()
        if h_max - h_min > 1e-6:
            height_map = (height_map - h_min) / (h_max - h_min) * 2.0 - 1.0
        else:
            return frame

        # Upscale to frame size
        if height_map.shape[:2] != frame.shape[:2]:
            height_map = cv2.resize(height_map, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

        # Apply as brightness modulation
        is_float = frame.dtype == np.float32
        if not is_float:
            work = frame.astype(np.float32)
        else:
            work = frame

        modulation = 1.0 + height_map * strength * 0.5
        result = work * modulation[:, :, np.newaxis]
        result = np.clip(result, 0, 255)

        if not is_float:
            return result.astype(np.uint8)
        return result
