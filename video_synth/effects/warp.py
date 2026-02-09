import cv2
import numpy as np
from noise import pnoise2

from effects.base import EffectBase
from effects.enums import WarpType
from common import Widget

PERLIN_SCALE=1700 #???
class Warp(EffectBase):

    def __init__(self, params, image_width: int, image_height: int, group=None):
        subgroup = self.__class__.__name__
        self.params = params
        self.width = image_width
        self.height = image_height
        self.warp_type = params.add("warp_type",
                                    min=0, max=len(WarpType)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=WarpType)
        self.warp_angle_amt = params.add("warp_angle_amt",
                                         min=0, max=360, default=30,
                                         subgroup=subgroup, group=group)
        self.warp_radius_amt = params.add("warp_radius_amt",
                                          min=0, max=360, default=30,
                                          subgroup=subgroup, group=group)
        self.warp_speed = params.add("warp_speed",
                                     min=0, max=100, default=10,
                                     subgroup=subgroup, group=group)
        self.warp_use_fractal = params.add("warp_use_fractal",
                                           min=0, max=1, default=0,
                                           subgroup=subgroup, group=group)
        self.warp_octaves = params.add("warp_octaves",
                                       min=1, max=8, default=4,
                                       subgroup=subgroup, group=group)
        self.warp_gain = params.add("warp_gain",
                                    min=0.0, max=1.0, default=0.5,
                                    subgroup=subgroup, group=group)
        self.warp_lacunarity = params.add("warp_lacunarity",
                                          min=1.0, max=4.0, default=2.0,
                                          subgroup=subgroup, group=group)
        self.x_speed = params.add("x_speed",
                                  min=0.0, max=100.0, default=1.0,
                                  subgroup=subgroup, group=group)
        self.x_size = params.add("x_size",
                                 min=0.25, max=100.0, default=20.0,
                                 subgroup=subgroup, group=group)
        self.y_speed = params.add("y_speed",
                                  min=0.0, max=10.0, default=1.0,
                                  subgroup=subgroup, group=group)
        self.y_size = params.add("y_size",
                                 min=0.25, max=100.0, default=10.0,
                                 subgroup=subgroup, group=group)

        self.t = 0

    def _generate_perlin_flow(self, t, amp_x, amp_y, freq_x, freq_y):
        fx = np.zeros((self.height, self.width), dtype=np.float32)
        fy = np.zeros((self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                nx = x * freq_x * PERLIN_SCALE
                ny = y * freq_y * PERLIN_SCALE
                fx[y, x] = amp_x * pnoise2(nx, ny, base=int(t))
                fy[y, x] = amp_y * pnoise2(nx + 1000, ny + 1000, base=int(t))
        return fx, fy

    def _generate_fractal_flow(
        self, t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity
    ):
        fx = np.zeros((self.height, self.width), dtype=np.float32)
        fy = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                nx = x * freq_x * PERLIN_SCALE
                ny = y * freq_y * PERLIN_SCALE

                noise_x = 0.0
                noise_y = 0.0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(octaves):
                    noise_x += amplitude * pnoise2(
                        nx * frequency, ny * frequency, base=int(t)
                    )
                    noise_y += amplitude * pnoise2(
                        (nx + 1000) * frequency, (ny + 1000) * frequency, base=int(t)
                    )
                    amplitude *= gain
                    frequency *= lacunarity

                fx[y, x] = amp_x * noise_x
                fy[y, x] = amp_y * noise_y
        return fx, fy

    def _polar_warp(self, frame, t, angle_amt, radius_amt, speed):
        cx, cy = self.width / 2, self.height / 2
        y, x = np.indices((self.height, self.width), dtype=np.float32)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        a = np.arctan2(dy, dx)

        # Modify radius and angle
        r_mod = r + np.sin(a * 5 + t * speed) * radius_amt
        a_mod = a + np.cos(r * 0.02 + t * speed) * (angle_amt * np.pi / 180)

        # Back to Cartesian
        map_x = (r_mod * np.cos(a_mod) + cx).astype(np.float32)
        map_y = (r_mod * np.sin(a_mod) + cy).astype(np.float32)

        return cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _first_warp(self, frame: np.ndarray):
        # self.height, self.width = frame.shape[:2]
        # frame = cv2.resize(frame, (self.width, self.height))

        # Create meshgrid for warping effect
        x_indices, y_indices = np.meshgrid(
            np.arange(self.width), np.arange(self.height)
        )

        # Calculate warped indices using sine function
        time = cv2.getTickCount() / cv2.getTickFrequency()
        x_warp = x_indices + self.x_size.value * np.sin(
            y_indices / 20.0 + time * self.x_speed.value
        )
        y_warp = y_indices + self.y_size.value * np.sin(
            x_indices / 20.0 + time * self.y_speed.value
        )

        # Bound indices within valid range
        x_warp = np.clip(x_warp, 0, self.width - 1).astype(np.float32)
        y_warp = np.clip(y_warp, 0, self.height - 1).astype(np.float32)

        # Remap frame using warped indices
        frame = cv2.remap(
            frame, x_warp, y_warp, interpolation=cv2.INTER_LINEAR
        )

        return frame

    def warp(self, frame):
        self.t += 0.1

        match WarpType(self.warp_type.value):
            case WarpType.NONE:
                return frame

            case WarpType.SINE:
                fx = np.sin(np.linspace(0, np.pi * 2, self.width)[None, :] + self.t) * self.x_size.value
                fy = np.cos(np.linspace(0, np.pi * 2, self.height)[:, None] + self.t) * self.y_size.value
                map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
                map_x = (map_x + fx).astype(np.float32)
                map_y = (map_y + fy).astype(np.float32)
                return cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            case WarpType.FRACTAL:
                if self.warp_use_fractal.value > 0:  # TODO: CHANGE TO TOGGLE
                    fx, fy = self._generate_fractal_flow(
                        self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value, octaves, gain, lacunarity
                    )
                else:
                    fx, fy = self._generate_fractal_flow(
                        self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value, 1, 1.0, 1.0
                    )
                map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
                map_x = (map_x + fx).astype(np.float32)
                map_y = (map_y + fy).astype(np.float32)
                return cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            case WarpType.RADIAL:
                return self._polar_warp(
                    frame,
                    self.t,
                    self.warp_angle_amt.value,
                    self.warp_radius_amt.value,
                    self.warp_speed.value
                )

            case WarpType.PERLIN:
                fx, fy = self._generate_perlin_flow(self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value)
                map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
                map_x = (map_x + fx).astype(np.float32)
                map_y = (map_y + fy).astype(np.float32)
                return cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            case WarpType.WARP0:
                return self._first_warp(frame)

            case _:
                return frame
