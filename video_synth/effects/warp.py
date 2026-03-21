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
        self.warp_type = params.new("warp_type",
                                    min=0, max=len(WarpType)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=WarpType)
        self.warp_angle_amt = params.new("warp_angle_amt",
                                         min=0, max=360, default=30,
                                         subgroup=subgroup, group=group)
        self.warp_radius_amt = params.new("warp_radius_amt",
                                          min=0, max=360, default=30,
                                          subgroup=subgroup, group=group)
        self.warp_speed = params.new("warp_speed",
                                     min=0, max=100, default=10,
                                     subgroup=subgroup, group=group)
        self.warp_use_fractal = params.new("warp_use_fractal",
                                           min=0, max=1, default=0,
                                           subgroup=subgroup, group=group)
        self.warp_octaves = params.new("warp_octaves",
                                       min=1, max=8, default=4,
                                       subgroup=subgroup, group=group)
        self.warp_gain = params.new("warp_gain",
                                    min=0.0, max=1.0, default=0.5,
                                    subgroup=subgroup, group=group)
        self.warp_lacunarity = params.new("warp_lacunarity",
                                          min=1.0, max=4.0, default=2.0,
                                          subgroup=subgroup, group=group)
        self.x_speed = params.new("x_speed",
                                  min=0.0, max=100.0, default=1.0,
                                  subgroup=subgroup, group=group)
        self.x_size = params.new("x_size",
                                 min=0.25, max=100.0, default=20.0,
                                 subgroup=subgroup, group=group)
        self.y_speed = params.new("y_speed",
                                  min=0.0, max=10.0, default=1.0,
                                  subgroup=subgroup, group=group)
        self.y_size = params.new("y_size",
                                 min=0.25, max=100.0, default=10.0,
                                 subgroup=subgroup, group=group)

        self.fb_warp_decay = params.new("fb_warp_decay",
                                         min=0.0, max=1.0, default=0.95,
                                         subgroup=subgroup, group=group)
        self.fb_warp_strength = params.new("fb_warp_strength",
                                            min=0.0, max=50.0, default=5.0,
                                            subgroup=subgroup, group=group)
        self.fb_warp_freq = params.new("fb_warp_freq",
                                        min=0.1, max=20.0, default=3.0,
                                        subgroup=subgroup, group=group)

        # Displacement feedback params
        self.disp_strength = params.new("disp_strength",
                                         min=0.0, max=30.0, default=5.0,
                                         subgroup=subgroup, group=group)
        self.disp_decay = params.new("disp_decay",
                                      min=0.0, max=1.0, default=0.92,
                                      subgroup=subgroup, group=group)
        self.disp_blur = params.new("disp_blur",
                                     min=1, max=15, default=5,
                                     subgroup=subgroup, group=group)

        # Convection params
        self.conv_rise_speed = params.new("conv_rise_speed",
                                           min=0.0, max=10.0, default=2.0,
                                           subgroup=subgroup, group=group)
        self.conv_diffusion = params.new("conv_diffusion",
                                          min=0.0, max=1.0, default=0.5,
                                          subgroup=subgroup, group=group)
        self.conv_turbulence = params.new("conv_turbulence",
                                           min=0.0, max=1.0, default=0.3,
                                           subgroup=subgroup, group=group)
        self.conv_decay = params.new("conv_decay",
                                      min=0.0, max=1.0, default=0.95,
                                      subgroup=subgroup, group=group)

        # Reaction-Diffusion warp params
        self.rd_warp_strength = params.new("rd_warp_strength",
                                            min=0.0, max=30.0, default=10.0,
                                            subgroup=subgroup, group=group)
        self.rd_warp_feed = params.new("rd_warp_feed",
                                        min=0.01, max=0.1, default=0.055,
                                        subgroup=subgroup, group=group)
        self.rd_warp_kill = params.new("rd_warp_kill",
                                        min=0.03, max=0.08, default=0.062,
                                        subgroup=subgroup, group=group)
        self.rd_warp_speed = params.new("rd_warp_speed",
                                         min=0.1, max=5.0, default=1.0,
                                         subgroup=subgroup, group=group)

        self.t = 0

        # Persistent displacement maps for feedback warp (identity = no displacement)
        y_id, x_id = np.meshgrid(np.arange(self.height, dtype=np.float32),
                                  np.arange(self.width, dtype=np.float32), indexing='ij')
        self._fb_base_x = x_id.copy()
        self._fb_base_y = y_id.copy()
        self._fb_map_x = x_id.copy()
        self._fb_map_y = y_id.copy()

        # Displacement feedback: accumulated displacement from luminance gradients
        self._disp_offset_x = np.zeros((self.height, self.width), dtype=np.float32)
        self._disp_offset_y = np.zeros((self.height, self.width), dtype=np.float32)
        self._prev_gray = None

        # Convection: accumulated heat-rise displacement
        self._conv_offset_x = np.zeros((self.height, self.width), dtype=np.float32)
        self._conv_offset_y = np.zeros((self.height, self.width), dtype=np.float32)

        # Reaction-Diffusion warp: two chemical grids at 1/4 resolution
        rd_h, rd_w = self.height // 4, self.width // 4
        self._rd_A = np.ones((rd_h, rd_w), dtype=np.float32)
        self._rd_B = np.zeros((rd_h, rd_w), dtype=np.float32)
        # Seed with random spots
        cx, cy = rd_w // 2, rd_h // 2
        r = min(rd_w, rd_h) // 4
        yy, xx = np.ogrid[:rd_h, :rd_w]
        seed_mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        self._rd_B[seed_mask] = 1.0
        # Add scattered seed points
        for _ in range(20):
            sx = np.random.randint(0, rd_w)
            sy = np.random.randint(0, rd_h)
            sr = np.random.randint(2, max(3, r // 3))
            spot = ((xx - sx)**2 + (yy - sy)**2) < sr**2
            self._rd_B[spot] = 1.0

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

    def _feedback_warp(self, frame):
        """Feedback warp: displacement maps accumulate over time, creating chaotic self-similar patterns."""
        strength = self.fb_warp_strength.value
        decay = self.fb_warp_decay.value
        freq = self.fb_warp_freq.value

        # Generate a small per-frame perturbation using sine waves
        dx = np.sin(self._fb_base_y * freq * 0.01 + self.t) * strength
        dy = np.cos(self._fb_base_x * freq * 0.01 + self.t * 0.7) * strength

        # Accumulate: add perturbation to the displacement offset from identity
        offset_x = self._fb_map_x - self._fb_base_x
        offset_y = self._fb_map_y - self._fb_base_y

        # Decay existing displacement toward zero, then add new perturbation
        offset_x = offset_x * decay + dx
        offset_y = offset_y * decay + dy

        # Self-distort: remap the displacement maps through themselves
        # This is what creates the chaotic, fractal-like folding
        new_map_x = self._fb_base_x + offset_x
        new_map_y = self._fb_base_y + offset_y

        # Remap the offset maps through the accumulated warp (self-referential feedback)
        warped_offset_x = cv2.remap(offset_x, new_map_x, new_map_y,
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        warped_offset_y = cv2.remap(offset_y, new_map_x, new_map_y,
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

        # Blend: mix self-distorted offsets back with a fraction of the straight offsets
        # to control how chaotic it gets
        self._fb_map_x = self._fb_base_x + warped_offset_x * 0.7 + offset_x * 0.3
        self._fb_map_y = self._fb_base_y + warped_offset_y * 0.7 + offset_y * 0.3

        # Clamp to valid pixel range
        map_x = np.clip(self._fb_map_x, 0, self.width - 1)
        map_y = np.clip(self._fb_map_y, 0, self.height - 1)

        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _displacement_warp(self, frame):
        """Luminance-gradient displacement: bright areas push pixels outward,
        creating melting-wax effects that accumulate over feedback frames."""
        strength = self.disp_strength.value
        decay = self.disp_decay.value
        blur_k = int(self.disp_blur.value) | 1  # ensure odd

        # Convert current frame to grayscale
        if frame.dtype != np.uint8:
            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Smooth to reduce noise
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # Compute luminance gradients (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 255.0
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 255.0

        # Accumulate: decay old displacement, add new gradient-driven push
        self._disp_offset_x = self._disp_offset_x * decay + gx * strength
        self._disp_offset_y = self._disp_offset_y * decay + gy * strength

        # Build remap coordinates
        map_x = self._fb_base_x + self._disp_offset_x
        map_y = self._fb_base_y + self._disp_offset_y

        map_x = np.clip(map_x, 0, self.width - 1)
        map_y = np.clip(map_y, 0, self.height - 1)

        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _convection_warp(self, frame):
        """Heat-rise convection: diffuses frame upward with turbulence,
        creating candle-flame dissolve effects."""
        rise = self.conv_rise_speed.value
        diffusion = self.conv_diffusion.value
        turb = self.conv_turbulence.value
        decay = self.conv_decay.value

        # Decay existing displacement
        self._conv_offset_x *= decay
        self._conv_offset_y *= decay

        # Upward bias: negative y = upward in image coordinates
        self._conv_offset_y -= rise * 0.5

        # Turbulence: sine-based horizontal wobble
        turb_x = np.sin(self._fb_base_y * 0.05 + self.t * 0.7) * turb * 3.0
        turb_x += np.sin(self._fb_base_x * 0.03 + self.t * 1.1) * turb * 1.5
        self._conv_offset_x += turb_x * 0.1

        # Diffusion: blur the displacement maps themselves
        if diffusion > 0.01:
            k = max(3, int(diffusion * 15) | 1)
            self._conv_offset_x = cv2.GaussianBlur(self._conv_offset_x, (k, k), 0)
            self._conv_offset_y = cv2.GaussianBlur(self._conv_offset_y, (k, k), 0)

        # Build remap
        map_x = self._fb_base_x + self._conv_offset_x
        map_y = self._fb_base_y + self._conv_offset_y

        map_x = np.clip(map_x, 0, self.width - 1)
        map_y = np.clip(map_y, 0, self.height - 1)

        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _rd_warp(self, frame):
        """Reaction-Diffusion warp: run a Gray-Scott RD simulation and use
        the chemical gradient as a continuously evolving displacement field.
        Creates organic cell-boundary warps that split, merge, and breathe."""
        strength = self.rd_warp_strength.value
        feed = self.rd_warp_feed.value
        kill = self.rd_warp_kill.value
        speed = self.rd_warp_speed.value

        A, B = self._rd_A, self._rd_B
        dt = 1.0 * speed

        # Run a few RD steps per frame for visible evolution
        for _ in range(4):
            # Laplacian via convolution (fast)
            LA = cv2.Laplacian(A, cv2.CV_32F)
            LB = cv2.Laplacian(B, cv2.CV_32F)

            AB2 = A * B * B
            A += dt * (0.21 * LA - AB2 + feed * (1.0 - A))
            B += dt * (0.105 * LB + AB2 - (kill + feed) * B)

            A = np.clip(A, 0, 1)
            B = np.clip(B, 0, 1)

        self._rd_A = A
        self._rd_B = B

        # Re-seed if B dies out
        if B.max() < 0.01:
            rd_h, rd_w = B.shape
            for _ in range(10):
                sx = np.random.randint(0, rd_w)
                sy = np.random.randint(0, rd_h)
                sr = np.random.randint(2, 5)
                yy, xx = np.ogrid[:rd_h, :rd_w]
                spot = ((xx - sx)**2 + (yy - sy)**2) < sr**2
                B[spot] = 1.0

        # Upscale B to frame size
        B_full = cv2.resize(B, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Use gradient of B as displacement vectors
        gx = cv2.Sobel(B_full, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(B_full, cv2.CV_32F, 0, 1, ksize=3)

        # Build remap
        map_x = self._fb_base_x + gx * strength
        map_y = self._fb_base_y + gy * strength

        map_x = np.clip(map_x, 0, self.width - 1)
        map_y = np.clip(map_y, 0, self.height - 1)

        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

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

            case WarpType.FEEDBACK:
                return self._feedback_warp(frame)

            case WarpType.DISPLACEMENT:
                return self._displacement_warp(frame)

            case WarpType.CONVECTION:
                return self._convection_warp(frame)

            case WarpType.RD_WARP:
                return self._rd_warp(frame)

            case _:
                return frame
