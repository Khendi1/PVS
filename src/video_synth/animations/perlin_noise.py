import cv2
import numpy as np
from enum import IntEnum, auto
from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class PerlinNoiseType(IntEnum):
    PERLIN   = 0
    TURBULENCE = auto()
    RIDGED   = auto()
    FBM      = auto()


class PerlinNoise(Animation):
    """
    Generative noise source using Perlin/simplex noise (via the 'noise' package).
    Computes noise on a downscaled grid then bicubically upscales for real-time performance.
    Supports Perlin, Turbulence (|noise|), Ridged (1-|noise|), and FBM modes.
    """

    _DOWN = 8  # spatial downsample factor: 640/8=80, 480/8=60 -> ~4800 noise calls/frame

    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        self.t = 0.0

        self.pnoise_type = params.new(
            "pnoise_type",
            min=0, max=len(PerlinNoiseType) - 1, default=0,
            group=group, subgroup=subgroup,
            type=Widget.DROPDOWN, options=PerlinNoiseType,
        )
        self.pnoise_scale = params.new(
            "pnoise_scale",
            min=0.002, max=0.1, default=0.01,
            group=group, subgroup=subgroup,
        )
        self.pnoise_speed = params.new(
            "pnoise_speed",
            min=0.0, max=2.0, default=0.2,
            group=group, subgroup=subgroup,
        )
        self.pnoise_octaves = params.new(
            "pnoise_octaves",
            min=1, max=8, default=4,
            group=group, subgroup=subgroup,
        )
        self.pnoise_persistence = params.new(
            "pnoise_persistence",
            min=0.1, max=1.0, default=0.5,
            group=group, subgroup=subgroup,
        )
        self.pnoise_lacunarity = params.new(
            "pnoise_lacunarity",
            min=1.0, max=4.0, default=2.0,
            group=group, subgroup=subgroup,
        )
        self.pnoise_colormap = params.new(
            "pnoise_colormap",
            min=0, max=len(Colormap) - 1, default=int(Colormap.INFERNO),
            group=group, subgroup=subgroup,
            type=Widget.DROPDOWN, options=Colormap,
        )
        self.pnoise_offset_x = params.new(
            "pnoise_offset_x",
            min=-100.0, max=100.0, default=0.0,
            group=group, subgroup=subgroup,
        )
        self.pnoise_offset_y = params.new(
            "pnoise_offset_y",
            min=-100.0, max=100.0, default=0.0,
            group=group, subgroup=subgroup,
        )

    def get_frame(self, frame=None):
        try:
            import noise as noise_lib
        except ImportError:
            # Fallback: solid grey if library not installed
            return np.full((self.height, self.width, 3), 64, dtype=np.float32)

        self.t += float(self.pnoise_speed.value) * 0.01
        scale       = float(self.pnoise_scale.value)
        octaves     = max(1, int(self.pnoise_octaves.value))
        persistence = float(self.pnoise_persistence.value)
        lacunarity  = float(self.pnoise_lacunarity.value)
        ox = float(self.pnoise_offset_x.value) * 0.1
        oy = float(self.pnoise_offset_y.value) * 0.1

        small_h = max(1, self.height // self._DOWN)
        small_w = max(1, self.width  // self._DOWN)

        noise_flat = np.empty(small_h * small_w, dtype=np.float32)
        idx = 0
        for y in range(small_h):
            for x in range(small_w):
                noise_flat[idx] = noise_lib.pnoise3(
                    x * scale + ox,
                    y * scale + oy,
                    self.t,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=10000,
                    repeaty=10000,
                    repeatz=10000,
                )
                idx += 1

        noise_grid = noise_flat.reshape(small_h, small_w)

        noise_type = int(self.pnoise_type.value)
        if noise_type == PerlinNoiseType.TURBULENCE:
            noise_grid = np.abs(noise_grid)
        elif noise_type == PerlinNoiseType.RIDGED:
            noise_grid = 1.0 - np.abs(noise_grid)
        # PERLIN and FBM: pnoise3 with octaves already gives FBM; plain PERLIN uses octaves=1

        n_min, n_max = float(noise_grid.min()), float(noise_grid.max())
        if n_max > n_min:
            gray_small = ((noise_grid - n_min) / (n_max - n_min) * 255).astype(np.uint8)
        else:
            gray_small = np.zeros((small_h, small_w), dtype=np.uint8)

        # Bicubic upscale to full resolution
        gray = cv2.resize(gray_small, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        # Apply colormap
        colormap_idx = int(self.pnoise_colormap.value) % len(COLORMAP_OPTIONS)
        colored = cv2.applyColorMap(gray, COLORMAP_OPTIONS[colormap_idx])

        return colored.astype(np.float32)
