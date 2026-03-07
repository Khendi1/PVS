"""
Lenia - Continuous cellular automata that produces organic, amoeba-like
forms that breathe, pulse, and slowly evolve. A smooth generalization
of Conway's Game of Life.
"""

import cv2
import numpy as np

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


class Lenia(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        self.lenia_dt = params.add("lenia_dt",
                                    min=0.01, max=0.5, default=0.1,
                                    subgroup=subgroup, group=group)
        self.lenia_mu = params.add("lenia_mu",
                                    min=0.05, max=0.5, default=0.15,
                                    subgroup=subgroup, group=group)
        self.lenia_sigma = params.add("lenia_sigma",
                                       min=0.005, max=0.1, default=0.017,
                                       subgroup=subgroup, group=group)
        self.lenia_radius = params.add("lenia_radius",
                                        min=5, max=30, default=13,
                                        subgroup=subgroup, group=group)
        self.lenia_colormap = params.add("lenia_colormap",
                                          min=0, max=len(COLORMAP_OPTIONS)-1, default=int(Colormap.INFERNO),
                                          subgroup=subgroup, group=group,
                                          type=Widget.DROPDOWN, options=Colormap)
        self.lenia_seed_density = params.add("lenia_seed_density",
                                              min=0.01, max=0.5, default=0.15,
                                              subgroup=subgroup, group=group)

        # Work at reduced resolution for performance
        self.rw = width // 4
        self.rh = height // 4

        self._prev_radius = int(self.lenia_radius.value)
        self._build_kernel()
        self._seed()

    def _build_kernel(self):
        """Build the ring-shaped convolution kernel."""
        r = int(self.lenia_radius.value)
        size = 2 * r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        dist = np.sqrt(x*x + y*y) / r

        # Ring kernel: Gaussian bump centered at distance ~0.5 from center
        self.kernel = np.exp(-((dist - 0.5) ** 2) / (2 * 0.15 ** 2))
        self.kernel[dist > 1] = 0
        total = self.kernel.sum()
        if total > 0:
            self.kernel /= total
        self.kernel = self.kernel.astype(np.float32)

    def _seed(self):
        """Initialize the grid with random blobs."""
        density = self.lenia_seed_density.value
        self.grid = np.zeros((self.rh, self.rw), dtype=np.float32)

        # Place a few random circular blobs
        num_blobs = max(1, int(density * 20))
        for _ in range(num_blobs):
            cx = np.random.randint(self.rw // 4, 3 * self.rw // 4)
            cy = np.random.randint(self.rh // 4, 3 * self.rh // 4)
            r = np.random.randint(3, max(4, self.rw // 8))
            y, x = np.ogrid[:self.rh, :self.rw]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            self.grid[mask] = np.random.uniform(0.2, 1.0, mask.sum()).astype(np.float32)

    def _growth(self, u):
        """Growth function: Gaussian bump centered at mu with width sigma."""
        mu = self.lenia_mu.value
        sigma = self.lenia_sigma.value
        return 2.0 * np.exp(-((u - mu) ** 2) / (2.0 * sigma ** 2)) - 1.0

    def get_frame(self, frame=None):
        # Check if kernel needs rebuilding
        current_radius = int(self.lenia_radius.value)
        if current_radius != self._prev_radius:
            self._build_kernel()
            self._prev_radius = current_radius

        # Compute neighborhood potential via convolution
        potential = cv2.filter2D(self.grid, -1, self.kernel,
                                 borderType=cv2.BORDER_WRAP)

        # Apply growth function and update
        dt = self.lenia_dt.value
        self.grid = np.clip(self.grid + dt * self._growth(potential), 0, 1)

        # If the grid dies out, reseed
        if self.grid.max() < 0.01:
            self._seed()

        # Convert to colormap
        display = (self.grid * 255).astype(np.uint8)
        cmap_idx = int(self.lenia_colormap.value)
        colored = cv2.applyColorMap(display, COLORMAP_OPTIONS[cmap_idx])

        # Upscale
        if colored.shape[:2] != (self.height, self.width):
            colored = cv2.resize(colored, (self.width, self.height),
                                 interpolation=cv2.INTER_LINEAR)

        return colored
