import cv2
import numpy as np

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget, Toggle

class Chladni(Animation):
    """
    Chladni Patterns - standing wave patterns on a vibrating plate.
    Particles accumulate at nodal lines where vibration amplitude is zero.
    """
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Wave parameters
        self.freq_m = params.add("chladni_freq_m",
                                 min=1, max=20, default=5.0,
                                 subgroup=subgroup, group=group)
        self.freq_n = params.add("chladni_freq_n",
                                 min=1, max=20, default=3.0,
                                 subgroup=subgroup, group=group)
        self.amplitude = params.add("chladni_amplitude",
                                    min=0.1, max=2.0, default=1.0,
                                    subgroup=subgroup, group=group)
        self.animation_speed = params.add("chladni_speed",
                                          min=0.0, max=2.0, default=0.5,
                                          subgroup=subgroup, group=group)
        self.pattern_blend = params.add("chladni_blend",
                                        min=0.0, max=1.0, default=0.5,
                                        subgroup=subgroup, group=group)

        # Particle simulation
        self.num_particles = params.add("chladni_particles",
                                        min=1000, max=50000, default=10000,
                                        subgroup=subgroup, group=group)
        self.particle_speed = params.add("chladni_particle_speed",
                                         min=0.1, max=5.0, default=1.0,
                                         subgroup=subgroup, group=group)
        self.friction = params.add("chladni_friction",
                                   min=0.8, max=0.99, default=0.95,
                                   subgroup=subgroup, group=group)

        # Visual parameters
        self.show_wave = params.add("chladni_show_wave",
                                    min=0, max=1, default=1,
                                    subgroup=subgroup, group=group,
                                    type=Widget.TOGGLE)
        self.colormap = params.add("chladni_colormap",
                                   min=0, max=len(COLORMAP_OPTIONS)-1, default=2,
                                   subgroup=subgroup, group=group,
                                   type=Widget.DROPDOWN, options=Colormap)
        self.particle_r = params.add("chladni_particle_r",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group)
        self.particle_g = params.add("chladni_particle_g",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group)
        self.particle_b = params.add("chladni_particle_b",
                                     min=0, max=255, default=200,
                                     subgroup=subgroup, group=group)

        self.time = 0.0
        self._init_particles()
        self.prev_num_particles = self.num_particles.value

        # Performance: cache meshgrid
        x_coords = np.linspace(0, self.width - 1, self.width)
        y_coords = np.linspace(0, self.height - 1, self.height)
        self._X, self._Y = np.meshgrid(x_coords, y_coords)

    def _init_particles(self):
        """Initialize particle positions and velocities."""
        n = int(self.num_particles.value)
        self.particles = np.random.uniform(0, 1, (n, 2)).astype(np.float32)
        self.particles[:, 0] *= self.width
        self.particles[:, 1] *= self.height
        self.velocities = np.zeros((n, 2), dtype=np.float32)

    def _chladni_value(self, x, y, m, n, t):
        """
        Calculate Chladni pattern value at position.
        Uses superposition of two wave modes.
        """
        # Normalize coordinates to [-1, 1]
        nx = (2.0 * x / self.width - 1.0)
        ny = (2.0 * y / self.height - 1.0)

        # Two orthogonal modes with phase offset
        phase = t * self.animation_speed.value
        mode1 = np.cos(m * np.pi * nx) * np.cos(n * np.pi * ny + phase)
        mode2 = np.cos(n * np.pi * nx + phase * 0.7) * np.cos(m * np.pi * ny)

        # Blend between modes
        blend = self.pattern_blend.value
        return mode1 * (1 - blend) + mode2 * blend

    def _chladni_gradient(self, x, y, m, n, t):
        """Calculate gradient of Chladni pattern for particle movement."""
        eps = 1.0
        val_c = self._chladni_value(x, y, m, n, t)
        val_x = self._chladni_value(x + eps, y, m, n, t)
        val_y = self._chladni_value(x, y + eps, m, n, t)
        dx = (val_x - val_c) / eps
        dy = (val_y - val_c) / eps
        return dx, dy

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:

        # Check for particle count change
        if self.num_particles.value != self.prev_num_particles:
            self._init_particles()
            self.prev_num_particles = self.num_particles.value

        self.time += 0.016  # ~60fps

        m = self.freq_m.value
        n = self.freq_n.value
        amp = self.amplitude.value

        # Use cached meshgrid
        wave = self._chladni_value(self._X, self._Y, m, n, self.time)
        wave = np.abs(wave) * amp

        # Normalize to 0-255
        wave_normalized = (wave / wave.max() * 255).astype(np.uint8) if wave.max() > 0 else np.zeros_like(wave, dtype=np.uint8)

        # Create output frame
        if self.show_wave.value:
            pattern = cv2.applyColorMap(wave_normalized, COLORMAP_OPTIONS[int(self.colormap.value)])
        else:
            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Update particles - they move toward nodal lines (where wave = 0)
        speed = self.particle_speed.value
        friction = self.friction.value

        # Vectorized gradient calculation for all particles
        px = self.particles[:, 0]
        py = self.particles[:, 1]

        # Calculate gradients at particle positions
        eps = 2.0
        val_c = self._chladni_value(px, py, m, n, self.time)
        val_xp = self._chladni_value(px + eps, py, m, n, self.time)
        val_yp = self._chladni_value(px, py + eps, m, n, self.time)

        grad_x = (val_xp - val_c) / eps
        grad_y = (val_yp - val_c) / eps

        # Particles move along gradient toward zero (nodal lines)
        # The force is proportional to the value and direction is along gradient
        force_x = -val_c * grad_x * speed
        force_y = -val_c * grad_y * speed

        # Update velocities with friction
        self.velocities[:, 0] = self.velocities[:, 0] * friction + force_x
        self.velocities[:, 1] = self.velocities[:, 1] * friction + force_y

        # Update positions
        self.particles[:, 0] += self.velocities[:, 0]
        self.particles[:, 1] += self.velocities[:, 1]

        # Wrap around boundaries
        self.particles[:, 0] = np.mod(self.particles[:, 0], self.width)
        self.particles[:, 1] = np.mod(self.particles[:, 1], self.height)

        # Render particles
        particle_color = (int(self.particle_b.value), int(self.particle_g.value), int(self.particle_r.value))
        for px, py in self.particles:
            xi, yi = int(px), int(py)
            if 0 <= xi < self.width and 0 <= yi < self.height:
                pattern[yi, xi] = particle_color

        return pattern
