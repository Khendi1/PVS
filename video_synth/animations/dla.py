import cv2
import numpy as np
import math

from animations.base import Animation
from common import Widget, Toggle

class DLA(Animation):
    """
    Diffusion-Limited Aggregation - particles random-walk until they stick
    to a growing crystal structure, creating organic dendrite-like fractals.
    """
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Growth parameters
        self.num_particles = params.add("dla_num_particles",
                                        min=10, max=500, default=100,
                                        subgroup=subgroup, group=group)
        self.stickiness = params.add("dla_stickiness",
                                     min=0.1, max=1.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.spawn_radius_ratio = params.add("dla_spawn_radius",
                                             min=1.1, max=2.0, default=1.3,
                                             subgroup=subgroup, group=group)
        self.particle_speed = params.add("dla_particle_speed",
                                         min=1, max=10, default=3,
                                         subgroup=subgroup, group=group)
        self.branch_bias = params.add("dla_branch_bias",
                                      min=-1.0, max=1.0, default=0.0,
                                      subgroup=subgroup, group=group)
        self.fade = params.add("dla_fade",
                               min=0.0, max=1.0, default=0.99,
                               subgroup=subgroup, group=group)

        # Color parameters
        self.crystal_r = params.add("dla_crystal_r",
                                    min=0, max=255, default=100,
                                    subgroup=subgroup, group=group)
        self.crystal_g = params.add("dla_crystal_g",
                                    min=0, max=255, default=200,
                                    subgroup=subgroup, group=group)
        self.crystal_b = params.add("dla_crystal_b",
                                    min=0, max=255, default=255,
                                    subgroup=subgroup, group=group)
        self.particle_r = params.add("dla_particle_r",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group)
        self.particle_g = params.add("dla_particle_g",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group)
        self.particle_b = params.add("dla_particle_b",
                                     min=0, max=255, default=255,
                                     subgroup=subgroup, group=group)

        # Reset trigger
        self.reset_trigger = params.add("dla_reset",
                                        min=0, max=1, default=0,
                                        subgroup=subgroup, group=group,
                                        type=Widget.RADIO, options=Toggle)

        self._initialize_simulation()

    def _initialize_simulation(self):
        """Initialize the DLA simulation state."""
        # Crystal grid - True where crystal exists
        self.crystal = np.zeros((self.height, self.width), dtype=bool)
        # Age map for color variation
        self.crystal_age = np.zeros((self.height, self.width), dtype=np.float32)
        # Seed crystal at center
        cx, cy = self.width // 2, self.height // 2
        self.crystal[cy-2:cy+2, cx-2:cx+2] = True
        self.crystal_age[cy-2:cy+2, cx-2:cx+2] = 1.0
        # Current growth radius
        self.max_radius = 5
        self.age_counter = 1.0
        # Particles: [x, y] positions
        self._spawn_particles()
        self.prev_reset = 0

    def _spawn_particles(self):
        """Spawn particles at random positions on spawn circle."""
        n = int(self.num_particles.value)
        spawn_r = self.max_radius * self.spawn_radius_ratio.value
        spawn_r = max(spawn_r, 20)
        angles = np.random.uniform(0, 2 * np.pi, n)
        cx, cy = self.width // 2, self.height // 2
        self.particles = np.zeros((n, 2), dtype=np.float32)
        self.particles[:, 0] = cx + spawn_r * np.cos(angles)
        self.particles[:, 1] = cy + spawn_r * np.sin(angles)

    def _respawn_particle(self, idx):
        """Respawn a single particle on the spawn circle."""
        spawn_r = self.max_radius * self.spawn_radius_ratio.value
        spawn_r = max(spawn_r, 20)
        angle = np.random.uniform(0, 2 * np.pi)
        cx, cy = self.width // 2, self.height // 2
        self.particles[idx, 0] = cx + spawn_r * np.cos(angle)
        self.particles[idx, 1] = cy + spawn_r * np.sin(angle)

    def _check_neighbors(self, x, y):
        """Check if position has neighboring crystal."""
        xi, yi = int(x), int(y)
        if xi < 1 or xi >= self.width - 1 or yi < 1 or yi >= self.height - 1:
            return False
        # Check 8-connected neighbors
        return np.any(self.crystal[yi-1:yi+2, xi-1:xi+2])

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:

        # Check for reset trigger
        if self.reset_trigger.value == 1 and self.prev_reset == 0:
            self._initialize_simulation()
        self.prev_reset = self.reset_trigger.value

        # Apply fade to previous frame
        if frame is None:
            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            pattern = (frame * self.fade.value).astype(np.uint8)

        speed = int(self.particle_speed.value)
        bias = self.branch_bias.value
        cx, cy = self.width // 2, self.height // 2

        # Move particles with random walk
        for i in range(len(self.particles)):
            for _ in range(speed):
                # Random walk with optional bias toward/away from center
                dx = np.random.choice([-1, 0, 1])
                dy = np.random.choice([-1, 0, 1])

                # Apply radial bias
                if bias != 0:
                    px, py = self.particles[i]
                    to_center_x = cx - px
                    to_center_y = cy - py
                    dist = math.sqrt(to_center_x**2 + to_center_y**2) + 0.001
                    if np.random.random() < abs(bias):
                        if bias > 0:  # Bias toward center
                            dx += int(np.sign(to_center_x))
                            dy += int(np.sign(to_center_y))
                        else:  # Bias away
                            dx -= int(np.sign(to_center_x))
                            dy -= int(np.sign(to_center_y))

                self.particles[i, 0] += dx
                self.particles[i, 1] += dy

                x, y = self.particles[i]

                # Check if stuck to crystal
                if self._check_neighbors(x, y):
                    if np.random.random() < self.stickiness.value:
                        xi, yi = int(x), int(y)
                        if 0 <= xi < self.width and 0 <= yi < self.height:
                            self.crystal[yi, xi] = True
                            self.age_counter += 0.001
                            self.crystal_age[yi, xi] = self.age_counter
                            # Update max radius
                            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                            self.max_radius = max(self.max_radius, dist + 5)
                        self._respawn_particle(i)
                        break

                # Respawn if too far or out of bounds
                dist_from_center = math.sqrt((x - cx)**2 + (y - cy)**2)
                kill_radius = self.max_radius * self.spawn_radius_ratio.value * 1.5
                if (dist_from_center > kill_radius or
                    x < 0 or x >= self.width or y < 0 or y >= self.height):
                    self._respawn_particle(i)
                    break

        # Render crystal with age-based coloring
        crystal_color = np.array([self.crystal_b.value, self.crystal_g.value, self.crystal_r.value])
        # Normalize ages for color variation
        max_age = self.age_counter if self.age_counter > 0 else 1.0
        normalized_age = self.crystal_age / max_age

        for c in range(3):
            # Vary color based on age
            color_val = crystal_color[c] * (0.5 + 0.5 * normalized_age)
            pattern[:, :, c] = np.where(self.crystal, color_val.astype(np.uint8), pattern[:, :, c])

        # Render particles
        particle_color = (int(self.particle_b.value), int(self.particle_g.value), int(self.particle_r.value))
        for px, py in self.particles:
            xi, yi = int(px), int(py)
            if 0 <= xi < self.width and 0 <= yi < self.height:
                cv2.circle(pattern, (xi, yi), 1, particle_color, -1)

        return pattern
