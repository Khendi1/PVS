import cv2
import numpy as np
import math

from animations.base import Animation
from animations.enums import AttractorType
from common import Widget

class StrangeAttractor(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Attractor type selector
        self.attractor_type = params.add("attractor_type",
                                         min=0, max=len(AttractorType)-1, default=0,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options=AttractorType)
        self.prev_attractor_type = self.attractor_type.value

        # Common parameters
        self.dt = params.add("attractor_dt",
                             min=0.001, max=0.05, default=0.01,
                             subgroup=subgroup, group=group)
        self.num_steps = params.add("attractor_num_steps",
                                    min=1, max=50, default=10,
                                    subgroup=subgroup, group=group)
        self.scale = params.add("attractor_scale",
                                min=1.0, max=20.0, default=5.0,
                                subgroup=subgroup, group=group)
        self.line_width = params.add("attractor_line_width",
                                     min=1, max=5, default=1,
                                     subgroup=subgroup, group=group)
        self.fade = params.add("attractor_fade",
                               min=0.0, max=1.0, default=0.95,
                               subgroup=subgroup, group=group)

        # Color parameters
        self.attractor_r = params.add("attractor_r",
                                      min=0, max=255, default=255,
                                      subgroup=subgroup, group=group)
        self.attractor_g = params.add("attractor_g",
                                      min=0, max=255, default=255,
                                      subgroup=subgroup, group=group)
        self.attractor_b = params.add("attractor_b",
                                      min=0, max=255, default=255,
                                      subgroup=subgroup, group=group)

        # Lorenz Attractor parameters (3D)
        self.lorenz_sigma = params.add("lorenz_sigma",
                                       min=1.0, max=20.0, default=10.0,
                                       subgroup=subgroup, group=group)
        self.lorenz_rho = params.add("lorenz_rho",
                                     min=1.0, max=50.0, default=28.0,
                                     subgroup=subgroup, group=group)
        self.lorenz_beta = params.add("lorenz_beta",
                                      min=0.1, max=5.0, default=2.667,
                                      subgroup=subgroup, group=group)

        # Clifford Attractor parameters (2D) - Beautiful spiraling patterns
        self.clifford_a = params.add("clifford_a",
                                     min=-3.0, max=3.0, default=-1.4,
                                     subgroup=subgroup, group=group)
        self.clifford_b = params.add("clifford_b",
                                     min=-3.0, max=3.0, default=1.6,
                                     subgroup=subgroup, group=group)
        self.clifford_c = params.add("clifford_c",
                                     min=-3.0, max=3.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.clifford_d = params.add("clifford_d",
                                     min=-3.0, max=3.0, default=0.7,
                                     subgroup=subgroup, group=group)

        # De Jong Attractor parameters (2D) - Similar elegance, different character
        self.dejong_a = params.add("dejong_a",
                                   min=-3.0, max=3.0, default=-2.0,
                                   subgroup=subgroup, group=group)
        self.dejong_b = params.add("dejong_b",
                                   min=-3.0, max=3.0, default=-2.0,
                                   subgroup=subgroup, group=group)
        self.dejong_c = params.add("dejong_c",
                                   min=-3.0, max=3.0, default=-1.2,
                                   subgroup=subgroup, group=group)
        self.dejong_d = params.add("dejong_d",
                                   min=-3.0, max=3.0, default=2.0,
                                   subgroup=subgroup, group=group)

        # Aizawa Attractor parameters (3D) - Organic chaotic system
        self.aizawa_a = params.add("aizawa_a",
                                   min=0.1, max=1.5, default=0.95,
                                   subgroup=subgroup, group=group)
        self.aizawa_b = params.add("aizawa_b",
                                   min=0.1, max=1.5, default=0.7,
                                   subgroup=subgroup, group=group)
        self.aizawa_c = params.add("aizawa_c",
                                   min=0.1, max=1.0, default=0.6,
                                   subgroup=subgroup, group=group)
        self.aizawa_d = params.add("aizawa_d",
                                   min=0.1, max=5.0, default=3.5,
                                   subgroup=subgroup, group=group)
        self.aizawa_e = params.add("aizawa_e",
                                   min=0.0, max=1.0, default=0.25,
                                   subgroup=subgroup, group=group)
        self.aizawa_f = params.add("aizawa_f",
                                   min=0.0, max=0.5, default=0.1,
                                   subgroup=subgroup, group=group)

        # Thomas Attractor parameters (3D) - Smooth, ribbon-like trajectories
        self.thomas_b = params.add("thomas_b",
                                   min=0.1, max=0.3, default=0.208186,
                                   subgroup=subgroup, group=group)

        # State variables for each attractor
        self._init_attractor_states()

    def _init_attractor_states(self):
        """Initialize state variables for all attractors."""
        # Lorenz state (3D)
        self.lorenz_x, self.lorenz_y, self.lorenz_z = 0.1, 0.0, 0.0
        # Clifford state (2D)
        self.clifford_x, self.clifford_y = 0.1, 0.1
        # De Jong state (2D)
        self.dejong_x, self.dejong_y = 0.1, 0.1
        # Aizawa state (3D)
        self.aizawa_x, self.aizawa_y, self.aizawa_z = 0.1, 0.0, 0.0
        # Thomas state (3D)
        self.thomas_x, self.thomas_y, self.thomas_z = 1.0, 1.0, 1.0

    def _reset_current_attractor(self):
        """Reset only the current attractor's state."""
        atype = self.attractor_type.value
        if atype == AttractorType.LORENZ:
            self.lorenz_x, self.lorenz_y, self.lorenz_z = 0.1, 0.0, 0.0
        elif atype == AttractorType.CLIFFORD:
            self.clifford_x, self.clifford_y = 0.1, 0.1
        elif atype == AttractorType.DE_JONG:
            self.dejong_x, self.dejong_y = 0.1, 0.1
        elif atype == AttractorType.AIZAWA:
            self.aizawa_x, self.aizawa_y, self.aizawa_z = 0.1, 0.0, 0.0
        elif atype == AttractorType.THOMAS:
            self.thomas_x, self.thomas_y, self.thomas_z = 1.0, 1.0, 1.0

    # --- Lorenz Attractor (3D) ---
    def _lorenz_deriv(self, x, y, z):
        """Lorenz Attractor derivatives."""
        sigma = self.lorenz_sigma.value
        rho = self.lorenz_rho.value
        beta = self.lorenz_beta.value
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    def _step_lorenz(self, dt):
        """Runge-Kutta 4 integration for Lorenz."""
        x, y, z = self.lorenz_x, self.lorenz_y, self.lorenz_z
        k1x, k1y, k1z = self._lorenz_deriv(x, y, z)
        k2x, k2y, k2z = self._lorenz_deriv(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
        k3x, k3y, k3z = self._lorenz_deriv(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
        k4x, k4y, k4z = self._lorenz_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z)
        self.lorenz_x += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.lorenz_y += (dt/6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        self.lorenz_z += (dt/6.0) * (k1z + 2*k2z + 2*k3z + k4z)
        return self.lorenz_x, self.lorenz_y

    def _map_lorenz(self, x, y, scale):
        """Map Lorenz coordinates to screen."""
        x_min, x_max = -30.0, 30.0
        y_min, y_max = -30.0, 30.0
        sx = int((x - x_min) / (x_max - x_min) * self.width * scale / 5 + self.width * (1 - scale/5) / 2)
        sy = int((y - y_min) / (y_max - y_min) * self.height * scale / 5 + self.height * (1 - scale/5) / 2)
        return sx, sy

    # --- Clifford Attractor (2D) ---
    def _step_clifford(self):
        """Clifford attractor iteration: x' = sin(a*y) + c*cos(a*x), y' = sin(b*x) + d*cos(b*y)"""
        a = self.clifford_a.value
        b = self.clifford_b.value
        c = self.clifford_c.value
        d = self.clifford_d.value
        x, y = self.clifford_x, self.clifford_y
        new_x = math.sin(a * y) + c * math.cos(a * x)
        new_y = math.sin(b * x) + d * math.cos(b * y)
        self.clifford_x, self.clifford_y = new_x, new_y
        return new_x, new_y

    def _map_clifford(self, x, y, scale):
        """Map Clifford coordinates to screen (typically in [-3, 3] range)."""
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
        sx = int((x - x_min) / (x_max - x_min) * self.width * scale / 5 + self.width * (1 - scale/5) / 2)
        sy = int((y - y_min) / (y_max - y_min) * self.height * scale / 5 + self.height * (1 - scale/5) / 2)
        return sx, sy

    # --- De Jong Attractor (2D) ---
    def _step_dejong(self):
        """De Jong attractor iteration: x' = sin(a*y) - cos(b*x), y' = sin(c*x) - cos(d*y)"""
        a = self.dejong_a.value
        b = self.dejong_b.value
        c = self.dejong_c.value
        d = self.dejong_d.value
        x, y = self.dejong_x, self.dejong_y
        new_x = math.sin(a * y) - math.cos(b * x)
        new_y = math.sin(c * x) - math.cos(d * y)
        self.dejong_x, self.dejong_y = new_x, new_y
        return new_x, new_y

    def _map_dejong(self, x, y, scale):
        """Map De Jong coordinates to screen (typically in [-2.5, 2.5] range)."""
        x_min, x_max = -2.5, 2.5
        y_min, y_max = -2.5, 2.5
        sx = int((x - x_min) / (x_max - x_min) * self.width * scale / 5 + self.width * (1 - scale/5) / 2)
        sy = int((y - y_min) / (y_max - y_min) * self.height * scale / 5 + self.height * (1 - scale/5) / 2)
        return sx, sy

    # --- Aizawa Attractor (3D) ---
    def _aizawa_deriv(self, x, y, z):
        """Aizawa attractor derivatives."""
        a = self.aizawa_a.value
        b = self.aizawa_b.value
        c = self.aizawa_c.value
        d = self.aizawa_d.value
        e = self.aizawa_e.value
        f = self.aizawa_f.value
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - (z**3) / 3.0 - (x**2 + y**2) * (1 + e * z) + f * z * (x**3)
        return dx, dy, dz

    def _step_aizawa(self, dt):
        """Runge-Kutta 4 integration for Aizawa."""
        x, y, z = self.aizawa_x, self.aizawa_y, self.aizawa_z
        k1x, k1y, k1z = self._aizawa_deriv(x, y, z)
        k2x, k2y, k2z = self._aizawa_deriv(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
        k3x, k3y, k3z = self._aizawa_deriv(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
        k4x, k4y, k4z = self._aizawa_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z)
        self.aizawa_x += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.aizawa_y += (dt/6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        self.aizawa_z += (dt/6.0) * (k1z + 2*k2z + 2*k3z + k4z)
        return self.aizawa_x, self.aizawa_y

    def _map_aizawa(self, x, y, scale):
        """Map Aizawa coordinates to screen (typically in [-2, 2] range)."""
        x_min, x_max = -2.0, 2.0
        y_min, y_max = -2.0, 2.0
        sx = int((x - x_min) / (x_max - x_min) * self.width * scale / 5 + self.width * (1 - scale/5) / 2)
        sy = int((y - y_min) / (y_max - y_min) * self.height * scale / 5 + self.height * (1 - scale/5) / 2)
        return sx, sy

    # --- Thomas Attractor (3D) ---
    def _thomas_deriv(self, x, y, z):
        """Thomas attractor derivatives: smooth, ribbon-like trajectories."""
        b = self.thomas_b.value
        dx = math.sin(y) - b * x
        dy = math.sin(z) - b * y
        dz = math.sin(x) - b * z
        return dx, dy, dz

    def _step_thomas(self, dt):
        """Runge-Kutta 4 integration for Thomas."""
        x, y, z = self.thomas_x, self.thomas_y, self.thomas_z
        k1x, k1y, k1z = self._thomas_deriv(x, y, z)
        k2x, k2y, k2z = self._thomas_deriv(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
        k3x, k3y, k3z = self._thomas_deriv(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
        k4x, k4y, k4z = self._thomas_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z)
        self.thomas_x += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.thomas_y += (dt/6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        self.thomas_z += (dt/6.0) * (k1z + 2*k2z + 2*k3z + k4z)
        return self.thomas_x, self.thomas_y

    def _map_thomas(self, x, y, scale):
        """Map Thomas coordinates to screen (typically in [-5, 5] range)."""
        x_min, x_max = -5.0, 5.0
        y_min, y_max = -5.0, 5.0
        sx = int((x - x_min) / (x_max - x_min) * self.width * scale / 5 + self.width * (1 - scale/5) / 2)
        sy = int((y - y_min) / (y_max - y_min) * self.height * scale / 5 + self.height * (1 - scale/5) / 2)
        return sx, sy

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """Generates a Strange Attractor pattern based on selected type."""

        # Check if attractor type changed - reset state if so
        if self.attractor_type.value != self.prev_attractor_type:
            self._reset_current_attractor()
            self.prev_attractor_type = self.attractor_type.value
            frame = None  # Clear frame on type change

        if frame is None:
            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            pattern = (frame * self.fade.value).astype(np.uint8)

        dt = self.dt.value
        num_steps = int(self.num_steps.value)
        scale = self.scale.value
        line_width = int(self.line_width.value)
        color = (int(self.attractor_b.value), int(self.attractor_g.value), int(self.attractor_r.value))

        atype = self.attractor_type.value

        # Get initial screen position based on attractor type
        if atype == AttractorType.LORENZ:
            prev_sx, prev_sy = self._map_lorenz(self.lorenz_x, self.lorenz_y, scale)
        elif atype == AttractorType.CLIFFORD:
            prev_sx, prev_sy = self._map_clifford(self.clifford_x, self.clifford_y, scale)
        elif atype == AttractorType.DE_JONG:
            prev_sx, prev_sy = self._map_dejong(self.dejong_x, self.dejong_y, scale)
        elif atype == AttractorType.AIZAWA:
            prev_sx, prev_sy = self._map_aizawa(self.aizawa_x, self.aizawa_y, scale)
        elif atype == AttractorType.THOMAS:
            prev_sx, prev_sy = self._map_thomas(self.thomas_x, self.thomas_y, scale)
        else:
            prev_sx, prev_sy = self.width // 2, self.height // 2

        for _ in range(num_steps):
            # Step the attractor and get new screen coordinates
            if atype == AttractorType.LORENZ:
                x, y = self._step_lorenz(dt)
                curr_sx, curr_sy = self._map_lorenz(x, y, scale)
            elif atype == AttractorType.CLIFFORD:
                x, y = self._step_clifford()
                curr_sx, curr_sy = self._map_clifford(x, y, scale)
            elif atype == AttractorType.DE_JONG:
                x, y = self._step_dejong()
                curr_sx, curr_sy = self._map_dejong(x, y, scale)
            elif atype == AttractorType.AIZAWA:
                x, y = self._step_aizawa(dt)
                curr_sx, curr_sy = self._map_aizawa(x, y, scale)
            elif atype == AttractorType.THOMAS:
                x, y = self._step_thomas(dt)
                curr_sx, curr_sy = self._map_thomas(x, y, scale)
            else:
                curr_sx, curr_sy = prev_sx, prev_sy

            cv2.line(pattern, (prev_sx, prev_sy), (curr_sx, curr_sy), color, line_width)
            prev_sx, prev_sy = curr_sx, curr_sy

        return pattern
