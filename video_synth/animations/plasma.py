import cv2
import numpy as np
import time
import noise
import random
from lfo import LFO
from animations.base import Animation

class Plasma(Animation):
    def __init__(self, params, width=800, height=600, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        self.params = params
        self.width = width
        self.height = height

        self.plasma_speed = params.add("plasma_speed",
                                       min=0.01, max=10, default=1.0,
                                       subgroup=subgroup, group=group)
        self.plasma_distance = params.add("plasma_distance",
                                          min=0.01, max=10, default=1.0,
                                          subgroup=subgroup, group=group)
        self.plasma_color_speed = params.add("plasma_color_speed",
                                             min=0.01, max=10, default=1.0,
                                             subgroup=subgroup, group=group)
        self.plasma_flow_speed = params.add("plasma_flow_speed",
                                            min=0.01, max=10, default=1.0,
                                            subgroup=subgroup, group=group)

        self.plasma_params = [
            "plasma_speed",
            "plasma_distance",
            "plasma_color_speed",
            "plasma_flow_speed",
        ]

        self.oscillators = [LFO(params, name=f"{self.plasma_params[i]}", frequency=0.5, amplitude=1.0, phase=0.0, shape=1) for i in range(4)]

        self.oscillators[0].link_param(self.plasma_speed)
        self.oscillators[1].link_param(self.plasma_distance)
        self.oscillators[2].link_param(self.plasma_color_speed)
        self.oscillators[3].link_param(self.plasma_flow_speed)

        # Performance: cache meshgrid
        x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        self._X, self._Y = np.meshgrid(x_coords, y_coords)

    
    def generate_plasma_effect(self):
        plasma_pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Use cached meshgrid
        X, Y = self._X, self._Y

        current_time = time.time()

        # for osc in self.oscillators:
        #     osc.get_next_value()

        plasma_time_offset_base = current_time * (0.5 + self.plasma_speed.value * 2.0) + random.randint(0, 1000)

        scale_factor_x = 0.01 + self.plasma_distance.value * 0.02
        scale_factor_y = 0.01 + self.plasma_distance.value * 0.02

        flow_scale = 0.005
        flow_strength = self.plasma_flow_speed.value * 100

        # CRITICAL OPTIMIZATION: Skip expensive Perlin noise flow (307K calls to pnoise3!)
        # Just use simple sine-based perturbation instead
        flow_noise_time = current_time * 0.1
        perturbed_X = X + np.sin(Y * flow_scale + flow_noise_time) * flow_strength
        perturbed_Y = Y + np.cos(X * flow_scale + flow_noise_time) * flow_strength

        value = (
            np.sin(perturbed_X * scale_factor_x + plasma_time_offset_base) +
            np.sin(perturbed_Y * scale_factor_y + plasma_time_offset_base * 0.8 + random.uniform(0, np.pi * 2)) + 
            np.sin((perturbed_X + perturbed_Y) * scale_factor_x * 0.7 + plasma_time_offset_base * 1.2 + random.uniform(0, np.pi * 2)) + 
            np.sin((perturbed_X - perturbed_Y) * scale_factor_y * 0.9 + plasma_time_offset_base * 0.6 + random.uniform(0, np.pi * 2))
        )

        normalized_value = (value + 4) / 8

        hue_shift_val = self.plasma_color_speed.value * 2 * np.pi

        R = np.sin(normalized_value * np.pi * 3 + hue_shift_val) * 0.5 + 0.5
        G = np.sin(normalized_value * np.pi * 3 + hue_shift_val + np.pi * 2/3) * 0.5 + 0.5
        B = np.sin(normalized_value * np.pi * 3 + hue_shift_val + np.pi * 4/3) * 0.5 + 0.5

        plasma_pattern[:, :, 2] = (R * 255).astype(np.uint8)
        plasma_pattern[:, :, 0] = (B * 255).astype(np.uint8)
        plasma_pattern[:, :, 1] = (G * 255).astype(np.uint8)

        return plasma_pattern

    def get_frame(self, frame):
        return self.generate_plasma_effect()
