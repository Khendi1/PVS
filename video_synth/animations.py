import cv2
import numpy as np
import time
import noise
import random
from lfo import LFO
from abc import ABC, abstractmethod
from enum import IntEnum, auto
import logging
import math
from common import *
import moderngl
from enum import Enum, IntEnum, auto


log = logging.getLogger(__name__)

class Colormap(IntEnum):
    JET = 0
    VIRIDIS = auto()
    MAGMA = auto()
    PLASMA = auto()
    RAINBOW = auto()
    OCEAN = auto()
    SPRING = auto()
    COOL = auto()

COLORMAP_OPTIONS = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_RAINBOW,
    cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SPRING,
    cv2.COLORMAP_COOL,
]


class MoirePattern(IntEnum):
    LINE = 0
    RADIAL = auto()
    GRID = auto()


class MoireBlend(IntEnum):
    MULTIPLY = 0
    ADD = auto()
    SUB = auto()


class ShaderType(IntEnum):
    FRACTAL_0 = 0
    FRACTAL = auto()
    GRID = auto()
    PLASMA = auto()
    CLOUD = auto()
    MANDALA = auto()
    GALAXY = auto()
    TECTONIC = auto()
    BIOLUMINESCENT = auto()
    AURORA = auto()
    CRYSTAL = auto()


class AttractorType(IntEnum):
    LORENZ = 0
    CLIFFORD = auto()
    DE_JONG = auto()
    AIZAWA = auto()
    THOMAS = auto()


class AnimSource(IntEnum):
    PLASMA = 1
    REACTION_DIFFUSION = 2
    METABALLS = 3
    MOIRE = 4
    STRANGE_ATTRACTOR = 5
    PHYSARUM = 6
    SHADERS = 7
    DLA = 8
    CHLADNI = 9
    VORONOI = 10

# ---- End Enum classes, begin animation base&concrete classes

class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, params, width=640, height=480, group=None):
        self.params = params
        self.width = width
        self.height = height
        self.group = group


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("subgroupes should implement this method.")


class Plasma(Animation):
    def __init__(self, params, width=800, height=600, group=None):
        super().__init__(params, group=group)
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

    
    def generate_plasma_effect(self):
        plasma_pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords)

        current_time = time.time()

        # for osc in self.oscillators:
        #     osc.get_next_value()

        plasma_time_offset_base = current_time * (0.5 + self.plasma_speed.value * 2.0) + random.randint(0, 1000)

        scale_factor_x = 0.01 + self.plasma_distance.value * 0.02
        scale_factor_y = 0.01 + self.plasma_distance.value * 0.02

        flow_scale = 0.005
        flow_strength = self.plasma_flow_speed.value * 100

        noise_x_perturb = np.zeros_like(X)
        noise_y_perturb = np.zeros_like(Y)

        flow_noise_time = current_time * 0.1

        random_base_x = random.randint(0, 1000)
        random_base_y = random.randint(0, 1000) + 500

        for y in range(self.height):
            for x in range(self.width):
                nx = x * flow_scale
                ny = y * flow_scale

                noise_x_perturb[y, x] = noise.pnoise3(nx, ny, flow_noise_time, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_x)
                noise_y_perturb[y, x] = noise.pnoise3(nx + 100, ny + 100, flow_noise_time + 100, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_y)
        
        perturbed_X = X + noise_x_perturb * flow_strength
        perturbed_Y = Y + noise_y_perturb * flow_strength

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


class ReactionDiffusion(Animation):

    def __init__(self, params, width=500, height=500, group=None):
        super().__init__(params, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        da=1.0
        db=0.5
        feed=0.055
        kill=0.062
        randomize_seed=False
        max_seed_size=50
        num_seeds=15

        self.da = params.add("da",
                             min=0, max=2.0, default=da,
                             subgroup=subgroup, group=group)
        self.db = params.add("db",
                             min=0, max=2.0, default=db,
                             subgroup=subgroup, group=group)

        self.feed = params.add("feed",
                               min=0, max=0.1, default=feed,
                               subgroup=subgroup, group=group)
        self.kill = params.add("kill",
                               min=0, max=0.1, default=kill,
                               subgroup=subgroup, group=group)
        self.iterations_per_frame = params.add("iterations_per_frame",
                                               min=5, max=100, default=50,
                                               subgroup=subgroup, group=group)
        
        self.dt = 0.15
        self.current_A = np.ones((height, width), dtype=np.float32)
        self.current_B = np.zeros((height, width), dtype=np.float32)
        self.next_A = np.copy(self.current_A)
        self.next_B = np.copy(self.current_B)
        
        self.randomize_seed = randomize_seed
        self.max_seed_size = max_seed_size
        self.num_seeds = num_seeds 

        self.initialize_seed()

    def initialize_seed(self):
        self.current_A.fill(1.0)
        self.current_B.fill(0.0)

        if self.randomize_seed:
            for _ in range(self.num_seeds):
                seed_size = random.randint(5, self.max_seed_size)
                center_x = random.randint(seed_size // 2, self.width - seed_size // 2 - 1)
                center_y = random.randint(seed_size // 2, self.height - seed_size // 2 - 1)
                
                self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
                self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0
        else:
            seed_size = 20
            center_x, center_y = self.width // 2, self.height // 2
            
            self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
            self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0


    def update_simulation(self):
        lap_A = (
            np.roll(self.current_A, 1, axis=0) +
            np.roll(self.current_A, -1, axis=0) +
            np.roll(self.current_A, 1, axis=1) +
            np.roll(self.current_A, -1, axis=1) -
            4 * self.current_A
        )

        lap_B = (
            np.roll(self.current_B, 1, axis=0) +
            np.roll(self.current_B, -1, axis=0) +
            np.roll(self.current_B, 1, axis=1) +
            np.roll(self.current_B, -1, axis=1) -
            4 * self.current_B
        )

        diff_A = self.da.value * lap_A - self.current_A * self.current_B**2 + self.feed.value * (1 - self.current_A)
        diff_B = self.db.value * lap_B + self.current_A * self.current_B**2 - (self.kill.value + self.feed.value) * self.current_B

        self.next_A = np.clip(self.current_A + diff_A * self.dt, 0.0, 1.0)
        self.next_B = np.clip(self.current_B + diff_B * self.dt, 0.0, 1.0)

        self.current_A, self.current_B = self.next_A, self.next_B


    def run(self):
        for _ in range(self.iterations_per_frame.value):
            self.update_simulation()

        hue = (self.current_A * 120).astype(np.uint8) 
        saturation = (self.current_B * 255).astype(np.uint8)
        value = ((self.current_A + self.current_B) / 2 * 255).astype(np.uint8)
        hsv_image = cv2.merge([hue, saturation, value])
        
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def get_frame(self, frame):
        return self.run()


class Metaballs(Animation):
    def __init__(self, params, width=800, height=600, group=None):
        super().__init__(params, group=group)
        subgroup=self.__class__.__name__
        p_name = group.name.lower()
        self.metaballs = []
        
        self.num_metaballs = params.add("num_metaballs",
                                        min=2, max=10, default=5,
                                        subgroup=subgroup, group=group)
        self.min_radius = params.add("min_radius",
                                     min=20, max=100, default=40,
                                     subgroup=subgroup, group=group)
        self.max_radius = params.add("max_radius",
                                     min=40, max=200, default=80,
                                     subgroup=subgroup, group=group)
        self.radius_multiplier = params.add("radius_multiplier",
                                            min=1.0, max=3.0, default=1.0,
                                            subgroup=subgroup, group=group)
        self.max_speed = params.add("max_speed",
                                    min=1, max=10, default=3,
                                    subgroup=subgroup, group=group)
        self.speed_multiplier = params.add("speed_multiplier",
                                           min=1.0, max=3.0, default=1.0,
                                           subgroup=subgroup, group=group)
        self.threshold = params.add("threshold",
                                    min=0.5, max=3.0, default=1.6,
                                    subgroup=subgroup, group=group)
        self.smooth_coloring_max_field = params.add("smooth_coloring_max_field",
                                                    min=1.0, max=3.0, default=1.5,
                                                    subgroup=subgroup, group=group)
        self.skew_angle = params.add("metaball_skew_angle",
                                     min=0.0, max=360.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.skew_intensity = params.add("metaball_skew_intensity",
                                         min=0.0, max=1.0, default=0.0,
                                         subgroup=subgroup, group=group)
        self.zoom = params.add("metaball_zoom",
                               min=1.0, max=3.0, default=1.0,
                               subgroup=subgroup, group=group)
        self.colormap = params.add("metaball_colormap",
                                   min=0, max=len(COLORMAP_OPTIONS)-1, default=0,
                                   group=group, subgroup=subgroup,
                                   type=Widget.DROPDOWN, options=Colormap)
        self.feedback_alpha = params.add("metaballs_feedback",
                                         min=0.0, max=1.0, default=0.95,
                                         subgroup=subgroup, group=group)

        self.current_num_metaballs = self.num_metaballs.value
        self.current_radius_multiplier = self.radius_multiplier.value
        self.current_speed_multiplier = self.speed_multiplier.value
        self.previous_frame = None

        self.setup_metaballs()

    def adjusteters(self):
        if self.current_radius_multiplier != self.radius_multiplier.value:
            for ball in self.metaballs:
                ball['radius'] = int(ball['radius'] * self.radius_multiplier.value / self.current_radius_multiplier)
            self.current_radius_multiplier = self.radius_multiplier.value

        if self.current_speed_multiplier != self.speed_multiplier.value:
            for ball in self.metaballs:
                ball['vx'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
                ball['vy'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
            self.current_speed_multiplier = self.speed_multiplier.value

    def setup_metaballs(self):
        num_metaballs = self.num_metaballs.value
        if len(self.metaballs) > num_metaballs:
            self.metaballs = self.metaballs[:num_metaballs]
        else:
            delta = num_metaballs - len(self.metaballs)
            for _ in range(delta):
                r = np.random.randint(self.min_radius.value, self.max_radius.value) * self.radius_multiplier.value
                x = np.random.randint(self.max_radius.value, self.width - self.max_radius.value)
                y = np.random.randint(self.max_radius.value, self.height - self.max_radius.value)
                vx = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                vy = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                self.metaballs.append({'x': x, 'y': y, 'radius': r, 'vx': vx, 'vy': vy})
        self.current_num_metaballs = num_metaballs

    def create_metaball_frame(self, metaballs, threshold, max_field_strength=None):
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        X, Y = np.meshgrid(x_coords, y_coords)

        center_x, center_y = self.width / 2, self.height / 2
        X_centered = X - center_x
        Y_centered = Y - center_y
        X_processed = X_centered / self.zoom.value
        Y_processed = Y_centered / self.zoom.value
        if self.skew_intensity.value > 0:
            angle_rad = np.radians(self.skew_angle.value)
            X_processed += Y_processed * self.skew_intensity.value * np.cos(angle_rad)
            Y_processed += X_processed * self.skew_intensity.value * np.sin(angle_rad)
        X_transformed = X_processed + center_x
        Y_transformed = Y_processed + center_y
        
        field_strength = np.zeros((self.height, self.width), dtype=np.float32)
        for ball in metaballs:
            mx, my, r = ball['x'], ball['y'], ball['radius']
            dist_sq = (X_transformed - mx)**2 + (Y_transformed - my)**2 + 1e-6
            field_strength += (r**2) / dist_sq

        if max_field_strength is not None:
            normalized_field = np.clip(field_strength / max_field_strength, 0, 1)
            grayscale_image = (normalized_field * 255).astype(np.uint8)
            image = cv2.applyColorMap(grayscale_image, COLORMAP_OPTIONS[self.colormap.value])
        else:
            image = ((field_strength >= threshold) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image

    def do_metaballs(self, frame: np.ndarray):
        if self.num_metaballs.value != self.current_num_metaballs:
            self.setup_metaballs()
        
        if self.current_radius_multiplier != self.radius_multiplier.value or self.current_speed_multiplier != self.speed_multiplier.value:
            self.adjusteters()

        for ball in self.metaballs:
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']

            if not (ball['radius'] < ball['x'] < self.width - ball['radius']):
                ball['vx'] *= -1
            if not (ball['radius'] < ball['y'] < self.height - ball['radius']):
                ball['vy'] *= -1

        current_frame = self.create_metaball_frame(self.metaballs,
                                                threshold=self.threshold.value,
                                                max_field_strength=self.smooth_coloring_max_field.value)
        
        if self.previous_frame is None:
            self.previous_frame = current_frame.astype(np.float32)
        else:
            current_frame = cv2.addWeighted(current_frame.astype(np.float32), 1-self.feedback_alpha.value, 
                                            self.previous_frame, self.feedback_alpha.value, 0)
            self.previous_frame = current_frame

        return current_frame

    def get_frame(self, frame: np.ndarray = None):
        return self.do_metaballs(frame)


class Moire(Animation):
    def __init__(self, params, width=800, height=600, group=None):
        super().__init__(params, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        
        self.blend_mode = params.add("moire_blend",
                                      min=0, max=len(MoireBlend)-1, default=0,
                                      group=group, subgroup=subgroup,
                                      type=Widget.DROPDOWN, options=MoireBlend)

        center_x, center_y = self.width//2, self.height//2

        self.pattern_1 = params.add("moire_type_1",
                                    min=0, max=len(MoirePattern)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=MoirePattern)
        self.freq_1 = params.add("spatial_freq_1",
                                 min=0.01, max=25, default=10.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER)
        self.angle_1 = params.add("angle_1",
                                  min=0, max=360, default=90.0,
                                  group=group, subgroup=subgroup,
                                  type=Widget.SLIDER)
        self.zoom_1 = params.add("zoom_1",
                                 min=0.05, max=1.5, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER)
        self.center_x_1 = params.add("moire_center_x_1",
                                     min=0, max=self.width, default=center_x,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER)
        self.center_y_1 = params.add("moire_center_y_1",
                                     min=0, max=self.height, default=center_y,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER)

        self.pattern_2 = params.add("moire_type_2",
                                    min=0, max=len(MoirePattern)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=MoirePattern)
        self.freq_2 = params.add("spatial_freq_2",
                                 min=0.01, max=25, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER)
        self.angle_2 = params.add("angle_2",
                                  min=0, max=360, default=0.0,
                                  group=group, subgroup=subgroup,
                                  type=Widget.SLIDER)
        self.zoom_2 = params.add("zoom_2",
                                 min=0.05, max=1.5, default=1.0,
                                 group=group, subgroup=subgroup,
                                 type=Widget.SLIDER)
        self.center_x_2 = params.add("moire_center_x_2",
                                     min=0, max=self.width, default=center_x,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER)
        self.center_y_2 = params.add("moire_center_y_2",
                                     min=0, max=self.height, default=center_y,
                                     group=group, subgroup=subgroup,
                                     type=Widget.SLIDER)

    def _generate_single_pattern(self, X_shifted, Y_shifted, frequency, angle_rad, zoom, pattern_type):
        angle_rad = math.radians(angle_rad)
        X_z, Y_z = X_shifted * zoom, Y_shifted * zoom

        if pattern_type == MoirePattern.LINE.value:
            P = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            pattern = np.sin(P * frequency/2)
        elif pattern_type == MoirePattern.RADIAL.value:
            R = np.sqrt(X_z**2 + Y_z**2)
            pattern = np.sin(R * frequency/2)
        elif pattern_type == MoirePattern.GRID.value:
            freq_x = frequency * (1.0 + np.sin(angle_rad) * .01)
            freq_y = frequency * (1.0 + np.cos(angle_rad) * .01)
            pattern = np.sin(X_z * freq_x) + np.sin(Y_z * freq_y)
        else:
            pattern = np.zeros_like(X_shifted)

        SCALE_FACTOR = 127.5
        return (pattern * (SCALE_FACTOR / (1 if pattern_type != MoirePattern.GRID.value else 2))) + SCALE_FACTOR

    def get_frame(self, frame):
        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

        x1, y1 = X - self.center_x_1.value, Y - self.center_y_1.value
        x2, y2 = X - self.center_x_2.value, Y - self.center_y_2.value

        pattern1_float = self._generate_single_pattern(
            x1, y1, self.freq_1.value, self.angle_1.value, self.zoom_1.value, self.pattern_1.value
        )
        pattern2_float = self._generate_single_pattern(
            x2, y2, self.freq_2.value, self.angle_2.value, self.zoom_2.value, self.pattern_2.value
        )

        blend_mode = self.blend_mode.value
        if blend_mode == MoireBlend.MULTIPLY.value:
            combined_pattern = (pattern1_float / 255.0) * (pattern2_float / 255.0)
            moire_image = (combined_pattern * 255).astype(np.uint8)
        elif blend_mode == MoireBlend.ADD.value:
            combined_pattern = pattern1_float + pattern2_float
            moire_image = np.clip(combined_pattern, 0, 255).astype(np.uint8)
        elif blend_mode == MoireBlend.SUB.value:
            combined_pattern = pattern1_float - pattern2_float
            moire_image = np.clip(combined_pattern, 0, 255).astype(np.uint8)
        else:
            moire_image = pattern1_float.astype(np.uint8)

        moire_image = cv2.equalizeHist(moire_image)
        return cv2.cvtColor(moire_image, cv2.COLOR_GRAY2BGR)


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
        log.debug("Entering StrangeAttractor.get_frame")

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


class Physarum(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Simulation parameters
        self.num_agents = params.add("phys_num_agents",
                                     min=1000, max=10000, default=1000,
                                     subgroup=subgroup, group=group)
        self.sensor_angle_spacing = params.add("phys_sensor_angle_spacing",
                                               min=0.0, max=np.pi/2, default=np.pi/8,
                                               subgroup=subgroup, group=group) # Radians
        self.sensor_distance = params.add("phys_sensor_distance",
                                          min=1, max=20, default=9,
                                          subgroup=subgroup, group=group)
        self.turn_angle = params.add("phys_turn_angle",
                                     min=0.0, max=np.pi/2, default=np.pi/4,
                                     subgroup=subgroup, group=group) # Radians
        self.step_distance = params.add("phys_step_distance",
                                        min=1, max=10, default=1,
                                        subgroup=subgroup, group=group)
        self.decay_factor = params.add("phys_decay_factor",
                                       min=0.0, max=1.0, default=0.1,
                                       subgroup=subgroup, group=group)
        self.diffuse_factor = params.add("phys_diffuse_factor",
                                         min=0.0, max=1.0, default=0.5,
                                         subgroup=subgroup, group=group)
        self.deposit_amount = params.add("phys_deposit_amount",
                                         min=0.1, max=5.0, default=1.0,
                                         subgroup=subgroup, group=group)
        self.grid_resolution_scale = params.add("phys_grid_res_scale",
                                                min=0.1, max=1.0, default=0.5,
                                                subgroup=subgroup, group=group)
        self.wrap_around = params.add("phys_wrap_around",
                                      min=0, max=1, default=1,
                                      group=group, subgroup=subgroup,
                                      type=Widget.RADIO, options=Toggle) # Boolean as int

        # Color parameters
        self.trail_r = params.add("phys_trail_r",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.trail_g = params.add("phys_trail_g",
                                  min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.trail_b = params.add("phys_trail_b",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_r = params.add("phys_agent_r",
                                  min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.agent_g = params.add("phys_agent_g",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_b = params.add("phys_agent_b",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_size = params.add("phys_agent_size",
                                     min=1, max=5, default=1,
                                     subgroup=subgroup, group=group)


        self.grid_width = int(self.width * self.grid_resolution_scale.value)
        self.grid_height = int(self.height * self.grid_resolution_scale.value)

        self.trail_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.agents = self._initialize_agents()

        # Store previous parameter values to detect changes for re-initialization
        self.prev_num_agents = self.num_agents.value
        self.prev_grid_resolution_scale = self.grid_resolution_scale.value

    def _reinitialize_simulation(self):
        log.debug("Reinitializing Physarum simulation due to parameter change.")
        self.grid_width = int(self.width * self.grid_resolution_scale.value)
        self.grid_height = int(self.height * self.grid_resolution_scale.value)
        self.trail_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.agents = self._initialize_agents()

    def _initialize_agents(self):
        # Agents: [x, y, angle]
        agents = np.zeros((self.num_agents.value, 3), dtype=np.float32)
        agents[:, 0] = np.random.uniform(0, self.grid_width, self.num_agents.value)   # x
        agents[:, 1] = np.random.uniform(0, self.grid_height, self.num_agents.value)  # y
        agents[:, 2] = np.random.uniform(0, 2 * np.pi, self.num_agents.value)         # angle
        return agents

    def _sense(self, agent_x, agent_y, agent_angle):
        # Sensor angles for front-left, front, front-right
        angle_f = agent_angle
        angle_r = agent_angle + self.sensor_angle_spacing.value
        angle_l = agent_angle - self.sensor_angle_spacing.value

        # Sensor positions
        dist = self.sensor_distance.value
        xf, yf = agent_x + dist * np.cos(angle_f), agent_y + dist * np.sin(angle_f)
        xr, yr = agent_x + dist * np.cos(angle_r), agent_y + dist * np.sin(angle_r)
        xl, yl = agent_x + dist * np.cos(angle_l), agent_y + dist * np.sin(angle_l)

        # Ensure sensor positions are within bounds or wrap around
        xf, yf = self._get_safe_coords(xf, yf)
        xr, yr = self._get_safe_coords(xr, yr)
        xl, yl = self._get_safe_coords(xl, yl)

        # Read trail map at sensor positions
        # Ensure coordinates are integers before indexing
        return (
            self.trail_map[int(yf), int(xf)],
            self.trail_map[int(yr), int(xr)],
            self.trail_map[int(yl), int(yl)]
        )

    def _get_safe_coords(self, x, y):
        if self.wrap_around.value:
            x = x % self.grid_width
            y = y % self.grid_height
        else:
            x = np.clip(x, 0, self.grid_width - 1)
            y = np.clip(y, 0, self.grid_height - 1)
        return x, y

    def _move_agents(self):
        new_angles = np.copy(self.agents[:, 2]) # Start with current angles

        for i, agent in enumerate(self.agents):
            x, y, angle = agent
            val_f, val_r, val_l = self._sense(x, y, angle)

            # Decision making based on sensed values
            if val_f > val_l and val_f > val_r:
                # Move forward (no change to angle)
                pass
            elif val_l > val_f and val_l > val_r:
                # Turn left
                new_angles[i] -= self.turn_angle.value
            elif val_r > val_f and val_r > val_l:
                # Turn right
                new_angles[i] += self.turn_angle.value
            else:
                # Random turn if no clear direction
                new_angles[i] += np.random.uniform(-self.turn_angle.value, self.turn_angle.value)

        self.agents[:, 2] = new_angles % (2 * np.pi) # Update angles and wrap around 2pi

        # Update positions
        step = self.step_distance.value
        self.agents[:, 0] += step * np.cos(self.agents[:, 2])
        self.agents[:, 1] += step * np.sin(self.agents[:, 2])

        # Apply boundary conditions
        if self.wrap_around.value:
            self.agents[:, 0] = self.agents[:, 0] % self.grid_width
            self.agents[:, 1] = self.agents[:, 1] % self.grid_height
        else:
            self.agents[:, 0] = np.clip(self.agents[:, 0], 0, self.grid_width - 1)
            self.agents[:, 1] = np.clip(self.agents[:, 1], 0, self.grid_height - 1)

    def _deposit_trails(self):
        # Deposit chemical at agent's current position
        for agent in self.agents:
            x, y = int(agent[0]), int(agent[1])
            # Ensure y, x are within bounds after potential clipping in _get_safe_coords
            y = np.clip(y, 0, self.grid_height - 1)
            x = np.clip(x, 0, self.grid_width - 1)
            self.trail_map[y, x] += self.deposit_amount.value
            self.trail_map[y, x] = np.clip(self.trail_map[y, x], 0.0, 1.0) # Clip to max trail value

    def _diffuse_and_decay(self):
        # Decay
        self.trail_map *= (1.0 - self.decay_factor.value)

        # Diffusion (simple blur)
        if self.diffuse_factor.value > 0:
            kernel_size = 3 # For a simple blur
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.trail_map = cv2.GaussianBlur(self.trail_map, (kernel_size, kernel_size), self.diffuse_factor.value)

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        log.debug("Entering Physarum.get_frame")

        # Check for re-initialization based on parameters
        current_num_agents = self.num_agents.value
        current_grid_res_scale = self.grid_resolution_scale.value

        if (current_num_agents != self.prev_num_agents or
            current_grid_res_scale != self.prev_grid_resolution_scale):
            
            self._reinitialize_simulation()
            # Update previous values
            self.prev_num_agents = current_num_agents
            self.prev_grid_resolution_scale = current_grid_res_scale

        # Update simulation
        self._move_agents()
        self._deposit_trails()
        self._diffuse_and_decay()

        # Render to frame
        if frame is None:
            output_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            output_frame = frame.copy()

        # Scale trail_map to full frame size
        display_trail_map = cv2.resize(self.trail_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Color the trails
        trail_color_bgr = (self.trail_b.value, self.trail_g.value, self.trail_r.value)
        colored_trails = np.zeros_like(output_frame, dtype=np.uint8)
        colored_trails[:,:,0] = (display_trail_map * trail_color_bgr[0]).astype(np.uint8)
        colored_trails[:,:,1] = (display_trail_map * trail_color_bgr[1]).astype(np.uint8)
        colored_trails[:,:,2] = (display_trail_map * trail_color_bgr[2]).astype(np.uint8)

        # Blend trails onto the frame
        output_frame = cv2.addWeighted(output_frame, 1.0, colored_trails, 1.0, 0)

        # Draw agents
        agent_color_bgr = (self.agent_b.value, self.agent_g.value, self.agent_r.value)
        agent_radius = self.agent_size.value
        for agent in self.agents:
            # Map agent coordinates from simulation grid to display grid
            display_x = int(agent[0] / self.grid_resolution_scale.value)
            display_y = int(agent[1] / self.grid_resolution_scale.value)
            cv2.circle(output_frame, (display_x, display_y), agent_radius, agent_color_bgr, -1)
        
        return output_frame


class Shaders(Animation):
    def __init__(self, params, width=1280, height=720, group=None):
        super().__init__(params, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        self.width, self.height = width, height
        self.ctx = moderngl.create_context(standalone=True)
        self.time = time.time()

        self.current_shader = params.add("s_type",
                                         min=0, max=len(ShaderType)-1, default=0,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options=ShaderType)
        self.zoom = params.add("s_zoom",
                               min=0.1, max=5.0, default=1.5,
                               subgroup=subgroup, group=group)
        self.distortion = params.add("s_distortion",
                                     min=0.0, max=1.0, default=0.5,
                                     subgroup=subgroup, group=group)
        self.iterations = params.add("s_iterations",
                                     min=1.0, max=10.0, default=4.0,
                                     subgroup=subgroup, group=group)
        self.color_shift = params.add("s_color_shift",
                                      min=0.5, max=3.0, default=1.0,
                                      subgroup=subgroup, group=group)
        self.brightness = params.add("s_brightness",
                                     min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.hue_shift = params.add("s_hue_shift",
                                    min=0.0, max=7, default=0.0,
                                    subgroup=subgroup, group=group)
        self.saturation = params.add("s_saturation",
                                     min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.x_shift = params.add("s_x_shift",
                                  min=-5.0, max=5.0, default=0.0,
                                  subgroup=subgroup, group=group)
        self.y_shift = params.add("s_y_shift",
                                  min=-5.0, max=5.0, default=0.0,
                                  subgroup=subgroup, group=group)
        self.rotation = params.add("s_rotation",
                                   min=-3.14, max=3.14, default=0.0,
                                   subgroup=subgroup, group=group)
        self.speed = params.add("s_speed",
                                min=0.0, max=2.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.prev_shader = self.current_shader.value
        
        vertex_shader, code = self.get_shader_code()
        
        self.programs = {}
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        
        for name, frag_code in code.items():
            try:
                prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=frag_code)
                vao = self.ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])
                self.programs[name] = {'prog': prog, 'vao': vao}
            except Exception as e:
                log.error(f"Error compiling shader {name.name}: {e}")

        self.fbo_texture = self.ctx.texture((width, height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])
        
    def render(self, params):
        if self.current_shader.value not in self.programs:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        prog = self.programs[self.current_shader.value]['prog']
        vao = self.programs[self.current_shader.value]['vao']
        
        for name, value in params.items():
            if name in prog:
                prog[name].value = value
        
        self.ctx.clear(0, 0, 0)
        self.fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)
        
        data = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        img = data.reshape((self.height, self.width, 3))
        return cv2.flip(img, 0)

    def get_frame(self, frame: np.ndarray = None):
        if self.current_shader.value != self.prev_shader:
            self.prev_shader = self.current_shader.value

        params = {
            'u_resolution': (self.width, self.height),
            'u_time': (time.time() - self.time) * self.speed.value,
            'u_zoom': self.zoom.value,
            'u_distortion': self.distortion.value,
            'u_iterations': self.iterations.value,
            'u_color_shift': self.color_shift.value,
            'u_brightness': self.brightness.value,
            'u_hue_shift': self.hue_shift.value,
            'u_saturation': self.saturation.value,
            'u_scroll_x': self.x_shift.value,
            'u_scroll_y': self.y_shift.value,
            'u_rotation': self.rotation.value,
        }
        return self.render(params)

    def get_shader_code(self):
        vertex_shader = '''
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        '''
        code = {
            ShaderType.FRACTAL_0: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;

                out vec4 f_color;

                vec3 palette(float t) {
                    vec3 a = vec3(0.5, 0.5, 0.5);
                    vec3 b = vec3(0.5, 0.5, 0.5);
                    vec3 c = vec3(1.0, 1.0, 1.0);
                    vec3 d = vec3(0.263, 0.416, 0.557);
                    return a + b * cos(6.28318 * (c * t * u_color_shift + d + u_hue_shift));
                }

                mat2 rotate2d(float angle){
                    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
                    
                    // Apply rotation
                    uv *= rotate2d(u_rotation);
                    
                    // Apply scrolling
                    uv += vec2(u_scroll_x, u_scroll_y);
                    
                    vec2 uv0 = uv;
                    vec3 finalColor = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        uv = fract(uv * u_zoom) - 0.5;
                        float d = length(uv) * exp(-length(uv0));
                        vec3 col = palette(length(uv0) + i * 0.4 + u_time * 0.4);
                        d = sin(d * 8.0 + u_time) / 8.0;
                        d = abs(d);
                        d = pow(0.01 / d, 1.2 + u_distortion * 0.5);
                        finalColor += col * d;
                    }
                    
                    // Apply saturation
                    float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                    finalColor = mix(vec3(gray), finalColor, u_saturation);
                    
                    f_color = vec4(finalColor * u_brightness, 1.0);
                }''',
            ShaderType.FRACTAL: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                out vec4 f_color;

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28 * (t * u_color_shift + vec3(0.0, 0.33, 0.67)));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    vec2 uv0 = uv;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        uv = fract(uv * u_zoom) - 0.5;
                        float d = length(uv) * exp(-length(uv0));
                        col += palette(length(uv0) + i * 0.4 + u_time * 0.4) * (0.01 / abs(d));
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.GRID: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        vec2 gv = fract(uv * rot(u_time * (1.0 + u_distortion) * 0.2 + i)) - 0.5;
                        float d = length(gv);
                        col += (0.5 + 0.5 * cos(u_time + vec3(0,2,4))) * smoothstep(0.4, 0.1, d);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.PLASMA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0;
                    uv.x *= u_resolution.x / u_resolution.y;
                    uv *= u_zoom * 5.0;
                    
                    float v = 0.0;
                    for (float i = 1.0; i <= u_iterations; i++) {
                        v += sin(uv.x * i + u_time) + sin(uv.y * i + u_time);
                        uv = mat2(cos(i), -sin(i), sin(i), cos(i)) * uv;
                    }
                    
                    vec3 col = 0.5 + 0.5 * cos(v * 3.14 + u_time + vec3(0, 2, 4));
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.CLOUD: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    float t = u_time * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 1.0; i <= u_iterations; i++) {
                        uv += sin(uv.yx * (2.0 + u_distortion) + t + i) * 0.4;
                        uv *= rot(t * 0.05 + i);
                        float d = length(uv);
                        float val = smoothstep(0.0, 0.8, abs(sin(d * (10.0 + u_distortion) - t * 2.0)));
                        col += (0.5 + 0.5 * cos(t + d + vec3(0, 2, 4))) * val / i;
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.MANDALA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float breath = sin(u_time * 0.2) * 0.2 + 1.0;
                    uv *= u_zoom * 2.0 * breath;
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float t = u_time * 0.1 * (i + 1.0) * (1.0 + u_distortion * 0.5);
                        float segments = 6.0 + i * 2.0;
                        float a = mod(angle + t, 6.28 / segments) - 3.14 / segments;
                        vec2 p = vec2(cos(a), sin(a)) * dist;
                        float d = sin(length(p - vec2(0.5, 0.0)) * 20.0 - u_time * 2.0);
                        col += (0.5 + 0.5 * cos(u_time * 0.5 + i + dist + vec3(0, 2, 4))) * smoothstep(0.1, 0.2, abs(d));
                    }
                    col *= 1.0 - smoothstep(0.5, 1.5, dist);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.GALAXY: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    float t = u_time * 0.5;
                    uv *= rot(t * 0.1);
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float armAngle = angle + dist * (2.0 + u_distortion) - t * (0.3 + i * 0.05);
                        float armDist = sin(armAngle * (3.0 + i)) * 0.3;
                        float arm = smoothstep(0.2, 0.0, abs(dist - 0.5 - armDist));
                        
                        vec2 starCoord = vec2(angle * 5.0 + i, dist * 10.0);
                        float stars = smoothstep(0.98, 1.0, hash(floor(starCoord + t * 0.1))) * smoothstep(0.3, 0.8, dist);
                        
                        col += (0.5 + 0.5 * cos(t * 0.3 + i * 2.0 + vec3(0, 2, 4))) * arm + vec3(1.0, 0.95, 0.9) * stars;
                    }
                    col += vec3(1.0, 0.8, 0.6) * exp(-dist * 3.0) * 0.5;
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.TECTONIC: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                float fbm(vec2 p) {
                    float v = 0.0, a = 0.5;
                    for (int i = 0; i < 6; i++) { v += a * hash(p); p *= 2.0; a *= 0.5; }
                    return v;
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    float t = u_time * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        vec2 drift = vec2(sin(t * 0.2 + i), cos(t * 0.15 + i * 1.5)) * t * 0.05;
                        vec2 pUV = uv + drift;
                        pUV += vec2(fbm(pUV * (2.0 + u_distortion) + t * 0.1), fbm(pUV * (2.0 + u_distortion) + t * 0.1 + 100.0)) * 0.3;
                        float e = fbm(pUV);
                        
                        vec3 c = e < 0.4 ? vec3(0.1, 0.2, 0.5) : e < 0.6 ? vec3(0.2, 0.6, 0.3) : vec3(0.7, 0.7, 0.8);
                        c *= 0.5 + 0.5 * cos(t * 0.1 + e);
                        col += c / (i + 1.0);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.BIOLUMINESCENT: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    float t = u_time * 0.5;
                    uv.y += sin(t * 0.1) * 0.3;
                    vec3 col = vec3(0.0, 0.05, 0.15);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        vec2 c = uv * (5.0 + i * 2.0);
                        c.x += t * (0.1 + i * 0.05);
                        c.y += sin(c.x * 2.0 + t * 0.2) * (0.2 + u_distortion * 0.2);
                        
                        vec2 id = floor(c);
                        vec2 gv = fract(c) - 0.5;
                        float h = hash(id + i);
                        float pulse = smoothstep(0.3, 0.8, sin(t * (0.5 + h * 2.0) + h * 6.28) * 0.5 + 0.5);
                        float org = smoothstep(0.3, 0.1, length(gv)) * pulse;
                        
                        col += (0.5 + 0.5 * cos(h * 6.28 + vec3(0, 2, 4))) * org / sqrt(i + 1.0);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.AURORA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    float t = u_time * 0.6;
                    float season = sin(t * 0.05) * 0.5 + 0.5;
                    vec3 col = vec3(0.01, 0.01, 0.03);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float x = uv.x * (3.0 + i);
                        float wave = sin(x + t * (0.3 + i * 0.1)) * (0.5 + u_distortion);
                        float curtain = abs(uv.y - wave);
                        float intensity = smoothstep(0.5, 0.0, curtain) * (0.5 + season * 0.5);
                        float shimmer = hash(vec2(x * 20.0, t * 2.0 + i));
                        intensity *= 0.7 + shimmer * 0.3;
                        
                        col += (0.5 + 0.5 * cos(t * 2.0 + i * 1.5 + vec3(0, 2, 4))) * intensity;
                    }
                    col += smoothstep(0.99, 1.0, hash(uv * 100.0)) * 0.3;
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.CRYSTAL: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float t = u_time * 0.2;
                    float growth = smoothstep(0.0, 10.0, t) * (1.0 + sin(t * 0.1) * 0.2);
                    uv *= u_zoom * (1.0 + growth * 0.5);

                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float a = mod(angle + t * 0.05 * (i + 1.0), 6.28 / 6.0);
                        float face = abs(sin(a * 3.0 + t * 0.1));
                        float layer = sin(dist * (10.0 + i * 3.0) - t * 0.3 + face * u_distortion) * 0.5 + 0.5;
                        layer *= smoothstep(i * 0.5, i * 0.5 + 2.0, t);
                        col += (0.5 + 0.5 * cos(dist + i + vec3(0, 2, 4))) * layer * smoothstep(1.0, 0.5, dist);
                    }
                    col += vec3(1.0, 1.0, 0.9) * exp(-dist * 5.0);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            '''
        }
        return vertex_shader, code


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
        log.debug("Entering DLA.get_frame")

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
                                 min=1, max=20, default=5,
                                 subgroup=subgroup, group=group)
        self.freq_n = params.add("chladni_freq_n",
                                 min=1, max=20, default=3,
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
                                    type=Widget.RADIO, options=Toggle)
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
        log.debug("Entering Chladni.get_frame")

        # Check for particle count change
        if self.num_particles.value != self.prev_num_particles:
            self._init_particles()
            self.prev_num_particles = self.num_particles.value

        self.time += 0.016  # ~60fps

        m = int(self.freq_m.value)
        n = int(self.freq_n.value)
        amp = self.amplitude.value

        # Generate wave pattern
        x_coords = np.linspace(0, self.width - 1, self.width)
        y_coords = np.linspace(0, self.height - 1, self.height)
        X, Y = np.meshgrid(x_coords, y_coords)

        wave = self._chladni_value(X, Y, m, n, self.time)
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


class Voronoi(Animation):
    """
    Voronoi Relaxation - points iteratively move toward cell centroids,
    creating organic, cell-like tessellations that breathe and flow.
    """
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Point parameters
        self.num_points = params.add("voronoi_num_points",
                                     min=5, max=200, default=50,
                                     subgroup=subgroup, group=group)
        self.relaxation_speed = params.add("voronoi_relax_speed",
                                           min=0.01, max=0.5, default=0.1,
                                           subgroup=subgroup, group=group)
        self.jitter = params.add("voronoi_jitter",
                                 min=0.0, max=5.0, default=0.5,
                                 subgroup=subgroup, group=group)

        # Visual parameters
        self.show_edges = params.add("voronoi_show_edges",
                                     min=0, max=1, default=1,
                                     subgroup=subgroup, group=group,
                                     type=Widget.RADIO, options=Toggle)
        self.show_points = params.add("voronoi_show_points",
                                      min=0, max=1, default=1,
                                      subgroup=subgroup, group=group,
                                      type=Widget.RADIO, options=Toggle)
        self.fill_cells = params.add("voronoi_fill_cells",
                                     min=0, max=1, default=1,
                                     subgroup=subgroup, group=group,
                                     type=Widget.RADIO, options=Toggle)
        self.edge_thickness = params.add("voronoi_edge_thickness",
                                         min=1, max=5, default=2,
                                         subgroup=subgroup, group=group)
        self.point_size = params.add("voronoi_point_size",
                                     min=2, max=10, default=5,
                                     subgroup=subgroup, group=group)

        # Color parameters
        self.edge_r = params.add("voronoi_edge_r",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.edge_g = params.add("voronoi_edge_g",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.edge_b = params.add("voronoi_edge_b",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.colormap = params.add("voronoi_colormap",
                                   min=0, max=len(COLORMAP_OPTIONS)-1, default=0,
                                   subgroup=subgroup, group=group,
                                   type=Widget.DROPDOWN, options=Colormap)

        # Animation
        self.color_cycle_speed = params.add("voronoi_color_speed",
                                            min=0.0, max=2.0, default=0.2,
                                            subgroup=subgroup, group=group)

        self.time = 0.0
        self._init_points()
        self.prev_num_points = self.num_points.value

    def _init_points(self):
        """Initialize Voronoi seed points."""
        n = int(self.num_points.value)
        # Random initial positions with margin
        margin = 20
        self.points = np.random.uniform(
            [margin, margin],
            [self.width - margin, self.height - margin],
            (n, 2)
        ).astype(np.float32)
        # Generate random colors for each cell
        self.cell_colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)

    def _compute_voronoi_image(self):
        """Compute Voronoi diagram using distance transform."""
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:self.height, :self.width]

        # Calculate distance from each pixel to each point
        # Use broadcasting: points shape (n, 2), coords shape (h, w)
        n = len(self.points)
        min_dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
        cell_indices = np.zeros((self.height, self.width), dtype=np.int32)

        for i, (px, py) in enumerate(self.points):
            dist = (x_coords - px) ** 2 + (y_coords - py) ** 2
            mask = dist < min_dist
            min_dist = np.where(mask, dist, min_dist)
            cell_indices = np.where(mask, i, cell_indices)

        return cell_indices, np.sqrt(min_dist)

    def _compute_centroids(self, cell_indices):
        """Compute centroid of each Voronoi cell."""
        n = len(self.points)
        centroids = np.zeros((n, 2), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:self.height, :self.width]

        for i in range(n):
            mask = cell_indices == i
            if np.any(mask):
                centroids[i, 0] = np.mean(x_coords[mask])
                centroids[i, 1] = np.mean(y_coords[mask])
                counts[i] = np.sum(mask)
            else:
                # Keep current position if cell is empty
                centroids[i] = self.points[i]

        return centroids

    def _detect_edges(self, cell_indices):
        """Detect edges between Voronoi cells."""
        # Shift and compare to detect boundaries
        edges = np.zeros((self.height, self.width), dtype=bool)

        # Compare with shifted versions
        if self.height > 1:
            edges[:-1, :] |= cell_indices[:-1, :] != cell_indices[1:, :]
        if self.width > 1:
            edges[:, :-1] |= cell_indices[:, :-1] != cell_indices[:, 1:]

        return edges

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        log.debug("Entering Voronoi.get_frame")

        # Check for point count change
        if self.num_points.value != self.prev_num_points:
            self._init_points()
            self.prev_num_points = self.num_points.value

        self.time += 0.016

        # Add jitter to points
        jitter_amount = self.jitter.value
        if jitter_amount > 0:
            jitter = np.random.normal(0, jitter_amount, self.points.shape).astype(np.float32)
            jittered_points = self.points + jitter
        else:
            jittered_points = self.points.copy()

        # Temporarily use jittered points for rendering
        original_points = self.points.copy()
        self.points = jittered_points

        # Compute Voronoi diagram
        cell_indices, distances = self._compute_voronoi_image()

        # Create output frame
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fill cells with colors
        if self.fill_cells.value:
            # Color offset for animation
            color_offset = int(self.time * self.color_cycle_speed.value * 50) % 256

            for i in range(len(self.points)):
                mask = cell_indices == i
                # Cycle colors over time
                color_idx = (i * 5 + color_offset) % 256
                # Use colormap for coloring
                base_color = cv2.applyColorMap(
                    np.array([[color_idx]], dtype=np.uint8),
                    COLORMAP_OPTIONS[int(self.colormap.value)]
                )[0, 0]
                pattern[mask] = base_color

        # Restore original points for relaxation
        self.points = original_points

        # Draw edges
        if self.show_edges.value:
            edges = self._detect_edges(cell_indices)
            edge_color = (int(self.edge_b.value), int(self.edge_g.value), int(self.edge_r.value))
            thickness = int(self.edge_thickness.value)

            if thickness == 1:
                pattern[edges] = edge_color
            else:
                # Dilate edges for thicker lines
                kernel = np.ones((thickness, thickness), np.uint8)
                edges_thick = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)
                pattern[edges_thick > 0] = edge_color

        # Draw points
        if self.show_points.value:
            point_size = int(self.point_size.value)
            for px, py in self.points:
                cv2.circle(pattern, (int(px), int(py)), point_size, (255, 255, 255), -1)
                cv2.circle(pattern, (int(px), int(py)), point_size, (0, 0, 0), 1)

        # Lloyd's relaxation - move points toward cell centroids
        centroids = self._compute_centroids(cell_indices)
        relax_speed = self.relaxation_speed.value
        self.points += (centroids - self.points) * relax_speed

        # Keep points within bounds
        margin = 10
        self.points[:, 0] = np.clip(self.points[:, 0], margin, self.width - margin)
        self.points[:, 1] = np.clip(self.points[:, 1], margin, self.height - margin)

        return pattern
