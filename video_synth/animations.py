import cv2
import numpy as np
import time
import noise
import random
from generators import Oscillator
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from gui_elements import *
import logging
import math
from config import WidgetType
import moderngl


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

# ---- End Enum classes, begin animation base&concrete classes

class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, params, toggles, width=640, height=480, parent=None):
        self.params = params
        self.toggles = toggles
        self.width = width
        self.height = height
        self.parent = parent


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Plasma(Animation):
    def __init__(self, params, toggles, width=800, height=600, parent=None):
        super().__init__(params, toggles, parent=parent)
        subclass = self.__class__.__name__
        p_name = parent.name.lower()
        self.params = params
        self.width = width
        self.height = height

        self.plasma_speed = params.add("plasma_speed", 0.01, 10, 1.0, subclass, parent)
        self.plasma_distance = params.add("plasma_distance", 0.01, 10, 1.0, subclass, parent)
        self.plasma_color_speed = params.add("plasma_color_speed", 0.01, 10, 1.0, subclass, parent)
        self.plasma_flow_speed = params.add("plasma_flow_speed", 0.01, 10, 1.0, subclass, parent)

        self.plasma_params = [
            "plasma_speed",
            "plasma_distance",
            "plasma_color_speed",
            "plasma_flow_speed",
        ]

        self.oscillators = [Oscillator(params, name=f"{self.plasma_params[i]}", frequency=0.5, amplitude=1.0, phase=0.0, shape=1) for i in range(4)]

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

        for osc in self.oscillators:
            osc.get_next_value()

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

    def __init__(self, params, toggles, width=500, height=500, parent=None):
        super().__init__(params, toggles, parent=parent)
        subclass = self.__class__.__name__
        p_name = parent.name.lower()
        da=1.0
        db=0.5
        feed=0.055
        kill=0.062
        randomize_seed=False
        max_seed_size=50
        num_seeds=15

        self.da = params.add("da", 0, 2.0, da, subclass, parent)
        self.db = params.add("db", 0, 2.0, db, subclass, parent)

        self.feed = params.add("feed", 0, 0.1, feed, subclass, parent)
        self.kill = params.add("kill", 0, 0.1, kill, subclass, parent)
        self.iterations_per_frame = params.add("iterations_per_frame", 5, 100, 50, subclass, parent)
        
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
    def __init__(self, params, toggles, width=800, height=600, parent=None):
        super().__init__(params, toggles, parent=parent)
        subclass=self.__class__.__name__
        p_name = parent.name.lower()
        self.metaballs = []
        
        self.num_metaballs_param = params.add("num_metaballs", 2, 10, 5, subclass, parent)
        self.min_radius_param = params.add("min_radius", 20, 100, 40, subclass, parent)
        self.max_radius_param = params.add("max_radius", 40, 200, 80, subclass, parent)
        self.radius_multiplier_param = params.add("radius_multiplier", 1.0, 3.0, 1.0, subclass, parent)
        self.max_speed_param = params.add("max_speed", 1, 10, 3, subclass, parent)
        self.speed_multiplier_param = params.add("speed_multiplier", 1.0, 3.0, 1.0, subclass, parent)
        self.threshold_param = params.add("threshold", 0.5, 3.0, 1.6, subclass, parent)
        self.smooth_coloring_max_field_param = params.add("smooth_coloring_max_field", 1.0, 3.0, 1.5, subclass, parent)
        self.skew_angle_param = params.add("metaball_skew_angle", 0.0, 360.0, 0.0, subclass, parent)
        self.skew_intensity_param = params.add("metaball_skew_intensity", 0.0, 1.0, 0.0, subclass, parent)
        self.zoom_param = params.add("metaball_zoom", 1.0, 3.0, 1.0, subclass, parent)
        self.colormap_param = params.add("metaball_colormap", 0, len(COLORMAP_OPTIONS) - 1, 0, subclass, parent, WidgetType.DROPDOWN, Colormap)
        self.feedback_alpha_param = params.add("metaballs_feedback", 0.0, 1.0, 0.95, subclass, parent)

        self.current_num_metaballs = self.num_metaballs_param.value
        self.current_radius_multiplier = self.radius_multiplier_param.value
        self.current_speed_multiplier = self.speed_multiplier_param.value
        self.previous_frame = None

        self.setup_metaballs()

    def adjust_parameters(self):
        if self.current_radius_multiplier != self.radius_multiplier_param.value:
            for ball in self.metaballs:
                ball['radius'] = int(ball['radius'] * self.radius_multiplier_param.value / self.current_radius_multiplier)
            self.current_radius_multiplier = self.radius_multiplier_param.value

        if self.current_speed_multiplier != self.speed_multiplier_param.value:
            for ball in self.metaballs:
                ball['vx'] *= (self.speed_multiplier_param.value / self.current_speed_multiplier)
                ball['vy'] *= (self.speed_multiplier_param.value / self.current_speed_multiplier)
            self.current_speed_multiplier = self.speed_multiplier_param.value

    def setup_metaballs(self):
        num_metaballs = self.num_metaballs_param.value
        if len(self.metaballs) > num_metaballs:
            self.metaballs = self.metaballs[:num_metaballs]
        else:
            delta = num_metaballs - len(self.metaballs)
            for _ in range(delta):
                r = np.random.randint(self.min_radius_param.value, self.max_radius_param.value) * self.radius_multiplier_param.value
                x = np.random.randint(self.max_radius_param.value, self.width - self.max_radius_param.value)
                y = np.random.randint(self.max_radius_param.value, self.height - self.max_radius_param.value)
                vx = np.random.uniform(-self.max_speed_param.value, self.max_speed_param.value) * self.speed_multiplier_param.value
                vy = np.random.uniform(-self.max_speed_param.value, self.max_speed_param.value) * self.speed_multiplier_param.value
                self.metaballs.append({'x': x, 'y': y, 'radius': r, 'vx': vx, 'vy': vy})
        self.current_num_metaballs = num_metaballs

    def create_metaball_frame(self, metaballs, threshold, max_field_strength=None):
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        X, Y = np.meshgrid(x_coords, y_coords)

        center_x, center_y = self.width / 2, self.height / 2
        X_centered = X - center_x
        Y_centered = Y - center_y
        X_processed = X_centered / self.zoom_param.value
        Y_processed = Y_centered / self.zoom_param.value
        if self.skew_intensity_param.value > 0:
            angle_rad = np.radians(self.skew_angle_param.value)
            X_processed += Y_processed * self.skew_intensity_param.value * np.cos(angle_rad)
            Y_processed += X_processed * self.skew_intensity_param.value * np.sin(angle_rad)
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
            image = cv2.applyColorMap(grayscale_image, COLORMAP_OPTIONS[self.colormap_param.value])
        else:
            image = ((field_strength >= threshold) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image

    def do_metaballs(self, frame: np.ndarray):
        if self.num_metaballs_param.value != self.current_num_metaballs:
            self.setup_metaballs()
        
        if self.current_radius_multiplier != self.radius_multiplier_param.value or self.current_speed_multiplier != self.speed_multiplier_param.value:
            self.adjust_parameters()

        for ball in self.metaballs:
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']

            if not (ball['radius'] < ball['x'] < self.width - ball['radius']):
                ball['vx'] *= -1
            if not (ball['radius'] < ball['y'] < self.height - ball['radius']):
                ball['vy'] *= -1

        current_frame = self.create_metaball_frame(self.metaballs,
                                                threshold=self.threshold_param.value,
                                                max_field_strength=self.smooth_coloring_max_field_param.value)
        
        if self.previous_frame is None:
            self.previous_frame = current_frame.astype(np.float32)
        else:
            current_frame = cv2.addWeighted(current_frame.astype(np.float32), 1-self.feedback_alpha_param.value, 
                                            self.previous_frame, self.feedback_alpha_param.value, 0)
            self.previous_frame = current_frame

        return current_frame

    def get_frame(self, frame: np.ndarray = None):
        return self.do_metaballs(frame)


class Moire(Animation):
    def __init__(self, params, toggles, width=800, height=600, parent=None):
        super().__init__(params, toggles, parent=parent)
        subclass = self.__class__.__name__
        p_name = parent.name.lower()
        
        self.blend_mode_param = params.add("moire_blend", 0, len(MoireBlend)-1, 0, subclass, parent, WidgetType.DROPDOWN, MoireBlend)
        
        center_x, center_y = self.width//2, self.height//2

        self.pattern_1_param = params.add("moire_type_1", 0, len(MoirePattern)-1, 0, subclass, parent, WidgetType.DROPDOWN, MoirePattern)
        self.freq_1_param = params.add("spatial_freq_1", 0.01, 25, 10.0, subclass, parent, WidgetType.SLIDER)
        self.angle_1_param = params.add("angle_1", 0, 360, 90.0, subclass, parent, WidgetType.SLIDER)
        self.zoom_1_param = params.add("zoom_1", 0.05, 1.5, 1.0, subclass, parent, WidgetType.SLIDER)
        self.center_x_1_param = params.add("moire_center_x_1", 0, self.width, center_x, subclass, parent, WidgetType.SLIDER)
        self.center_y_1_param = params.add("moire_center_y_1", 0, self.height, center_y, subclass, parent, WidgetType.SLIDER)

        self.pattern_2_param = params.add("moire_type_2", 0, len(MoirePattern)-1, 0, subclass, parent, WidgetType.DROPDOWN, MoirePattern)
        self.freq_2_param = params.add("spatial_freq_2", 0.01, 25, 1.0, subclass, parent, WidgetType.SLIDER)
        self.angle_2_param = params.add("angle_2", 0, 360, 0.0, subclass, parent, WidgetType.SLIDER)
        self.zoom_2_param = params.add("zoom_2", 0.05, 1.5, 1.0, subclass, parent, WidgetType.SLIDER)
        self.center_x_2_param = params.add("moire_center_x_2", 0, self.width, center_x, subclass, parent, WidgetType.SLIDER)
        self.center_y_2_param = params.add("moire_center_y_2", 0, self.height, center_y, subclass, parent, WidgetType.SLIDER)

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

        x1, y1 = X - self.center_x_1_param.value, Y - self.center_y_1_param.value
        x2, y2 = X - self.center_x_2_param.value, Y - self.center_y_2_param.value

        pattern1_float = self._generate_single_pattern(
            x1, y1, self.freq_1_param.value, self.angle_1_param.value, self.zoom_1_param.value, self.pattern_1_param.value
        )
        pattern2_float = self._generate_single_pattern(
            x2, y2, self.freq_2_param.value, self.angle_2_param.value, self.zoom_2_param.value, self.pattern_2_param.value
        )

        blend_mode = self.blend_mode_param.value
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


class Shaders(Animation):
    def __init__(self, params, toggles, width=1280, height=720, parent=None):
        super().__init__(params, toggles, parent=parent)
        subclass = self.__class__.__name__
        p_name = parent.name.lower()
        self.width, self.height = width, height
        self.ctx = moderngl.create_context(standalone=True)
        self.time = time.time()

        self.current_shader = params.add("s_type", 0, len(ShaderType)-1, 0, subclass, parent, WidgetType.DROPDOWN, ShaderType)
        self.zoom = params.add("s_zoom", 0.1, 5.0, 1.5, subclass, parent)
        self.distortion = params.add("s_distortion", 0.0, 1.0, 0.5, subclass, parent)
        self.iterations = params.add("s_iterations", 1.0, 10.0, 4.0, subclass, parent)
        self.color_shift = params.add("s_color_shift", 0.5, 3.0, 1.0, subclass, parent)
        self.brightness = params.add("s_brightness", 0.0, 2.0, 1.0, subclass, parent)
        self.hue_shift = params.add("s_hue_shift", 0.0, 7, 0.0, subclass, parent)
        self.saturation = params.add("s_saturation", 0.0, 2.0, 1.0, subclass, parent)
        self.x_shift = params.add("s_x_shift", -5.0, 5.0, 0.0, subclass, parent)
        self.y_shift = params.add("s_y_shift", -5.0, 5.0, 0.0, subclass, parent)
        self.rotation = params.add("s_rotation", -3.14, 3.14, 0.0, subclass, parent)
        self.speed = params.add("s_speed", 0.0, 2.0, 1.0, subclass, parent)
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
