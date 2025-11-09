import cv2
import numpy as np
import time
import noise
import random
from generators import Oscillator
from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg
from enum import IntEnum, auto
from gui_elements import *
import logging


log = logging.getLogger(__name__)


class MoirePattern(IntEnum):
    SINE = 0
    RADIAL = auto()
    GRID = auto()


class MoireBlend(IntEnum):
    MULTIPLY = 0
    ADD = auto()
    SUB = auto()

# ---- End Enum classes, begin animation base&concrete classes

class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, params, toggles, width=800, height=600):
        self.params = params
        self.toggles = toggles
        self.width = width
        self.height = height


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Plasma(Animation):
    def __init__(self, params, toggles, width=800, height=600):
        super().__init__(params, toggles)
        self.params = params
        self.width = width
        self.height = height

        self.plasma_speed = params.add("plasma_speed", 0.01, 10, 1.0)
        self.plasma_distance = params.add("plasma_distance", 0.01, 10, 1.0)
        self.plasma_color_speed = params.add("plasma_color_speed", 0.01, 10, 1.0)
        self.plasma_flow_speed = params.add("plasma_flow_speed", 0.01, 10, 1.0)

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

        # TODO: fix plasma oscillators
        # osc1_norm = (osc1_val + oscillator1_amp) / (2 * oscillator1_amp) if oscillator1_amp > 0 else 0.5
        # osc2_norm = (osc2_val + oscillator2_amp) / (2 * oscillator2_amp) if oscillator2_amp > 0 else 0.5
        # osc3_norm = (osc3_val + oscillator3_amp) / (2 * oscillator3_amp) if oscillator3_amp > 0 else 0.5
        # osc4_norm = (osc4_val + oscillator4_amp) / (2 * oscillator4_amp) if oscillator4_amp > 0 else 0.5

        x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords)

        current_time = time.time()

        for osc in self.oscillators:
            osc.get_next_value()

        # Base time offset for overall plasma evolution, influenced by Osc1
        # Adding a large random base to offset global direction
        plasma_time_offset_base = current_time * (0.5 + self.plasma_speed.value * 2.0) + random.randint(0, 1000)

        # Spatial scaling for the main plasma, influenced by Osc2
        scale_factor_x = 0.01 + self.plasma_distance.value * 0.02
        scale_factor_y = 0.01 + self.plasma_distance.value * 0.02 #todo: make this different from x

        #  Generate Flow Fields (Domain Warping) using Perlin Noise 
        flow_scale = 0.005
        flow_strength = self.plasma_flow_speed.value * 100

        noise_x_perturb = np.zeros_like(X)
        noise_y_perturb = np.zeros_like(Y)

        # Time component for flow field evolution
        flow_noise_time = current_time * 0.1

        # Add random offsets to the base of Perlin noise for more varied flow
        random_base_x = random.randint(0, 1000)
        random_base_y = random.randint(0, 1000) + 500 # Ensure different from X

        for y in range(self.height):
            for x in range(self.width):
                nx = x * flow_scale
                ny = y * flow_scale

                noise_x_perturb[y, x] = noise.pnoise3(nx, ny, flow_noise_time, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_x)
                noise_y_perturb[y, x] = noise.pnoise3(nx + 100, ny + 100, flow_noise_time + 100, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_y)
        
        perturbed_X = X + noise_x_perturb * flow_strength
        perturbed_Y = Y + noise_y_perturb * flow_strength

        # Combine multiple sine waves with time offsets break direction
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


    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        plasma_freq_sliders = []
        plasma_amp_sliders = []
        plasma_phase_sliders = []
        plasma_seed_sliders = []
        plasma_shape_sliders = []
        plasma_params = [
            "plasma_speed",
            "plasma_distance",
            "plasma_color_speed",
            "plasma_flow_speed",
        ]
        with dpg.collapsing_header(label=f"\tPlasma Oscillator", tag="plasma_oscillator") as h:
            dpg.bind_item_theme(h, theme)
            for i in range(len(plasma_params)):
                with dpg.collapsing_header(label=f"\t{plasma_params[i]} panel", tag=f"{plasma_params[i]}_panel"):
                    plasma_shape_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Shape", 
                         self.params.get(f"{plasma_params[i]}_shape"), 
                        default_font_id))
                    
                    plasma_freq_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Freq", 
                         self.params.get(f"{plasma_params[i]}_frequency"), 
                        default_font_id))
                    
                    plasma_amp_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Amp", 
                         self.params.get(f"{plasma_params[i]}_amplitude"),
                        default_font_id))
                    
                    plasma_phase_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Phase", 
                         self.params.get(f"{plasma_params[i]}_phase"),
                        default_font_id))
                    
                    plasma_seed_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Seed", 
                         self.params.get(f"{plasma_params[i]}_seed"),
                        default_font_id))
                dpg.bind_item_font(f"{plasma_params[i]}_panel", global_font_id)
        dpg.bind_item_font("plasma_oscillator", global_font_id)


class ReactionDiffusion(Animation):

    def __init__(self, params, toggles, width=500, height=500):
        super().__init__(params, toggles)
        da=1.0
        db=0.5
        feed=0.055
        kill=0.062
        randomize_seed=False
        max_seed_size=50
        num_seeds=15

        self.da = params.add("da", 0, 2.0, da)
        self.db = params.add("db", 0, 2.0, db)

        example_patterns = {
            "worms": (0.055, 0.062),
            "spots": (0.035, 0.065),
            "maze": (0.029, 0.057),
            "coral": (0.054, 0.063)
        }
        self.pattern = example_patterns.get("coral", (feed, kill))

        # Feed rate (f): How much chemical A is added to the system
        self.feed = params.add("feed", 0, 0.1, feed)
        # Kill rate (k): How much chemical B is removed from the system
        self.kill = params.add("kill", 0, 0.1, kill)
        # Time step for the simulation. Smaller values increase stability but require more iterations.
        self.dt = 0.15
        # Number of simulation steps per displayed frame. Increased to compensate for smaller dt.
        self.iterations_per_frame = params.add("iterations_per_frame", 5, 100, 50)
        self.current_A = np.ones((height, width), dtype=np.float32)
        self.current_B = np.zeros((height, width), dtype=np.float32)
        self.next_A = np.copy(self.current_A)
        self.next_B = np.copy(self.current_B)
        
        self.randomize_seed = randomize_seed
        self.max_seed_size = max_seed_size
        self.num_seeds = num_seeds 

        self.initialize_seed()

    def initialize_seed(self):
        """
        Seeds the grid with chemical B, either at a fixed center
        or with multiple random sizes and locations, and removes chemical A from those areas.
        This initial perturbation is necessary to kickstart pattern formation.
        """
        # Reset the grid before seeding to ensure a clean start for new seeds
        self.current_A.fill(1.0)
        self.current_B.fill(0.0)

        if self.randomize_seed:
            for _ in range(self.num_seeds): # Loop for multiple seeds
                # Randomize seed size
                seed_size = random.randint(5, self.max_seed_size)
                
                # Randomize seed location, ensuring it's within bounds
                # The seed_size // 2 offset keeps the square fully within the grid
                center_x = random.randint(seed_size // 2, self.width - seed_size // 2 - 1)
                center_y = random.randint(seed_size // 2, self.height - seed_size // 2 - 1)
                
                # Apply the seed
                self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
                self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0
        else:
            # Use fixed seed size and location if not randomizing (single seed at center)
            seed_size = 20
            center_x, center_y = self.width // 2, self.height // 2
            
            # Apply the single seed
            self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
            self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0


    def update_simulation(self):
        """
        Performs one step of the Gray-Scott reaction-diffusion simulation using
        NumPy's vectorized operations for efficiency.
        """
        # Calculate Laplacians using array slicing and rolling for periodic boundary conditions
        lap_A = (
            np.roll(self.current_A, 1, axis=0) +  # Up
            np.roll(self.current_A, -1, axis=0) + # Down
            np.roll(self.current_A, 1, axis=1) +  # Left
            np.roll(self.current_A, -1, axis=1) - # Right
            4 * self.current_A
        )

        lap_B = (
            np.roll(self.current_B, 1, axis=0) +  # Up
            np.roll(self.current_B, -1, axis=0) + # Down
            np.roll(self.current_B, 1, axis=1) +  # Left
            np.roll(self.current_B, -1, axis=1) - # Right
            4 * self.current_B
        )

        # Apply Gray-Scott equations using vectorized operations
        # dA/dt = Da * Laplacian(A) - A * B*B + f * (1 - A)
        # dB/dt = Db * Laplacian(B) + A * B*B - (k + f) * B
        diff_A = self.da.value * lap_A - self.current_A * self.current_B**2 + self.feed.value * (1 - self.current_A)
        diff_B = self.db.value * lap_B + self.current_A * self.current_B**2 - (self.kill.value + self.feed.value) * self.current_B

        # Update next state arrays and clip values to stay within [0, 1] range
        self.next_A = np.clip(self.current_A + diff_A * self.dt, 0.0, 1.0)
        self.next_B = np.clip(self.current_B + diff_B * self.dt, 0.0, 1.0)

        # Swap current and next states for the next iteration
        temp_A = self.current_A
        temp_B = self.current_B
        self.current_A = self.next_A
        self.current_B = self.next_B
        self.next_A = temp_A 
        self.next_B = temp_B 


    def run(self):
        """
        Runs the simulation for a specified number of iterations and returns the display image.
        """
        for _ in range(self.iterations_per_frame.value):
            self.update_simulation()

        # Hue (H): Map chemical A concentration to hue (0-179 for OpenCV)
        hue = (self.current_A * 120).astype(np.uint8) 

        # Saturation (S): Map chemical B concentration to saturation (0-255)
        saturation = (self.current_B * 255).astype(np.uint8)

        # Value (V): Map overall activity or a combination to brightness (0-255)
        value = ((self.current_A + self.current_B) / 2 * 255).astype(np.uint8)

        hsv_image = cv2.merge([hue, saturation, value])
        
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def get_frame(self, frame):
        """
        Public method to get the next frame of the simulation.
        """
        return self.run()

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\tReaction Diffusion", tag="reaction_diffusion") as h:
            dpg.bind_item_theme(h, theme)
            rd_diffusion_rate_a_slider = TrackbarRow(
                "Diffusion Rate A",
                 self.params.get("da"),
                default_font_id)
            
            rd_diffusion_rate_b_slider = TrackbarRow(
                "Diffusion Rate B",
                 self.params.get("db"),
                default_font_id)
            
            rd_feed_rate_slider = TrackbarRow(
                "Feed Rate",
                self.params.get("feed"),
                default_font_id)
            
            rd_kill_rate_slider = TrackbarRow(
                "Kill Rate",
                self.params.get("kill"),
                default_font_id)
        
        dpg.bind_item_font("reaction_diffusion", global_font_id)


class Metaballs(Animation):
    # BUG: find bug; when increasing num_metaballs, their sizes seem to get smaller
    # TODO: add parameters to control metaball colors, blending modes, and feedback intensity
    # BUG: find bug: when reducing metaball size then returning to original/larger sizes, they get smaller each time
    def __init__(self, params, toggles, width=800, height=600):
        """
        Initializes the Metaballs with given dimensions.
        """
        super().__init__(params, toggles)
        self.metaballs = []
        self.num_metaballs = 5        # Number of metaballs
        self.min_radius = 40          # Minimum radiuWSs of a metaball
        self.max_radius = 80          # Maximum radius of a metaball
        self.max_speed = 3            # Maximum pixels per frame movement
        self.threshold = 1.6          # Threshold for metaball field strength (adjust for different blob shapes)
        
        # Adjusted smooth_coloring_max_field for a better gradient.
        # This value should be high enough to encompass the range of field strengths
        # you want to map to a gradient, preventing premature clipping to pure white.
        # It's scaled by max_radius to better fit the r^2/dist_sq field function.
        self.smooth_coloring_max_field = 1.5 # Experiment with this multiplier (e.g., 1.0 to 3.0)

        # Feedback effect parameters
        self.feedback_alpha = 0.950  # Weight of the new frame (0.0 - 1.0)
        self.previous_frame = None  # Stores the previous frame for feedback

        self.num_metaballs = params.add("num_metaballs", 2, 10, self.num_metaballs)
        self.current_num_metaballs = self.num_metaballs.value

        self.min_radius = params.add("min_radius", 20, 100, 40)
        self.max_radius = params.add("max_radius", 40, 200, 80)
        self.radius_multplier = params.add("radius_multiplier", 1.0, 3.0, 1.0)
        self.current_radius_multiplier = self.radius_multplier.value

        self.max_speed = params.add("max_speed", 1, 10, self.max_speed)
        self.speed_multiplier = params.add("speed_multiplier", 1.0, 3.0, 1.0)
        self.current_speed_multiplier = self.speed_multiplier.value

        self.threshold = params.add("threshold", 0.5, 3.0, self.threshold)
        self.smooth_coloring_max_field = params.add("smooth_coloring_max_field", 1.0, 3.0, self.smooth_coloring_max_field)

        self.skew_angle = params.add("metaball_skew_angle", 0.0, 360.0, 0.0)  # Angle to skew the metaballs
        self.skew_intensity = params.add("metaball_skew_intensity", 0.0, 1.0, 0.0)  # Intensity of the skew effect

        self.zoom = params.add("metaball_zoom", 1.0, 3.0, 1.0)  # Zoom level for the metaballs

        self.hue = params.add("metaball_hue", 0.0, 255.0, 0.0)  # Hue shift for the metaballs
        self.saturation = params.add("metaball_saturation", 0.0, 255.0, 255.0)  # Saturation for the metaballs
        self.value = params.add("metaball_value", 0.0, 255.0, 255.0)  # Value for the metaballs
        
        # apply feedback to the metaball frame
        self.feedback_alpha = params.add("metaballs_feedback", 0.0, 1.0, self.feedback_alpha)

        self.setup_metaballs()


    def adjust_parameters(self):
        """
        Adjusts the parameters based on the current values in the config.
        """
        if self.current_num_metaballs != self.num_metaballs.value:
            self.setup_metaballs()
            self.current_num_metaballs = self.num_metaballs.value

        if self.current_radius_multiplier != self.radius_multplier.value:
            for ball in self.metaballs:
                ball['radius'] = int(ball['radius'] * self.radius_multplier.value / self.current_radius_multiplier)
            self.current_radius_multiplier = self.radius_multplier.value

        if self.current_speed_multiplier != self.speed_multiplier.value:
            for ball in self.metaballs:
                ball['vx'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
                ball['vy'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
            self.current_speed_multiplier = self.speed_multiplier.value


    def setup_metaballs(self):
        """
        Initializes the positions, radii, and velocities of the metaballs.
        """
        num_metaballs = self.num_metaballs.value
        if len(self.metaballs) > num_metaballs:
            # If reducing the number of metaballs, truncate the list
            self.metaballs = self.metaballs[:num_metaballs]
            self.current_num_metaballs = num_metaballs
        else:
            # If increasing the number of metaballs, append new ones
            # Random initial position within the frame, ensuring balls start within bounds
            delta = num_metaballs - len(self.metaballs)
            for i in range(delta):
                x = np.random.randint(self.max_radius.value, self.width - self.max_radius.value)
                y = np.random.randint(self.max_radius.value, self.height - self.max_radius.value)
                # Random radius within the defined range
                r = np.random.randint(self.min_radius.value, self.max_radius.value) * self.radius_multplier.value
                # Random velocity components
                vx = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                vy = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                self.metaballs.append({'x': x, 'y': y, 'radius': r, 'vx': vx, 'vy': vy})

    def create_metaball_frame(self, metaballs, threshold, max_field_strength=None):
        """
        Generates a single frame with metaball blobs based on their properties.

        Args:
            metaballs (list): A list of dictionaries, where each dictionary
                              represents a metaball with 'x', 'y', and 'radius' keys.
            threshold (float): The field strength value above which pixels are colored.
            max_field_strength (float, optional): If provided, the field strength will be
                                                  normalized by this value for smoother coloring.
                                                  If None, a simple binary coloring is used.

        Returns:
            numpy.ndarray: An 8-bit grayscale or BGR image (frame) representing the metaballs.
        """
        # Create a grid of pixel coordinates for efficient calculation using NumPy
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Initialize a 2D array to store the total field strength for each pixel
        field_strength = np.zeros((self.height, self.width), dtype=np.float32)

        # Iterate through each metaball and add its contribution to the field
        for ball in metaballs:
            mx, my, r = ball['x'], ball['y'], ball['radius']

            # Calculate squared distance from each pixel to the metaball center
            # Adding a small epsilon (1e-6) to avoid division by zero if a pixel
            # is exactly at the metaball's center.
            dist_sq = (X - mx)**2 + (Y - my)**2 + 1e-6

            # Add the field contribution of this metaball.
            # The field strength decreases with the squared distance from the center.
            field_strength += (r**2) / dist_sq

        #  Coloring the frame based on field strength 
        if max_field_strength is not None:
            # Normalize the field strength to a 0-1 range based on max_field_strength.
            # This allows for a gradient effect, like a real lava lamp.
            # np.clip ensures values stay within 0 and 1 before scaling to 0-255.
            normalized_field = np.clip(field_strength / max_field_strength, 0, 1)

            # Map normalized field strength to a grayscale value (0-255)
            grayscale_image = (normalized_field * 255).astype(np.uint8)

            # Apply a colormap to create vibrant colors
            # cv2.COLORMAP_JET is a good general-purpose colormap.
            # You can try others like cv2.COLORMAP_HSV, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_TURBO
            image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)
        else:
            # Simple binary thresholding: pixels above threshold are white (255), others black (0)
            image = (field_strength >= threshold) * 255
            image = image.astype(np.uint8)
            # Convert grayscale to BGR to match the output type of the colormap branch
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

        return image

    def do_metaballs(self, frame: np.ndarray):
        """
        Updates metaball positions and generates the current frame, applying feedback.
        """

        if self.num_metaballs.value != len(self.metaballs):
            self.setup_metaballs()
        
        if self.current_radius_multiplier != self.radius_multplier.value or self.current_speed_multiplier != self.speed_multiplier.value:
            self.adjust_parameters()

        for ball in self.metaballs:
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']

            # Simple bouncing off the window edges
            # Check horizontal bounds
            if ball['x'] - ball['radius'] < 0:
                ball['x'] = ball['radius']
                ball['vx'] *= -1
            elif ball['x'] + ball['radius'] > self.width:
                ball['x'] = self.width - ball['radius']
                ball['vx'] *= -1

            # Check vertical bounds
            if ball['y'] - ball['radius'] < 0:
                ball['y'] = ball['radius']
                ball['vy'] *= -1
            elif ball['y'] + ball['radius'] > self.height:
                ball['y'] = self.height - ball['radius']
                ball['vy'] *= -1

        # Generate the current frame using the class's configuration
        current_frame = self.create_metaball_frame(self.metaballs,
                                                threshold=self.threshold.value,
                                                max_field_strength=self.smooth_coloring_max_field.value)
        
        # Apply feedback effect
        if self.previous_frame is None:
            self.previous_frame = current_frame
        else:
            # Blend the current frame with the previous frame
            current_frame = cv2.addWeighted(current_frame, 1-self.feedback_alpha.value, 
                                            self.previous_frame, self.feedback_alpha.value, 0)
            self.previous_frame = current_frame # Store this blended frame for the next iteration

        return current_frame

    def get_frame(self, frame: np.ndarray = None):
        """
        Public method to get the next frame of the metaball animation.
        """
        return self.do_metaballs(frame)


    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):

        with dpg.collapsing_header(label=f"\tMetaballs", tag="metaballs") as h:
            dpg.bind_item_theme(h, theme)

            num_metaballs_slider = TrackbarRow(
                "Num Metaballs", self.num_metaballs, default_font_id
            )
            
            min_radius_slider = TrackbarRow(
                "Min Radius", self.min_radius, default_font_id
            )
            
            max_radius_slider = TrackbarRow(
                "Max Radius", self.max_radius, default_font_id
            )
            
            radius_multiplier = TrackbarRow(
                "Radius Multiplier", self.radius_multplier, default_font_id
            )
            
            max_speed_slider = TrackbarRow(
                "Max Speed", self.max_speed, default_font_id
            )

            speed_multiplier = TrackbarRow(
                "Speed Multiplier", self.speed_multiplier, default_font_id
            )
            
            threshold_slider = TrackbarRow(
                "Threshold", self.threshold, default_font_id
            )
            
            smooth_coloring_max_field_slider = TrackbarRow(
                "Smooth Coloring Max Field", self.smooth_coloring_max_field, default_font_id
            )
            
            feedback_alpha_slider = TrackbarRow(
                "Feedback Alpha", self.feedback_alpha, default_font_id
            )

            skew_angle = TrackbarRow(
                "Skew Angle", self.skew_angle, default_font_id
            )

            skew_intensity = TrackbarRow(
                "Skew Instensity", self.skew_intensity, default_font_id
            )

            zoom = TrackbarRow(
                "Zoom", self.zoom, default_font_id
            )

            hue = TrackbarRow(
                "Hue", self.hue, default_font_id
            )

            sat = TrackbarRow(
                "Sat", self.saturation, default_font_id
            )

            val = TrackbarRow(
                "Val", self.value, default_font_id
            )

        dpg.bind_item_font("metaballs", global_font_id)


class Moire(Animation):
    def __init__(self, params, toggles, width=800, height=600):
        super().__init__(params, toggles)
        self.blend_mode = params.add("moire_blend", 0, len(MoireBlend)-1, 0)
        
        center_x = self.width//2
        center_y = self.height//2

        self.pattern_1 = params.add("moire_type_1", 0, len(MoirePattern)-1, 0)
        self.freq_1 = params.add("spatial_freq_1", 0, 100, 10.0)
        self.angle_1 = params.add("angle_1", 0, 100, 10.0)
        self.zoom_1 = params.add("zoom_1", 0.8, 1.5, 1.0)
        self.center_x_1 = params.add("moire_center_x_1", 0, self.width, center_x)
        self.center_y_1 = params.add("moire_center_y_1", 0, self.width, center_x)

        self.pattern_2 = params.add("moire_type_2", 0, len(MoirePattern)-1, 0)
        self.freq_2 = params.add("spatial_freq_2", 0, 100, 10.0)
        self.angle_2 = params.add("angle_2", 0, 100, 10.0)
        self.zoom_2 = params.add("zoom_2", 0.8, 1.5, 1.0)    
        self.center_x_2 = params.add("moire_center_x_2", 0, self.height, center_y)
        self.center_y_2 = params.add("moire_center_y_2", 0, self.height, center_y)

    def _generate_single_pattern(self, X_shifted, Y_shifted, frequency, angle_rad, zoom, pattern_type):
        """Internal helper to generate one of the two interfering patterns."""
        
        # apply zoom
        X_z = X_shifted * zoom
        Y_z = Y_shifted * zoom

        if pattern_type == MoirePattern.SINE.value:
            # Creates lines
            P = X_z * np.cos(angle_rad) + Y_z * np.sin(angle_rad)
            pattern = np.sin(P * frequency)
            
        elif pattern_type == MoirePattern.RADIAL.value:
            # Creates circles
            R = np.sqrt(X_z**2 + Y_z**2)
            pattern = np.sin(R * frequency)
            
        elif pattern_type == MoirePattern.GRID.value:
            # Creates grid
            freq_x = frequency * (1.0 + np.sin(angle_rad) * 0.05)
            freq_y = frequency * (1.0 + np.cos(angle_rad) * 0.05)
            
            # Additive combination of two sine waves forms a diamond-like grid structure
            pattern = np.sin(X_z * freq_x) + np.sin(Y_z * freq_y)

        # TODO: define moire pattern magic numbers
        # Scale the sine wave output (which is between -1 and 1 or -2 and 2 for grid) to 0-255
        if pattern_type == MoirePattern.GRID.value:
            # Grid sine output is -2 to 2, scale to 0-255
            return (pattern * 63.75 + 127.5).astype(np.float32)
        else:
            # Standard sine output is -1 to 1, scale to 0-255
            return (pattern * 127.5 + 127.5).astype(np.float32)


    def get_frame(self, frame):
        """
        Generates a Moiré pattern based on the specified mode and parameters,
        including independent zoom factors for each grating. Takes no args, but
        utilizes class params.

        Returns:
            np.ndarray: The resulting Moiré pattern image (grayscale, 0-255).
        """
        
        # Create coordinate grids
        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Apply Center Shift: Subtract the desired center from coordinates
        x1 = X - self.center_x_1.value
        y1 = Y - self.center_y_1.value
        x2 = X - self.center_x_2.value
        y2 = Y - self.center_y_2.value

        # generate patterns to be blended
        pattern1_float = self._generate_single_pattern(
            x1, y1, self.freq_1.value, self.angle_1.value, self.zoom_1.value, self.pattern_1.value
        )
        
        pattern2_float = self._generate_single_pattern(
            x2, y2, self.freq_2.value, self.angle_2.value, self.zoom_2.value, self.pattern_2.value
        )

        # blend pattens according to blend_mode
        if self.blend_mode.value == MoireBlend.MULTIPLY:
            # Multiply (normalized float 0-1) for a strong interference
            combined_pattern = (pattern1_float / 255.0) * (pattern2_float / 255.0)
            # Scale back to 0-255 range and convert to 8-bit integer
            moire_image = (combined_pattern * 255).astype(np.uint8)
        elif self.blend_mode.value == MoireBlend.ADD.value:
            # Add the two patterns and clip to 255
            combined_pattern = pattern1_float + pattern2_float
            moire_image = np.clip(combined_pattern / 2.0, 0, 255).astype(np.uint8)
        elif self.blend_mode.value == MoireBlend.SUB.value:
            # Add the two patterns and clip to 255
            combined_pattern = pattern1_float - pattern2_float
            moire_image = np.clip(combined_pattern / 2.0, 0, 255).astype(np.uint8)
        else:
            # Default to multiply
            combined_pattern = (pattern1_float / 255.0) * (pattern2_float / 255.0)
            moire_image = (combined_pattern * 255).astype(np.uint8)

        # apply a contrast stretch (optional but makes pattern clearer)
        moire_image = cv2.equalizeHist(moire_image)

        return cv2.cvtColor(moire_image, cv2.COLOR_GRAY2BGR)
    
    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\t Moire animation", tag="moire_animation") as h:
            dpg.bind_item_theme(h, theme)
            RadioButtonRow(
                "Blend Mode",
                MoireBlend,
                self.blend_mode,
                default_font_id
            )
            
            RadioButtonRow(
                "Pattern 1",
                MoirePattern,
                self.pattern_1,
                default_font_id
            )
            TrackbarRow(
                "Freq 1",
                self.freq_1,
                default_font_id
            )
            TrackbarRow(
                "Zoom 1",
                self.zoom_1,
                default_font_id
            )
            TrackbarRow(
                "Angle 1",
                self.angle_1,
                default_font_id
            )
            TrackbarRow(
                "Center X 1",
                self.center_x_1,
                default_font_id
            )
            TrackbarRow(
                "Center Y 1",
                self.center_y_1,
                default_font_id
            )
    
            RadioButtonRow(
                "Pattern 2",
                MoirePattern,
                self.pattern_2,
                default_font_id
            )
            TrackbarRow(
                "Freq 2",
                self.freq_2,
                default_font_id
            )
            TrackbarRow(
                "Zoom 2",
                self.zoom_2,
                default_font_id
            )
            TrackbarRow(
                "Angle 2",
                self.angle_2,
                default_font_id
            )
            TrackbarRow(
                "Center X 2",
                self.center_x_2,
                default_font_id
            )
            TrackbarRow(
                "Center Y 2",
                self.center_y_2,
                default_font_id
            )
 
    