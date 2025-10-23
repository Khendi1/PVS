import cv2
import numpy as np
import time
import noise
import random
from generators import Oscillator
from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg
from enum import IntEnum, auto
from gui_elements import TrackbarRow
import logging
from custom_types import MoireType

log = logging.getLogger(__name__)

class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Plasma(Animation):
    def __init__(self, params, width=800, height=600):
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


    def create_gui_panel(self, default_font_id=None, global_font_id=None):
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
        with dpg.collapsing_header(label=f"\plasma Oscillator", tag="plasma_oscillator"):
            for i in range(len(plasma_params)):
                with dpg.collapsing_header(label=f"\t{plasma_params[i]}", tag=f"{plasma_params[i]}"):
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
                dpg.bind_item_font(f"{plasma_params[i]}", global_font_id)
        dpg.bind_item_font("plasma_oscillator", global_font_id)


class ReactionDiffusionSimulator(Animation):

    def __init__(self, params, width=500, height=500, da=1.0, db=0.5, feed=0.055, kill=0.062, randomize_seed=False, max_seed_size=50, num_seeds=15):
        self.params = params
        self.width = width
        self.height = height
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
        self.dt = 0.25
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


    def create_gui_panel(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tReaction Diffusion", tag="reaction_diffusion"):
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
    def __init__(self, params, width=800, height=600):
        """
        Initializes the Metaballs with given dimensions.
        """
        self.params = params
        self.width = width if width else 800
        self.height = height if height else 600
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


    def create_gui_panel(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tMetaballs", tag="metaballs"):
            num_metaballs_slider = TrackbarRow(
                "Num Metaballs",
                self.params.get("num_metaballs"),
                default_font_id)
            
            min_radius_slider = TrackbarRow(
                "Min Radius",
                 self.params.get("min_radius"),
                default_font_id)
            
            max_radius_slider = TrackbarRow(
                "Max Radius",
                 self.params.get("max_radius"),
                default_font_id)
            
            max_speed_slider = TrackbarRow(
                "Max Speed",
                 self.params.get("max_speed"),
                default_font_id)
            
            threshold_slider = TrackbarRow(
                "Threshold",
                 self.params.get("threshold"),
                default_font_id)
            
            smooth_coloring_max_field_slider = TrackbarRow(
                "Smooth Coloring Max Field",
                 self.params.get("smooth_coloring_max_field"),
                default_font_id)
            
            feedback_alpha_slider = TrackbarRow(
                "Feedback Alpha",
                 self.params.get("metaballs_feedback"),
                default_font_id)

        dpg.bind_item_font("metaballs", global_font_id)


class Moire(Animation):
    def __init__(self, params):
        self.params = params
        self.mode = params.add("moire_mode", 0, max(MoireType.values()), 0)
        self.freq = params.add("spatial_freq_1", 0,)

    def create_moire_pattern(size=512, f1=20, f2=20, angle1_deg=0, angle2_deg=5, zoom1=1.0, zoom2=1.0, mode='rotational'):
        """
        Generates a Moiré pattern based on the specified mode and parameters,
        including independent zoom factors for each grating.

        Args:
            size (int): The width and height of the square image in pixels.
            f1 (float): Spatial frequency/pitch for the first grating.
            f2 (float): Spatial frequency/pitch for the second grating.
            angle1_deg (float): Rotation angle for the first grating in degrees.
            angle2_deg (float): Rotation angle for the second grating in degrees.
            zoom1 (float): Zoom factor for the first grating (scales the pattern).
            zoom2 (float): Zoom factor for the second grating.
            mode (str): 'rotational', 'translational', or 'circular'.

        Returns:
            np.ndarray: The resulting Moiré pattern image (grayscale, 0-255).
        """
        
        #C reate coordinate grids (from -1 to 1, centered at 0)
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        
        # Initialize gratings
        grating1 = np.zeros_like(x)
        grating2 = np.zeros_like(x)

        if mode == MoireType.ROTATIONAL.value:

            angle1_rad = np.radians(angle1_deg)
            angle2_rad = np.radians(angle2_deg)

            # Grating 1
            x_rotated1 = x * np.cos(angle1_rad) + y * np.sin(angle1_rad)
            x_scaled1 = x_rotated1 / zoom1 #
            grating1 = 0.5 + 0.5 * np.cos(2 * np.pi * f1 * x_scaled1)

            # Grating 2
            x_rotated2 = x * np.cos(angle2_rad) + y * np.sin(angle2_rad)
            x_scaled2 = x_rotated2 / zoom2
            grating2 = 0.5 + 0.5 * np.cos(2 * np.pi * f2 * x_scaled2)

        elif mode == MoireType.TRANSLATIONAL.value:

            x_scaled1 = x / zoom1
            x_scaled2 = x / zoom2
            
            grating1 = 0.5 + 0.5 * np.cos(2 * np.pi * f1 * x_scaled1)
            grating2 = 0.5 + 0.5 * np.cos(2 * np.pi * f2 * x_scaled2)

        elif mode == MoireType.CIRCULAR.value:

            r = np.sqrt(x**2 + y**2)

            # Grating 1 (
            r_scaled1 = r / zoom1
            grating1 = 0.5 + 0.5 * np.cos(2 * np.pi * f1 * r_scaled1)

            # Grating 2 
            r_scaled2 = r / zoom2
            grating2 = 0.5 + 0.5 * np.cos(2 * np.pi * f2 * r_scaled2)

        else:
            # Fallback or unrecognized mode
            print(f"Error: Unrecognized Moiré mode: {mode}. Defaulting to empty image.")
            return np.zeros((size, size), dtype=np.uint8)


        # 2. Combine the Gratings (Multiplication is the common method for Moiré)
        moire_float = grating1 * grating2

        # 3. Convert to 8-bit unsigned integer (0-255) for OpenCV display
        moire_uint8 = (moire_float * 255).astype(np.uint8)

        return moire_uint8