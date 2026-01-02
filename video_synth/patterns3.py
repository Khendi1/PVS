import numpy as np
import cv2
from enum import Enum
import noise 
from generators import Oscillator, OscillatorShape
from gui_elements import RadioButtonRow
import dearpygui.dearpygui as dpg
import logging
import math
from config import WidgetType, enum_names

posc_bank = []  

log = logging.getLogger(__name__)

class BarMode(Enum):
    """Enumeration for different bar animation modes."""
    GROWING_SPACING = 0
    FIXED_SCROLL = 1

    def __str__(self):
        return self.name.replace('_', ' ').title()


class PatternType(Enum):
    """Enumeration for different visual pattern types."""
    NONE = 0
    BARS = 1
    WAVES = 2
    CHECKERS = 3
    RADIAL = 4
    PERLIN_BLOBS = 5
    FRACTAL_SINE = 6 # (sum of sines)
    XY_BARS = 7 

    def __str__(self):
        return self.name.replace('_', ' ').title()


class Patterns:
    """
    Generates various animated patterns using OpenCV and modulates them
    with its own bank of Oscillators.
    """
    def __init__(self, params, width, height, parent=None):
        """
        Initializes the PatternGenerator.
        Args:
            params_table (ParamTable): An instance of ParamTable to register pattern-specific parameters.
            width (int): Width of the generated pattern image.
            height (int): Height of the generated pattern image.
        """
        self.params = params
        self.width = width
        self.height = height
        subclass = self.__class__.__name__

        # Create coordinate grids for vectorized operations (much faster than loops)
        self.x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        self.y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        # Define pattern-specific parameters (using the global params_table)
        self.pattern_type = params.add("pattern_type", PatternType.NONE.value, len(PatternType)-1, 0, subclass, parent, WidgetType.DROPDOWN, PatternType) 
        self.prev_pattern_type = self.pattern_type.value 
        self.pattern_alpha = params.add("pattern_alpha", 0.0, 1.0, 0.5, subclass, parent)

        # perlin params
        self.octaves = params.add("pattern_octaves", 1, 8, 4, subclass, parent) # Number of octaves for fractal noise
        self.gain = params.add("pattern_gain", 0.0, 1.0, 0.2, subclass, parent) # Gain for fractal noise
        self.lacunarity = params.add("pattern_lacunarity", 1.0, 4.0, 2.0, subclass, parent) # Lacunarity for fractal noise

        # controls density of bars
        self.bar_x_freq = params.add("bar_x_freq", 0.01, 0.75, 0.01, subclass, parent)
        self.bar_y_freq = params.add("bar_y_freq", 0.01, 0.75, 0.01, subclass, parent)
        # controls bar scrolling speed
        self.bar_x_offset = params.add("bar_x_offset", -100, 100, 2.0, subclass, parent) # Offset for X bars
        self.bar_y_offset = params.add("bar_y_offset", -10, 10, 1.0, subclass, parent) # Offset for Y bars

        self.rotation = params.add("pattern_rotation", -360, 360, 0.0, subclass, parent)

        # creates interesting color patterns
        self.mod = params.add("pattern_mod", 0.1, 2.0, 1.0, subclass, parent) # Modulation factor for bar patterns

        # Color parameters for patterns # TODO: mapping seems off
        self.r = params.add("pattern_r", 0, 180, 127, subclass, parent)
        self.g = params.add("pattern_g", 0, 180, 127, subclass, parent)
        self.b = params.add("pattern_b", 0, 180, 127, subclass, parent)

        self.grid_size = params.add("pattern_grid_size", 10, 100, 30, subclass, parent) # Base grid size for checkers
        self.color_shift = params.add("pattern_color_shift", 0, 255, 127, subclass, parent)
        self.color_blend = params.add("pattern_color_blend", 0, 255, 127, subclass, parent)

        self.wave_freq_x = params.add("pattern_wave_freq_x", 0.0, 100, 0.05, subclass, parent)
        self.wave_freq_y = params.add("pattern_wave_freq_y", 0.0, 100, 0.05, subclass, parent)
        self.brightness = params.add("pattern_brightness", 0.0, 100.0, 50.0, subclass, parent) 

        self.radial_freq = params.add("pattern_radial_freq", 1, 100, 30, subclass, parent) # Angle amount for polar warp
        self.angular_freq = params.add("pattern_angular_freq", 1, 40, 1, subclass, parent) # Radius amount for polar warp
        self.radial_mod = params.add("pattern_radial_mod", 0.1, 10.0, 1.0, subclass, parent) # Modulation factor for radial patterns
        self.angle_mod = params.add("pattern_angle_mod", 0.1, 10.0, 1.0, subclass, parent) # Modulation factor for angle patterns

        self.x_hue = params.add("x_hue", 0.0, 1.0, 0.5, subclass, parent) # Hue for X bars
        self.y_hue = params.add("y_hue", 0.0, 1.0, 0.5, subclass, parent) # Hue for Y bars

        self.x_scale = params.add("pperlin_scale_x", 0.001, 0.05, 0.005, subclass, parent)
        self.y_scale = params.add("pperlin_scale_y", 0.001, 0.05, 0.005, subclass, parent)

        self.octaves = params.add("pperlin_octaves", 1, 10, 6, subclass, parent)
        self.persistence = params.add("pperlin_persistence", 0.1, 1.0, 0.5, subclass, parent)
        self.lacunarity= params.add("pperlin_lacunarity", 1.0, 4.0, 2.0, subclass, parent)
        self.time_speed=params.add("pperlin_time_speed", 0.01, 1.0, 0.1, subclass, parent)

        self.famp=params.add("pfractal_amplitude", 0.5, 5.0, 1.5, subclass, parent)
        self.foct=params.add("pfractal_octaves", 1, 8, 4, subclass, parent)

        # Fractal Sine parameters
        self.x_perturb = params.add("x_perturb", 0, 50, 25.0, subclass, parent)
        self.y_perturb = params.add("y_perturb", 0, 50, 25.0, subclass, parent)
        self.phase_speed = params.add("phase_speed", 0.01, 10.0, 1.0, subclass, parent) 

        self.num_osc = 5
        self.posc_bank = [] # List to hold Oscillator instances

        for i in range(self.num_osc):
            self.posc_bank.append(Oscillator(params, name=f"posc{i}", frequency=0.1, 
                                             amplitude=100.0, phase=0.0, shape=4,
                                             max_amplitude=self.width, min_amplitude=-self.width
                                             )) 

        self.prev = None

    def _set_osc_params(self):
        if self.pattern_type.value != self.prev_pattern_type:
            log.info(f"Pattern type changed from {self.prev_pattern_type} to {self.pattern_type.value}.")
            self.prev_pattern_type = self.pattern_type.value
            if self.pattern_type.value == PatternType.BARS.value:
                # log.info("PatternType is set to BARS; Linking bar frequency to osc 0.")
                self.posc_bank[0].link_param(self.bar_x_offset, True)
                self.posc_bank[1].link_param(self.rotation, True)
                self.posc_bank[1].frequency.value = 0 # stop auto oscillation

            elif self.pattern_type.value == PatternType.WAVES.value:
                # log.info("PatternType is set to WAVES; Linking wave frequencies to osc 0, 1.")
                self.posc_bank[0].link_param(self.wave_freq_x)
                self.posc_bank[1].link_param(self.wave_freq_y)
                self.posc_bank[2].link_param(self.rotation)
            elif self.pattern_type.value == PatternType.CHECKERS.value:
                # log.info("PatternType is set to CHECKERS; Linking grid size to osc 0.")
                self.posc_bank[0].link_param(self.grid_size, True)
                self.posc_bank[1].link_param(self.color_shift, True)
                self.posc_bank[2].link_param(self.color_blend, True)
            elif self.pattern_type.value == PatternType.RADIAL.value:
                log.info("PatternType is set to RADIAL; Linking radial parameters to osc 0.")
                # self.posc_bank[0].link_param(self.radial_freq)
                # self.posc_bank[1].link_param(self.angular_freq)
                # self.posc_bank[2].link_param(self.radial_mod)
                # self.posc_bank[3].link_param(self.angle_mod)
            elif self.pattern_type.value == PatternType.PERLIN_BLOBS.value:
                pass
                # log.info("PatternType is set to PERLIN BLOBS; Linking Perlin noise parameters to oscillators.")
                # Link Perlin noise parameters to oscillators if needed
            elif self.pattern_type.value == PatternType.FRACTAL_SINE.value:
                # log.info("PatternType is set to FRACTAL SINE; Linking fractal sine parameters to oscillators.")
                self.posc_bank[0].link_param(self.bar_x_offset)
                self.posc_bank[1].link_param(self.bar_y_offset)
            elif self.pattern_type.value == PatternType.XY_BARS.value:
                # log.info("PatternType is set to XY_BARS; Linking XY bar parameters to oscillators.")
                self.posc_bank[0].link_param(self.bar_x_offset)
                self.posc_bank[1].link_param(self.bar_y_offset)
                

    def generate_pattern_frame(self, frame: np.ndarray):
        """
        Generates a single frame of the current pattern.
        Args:
            frame_time (float): The current time in seconds, used for animation.
        Returns:
            np.ndarray: The generated pattern image (height x width x 3, uint8 BGR).
        """
        # Initialize a black image
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._set_osc_params() # TODO: use conditional to compare current/last setting so not calling every time
        posc_vals = [osc.get_next_value() for osc in self.posc_bank if osc.linked_param is not None]

        # Dispatch to the appropriate pattern generation function
        if self.pattern_type.value == PatternType.NONE.value:
            return frame
        elif self.pattern_type.value == PatternType.BARS.value:
            pattern = self._generate_bars(pattern)
        elif self.pattern_type.value == PatternType.WAVES.value:
            pattern = self._generate_waves(pattern, self.X, self.Y)
        elif self.pattern_type.value == PatternType.CHECKERS.value:
            pattern = self._generate_checkers(pattern, self.X, self.Y)
        elif self.pattern_type.value == PatternType.RADIAL.value:
            pattern = self._generate_radial(pattern, self.X, self.Y)
        elif self.pattern_type.value == PatternType.PERLIN_BLOBS.value:
            pass
            # pattern = self._generate_perlin_blobs(norm_osc_vals, frame_time)
        elif self.pattern_type == PatternType.FRACTAL_SINE.value:
            pass
            # pattern = self._generate_fractal_sine(pattern, self.X, self.Y, norm_osc_vals, frame_time)
        elif self.pattern_type == PatternType.XY_BARS.value:
            pattern = self._generate_xy_bars(pattern, self.X, self.Y)
        else:
            log.warning(f"Unknown pattern type {self.pattern_type.value}. Returning black frame.")

        # return pattern
        alpha = self.pattern_alpha.value
        blended_frame = cv2.addWeighted(frame.astype(np.float32), 1 - alpha, pattern.astype(np.float32), alpha, 0)
        return blended_frame.astype('uint8')

    def _generate_bars(self, pattern: np.ndarray) -> np.ndarray:
        """
        Generates bars that can be rotated, shifting color and position based on oscillator values.
        """
        height, width, _ = pattern.shape

        density = self.bar_x_freq.value
        offset = self.bar_x_offset.value / 4  # linked to posc 0 for scrolling
        mod = self.mod.value # gradient modulation for stripy bar patterns
        
        rotation_rad = math.radians(self.rotation.value)
        # log.debug(f'pattern rotation: {self.rotation.value}\u00b0->{rotation_rad}')

        # Create normalized X and Y coordinate grids (e.g., from -1 to 1 or 0 to 1)
        # This example uses normalized coordinates from 0 to 1, then scales/centers them
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        # Combine the new coordinates into a single axis for modulation
        rotated_axis = (
            xx * math.cos(rotation_rad) - yy * math.sin(rotation_rad)
        )

        # color modulation: clamp bar_mod to avoid overflow errors
        bar_mod = (np.sin(rotated_axis * density * width + offset) + 1) / mod
        bar_mod = np.clip(bar_mod, 0, 1)

        # Apply colors based on modulated brightness and other oscillators
        # BGR format for OpenCV
        blue_channel = (bar_mod * 255 * self.b.value / 255).astype(np.uint8)
        green_channel = (bar_mod * 255 * self.g.value / 255).astype(np.uint8)
        red_channel = (bar_mod * 255 * self.r.value / 255).astype(np.uint8)

        pattern[:, :, 0] = blue_channel
        pattern[:, :, 1] = green_channel
        pattern[:, :, 2] = red_channel
        return pattern

    def _generate_xy_bars(self, pattern: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generates a pattern of bars on both X and Y axes, controlled by static parameters.
        """

        x_offset = self.bar_x_offset.value / 10 # linked to posc 0
        y_offset = self.bar_y_offset.value / 10 # linked to posc 1

        # Generate bars per axis
        x_bars = (np.sin(X * self.bar_x_freq.value + x_offset) + 1) / 2 # Range 0-1
        y_bars = (np.sin(Y * self.bar_y_freq.value + y_offset) + 1) / 2 # Range 0-1

        # Combine the bar patterns. Multiplying creates a grid-like intersection.
        # Adding would create overlapping bars.
        combined_bars = x_bars * y_bars # This creates a grid where both X and Y bars are bright

        # Apply colors; make X bars one color and Y bars another, with overlap.
        red_x = (combined_bars * 255 * self.x_hue.value ).astype(np.uint8)
        green_x = (combined_bars * 255 * (1 - self.x_hue.value )).astype(np.uint8)
        blue_x = np.zeros_like(combined_bars, dtype=np.uint8) # Minimal blue for X
        red_y = np.zeros_like(combined_bars, dtype=np.uint8) # Minimal red for Y
        green_y = (combined_bars * 255 * (1 - self.y_hue.value )).astype(np.uint8)
        blue_y = (combined_bars * 255 * self.y_hue.value ).astype(np.uint8)

        # Combine colors. Additive blending for overlapping effects.
        pattern[:, :, 0] = np.clip(blue_x + blue_y, 0, 255) # Blue channel
        pattern[:, :, 1] = np.clip(green_x + green_y, 0, 255) # Green channel
        pattern[:, :, 2] = np.clip(red_x + red_y, 0, 255) # Red channel

        return pattern

    def _generate_waves(self, pattern: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generates horizontal/vertical waves that ripple and change color based on oscillators.
        """
        # posc0: controls horizontal wave frequency/speed
        # posc1: controls vertical wave frequency/speed
        # posc2: controls overall brightness/color shift

        # Growing/spacing mode: use oscillators to modulate wave frequencies 
        freq_x = 0.03 + self.wave_freq_x.value / 0.05 # testing w/ value 5, original is 0.05
        freq_y = 0.03 + self.wave_freq_y.value / 0.05
        # Combine waves and modulate with osc2 for overall brightness/color
        val_x = np.sin(X * freq_x + self.bar_x_offset.value * 5) # Horizontal wave
        val_y = np.sin(Y * freq_y + self.bar_y_offset.value * 5) # Vertical wave

        total_val = (val_x + val_y) / 2 # Range -1 to 1
        brightness = ((total_val + self.brightness.value) / 2 + 1) / self.mod.value * 255 # Map to 0-255

        # Apply color based on brightness and oscillator values
        blue_channel = (brightness * (1 - self.b.value)).astype(np.uint8)
        green_channel = (brightness * self.g.value).astype(np.uint8)
        red_channel = (brightness * self.r.value).astype(np.uint8)

        pattern[:, :, 0] = blue_channel
        pattern[:, :, 1] = green_channel
        pattern[:, :, 2] = red_channel
        return pattern

    def _generate_checkers(self, pattern: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generates a checkerboard pattern whose square size and colors shift.
        """
        # posc0: controls grid size
        # posc1: controls color blend
        # posc2: controls color shift

        grid_size_base = 30 # Base size in pixels
        grid_size_mod = self.grid_size.value * 40 # Modulation amount
        grid_size_x = grid_size_base + grid_size_mod
        grid_size_y = grid_size_base + grid_size_mod

        # Create checkerboard mask
        # Ensure grid_size is at least 1 to prevent division by zero
        grid_size_x = max(1, grid_size_x)
        grid_size_y = max(1, grid_size_y)

        checker_mask = ((X // grid_size_x).astype(int) % 2 == (Y // grid_size_y).astype(int) % 2)
        
        # Define two colors, modulated by oscillators
        # color_shift = norm_osc_vals[2] * 255
        color_shift = self.color_shift.value # Modulation for color shift
        color_blend = self.color_blend.value / 255 # Modulation for color blending

        # Color 1: Changes based on osc2 (color_shift)
        c1_b = int(color_shift)
        c1_g = int(255 - color_shift)
        c1_r = int(127 + color_shift / 2)

        # Color 2: Inverse or complementary to Color 1, also blended by osc1
        c2_b = int((255 - color_shift) * color_blend)
        c2_g = int(color_shift * color_blend)
        c2_r = int((127 - color_shift / 2) * color_blend)
        
        # Apply colors based on the mask
        # Use np.where for efficient assignment
        pattern[:, :, 0] = np.where(checker_mask, c1_b, c2_b).astype(np.uint8) # Blue
        pattern[:, :, 1] = np.where(checker_mask, c1_g, c2_g).astype(np.uint8) # Green
        pattern[:, :, 2] = np.where(checker_mask, c1_r, c2_r).astype(np.uint8) # Red

        return pattern

    def _generate_radial(self, pattern: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generates a radial pattern that pulsates and rotates based on oscillators.
        """
        # posc0: controls radial wave density/pulsation
        # posc1: controls angular wave density/rotation speed
        # posc2: controls overall color hue

        center_x, center_y = self.width / 2, self.height / 2
        
        DX = X - center_x
        DY = Y - center_y
        distance = np.sqrt(DX**2 + DY**2)
        angle = np.arctan2(DY, DX) # Range -pi to pi

        # Modulate radial distance and angle with oscillators
        radial_freq = 0.05 * self.radial_freq.value * 0.05 # Radial wave frequency
        angular_freq = self.angular_freq.value #* 5 # Angular wave frequency
        
        radial_mod = np.sin(distance * radial_freq + self.radial_mod.value * 10) # Radial wave
        angle_mod = np.sin(angle * angular_freq + self.angle_mod.value * 5) # Angular wave

        # Combine for brightness, modulate color with osc2
        brightness_base = ((radial_mod + angle_mod) / 2 + 1) / 2 * 255 # Map to 0-255
        
        # Color mapping with oscillators - ensure final values are arrays
        blue_channel = (brightness_base * (1 - self.b.value)).astype(np.uint8)
        green_channel = (brightness_base * self.g.value).astype(np.uint8)
        red_channel = (brightness_base * ((self.b.value + self.g.value)/2)).astype(np.uint8)

        pattern[:, :, 0] = blue_channel
        pattern[:, :, 1] = green_channel
        pattern[:, :, 2] = red_channel
        return pattern

    def _generate_perlin_blobs(self, norm_osc_vals: tuple, frame_time: float) -> np.ndarray:
        """
        Generates evolving Perlin noise blobs, modulated by oscillators.
        """
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Perlin noise parameters from ParamTable, modulated by oscillators
        # posc0: modulates X spatial scale
        # posc1: modulates Y spatial scale
        # posc2: modulates persistence and time evolution speed

        scale_x = self.x_scale + norm_osc_vals[0] * 0.02
        scale_y = self.y_scale + norm_osc_vals[1] * 0.02
        octaves = int(self.octaves.value)
        persistence = self.persistence.value + norm_osc_vals[2] * 0.2
        lacunarity = self.lacunarity.value
        
        # Use time as a Z-axis for 3D Perlin noise to ensure continuous evolution
        time_speed = self.time_speed.value + norm_osc_vals[2] * 0.2
        time_factor = frame_time * time_speed

        # Pre-calculate noise for the entire grid for efficiency
        # Create a grid of x,y coordinates
        x_grid = self.X * scale_x
        y_grid = self.Y * scale_y

        # Generate 3D Perlin noise for the entire grid
        # This is more efficient than looping pixel by pixel for noise generation
        noise_val_shape = np.array([
            [noise.pnoise3(x_val, y_val, time_factor,
                           octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                           repeatx=1024, repeaty=1024, repeatz=1024, base=0)
             for x_val, y_val in zip(row, y_row)] for row, y_row in zip(x_grid, y_grid)
        ])

        # Map noise_val_shape from (-1, 1) range to (0, 1)
        normalized_noise_val = (noise_val_shape + 1) / 2

        # Generate separate noise for color channels, also evolving
        noise_val_color_r = np.array([
            [noise.pnoise3(x_val * 0.8, y_val * 1.2, time_factor * 0.7,
                           octaves=4, persistence=0.6, lacunarity=2.2, base=1)
             for x_val, y_val in zip(row, y_row)] for row, y_row in zip(x_grid, y_grid)
        ])
        noise_val_color_g = np.array([
            [noise.pnoise3(x_val * 1.1, y_val * 0.9, time_factor * 1.1,
                           octaves=4, persistence=0.7, lacunarity=1.8, base=2)
             for x_val, y_val in zip(row, y_row)] for row, y_row in zip(x_grid, y_grid)
        ])
        noise_val_color_b = np.array([
            [noise.pnoise3(x_val * 0.9, y_val * 1.0, time_factor * 0.9,
                           octaves=4, persistence=0.5, lacunarity=2.0, base=3)
             for x_val, y_val in zip(row, y_row)] for row, y_row in zip(x_grid, y_grid)
        ])
        
        # Normalize color noise values to 0-1 and scale to 0-255
        r = ((noise_val_color_r + 1) / 2 * 255).astype(np.uint8)
        g = ((noise_val_color_g + 1) / 2 * 255).astype(np.uint8)
        b = ((noise_val_color_b + 1) / 2 * 255).astype(np.uint8)

        # Apply shape to color:
        # Blend with black based on normalized_noise_val (brighter areas reveal more color)
        final_r = (r * normalized_noise_val).astype(np.uint8)
        final_g = (g * normalized_noise_val).astype(np.uint8)
        final_b = (b * normalized_noise_val).astype(np.uint8)

        # OpenCV uses BGR
        pattern[:, :, 0] = final_b
        pattern[:, :, 1] = final_g
        pattern[:, :, 2] = final_r
        return pattern

    def _generate_fractal_sine(self, pattern: np.ndarray, X: np.ndarray, Y: np.ndarray, frame_time: float) -> np.ndarray:
        """
        Generates a fractal pattern by summing layered, modulated sine waves.
        """
        total_val_accumulator = np.zeros_like(X) # Accumulate values for each pixel
        base_freq_x = self.bar_x_freq.value * 0.01
        base_freq_y = self.bar_y_freq.value * 0.01

        # Use osc for overall time-based movement / phase shift
        phase_speed = self.phase_speed.value * 2
        time_offset = frame_time * phase_speed

        for i in range(self.foct.value+1):
            freq_x = base_freq_x * (2 ** i) # Frequency doubles each octave
            freq_y = base_freq_y * (2 ** i)
            amplitude = self.famp.value / (2 ** i) # Amplitude halves each octave

            # Perturb coordinates using oscillators
            current_x = X + self.x_perturb.value # * 50
            current_y = Y + self.y_perturb.value # * 50

            # Calculate wave for current octave
            wave = (amplitude * np.sin(current_x * freq_x + time_offset * i) +
                    amplitude * np.sin(current_y * freq_y + time_offset * i))
            
            total_val_accumulator += wave
        
        # Normalize and scale the accumulated value to 0-255
        # Calculate max possible sum of amplitudes
        max_possible_val = self.famp.value * (2 - (1 / (2**(self.foct.value)))) 
        
        if max_possible_val > 0.001: # Avoid division by zero
            brightness = ( (total_val_accumulator / max_possible_val / 2 + 0.5) * 255).astype(np.uint8)
        else:
            brightness = np.zeros_like(X, dtype=np.uint8) # Default to black if no meaningful range
        
        # Create a color based on brightness and hue shift
        # For simplicity, let's mix colors based on hue_shift
        blue_channel = (brightness * (1 - hue_shift)).astype(np.uint8)
        green_channel = (brightness * hue_shift).astype(np.uint8)
        red_channel = (brightness * (1 - np.abs(hue_shift - 0.5) * 2)).astype(np.uint8) # Peaks in middle

        pattern[:, :, 0] = blue_channel # Blue
        pattern[:, :, 1] = green_channel # Green
        pattern[:, :, 2] = red_channel # Red
        return pattern
    