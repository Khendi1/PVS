import numpy as np
import cv2
from enum import Enum
import noise 
from lfo import LFO, LFOShape
import logging
import math
from common import Widget

posc_bank = []  

log = logging.getLogger(__name__)

class BarMode(Enum):
    """Enumeration for different bar animation modes."""
    GROWING_SPACING = 0
    FIXED_SCROLL = 1


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


class Patterns:
    """
    Generates various animated patterns using OpenCV and modulates them
    with its own bank of LFOs.
    """
    def __init__(self, params, oscs, width, height, group=None):
        """
        Initializes the PatternGenerator.
        Args:
            params_table (ParamTable): An instance of ParamTable to register pattern-specific parameters.
            width (int): Width of the generated pattern image.
            height (int): Height of the generated pattern image.
        """
        self.params = params
        self.oscs = oscs
        self.width = width
        self.height = height
        subgroup = self.__class__.__name__

        # Create coordinate grids for vectorized operations (much faster than loops)
        self.x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        self.y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        # Define pattern-specific parameters (using the global params_table)
        self.pattern_type = params.add("pattern_type",
                                       min=PatternType.NONE.value, max=len(PatternType)-1, default=0,
                                       group=group, subgroup=subgroup,
                                       type=Widget.DROPDOWN, options=PatternType)
        self.prev_pattern_type = self.pattern_type.value
        self.pattern_alpha = params.add("pattern_alpha",
                                        min=0.0, max=1.0, default=0.5,
                                        subgroup=subgroup, group=group)

        # perlin params
        self.octaves = params.add("pattern_octaves",
                                  min=1, max=8, default=4,
                                  subgroup=subgroup, group=group) # Number of octaves for fractal noise
        self.gain = params.add("pattern_gain",
                               min=0.0, max=1.0, default=0.2,
                               subgroup=subgroup, group=group) # Gain for fractal noise
        self.lacunarity = params.add("pattern_lacunarity",
                                     min=1.0, max=4.0, default=2.0,
                                     subgroup=subgroup, group=group) # Lacunarity for fractal noise

        # controls density of bars
        self.bar_x_freq = params.add("bar_x_freq",
                                     min=0.01, max=0.75, default=0.01,
                                     subgroup=subgroup, group=group)
        self.bar_y_freq = params.add("bar_y_freq",
                                     min=0.01, max=0.75, default=0.01,
                                     subgroup=subgroup, group=group)
        # controls bar scrolling speed
        self.bar_x_offset = params.add("bar_x_offset",
                                       min=-100, max=100, default=2.0,
                                       subgroup=subgroup, group=group) # Offset for X bars
        self.bar_y_offset = params.add("bar_y_offset",
                                       min=-10, max=10, default=1.0,
                                       subgroup=subgroup, group=group) # Offset for Y bars

        self.rotation = params.add("pattern_rotation",
                                   min=-360, max=360, default=0.0,
                                   subgroup=subgroup, group=group)

        # creates interesting color patterns
        self.mod = params.add("pattern_mod",
                              min=0.1, max=2.0, default=1.0,
                              subgroup=subgroup, group=group) # Modulation factor for bar patterns

        # Color parameters for patterns # TODO: mapping seems off
        self.r = params.add("pattern_r",
                            min=0, max=180, default=127,
                            subgroup=subgroup, group=group)
        self.g = params.add("pattern_g",
                            min=0, max=180, default=127,
                            subgroup=subgroup, group=group)
        self.b = params.add("pattern_b",
                            min=0, max=180, default=127,
                            subgroup=subgroup, group=group)

        self.grid_size = params.add("pattern_grid_size",
                                    min=10, max=100, default=30,
                                    subgroup=subgroup, group=group) # Base grid size for checkers
        self.color_shift = params.add("pattern_color_shift",
                                      min=0, max=255, default=127,
                                      subgroup=subgroup, group=group)
        self.color_blend = params.add("pattern_color_blend",
                                      min=0, max=255, default=127,
                                      subgroup=subgroup, group=group)

        self.wave_freq_x = params.add("pattern_wave_freq_x",
                                      min=0.0, max=100, default=0.05,
                                      subgroup=subgroup, group=group)
        self.wave_freq_y = params.add("pattern_wave_freq_y",
                                      min=0.0, max=100, default=0.05,
                                      subgroup=subgroup, group=group)
        self.brightness = params.add("pattern_brightness",
                                     min=0.0, max=100.0, default=50.0,
                                     subgroup=subgroup, group=group)

        self.radial_freq = params.add("pattern_radial_freq",
                                      min=1, max=100, default=30,
                                      subgroup=subgroup, group=group) # Angle amount for polar warp
        self.angular_freq = params.add("pattern_angular_freq",
                                       min=1, max=40, default=1,
                                       subgroup=subgroup, group=group) # Radius amount for polar warp
        self.radial_mod = params.add("pattern_radial_mod",
                                     min=0.1, max=10.0, default=1.0,
                                     subgroup=subgroup, group=group) # Modulation factor for radial patterns
        self.angle_mod = params.add("pattern_angle_mod",
                                    min=0.1, max=10.0, default=1.0,
                                    subgroup=subgroup, group=group) # Modulation factor for angle patterns

        self.x_hue = params.add("x_hue",
                                min=0.0, max=1.0, default=0.5,
                                subgroup=subgroup, group=group) # Hue for X bars
        self.y_hue = params.add("y_hue",
                                min=0.0, max=1.0, default=0.5,
                                subgroup=subgroup, group=group) # Hue for Y bars

        self.x_scale = params.add("pperlin_scale_x",
                                  min=0.001, max=0.05, default=0.005,
                                  subgroup=subgroup, group=group)
        self.y_scale = params.add("pperlin_scale_y",
                                  min=0.001, max=0.05, default=0.005,
                                  subgroup=subgroup, group=group)

        self.octaves = params.add("pperlin_octaves",
                                  min=1, max=10, default=6,
                                  subgroup=subgroup, group=group)
        self.persistence = params.add("pperlin_persistence",
                                      min=0.1, max=1.0, default=0.5,
                                      subgroup=subgroup, group=group)
        self.lacunarity = params.add("pperlin_lacunarity",
                                     min=1.0, max=4.0, default=2.0,
                                     subgroup=subgroup, group=group)
        self.time_speed = params.add("pperlin_time_speed",
                                     min=0.01, max=1.0, default=0.1,
                                     subgroup=subgroup, group=group)

        self.famp = params.add("pfractal_amplitude",
                               min=0.5, max=5.0, default=1.5,
                               subgroup=subgroup, group=group)
        self.foct = params.add("pfractal_octaves",
                               min=1, max=8, default=4,
                               subgroup=subgroup, group=group)

        # Fractal Sine parameters
        self.x_perturb = params.add("x_perturb",
                                    min=0, max=50, default=25.0,
                                    subgroup=subgroup, group=group)
        self.y_perturb = params.add("y_perturb",
                                    min=0, max=50, default=25.0,
                                    subgroup=subgroup, group=group)
        self.phase_speed = params.add("phase_speed",
                                      min=0.01, max=10.0, default=1.0,
                                      subgroup=subgroup, group=group) 

        self.pattern_oscs = []
        self.prev = None

    def _cleanup_pattern_oscs(self):
        for osc in self.pattern_oscs:
            if osc.linked_param:
                osc.unlink_param()
            self.oscs.remove_oscillator(osc)
        self.pattern_oscs = []

    def _set_osc_params(self):
        if self.pattern_type.value != self.prev_pattern_type:
            self._cleanup_pattern_oscs()
            log.info(f"Pattern type changed from {self.prev_pattern_type} to {self.pattern_type.value}.")
            self.prev_pattern_type = self.pattern_type.value
            if self.pattern_type.value == PatternType.BARS.value:
                osc1 = self.oscs.add_oscillator(name=f"p_{self.bar_x_offset.name}")
                osc1.link_param(self.bar_x_offset)
                self.bar_x_offset.linked_oscillator = osc1
                self.pattern_oscs.append(osc1)
                
                osc2 = self.oscs.add_oscillator(name=f"p_{self.rotation.name}")
                osc2.link_param(self.rotation)
                self.rotation.linked_oscillator = osc2
                osc2.frequency.value = 0
                self.pattern_oscs.append(osc2)

            elif self.pattern_type.value == PatternType.WAVES.value:
                osc1 = self.oscs.add_oscillator(name=f"p_{self.wave_freq_x.name}")
                osc1.link_param(self.wave_freq_x)
                self.wave_freq_x.linked_oscillator = osc1
                self.pattern_oscs.append(osc1)

                osc2 = self.oscs.add_oscillator(name=f"p_{self.wave_freq_y.name}")
                osc2.link_param(self.wave_freq_y)
                self.wave_freq_y.linked_oscillator = osc2
                self.pattern_oscs.append(osc2)

                osc3 = self.oscs.add_oscillator(name=f"p_{self.rotation.name}")
                osc3.link_param(self.rotation)
                self.rotation.linked_oscillator = osc3
                self.pattern_oscs.append(osc3)

            elif self.pattern_type.value == PatternType.CHECKERS.value:
                osc1 = self.oscs.add_oscillator(name=f"p_{self.grid_size.name}")
                osc1.link_param(self.grid_size)
                self.grid_size.linked_oscillator = osc1
                self.pattern_oscs.append(osc1)

                osc2 = self.oscs.add_oscillator(name=f"p_{self.color_shift.name}")
                osc2.link_param(self.color_shift)
                self.color_shift.linked_oscillator = osc2
                self.pattern_oscs.append(osc2)

                osc3 = self.oscs.add_oscillator(name=f"p_{self.color_blend.name}")
                osc3.link_param(self.color_blend)
                self.color_blend.linked_oscillator = osc3
                self.pattern_oscs.append(osc3)

            elif self.pattern_type.value == PatternType.FRACTAL_SINE.value:
                osc1 = self.oscs.add_oscillator(name=f"p_{self.bar_x_offset.name}")
                osc1.link_param(self.bar_x_offset)
                self.bar_x_offset.linked_oscillator = osc1
                self.pattern_oscs.append(osc1)

                osc2 = self.oscs.add_oscillator(name=f"p_{self.bar_y_offset.name}")
                osc2.link_param(self.bar_y_offset)
                self.bar_y_offset.linked_oscillator = osc2
                self.pattern_oscs.append(osc2)

            elif self.pattern_type.value == PatternType.XY_BARS.value:
                osc1 = self.oscs.add_oscillator(name=f"p_{self.bar_x_offset.name}")
                osc1.link_param(self.bar_x_offset)
                self.bar_x_offset.linked_oscillator = osc1
                self.pattern_oscs.append(osc1)

                osc2 = self.oscs.add_oscillator(name=f"p_{self.bar_y_offset.name}")
                osc2.link_param(self.bar_y_offset)
                self.bar_y_offset.linked_oscillator = osc2
                self.pattern_oscs.append(osc2)
                
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

        self._set_osc_params()

        # log.debug(f"Generating pattern frame for type: {self.pattern_type.value}")

        # Dispatch to the appropriate pattern generation function
        if self.pattern_type.value == PatternType.NONE.value:
            return frame
        elif self.pattern_type.value == PatternType.BARS.value:
            try:
                log.debug("Calling _generate_bars")
                pattern = self._generate_bars(pattern)
            except Exception as e:
                log.error(f"Error generating BARS pattern: {e}")
                return frame # Return original frame on error
        elif self.pattern_type.value == PatternType.WAVES.value:
            try:
                log.debug("Calling _generate_waves")
                pattern = self._generate_waves(pattern, self.X, self.Y)
            except Exception as e:
                log.error(f"Error generating WAVES pattern: {e}")
                return frame
        elif self.pattern_type.value == PatternType.CHECKERS.value:
            try:
                log.debug("Calling _generate_checkers")
                pattern = self._generate_checkers(pattern, self.X, self.Y)
            except Exception as e:
                log.error(f"Error generating CHECKERS pattern: {e}")
                return frame
        elif self.pattern_type.value == PatternType.RADIAL.value:
            try:
                log.debug("Calling _generate_radial")
                pattern = self._generate_radial(pattern, self.X, self.Y)
            except Exception as e:
                log.error(f"Error generating RADIAL pattern: {e}")
                return frame
        elif self.pattern_type.value == PatternType.PERLIN_BLOBS.value:
            try:
                log.debug("Calling _generate_perlin_blobs")
                pattern = self._generate_perlin_blobs(0) # frame_time not available here, using 0
            except Exception as e:
                log.error(f"Error generating PERLIN_BLOBS pattern: {e}")
                return frame
        elif self.pattern_type.value == PatternType.FRACTAL_SINE.value:
            try:
                log.debug("Calling _generate_fractal_sine")
                pattern = self._generate_fractal_sine(pattern, self.X, self.Y, 0) # frame_time not available here, using 0
            except Exception as e:
                log.error(f"Error generating FRACTAL_SINE pattern: {e}")
                return frame
        elif self.pattern_type.value == PatternType.XY_BARS.value:
            try:
                log.debug("Calling _generate_xy_bars")
                pattern = self._generate_xy_bars(pattern, self.X, self.Y)
            except Exception as e:
                log.error(f"Error generating XY_BARS pattern: {e}")
                return frame
        alpha = self.pattern_alpha.value
        blended_frame = cv2.addWeighted(frame.astype(np.float32), 1 - alpha, pattern.astype(np.float32), alpha, 0)
        return blended_frame.astype('uint8')

    def _generate_bars(self, pattern: np.ndarray) -> np.ndarray:
        """
        Generates bars that can be rotated, shifting color and position based on oscillator values.
        """
        log.debug("Entering _generate_bars")
        height, width, _ = pattern.shape

        density = self.bar_x_freq.value
        log.debug(f"bar_x_freq.value (density): {density}")
        offset = self.bar_x_offset.value / 4  # linked to posc 0 for scrolling
        log.debug(f"bar_x_offset.value (offset): {self.bar_x_offset.value}, calculated offset: {offset}")
        mod = self.mod.value # gradient modulation for stripy bar patterns
        log.debug(f"mod.value (mod): {mod}")
        
        rotation_rad = math.radians(self.rotation.value)
        log.debug(f"rotation.value: {self.rotation.value}, rotation_rad: {rotation_rad}")

        # Create normalized X and Y coordinate grids (e.g., from -1 to 1 or 0 to 1)
        # This example uses normalized coordinates from 0 to 1, then scales/centers them
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        log.debug("Generated xx and yy meshgrids")
        
        # Combine the new coordinates into a single axis for modulation
        rotated_axis = (
            xx * math.cos(rotation_rad) - yy * math.sin(rotation_rad)
        )
        log.debug("Calculated rotated_axis")

        # color modulation: clamp bar_mod to avoid overflow errors
        bar_mod = (np.sin(rotated_axis * density * width + offset) + 1) / mod
        log.debug(f"Calculated bar_mod (before clip): min={np.min(bar_mod)}, max={np.max(bar_mod)}")
        bar_mod = np.clip(bar_mod, 0, 1)
        log.debug(f"Calculated bar_mod (after clip): min={np.min(bar_mod)}, max={np.max(bar_mod)}")

        # Apply colors based on modulated brightness and other oscillators
        # BGR format for OpenCV
        blue_channel = (bar_mod * 255 * self.b.value / 255).astype(np.uint8)
        green_channel = (bar_mod * 255 * self.g.value / 255).astype(np.uint8)
        red_channel = (bar_mod * 255 * self.r.value / 255).astype(np.uint8)
        log.debug("Calculated color channels")

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
        log.debug("Entering _generate_checkers")
        # posc0: controls grid size
        # posc1: controls color blend
        # posc2: controls color shift

        grid_size_base = 30 # Base size in pixels
        log.debug(f"grid_size_base: {grid_size_base}")
        grid_size_mod = self.grid_size.value * 40 # Modulation amount
        log.debug(f"grid_size.value: {self.grid_size.value}, grid_size_mod: {grid_size_mod}")
        grid_size_x = grid_size_base + grid_size_mod
        grid_size_y = grid_size_base + grid_size_mod
        log.debug(f"grid_size_x: {grid_size_x}, grid_size_y: {grid_size_y}")

        # Create checkerboard mask
        # Ensure grid_size is at least 1 to prevent division by zero
        grid_size_x = max(1, grid_size_x)
        grid_size_y = max(1, grid_size_y)
        log.debug(f"grid_size_x (after max(1)): {grid_size_x}, grid_size_y (after max(1)): {grid_size_y}")

        checker_mask = ((X // grid_size_x).astype(int) % 2 == (Y // grid_size_y).astype(int) % 2)
        log.debug("Calculated checker_mask")
        
        # Define two colors, modulated by oscillators
        # color_shift = norm_osc_vals[2] * 255
        color_shift = self.color_shift.value # Modulation for color shift
        log.debug(f"color_shift.value: {color_shift}")
        color_blend = self.color_blend.value / 255 # Modulation for color blending
        log.debug(f"color_blend.value: {self.color_blend.value}, calculated color_blend: {color_blend}")

        # Color 1: Changes based on osc2 (color_shift)
        c1_b = int(color_shift)
        c1_g = int(255 - color_shift)
        c1_r = int(127 + color_shift / 2)
        log.debug(f"Color 1: c1_b={c1_b}, c1_g={c1_g}, c1_r={c1_r}")

        # Color 2: Inverse or complementary to Color 1, also blended by osc1
        c2_b = int((255 - color_shift) * color_blend)
        c2_g = int(color_shift * color_blend)
        c2_r = int((127 - color_shift / 2) * color_blend)
        log.debug(f"Color 2: c2_b={c2_b}, c2_g={c2_g}, c2_r={c2_r}")
        
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

    def _generate_perlin_blobs(self, frame_time: float) -> np.ndarray:
        """
        Generates evolving Perlin noise blobs, modulated by oscillators.
        """
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Perlin noise parameters from ParamTable, modulated by oscillators
        # posc0: modulates X spatial scale
        # posc1: modulates Y spatial scale
        # posc2: modulates persistence and time evolution speed

        scale_x = self.x_scale.value
        scale_y = self.y_scale.value
        octaves = int(self.octaves.value)
        persistence = self.persistence.value
        lacunarity = self.lacunarity.value
        
        # Use time as a Z-axis for 3D Perlin noise to ensure continuous evolution
        time_speed = self.time_speed.value
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
        # For simplicity, let's mix colors based on self.r.value
        blue_channel = (brightness * (1 - self.r.value)).astype(np.uint8)
        green_channel = (brightness * self.r.value).astype(np.uint8)
        red_channel = (brightness * (1 - np.abs(self.r.value - 0.5) * 2)).astype(np.uint8) # Peaks in middle

        pattern[:, :, 0] = blue_channel # Blue
        pattern[:, :, 1] = green_channel # Green
        pattern[:, :, 2] = red_channel # Red
        return pattern
    