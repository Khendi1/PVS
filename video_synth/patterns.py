from config import osc_bank, params
import numpy as np
import time
from enum import IntEnum

class PatternType(IntEnum):
    NONE = 0
    BARS = 1
    WAVES = 2
    CHECKERS = 3
    RADIAL = 4
    PERLIN = 5
    FRACTAL_PERLIN = 6
    PERLIN_BLOBS = 7

class Patterns:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # # Initialize oscillators
        # self.osc_bank = [Oscillator(f"osc_{i}", 0.0, 1.0) for i in range(3)]
        
        # # Initialize warp parameters
        self.amp_x = 10.0
        self.amp_y = 10.0
        self.freq_x = 0.1
        self.freq_y = 0.1
        self.angle_amt = params.add("angle_amt", 30, 180, 30) # Angle amount for polar warp
        self.radius_amt = params.add("radius_amt", 30, 180, 30) # Radius amount for polar warp
        self.speed = params.add("speed", 0.01, 100, 1.0) # Speed of warp effect
        self.use_fractal = params.add("use_fractal", 0, 1, 0) # Toggle for fractal noise
        self.octaves = params.add("octaves", 1, 8, 4) # Number of octaves for fractal noise
        self.gain = params.add("gain", 0.0, 1.0, 1.0) # Gain for fractal noise
        self.lacunarity = params.add("lacunarity", 1.0, 4.0, 2.0) # Lacunarity for fractal noise
        self.pattern_mode = params.add("pattern_mode", PatternType.NONE, PatternType.PERLIN_BLOBS, PatternType.NONE)

    def set_warp_params(self, warp_type, amp_x, amp_y, freq_x, freq_y, angle_amt, radius_amt, speed, use_fractal=False, octaves=4, gain=1.0, lacunarity=2.0):
        self.warp_type = warp_type
        self.amp_x = amp_x
        self.amp_y = amp_y
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.angle_amt = angle_amt
        self.radius_amt = radius_amt
        self.speed = speed
        self.use_fractal = use_fractal
        self.octaves = octaves
        self.gain = gain
        self.lacunarity = lacunarity

    def get_warp_params(self):
        return {
            'warp_type': self.warp_type,
            'amp_x': self.amp_x,
            'amp_y': self.amp_y,
            'freq_x': self.freq_x,
            'freq_y': self.freq_y,
            'angle_amt': self.angle_amt,
            'radius_amt': self.radius_amt,
            'speed': self.speed,
            'use_fractal': self.use_fractal,
            'octaves': self.octaves,
            'gain': self.gain,
            'lacunarity': self.lacunarity,
        }
    
    def generate_bars(self, pattern, x, norm_osc_vals):
        # Vertical bars that shift color based on osc values
        # Modulate bar position/color with oscillators
        # Using vectorized operations
        bar_mod = (np.sin(x * 0.05 + norm_osc_vals[0] * 10) + 1) / 2 # 0-1
        
        # brightness_green_channel is a 2D array, so .astype is fine here
        brightness_green_channel = (bar_mod * 255).astype(np.uint8)
        
        # osc2_norm * 255 and osc1_norm * 255 are single float values.
        # Assign them directly. NumPy will cast them to uint8 for the entire channel slice.
        pattern[:, :, 0] = int(norm_osc_vals[2] * 255) # Blue channel (B G R)
        pattern[:, :, 1] = brightness_green_channel # Green channel (this is a 2D array)
        pattern[:, :, 2] = int(norm_osc_vals[1] * 255) # Red channel

        return pattern

    def generate_waves(self, pattern, x, y, norm_osc_vals):
        # Horizontal/Vertical waves that ripple based on oscillator values
        # Combined spatial and temporal modulation
        val_x = np.sin(x * 0.03 + norm_osc_vals[0] * 5) # Horizontal wave
        val_y = np.sin(y * 0.03 + norm_osc_vals[1] * 5) # Vertical wave
        
        # Combine waves and modulate with osc2 for overall brightness/color
        total_val = (val_x + val_y) / 2 # Range -1 to 1
        brightness = ((total_val + norm_osc_vals[2]) / 2 + 1) / 2 * 255
        
        pattern = np.stack([brightness, brightness, brightness], axis=-1).astype(np.uint8) # Grayscale
        return pattern

    def generate_checkers(self, pattern, x, y, norm_osc_vals):
        # Checkerboard pattern whose square size/color shifts
        # TODO: parametrize grid size and colors
        grid_size_x = 50 * (1 + norm_osc_vals[0] * 0.5) # Dynamic grid size
        grid_size_y = 50 * (1 + norm_osc_vals[1] * 0.5)

        # Create checkerboard mask
        checker_mask = ((x // grid_size_x).astype(int) % 2 == (y // grid_size_y).astype(int) % 2)
        
        # These are single float values, cast to int for assignment
        color1 = int(norm_osc_vals[2] * 255)
        color2 = int((1 - norm_osc_vals[2]) * 255)
        
        # Apply colors based on the mask
        # NumPy will broadcast the scalar int values to the array elements
        pattern[checker_mask] = [color1, color1, color1]
        pattern[~checker_mask] = [color2, color2, color2]
        return pattern
    
    def generate_radial(self, pattern, x, y, norm_osc_vals):
        # Radial pattern that pulsates/rotates
        center_x, center_y = self.width / 2, self.height / 2
        
        DX = x - center_x
        DY = y - center_y
        distance = np.sqrt(DX**2 + DY**2)
        angle = np.arctan2(DY, DX)

        # Modulate radial distance and angle with oscillators
        # TODO: parameterize wave distance
        radial_mod = np.sin(distance * 0.05 + norm_osc_vals[0] * 10) # Radial wave based on distance
        # TODO: parameterize wave angle
        angle_mod = np.sin(angle * 5 + norm_osc_vals[1] * 5) # Angular wave based on angle

        # Combine for brightness, modulate color with osc2
        brightness_base = ((radial_mod + angle_mod) / 2 + 1) / 2 * 255
        
        # Color mapping with oscillators - ensure final values are arrays before stacking
        red_channel = (brightness_base * (1 - norm_osc_vals[2])).astype(np.uint8)
        green_channel = (brightness_base * norm_osc_vals[2]).astype(np.uint8)
        blue_channel = (brightness_base * ((norm_osc_vals[0] + norm_osc_vals[1])/2)).astype(np.uint8)

        pattern = np.stack([blue_channel, green_channel, red_channel], axis=-1)
        # Ensure pattern is in uint8 format
        return pattern
    
    def generate_fractal1(self, pattern, x, y, norm_osc_vals):
        # --- FRACTAL EFFECT: Sum of layered, modulated sine waves ---
        total_val_accumulator = np.zeros_like(x) # Accumulate values for each pixel

        base_freq_x = 0.01  # #TODO: this is  extremely interesting (see change from 0.01 to 0,1)
        base_freq_y = 0.01 
        base_amplitude_for_fractal = 1.5 # Max amplitude for the first layer of the fractal
        num_octaves = 4      # Number of layers/octaves # TODO: use param to control this

        # Oscillator values to perturb coordinates or phase
        # Use a scaling factor to make oscillator influence more noticeable
        x_perturb = norm_osc_vals[0] * 50 # Amount of x perturbation
        y_perturb = norm_osc_vals[1] * 50 # Amount of y perturbation
        phase_shift = norm_osc_vals[2] * np.pi * 2 # Phase shift for layers

        for i in range(num_octaves):
            freq_x = base_freq_x * (2 ** i) # Frequency doubles each octave
            freq_y = base_freq_y * (2 ** i)
            amplitude = base_amplitude_for_fractal / (2 ** i) # Amplitude halves each octave

            # Perturb coordinates using oscillators
            current_x = x + x_perturb
            current_y = y + y_perturb

            # Calculate wave for current octave
            # Adding phase_shift to make each layer move differently
            # TODO: i * i * i PHASE; slider to exponentially increase phase shift
            wave = (amplitude * np.sin(current_x * freq_x + phase_shift * i) +
                    amplitude * np.sin(current_y * freq_y + phase_shift * i))
            
            total_val_accumulator += wave
        
        # Normalize and scale the accumulated value to 0-255
        max_possible_val = base_amplitude_for_fractal * (2 - (1 / (2**(num_octaves-1)))) 
        
        if max_possible_val > 0.001:
            brightness = ( (total_val_accumulator / max_possible_val / 2 + 0.5) * 255).astype(np.uint8)
        else:
            brightness = np.zeros_like(x, dtype=np.uint8) # Default to black if no meaningful range

        # Apply a color based on some oscillator or fixed color for the fractal
        pattern[:, :, 0] = brightness # Blue
        pattern[:, :, 1] = brightness # Green
        pattern[:, :, 2] = brightness # Red (Grayscale for now)
        return pattern

    def generate_fractal2(self, pattern, x, y, norm_osc_vals):
        # --- ENHANCED FRACTAL EFFECT: Sum of layered, modulated sine waves ---
        total_val_accumulator = np.zeros_like(x) # Accumulate values for each pixel

        base_freq_x = 0.11  # Base spatial frequency
        base_freq_y = 0.01
        base_amplitude_for_fractal = 2.0 # Max amplitude for the first layer of the fractal
        num_octaves = 4      # Increased octaves for more detail
        
        # Dynamic lacunarity and persistence based on oscillators
        # Lacunarity controls how much frequency increases per octave (standard is 2.0)
        lacunarity = 2.5 + norm_osc_vals[0] * 0.7 # Range from 1.8 to 2.5
        # Persistence controls how much amplitude decreases per octave (standard is 0.5)
        persistence = 0.7 + norm_osc_vals[1] * 0.3 # Range from 0.4 to 0.7

        # Oscillator for overall time-based movement / phase shift
        time_offset = osc_bank[1] * 10 # Use raw osc value for direct offset, scaled

        for i in range(num_octaves):
            freq_x = base_freq_x * (lacunarity ** i) # Frequency multiplies by lacunarity each octave
            freq_y = base_freq_y * (lacunarity ** i)
            amplitude = base_amplitude_for_fractal * (persistence ** i) # Amplitude multiplies by persistence

            # Complex coordinate perturbation (domain warping)
            # Use a sine wave based on time_offset to perturb x and y differently
            # This creates a flowing, evolving distortion
            perturbed_x = x + np.sin(y * 0.01 + time_offset * 0.05*i) * norm_osc_vals[0] * 15 # x perturbed by y-wave
            perturbed_y = y + np.sin(x * 0.01 + time_offset * 0.05) * norm_osc_vals[1] * 15 # y perturbed by x-wave

            # Calculate wave for current octave
            wave = (amplitude * np.sin(perturbed_x * freq_x + time_offset) +
                    amplitude * np.sin(perturbed_y * freq_y + time_offset))
            
            total_val_accumulator += wave

    # --- MODIFIED generate_pattern function ---
    def generate_pattern(self):
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Scale oscillator values to a useful range for modulation
        # Normalize to 0-1 from -1 to 1 range (assuming amplitude 1, offset 0 for initial calc)
        # The get_value() method already incorporates amplitude, so we just normalize its output
        osc0_norm = osc_bank[0].value + np.abs(osc_bank[0].amplitude) / (2 * np.abs(osc_bank[0].amplitude)) if np.abs(osc_bank[0].amplitude) > 0 else 0.5
        osc1_norm = osc_bank[1].value + np.abs(osc_bank[1].amplitude) / (2 * np.abs(osc_bank[1].amplitude)) if np.abs(osc_bank[1].amplitude) > 0 else 0.5
        osc2_norm = osc_bank[2].value + np.abs(osc_bank[2].amplitude) / (2 * np.abs(osc_bank[2].amplitude)) if np.abs(osc_bank[2].amplitude) > 0 else 0.5

        norm_osc_vals = (osc0_norm, osc1_norm, osc2_norm)

        # Create coordinate grids for vectorized operations (much faster than loops)
        x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords)

        if self.pattern_type == PatternType.NONE:
            pass
        elif self.pattern_type == PatternType.BARS:
            pattern = self.generate_bars(pattern, X, norm_osc_vals)
        elif self.pattern_type == PatternType.WAVES:
            pattern = self.generate_waves(pattern, X, Y, norm_osc_vals)
        elif self.pattern_type == PatternType.CHECKERS:
            pattern = self.generate_checkers(pattern, X, Y, norm_osc_vals)
        elif self.pattern_type == PatternType.RADIAL:
            pattern = self.generate_radial(pattern, X, Y, norm_osc_vals)
        elif self.pattern_type == PatternType.PERLIN:
            pattern = self.generate_fractal1(pattern, X, Y, norm_osc_vals)
        elif self.pattern_type == PatternType.FRACTAL_PERLIN:
            pattern = self.generate_fractal2(pattern, X, Y, norm_osc_vals)
        elif self.pattern_type == PatternType.PERLIN_BLOBS:
            pattern = self.generate_perlin_blobs(norm_osc_vals) # TODO: implement

        return pattern

    # TODO: implement
    def generate_perlin_blobs(self, norm_osc_vals):
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Perlin noise parameters, modulated by oscillators for evolution
        scale_x = 0.005 + norm_osc_vals[0] * 0.01  # Modulate X spatial scale (roughness)
        scale_y = 0.005 + norm_osc_vals[1] * 0.01  # Modulate Y spatial scale (roughness)
        octaves = 6                         # Number of layers of noise
        persistence = 0.5 + norm_osc_vals[2] * 0.2 # Modulate persistence (how much each octave contributes)
        lacunarity = 2.0                    # How much frequency increases per octave (standard)

        # Use time as a Z-axis for 3D Perlin noise to ensure continuous evolution
        # The higher the frequency of osc2, the faster the blobs will "flow"
        time_factor = time.time() * (0.1 + norm_osc_vals[2] * 0.5) # Modulate evolution speed

        for y in range(self.height):
            for x in range(self.width):
                # Map x, y to a smaller range for noise function
                nx = x * scale_x
                ny = y * scale_y

                # Get Perlin noise value for each pixel using 3D noise (x, y, time)
                # This is the "shape" of the blob
                noise_val_shape = noise.pnoise3(nx, ny, time_factor,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=1024, repeaty=1024, repeatz=1024,
                                                base=0) # Base is like a seed offset

                # Map noise_val_shape from (-1, 1) range to (0, 1)
                normalized_noise_val = (noise_val_shape + 1) / 2

                # Use another noise sample for color, also evolving with time
                # Slightly different scales/parameters to make color vary independently of shape
                noise_val_color_r = noise.pnoise3(nx * 0.8, ny * 1.2, time_factor * 0.7,
                                                octaves=4, persistence=0.6, lacunarity=2.2, base=1)
                noise_val_color_g = noise.pnoise3(nx * 1.1, ny * 0.9, time_factor * 1.1,
                                                octaves=4, persistence=0.7, lacunarity=1.8, base=2)
                noise_val_color_b = noise.pnoise3(nx * 0.9, ny * 1.0, time_factor * 0.9,
                                                octaves=4, persistence=0.5, lacunarity=2.0, base=3)
                
                # Normalize color noise values
                r = int(((noise_val_color_r + 1) / 2) * 255)
                g = int(((noise_val_color_g + 1) / 2) * 255)
                b = int(((noise_val_color_b + 1) / 2) * 255)

                # Apply shape to color: lower noise_val_shape means darker/more transparent area
                # We'll use normalized_noise_val to control brightness or alpha
                # For simplicity, let's blend with black based on normalized_noise_val
                # Brighter areas of noise_val_shape will reveal more color
                final_r = int(r * normalized_noise_val)
                final_g = int(g * normalized_noise_val)
                final_b = int(b * normalized_noise_val)

                pattern[y, x] = [final_b, final_g, final_r] # OpenCV uses BGR

        return pattern