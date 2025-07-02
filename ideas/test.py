import cv2
import numpy as np
import time
import keyboard  # pip install keyboard

class Oscillator:
    def __init__(self, frequency=1.0, amplitude=1.0, offset=0.0, waveform='sine'):
        self.frequency = frequency
        self.amplitude = amplitude
        # Ensure amplitude is always positive and within a reasonable range for control
        self.amplitude = np.clip(amplitude, 0.0, 2.0) # Clip amplitude to prevent extreme values
        self.offset = offset
        self.waveform = waveform
        self.start_time = time.time()

    def get_value(self):
        current_time = time.time() - self.start_time
        phase = 2 * np.pi * self.frequency * current_time

        if self.waveform == 'sine':
            return self.amplitude * np.sin(phase) + self.offset
        elif self.waveform == 'square':
            return self.amplitude * np.sign(np.sin(phase)) + self.offset
        elif self.waveform == 'sawtooth':
            # Normalized sawtooth from -1 to 1
            return self.amplitude * (2 * ((phase / (2 * np.pi)) % 1) - 1) + self.offset
        elif self.waveform == 'triangle':
            # Normalized triangle from -1 to 1
            return self.amplitude * (2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1) + self.offset
        else:
            return 0.0

# --- MODIFIED generate_pattern function ---
def generate_pattern(frame_width, frame_height, osc1_val, osc2_val, osc3_val, pattern_type='bars'):
    pattern = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Scale oscillator values to a useful range for modulation
    # Normalize to 0-1 from -1 to 1 range (assuming amplitude 1, offset 0 for initial calc)
    # The get_value() method already incorporates amplitude, so we just normalize its output
    osc1_norm = (osc1_val + np.abs(oscillator1.amplitude)) / (2 * np.abs(oscillator1.amplitude)) if np.abs(oscillator1.amplitude) > 0 else 0.5
    osc2_norm = (osc2_val + np.abs(oscillator2.amplitude)) / (2 * np.abs(oscillator2.amplitude)) if np.abs(oscillator2.amplitude) > 0 else 0.5
    osc3_norm = (osc3_val + np.abs(oscillator3.amplitude)) / (2 * np.abs(oscillator3.amplitude)) if np.abs(oscillator3.amplitude) > 0 else 0.5


    # Create coordinate grids for vectorized operations (much faster than loops)
    x_coords = np.linspace(0, frame_width - 1, frame_width, dtype=np.float32)
    y_coords = np.linspace(0, frame_height - 1, frame_height, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords)

    if pattern_type == 'bars':
        # Vertical bars that shift color based on osc values
        # Modulate bar position/color with oscillators
        # Using vectorized operations
        bar_mod = (np.sin(X * 0.05 + osc1_norm * 10) + 1) / 2 # 0-1
        
        # brightness_green_channel is a 2D array, so .astype is fine here
        brightness_green_channel = (bar_mod * 255).astype(np.uint8)
        
        # osc3_norm * 255 and osc2_norm * 255 are single float values.
        # Assign them directly. NumPy will cast them to uint8 for the entire channel slice.
        pattern[:, :, 0] = int(osc3_norm * 255) # Blue channel (B G R)
        pattern[:, :, 1] = brightness_green_channel # Green channel (this is a 2D array)
        pattern[:, :, 2] = int(osc2_norm * 255) # Red channel

    elif pattern_type == 'waves':
        # Horizontal/Vertical waves that ripple based on oscillator values
        # Combined spatial and temporal modulation
        val_x = np.sin(X * 0.03 + osc1_norm * 5) # Horizontal wave
        val_y = np.sin(Y * 0.03 + osc2_norm * 5) # Vertical wave
        
        # Combine waves and modulate with osc3 for overall brightness/color
        total_val = (val_x + val_y) / 2 # Range -1 to 1
        brightness = ((total_val + osc3_norm) / 2 + 1) / 2 * 255
        
        pattern = np.stack([brightness, brightness, brightness], axis=-1).astype(np.uint8) # Grayscale

    elif pattern_type == 'checker':
        # Checkerboard pattern whose square size/color shifts
        grid_size_x = 50 * (1 + osc1_norm * 0.5) # Dynamic grid size
        grid_size_y = 50 * (1 + osc2_norm * 0.5)

        # Create checkerboard mask
        checker_mask = ((X // grid_size_x).astype(int) % 2 == (Y // grid_size_y).astype(int) % 2)
        
        # These are single float values, cast to int for assignment
        color1 = int(osc3_norm * 255)
        color2 = int((1 - osc3_norm) * 255)
        
        # Apply colors based on the mask
        # NumPy will broadcast the scalar int values to the array elements
        pattern[checker_mask] = [color1, color1, color1]
        pattern[~checker_mask] = [color2, color2, color2]

    elif pattern_type == 'radial':
        # Radial pattern that pulsates/rotates
        center_x, center_y = frame_width / 2, frame_height / 2
        
        DX = X - center_x
        DY = Y - center_y
        distance = np.sqrt(DX**2 + DY**2)
        angle = np.arctan2(DY, DX)

        # Modulate radial distance and angle with oscillators
        radial_mod = np.sin(distance * 0.05 + osc1_norm * 10) # Radial wave based on distance
        angle_mod = np.sin(angle * 5 + osc2_norm * 5) # Angular wave based on angle

        # Combine for brightness, modulate color with osc3
        brightness_base = ((radial_mod + angle_mod) / 2 + 1) / 2 * 255
        
        # Color mapping with oscillators - ensure final values are arrays before stacking
        red_channel = (brightness_base * (1 - osc3_norm)).astype(np.uint8)
        green_channel = (brightness_base * osc3_norm).astype(np.uint8)
        blue_channel = (brightness_base * ((osc1_norm + osc2_norm)/2)).astype(np.uint8)

        pattern = np.stack([blue_channel, green_channel, red_channel], axis=-1)

    elif pattern_type == 'fractal_layered_waves':
        # --- FRACTAL EFFECT: Sum of layered, modulated sine waves ---
        total_val_accumulator = np.zeros_like(X) # Accumulate values for each pixel

        base_freq_x = 0.01  # #TODO: this is  extremely interesting (see change from 0.01 to 0,1)
        base_freq_y = 0.01 
        base_amplitude_for_fractal = 1.5 # Max amplitude for the first layer of the fractal
        num_octaves = 4      # Number of layers/octaves

        # Oscillator values to perturb coordinates or phase
        # Use a scaling factor to make oscillator influence more noticeable
        x_perturb = osc1_norm * 50 # Amount of x perturbation
        y_perturb = osc2_norm * 50 # Amount of y perturbation
        phase_shift = osc3_norm * np.pi * 2 # Phase shift for layers

        for i in range(num_octaves):
            freq_x = base_freq_x * (2 ** i) # Frequency doubles each octave
            freq_y = base_freq_y * (2 ** i)
            amplitude = base_amplitude_for_fractal / (2 ** i) # Amplitude halves each octave

            # Perturb coordinates using oscillators
            current_x = X + x_perturb
            current_y = Y + y_perturb

            # Calculate wave for current octave
            # Adding phase_shift to make each layer move differently
            # TODO: i * i * i PHASE
            wave = (amplitude * np.sin(current_x * freq_x + phase_shift * i) +
                    amplitude * np.sin(current_y * freq_y + phase_shift * i))
            
            total_val_accumulator += wave
        
        # Normalize and scale the accumulated value to 0-255
        max_possible_val = base_amplitude_for_fractal * (2 - (1 / (2**(num_octaves-1)))) 
        
        if max_possible_val > 0.001:
            brightness = ( (total_val_accumulator / max_possible_val / 2 + 0.5) * 255).astype(np.uint8)
        else:
            brightness = np.zeros_like(X, dtype=np.uint8) # Default to black if no meaningful range

        # Apply a color based on some oscillator or fixed color for the fractal
        pattern[:, :, 0] = brightness # Blue
        pattern[:, :, 1] = brightness # Green
        pattern[:, :, 2] = brightness # Red (Grayscale for now)

    elif pattern_type == 'perlin_blobs':
        pattern = generate_perlin_blobs(frame_width, frame_height, osc1_val, osc2_val, osc3_val, osc1_amp, osc2_amp, osc3_amp)
    return pattern


# --- NEW generate_perlin_blobs function ---
def generate_perlin_blobs(frame_width, frame_height, osc1_val, osc2_val, osc3_val,
                          oscillator1_amp, oscillator2_amp, oscillator3_amp):
    pattern = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Normalize oscillator values from their amplitude range to 0-1
    osc1_norm = (osc1_val + oscillator1_amp) / (2 * oscillator1_amp) if oscillator1_amp > 0 else 0.5
    osc2_norm = (osc2_val + oscillator2_amp) / (2 * oscillator2_amp) if oscillator2_amp > 0 else 0.5
    osc3_norm = (osc3_val + oscillator3_amp) / (2 * oscillator3_amp) if oscillator3_amp > 0 else 0.5

    # Perlin noise parameters, modulated by oscillators for evolution
    scale_x = 0.005 + osc1_norm * 0.01  # Modulate X spatial scale (roughness)
    scale_y = 0.005 + osc2_norm * 0.01  # Modulate Y spatial scale (roughness)
    octaves = 6                         # Number of layers of noise
    persistence = 0.5 + osc3_norm * 0.2 # Modulate persistence (how much each octave contributes)
    lacunarity = 2.0                    # How much frequency increases per octave (standard)

    # Use time as a Z-axis for 3D Perlin noise to ensure continuous evolution
    # The higher the frequency of osc3, the faster the blobs will "flow"
    time_factor = time.time() * (0.1 + osc3_norm * 0.5) # Modulate evolution speed

    for y in range(frame_height):
        for x in range(frame_width):
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

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    oscillator1 = Oscillator(frequency=0.5, amplitude=1.0, offset=0.0, waveform='sine')
    oscillator2 = Oscillator(frequency=0.7, amplitude=1.0, offset=0.0, waveform='square')
    oscillator3 = Oscillator(frequency=0.3, amplitude=1.0, offset=0.0, waveform='sawtooth')

    osc_freq_step = 0.05 # Smaller step for finer control
    osc_amp_step = 0.05  # Step for amplitude control
    
    current_pattern_type = 'bars'
    pattern_types = ['bars', 'waves', 'checker', 'radial', 'fractal_layered_waves']
    pattern_type_index = 0

    print("Controls:")
    print("  Q/A: Adjust Oscillator 1 Frequency (+/-)")
    print("  W/S: Adjust Oscillator 2 Frequency (+/-)")
    print("  E/D: Adjust Oscillator 3 Frequency (+/-)")
    print("  F/G: Adjust Oscillator 1 Amplitude (+/-)  <-- NEW")
    print("  H/J: Adjust Oscillator 2 Amplitude (+/-)  <-- NEW")
    print("  K/L: Adjust Oscillator 3 Amplitude (+/-)  <-- NEW")
    print("  Z: Change Oscillator 1 Waveform")
    print("  X: Change Oscillator 2 Waveform")
    print("  C: Change Oscillator 3 Waveform")
    print("  T: Change Pattern Type")
    print("  Spacebar: Reset Oscillator Timers")
    print("  ESC: Exit")

    waveforms = ['sine', 'square', 'sawtooth', 'triangle']
    osc1_waveform_index = 0
    osc2_waveform_index = 1
    osc3_waveform_index = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        osc1_value = oscillator1.get_value()
        osc2_value = oscillator2.get_value()
        osc3_value = oscillator3.get_value()

        pattern = generate_pattern(frame_width, frame_height, osc1_value, osc2_value, osc3_value, current_pattern_type)

        alpha = 0.5 # Adjust blend transparency (0.0 to 1.0)
        blended_frame = cv2.addWeighted(frame, 1 - alpha, pattern, alpha, 0)

        cv2.imshow('Video with Oscillator Pattern', blended_frame)

        # --- Keyboard Controls ---
        if keyboard.is_pressed('q'):
            oscillator1.frequency += osc_freq_step
            print(f"Oscillator 1 Frequency: {oscillator1.frequency:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('a'):
            oscillator1.frequency -= osc_freq_step
            print(f"Oscillator 1 Frequency: {oscillator1.frequency:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('w'):
            oscillator2.frequency += osc_freq_step
            print(f"Oscillator 2 Frequency: {oscillator2.frequency:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('s'):
            oscillator2.frequency -= osc_freq_step
            print(f"Oscillator 2 Frequency: {oscillator2.frequency:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('e'):
            oscillator3.frequency += osc_freq_step
            print(f"Oscillator 3 Frequency: {oscillator3.frequency:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('d'):
            oscillator3.frequency -= osc_freq_step
            print(f"Oscillator 3 Frequency: {oscillator3.frequency:.2f}")
            time.sleep(0.1)
        
        # --- New Amplitude Controls ---
        elif keyboard.is_pressed('f'):
            oscillator1.amplitude = np.clip(oscillator1.amplitude + osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 1 Amplitude: {oscillator1.amplitude:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('g'):
            oscillator1.amplitude = np.clip(oscillator1.amplitude - osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 1 Amplitude: {oscillator1.amplitude:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('h'):
            oscillator2.amplitude = np.clip(oscillator2.amplitude + osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 2 Amplitude: {oscillator2.amplitude:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('j'):
            oscillator2.amplitude = np.clip(oscillator2.amplitude - osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 2 Amplitude: {oscillator2.amplitude:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('k'):
            oscillator3.amplitude = np.clip(oscillator3.amplitude + osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 3 Amplitude: {oscillator3.amplitude:.2f}")
            time.sleep(0.1)
        elif keyboard.is_pressed('l'):
            oscillator3.amplitude = np.clip(oscillator3.amplitude - osc_amp_step, 0.0, 2.0)
            print(f"Oscillator 3 Amplitude: {oscillator3.amplitude:.2f}")
            time.sleep(0.1)

        elif keyboard.is_pressed('z'):
            osc1_waveform_index = (osc1_waveform_index + 1) % len(waveforms)
            oscillator1.waveform = waveforms[osc1_waveform_index]
            print(f"Oscillator 1 Waveform: {oscillator1.waveform}")
            time.sleep(0.2)
        elif keyboard.is_pressed('x'):
            osc2_waveform_index = (osc2_waveform_index + 1) % len(waveforms)
            oscillator2.waveform = waveforms[osc2_waveform_index]
            print(f"Oscillator 2 Waveform: {osc2_waveform_index}")
            time.sleep(0.2)
        elif keyboard.is_pressed('c'):
            osc3_waveform_index = (osc3_waveform_index + 1) % len(waveforms)
            oscillator3.waveform = waveforms[osc3_waveform_index]
            print(f"Oscillator 3 Waveform: {osc3_waveform_index}")
            time.sleep(0.2)
        
        elif keyboard.is_pressed('t'):
            pattern_type_index = (pattern_type_index + 1) % len(pattern_types)
            current_pattern_type = pattern_types[pattern_type_index]
            print(f"Pattern Type: {current_pattern_type}")
            time.sleep(0.3)

        elif keyboard.is_pressed('space'):
            oscillator1.start_time = time.time()
            oscillator2.start_time = time.time()
            oscillator3.start_time = time.time()
            print("Oscillator Timers Reset")
            time.sleep(0.2)
        elif keyboard.is_pressed('esc'):
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
