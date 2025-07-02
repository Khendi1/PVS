import cv2
import numpy as np
import time
import keyboard  # pip install keyboard
import noise   # pip install noise

class Oscillator:
    def __init__(self, frequency=1.0, amplitude=1.0, offset=0.0, waveform='sine'):
        self.frequency = frequency
        self.amplitude = amplitude
        self.amplitude = np.clip(amplitude, 0.0, 2.0)
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
            return self.amplitude * (2 * ((phase / (2 * np.pi)) % 1) - 1) + self.offset
        elif self.waveform == 'triangle':
            return self.amplitude * (2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1) + self.offset
        else:
            return 0.0

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

    osc_freq_step = 0.05 
    osc_amp_step = 0.05  
    
    current_pattern_type = 'perlin_blobs' # Set default to the new effect
    pattern_types = ['bars', 'waves', 'checker', 'radial', 'fractal_layered_waves', 'perlin_blobs']
    pattern_type_index = len(pattern_types) - 1 # Start at 'perlin_blobs'

    print("Controls:")
    print("  Q/A: Adjust Oscillator 1 Frequency (+/-)")
    print("  W/S: Adjust Oscillator 2 Frequency (+/-)")
    print("  E/D: Adjust Oscillator 3 Frequency (+/-)")
    print("  F/G: Adjust Oscillator 1 Amplitude (+/-)")
    print("  H/J: Adjust Oscillator 2 Amplitude (+/-)")
    print("  K/L: Adjust Oscillator 3 Amplitude (+/-)")
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

        if current_pattern_type == 'perlin_blobs':
            pattern = generate_perlin_blobs(frame_width, frame_height, 
                                            osc1_value, osc2_value, osc3_value, 
                                            oscillator1.amplitude, oscillator2.amplitude, oscillator3.amplitude)
        else:
            # Fallback to the previous generate_pattern if you want to keep other effects
            # You would need to re-include the old generate_pattern function or adapt this structure
            # For this script, I'm assuming 'perlin_blobs' is the primary new focus.
            # If you want to keep ALL previous patterns AND 'perlin_blobs',
            # you'd merge this logic back into the original generate_pattern function
            # or create a dispatch system.
            # For now, let's keep it simple and just show the Perlin blobs.
            # If other patterns are desired, the previous generate_pattern function needs to be explicitly defined.
            # For demonstration, let's make it a placeholder or raise an error for other types.
            print(f"Warning: Only 'perlin_blobs' implemented in this version. Switching to Perlin.")
            current_pattern_type = 'perlin_blobs' # Force to perlin_blobs if T is pressed

            # Placeholder for previous patterns - you would copy the original generate_pattern here
            # Or make a consolidated generate_pattern function.
            # For simplicity, this script ONLY implements perlin_blobs effectively.
            # To avoid an error if you press T and cycle away:
            pattern = np.zeros((frame_height, frame_width, 3), dtype=np.uint8) # Blank if not perlin

        alpha = 0.5 
        blended_frame = cv2.addWeighted(frame, 1 - alpha, pattern, alpha, 0)

        cv2.imshow('Video with Oscillator Pattern', blended_frame)

        # --- Keyboard Controls (Same as previous script) ---
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