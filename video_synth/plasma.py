import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import noise
import random # Import random for randomized offsets
from generators import Oscillator
from config import params

# TODO: init w loop
plasma_speed = params.add("plasma_speed", 0.01, 10, 1.0)
plasma_distance = params.add("plasma_distance", 0.01, 10, 1.0)
plasma_color_speed = params.add("plasma_color_speed", 0.01, 10, 1.0)
plasma_flow_speed = params.add("plasma_flow_speed", 0.01, 10, 1.0)

plasma_params = [
    "plasma_speed",
    "plasma_distance",
    "plasma_color_speed",
    "plasma_flow_speed",
]

oscillators = [Oscillator(name=f"{plasma_params[i]}", frequency=0.5, amplitude=1.0, phase=0.0, shape=1) for i in range(4)]

oscillators[0].link_param(plasma_speed)
oscillators[1].link_param(plasma_distance)
oscillators[2].link_param(plasma_color_speed)
oscillators[3].link_param(plasma_flow_speed)

# --- MODIFIED generate_plasma_effect function ---
def generate_plasma_effect(frame_width, frame_height):
    plasma_pattern = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # osc1_norm = (osc1_val + oscillator1_amp) / (2 * oscillator1_amp) if oscillator1_amp > 0 else 0.5
    # osc2_norm = (osc2_val + oscillator2_amp) / (2 * oscillator2_amp) if oscillator2_amp > 0 else 0.5
    # osc3_norm = (osc3_val + oscillator3_amp) / (2 * oscillator3_amp) if oscillator3_amp > 0 else 0.5
    # osc4_norm = (osc4_val + oscillator4_amp) / (2 * oscillator4_amp) if oscillator4_amp > 0 else 0.5

    x_coords = np.linspace(0, frame_width - 1, frame_width, dtype=np.float32)
    y_coords = np.linspace(0, frame_height - 1, frame_height, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords)

    current_time = time.time()

    for osc in oscillators:
        osc.get_next_value()

    # Base time offset for overall plasma evolution, influenced by Osc1
    # Adding a large random base to offset global direction
    plasma_time_offset_base = current_time * (0.5 + plasma_speed.value * 2.0) + random.randint(0, 1000)

    # Spatial scaling for the main plasma, influenced by Osc2
    scale_factor_x = 0.01 + plasma_distance.value * 0.02
    scale_factor_y = 0.01 + plasma_distance.value * 0.02 #todo: make this different from x

    # --- Generate Flow Fields (Domain Warping) using Perlin Noise ---
    flow_scale = 0.005
    flow_strength = plasma_flow_speed.value * 100

    noise_x_perturb = np.zeros_like(X)
    noise_y_perturb = np.zeros_like(Y)

    # Time component for flow field evolution
    flow_noise_time = current_time * 0.1

    # Add random offsets to the base of Perlin noise for more varied flow
    # These offsets should be large enough to jump to different parts of the noise space
    random_base_x = random.randint(0, 1000)
    random_base_y = random.randint(0, 1000) + 500 # Ensure different from X

    for y in range(frame_height):
        for x in range(frame_width):
            nx = x * flow_scale
            ny = y * flow_scale

            noise_x_perturb[y, x] = noise.pnoise3(nx, ny, flow_noise_time, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_x)
            noise_y_perturb[y, x] = noise.pnoise3(nx + 100, ny + 100, flow_noise_time + 100, octaves=4, persistence=0.5, lacunarity=2.0, base=random_base_y)
    
    perturbed_X = X + noise_x_perturb * flow_strength
    perturbed_Y = Y + noise_y_perturb * flow_strength

    # --- Combine multiple sine waves for the core plasma "value" using perturbed coordinates ---
    # Introduce different time offsets for each sine wave component to break global direction
    value = (
        np.sin(perturbed_X * scale_factor_x + plasma_time_offset_base) +
        np.sin(perturbed_Y * scale_factor_y + plasma_time_offset_base * 0.8 + random.uniform(0, np.pi * 2)) + # Added random phase
        np.sin((perturbed_X + perturbed_Y) * scale_factor_x * 0.7 + plasma_time_offset_base * 1.2 + random.uniform(0, np.pi * 2)) + # Added random phase
        np.sin((perturbed_X - perturbed_Y) * scale_factor_y * 0.9 + plasma_time_offset_base * 0.6 + random.uniform(0, np.pi * 2)) # Added random phase
    )

    normalized_value = (value + 4) / 8

    hue_shift_val = plasma_color_speed.value * 2 * np.pi

    R = np.sin(normalized_value * np.pi * 3 + hue_shift_val) * 0.5 + 0.5
    G = np.sin(normalized_value * np.pi * 3 + hue_shift_val + np.pi * 2/3) * 0.5 + 0.5
    B = np.sin(normalized_value * np.pi * 3 + hue_shift_val + np.pi * 4/3) * 0.5 + 0.5

    plasma_pattern[:, :, 2] = (R * 255).astype(np.uint8)
    plasma_pattern[:, :, 0] = (B * 255).astype(np.uint8)
    plasma_pattern[:, :, 1] = (G * 255).astype(np.uint8)

    return plasma_pattern

# --- Main Application Class for GUI and Video (Same as before, except for osc initializers) ---
class VideoSynthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plasma Video Synth GUI")
        self.root.geometry("1000x800")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.root.destroy()
            return

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.oscillators = [
            Oscillator(frequency=0.1, amplitude=1.0, offset=0.0, waveform='sine'),
            Oscillator(frequency=0.05, amplitude=1.0, offset=0.0, waveform='sine'),
            Oscillator(frequency=0.08, amplitude=1.0, offset=0.0, waveform='sine'),
            Oscillator(frequency=0.03, amplitude=1.0, offset=0.0, waveform='sine')
        ]
        self.waveforms = ['sine', 'square', 'sawtooth', 'triangle']
        self.osc_waveform_indices = [0, 0, 0, 0]

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_frame = ttk.Label(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.controls_frame = ttk.Frame(self.main_frame, padding="10", relief=tk.GROOVE, borderwidth=2)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.create_controls()

        self.update_video()

    def create_controls(self):
        for i, osc in enumerate(self.oscillators):
            label_text = f"Oscillator {i+1}"
            osc_frame = ttk.LabelFrame(self.controls_frame, text=label_text, padding="5")
            osc_frame.pack(fill=tk.X, pady=5)

            ttk.Label(osc_frame, text="Frequency").pack(side=tk.TOP, anchor=tk.W)
            freq_slider = ttk.Scale(osc_frame, from_=0.001, to=1.0, value=osc.frequency,
                                    orient=tk.HORIZONTAL, length=200,
                                    command=lambda val, idx=i: self.update_frequency(idx, float(val)))
            freq_slider.pack(fill=tk.X, expand=True)
            setattr(self, f'freq_slider_{i}', freq_slider)

            ttk.Label(osc_frame, text="Amplitude").pack(side=tk.TOP, anchor=tk.W)
            amp_slider = ttk.Scale(osc_frame, from_=0.0, to=2.0, value=osc.amplitude,
                                   orient=tk.HORIZONTAL, length=200,
                                   command=lambda val, idx=i: self.update_amplitude(idx, float(val)))
            amp_slider.pack(fill=tk.X, expand=True)
            setattr(self, f'amp_slider_{i}', amp_slider)

            waveform_button = ttk.Button(osc_frame, text=f"Waveform: {osc.waveform.capitalize()}",
                                         command=lambda idx=i: self.cycle_waveform(idx))
            waveform_button.pack(fill=tk.X, pady=5)
            setattr(self, f'waveform_button_{i}', waveform_button)

        global_frame = ttk.LabelFrame(self.controls_frame, text="Global Controls", padding="5")
        global_frame.pack(fill=tk.X, pady=10)

        reset_button = ttk.Button(global_frame, text="Reset Oscillators", command=self.reset_oscillators)
        reset_button.pack(fill=tk.X, pady=5)

        exit_button = ttk.Button(global_frame, text="Exit", command=self.on_closing)
        exit_button.pack(fill=tk.X, pady=5)

    def update_frequency(self, osc_idx, value):
        self.oscillators[osc_idx].frequency = value

    def update_amplitude(self, osc_idx, value):
        self.oscillators[osc_idx].amplitude = value

    def cycle_waveform(self, osc_idx):
        self.osc_waveform_indices[osc_idx] = (self.osc_waveform_indices[osc_idx] + 1) % len(self.waveforms)
        new_waveform = self.waveforms[self.osc_waveform_indices[osc_idx]]
        self.oscillators[osc_idx].waveform = new_waveform
        getattr(self, f'waveform_button_{osc_idx}')['text'] = f"Waveform: {new_waveform.capitalize()}"

    def reset_oscillators(self):
        for osc in self.oscillators:
            osc.start_time = time.time()
        print("Oscillator Timers Reset")

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            self.on_closing()
            return

        osc1_val = self.oscillators[0].get_value()
        osc2_val = self.oscillators[1].get_value()
        osc3_val = self.oscillators[2].get_value()
        osc4_val = self.oscillators[3].get_value()

        osc1_amp = self.oscillators[0].amplitude
        osc2_amp = self.oscillators[1].amplitude
        osc3_amp = self.oscillators[2].amplitude
        osc4_amp = self.oscillators[3].amplitude

        plasma_pattern = generate_plasma_effect(self.frame_width, self.frame_height, 
                                                osc1_val, osc2_val, osc3_val, osc4_val,
                                                osc1_amp, osc2_amp, osc3_amp, osc4_amp)

        alpha = 0.6
        blended_frame = cv2.addWeighted(frame, 1 - alpha, plasma_pattern, alpha, 0)

        rgb_image = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.photo_image = ImageTk.PhotoImage(image=pil_image)
        
        self.video_frame.config(image=self.photo_image)
        self.video_frame.image = self.photo_image

        self.root.after(10, self.update_video)

    def on_closing(self):
        print("Closing application...")
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSynthApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()