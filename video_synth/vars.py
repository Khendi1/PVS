
# global control_image
# global sliders, buttons
# global save_index
# global interp, fade

# sliders = {}

image_height = None
image_width = None

# Default values for hue, saturation, and value shifts
hue_shift = 100  # Value to shift the hue (can be positive or negative)
sat_shift = 100  # Value to shift the saturation (0 to 255)
val_shift = 50  # Value to shift the value (0 to 255)
alpha = 95  # Adjust for desired feedback intensity
num_glitches = 0  # Number of glitches to apply
glitch_size = 1  # Size of each glitch
val_threshold = 0  # Initial saturation threshold
val_hue_shift = 0  # Initial partial hue shift
blur_kernel_size = 0  # Initial blur kernel size
x_speed = 0
y_speed = 0
x_size = 0
y_size = 0
x_shift = 0
y_shift = 0
r_shift= 0
x_shift_mode = 0  # 0: manual, 1: perlin, 2: sine, 3: square, 4: sawtooth, 5: triangle
y_shift_mode = 0  # 0: manual, 1: perlin, 2: sine, 3: square, 4: sawtooth, 5: triangle
polar_x = 0 # init to frame center
palar_y = 0 # init to center
polar_coord_mode = 0 # init to current config, create button for alt config


perlin_amplitude = 0
perlin_frequency = 0
perlin_octaves = 0
perlin_noise = 0
noise = 1

osc_mode = 0

freq = 0
amp = 0
