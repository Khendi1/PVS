from generators import PerlinNoise, Interp, Oscillator
from param import Param
from math import floor

save_index = 0

image_height = None
image_width = None

val_threshold = 0  # Initial saturation threshold
val_hue_shift = 0  # Initial partial hue shift

enable_polar_transform = False

noise = 1
perlin_noise = PerlinNoise(noise)

NUM_OSCILLATORS = 6

osc_bank = []

params = {
    "hue_shift": Param("hue_shift", 0, 180, 100),
    "sat_shift": Param("sat_shift", 0, 255, 100),
    "val_shift":  Param("val_shift", 0, 255, 50),

    "alpha": Param("alpha", 0.0, 1.0, 0.9),
    "blur_kernel_size": Param("blur_kernel_size", 0, 100, 0),

    "num_glitches": Param("num_glitches", 0, 100, 0),
    "glitch_size": Param("glitch_size", 1, 100, 0),

    "val_threshold": Param("val_threshold", 0, 255, 0),
    "val_hue_shift": Param("val_hue_shift", -255, 255, 0),
    
    "x_shift": Param("x_shift", -1000, 1000, 0), # min/max depends on image size
    "y_shift": Param("y_shift", -1000, 1000, 0), # min/max depends on image size
    "zoom": Param("zoom", 0.5, 3, 1.0),
    "r_shift": Param("r_shift", -360, 360, 0),

    "polar_x": Param("polar_x", -1000, 1000, 0), # min/max depends on image size
    "polar_y": Param("polar_y", -1000, 1000, 0), # min/max depends on image size
    "polar_radius": Param("polar_radius", 0.1, 100, 1.0),
    
    "perlin_amplitude": Param("perlin_amplitude", 0.1, 1000, 1.0), # min/max depends on linked param
    "perlin_frequency": Param("perlin_frequency", 0.1, 500, 1.0),
    "perlin_octaves": Param("perlin_octaves", 0.1, 500, 1.0),

    "contrast": Param("contrast", 1.0, 3.0, 1.0),
    "brightness": Param("brightness", 0, 100, 0),
    "temporal_filter": Param("temporal_filter", 0, 1.0, 0.95)
    # Param("enable_polar_transform", False, True, enable_polar_transform),

}

def map_value(value, from_min, from_max, to_min, to_max):
  """
  Maps a value from one range to another.

  Args:
    value: The value to map.
    from_min: The minimum value of the original range.
    from_max: The maximum value of the original range.
    to_min: The minimum value of the target range.
    to_max: The maximum value of the target range.
  Returns:
    The mapped value in the target range, rounded down to the nearest integer.
  """
  # Calculate the proportion of the value within the original range
  proportion = (value - from_min) / (from_max - from_min)

  # Map the proportion to the target range
  mapped_value = to_min + proportion * (to_max - to_min)

  return floor(mapped_value)

def add_param(name, min_val, max_val, default_val):
    if name in params:
        raise ValueError(f"Parameter '{name}' already exists.")
    params[name] = Param(name, min_val, max_val, default_val)
    return params[name]