# from generators import PerlinNoise
from param import Param, ParamTable
from math import floor

save_index = 0

image_height = None
image_width = None

enable_polar_transform = False

noise = 1
# perlin_noise = PerlinNoise(noise)
# "perlin_amplitude": Param("perlin_amplitude", 0.1, 1000, 1.0), # min/max depends on linked param
# "perlin_frequency": Param("perlin_frequency", 0.1, 500, 1.0),
# "perlin_octaves": Param("perlin_octaves", 0.1, 500, 1.0)

NUM_OSCILLATORS = 6
osc_bank = []

FPS = 30 # Desired frame rate

panels = {"""<family>:[param1_name,...],
          "HSV": ["hue_shift","sat_shift",...]"""}



params = ParamTable()

def add(name, min_val, max_val, default_val, family=None):
    """
    Adds a new parameter to the params dictionary.
    If a parameter with the same name already exists, raises a ValueError.
    Args:
        name: The name of the parameter.
        min_val: The minimum value for the parameter.
        max_val: The maximum value for the parameter.
        default_val: The default value for the parameter.
    Returns:
        The newly created Param object.
    Raises:
        ValueError: If a parameter with the same name already exists.
    """
    if name in params:
        raise ValueError(f"Parameter '{name}' already exists.")
    params[name] = Param(name, min_val, max_val, default_val, family=family)
       
    return params[name]

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