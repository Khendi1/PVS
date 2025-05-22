from generators import PerlinNoise, Interp, Oscillator
from param import Param

save_index = 0

image_height = None
image_width = None

val_threshold = 0  # Initial saturation threshold
val_hue_shift = 0  # Initial partial hue shift

enable_polar_transform = False

noise = 1
perlin_noise = PerlinNoise(noise)

osc_bank = []
osc_vals = []

params = {
    "hue_shift": Param("hue_shift", 0, 180, 100),
    "sat_shift": Param("sat_shift", 0, 255, 100),
    "val_shift":  Param("val_shift", 0, 255, 50),
    "alpha": Param("alpha", 0.0, 1.0, 0.9),
    "num_glitches": Param("num_glitches", 0, 100, 0),
    "glitch_size": Param("glitch_size", 1, 100, 0),
    "val_threshold": Param("val_threshold", 0, 255, 0),
    "val_hue_shift": Param("val_hue_shift", -255, 255, 0),
    "blur_kernel_size": Param("blur_kernel_size", 0, 100, 0),
    "x_shift": Param("x_shift", -1000, 1000, 0), # min/max depends on image size
    "y_shift": Param("y_shift", -1000, 1000, 0), # min/max depends on image size
    "r_shift": Param("r_shift", -360, 360, 0),
    "polar_x": Param("polar_x", -1000, 1000, 0), # min/max depends on image size
    "polar_y": Param("polar_y", -1000, 1000, 0), # min/max depends on image size
    "polar_radius": Param("polar_radius", 0.1, 100, 1.0),
    "perlin_amplitude": Param("perlin_amplitude", 0.1, 1000, 1.0), # min/max depends on linked param
    "perlin_frequency": Param("perlin_frequency", 0.1, 500, 1.0),
    "perlin_octaves": Param("perlin_octaves", 0.1, 500, 1.0),
    "contrast": Param("contrast", 1.0, 3.0, 1.0),
    "brightness": Param("brightness", 0, 100, 0)
    # Param("enable_polar_transform", False, True, enable_polar_transform),
}

indices = {
}
