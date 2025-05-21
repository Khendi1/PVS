from generators import PerlinNoise, Interp, Oscillator

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
    "hue_shift": Param("hue_shift", -180, 180, 100),
    "hue_shift": Param("sat_shift", 0, 255, 100),
    "val_shift":  Param("val_shift", 0, 255, 50),
    "alpha": Param("alpha", 0.0, 1.0, 0.9),
    "num_glitches": Param("num_glitches", 0, 100, 0),
    "glitch_size": Param("glitch_size", 1, 100, 0),
    "val_threshold": Param("val_threshold", 0, 255, 0),
    "val_hue_shift": Param("val_hue_shift", -255, 255, 0),
    "blur_kernel_size": Param("blur_kernel_size", 1, 100, 0),
    "x_shift": Param("x_shift", -1000, 1000, 0), # min/max depends on image size
    "y_shift": Param("y_shift", -1000, 1000, 0), # min/max depends on image size
    "r_shift": Param("r_shift", -360, 360, 0),
    "polar_x": Param("polar_x", -1000, 1000, 0), # min/max depends on image size
    "polar_y": Param("polar_y", -1000, 1000, 0), # min/max depends on image size
    "polar_radius": Param("polar_radius", 0.1, 100, 1),
    "perlin_amplitude": Param("perlin_amplitude", 0.1, 1000, 1), # min/max depends on linked param
    "perlin_frequency": Param("perlin_frequency", 0.1, 1000, 1),
    "perlin_octaves": Param("perlin_octaves", 0.1, 1000, 1),
    "contrast": Param("contrast", 1.0, 3.0, 1.0),
    "brightness": Param("brightness", 0, 100, 0)
    # Param("enable_polar_transform", False, True, enable_polar_transform),
}

class Param:
    def __init__(self, name, min_val, max_val, default_val):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val
        self.value = default_val

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        if isinstance(other, Param):
            return self.value + other.value
        elif isinstance(other, (int, float)):
            return self.value + other
        else:
            raise TypeError("Unsupported type for addition")
    
    def __sub__(self, other):
        if isinstance(other, Param):
            return self.value - other.value
        elif isinstance(other, (int, float)):
            return self.value - other
        else:
            raise TypeError("Unsupported type for subtraction")

    def __mul__(self, other):
        if isinstance(other, Param):
            return self.value * other.value
        elif isinstance(other, (int, float)):
            return self.value * other
        else:
            raise TypeError("Unsupported type for multiplication")
    
    def __truediv__(self, other):
        if isinstance(other, Param):
            return self.value / other.value
        elif isinstance(other, (int, float)):
            return self.value / other
        else:
            raise TypeError("Unsupported type for division")
    
    def reset(self):
        self.value = self.default_val
        return self.value
    
    def set_max(self, max_val):
        self.max_val = max_val
        if self.value > max_val:
            self.value = max_val
        return self.value
    
    def set_min(self, min_val):
        self.min_val = min_val
        if self.value < min_val:
            self.value = min_val
        return self.value

    def set_min_max(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        if self.value < min_val:
            self.value = min_val
        elif self.value > max_val:
            self.value = max_val
        return self.value
    
    def set_value(self, value):
        if value < self.min_val:
            self.value = self.min_val
        elif value > self.max_val:
            self.value = self.max_val
        else:
            self.value = value
        return self.value

    

