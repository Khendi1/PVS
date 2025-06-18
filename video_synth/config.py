from param import ParamTable

NUM_OSCILLATORS = 6
osc_bank = []

FPS = 30 # Desired frame rate

save_index = 0

image_height = None
image_width = None

enable_polar_transform = False

panels = {}
params = ParamTable()

toggles = {
    "invert": False,
    "grayscale": False,
    "edge_detect": False,
    "cartoonify": False,
    "sepia": False,
    "posterize": False,
    "solarize": False,
    "negative": False,
    "swirl": False,
    "fisheye": False,
    "kaleidoscope": False,
    "pixelate": False,
    "vignette": False,
    "noise_overlay": False,
    "feedback": False,
    "glitch": False,
    "gaussian_blur": False,
    "shift": False,
    "brightness_contrast": False,
    "shapes": False,
}
