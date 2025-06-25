from param import ParamTable
from buttons import Buttons

NUM_OSCILLATORS = 6
osc_bank = []

FPS = 30 # Desired frame rate

save_index = 0

image_height = None
image_width = None

enable_polar_transform = False

panels = {}
params = ParamTable()
toggles = Buttons()
