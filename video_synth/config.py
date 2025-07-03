from param import ParamTable
from buttons import Buttons

FPS = 30 # Desired frame rate

NUM_OSCILLATORS = 6 # TODO: make this a command line arg via argparse
osc_bank = []
# osc_vals = [osc.get_next_value() for osc in osc_bank]

# Index for loading saved patches from a file
save_index = 0

# Image dimensions; these are set after the first frame is read
image_height = None
image_width = None

# TODO: implement via buttons.py/toggles
enable_polar_transform = False

# TODO: implement programatic generation of control panels by using the params and toggles
# this is currently broken and needs investigation. (only populates last param in list for each panel)
panels = {}

# Initialize parameters; this is to be populated by individual classes/effects/generators
# and used to create the control panel
params = ParamTable()
toggles = Buttons()
