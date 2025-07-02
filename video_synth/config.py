from param import ParamTable
from buttons import Buttons
from generators import Oscillator

FPS = 30 # Desired frame rate

NUM_OSCILLATORS = 6 # TODO: make this a command line arg via argparse
osc_bank = [Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4) \
            for i in range(NUM_OSCILLATORS)]

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
