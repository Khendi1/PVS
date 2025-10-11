from param import ParamTable
from buttons import Buttons

"""
This module stores common variables used across main, the gui, and most effects
Note that the effects classes are initialized in shared_object.py to avoid circular import
"""

# Initialize parameters; this is to be populated by individual classes/effects/generators
# and used to create the control panel
params = ParamTable()
toggles = Buttons()

FPS = 45 # Desired frame rate

cap1 = None
cap2 = None

# TODO: make this a command line arg via argparse
NUM_OSCILLATORS = 4 
osc_bank = [] 

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


