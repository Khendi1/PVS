from param import ParamTable
from buttons import Buttons

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

# TODO: move to a better location (the class that uses it)
posc_bank = []  

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

