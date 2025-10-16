from param import ParamTable
from buttons import ButtonsTable

"""
This module stores required config data and common params used across main, the gui, and most effects

Note that the effects classes are initialized in main using shared_object.py to avoid circular import
"""


# Initialize parameters; this is to be populated by individual classes/effects/generators
# and used to create the control panel
# params = ParamTable()
toggles = ButtonsTable()

# TODO: make this a command line arg via argparse
NUM_OSCILLATORS = 4 
osc_bank = []

# TODO: implement programatic generation of control panels by using the params and toggles
# this is currently broken and needs investigation. (only populates last param in list for each panel)
panels = {}
