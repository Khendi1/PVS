from buttons import ButtonsTable

"""
This module is being removed. 

Note that the effects classes are initialized in main using shared_object.py to avoid circular import
"""

toggles = ButtonsTable()

# TODO: make this a command line arg via argparse
# TODO: Pass into dependencies rather that using global varial (only gui is dep)
NUM_OSCILLATORS = 4 
osc_bank = []

# TODO: implement programatic generation of control panels by using the params and toggles
# this is currently broken and needs investigation. (only populates last param in list for each panel)
panels = {}
