from effects import EffectManager

"""
This module stores shared objects used across the application.

Now classes with modifyable params are expected to define a create_gui_panel function and call it gui.py 
gui panels within
"""

CONTROL_WINDOW_TITLE = "Control"
VIDEO_OUTPUT_WINDOW_TITLE = "Synthesizer Output"

WIDTH = None
HEIGHT = None

effects = EffectManager() # initialized with params in main.py

