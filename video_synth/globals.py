from enum import StrEnum, auto

"""
This module initializes shared objects used across the application.
Now classes with modifyable params are expected to define a create_sliders function and call it gui.py 
gui panels within
"""

# Convenient dictionary of effects to be passed to the apply_effects function
fx_dict = {}

class FX(StrEnum):
    FEEDBACK = auto()
    COLOR = auto()
    PIXELS = auto()
    NOISE = auto()
    PTZ = auto()
    SHAPES = auto()
    PATTERNS = auto()
    REFLECTOR = auto()
    SYNC = auto()
    WARP = auto()
    GLITCH = auto()
    LISSAJOUS = auto()
