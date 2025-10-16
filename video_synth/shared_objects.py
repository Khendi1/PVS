from fx import *
from patterns3 import Patterns    
from enum import StrEnum, auto
from mix import Mixer
from config import *

"""
This module initializes shared objects used across the application.
Now classes with modifyable params are expected to define a create_sliders function and call it gui.py 
gui panels within
"""

global params

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

def init_effects(params, width, height):
    global fx_dict
    global feedback, color, pixels, noise, shapes, patterns
    global reflector, sync, warp, glitch

    feedback = Feedback(params, width, height)
    color = Color(params)
    pixels = Pixels(params, width, height)
    noise = ImageNoiser(params, NoiseType.NONE)
    shapes = ShapeGenerator(params, width, height)
    patterns = Patterns(params, width, height)
    reflector = Reflector(params, )                    
    sync = Sync(params) 
    warp = Warp(params, width, height)
    glitch = GlitchEffect(params)
    ptz = PTZ(params, width, height)

    # Convenient dictionary of effects to be passed to the apply_effects function
    fx_dict.clear() # Clear any previous state
    fx_dict.update({
        FX.FEEDBACK: feedback,
        FX.COLOR: color,
        FX.PIXELS: pixels,
        FX.NOISE: noise,
        FX.SHAPES: shapes,
        FX.PATTERNS: patterns,
        FX.REFLECTOR: reflector,
        FX.SYNC: sync,
        FX.WARP: warp,
        FX.GLITCH: glitch,
        FX.PTZ: ptz
    })