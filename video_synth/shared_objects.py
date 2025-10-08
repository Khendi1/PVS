from fx import *
from shapes import ShapeGenerator
from patterns3 import Patterns    
from enum import StrEnum, auto

"""
This module initializes and holds shared objects used across the application.
This simplifies code in main and imports into gui.py.
"""

class FX(StrEnum):
    BASIC = auto()
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

# Initialize effects classes; these contain Params to be modified by the generators
basic = None
color = None
pixels = None
noise = None
shapes = None
patterns = None
reflector = None
sync = None
warp = None
glitch = None
ptz = None

# Convenient dictionary of effects to be passed to the apply_effects function
fx = {}

def init_shared_objects(width, height):
    global basic, color, pixels, noise, shapes, patterns
    global reflector, sync, warp, glitch

    basic = Effects(width, height)
    color = Color()
    pixels = Pixels(width, height)
    noise = ImageNoiser(NoiseType.NONE)
    shapes = ShapeGenerator(width, height)
    patterns = Patterns(width, height)
    reflector = Reflector()                    
    sync = Sync() 
    warp = Warp(width, height)
    glitch = GlitchEffect()
    ptz = PTZ(width, height)

    # Convenient dictionary of effects to be passed to the apply_effects function
    fx.clear() # Clear any previous state
    fx.update({
        FX.BASIC: basic,
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