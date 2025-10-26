from enum import IntEnum, Enum, auto

"""This module stores all custom enum classes"""

class LumaMode(IntEnum):
    NONE = 0
    WHITE = auto()
    BLACK = auto()

class NoiseType(IntEnum):
    NONE = 0
    GAUSSIAN = auto()
    POISSON = auto()
    SALT_AND_PEPPER = auto()
    SPECKLE = auto()
    SPARSE = auto()
    RANDOM = auto()

class WarpType(IntEnum):
    NONE = 0
    SINE = auto()
    RADIAL = auto()
    FRACTAL = auto()
    PERLIN = auto()
    WARP0 = auto()  # placeholder for old warp_frame method; yet to be tested

"""Enumeration of blur modes"""
class BlurType(IntEnum):
    NONE = 0
    GAUSSIAN = 1
    MEDIAN = 2
    BOX = 3
    BILATERAL = 4

"""Enumeration of sharpening modes"""
class SharpenType(IntEnum):
    NONE = 0
    SHARPEN = 1
    UNSHARP_MASK = 2

"""Enum to access hsv tuple indicies"""
class HSV(IntEnum):
    H = 0
    S = 1
    V = 2

class ReflectionMode(Enum):
    """Enumeration for different image reflection modes."""

    NONE = 0  # No reflection
    HORIZONTAL = 1  # Reflect across the Y-axis (flip horizontally)
    VERTICAL = 2  # Reflect across the X-axis (flip vertically)
    BOTH = 3  # Reflect across both X and Y axes (flip horizontally and vertically)
    QUAD_SYMMETRY = 4  # Reflect across both axes with quadrants (not implemented)
    SPLIT = 5  # Reflect left half onto right half

    def __str__(self):
        return self.name.replace("_", " ").title()


class Shape(IntEnum):

    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2
    LINE = 3
    DIAMOND = 4
    NONE = 5


class MoireType(IntEnum):
    ROTATIONAL = 0
    TRANSLATIONAL = auto()
    CIRCULAR = auto()


class OscillatorShape(Enum):
    NONE = 0
    SINE = 1
    SQUARE = 2
    TRIANGLE = 3
    SAWTOOTH = 4
    PERLIN = 5

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            value = value.lower()
            if value == "sine":
                return cls.SINE
            elif value == "square":
                return cls.SQUARE
            elif value == "triangle":
                return cls.TRIANGLE
            elif value == "sawtooth":
                return cls.SAWTOOTH
            elif value == "perlin":
                return cls.PERLIN
        return cls(value)
    

class MixModes(IntEnum):
    BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2

""" 
The MixSources Enum class is used to standardize strings 

For cv2 sources (devices, images, video files), there is an A and B enum
so that different they can be used on source 1 and source 2 simultaneously

Note that if you want to mix two video files, the sources must be set to 
"""
class MixSources(Enum):
    DEVICE_1 = 0
    DEVICE_2 = auto()
    VIDEO_FILE_1 = auto()
    VIDEO_FILE_2 = auto()
    IMAGE_FILE_1 = auto()
    IMAGE_FILE_2 = auto()
    METABALLS_ANIM = auto()
    PLASMA_ANIM = auto()
    REACTION_DIFFUSION_ANIM = auto()
    MOIRE_ANIM = auto()
    SHADER_ANIM = auto()