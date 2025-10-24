from enum import IntEnum, Enum, auto

"""This module stores all custom enum classes"""

class LumaMode(IntEnum):
    NONE = 0
    WHITE = auto()
    BLACK = auto()

class NoiseType(IntEnum):
    NONE = 0
    GAUSSIAN = 1
    POISSON = 2
    SALT_AND_PEPPER = 3
    SPECKLE = 4
    SPARSE = 5
    RANDOM = 6

class WarpType(IntEnum):
    NONE = 0
    SINE = 1
    RADIAL = 2
    FRACTAL = 3
    PERLIN = 4
    WARP0 = 5  # this is a placeholder for the old warp_frame method; yet to be tested

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
