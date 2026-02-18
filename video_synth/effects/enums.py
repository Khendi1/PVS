from enum import IntEnum, Enum, auto

class BlendModes(Enum):
    NONE = 0
    REPLACE = 1
    BLEND = 2
    ADD = 3

"""This section stores all local custom enum classes"""
class NoiseType(Enum):
    NONE = 0
    GAUSSIAN = 1
    POISSON = 2
    SALT_AND_PEPPER = 2
    SPECKLE = 4
    SPARSE = 5
    RANDOM = 6


class WarpType(Enum):
    NONE = 0
    SINE = auto()
    RADIAL = auto()
    FRACTAL = auto()
    PERLIN = auto()
    WARP0 = auto()
    FEEDBACK = auto()


"""Enumeration of blur modes"""
class BlurType(Enum):
    NONE = 0
    GAUSSIAN = auto()
    MEDIAN = auto()
    BOX = auto()
    BILATERAL = auto()


"""Enumeration of sharpening modes"""
class SharpenType(Enum):
    NONE = 0
    TEST = auto()
    KERNEL = auto()
    UNSHARP = auto()
    LAPLACIAN = auto()


"""Enum to access hsv tuple indicies"""
class HSV(IntEnum):
    H = 0
    S = 1
    V = 2


class ReflectionMode(Enum):
    """Enumeration for different image reflection modes."""

    NONE = 0  # No reflection
    HORIZONTAL = auto()  # Reflect across the Y-axis (flip horizontally)
    VERTICAL = auto() # Reflect across the X-axis (flip vertically)
    BOTH = auto()  # Reflect across both X and Y axes (flip horizontally and vertically)
    QUAD_SYMMETRY = auto()  # Reflect across both axes with quadrants (not implemented)
    SPLIT = auto()  # Reflect left half onto right half
    KALEIDOSCOPE = auto()


class Shape(Enum):
    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2
    LINE = 3
    DIAMOND = 4
    NONE = 5
