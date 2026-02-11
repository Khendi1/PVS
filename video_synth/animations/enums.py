from enum import IntEnum, auto
import cv2

class Colormap(IntEnum):
    AUTUMN = 0
    BONE = auto()
    JET = auto()
    WINTER = auto()
    RAINBOW = auto()
    OCEAN = auto()
    SUMMER = auto()
    SPRING = auto()
    COOL = auto()
    HSV = auto()
    PINK = auto()
    HOT = auto()
    PARULA = auto()
    MAGMA = auto()
    INFERNO = auto()
    PLASMA = auto()
    VIRIDIS = auto()
    CIVIDIS = auto()
    TWILIGHT = auto()
    TWILIGHT_SHIFTED = auto()
    TURBO = auto()
    DEEPGREEN = auto()

COLORMAP_OPTIONS = [
    cv2.COLORMAP_AUTUMN,
    cv2.COLORMAP_BONE,
    cv2.COLORMAP_JET,
    cv2.COLORMAP_WINTER,
    cv2.COLORMAP_RAINBOW,
    cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SUMMER,
    cv2.COLORMAP_SPRING,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_HSV,
    cv2.COLORMAP_PINK,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_PARULA,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_CIVIDIS,
    cv2.COLORMAP_TWILIGHT,
    cv2.COLORMAP_TWILIGHT_SHIFTED,
    cv2.COLORMAP_TURBO,
    cv2.COLORMAP_DEEPGREEN
]


class MoirePattern(IntEnum):
    LINE = 0
    RADIAL = auto()
    GRID = auto()
    SPIRAL = auto()
    DIAMOND = auto()
    CHECKERBOARD = auto()
    HEXAGONAL = auto()


class MoireBlend(IntEnum):
    MULTIPLY = 0
    ADD = auto()
    SUB = auto()


class ShaderType(IntEnum):
    FRACTAL_0 = 0
    FRACTAL = auto()
    GRID = auto()
    PLASMA = auto()
    CLOUD = auto()
    MANDALA = auto()
    GALAXY = auto()
    TECTONIC = auto()
    BIOLUMINESCENT = auto()
    AURORA = auto()
    CRYSTAL = auto()


class AttractorType(IntEnum):
    LORENZ = 0
    CLIFFORD = auto()
    DE_JONG = auto()
    AIZAWA = auto()
    THOMAS = auto()


class AnimSource(IntEnum):
    PLASMA = 1
    REACTION_DIFFUSION = 2
    METABALLS = 3
    MOIRE = 4
    STRANGE_ATTRACTOR = 5
    PHYSARUM = 6
    SHADERS = 7
    DLA = 8
    CHLADNI = 9
    VORONOI = 10
