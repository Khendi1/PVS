from enum import IntEnum, auto
import cv2

class Colormap(IntEnum):
    JET = 0
    VIRIDIS = auto()
    MAGMA = auto()
    PLASMA = auto()
    RAINBOW = auto()
    OCEAN = auto()
    SPRING = auto()
    COOL = auto()

COLORMAP_OPTIONS = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_RAINBOW,
    cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SPRING,
    cv2.COLORMAP_COOL,
]


class MoirePattern(IntEnum):
    LINE = 0
    RADIAL = auto()
    GRID = auto()


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
