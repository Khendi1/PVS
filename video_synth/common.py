from enum import Enum, StrEnum, IntEnum, auto

"""Module to store global enums"""

class LayoutType(Enum):
    SPLIT = 0
    QUAD_PREVIEW = 1
    QUAD_FULL = 2


class OutputMode(Enum):
    NONE = 0
    WINDOW = 1
    FULLSCREEN = 2


class ParentClass(StrEnum):
    """Enumeration for different parent classes of parameters."""
    SRC_1_EFFECTS = "#4CAF50"
    SRC_2_EFFECTS = "#28702A"
    POST_EFFECTS = "#113F13"
    SRC_1_ANIMATIONS = "#2196F3"
    SRC_2_ANIMATIONS = "#0B5C9C"
    MIXER = "#B53D3D"
    GENERAL_LFOS = "#FF9800"
    SETTINGS = "#9C27B0"


class SourceIndex(IntEnum):
    SRC_1 = 0
    SRC_2 = 1
    POST = 2


class WidgetType(StrEnum):
    SLIDER = auto()
    DROPDOWN = auto()
    RADIO = auto()
    SPLIT_ROW_SLIDER = auto()
    SPLIT_ROW_DROPDOWN = auto()

