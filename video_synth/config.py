import logging
from enum import StrEnum, IntEnum, auto
# from midi_input import *

"""Module to store basic global variables and effects manager"""


class ParentClass(StrEnum):
    """Enumeration for different parent classes of parameters."""
    SRC_1_EFFECTS = "#4CAF50"
    SRC_2_EFFECTS = "#28702A"
    POST_EFFECTS = "#113F13"
    SRC_1_ANIMATIONS = "#2196F3"
    SRC_2_ANIMATIONS = "#0B5C9C"
    MIXER = "#B53D3D"


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

# default argparse values
DEFAULT_NUM_OSC = 5 
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_NUM_DEVICES = 5
DEFAULT_PATCH_INDEX = 0
DEFAULT_SAVE_FILE = "saved_values.yaml"

CONTROL_WINDOW_TITLE = "Control"
VIDEO_OUTPUT_WINDOW_TITLE = "Synthesizer Output"

WIDTH = None
HEIGHT = None

ESCAPE_KEYS = [ord('q'), ord('Q'), 27] # 27 is escape key

# effects = EffectManager(ParentClass.SRC_1_EFFECTS) # initialized with params in main.py

def enum_names(enum):
    return [enum(e).name for e in enum]
