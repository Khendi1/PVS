from enum import Enum, StrEnum, IntEnum, auto
import logging

"""Module to store global enums and constants"""

# Set to the number of frames to average together per 
# performance profiling output message, 0 to disable
DEFAULT_PROFILING_FRAMES = 0 # 100 
DEBUG = DEFAULT_PROFILING_FRAMES > 0
DEFAULT_LOG_LEVEL = logging.INFO

# Number of USB video devices to search for on boot, 
# will safely ignore extra devices if not found
DEFAULT_NUM_DEVICES = 5 
MAX_DEVICES = 10

DEFAULT_SAVE_FILE = "saved_values.yaml"
DEFAULT_PATCH_INDEX = 0

VIDEO_OUTPUT_WINDOW_TITLE = "Synthesizer Output"

WIDTH = 640
HEIGHT = 480

# Quit keys: 'q', 'Q', or 'ESC'
ESCAPE_KEYS = [ord('q'), ord('Q'), 27]

class Layout(Enum):
    """Enumeration for different control layouts, which determine how parameters are grouped and displayed in the GUI."""
    SPLIT = 0           
    QUAD_PREVIEW = 1
    QUAD_FULL = 2


class OutputMode(Enum):
    NONE = 0
    WINDOW = 1
    FULLSCREEN = 2


class Groups(StrEnum):
    """Enumeration for different group classes of parameters."""
    SRC_1_EFFECTS = "#4CAF50"
    SRC_2_EFFECTS = "#28702A"
    POST_EFFECTS = "#113F13"
    SRC_1_ANIMATIONS = "#2196F3"
    SRC_2_ANIMATIONS = "#0B5C9C"
    MIXER = "#B53D3D"
    AUDIO_REACTIVE = "#FF9800"
    OBS = "#9C27B0"
    USER_SETTINGS = "#F0F0F0"
    UNCATEGORIZED = "#000000"


class MixerSource(IntEnum):
    """
    Enumeration for mixer sources, used to identify which source is being processed in a consistent way.
    """
    SRC_1 = 0
    SRC_2 = 1
    POST = 2


class Widget(StrEnum):
    """ Enumeration for different types of GUI widgets to represent parameters in the PyQT interface.
    Each type corresponds to a specific way of displaying and interacting with the parameter in the GUI."""
    SLIDER = auto()
    DROPDOWN = auto()
    RADIO = auto()
    TOGGLE = auto()
    SPLIT_ROW_SLIDER = auto()
    SPLIT_ROW_DROPDOWN = auto()


class Toggle(Enum):
    OFF = 0
    ON = 1