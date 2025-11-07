from effects import EffectManager
import logging
# from midi_input import *

"""Module to store basic global variables and effects manager"""

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

effects = EffectManager() # initialized with params in main.py

