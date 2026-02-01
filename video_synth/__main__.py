"""
Main module for the video synthesizer application.
This module initializes the gui and video mixer, applies effects, and manages the main loop.

All effects classes are initialized, sequenced and applied by the EffectsManager.
The Effects manager is stored in globals.py for easy sharing with the GUI, but is initialized in main.

See the Program Architecture section in README.md for in-depth explainations
of module function and interation.

Author: Kyle Henderson 
"""

import sys
import argparse
import logging
import cv2
import signal
import threading
import numpy as np

from settings import UserSettings
from common import *
from midi_input import *
from mix import Mixer
from effects import EffectManager

from PyQt6.QtWidgets import QApplication, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from pyqt_gui import PyQTGUI


# List of MIDI controller class names to identify and initialize
CONTROLLER_NAMES = [SMC_Mixer.__name__, MidiMix.__name__]

# Number of USB video devices to search for on boot, 
# will safely ignore extra devices if not found
DEFAULT_NUM_DEVICES = 5 

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_SAVE_FILE = "saved_values.yaml"
DEFAULT_PATCH_INDEX = 0

VIDEO_OUTPUT_WINDOW_TITLE = "Synthesizer Output"

WIDTH = 640
HEIGHT = 480

# Quit keys: 'q', 'Q', or 'ESC'
ESCAPE_KEYS = [ord('q'), ord('Q'), 27]

"""Creates ArgumentParser, configures arguments, returns parser"""
def parse_args():
    parser = argparse.ArgumentParser(description='Video Synthesizer initialization arguments')
    parser.add_argument(
        '-l',
        '--log-level',
        default=DEFAULT_LOG_LEVEL,  
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='Set the logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG)'
    )
    parser.add_argument(
        '-d',
        '--devices',
        default=DEFAULT_NUM_DEVICES,
        choices=[i for i in range(1,11)],
        type=int,
        help='Number of USB video capture devices to search for on boot. Will safely ignore a extra devices if not found'
    )
    parser.add_argument(
        '-p',
        '--patch',
        default=DEFAULT_PATCH_INDEX,
        type=int,
        help='Initialize program with a saved patch. Defaults to using "saved_values.yaml", but can be changed'
    )
    parser.add_argument(
        '-f',
        '--file',
        default= DEFAULT_SAVE_FILE,
        type=str,
        help='Use an alternate save file. Must still be located in the save directory'
    )
    parser.add_argument(
        '-c',
        '--control-layout',
        default=LayoutType.QUAD_PREVIEW.name,
        choices=[item.name for item in LayoutType],
        help='Choose the GUI layout: "tabbed" for 1x2 grid, or "quad" for a 2x2 grid.'
    )
    parser.add_argument(
        '-o',
        '--output-mode',
        default=OutputMode.NONE.name,
        choices=[item.name for item in OutputMode],
        help='Use an external window for video output'
    )
    parser.print_help()
    return parser.parse_args()


""" Global logging module configuration using """
def config_log(log_level):
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s,%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    log = logging.getLogger(__name__)
    return log


"""Video processing loop"""
def video_loop(mixer, effects, should_quit, gui, settings):
    wet_frame = dry_frame = mixer.get_mixed_frame()
    if dry_frame is None:
        log.error("Failed to get initial frame from mixer. Exiting video loop.")
        return

    prev_frame = dry_frame.copy()
    frame_count = 0

    if settings.output_mode.value != OutputMode.NONE.value:
        cv2.namedWindow(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        if settings.output_mode.value == OutputMode.FULLSCREEN.value:  # Fullscreen mode
            cv2.setWindowProperty(VIDEO_OUTPUT_WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not should_quit.is_set():
        dry_frame = mixer.get_mixed_frame()

        if mixer.skip or dry_frame is None:
            mixer.skip = False
            log.warning("Skipping frame due to source read failure")
            continue

        for effect_manager in effects:
            effect_manager.oscs.update()

        prev_frame, wet_frame = effects[2].get_frames(
            dry_frame, wet_frame, prev_frame, frame_count
        )
        frame_count += 1
        
        if settings.output_mode.value != OutputMode.NONE.value:
            cv2.imshow(VIDEO_OUTPUT_WINDOW_TITLE, wet_frame.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF in ESCAPE_KEYS:
                break
        else:
            # Convert frame to QImage and emit signal
            rgb_image = cv2.cvtColor(wet_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            gui.video_frame_ready.emit(qt_image)

    if settings.output_mode.value != OutputMode.NONE.value:
        cv2.destroyAllWindows()
    log.info("Video loop has gracefully stopped.")


""" Main app setup and loop """
def main(settings, controller_names):

    log.info("Initializing video synthesizer... Press 'q' or 'ESC' to quit")

    src_1_effects = EffectManager(ParentClass.SRC_1_EFFECTS, WIDTH, HEIGHT)
    src_2_effects = EffectManager(ParentClass.SRC_2_EFFECTS, WIDTH, HEIGHT)
    post_effects = EffectManager(ParentClass.POST_EFFECTS, WIDTH, HEIGHT)
    effects = (src_1_effects, src_2_effects, post_effects)

    mixer = Mixer(effects, settings.num_devices, WIDTH, HEIGHT)

    # Automatically identify and initialize midi controllers before creating the GUI
    CONTROLLER_NAMES = [SMC_Mixer.__name__, MidiMix.__name__]
    controllers = identify_midi_ports(controller_names, 
                                      src_1_effects.params, 
                                      src_2_effects.params, 
                                      post_effects.params)

    # log.info(f'Starting program with {len(params.keys())} tunable parameters')

    # Handle Ctrl+C from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    main_window = PyQTGUI(effects, settings.layout.value, mixer)

    main_window.show() 
    
    should_quit = threading.Event()
    
    video_thread = threading.Thread(
        target=video_loop,
        args=(mixer, effects, should_quit, main_window, settings)
    )
    video_thread.start()
    
    log.info("Starting PyQt event loop.")
    exit_code = app.exec()
    log.info("PyQt event loop finished.")
    
    should_quit.set()
    video_thread.join()
    sys.exit(exit_code)

    if controllers:
        for c in controllers:
            c.thread_stop = True
            c.thread.join(timeout=5)
            if c.thread.is_alive():
                log.warning("MIDI thread did not terminate gracefully. Forcing exit.")
            else:
                log.info("MIDI thread stopped successfully.")

    log.info("Goodbye!")


if __name__ == "__main__":
    args = parse_args()
    log = config_log(args.log_level)
    settings = UserSettings(**args.__dict__)
    main(settings, CONTROLLER_NAMES)
