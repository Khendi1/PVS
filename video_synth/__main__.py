"""
Main module for the video synthesizer application.
This module initializes the video mixer, applies effects, and manages the main loop.

All effects classes use Parameters and Toggles for control.
- Parameters are numerical data in a range (>2) of minimum and maximum values.
- Toggles are boolean.
These are managed by the ParamsTable and ButtonsTable respectively.
These structures must be passed to each effect class that wishes to permit user-
control, oscillator linking, gui sliders/buttons, effect sequencing, etc.

All effects classes are managed by the EffectsManager (init'd in globals.py for easy sharing with the GUI)

To create new effects with user controlable parameters and their corresponding 
sliders, you must first create a Param and add it to the ParamTable.
This is typically done in class __init__ functions.
Then you must create a panel/trackbar for it manually (for now).
Existing classes expose a create_gui_panel function which are called in gui.py

Parameter values can be modified via the GUI or linked to MIDI controllers.
See how sample controllers are mapped in midi.py

Author: Kyle Hendersons
"""

import argparse
import logging
import cv2
import dearpygui.dearpygui as dpg 
from globals import effects
from gui import Interface
from generators import OscBank
from midi_input import MidiInputController, MidiMix, SMC_Mixer
from param import ParamTable
from mix import Mixer
from gui_elements import ButtonsTable
import os

os.environ["OPENCV_LOG_LEVEL"] = "INFO"

# default argparse&log values
DEFAULT_NUM_OSC = 4 
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_PATCH_INDEX = 0


# Global logging module config 
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format='[%(asctime)s,%(msecs)03d] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def parse_args():
    """
    Create ArgumentParser, configure arguments, return parser
    """
    parser = argparse.ArgumentParser(description='Video Synthesizer initialization arguments')
    parser.add_argument(
        '--osc', 
        type=int, 
        default=DEFAULT_NUM_OSC, 
        help='Number of general purpose oscillators')
    parser.add_argument(
        '--log-level',
        default=DEFAULT_LOG_LEVEL,  
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='Set the logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG)'
    )
    parser.add_argument(
        '--patch',
        default=DEFAULT_PATCH_INDEX,
        type=int,
        help='Initialize program with a saved patch'
    )
    return parser.parse_args()


""" Main app setup and loop """
def main(num_osc, log_level):
    
    log.info("Initializing video synthesizer...")

    # all user modifiable parameters are stored here
    params = ParamTable()       # params have a min and max value
    toggles = ButtonsTable()    # toggles are binary

    # initialize general purpose oscillators for linking to params
    osc_bank = OscBank(params, num_osc)

    # Initialize video mixer, get a frame, create copies for feedback
    mixer = Mixer(params)
    dry_frame = mixer.get_frame()  
    wet_frame = dry_frame.copy()
    prev_frame = dry_frame.copy()

    image_height, image_width = dry_frame.shape[:2]

    frame_count = 0

    # Initialize effects classes with image dimensions
    effects.init(params, toggles, image_width, image_height)

    # TODO: This assumes both controllers are always connected in a specific order; improve this
    # test_ports()

    # Initialize the midi input controller before creating the GUI
    controller1 = MidiInputController(controller=MidiMix(params))
    controller2 = MidiInputController(controller=SMC_Mixer(params))

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface(params, osc_bank, toggles)
    gui.create_control_window(params, mixer=mixer, osc_bank=osc_bank)

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Modified Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    log.info(f'Starting program with {len(params.keys())} tunable parameters')
    # for p in params.values():
    #     log.info(f'{p.name}: {p.min}-{p.max}, {p.value}')

    try:
        while True:

            # retreive and mix frames from the selected sources
            dry_frame = mixer.get_frame()

            # frame retrieval may fail when changing sources; skip
            if mixer.skip1 or dry_frame is None:
                mixer.skip1 = False
                log.warning("Skipping frame due to source read failure")
                continue

            # update osc values if linked to params
            osc_bank.update()

            prev_frame, wet_frame = effects.modify_frames(
                dry_frame, wet_frame, prev_frame, frame_count
            )

            frame_count += 1

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', wet_frame)
            dpg.render_dearpygui_frame()

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info(f"Received quit command, begining shutdown")
                break

    except KeyboardInterrupt:
        log.warning("Quit command detected, signaling MIDI thread to stop...")
        
        controller1.thread_stop, controller2.thread_stop = True, True
       
        # Wait for the MIDI thread to finish, with a timeout
        controller1.thread.join(timeout=5)
        controller2.thread.join(timeout=5)

        if controller1.thread.is_alive() or controller2.thread.is_alive():
            log.warning("MIDI thread did not terminate gracefully. Forcing exit.")
        else:
            log.info("MIDI thread stopped successfully.")

    finally:

        # Destroy all windows 
        dpg.destroy_context()
        cv2.destroyAllWindows()

        # close capture streams
        for cap in mixer.live_caps:
            if cap and cap.isOpened():
                cap.release()
        
        log.info("Goodbye!")


if __name__ == "__main__":
    args = parse_args()
    main(args.osc, args.log_level)