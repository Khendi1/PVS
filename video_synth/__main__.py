"""
Main module for the video synthesizer application.
This module initializes the gui and video mixer, applies effects, and manages the main loop.

All effects classes are initialized, sequenced and applied by the EffectsManager.
The Effects manager is stored in globals.py for easy sharing with the GUI, but is initialized in main.

See the Program Architecture section in README.md for in-depth explainations
of module function and interation.

Author: Kyle Henderson
"""

import argparse
import logging
import cv2
import dearpygui.dearpygui as dpg 
from config import *
from gui import Interface
from generators import OscBank
from midi_input import *
from param import ParamTable
from mix import Mixer
from gui_elements import ButtonsTable

"""Creates ArgumentParser, configures arguments, returns parser"""
def parse_args():
    parser = argparse.ArgumentParser(description='Video Synthesizer initialization arguments')
    parser.add_argument(
        '-o',
        '--osc', 
        type=int, 
        default=DEFAULT_NUM_OSC, 
        help='Number of general purpose oscillators')
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

""" Main app setup and loop """
def main(num_osc, log_level, devices):
    global effects

    log.info("Initializing video synthesizer... Press 'q' or 'ESC' to quit")

    # all user modifiable parameters are stored here
    params = ParamTable()       # params have a min and max value
    toggles = ButtonsTable()    # toggles are binary

    # initialize general purpose oscillators for linking to params
    osc_bank = OscBank(params, num_osc)

    # Initialize video mixer, get a frame, create copies for feedback
    mixer = Mixer(params, devices)
    dry_frame = mixer.get_frame()  
    wet_frame = dry_frame.copy()
    prev_frame = dry_frame.copy()

    image_height, image_width = dry_frame.shape[:2]

    frame_count = 0

    # Initialize effects classes with image dimensions
    effects.init(params, toggles, image_width, image_height)

    # Automatically identify and initialize midi input controllers before creating the GUI
    # The midi controller must have an existing interface class, with its name appended to the names list
    controllers = identify_midi_ports(params)

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
            if mixer.skip or dry_frame is None:
                mixer.skip = False
                log.warning("Skipping frame due to source read failure")
                continue

            # update osc values if linked to params
            osc_bank.update()

            # apply effects sequence
            prev_frame, wet_frame = effects.modify_frames(
                dry_frame, wet_frame, prev_frame, frame_count
            )

            frame_count += 1

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', wet_frame)
            dpg.render_dearpygui_frame()

            # Break the loop if a quit key is pressed
            key = cv2.waitKey(1) & 0xFF
            if key in ESCAPE_KEYS:
                log.info(f"Received quit command, begining shutdown")
                break

    except KeyboardInterrupt:
        log.warning("Quit command detected, signaling thread stop...")
        
        if controllers:
            for c in controllers:
                c.thread_stop = True
                c.thread.join(timeout=5)
            # for c in controllers:     
                if c.thread.is_alive():
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
    log = config_log(args.log_level)
    main(args.osc, args.log_level, args.devices)