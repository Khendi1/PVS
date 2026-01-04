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

from config import *
# from generators import OscBank
from midi_input import *
from mix import Mixer
from effects import EffectManager

from PyQt6.QtWidgets import QApplication, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from pyqt_gui import PyQTGUI


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
        '--fullscreen',
        action='store_true',  # This makes it a boolean flag
        help='Launch the video output window in fullscreen mode.'
    )
    parser.add_argument(
        '--layout',
        default='quad',
        choices=['tabbed', 'quad'],
        help='Choose the GUI layout: "tabbed" for 1x2 grid, or "quad" for a 2x2 grid.'
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


""" Identifies midi controller ports from controller_names"""
def identify_midi_ports(params, controller_names):
    
    # Mido uses a default backend (often python-rtmidi) which handles platform differences.    
    controllers = []
    ports_found = False
    
    try:
        # List all available MIDI input ports. Ex: ["MIDI Mix 0", "SMC Mixer 1"]
        input_ports = mido.get_input_names()
        # Note that the controller_names list should follow a similar naming convention,
        # with the port number omitted

        if input_ports:
            string = "\nFound MIDI Input Devices:"

            # attempt to match found ports with known controller names
            for i, port_name in enumerate(input_ports):
                for name in controller_names:
                    if name in port_name:
                        found_controller = MidiControllerInterface(params, name=name, port_name=port_name)
                        # add to list of controllers so threads can be gracefully stopped on exit
                        controllers.append(found_controller)
                        # build output string with formatting so we only have to log once
                        string += f"\n\tInitialized midi controller: {name}"
        
            log.info(string)
            ports_found = True
        
        # List all available MIDI output ports
        output_ports = mido.get_output_names()
        if output_ports:
            string = "\nFound MIDI Output Devices:"
            for i, name in enumerate(output_ports):
                string += (f"\n\t[{i+1}] {name}")
            ports_found = True
            log.info(string)
            
    except Exception as e:
        log.exception(f"\nAn unexpected error occurred during port scan: {e}")
        return    
    if not ports_found:
        log.warning("No MIDI ports found by the operating system.")

    return controllers


"""Video processing loop"""
def video_loop(mixer, effects, should_quit, gui, fullscreen=False):
    wet_frame = dry_frame = mixer.get_mixed_frame()
    if dry_frame is None:
        log.error("Failed to get initial frame from mixer. Exiting video loop.")
        return

    prev_frame = dry_frame.copy()
    frame_count = 0
    
    # Using a CV2 window is now conditional
    use_cv2_window = not isinstance(gui.central_widget.layout(), QGridLayout)

    if use_cv2_window:
        cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("Modified Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not should_quit.is_set():
        dry_frame = mixer.get_mixed_frame()

        if mixer.skip or dry_frame is None:
            mixer.skip = False
            log.warning("Skipping frame due to source read failure")
            continue

        for effect_manager in effects:
            effect_manager.oscs.update()

        prev_frame, wet_frame = effects[2].modify_frames(
            dry_frame, wet_frame, prev_frame, frame_count
        )
        frame_count += 1
        
        if use_cv2_window:
            cv2.imshow('Modified Frame', wet_frame.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF in ESCAPE_KEYS:
                break
        else:
            # Convert frame to QImage and emit signal
            rgb_image = cv2.cvtColor(wet_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            gui.video_frame_ready.emit(qt_image)

    if use_cv2_window:
        cv2.destroyAllWindows()
    log.info("Video loop has gracefully stopped.")


""" Main app setup and loop """
def main(devices, controller_names, fullscreen, layout):

    log.info("Initializing video synthesizer... Press 'q' or 'ESC' to quit")

    src_1_effects = EffectManager(ParentClass.SRC_1_EFFECTS, WIDTH, HEIGHT)
    src_2_effects = EffectManager(ParentClass.SRC_2_EFFECTS, WIDTH, HEIGHT)
    post_effects = EffectManager(ParentClass.POST_EFFECTS, WIDTH, HEIGHT)
    effects = (src_1_effects, src_2_effects, post_effects)

    mixer = Mixer(effects, devices, WIDTH, HEIGHT)

    # Automatically identify and initialize midi controllers before creating the GUI
    CONTROLLER_NAMES = [] # Placeholder for now, assumed to be defined elsewhere
    controllers = [] #identify_midi_ports(src_1_params, CONTROLLER_NAMES) # TODO: which params to use?

    # log.info(f'Starting program with {len(params.keys())} tunable parameters')

    # Handle Ctrl+C from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    main_window = PyQTGUI(effects, layout, mixer)

    main_window.show()
    
    should_quit = threading.Event()
    
    video_thread = threading.Thread(
        target=video_loop, 
        args=(mixer, effects, should_quit, main_window, fullscreen)
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
    main(args.devices, CONTROLLER_NAMES, args.fullscreen, args.layout)