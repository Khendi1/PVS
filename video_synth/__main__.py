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
from globals import effects
from gui import Interface
from generators import OscBank
from midi_input import MidiInputController, MidiMix, SMC_Mixer
from param import ParamTable
from mix import Mixer
from gui_elements import ButtonsTable
from usbmonitor import USBMonitor
from usbmonitor.attributes import *


# default argparse values
DEFAULT_NUM_OSC = 4 
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_PATCH_INDEX = 0
DEFAULT_SAVE_FILE = "saved_values.yaml"
DEFAULT_CONFIG_HELPER = False

# Global logging module config 
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format='[%(asctime)s,%(msecs)03d] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

"""Creates ArgumentParser, configures arguments, returns parser"""
def parse_args():
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
        help='Initialize program with a saved patch. Defaults to using "saved_values.yaml", but can be changed'
    )
    parser.add_argument(
        '--file',
        default= DEFAULT_SAVE_FILE,
        type=str,
        help='Use an alternate save file. Must still be located in the save directory'
    )
    return parser.parse_args()

""" Find and initialize USB MIDI controllers, return list of controllers """
def id_midi_devices(controllers, params):

    monitor = USBMonitor()
    devices_dict = monitor.get_available_devices()

    log.info("--- Currently Connected USB Devices ---")
    if not devices_dict:
        log.warning("No USB devices found.")
    else:
        count = 0
        for device_id, device_info in devices_dict.items():
            
            # filter by device class 0103 (Class 01, Subclass 03).
            midi_usb_interface = "DevClass_01&SubClass_03"

            if midi_usb_interface in device_id:
                # The device_id is usually a system path or identifier
                model = device_info.get(ID_MODEL, 'N/A')
                model_id = device_info.get(ID_MODEL_ID, 'N/A')
                vendor_id = device_info.get(ID_VENDOR_ID, 'N/A')
                vendor_name = device_info.get(ID_VENDOR_FROM_DATABASE, 'N/A')
                interfaces = device_info.get(ID_USB_INTERFACES, 'N/A')
                device_name = device_info.get(DEVNAME, 'N/A')
                device_type = device_info.get(DEVTYPE, 'N/A')

                print(f"Device ID: {device_id}")
                print(f"  Model: {model}")
                print(f"  VID: {vendor_id}, PID: {model_id}")
                print(f"  INTERFACES: {interfaces}")
                print(f"  DEVICE NAME: {device_name}")
                print(f"  DEVICE TYPE: {device_type}")
                print(f"  VENDOR NAME: {vendor_name}")
                print("-" * 35)

                # initialize a controller in controller list 
                # by using some attribute to ID which type of device and
                # pass its corresponding controller interface class as arg
                if <id-attribute> in <list of known midi device attribute>:
                    
                    if <id-attribute> == <known midi device attribute>:
                        controllers[count] = MidiInputController(controller=MidiMix(params))
                    if <id-attribute> == <known midi device attribute>:
                        MidiInputController(controller=SMC_Mixer(params))
                    
                    count+=1

            else:
                log.warning("Found USB device without available MIDI or Video interface, skipping")

""" IDs USB devices, prompts user on whether to save device to config file for future use.
The user must still implement the controller class, map params, and initialize it in id_midi_devices """
def config_helper(controllers, params):
    pass

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
    controllers = []
    id_midi_devices(controllers)

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

            # apply effects sequence
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