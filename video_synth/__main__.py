"""
Main module for the video synthesizer application.
This module initializes the video mixer, applies effects, and manages the main loop.

All parameters are managed via the ParamTable class in config.py. 

To create new parameters and their corresponding sliders,
you must first create a param and add it to the ParamTable.
This is often done in class init functions.
Then you must create a trackbar for it manually (for now).

Existing classes expose a create_gui_panel function which are called in gui.py

Parameter values can be modified via the GUI or linked to MIDI controllers.
See how sample controllers are mapped in midi.py

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
from effects import *
from patterns3 import Patterns
from param import ParamTable
from mix import Mixer
from gui_elements import ButtonsTable


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


def apply_effects(frame, frame_count, frame_skip, effects: EffectManager):
    """ 
    Applies a sequence of visual effects to the input frame based on current parameters.
    Each effect is modular and can be enabled/disabled via the GUI.
    The order of effects can be adjusted to achieve different visual styles.
    
    Returns the modified frame.
    """
    
    global image_height, image_width

    # TODO: implement effect sequencer
    # TODO: use frame skip slider to control frame skip
    if frame_count % frame_skip == 0: 
        frame = effects.patterns.generate_pattern_frame(frame)
        frame = effects.ptz.shift_frame(frame)
        frame = effects.sync.sync(frame)
        frame = effects.reflector.apply_reflection(frame)
        frame = effects.color.polarize_frame_hsv(frame)
        frame = effects.color.modify_hsv(frame)
        frame = effects.color.adjust_brightness_contrast(frame)
        frame = effects.noise.apply_noise(frame)
        frame = effects.color.solarize_image(frame)
        frame = effects.color.posterize(frame)
        frame = effects.pixels.gaussian_blur(frame)
        frame = effects.pixels.sharpen_frame(frame)
        frame = effects.glitch.apply_glitch_effects(frame, frame_count)

        # TODO: test these effects, test ordering
        # frame = color limit_hues_kmeans(frame)
        # frame = fx.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = fx.apply_perlin_noise
        # warp_frame = fx.warp_frame(frame)

        # BUG: does lissajous need to be on black background to work properly?
        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = fx.lissajous_pattern(frame, t)

        # TODO: fix bug where shape hue affects the entire frame hue
        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

    return frame


def apply_feedback(dry_frame, wet_frame, prev_frame, frame_count, params, toggles, effects):

    if toggles.val("effects_first") == True:         
        wet_frame = apply_effects(wet_frame, frame_count, params.val("frame_skip")-1, effects)
        # Blend the current dry frame with the previous wet frame using the alpha param
        wet_frame = cv2.addWeighted(dry_frame, 1 - params.val("alpha"), wet_frame, params.val("alpha"), 0)
    else:
        wet_frame = cv2.addWeighted(dry_frame, 1 - params.val("alpha"), wet_frame, params.val("alpha"), 0)
        wet_frame = apply_effects(wet_frame, frame_count, params.val("frame_skip")-1, effects) 

    # Apply feedback effects
    wet_frame = effects.feedback.apply_temporal_filter(prev_frame, wet_frame)
    wet_frame = effects.feedback.avg_frame_buffer(wet_frame)
    wet_frame = effects.feedback.nth_frame_feedback(wet_frame)
    wet_frame = effects.feedback.apply_luma_feedback(prev_frame, wet_frame)
    # prev_frame = effects.feedback.scale_frame(wet_frame)
    prev_frame = wet_frame

    return prev_frame, wet_frame


def main(num_osc, log_level):
    global fx_dict
    
    log.info("Initializing video synthesizer...")

    # all user modifiable parameters are stored here
    params = ParamTable()       # params have a min and max value
    toggles = ButtonsTable()    # toggles are binary

    # initialize general purpose oscillators for linking to params
    osc_bank = OscBank(params, num_osc)

    # Initialize mixer video sources and retreive frame
    mixer = Mixer(params)
    dry_frame = mixer.get_frame()  

    # Create a copy of the frame for feedback and get its dimensions
    wet_frame = dry_frame.copy()
    prev_frame = dry_frame.copy()
    image_height, image_width = dry_frame.shape[:2]
    frame_count = 0

    # Initialize effects classes with image dimensions
    effects.init(params, image_width, image_height)

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Modified Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # TODO: This assumes both controllers are always connected in a specific order; improve this
    # test_ports()

    # Initialize the midi input controller before creating the GUI
    controller1 = MidiInputController(controller=MidiMix(params))
    controller2 = MidiInputController(controller=SMC_Mixer(params))

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface(params, osc_bank, toggles)
    gui.create_control_window(params, mixer=mixer)

    # for k, v in params.params.items():
    #     log.info(f'{k}: {v.min}-{v.max}, {v.value}')

    log.info(f'Enjoy {len(params.keys())} tunable parameters!')

    try:
        while True:
            # retreive and mix frames from the selected sources
            dry_frame = mixer.get_frame()
            if mixer.skip1 or dry_frame is None:
                mixer.skip1 = False
                log.warning("Skipping frame due to source read failure")
                continue

            # update osc values if linked to params
            osc_bank.update()

            prev_frame, wet_frame = apply_feedback(
                dry_frame, 
                wet_frame,
                prev_frame, 
                frame_count,
                params, 
                toggles, 
                effects
            )

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', wet_frame)
            dpg.render_dearpygui_frame()

            frame_count += 1

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info(f"Received quit command, begining shutdown")
                break

    except KeyboardInterrupt:
        log.exception("Quit command detected, signaling MIDI thread to stop...")
        
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
        
        log.info("P")


if __name__ == "__main__":
    args = parse_args()
    main(args.osc, args.log_level)