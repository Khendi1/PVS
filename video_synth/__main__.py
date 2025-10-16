"""
Main module for the video synthesizer application.
This module initializes the video mixer, applies effects, and manages the main loop.

All parameters are managed via the ParamTable class in config.py. 

To create new parameters and their corresponding sliders,
you must first create a param and add it to the ParamTable.
This is often done in class init functions.
Then you must create a trackbar for it manually (for now).
Existing classes expose a create_sliders function which is called in gui.py

Parameter values can be modified via the GUI or linked to MIDI controllers.
See how sample controllers are mapped in midi.py


Author: Kyle Henderson
"""

import argparse
import logging
import cv2
import dearpygui.dearpygui as dpg 
from globals import fx_dict, FX
from gui import Interface
from generators import OscBank
from midi_input import MidiInputController, MidiMix, SMC_Mixer
from fx import *
from patterns3 import Patterns
from param import ParamTable
from mix import Mixer
from gui_elements import ButtonsTable


NUM_OSC = 4 
LOG_LEVEL = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Video Synthesizer initialization arguments')
    parser.add_argument(
        '--osc', 
        type=int, 
        default=NUM_OSC, 
        help='Number of general purpose oscillators')
    parser.add_argument(
        "--log-level",
        default="INFO",  # Default log level
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help="Set the logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG)"
    )
    return parser.parse_args()


def init_effects(params, width, height):

    global fx_dict

    feedback = Feedback(params, width, height)
    color = Color(params)
    pixels = Pixels(params, width, height)
    noise = ImageNoiser(params, NoiseType.NONE)
    shapes = ShapeGenerator(params, width, height)
    patterns = Patterns(params, width, height)
    reflector = Reflector(params, )                    
    sync = Sync(params) 
    warp = Warp(params, width, height)
    glitch = GlitchEffect(params)
    ptz = PTZ(params, width, height)

    # Convenient dictionary of effects to be passed to the apply_effects function
    fx_dict.clear() # Clear any previous state
    fx_dict.update({
        FX.FEEDBACK: feedback,
        FX.COLOR: color,
        FX.PIXELS: pixels,
        FX.NOISE: noise,
        FX.SHAPES: shapes,
        FX.PATTERNS: patterns,
        FX.REFLECTOR: reflector,
        FX.SYNC: sync,
        FX.WARP: warp,
        FX.GLITCH: glitch,
        FX.PTZ: ptz
    })


def apply_effects(frame, frame_count, frame_skip, patterns: Patterns, feedback: Feedback, color: Color, 
                  pixels: Pixels, noise: ImageNoiser, reflector: Reflector, 
                  sync: Sync, warp: Warp, shapes: ShapeGenerator, glitch: GlitchEffect,
                  ptz: PTZ):
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
        frame = patterns.generate_pattern_frame(frame)
        frame = ptz.shift_frame(frame)
        frame = sync.sync(frame)
        frame = reflector.apply_reflection(frame)
        frame = color.polarize_frame_hsv(frame)
        frame = color.modify_hsv(frame)
        frame = color.adjust_brightness_contrast(frame)
        frame = noise.apply_noise(frame)
        frame = color.solarize_image(frame)
        frame = color.posterize(frame)
        frame = pixels.gaussian_blur(frame)
        frame = pixels.sharpen_frame(frame)
        frame = glitch.apply_glitch_effects(frame, frame_count)

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


def apply_feedback(frame, feedback_frame, prev_frame, frame_count, frame_skip, params, toggles, fx_dict):
# relevant section
    if toggles.val("effects_first") == True:         
        feedback_frame = apply_effects(feedback_frame, frame_count, frame_skip, **fx_dict)
        # Blend the current dry frame with the previous wet frame using the alpha param
        feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
    else:
        feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
        feedback_frame = apply_effects(feedback_frame, frame_count, frame_skip, **fx_dict) 

    # Apply feedback effects
    feedback_frame = fx_dict[FX.FEEDBACK].apply_temporal_filter(prev_frame, feedback_frame)
    feedback_frame = fx_dict[FX.FEEDBACK].avg_frame_buffer(feedback_frame)
    feedback_frame = fx_dict[FX.FEEDBACK].nth_frame_feedback(feedback_frame)
    feedback_frame = fx_dict[FX.FEEDBACK].apply_luma_feedback(prev_frame, feedback_frame)
    # prev_frame = fx_dict[FX.FEEDBACK].scale_frame(feedback_frame)
    prev_frame = feedback_frame

    return prev_frame, feedback_frame


def main(num_osc, log_level):
    global fx_dict
    
    print("Initializing video synthesizer...")

    # all user modifiable parameters are stored here
    params = ParamTable()       # params have a min and max value
    toggles = ButtonsTable()    # toggles are binary

    # initialize general purpose oscillators for linking to params
    osc_bank = OscBank(params, num_osc)

    # Initialize mixer video sources and retreive frame
    mixer = Mixer(params)
    frame = mixer.get_frame()  

    # Create a copy of the frame for feedback and get its dimensions
    feedback_frame = frame.copy()
    prev_frame = frame.copy()

    image_height, image_width = frame.shape[:2]

    frame_count = 0

    # Initialize effects classes with image dimensions
    # The mixer and all objects from fx.py are stored here so they may be used by both main and the gui
    init_effects(params=params, width=image_width, height=image_height)

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
    #     print(f'{k}: {v.min}-{v.max}, {v.value}')

    print(f'Enjoy {len(params.keys())} tunable parameters!\n')

    try:
        while True:
            # retreive and mix frames from the selected sources
            frame = mixer.get_frame()
            if mixer.skip1 or frame is None:
                mixer.skip1 = False
                print("Skipping frame due to source read failure...")
                continue

            # update osc values if linked to params
            osc_bank.update()


            prev_frame, feedback_frame = apply_feedback(
                frame, 
                feedback_frame,
                prev_frame, 
                frame_count,
                (params.val("frame_skip") + 1), 
                params, 
                toggles, 
                fx_dict
            )

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', feedback_frame)
            dpg.render_dearpygui_frame()

            frame_count += 1

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Key pressed: {key}")
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Signaling MIDI thread to stop...")
        controller1.thread_stop, controller2.thread_stop = True, True
        # Wait for the MIDI thread to finish, with a timeout
        controller1.thread.join(timeout=5)
        controller2.thread.join(timeout=5)
        if controller1.thread.is_alive() or controller2.thread.is_alive():
            print("MIDI thread did not terminate gracefully. Forcing exit.")
        else:
            print("MIDI thread stopped successfully.")
    finally:
        print("Exiting main program.")
        # Release the capture and destroy all windows
        dpg.destroy_context()
        for cap in mixer.live_caps:
            if cap and cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args.osc, args.log_level)