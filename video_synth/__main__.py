"""
Main module for the video synthesizer application.
This module initializes video sources, applies effects, and manages the main loop.
All components are modular and can be extended or modified independently.
All parameters are managed via the ParamTable class in config.py. 
Parameter values can be modified via the GUI or linked to MIDI controllers.
Soureces include webcam, video files, capture card, metaballs generator, and plasma generator.
"""

import cv2
import dearpygui.dearpygui as dpg 
from config import *
from gui import Interface
from generators import Oscillator
from midi_input import *
from animations import *
from mix import *
from fx import *
from shapes import ShapeGenerator
from patterns3 import *

image_height, image_width = None, None

def apply_effects(frame, frame_count, patterns: Patterns, basic: Effects, color: Color, 
                  pixels: Pixels, noise: ImageNoiser, reflector: Reflector, 
                  sync: Sync, warp: Warp, shapes: ShapeGenerator):
    """ 
    Applies a sequence of visual effects to the input frame based on current parameters.
    Each effect is modular and can be enabled/disabled via the GUI.
    The order of effects can be adjusted to achieve different visual styles.
    
    Returns the modified frame.
    """
    
    global image_height, image_width

    # TODO: implement effect sequencer
    # TODO: use frame skip slider to control frame skip
    if frame_count % (params.val("frame_skip") + 1) == 0: 
        frame = patterns.generate_pattern_frame(frame)
        frame = basic.shift_frame(frame)
        frame = sync.sync(frame)
        frame = reflector.apply_reflection(frame)
        frame = color.modify_hsv(frame)
        frame = color.adjust_brightness_contrast(frame)
        # frame = pixels.glitch_image(frame)
        frame = noise.apply_noise(frame)
        frame = color.polarize_frame_hsv(frame)
        frame = color.solarize_image(frame)
        frame = color.posterize(frame)
        frame = pixels.gaussian_blur(frame)
        frame = pixels.sharpen_frame(frame)

        #TODO: implement glitch effect class functions
        # frame = glitch.apply_evolving_pixel_shift(frame, ...)
        # frame = glitch.apply_gradual_color_split(frame, ...)
        # frame = glitch.apply_morphing_block_corruption(frame, ...)
        # frame = glitch.apply_horizontal_scroll_freeze_glitch(frame, ...)
        # frame = glitch.apply_vertical_jitter_glitch(frame, ...)
        # frame = glitch.scanline_distortion(frame, ...)

        # TODO: test these effects, test ordering
        # frame = color limit_hues_kmeans(frame)
        # image_height, image_width = frame.shape[:2]
        # frame = fx.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = fx.apply_perlin_noise
        # warp_frame = fx.warp_frame(frame)

        # BUG: does lissajous need to be on black background to work properly?
        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = fx.lissajous_pattern(frame, t)

        # TODO: fix bug where shape hue affects the entire frame hue
        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

    return frame


def main():
    global image_height, image_width, cap1, cap2
    print("Initializing video synthesizer...")

    # Initialize mixer video sources with their default settings
    mixer = Mixer()

    # Retrieve an initial frame to determine image dimensions; 
    # required for initializing effects
    frame = mixer.mix_sources()  
    frame_count = 0

    # Create a copy of the frame for feedback and get its dimensions
    feedback_frame = frame.copy()
    prev_frame = frame.copy()
    image_height, image_width = frame.shape[:2]

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Modified Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize the general purpose oscillator bank
    for i in range(NUM_OSCILLATORS):
        osc_bank.append(Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4))
    print(f"Oscillator bank initialized with {len(osc_bank)} oscillators.")
    
    # Initialize effects classes; these contain Params to be modified by the generators
    basic = Effects(image_width, image_height)
    color = Color()
    pixels = Pixels(image_width, image_height)
    noise = ImageNoiser(NoiseType.NONE)
    shapes = ShapeGenerator(image_width, image_height)
    patterns = Patterns(image_width, image_height)
    reflector = Reflector()                    
    sync = Sync() 
    warp = Warp(image_width, image_height)

    # Convenient dictionary of effects to be passed to the apply_effects function
    fx = {
        "basic": basic,
        "color": color,
        "pixels": pixels,
        "noise": noise,
        "shapes": shapes,
        "patterns": patterns,
        "reflector": reflector,
        "sync": sync,
        "warp": warp
    }

    # Initialize the midi input controller before creating the GUI
    # TODO: This assumes both controllers are always connected in a specific order; improve this
    # BUG: fix initialization of MIDI controllers in midi_input.py
    # test_ports()
    controller1 = MidiInputController(controller=MidiMix())
    controller2 = MidiInputController(controller=SMC_Mixer())

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface()
    gui.create_control_window(mixer=mixer)

    print(f'Enjoy {len(params.keys())} tunable parameters!\n')

    try:
        while True:
            # retreive and mix frames from the selected sources
            frame = mixer.mix_sources()
            if mixer.skip1 or frame is None:
                mixer.skip1 = False
                print("Skipping frame due to source read failure...")
                continue

            # update osc values
            osc_vals = [osc.get_next_value() for osc in osc_bank if osc.linked_param is not None]
            
            # print("\nframe dims:")
            # check_frame_dimensions(frame)
            # print("\nfeedback dims:")
            # check_frame_dimensions(feedback_frame)
            # frame, feedback_frame = resize_to_match(frame, feedback_frame)

            # relevant section
            if toggles.val("effects_first") == True:         
                feedback_frame = apply_effects(feedback_frame, frame_count, **fx)
                # Blend the current dry frame with the previous wet frame using the alpha param
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
            else:
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
                feedback_frame = apply_effects(feedback_frame, frame_count, **fx) 

            # Apply temporal filtering and frame buffer averaging to the resulting feedback frame
            feedback_frame = basic.apply_temporal_filter(prev_frame, feedback_frame)
            feedback_frame = basic.avg_frame_buffer(feedback_frame)
            prev_frame = feedback_frame.copy()

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
        controller1.thread_stop = True
        controller2.thread_stop = True
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
    main()