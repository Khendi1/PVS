"""
Main module for the video synthesizer application.
This module initializes video sources, applies effects, and manages the main loop.
All components are modular and can be extended or modified independently.
All parameters are managed via the ParamTable class in config.py. 
Parameter values can be modified via the GUI or linked to MIDI controllers.
Soureces include webcam, video files, capture card 1 (hdmi), capture card 2 (composite), metaballs generator, and plasma generator.
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

def apply_effects(frame, patterns: Patterns, basic: Effects, color: Color, 
                  pixels: Pixels, noise: ImageNoiser, reflector: Reflector, 
                  sync: Sync, warp: Warp, shapes: ShapeGenerator):
    global image_height, image_width

    # TODO: use frame skip slider to control frame skip
    if True: 
        frame = patterns.generate_pattern_frame(frame)
        frame = basic.shift_frame(frame)
        frame = sync.sync(frame)
        frame = reflector.apply_reflection(frame)
        frame = color.modify_hsv(frame)
        frame = color.adjust_brightness_contrast(frame)
        frame = pixels.glitch_image(frame)
        frame = noise.apply_noise(frame)
        frame = color.polarize_frame_hsv(frame)
        frame = color.solarize_image(frame)
        frame = color.posterize(frame)
        frame = pixels.gaussian_blur(frame)
        frame = pixels.sharpen_frame(frame)

        # TODO: test this,test ordering
        # image_height, image_width = frame.shape[:2]
        # frame = generate_plasma_effect(image_width, image_height)
        # frame = fx.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = fx.apply_perlin_noise
        # warp_frame = fx.warp_frame(frame)
        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = fx.lissajous_pattern(frame, t)
        # TODO: fix bug where shape hue affects the entire frame hue
        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

    return frame

def failback_camera():
    # TODO: implement a fallback camera source if the selected source fails
    pass

def debug_midi_controller_connections():
    test_ports()

def main():
    global image_height, image_width, skip1, cap1, cap2
    print("Initializing video capture...")

    mixer = Mixer()

    # Initialize mixer video sources with their default settings
    mixer.start_video(params.get("source1").val, 1) # TODO: move to init function
    mixer.start_video(params.get("source2").val, 2)
    frame = mixer.mix_sources(mode="blend")

    # Create a copy of the frame for feedback and get its dimensions
    feedback_frame = frame.copy()
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
    # key = Keying(image_width, image_height)     # TODO: test this
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
        # "key": key,    # TODO: test this
        "reflector": reflector,
        "sync": sync,
        "warp": warp
    }


    # Initialize the midi input controller before creating the GUI
    # TODO: This assumes both controllers are always connected in a specific order; improve this
    # debug_midi_controller_connections()
    controller1 = MidiInputController(controller=MidiMix())
    controller2 = MidiInputController(controller=SMC_Mixer())

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface()
    gui.create_control_window()

    # Create a copy of the feedback frame for temporal filtering
    prev_frame = feedback_frame.copy()

    print(f'Enjoy {len(params.keys())} tunable parameters!\n')

    try:
        while True:
            # retreive and mix frames from the selected sources
            frame = mix_sources()
            if skip1 or frame is None:
                skip1 = False
                continue

            # TODO: test this placement
            feedback_frame = frame.copy()

            # update osc values
            osc_vals = [osc.get_next_value() for osc in osc_bank if osc.linked_param is not None]
            
            # relevant section
            if toggles.val("effects_first") == True:
                feedback_frame = apply_effects(feedback_frame, **fx)
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
            else:
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
                feedback_frame = apply_effects(feedback_frame, **fx) 

            # Apply temporal filtering and frame buffer averaging to the resulting feedback frame
            feedback_frame = basic.apply_temporal_filter(prev_frame, feedback_frame)
            feedback_frame = basic.avg_frame_buffer(feedback_frame)
            prev_frame = feedback_frame.copy()

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', feedback_frame)
            dpg.render_dearpygui_frame()

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
        for cap in live_caps:
            if cap and cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()