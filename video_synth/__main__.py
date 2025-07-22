import cv2
import dearpygui.dearpygui as dpg 
from fx import Effects
from config import *
from noiser import ImageNoiser, NoiseType
from gui import Interface
from shapes import ShapeGenerator
from generators import PerlinNoise, Interp, Oscillator
from patterns3 import Patterns
from keying import Keying
import numpy as np
import time
from midi_input import *
from reflactor import Reflector
from metaballs import LavaLampSynth
from plasma import *

image_height, image_width = None, None

def apply_effects(frame, e: Effects, n: ImageNoiser, s: ShapeGenerator, t: float, p: Patterns, r: Reflector, l: LavaLampSynth):

    # TODO: use frame skip slider to control frame skip
    if True: 
        frame = l.do_metaballs(frame)
        frame = p.generate_pattern_frame(frame)
        frame = e.shift_frame(frame)
        frame = r.apply_reflection(frame) 
        frame = e.sync(frame)
        frame = e.modify_hsv(frame)
        frame = e.adjust_brightness_contrast(frame)
        frame = e.sharpen_frame(frame)
        frame = e.glitch_image(frame) 
        frame = n.apply_noise(frame)
        frame = e.polarize_frame_hsv(frame)
        frame = e.gaussian_blur(frame)
        frame = e.solarize_image(frame)
        frame = e.posterize(frame)

        # TODO: test this,test ordering
        # image_height, image_width = frame.shape[:2]
        # frame = generate_plasma_effect(image_width, image_height)
        # frame = e.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = e.apply_perlin_noise
        # warp_frame = e.warp_frame(frame)
        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = e.lissajous_pattern(frame, t)
        # TODO: fix bug where shape hue affects the entire frame hue
        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

    return frame

def main():

    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Note: Setting FPS may not work on all cameras, it depends on the camera's
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Create an initial empty frame
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame")
    
    # Create a copy of the frame for feedback
    feedback_frame = frame.copy()
    image_height, image_width = frame.shape[:2]

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)

    # Initialize the general purpose oscillator bank
    for i in range(NUM_OSCILLATORS):
        osc_bank.append(Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4))
    print(f"Oscillator bank initialized with {len(osc_bank)} oscillators.")

    # TODO: move this to generators class, set as a wave shape, equate phase with octaves
    pn = PerlinNoise(1, frequency=1.0, amplitude=1.0, octaves=1, interp=Interp.COSINE)
    
    # Initialize effects classes; these contain Params to be modified by the generators
    n = ImageNoiser(NoiseType.NONE)
    s = ShapeGenerator(image_width, image_height)
    e = Effects(image_width, image_height)
    p = Patterns(image_width, image_height)
    k = Keying(image_width, image_height)     # TODO: test this
    r = Reflector()  # Initialize the reflector
    l = LavaLampSynth(image_width, image_height)  # Initialize the Lava Lamp synth

    # Initialize the midi input controller before creating the GUI
    # This will allow the controller to be used in the GUI and to respond to MIDI input
    controller = MidiInputController(controller=MidiMix())
    controller2 = MidiInputController(controller=SMC_Mixer())

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface()
    gui.create_control_window()

    # Create a copy of the feedback frame for temporal filtering
    prev_frame = feedback_frame.copy()

    # TODO: [re]move this
    # noise = pn.get(noise)

    t = 0

    print(f'Enjoy {len(params.keys())} tunable parameters!\n')

    try:
        # Keep the main thread alive indefinitely so the MIDI input thread can run.
        # It sleeps periodically to prevent busy-waiting.

        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error")
                break

            # update osc values
            osc_vals = [osc.get_next_value() for osc in osc_bank if osc.linked_param is not None]
            # print(f"Oscillator values: {osc_vals}")
            
            prev_frame1 = feedback_frame.copy()

            # effect ordering leads to unique results
            if toggles.val("effects_first") == True:
                feedback_frame = apply_effects(feedback_frame, e, n, s, t, p, r, l)
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
            else:
                feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
                feedback_frame = apply_effects(feedback_frame, e, n, s, t, p, r, l) 

            frame = e.limit_hues_kmeans(frame)

            # Apply temporal filtering to the resulting feedback frame
            feedback_frame = e.apply_temporal_filter(prev_frame, feedback_frame)
            prev_frame = feedback_frame.copy()

            # Display the resulting frame and control panel
            cv2.imshow('Modified Frame', feedback_frame)
            dpg.render_dearpygui_frame()

            t += 0.1

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Key pressed: {key}")
                break

            # time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Signaling MIDI thread to stop...")
        controller.thread_stop = True
        controller2.thread_stop = True
        # Wait for the MIDI thread to finish, with a timeout
        controller.thread.join(timeout=5)
        controller2.thread.join(timeout=5)
        if controller.thread.is_alive() or controller2.thread.is_alive():
            print("MIDI thread did not terminate gracefully. Forcing exit.")
        else:
            print("MIDI thread stopped successfully.")
    finally:
        print("Exiting main program.")
        # Release the capture and destroy all windows
        dpg.destroy_context()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()