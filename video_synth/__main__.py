import cv2
import dearpygui.dearpygui as dpg 
from fx import Effects
from config import *
from noiser import ImageNoiser, NoiseType
from gui import Interface
from shapes import ShapeGenerator
from generators import PerlinNoise, Interp, Oscillator
from patterns import Patterns
from keying import Keying
import numpy as np

image_height, image_width = None, None

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
    
    # TODO: move this to generators class, set as a wave shape, equate phase with octaves
    pn = PerlinNoise(1, frequency=1.0, amplitude=1.0, octaves=1, interp=Interp.COSINE)

    # osc_bank = [Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4) for i in range(NUM_OSCILLATORS)]
    for i in range(NUM_OSCILLATORS):
        osc_bank.append(Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4))
    print(f"Oscillator bank initialized with {len(osc_bank)} oscillators.")
    # Initialize effects classes; these contain Params to be modified by the generators
    n = ImageNoiser(NoiseType.NONE)
    s = ShapeGenerator(image_width, image_height)
    e = Effects(image_width, image_height)
    p = Patterns(image_width, image_height)
    # TODO: test this
    k = Keying(image_width, image_height)

    print(f'Enjoy {len(params.keys())} tunable parameters!')
    
    # noise = pn.get(noise)

    # Create control panel after initializing objects that will be used in the GUI
    gui = Interface()
    gui.create_control_window()

    # Create a copy of the feedback frame for temporal filtering
    prev_frame = feedback_frame.copy()

    t = 0

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break

        # update osc values
        osc_vals = [osc.get_next_value() for osc in osc_bank if osc.linked_param is not None]

        # Update noise value
        # noise = c.perlin_noise.get(noise)
        
        # effect ordering leads to unique results
        if toggles.val("effects_first") == True:
            feedback_frame = apply_effects(feedback_frame, image_height, image_width, e, n, s, t, p)
            feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
        else:
            feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
            feedback_frame = apply_effects(feedback_frame, image_height, image_width, e, n, s, t, p) 
        
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

    # Release the capture and destroy all windows
    dpg.destroy_context()
    cap.release()
    cv2.destroyAllWindows()

def apply_effects(frame, height, width, e: Effects, n: ImageNoiser, s: ShapeGenerator, t: float, p: Patterns):

    # TODO: fix bug where shape hue affects the entire frame
    # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)
    
    # TODO: use frame skip slider to control frame skip
    # apply effects to every Nth frame
    if True: # if t % frame_skip == 0:
        frame = e.shift_frame(frame)
        frame = e.modify_hsv(frame)
        frame = e.adjust_brightness_contrast(frame)
        frame = e.glitch_image(frame) 
        frame = e.gaussian_blur(frame)

        # TODO: test this
        # frame = e.apply_perlin_noise

        # TODO: test this
        # warp_frame = e.warp_frame(frame)

        # TODO: test this
        # fromw = e.sync_wobble

        # TODO: test this
        # frame = e.generate_pattern

        if n.noise_type != NoiseType.NONE:
            frame = e.polarize_frame_hsv(frame)

        if enable_polar_transform == True:
            frame = e.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))

        # TODO: test this
        # TODO: test ordering
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame = e.lissajous_pattern(frame, t)

    return frame

if __name__ == "__main__":
    main()