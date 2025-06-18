import cv2
from fx import Effects
import config as c
from config import NUM_OSCILLATORS, params, FPS
from gui import Interface
import dearpygui.dearpygui as dpg 
from shapes import ShapeGenerator
from generators import PerlinNoise, Interp, Oscillator

def main():
    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Create an initial empty frame
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame")
    feedback_frame = frame.copy()
    c.image_height, c.image_width = frame.shape[:2]

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    
    # Initialize oscillators
    for i in range(NUM_OSCILLATORS):
        osc = Oscillator(name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4)
        c.osc_bank.append(osc)


    gui = Interface()
    s = ShapeGenerator(c.image_width, c.image_height)
    pn = PerlinNoise(1, frequency=1.0, amplitude=1.0, octaves=1, interp=Interp.COSINE)
    e = Effects(c.image_width, c.image_height)

    print(f'Enjoy {len(params.all().keys())} tunable parameters!')
    # noise = pn.get(noise)

    gui.create_control_window()

    prev_frame = feedback_frame.copy()

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break

        # update osc values
        osc_vals = [osc.get_next_value() for osc in c.osc_bank if osc.linked_param is not None]

        # Update noise value
        # noise = c.perlin_noise.get(noise)

        # Apply transformations to the frame for feedback effect
        frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)
        feedback_frame = cv2.addWeighted(frame, 1 - params.val("alpha"), feedback_frame, params.val("alpha"), 0)
        feedback_frame = e.glitch_image(feedback_frame) 
        feedback_frame = e.gaussian_blur(feedback_frame)
        feedback_frame = e.shift_frame(feedback_frame)
        feedback_frame = e.adjust_brightness_contrast(feedback_frame)
        if c.enable_polar_transform == True:
            feedback_frame = e.polar_transform(feedback_frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # feedback_frame = noisy("gauss", feedback_frame)        

        # Split image HSV channels for modifications and convert back to BGR color space.   
        hsv_image = e.modify_hsv(feedback_frame)
        feedback_frame = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Apply temporal filtering to the resulting feedback frame
        feedback_frame = e.apply_temporal_filter(prev_frame, feedback_frame)
        prev_frame = feedback_frame.copy()

        # Display the resulting frame and control panel
        cv2.imshow('Modified Frame', feedback_frame)
        dpg.render_dearpygui_frame()

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Key pressed: {key}")
            break

    # Release the capture and destroy all windows
    dpg.destroy_context()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()