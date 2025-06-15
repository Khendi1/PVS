import cv2
from fx import Effects, HSV
import config as p
from config import NUM_OSCILLATORS, params
from gui import Interface
import dearpygui.dearpygui as dpg 
from shapes import ShapeGenerator
from generators import PerlinNoise, Interp, Oscillator

CURRENT = 0
PREV = 1

fps = 20 # Desired frame rate

def main():
    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FPS, fps)

    # Create an initial empty frame
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame")
    feedback_frame = frame.copy()
    p.image_height, p.image_width = frame.shape[:2]

    params["x_shift"].set_min_max(-p.image_width, p.image_width)
    params["y_shift"].set_min_max(-p.image_height, p.image_height)
    params["polar_x"].set_min_max(-p.image_width, p.image_width)
    params["polar_y"].set_min_max(-p.image_height, p.image_height)

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    
    for i in range(NUM_OSCILLATORS):
        # Create a new oscillator with frequency 0.5 Hz and amplitude 1.0
        osc = Oscillator(name=f"osc{i}", frequency=1.0, amplitude=1.0, phase=0.0, shape=i)
        p.osc_bank.append(osc)

    s = ShapeGenerator(p.image_width, p.image_height)
    gui = Interface()
    gui.create_control_window()

    e = Effects()

    prev_frame = feedback_frame.copy()

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # update osc values
        osc_vals = [osc.get_next_value() for osc in p.osc_bank if osc.linked_param is not None]
        # print(p.osc_vals)

        # Update noise value
        # noise = p.perlin_noise.get(noise)
        # print(noise)

        # Apply transformations to the frame for feedback effect
        frame = s.draw_shapes_on_frame(frame, p.image_width, p.image_height)
        feedback_frame = cv2.addWeighted(frame, 1 - params["alpha"].value, feedback_frame, params["alpha"].value, 0)
        feedback_frame = e.glitch_image(feedback_frame, params["num_glitches"].value, params["glitch_size"].value) 
        feedback_frame = e.gaussian_blur(feedback_frame, params["blur_kernel_size"].value)
        feedback_frame = e.shift_frame(feedback_frame, params["x_shift"].value, params["y_shift"].value, params["r_shift"].value, params["zoom"].value)
        feedback_frame = e.adjust_brightness_contrast(feedback_frame, params["contrast"].value, params["brightness"].value)
        if p.enable_polar_transform == True:
            feedback_frame = e.polar_transform(feedback_frame, params["polar_x"].value, params["polar_y"].value, params["polar_radius"].value)
        # feedback_frame = noisy("gauss", feedback_frame)        

        # Split image HSV channels for modifications and convert back to BGR color space.   
        hsv_image = e.modify_hsv(feedback_frame, params["hue_shift"].value, params["sat_shift"].value, 
                                params["val_shift"].value, p.val_threshold, p.val_hue_shift)
        feedback_frame = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        feedback_frame = e.apply_temporal_filter(prev_frame, feedback_frame, params["temporal_filter"].value)
        prev_frame = feedback_frame.copy()

        # Display the resulting frame and control panel
        cv2.imshow('Modified Frame', feedback_frame)
        dpg.render_dearpygui_frame()

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Key pressed: {key}")
            break
        else:
            s.keyboard_controls(key)

    # Release the capture and destroy all windows
    dpg.destroy_context()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()