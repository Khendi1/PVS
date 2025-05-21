import os
import time
import sys
import math
from math import floor
from datetime import datetime
import yaml
from perlin import PerlinNoise, Interp
import cv2
from effects import Effects, HSV
import vars
from gui import Interface
import dearpygui.dearpygui as dpg 

CURRENT = 0
PREV = 1

def main():

    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Create an initial empty frame
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame")
    feedback_frame = frame.copy()
    vars.image_height, vars.image_width = frame.shape[:2]

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    
    gui = Interface()
    gui.create_control_window()

    e = Effects()
    perlin_noise = PerlinNoise(vars.noise)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # move to call back functions
        # perlin_noise.amplitude = perlin_amplitude
        # perlin_noise.frequency = perlin_frequency
        # perlin_noise.octaves = perlin_octaves
        # perlin_noise.interp = get_interp_btn_idx

        # noise = perlin_noise.get(noise)
        # print(noise)
        # x_shift = int(noise)

        # Apply transformations to the frame for feedback effect
        feedback_frame = cv2.addWeighted(frame, 1 - vars.alpha, feedback_frame, vars.alpha, 0)
        feedback_frame = e.glitch_image(feedback_frame, vars.num_glitches, vars.glitch_size) 

        if vars.blur_kernel_size >= 1:
            vars.blur_kernel_size = max(1, vars.blur_kernel_size | 1)
            feedback_frame = cv2.GaussianBlur(feedback_frame, (vars.blur_kernel_size, vars.blur_kernel_size), 0) 

        feedback_frame = e.shift_frame(feedback_frame, vars.x_shift, vars.y_shift, vars.r_shift)
        feedback_frame = e.polar_transform(feedback_frame)
        # feedback_frame = noisy("gauss", feedback_frame)        

        # Split image HSV channels for modifications (hsv shifts, shifts hue by x where val > y)
        # then merge the modified channels and convert back to BGR color space.   
        hsv_image = e.modify_hsv(feedback_frame, vars.hue_shift, vars.sat_shift, 
                                vars.val_shift, vars.val_threshold, vars.val_hue_shift)
        feedback_frame = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Display the resulting frame next to the original frame
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Modified Frame', feedback_frame)

        # Display the control panel
        dpg.render_dearpygui_frame()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    dpg.destroy_context()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()