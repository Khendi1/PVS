import cv2
import numpy as np
import os
import random
import time
import sys
import math
from math import floor
import dearpygui.dearpygui as dpg
from datetime import datetime
import yaml

H = 0
S = 1
V = 2

class SliderRow:
    def __init__(self, label, tag, default_value, min_value, max_value, callback, type, button_callback):
        self.label = label
        self.tag = tag
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.callback = callback
        self.button_callback = button_callback
        self.slider = None
        self.button = None
        self.value = default_value
        self.type = type
        self.create()

    def create(self):
        with dpg.group(horizontal=True):
            # self.button = dpg.add_button(label="Reset", callback=lambda x: self.reset, width=50)
            self.button = dpg.add_button(label="Reset", callback=self.button_callback, width=50, tag=self.tag + "_reset", user_data=self.tag)
            if self.type == 'float':
                self.slider = dpg.add_slider_float(label=self.label, tag=self.tag, default_value=self.default_value, min_value=self.min_value, max_value=self.max_value, callback=self.callback, width=-100)
            else:
                self.slider = dpg.add_slider_int(label=self.label, tag=self.tag, default_value=self.default_value, min_value=self.min_value, max_value=self.max_value, callback=self.callback, width=-100)

    def update(self, value):
        self.value = value
        dpg.set_value(self.label, value)

    def reset(self):
        print(f"Resetting {self.label} to {self.min_value}")
        self.update(self.min_value)
    
    def randomize(self):
        if self.max_value == 1:
            self.value = random.uniform(self.min_value, self.max_value)
        else:
            self.value = random.randint(self.min_value, self.max_value)

class Effects:
    def __init__(self):
            self.hue_shift = 0

    def shift_hue(self, hue, hue_shift):
        """
        Shifts the hue of an image by a specified amount, wrapping aroung in necessary.
        """
        return (hue + hue_shift) % 180

    def shift_sat(self, sat, sat_shift):
        """
        Shifts the saturation of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(sat + sat_shift, 0, 255)

    def shift_val(self, val, val_shift):
        """
        Shifts the value of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(val + val_shift, 0, 255)

    def shift_hsv(self, hsv, hue_shift, sat_shift, val_shift):
        """
        Shifts the hue, saturation, and value of an image by specified amounts.
        """
        # Convert the image to HSV color space.
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        new_h = self.shift_hue(hsv[H], hue_shift)
        new_s = self.shift_sat(hsv[S], sat_shift)
        new_v = self.shift_val(hsv[V], val_shift)

        return [new_h, new_s, new_v]

    def val_threshold_hue_shift(hsv, val_threshold, hue_shift):
        """
        Shifts the hue of pixels in an image that have a saturation value
        greater than the given threshold.

        Args:
            image (numpy.ndarray): The input BGR image.
            val_threshold (int): The minimum saturation value for hue shifting.
            hue_shift (int): The amount to shift the hue.

        Returns:
            numpy.ndarray: The output image with shifted hues.
        """

        # Create a mask for pixels with saturation above the threshold.
        mask = hsv[S] > val_threshold

        # Shift the hue values for the masked pixels.  We use modulo 180
        # to ensure the hue values stay within the valid range of 0-179.
        hsv[H][mask] = (hsv[H][mask] + hue_shift) % 180

        return hsv

    def glitch_image(self, image, num_glitches=50, glitch_size=10):
        height, width, _ = image.shape

        for _ in range(num_glitches):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            x_glitch_size = random.randint(1, glitch_size)
            y_glitch_size = random.randint(1, glitch_size)

            # Ensure the glitch area does not exceed image boundaries
            x_end = min(x + x_glitch_size, width)
            y_end = min(y + y_glitch_size, height)

            # Extract a random rectangle
            glitch_area = image[y:y_end, x:x_end].copy()

            # Shuffle the pixels
            glitch_area = glitch_area.reshape((-1, 3))
            np.random.shuffle(glitch_area)
            glitch_area = glitch_area.reshape((y_end - y, x_end - x, 3))

            # Apply the glitch
            image[y:y_end, x:x_end] = glitch_area
        return image

    def noisy(noise_typ,image):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy

    def warp_frame(feedback_frame, x_speed, y_speed, x_size, y_size):
        frame_height, frame_width = feedback_frame.shape[:2]
        feedback_frame = cv2.resize(feedback_frame, (frame_width, frame_height))

        # Create meshgrid for warping effect
        x_indices, y_indices = np.meshgrid(np.arange(frame_width), np.arange(frame_height))

        # Calculate warped indices using sine function
        time = cv2.getTickCount() / cv2.getTickFrequency()
        x_warp = x_indices + x_size * np.sin(y_indices / 20.0 + time * x_speed)
        y_warp = y_indices + y_size * np.sin(x_indices / 20.0 + time * y_speed)

        # Bound indices within valid range
        x_warp = np.clip(x_warp, 0, frame_width - 1).astype(np.float32)
        y_warp = np.clip(y_warp, 0, frame_height - 1).astype(np.float32)

        # Remap frame using warped indices
        feedback_frame = cv2.remap(feedback_frame, x_warp, y_warp, interpolation=cv2.INTER_LINEAR)  


    def val_threshold_hue_shift(self, hsv, val_threshold, hue_shift):
        """
        Shifts the hue of pixels in an image that have a saturation value
        greater than the given threshold.

        Args:
            image (numpy.ndarray): The input BGR image.
            val_threshold (int): The minimum saturation value for hue shifting.
            hue_shift (int): The amount to shift the hue.

        Returns:
            numpy.ndarray: The output image with shifted hues.
        """

        # Create a mask for pixels with saturation above the threshold.
        mask = hsv[S] > val_threshold

        # Shift the hue values for the masked pixels.  We use modulo 180
        # to ensure the hue values stay within the valid range of 0-179.
        hsv[H][mask] = (hsv[H][mask] + hue_shift) % 180

        return hsv

    def shift_frame(self, frame, shift_x, shift_y):
        """
        Shifts all pixels in an OpenCV frame by the specified x and y amounts,
        wrapping pixels that go beyond the frame boundaries.

        Args:
            frame: The input OpenCV frame (a numpy array).
            shift_x: The number of pixels to shift in the x-direction.
                    Positive values shift to the right, negative to the left.
            shift_y: The number of pixels to shift in the y-direction.
                    Positive values shift downwards, negative upwards.

        Returns:
            A new numpy array representing the shifted frame.
        """
        height, width = frame.shape[:2]  # Get height and width

        # Create a new array with the same shape and data type as the original frame
        shifted_frame = np.zeros_like(frame)

        # Create the mapping arrays for the indices.
        x_map = (np.arange(width) - shift_x) % width
        y_map = (np.arange(height) - shift_y) % height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        return shifted_frame

def get_slider_by_tag(tag):
    for s in sliders:
        if s.tag == tag:
            return s
        
def get_value_by_tag(tag):
    for s in sliders:
        if s.tag == tag:
            return s.value

def reset_button_callback(sender, app_data, user_data):
    s = get_slider_by_tag(user_data)
    s.value = s.min_value
    print(f"{user_data}: {s.value}")
    # s.reset()    
    dpg.set_value(user_data, s.min_value)

def slider_callback(sender, app_data):
    print(f"{sender}: {app_data}")
    # if sender == "x_shift" or sender == "y_shift":
    #     get_slider_by_tag(sender).value = app_data
    #     map_value(app_data, 0, 100, 0, image_width)  # Map app_data from [0, 100] to [0, image_width]
    #     print(f'Mapped x shift from {app_data} to {x_shift}')
    # else:
    #     get_slider_by_tag(sender).value = app_data
    get_slider_by_tag(sender).value = app_data

def on_x_shift_mode_button_press():
    global x_shift_mode
    x_shift_mode = (x_shift_mode + 1) % 6  # Cycle through modes 0-5
    print(f"X Shift Mode: {x_shift_mode}") 

def map_value(value, from_min, from_max, to_min, to_max):
  """
  Maps a value from one range to another.

  Args:
    value: The value to map.
    from_min: The minimum value of the original range.
    from_max: The maximum value of the original range.
    to_min: The minimum value of the target range.
    to_max: The maximum value of the target range.

  Returns:
    The mapped value.
  """
  # Calculate the proportion of the value within the original range
  proportion = (value - from_min) / (from_max - from_min)

  # Map the proportion to the target range
  mapped_value = to_min + proportion * (to_max - to_min)

  return floor(mapped_value)

def on_y_shift_mode_button_press():
    global y_shift_mode
    y_shift_mode = (y_shift_mode + 1) % 6

def on_reset_button_click():
    print("Reset button clicked!")
    reset_values()

def on_save_button_click():
    date_time_str = datetime.now().strftime("%m-%d-%Y %H-%M")
    print(f"Saving values at {date_time_str}")
    
    # Prepare the data to save
    data = {
        "timestamp": date_time_str,
        "hue_shift": hue_shift,
        "sat_shift": sat_shift,
        "val_shift": val_shift,
        "alpha": alpha,
        "num_glitches": num_glitches,
        "glitch_size": glitch_size,
        "val_threshold": val_threshold,
        "val_hue_shift": val_hue_shift,
        "blur_kernel_size": blur_kernel_size,
        "x_speed": x_speed,
        "y_speed": y_speed,
        "x_size": x_size,
        "y_size": y_size
    }
    
    # Append the data to the YAML file
    with open("saved_values.yaml", "a") as f:
        yaml.dump([data], f, default_flow_style=False)
    
    # Optionally, save the modified image
    cv2.imwrite(f"{date_time_str}.jpg", feedback_frame)
    print("Values saved successfully!")

def on_forward_button_click():
    global hue_shift, sat_shift, val_shift, alpha, num_glitches, glitch_size
    global val_threshold, val_hue_shift, blur_kernel_size, x_speed, y_speed, x_size, y_size

    try:
        with open("saved_values.yaml", "r") as f:
            saved_values = list(yaml.safe_load_all(f))
        
        if not hasattr(on_forward_button_click, "index"):
            on_forward_button_click.index = 0
        
        on_forward_button_click.index = (on_forward_button_click.index + 1) % len(saved_values)
        current_values = saved_values[on_forward_button_click.index][0]
        
        # Load the values into the global variables
        hue_shift = current_values["hue_shift"]
        sat_shift = current_values["sat_shift"]
        val_shift = current_values["val_shift"]
        alpha = current_values["alpha"]
        num_glitches = current_values["num_glitches"]
        glitch_size = current_values["glitch_size"]
        val_threshold = current_values["val_threshold"]
        val_hue_shift = current_values["val_hue_shift"]
        blur_kernel_size = current_values["blur_kernel_size"]
        x_speed = current_values["x_speed"]
        y_speed = current_values["y_speed"]
        x_size = current_values["x_size"]
        y_size = current_values["y_size"]
        
        print(f"Loaded values: {current_values}")
    except Exception as e:
        print(f"Error loading values: {e}")

def on_prev_button_click():
    global hue_shift, sat_shift, val_shift, alpha, num_glitches, glitch_size
    global val_threshold, val_hue_shift, blur_kernel_size, x_speed, y_speed, x_size, y_size

    try:
        with open("saved_values.yaml", "r") as f:
            saved_values = list(yaml.safe_load_all(f))
        
        if not hasattr(on_prev_button_click, "index"):
            on_prev_button_click.index = 0
        
        on_prev_button_click.index = (on_prev_button_click.index - 1) % len(saved_values)
        current_values = saved_values[on_prev_button_click.index]
        
        # Load the values into the global variables
        hue_shift = current_values["hue_shift"]
        sat_shift = current_values["sat_shift"]
        val_shift = current_values["val_shift"]
        alpha = current_values["alpha"]
        num_glitches = current_values["num_glitches"]
        glitch_size = current_values["glitch_size"]
        val_threshold = current_values["val_threshold"]
        val_hue_shift = current_values["val_hue_shift"]
        blur_kernel_size = current_values["blur_kernel_size"]
        x_speed = current_values["x_speed"]
        y_speed = current_values["y_speed"]
        x_size = current_values["x_size"]
        y_size = current_values["y_size"]
        
        print(f"Loaded values: {current_values}")
    except Exception as e:
        print(f"Error loading values: {e}")

def on_load_random_click():
    print("Load Random button clicked!")
    # get values from saved_values.yaml

def on_undo_button_click():
    print("Undo button clicked!")

def on_redo_button_click():
    print("Redo button clicked!")

def on_random_button_click():
    print("Random button clicked!")
    randomize_values()

def on_trap_button_click():
    print("Trap button clicked!")

def on_reload_button_click():
    print("Reload button clicked!")

def reset_values():
    global image, image_width, image_height
    global val_threshold, val_hue_shift 
    global hue_shift, sat_shift, val_shift
    global alpha, num_glitches, glitch_size, blur_kernel_size
    global x_speed, y_speed, x_size, y_size
    global x_shift, y_shift

    for s in sliders:
        s.reset()

    hue_shift = 0  # Value to shift the hue (can be positive or negative)
    sat_shift = 0  # Value to shift the saturation (0 to 255)
    val_shift = 0  # Value to shift the value (0 to 255)
    alpha = 0  # Adjust for desired feedback intensity
    num_glitches = 0  # Number of glitches to apply
    glitch_size = 1  # Size of each glitch
    val_threshold = 0  # Initial saturation threshold
    val_hue_shift = 0  # Initial partial hue shift
    blur_kernel_size = 1  # Initial blur kernel size
    x_speed = 0
    y_speed = 0
    x_size = 0
    y_size = 0
    x_shift = 0
    y_shift = 0

    dpg.set_value("hue_shift", hue_shift)
    dpg.set_value("sat_shift", sat_shift)
    dpg.set_value("val_shift", val_shift)
    dpg.set_value("feedback", alpha)
    dpg.set_value("glitch_qty", num_glitches)
    dpg.set_value("glitch_size", glitch_size)
    dpg.set_value("val_threshold", val_threshold)
    dpg.set_value("hue_val_shift", val_hue_shift)
    dpg.set_value("blur_kernel", blur_kernel_size)
    dpg.set_value("x_shift", x_shift)
    dpg.set_value("y_shift", y_shift)

def randomize_values():
    # global val_threshold, val_hue_shift
    # global hue_shift, sat_shift, val_shift
    # global alpha, num_glitches, glitch_size, blur_kernel_size
    # global x_speed, y_speed, x_size, y_size
    # global x_shift, y_shift
    # global sliders

    # for s in sliders:
    #     s.randomize()

    # hue_shift = random.randint(0, 180)  # Value to shift the hue (can be positive or negative)   
    # sat_shift = random.randint(0, 255)  # Value to shift the saturation (0 to 255)
    # val_shift = random.randint(0, 255)  # Value to shift the value (0 to 255)
    # alpha = random.uniform(0, 1)  # Adjust for desired feedback intensity
    # num_glitches = random.randint(0, 100)  # Number of glitches to apply
    # glitch_size = random.randint(1, 100)  # Size of each glitch
    # val_threshold = random.randint(0, 255)  # Initial saturation threshold
    # val_hue_shift = random.randint(0, 180)  # Initial partial hue shift
    # blur_kernel_size = max(1, random.randint(1, 31) | 1)  # Initial blur kernel size
    # x_speed = random.uniform(-1, 1)  # Random speed for X shift
    # y_speed = random.uniform(-1, 1)  # Random speed for Y shift
    # x_size = random.randint(0, 100)  # Random size for X shift
    # y_size = random.randint(0, 100)  # Random size for Y shift
    # x_shift = random.randint(0, image_width)
    # y_shift = random.randint(0, image_height)  # Random size for Y shift

    for s in sliders:
        s.randomize()
        dpg.set_value(s.tag, s.value)

    # dpg.set_value("hue_shift", hue_shift)
    # dpg.set_value("sat_shift", sat_shift)
    # dpg.set_value("val_shift", val_shift)
    # dpg.set_value("feedback", alpha)
    # dpg.set_value("glitch_qty", num_glitches)
    # dpg.set_value("glitch_size", glitch_size)
    # dpg.set_value("val_threshold", val_threshold)
    # dpg.set_value("hue_val_shift", val_hue_shift)
    # dpg.set_value("blur_kernel", blur_kernel_size)
    # dpg.set_value("x_shift", x_shift)
    # dpg.set_value("y_shift", y_shift)

def create_trackbars():
    global hue_shift, sat_shift, val_shift
    global alpha, num_glitches, glitch_size, blur_kernel_size
    global x_speed, y_speed, x_size, y_size
    global x_shift, y_shift
    global val_threshold, val_hue_shift
    global image_width, image_height
    global sliders

    hue_slider = SliderRow("Hue Shift", "hue_shift", hue_shift, 0, 180, slider_callback, 'int', reset_button_callback)
    sat_slider = SliderRow("Sat Shift", "sat_shift", sat_shift, 0, 255, slider_callback, 'int', reset_button_callback)
    val_slider = SliderRow("Val Shift", "val_shift", val_shift, 0, 255, slider_callback, 'int', reset_button_callback)
    alpha_slider = SliderRow("Feedback", "feedback", alpha, 0.0, 1.0, slider_callback, 'float', reset_button_callback)
    num_glitches_slider = SliderRow("Glitch Qty", "glitch_qty", num_glitches, 0, 100, slider_callback, 'int', reset_button_callback)
    glitch_size_slider = SliderRow("Glitch Size", "glitch_size", glitch_size, 1, 100, slider_callback, 'int', reset_button_callback)
    val_threshold_slider = SliderRow("Val Threshold", "val_threshold", val_threshold, 0, 255, slider_callback, 'int', reset_button_callback)
    val_hue_shift_slider = SliderRow("Hue Shift for Val", "hue_val_shift", val_hue_shift, 0, 180, slider_callback, 'int', reset_button_callback)
    blur_kernel_slider = SliderRow("Blur Kernel", "blur_kernel", blur_kernel_size, 1, 31, slider_callback, 'int', reset_button_callback)
    x_shift_slider = SliderRow("X Shift", "x_shift", x_shift, -image_width//2, image_width//2, slider_callback, 'int', reset_button_callback)
    y_shift_slider = SliderRow("Y Shift", "y_shift", y_shift, -image_height//2, image_height//2, slider_callback, 'int', reset_button_callback)

    sliders = [hue_slider, sat_slider, val_slider, alpha_slider, num_glitches_slider, glitch_size_slider, 
               val_threshold_slider, val_hue_shift_slider, blur_kernel_slider, x_shift_slider, y_shift_slider]

def create_buttons(width, height):
    width -= 20
    with dpg.group(horizontal=True):
        dpg.add_button(label="Random", callback=on_random_button_click, width=width)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Load <<", callback=on_prev_button_click, width=width//3)
        dpg.add_button(label="Load ??", callback=on_prev_button_click, width=width//3)
        dpg.add_button(label="Load >>", callback=on_forward_button_click, width=width//3)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Save", callback=on_save_button_click, width=width//2)
        dpg.add_button(label="Reset", callback=reset_values, width=width//2)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Undo", callback=on_undo_button_click, width=width//2)
        dpg.add_button(label="Redo", callback=on_redo_button_click, width=width//2)  

    with dpg.group(horizontal=True):
        dpg.add_button(label="Reload", callback=on_reload_button_click, width=width//2)
        dpg.add_button(label="Trap", callback=on_trap_button_click, width=width//2) 

def resize_buttons(sender, app_data):
    # Get the current width of the window
    window_width = dpg.get_item_width("Controls")
    
    # Set each button to half the window width (minus a small padding if you want)
    half_width = window_width // 2
    dpg.set_item_width(sender, half_width)
    # dpg.set_item_width("button2", half_width)

def create_control_window(width, height):
    dpg.create_context()

    with dpg.window(tag="Controls", label="Controls", width=width, height=height):
        create_trackbars()
        create_buttons(width, height)
        # dpg.set_viewport_resize_callback(resize_buttons)

    dpg.create_viewport(title='Controls', width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Controls", True)

def main():
    global image, feedback_frame, image_width, image_height
    global val_threshold, val_hue_shift 
    global hue_shift, sat_shift, val_shift
    global alpha, num_glitches, glitch_size, blur_kernel_size
    global x_speed, y_speed, x_size, y_size
    global x_shift, y_shift
    global control_image
    global window_width, window_height

    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)

    window_width = 550  # Set the desired width of the window
    window_height = 420  # Set the desired height of the window

    # Default values for hue, saturation, and value shifts
    hue_shift = 100  # Value to shift the hue (can be positive or negative)
    sat_shift = 100  # Value to shift the saturation (0 to 255)
    val_shift = 50  # Value to shift the value (0 to 255)
    alpha = 95  # Adjust for desired feedback intensity
    num_glitches = 0  # Number of glitches to apply
    glitch_size = 1  # Size of each glitch
    val_threshold = 0  # Initial saturation threshold
    val_hue_shift = 0  # Initial partial hue shift
    blur_kernel_size = 1  # Initial blur kernel size
    x_speed = 0
    y_speed = 0
    x_size = 0
    y_size = 0
    x_shift = 0
    y_shift = 0
    x_shift_mode = 0  # 0: manual, 1: perlin, 2: sine, 3: square, 4: sawtooth, 5: triangle
    y_shift_mode = 0  # 0: manual, 1: perlin, 2: sine, 3: square, 4: sawtooth, 5: triangle

    # Check if the camera opened successfully
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Create an initial empty frame
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame")
    feedback_frame = frame.copy()

    # image = cv2.imread('photo.jpg')

    image_height, image_width = frame.shape[:2]
    print(f"Image width: {image_width}, Image height: {image_height}")

    cv2.namedWindow('Modified Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, initial_width, initial_height) #set to initial frame size
    create_control_window(window_width, window_height)

    i = 0

    e = Effects()

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        alpha = get_slider_by_tag('feedback').value
        num_glitches = get_slider_by_tag('glitch_qty').value
        glitch_size = get_slider_by_tag('glitch_size').value
        hue_shift = get_slider_by_tag('hue_shift').value
        sat_shift = get_slider_by_tag('sat_shift').value
        val_shift = get_slider_by_tag('val_shift').value
        val_threshold = get_slider_by_tag('val_threshold').value
        val_hue_shift = get_slider_by_tag('hue_val_shift').value
        blur_kernel_size = get_slider_by_tag('blur_kernel').value
        x_shift = get_slider_by_tag('x_shift').value
        y_shift = get_slider_by_tag('y_shift').value

        # Apply transformations to the frame for feedback effect
        if i == 0:
            pass
        else:
            # alpha = 0.95  # Adjust for desired feedback intensity
            feedback_frame = cv2.addWeighted(frame, 1 - alpha, feedback_frame, alpha, 0)
            feedback_frame = e.glitch_image(feedback_frame, num_glitches, glitch_size) 
            feedback_frame = cv2.GaussianBlur(feedback_frame, (blur_kernel_size, blur_kernel_size), 0) 
            feedback_frame = e.shift_frame(feedback_frame, x_shift, y_shift)
            # feedback_frame = noisy("gauss", feedback_frame)        

        i += 1    

        hsv_image = cv2.cvtColor(feedback_frame, cv2.COLOR_BGR2HSV)        
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v]

        
        # Apply the hue, saturation, and value shifts to the image.
        hsv = e.shift_hsv(hsv, hue_shift, sat_shift, val_shift)
        hsv = e.val_threshold_hue_shift(hsv, val_threshold, hue_shift)

        # Merge the modified channels and convert back to BGR color space.
        modified_hsv_image = cv2.merge((hsv[H], hsv[S], hsv[V]))
        feedback_frame = cv2.cvtColor(modified_hsv_image, cv2.COLOR_HSV2BGR)

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


main()