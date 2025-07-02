import numpy as np
from enum import IntEnum
import cv2
from config import *
import random
import math
from noise import pnoise2
import time

PERLIN_SCALE = 0.01

class OscillatorShape(IntEnum):
    SINE = 0
    SQUARE = 1
    TRIANGLE = 2
    SAWTOOTH = 3

class WarpType(IntEnum):
    NONE = 0
    SINE = 1
    RADIAL = 2
    FRACTAL = 3
    PERLIN = 4

class PatternType(IntEnum):
    NONE = 0
    BARS = 1
    WAVES = 2
    RADIAL = 3
    PERLIN = 4
    FRACTAL_PERLIN = 5
    PERLIN_BLOBS = 6

class HSV(IntEnum):
    H = 0
    S = 1
    V = 2

class BlurType(IntEnum):
    NONE = 0
    GAUSSIAN = 1
    MEDIAN = 2
    BOX = 3
    BILATERAL = 4

class Effects:

    def __init__(self, image_width, image_height):

        self.height = image_height
        self.width = image_width

        self.hue_shift = params.add("hue_shift", 0, 180, 0)
        self.sat_shift = params.add("sat_shift", 0, 255, 0)
        self.val_shift = params.add("val_shift", 0, 255, 0)

        self.alpha = params.add("alpha", 0.0, 1.0, 0.0)
        self.temporal_filter = params.add("temporal_filter", 0, 1.0, 1.0)

        self.blur_type = params.add("blur_type", 0, 4, 0) # 1=Gaussian, 2=Median, 3=Box, 4=Bilateral
        self.blur_kernel_size = params.add("blur_kernel_size", 0, 100, 0)
        # TODO: implement additional required parameters for each blur type

        self.num_glitches = params.add("num_glitches", 0, 100, 0)
        self.glitch_size = params.add("glitch_size", 1, 100, 0)

        self.val_threshold = params.add("val_threshold", 0, 255, 0)
        self.val_hue_shift = params.add("val_hue_shift", 0, 255, 0)

        self.x_shift = params.add("x_shift", -image_width, image_width, 0, family="Pan") # min/max depends on image size
        self.y_shift = params.add("y_shift", -image_height, image_height, 0, family="Pan") # min/max depends on image size
        self.zoom = params.add("zoom", 0.75, 3, 1.0, family="Pan")
        self.r_shift = params.add("r_shift", -360, 360, 0, family="Pan")

        self.polar_x = params.add("polar_x", -image_width, image_width, 0)
        self.polar_y = params.add("polar_y", -image_height, image_height, 0)
        self.polar_radius = params.add("polar_radius", 0.1, 100, 1.0)

        self.contrast = params.add("contrast", 1.0, 3.0, 1.0)
        self.brightness = params.add("brightness", 0, 100, 0)

        self.key_upper_hue = params.add("key_upper_hue", 0, 180, 0)
        self.key_lower_hue = params.add("key_lower_hue", 0, 180, 0)
        self.key_upper_sat = params.add("key_upper_sat", 0, 255, 255)
        self.key_lower_sat = params.add("key_lower_sat", 0, 255, 0)
        self.key_upper_val = params.add("key_upper_val", 0, 255, 255)
        self.key_lower_val = params.add("key_lower_val", 0, 255, 0)
        # self.key_fuzz = params.add("key_fuzz", 0, 100, 0)
        # self.key_invert = params.add("key_invert", 0, 1, 0)
        # self.key_feather = params.add("key_feather", 0, 100, 0)

        self.hue_invert_angle = params.add("hue_invert_angle", 0, 360, 0)
        self.hue_invert_strength = params.add("hue_invert_strength", 0.0, 1.0, 0.0)

        self.frame_skip = params.add("frame_skip", 1, 10, 1)

        self.pattern_mode = params.add("pattern_mode", PatternType.NONE, PatternType.PERLIN_BLOBS, PatternType.NONE)

        self.warp_mode = params.add("warp_mode", 0, 1, 0) # 0=none, 1=polar



    def shift_hue(self, hue):
        """
        Shifts the hue of an image by a specified amount, wrapping aroung in necessary.
        """
        return (hue + self.hue_shift.val()) % 180

    def shift_sat(self, sat):
        """
        Shifts the saturation of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(sat + self.sat_shift.val(), 0, 255)

    def shift_val(self, val):
        """
        Shifts the value of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(val + self.val_shift.val(), 0, 255)

    def shift_hsv(self, hsv):
        """
        Shifts the hue, saturation, and value of an image by specified amounts.
        """
        return [self.shift_hue(hsv[HSV.H]), self.shift_sat(hsv[HSV.S]), self.shift_val(hsv[HSV.V])]

    def val_threshold_hue_shift(self, hsv):
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
        mask = hsv[HSV.V.value] > self.val_threshold.val()

        # Shift and wrap around the hue values for the masked pixels
        hsv[HSV.H.value][mask] = (hsv[HSV.H.value][mask] + self.val_hue_shift.val()) % 180

        return hsv
    
    def modify_hsv(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v]

        # Apply the hue, saturation, and value shifts to the image.
        hsv = self.shift_hsv(hsv)
        hsv = self.val_threshold_hue_shift(hsv)

        # Merge the modified channels and convert back to BGR color space.
        return cv2.cvtColor(cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value])), cv2.COLOR_HSV2BGR)

    # TODO: implement
    def glitch_image(self, image):
        height, width, _ = image.shape

        for _ in range(self.num_glitches.val()):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            x_glitch_size = random.randint(1, self.glitch_size.val())
            y_glitch_size = random.randint(1, self.glitch_size.val())

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

    # TODO: implement
    def warp_frame(feedback_frame, x_speed, y_speed, x_size, y_size):
        self.height, self.width = feedback_frame.shape[:2]
        feedback_frame = cv2.resize(feedback_frame, (self.width, self.height))

        # Create meshgrid for warping effect
        x_indices, y_indices = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Calculate warped indices using sine function
        time = cv2.getTickCount() / cv2.getTickFrequency()
        x_warp = x_indices + x_size * np.sin(y_indices / 20.0 + time * x_speed)
        y_warp = y_indices + y_size * np.sin(x_indices / 20.0 + time * y_speed)

        # Bound indices within valid range
        x_warp = np.clip(x_warp, 0, self.width - 1).astype(np.float32)
        y_warp = np.clip(y_warp, 0, self.height - 1).astype(np.float32)

        # Remap frame using warped indices
        feedback_frame = cv2.remap(feedback_frame, x_warp, y_warp, interpolation=cv2.INTER_LINEAR)  

        return feedback_frame
    
    def shift_frame(self, frame):
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
        (height, width) = frame.shape[:2]
        center = (width / 2, height / 2)

        # Create a new array with the same shape and data type as the original frame
        shifted_frame = np.zeros_like(frame)

        # Create the mapping arrays for the indices.
        x_map = (np.arange(width) - self.x_shift.val()) % width
        y_map = (np.arange(height) - self.y_shift.val()) % height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        # Use cv2.getRotationMatrix2D to get the rotation matrix
        M = cv2.getRotationMatrix2D(center, self.r_shift.val(), self.zoom.val())  # 1.0 is the scale

        # Perform the rotation using cv2.warpAffine
        rotated_frame = cv2.warpAffine(shifted_frame, M, (width, height))

        return rotated_frame

    def polar_transform(self, frame):
        """
        Transforms an image with horizontal bars into an image with concentric circles
        using a polar coordinate transform.
        """
        height, width = frame.shape[:2]
        center = (width // 2 + self.polar_x, height // 2 + self.polar_y.val())
        max_radius = np.sqrt((width // self.polar_radius)**2 + (height // self.polar_radius.val())**2)

        #    The flags parameter is important:
        #    cv2.INTER_LINEAR:  Bilinear interpolation (good quality)
        #    cv2.WARP_FILL_OUTLIERS:  Fills in any missing pixels
        #
        return cv2.warpPolar(
            frame,
            (width, height),  # Output size (can be different from input)
            center,
            max_radius,
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS # or +WARP_POLAR_LOG
        )

    # TODO: implement
    def gaussian_blur(self, frame, mode=1):
        if self.blur_kernel_size.val() == 0:
            pass
        elif mode == 1:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size.val(), self.blur_kernel_size.val()), 0) 
        elif mode == 2:
            frame = cv2.medianBlur(frame, self.blur_kernel_size.val())
        elif mode == 3:
            frame = cv2.blur(frame,(self.blur_kernel_size.val(), self.blur_kernel_size.val()))
        elif mode == 4:
            frame = cv2.bilateralFilter(frame,9,75,75)
        
        return frame

    def adjust_brightness_contrast(self, image):
        """
        Adjusts the brightness and contrast of an image.

        Args:
            image: The input image (NumPy array).
            alpha: Contrast control (1.0-3.0, default=1.0).
            beta: Brightness control (0-100, default=0).

        Returns:
            The adjusted image (NumPy array).
        """
        adjusted_image = cv2.convertScaleAbs(image, alpha=self.contrast.val(), beta=self.brightness.val())
        return adjusted_image

    def polarize_frame_hsv(self, frame):
        """
        Polarizes a frame by rotating hue in HSV color space.  This often gives
        a more visually interesting effect than rotating in BGR.

        Args:
            frame (numpy.ndarray): The input frame as a NumPy array (H, W, 3) in BGR format.
            angle (float): The polarization angle in degrees.
            strength (float): The strength of the polarization effect (0 to 1).

        Returns:
            numpy.ndarray: The polarized frame as a NumPy array (H, W, 3) in BGR format.
        """
        # Convert to HSV color space and extract hsv channel
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_channel = hsv_frame[:, :, 0]

        # Convert angle to OpenCV hue units (0-180)
        hue_shift = (self.hue_invert_angle.val() / 360.0) * 180

        # Apply the hue shift with strength
        shifted_hue = (hue_channel + hue_shift * self.hue_invert_strength.val()) % 180  # Wrap aroundS
        hsv_frame[:, :, 0] = shifted_hue

        # Convert back to BGR
        polarized_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return polarized_frame

    def apply_temporal_filter(self, prev_frame, cur_frame):
        """
        Applies a temporal filter (exponential moving average) to reduce noise and flicker in a video stream.

        Args:
            video_path (str, optional): Path to the video file. If None, uses the default webcam.
            alpha (float): The weighting factor for the current frame (0.0 to 1.0).
                        Higher alpha means less smoothing, more responsiveness to changes.
                        Lower alpha means more smoothing, less responsiveness.
        """

        # Convert the first frame to float for accurate averaging
        filtered_frame = prev_frame.astype(np.float32)

        # Convert current frame to float for calculations
        current_frame_float = cur_frame.astype(np.float32)

        # Apply the temporal filter (Exponential Moving Average)
        # filtered_frame = alpha * current_frame_float + (1 - alpha) * filtered_frame
        # This formula directly updates the filtered_frame based on the new current_frame.
        # It's a low-pass filter in the time domain.
        filtered_frame = cv2.addWeighted(current_frame_float, self.temporal_filter.val(), filtered_frame, 1 - self.temporal_filter.val(), 0)

        # Convert back to uint8 for display
        return cv2.convertScaleAbs(filtered_frame)

    # TODO: implement
    def apply_perlin_noise(self, frame, perlin_noise, amplitude=1.0, frequency=1.0, octaves=1):
        """ 
        Applies Perlin noise to a frame to create a textured effect.
        Args:
            frame (numpy.ndarray): The input frame as a NumPy array (H, W, 3) in BGR format.
            perlin_noise (PerlinNoise): An instance of PerlinNoise to generate noise.
            amplitude (float): The amplitude of the noise.
            frequency (float): The frequency of the noise.
            octaves (int): The number of octaves for the noise.
        Returns:
            numpy.ndarray: The frame with Perlin noise applied.
        """
        height, width = frame.shape[:2]
        noise = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                noise[y, x] = perlin_noise([x / frequency, y / frequency], octaves=octaves) * amplitude

        # Normalize the noise to [0, 255]
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert the noise to a 3-channel image
        noise_colored = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

        # Blend the original frame with the noise
        noisy_frame = cv2.addWeighted(frame, 1.0, noise_colored, 0.5, 0)
        
        return noisy_frame

    # TODO: implement   
    def lissajous_pattern(self, frame, t, width, height, A=100, B=100, a=3, b=2, delta=0):
        center_x, center_y = width // 2, height // 2
        for i in range(1000):
            x = int(center_x + A * math.sin(a * t + i * 0.01 + delta))
            y = int(center_y + B * math.sin(b * t + i * 0.01))
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        return frame
    
    # TODO: implement    
    def sync_wobble(self, frame, x_speed=20, y_speed=20):
        """
        Applies a raster wobble effect to the frame using sine waves.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        warped = np.zeros_like(frame)

        for y in range(frame.shape[0]):
            shift = int(x_speed * np.sin(y / 20.0 + cv2.getTickCount() / 1e7))
            warped[y] = np.roll(frame[y], shift, axis=0)
        
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

        return warped

#######################################################33 
    # TODO: implement        
    def _generate_perlin_flow(self, t, amp_x, amp_y, freq_x, freq_y):
        fx = np.zeros((self.height, self.width), dtype=np.float32)
        fy = np.zeros((self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                nx = x * freq_x * PERLIN_SCALE
                ny = y * freq_y * PERLIN_SCALE
                fx[y, x] = amp_x * pnoise2(nx, ny, base=int(t))
                fy[y, x] = amp_y * pnoise2(nx + 1000, ny + 1000, base=int(t))
        return fx, fy

    # TODO: implement
    def _generate_fractal_flow(self, t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity):
        fx = np.zeros((self.height, self.width), dtype=np.float32)
        fy = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                nx = x * freq_x * PERLIN_SCALE
                ny = y * freq_y * PERLIN_SCALE

                noise_x = 0.0
                noise_y = 0.0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(octaves):
                    noise_x += amplitude * pnoise2(nx * frequency, ny * frequency, base=int(t))
                    noise_y += amplitude * pnoise2((nx + 1000) * frequency, (ny + 1000) * frequency, base=int(t))
                    amplitude *= gain
                    frequency *= lacunarity

                fx[y, x] = amp_x * noise_x
                fy[y, x] = amp_y * noise_y
        return fx, fy

    # TODO: implement
    def _polar_warp(self, img, t, angle_amt, radius_amt, speed):
        cx, cy = self.width / 2, self.height / 2
        y, x = np.indices((self.height, self.width), dtype=np.float32)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        a = np.arctan2(dy, dx)

        # Modify radius and angle
        r_mod = r + np.sin(a * 5 + t * speed * 2) * radius_amt
        a_mod = a + np.cos(r * 0.02 + t * speed * 2) * (angle_amt * np.pi / 180)

        # Back to Cartesian
        map_x = (r_mod * np.cos(a_mod) + cx).astype(np.float32)
        map_y = (r_mod * np.sin(a_mod) + cy).astype(np.float32)

        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # TODO: implement
    def warp(self, img, t, mode, amp_x, amp_y, freq_x, freq_y, angle_amt, radius_amt, speed, use_fractal, octaves, gain, lacunarity):
        
        if mode == WarpType.NONE:  # No warp
            return img
        elif mode == WarpType.SINE: # Sine warp
            fx = np.sin(np.linspace(0, np.pi * 2, self.width)[None, :] + t) * amp_x
            fy = np.cos(np.linspace(0, np.pi * 2, self.height)[:, None] + t) * amp_y
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        elif mode == WarpType.FRACTAL:
            if use_fractal:
                fx, fy = self._generate_fractal_flow(t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity)
            else:
                fx, fy = self._generate_fractal_flow(t, amp_x, amp_y, freq_x, freq_y, 1, 1.0, 1.0)  # fallback: single octave
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        elif mode == WarpType.POLAR:
            return self.polar_warp(img, t, angle_amt, radius_amt, speed)
        
        elif mode == WarpType.PERLIN:  # Perlin warp
            # Fractal Perlin warp
            fx, fy = self._generate_perlin_flow(t, amp_x, amp_y, freq_x, freq_y)
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

##########################################################

    def generate_bars(self, pattern, x, norm_osc_vals):
        # Vertical bars that shift color based on osc values
        # Modulate bar position/color with oscillators
        # Using vectorized operations
        bar_mod = (np.sin(x * 0.05 + norm_osc_vals[0] * 10) + 1) / 2 # 0-1
        
        # brightness_green_channel is a 2D array, so .astype is fine here
        brightness_green_channel = (bar_mod * 255).astype(np.uint8)
        
        # osc2_norm * 255 and osc1_norm * 255 are single float values.
        # Assign them directly. NumPy will cast them to uint8 for the entire channel slice.
        pattern[:, :, 0] = int(norm_osc_vals[2] * 255) # Blue channel (B G R)
        pattern[:, :, 1] = brightness_green_channel # Green channel (this is a 2D array)
        pattern[:, :, 2] = int(norm_osc_vals[1] * 255) # Red channel

        return pattern

    def generate_waves(self, pattern, x, y, norm_osc_vals):
        # Horizontal/Vertical waves that ripple based on oscillator values
        # Combined spatial and temporal modulation
        val_x = np.sin(x * 0.03 + norm_osc_vals[0] * 5) # Horizontal wave
        val_y = np.sin(y * 0.03 + norm_osc_vals[1] * 5) # Vertical wave
        
        # Combine waves and modulate with osc2 for overall brightness/color
        total_val = (val_x + val_y) / 2 # Range -1 to 1
        brightness = ((total_val + norm_osc_vals[2]) / 2 + 1) / 2 * 255
        
        pattern = np.stack([brightness, brightness, brightness], axis=-1).astype(np.uint8) # Grayscale
        return pattern

    def generate_checkers(self, pattern, x, y, norm_osc_vals):
        # Checkerboard pattern whose square size/color shifts
        # TODO: parametrize grid size and colors
        grid_size_x = 50 * (1 + norm_osc_vals[0] * 0.5) # Dynamic grid size
        grid_size_y = 50 * (1 + norm_osc_vals[1] * 0.5)

        # Create checkerboard mask
        checker_mask = ((x // grid_size_x).astype(int) % 2 == (y // grid_size_y).astype(int) % 2)
        
        # These are single float values, cast to int for assignment
        color1 = int(norm_osc_vals[2] * 255)
        color2 = int((1 - norm_osc_vals[2]) * 255)
        
        # Apply colors based on the mask
        # NumPy will broadcast the scalar int values to the array elements
        pattern[checker_mask] = [color1, color1, color1]
        pattern[~checker_mask] = [color2, color2, color2]
        return pattern
    
    def generate_radial(self, pattern, x, y, norm_osc_vals):
        # Radial pattern that pulsates/rotates
        center_x, center_y = self.width / 2, self.height / 2
        
        DX = x - center_x
        DY = y - center_y
        distance = np.sqrt(DX**2 + DY**2)
        angle = np.arctan2(DY, DX)

        # Modulate radial distance and angle with oscillators
        # TODO: parameterize wave distance
        radial_mod = np.sin(distance * 0.05 + norm_osc_vals[0] * 10) # Radial wave based on distance
        # TODO: parameterize wave angle
        angle_mod = np.sin(angle * 5 + norm_osc_vals[1] * 5) # Angular wave based on angle

        # Combine for brightness, modulate color with osc2
        brightness_base = ((radial_mod + angle_mod) / 2 + 1) / 2 * 255
        
        # Color mapping with oscillators - ensure final values are arrays before stacking
        red_channel = (brightness_base * (1 - norm_osc_vals[2])).astype(np.uint8)
        green_channel = (brightness_base * norm_osc_vals[2]).astype(np.uint8)
        blue_channel = (brightness_base * ((norm_osc_vals[0] + norm_osc_vals[1])/2)).astype(np.uint8)

        pattern = np.stack([blue_channel, green_channel, red_channel], axis=-1)
        # Ensure pattern is in uint8 format
        return pattern
    
    def generate_fractal1(self, pattern, x, y, norm_osc_vals):
        # --- FRACTAL EFFECT: Sum of layered, modulated sine waves ---
        total_val_accumulator = np.zeros_like(x) # Accumulate values for each pixel

        base_freq_x = 0.01  # #TODO: this is  extremely interesting (see change from 0.01 to 0,1)
        base_freq_y = 0.01 
        base_amplitude_for_fractal = 1.5 # Max amplitude for the first layer of the fractal
        num_octaves = 4      # Number of layers/octaves # TODO: use param to control this

        # Oscillator values to perturb coordinates or phase
        # Use a scaling factor to make oscillator influence more noticeable
        x_perturb = norm_osc_vals[0] * 50 # Amount of x perturbation
        y_perturb = norm_osc_vals[1] * 50 # Amount of y perturbation
        phase_shift = norm_osc_vals[2] * np.pi * 2 # Phase shift for layers

        for i in range(num_octaves):
            freq_x = base_freq_x * (2 ** i) # Frequency doubles each octave
            freq_y = base_freq_y * (2 ** i)
            amplitude = base_amplitude_for_fractal / (2 ** i) # Amplitude halves each octave

            # Perturb coordinates using oscillators
            current_x = x + x_perturb
            current_y = y + y_perturb

            # Calculate wave for current octave
            # Adding phase_shift to make each layer move differently
            # TODO: i * i * i PHASE; slider to exponentially increase phase shift
            wave = (amplitude * np.sin(current_x * freq_x + phase_shift * i) +
                    amplitude * np.sin(current_y * freq_y + phase_shift * i))
            
            total_val_accumulator += wave
        
        # Normalize and scale the accumulated value to 0-255
        max_possible_val = base_amplitude_for_fractal * (2 - (1 / (2**(num_octaves-1)))) 
        
        if max_possible_val > 0.001:
            brightness = ( (total_val_accumulator / max_possible_val / 2 + 0.5) * 255).astype(np.uint8)
        else:
            brightness = np.zeros_like(x, dtype=np.uint8) # Default to black if no meaningful range

        # Apply a color based on some oscillator or fixed color for the fractal
        pattern[:, :, 0] = brightness # Blue
        pattern[:, :, 1] = brightness # Green
        pattern[:, :, 2] = brightness # Red (Grayscale for now)
        return pattern

    def generate_fractal2(self, pattern, x, y, norm_osc_vals):
        # --- ENHANCED FRACTAL EFFECT: Sum of layered, modulated sine waves ---
        total_val_accumulator = np.zeros_like(x) # Accumulate values for each pixel

        base_freq_x = 0.11  # Base spatial frequency
        base_freq_y = 0.01
        base_amplitude_for_fractal = 2.0 # Max amplitude for the first layer of the fractal
        num_octaves = 4      # Increased octaves for more detail
        
        # Dynamic lacunarity and persistence based on oscillators
        # Lacunarity controls how much frequency increases per octave (standard is 2.0)
        lacunarity = 2.5 + norm_osc_vals[0] * 0.7 # Range from 1.8 to 2.5
        # Persistence controls how much amplitude decreases per octave (standard is 0.5)
        persistence = 0.7 + norm_osc_vals[1] * 0.3 # Range from 0.4 to 0.7

        # Oscillator for overall time-based movement / phase shift
        time_offset = osc2_val * 10 # Use raw osc value for direct offset, scaled

        for i in range(num_octaves):
            freq_x = base_freq_x * (lacunarity ** i) # Frequency multiplies by lacunarity each octave
            freq_y = base_freq_y * (lacunarity ** i)
            amplitude = base_amplitude_for_fractal * (persistence ** i) # Amplitude multiplies by persistence

            # Complex coordinate perturbation (domain warping)
            # Use a sine wave based on time_offset to perturb x and y differently
            # This creates a flowing, evolving distortion
            perturbed_x = x + np.sin(y * 0.01 + time_offset * 0.05*i) * norm_osc_vals[0] * 15 # x perturbed by y-wave
            perturbed_y = y + np.sin(x * 0.01 + time_offset * 0.05) * norm_osc_vals[1] * 15 # y perturbed by x-wave

            # Calculate wave for current octave
            wave = (amplitude * np.sin(perturbed_x * freq_x + time_offset) +
                    amplitude * np.sin(perturbed_y * freq_y + time_offset))
            
            total_val_accumulator += wave

    # --- MODIFIED generate_pattern function ---
    def generate_pattern(self, pattern_type='bars'):
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Scale oscillator values to a useful range for modulation
        # Normalize to 0-1 from -1 to 1 range (assuming amplitude 1, offset 0 for initial calc)
        # The get_value() method already incorporates amplitude, so we just normalize its output
        osc0_norm = osc_bank[0].get_value() + np.abs(osc_bank[0].amplitude) / (2 * np.abs(osc_bank[0].amplitude)) if np.abs(osc_bank[0].amplitude) > 0 else 0.5
        osc1_norm = osc_bank[1].get_value() + np.abs(osc_bank[1].amplitude) / (2 * np.abs(osc_bank[1].amplitude)) if np.abs(osc_bank[1].amplitude) > 0 else 0.5
        osc2_norm = osc_bank[2].get_value() + np.abs(osc_bank[2].amplitude) / (2 * np.abs(osc_bank[2].amplitude)) if np.abs(osc_bank[2].amplitude) > 0 else 0.5

        norm_osc_vals = (osc0_norm, osc1_norm, osc2_norm)

        # Create coordinate grids for vectorized operations (much faster than loops)
        x_coords = np.linspace(0, self.width - 1, self.width, dtype=np.float32)
        y_coords = np.linspace(0, self.height - 1, self.height, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords)

        if pattern_type == PatternType.NONE:
            pass
        elif pattern_type == PatternType.BARS:
            pattern = self.generate_bars(pattern, X, norm_osc_vals)
        elif pattern_type == PatternType.WAVES:
            pattern = self.generate_waves(pattern, X, Y, norm_osc_vals)
        elif pattern_type == PatternType.CHECKERS:
            pattern = self.generate_checkers(pattern, X, Y, norm_osc_vals)
        elif pattern_type == PatternType.RADIAL:
            pattern = self.generate_radial(pattern, X, Y, norm_osc_vals)
        elif pattern_type == PatternType.PERLIN:
            pattern = self.generate_fractal1(pattern, X, Y, norm_osc_vals)
        elif pattern_type == PatternType.FRACTAL_PERLIN:
            pattern = self.generate_fractal2(pattern, X, Y, norm_osc_vals)
        elif pattern_type == PatternType.PERLIN_BLOBS:
            pattern = self.generate_perlin_blobs(norm_osc_vals) # TODO: implement

        return pattern


    # TODO: implement
    def generate_perlin_blobs(self, norm_osc_vals):
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Perlin noise parameters, modulated by oscillators for evolution
        scale_x = 0.005 + norm_osc_vals[0] * 0.01  # Modulate X spatial scale (roughness)
        scale_y = 0.005 + norm_osc_vals[1] * 0.01  # Modulate Y spatial scale (roughness)
        octaves = 6                         # Number of layers of noise
        persistence = 0.5 + norm_osc_vals[2] * 0.2 # Modulate persistence (how much each octave contributes)
        lacunarity = 2.0                    # How much frequency increases per octave (standard)

        # Use time as a Z-axis for 3D Perlin noise to ensure continuous evolution
        # The higher the frequency of osc2, the faster the blobs will "flow"
        time_factor = time.time() * (0.1 + norm_osc_vals[2] * 0.5) # Modulate evolution speed

        for y in range(self.height):
            for x in range(self.width):
                # Map x, y to a smaller range for noise function
                nx = x * scale_x
                ny = y * scale_y

                # Get Perlin noise value for each pixel using 3D noise (x, y, time)
                # This is the "shape" of the blob
                noise_val_shape = noise.pnoise3(nx, ny, time_factor,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=1024, repeaty=1024, repeatz=1024,
                                                base=0) # Base is like a seed offset

                # Map noise_val_shape from (-1, 1) range to (0, 1)
                normalized_noise_val = (noise_val_shape + 1) / 2

                # Use another noise sample for color, also evolving with time
                # Slightly different scales/parameters to make color vary independently of shape
                noise_val_color_r = noise.pnoise3(nx * 0.8, ny * 1.2, time_factor * 0.7,
                                                octaves=4, persistence=0.6, lacunarity=2.2, base=1)
                noise_val_color_g = noise.pnoise3(nx * 1.1, ny * 0.9, time_factor * 1.1,
                                                octaves=4, persistence=0.7, lacunarity=1.8, base=2)
                noise_val_color_b = noise.pnoise3(nx * 0.9, ny * 1.0, time_factor * 0.9,
                                                octaves=4, persistence=0.5, lacunarity=2.0, base=3)
                
                # Normalize color noise values
                r = int(((noise_val_color_r + 1) / 2) * 255)
                g = int(((noise_val_color_g + 1) / 2) * 255)
                b = int(((noise_val_color_b + 1) / 2) * 255)

                # Apply shape to color: lower noise_val_shape means darker/more transparent area
                # We'll use normalized_noise_val to control brightness or alpha
                # For simplicity, let's blend with black based on normalized_noise_val
                # Brighter areas of noise_val_shape will reveal more color
                final_r = int(r * normalized_noise_val)
                final_g = int(g * normalized_noise_val)
                final_b = int(b * normalized_noise_val)

                pattern[y, x] = [final_b, final_g, final_r] # OpenCV uses BGR

        return pattern