import numpy as np
from enum import IntEnum
import cv2
from config import *
import random
import math
from noise import pnoise2
import noise 
import time

class WarpType(IntEnum):
    NONE = 0
    SINE = 1
    RADIAL = 2
    FRACTAL = 3
    PERLIN = 4
    WARP0 = 5  # this is a placeholder for the old warp_frame method; yet to be tested

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

# class Color:
#     def __init__(self):

class Effects:

    def __init__(self, image_width: int, image_height: int):

        self.height = image_height
        self.width = image_width

        self.hue_shift = params.add("hue_shift", 0, 180, 0)
        self.sat_shift = params.add("sat_shift", 0, 255, 0)
        self.val_shift = params.add("val_shift", 0, 255, 0)

        self.alpha = params.add("alpha", 0.0, 1.0, 0.0)
        self.temporal_filter = params.add("temporal_filter", 0, 1.0, 1.0)

        self.blur_type = params.add("blur_type", 0, 4, 1) # 1=Gaussian, 2=Median, 3=Box, 4=Bilateral
        self.blur_kernel_size = params.add("blur_kernel_size", 1, 100, 1)
        # TODO: implement additional requirefd parameters for each blur type

        self.num_glitches = params.add("num_glitches", 0, 100, 0)
        self.glitch_size = params.add("glitch_size", 1, 100, 0)

        self.val_threshold = params.add("val_threshold", 0, 255, 0)
        self.val_hue_shift = params.add("val_hue_shift", 0, 255, 0)

        self.x_shift = params.add("x_shift", -image_width, image_width, 0, family="Pan") # min/max depends on image size
        self.y_shift = params.add("y_shift", -image_height, image_height, 0, family="Pan") # min/max depends on image size
        self.zoom = params.add("zoom", 0.75, 3, 1.0, family="Pan")
        self.r_shift = params.add("r_shift", -360, 360, 0.0, family="Pan")

        self.polar_x = params.add("polar_x", -image_width, image_width, 0)
        self.polar_y = params.add("polar_y", -image_height, image_height, 0)
        self.polar_radius = params.add("polar_radius", 0.1, 100, 1.0)

        self.contrast = params.add("contrast", 0.5, 3.0, 1.0)
        self.brightness = params.add("brightness", 0, 100, 0)

        self.hue_invert_angle = params.add("hue_invert_angle", 0, 360, 0)
        self.hue_invert_strength = params.add("hue_invert_strength", 0.0, 1.0, 0.0)

        self.frame_skip = params.add("frame_skip", 1, 10, 1)

        self.warp_type = params.add("warp_type", WarpType.NONE.value, WarpType.WARP0.value, WarpType.NONE.value)
        self.warp_angle_amt = params.add("warp_angle_amt", 0, 360, 30)
        self.warp_radius_amt = params.add("warp_radius_amt", 0, 360, 30)
        self.warp_speed = params.add("warp_speed", 0, 100, 10)
        self.warp_use_fractal = params.add("warp_use_fractal", 0, 1, 0)
        self.warp_octaves = params.add("warp_octaves", 1, 8, 4)
        self.warp_gain = params.add("warp_gain", 0.0, 1.0, 0.5)
        self.warp_lacunarity = params.add("warp_lacunarity", 1.0, 4.0, 2.0)
        #warp0/first_warp parameters
        self.x_speed = params.add("x_speed", 0.0, 100.0, 1.0)
        self.x_size = params.add("x_size", 0.25, 100.0, 20.0) 
        self.y_speed = params.add("y_speed", 0.0, 10.0, 1.0)
        self.y_size = params.add("y_size", 0.0, 100.0, 10.0)

        self.x_sync_freq = params.add("x_sync_freq", 0.1, 100.0, 1.0)
        self.x_sync_amp = params.add("x_sync_amp", -200, 200, 0.0)
        self.x_sync_speed = params.add("x_sync_speed", 5.0, 10.0, 9.0)

        self.y_sync_freq = params.add("y_sync_freq", 0.1, 100.0, 1.0)
        self.y_sync_amp = params.add("y_sync_amp", -200, 200, 00.0)
        self.y_sync_speed = params.add("y_sync_speed", 5.0, 10.0, 9.0)

        self.lissajous_A = params.add("lissajous_A", 0, 100, 50)
        self.lissajous_B = params.add("lissajous_B", 0, 100, 50)
        self.lissajous_a = params.add("lissajous_a", 0, 100, 50)
        self.lissajous_b = params.add("lissajous_b", 0, 100, 50)
        self.lissajous_delta = params.add("lissajous_delta", 0, 360, 0)

        # TODO: implement
        self.sequence = params.add("sequence", 0, 100, 0)
        self.sharpen_intensity = params.add("sharpen_intensity", 4.0, 8.0, 4.0)
        self.levels_per_channel = params.add("posterize_levels", 2, 128, 2.0)
        self.solarize_threshold = params.add("solarize_threshold", 0, 128, 0.0)
        self.num_hues = params.add("num_hues", 2, 10, 8)

    def shift_hue(self, hue: int):
        """
        Shifts the hue of an image by a specified amount, wrapping aroung in necessary.
        """
        return (hue + self.hue_shift.value) % 180

    def shift_sat(self, sat: int):
        """
        Shifts the saturation of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(sat + self.sat_shift.value, 0, 255)

    def shift_val(self, val: int):
        """
        Shifts the value of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(val + self.val_shift.value, 0, 255)

    def shift_hsv(self, hsv: list):
        """
        Shifts the hue, saturation, and value of an image by specified amounts.
        """
        return [self.shift_hue(hsv[HSV.H]), self.shift_sat(hsv[HSV.S]), self.shift_val(hsv[HSV.V])]

    def val_threshold_hue_shift(self, hsv: list):
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
        mask = hsv[HSV.V.value] > self.val_threshold.value

        # Shift and wrap around the hue values for the masked pixels
        hsv[HSV.H.value][mask] = (hsv[HSV.H.value][mask] + self.val_hue_shift.value) % 180

        return hsv
    
    def modify_hsv(self, image: np.ndarray):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v]

        # Apply the hue, saturation, and value shifts to the image.
        hsv = self.shift_hsv(hsv)
        hsv = self.val_threshold_hue_shift(hsv)

        # Merge the modified channels and convert back to BGR color space.
        return cv2.cvtColor(cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value])), cv2.COLOR_HSV2BGR)

    def limit_hues_kmeans(self, frame: np.ndarray):

        if self.num_hues.value <= self.num_hues.max:
            return frame

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)        

        # Reshape the image to a 2D array of pixels (N x 3) for K-Means
        # We'll focus on the Hue channel, so we can reshape to (N x 1) if we only want to cluster hue
        # For a more general color quantization (hue, saturation, value), keep it N x 3

        # Option 2: Quantize all HSV channels (more common for general color reduction)
        # This will result in a limited set of overall colors, which inherently limits hues.
        data = frame.reshape((-1, 3)).astype(np.float32)

        # Define criteria for K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Apply K-Means clustering
        # attempts: Number of times the algorithm is executed with different initial labellings.
        #           The best result of all attempts is returned.
        # flags: Specifies how initial centers are chosen. KMEANS_PP_CENTERS is a good choice.
        compactness, labels, centers = cv2.kmeans(
            data, self.num_hues.value, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Convert back to uint8 and reshape to original image dimensions
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(hsv_image.shape)

        # Convert back to BGR for display
        output_image = cv2.cvtColor(quantized_image, cv2.COLOR_HSV2BGR)
        return output_image

    # TODO: implement
    def glitch_image(self, image: np.ndarray):
        height, width, _ = image.shape

        for _ in range(self.num_glitches.value):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            x_glitch_size = random.randint(1, self.glitch_size.value)
            y_glitch_size = random.randint(1, self.glitch_size.value)

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
    
    def shift_frame(self, frame: np.ndarray):
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
        x_map = (np.arange(width) - self.x_shift.value) % width
        y_map = (np.arange(height) - self.y_shift.value) % height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        # Use cv2.getRotationMatrix2D to get the rotation matrix
        M = cv2.getRotationMatrix2D(center, self.r_shift.value, self.zoom.value)  # 1.0 is the scale

        # Perform the rotation using cv2.warpAffine
        rotated_frame = cv2.warpAffine(shifted_frame, M, (width, height))

        return rotated_frame

    def polar_transform(self, frame: np.ndarray):
        """
        Transforms an image with horizontal bars into an image with concentric circles
        using a polar coordinate transform.
        """
        height, width = frame.shape[:2]
        center = (width // 2 + self.polar_x, height // 2 + self.polar_y.value)
        max_radius = np.sqrt((width // self.polar_radius)**2 + (height // self.polar_radius.value)**2)

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
    def gaussian_blur(self, frame: np.ndarray):
        mode = self.blur_type.value
        if mode == BlurType.NONE:
            pass
        elif mode == BlurType.GAUSSIAN:
            # TODO: apply snapping to odd kernel size here rather than in the GUI callback, as other blur type *MAY* permit other sizes
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size.value, self.blur_kernel_size.value), 0) 
        elif mode == BlurType.MEDIAN:
            frame = cv2.medianBlur(frame, self.blur_kernel_size.value)
        elif mode == BlurType.BOX:
            frame = cv2.blur(frame,(self.blur_kernel_size.value, self.blur_kernel_size.value))
        elif mode == BlurType.BILATERAL:
            frame = cv2.bilateralFilter(frame,self.blur_kernel_size.value,75,75)
        
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
        adjusted_image = cv2.convertScaleAbs(image, alpha=self.contrast.value, beta=self.brightness.value)
        return adjusted_image

    def polarize_frame_hsv(self, frame: np.ndarray):
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
        if self.hue_invert_strength.value <= self.hue_invert_strength.min:
            return frame
        
        # Convert to HSV color space and extract hsv channel
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_channel = hsv_frame[:, :, 0]

        # Convert angle to OpenCV hue units (0-180)
        hue_shift = (self.hue_invert_angle.value / 360.0) * 180

        # Apply the hue shift with strength
        shifted_hue = (hue_channel + hue_shift * self.hue_invert_strength.value) % 180  # Wrap aroundS
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
        filtered_frame = cv2.addWeighted(current_frame_float, self.temporal_filter.value, filtered_frame, 1 - self.temporal_filter.value, 0)

        # Convert back to uint8 for display
        return cv2.convertScaleAbs(filtered_frame)

    def sharpen_frame(self, frame: np.ndarray):

        if self.sharpen_intensity.value <= self.sharpen_intensity.min + 0.01:
            return frame
        
        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, self.sharpen_intensity.value, -1],
            [0, -1, 0]
        ])

        # Apply the kernel to the frame using cv2.filter2D
        # -1 indicates that the output image will have the same depth (data type) as the input image.
        sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)

        return sharpened_frame

    def posterize(self, frame: np.ndarray):
        """
        Applies a posterization effect to an image.

        Args:
            image_path (str): The path to the input image.
            levels_per_channel (int): The number of distinct intensity levels
                                    to use for each color channel (e.g., 8, 16, 32).
                                    Must be between 2 and 255.
        Returns:
            numpy.ndarray: The posterized image.
        """
        if self.levels_per_channel.value <= self.levels_per_channel.min:
            return frame

        # Ensure the image is in 8-bit format (0-255)
        if frame.dtype != np.uint8:
            print("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)

        # Calculate the step size for quantization
        # Each pixel value will be mapped to one of 'levels_per_channel' distinct values.
        # For example, if levels_per_channel is 8, values 0-31 map to 0, 32-63 map to 32, etc.
        step = 256 // self.levels_per_channel.value
        
        # Calculate the half-step to round to the nearest quantization level
        half_step = step // 2

        # Apply posterization to each color channel (B, G, R)
        # This involves dividing by the step, multiplying by the step, and adding half_step for rounding.
        # The `np.clip` ensures values stay within 0-255.
        
        # Method 1: Simple quantization (floor division)
        # posterized_frame = (frame // step) * step

        # Method 2: Quantization with rounding (often looks better)
        posterized_frame = ((frame + half_step) // step) * step
        posterized_frame = np.clip(posterized_frame, 0, 255).astype(np.uint8)


        return posterized_frame

    def solarize_image(self, frame: np.ndarray):
        """
        Applies a solarize effect to an image.

        Args:
            image_path (str): The path to the input image.
            threshold (int): The intensity threshold (0-255). Pixels above this
                            value will be inverted. Default is 128 (mid-range).

        Returns:
            numpy.ndarray: The solarized image.
        """
        if self.solarize_threshold.value == 0:
            return frame

        # Ensure the image is in 8-bit format (0-255)
        if frame.dtype != np.uint8:
            print("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame) # Converts to uint8, scales if needed

        # Create a copy to modify
        solarized_frame = frame.copy()

        # Apply solarization logic.
        # We can use boolean indexing for efficient processing.
        # For each channel (B, G, R) and each pixel:
        # If the pixel value is greater than the threshold, invert it.

        # Option 1: Using NumPy's where function (very concise)
        solarized_frame = np.where(frame > self.solarize_threshold.value, 255 - frame, frame)

        # Option 2: Manual iteration (less efficient for large images, but illustrative)
        # height, width, channels = frame.shape
        # for y in range(height):
        #     for x in range(width):
        #         for c in range(channels):
        #             pixel_val = frame[y, x, c]
        #             if pixel_val > threshold:
        #                 solarized_frame[y, x, c] = 255 - pixel_val
        #             else:
        #                 solarized_frame[y, x, c] = pixel_val

        # Ensure the output is uint8
        solarized_frame = solarized_frame.astype(np.uint8)

        return solarized_frame
    
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

    # TODO: make this more engaging   
    def lissajous_pattern(self, frame, t):
        center_x, center_y = self.width // 2, self.height // 2
        for i in range(1000):
            x = int(center_x + self.lissajous_A.value * math.sin(self.lissajous_a.value * t + i * 0.01 + self.lissajous_delta.value * math.pi / 180))
            y = int(center_y + self.lissajous_B.value * math.sin(self.lissajous_b.value* t + i * 0.01))
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        return frame
    
    def sync(self, frame: np.ndarray):
        """
        Applies a raster wobble effect to the frame using sine waves on both X and Y axes.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        warped = np.zeros_like(frame)

        # X-axis wobble (horizontal shift per row)
        for y in range(height):
            shift_x = int(self.x_sync_amp.value * np.sin(
                y / self.x_sync_freq.value + cv2.getTickCount() / (10 ** self.x_sync_speed.value)))
            warped[y] = np.roll(frame[y], shift_x, axis=0)

        # Y-axis wobble (vertical shift per column)
        warped_y = np.zeros_like(warped)
        for x in range(width):
            shift_y = int(self.y_sync_amp.value * np.sin(
                x / self.y_sync_freq.value + cv2.getTickCount() / (10 ** self.y_sync_speed.value)))
            warped_y[:, x] = np.roll(warped[:, x], shift_y, axis=0)

        warped_y = cv2.cvtColor(warped_y, cv2.COLOR_RGB2BGR)
        return warped_y

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

    def _first_warp(self, frame: np.ndarray):
        self.height, self.width = feedback_frame.shape[:2]
        feedback_frame = cv2.resize(feedback_frame, (self.width, self.height))

        # Create meshgrid for warping effect
        x_indices, y_indices = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Calculate warped indices using sine function
        time = cv2.getTickCount() / cv2.getTickFrequency()
        x_warp = x_indices + self.x_size * np.sin(y_indices / 20.0 + time * self.x_speed)
        y_warp = y_indices + self.y_size * np.sin(x_indices / 20.0 + time * self.y_speed)

        # Bound indices within valid range
        x_warp = np.clip(x_warp, 0, self.width - 1).astype(np.float32)
        y_warp = np.clip(y_warp, 0, self.height - 1).astype(np.float32)

        # Remap frame using warped indices
        feedback_frame = cv2.remap(feedback_frame, x_warp, y_warp, interpolation=cv2.INTER_LINEAR)  

        return feedback_frame
    
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

        elif mode == WarpType.RADIAL:
            return self.polar_warp(img, t, angle_amt, radius_amt, speed)
        
        elif mode == WarpType.PERLIN:  # Perlin warp
            # Fractal Perlin warp
            fx, fy = self._generate_perlin_flow(t, amp_x, amp_y, freq_x, freq_y)
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        elif mode == WarpType.WARP0:
            return self._first_warp(img)

##########################################################
