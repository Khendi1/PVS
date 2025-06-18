import numpy as np
from enum import IntEnum
import cv2
from config import params
import random

class HSV(IntEnum):
    H = 0
    S = 1
    V = 2

class Effects:
    def __init__(self, image_width, image_height):
        self.hue_shift = params.add("hue_shift", 0, 180, 100)
        self.sat_shift = params.add("sat_shift", 0, 255, 100)
        self.val_shift = params.add("val_shift", 0, 255, 50)
        self.alpha = params.add("alpha", 0.0, 1.0, 0.9)
        self.blur_kernel_size = params.add("blur_kernel_size", 0, 100, 0)
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
        self.temporal_filter = params.add("temporal_filter", 0, 1.0, 0.95)
        self.cc_upper = params.add("cc_upper", 0, 255, 255)
        self.cc_lower = params.add("cc_lower", 0, 255, 0)

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
        return cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value]))

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

    def polarize_frame_hsv(self, frame, angle=0, strength=1.0):
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
        hue_shift = (angle / 360.0) * 180

        # Apply the hue shift with strength
        shifted_hue = (hue_channel + hue_shift * strength) % 180  # Wrap aroundS
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