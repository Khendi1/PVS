import numpy as np
from enum import Enum
import cv2
import config as p
import random

class HSV(Enum):
    H = 0
    S = 1
    V = 2

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
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)         # Convert the image to HSV color space.
        new_h = self.shift_hue(hsv[HSV.H.value], hue_shift)
        new_s = self.shift_sat(hsv[HSV.S.value], sat_shift)
        new_v = self.shift_val(hsv[HSV.V.value], val_shift)

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
        mask = hsv[HSV.V.value] > val_threshold

        # Shift the hue values for the masked pixels.  We use modulo 180
        # to ensure the hue values stay within the valid range of 0-179.
        hsv[HSV.H.value][mask] = (hsv[HSV.H.value][mask] + hue_shift) % 180

        return hsv
    
    def modify_hsv(self, image, hue_shift, sat_shift, val_shift, val_threshold, val_hue_shift):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v]

        # Apply the hue, saturation, and value shifts to the image.
        hsv = self.shift_hsv(hsv, hue_shift, sat_shift, val_shift)
        hsv = self.val_threshold_hue_shift(hsv, val_threshold, val_hue_shift)

        # Merge the modified channels and convert back to BGR color space.
        return cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value]))

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
        mask = hsv[HSV.S.value] > val_threshold

        # Shift the hue values for the masked pixels.  We use modulo 180
        # to ensure the hue values stay within the valid range of 0-179.
        hsv[HSV.H.value][mask] = (hsv[HSV.H.value][mask] + hue_shift) % 180

        return hsv

    def shift_frame(self, frame, shift_x, shift_y, shift_r=None, zoom=1.0):
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
        (height, width) = frame.shape[:2]  # Get height and width
        center = (width / 2, height / 2)

        # Create a new array with the same shape and data type as the original frame
        shifted_frame = np.zeros_like(frame)

        # Create the mapping arrays for the indices.
        x_map = (np.arange(width) - shift_x) % width
        y_map = (np.arange(height) - shift_y) % height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        # Use cv2.getRotationMatrix2D to get the rotation matrix
        M = cv2.getRotationMatrix2D(center, shift_r, zoom)  # 1.0 is the scale

        # Perform the rotation using cv2.warpAffine
        rotated_frame = cv2.warpAffine(shifted_frame, M, (width, height))

        return rotated_frame

    def polar_transform(self, frame, x_shift, y_shift, radius=2):
        """
        Transforms an image with horizontal bars into an image with concentric circles
        using a polar coordinate transform.
        """
        height, width = frame.shape[:2]
        center = (width // 2+x_shift, height // 2+y_shift)
        max_radius = np.sqrt((width // radius)**2 + (height // radius)**2)

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

    def create_horizontal_bars_image(width, height, bar_width, bar_spacing, bar_color=(255, 255, 255), bg_color=(0,0,0)):
        """
        Creates a synthetic image with two horizontal bars.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            bar_width (int): Thickness of each bar.
            bar_spacing (int): Distance between the centers of the two bars.
            bar_color (tuple, optional): Color of the bars (BGR). Defaults to white.
            bg_color (tuple, optional): Color of the background (BGR). Defaults to black.

        Returns:
            numpy.ndarray: A NumPy array representing the image.
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = bg_color  # Fill with background color

        # Calculate the y-coordinates of the bars
        center_y = height // 2
        bar1_y = center_y - bar_spacing // 2
        bar2_y = center_y + bar_spacing // 2

        # Draw the bars.  Ensure we handle cases where bar_width is even/odd
        bar1_start = max(0, bar1_y - bar_width // 2)
        bar1_end   = min(height, bar1_y + (bar_width + 1) // 2) # +1 for odd widths
        bar2_start = max(0, bar2_y - bar_width // 2)
        bar2_end   = min(height, bar2_y + (bar_width + 1) // 2)

        img[bar1_start:bar1_end, :, :] = bar_color
        img[bar2_start:bar2_end, :, :] = bar_color
        return img

    def gaussian_blur(self, frame, blur_kernel_size, mode=1):
        if blur_kernel_size == 0:
            pass
        elif mode == 1:
            frame = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 0) 
        elif mode == 2:
            frame = cv2.medianBlur(frame, blur_kernel_size)
        elif mode == 3:
            frame = cv2.blur(frame,(p.params["blur_kernel_size"].value, p.params["blur_kernel_size"].value))
        elif mode == 4:
            frame = cv2.bilateralFilter(frame,9,75,75)
        
        return frame

    def adjust_brightness_contrast(self, image, alpha=1.0, beta=0):
        """
        Adjusts the brightness and contrast of an image.

        Args:
            image: The input image (NumPy array).
            alpha: Contrast control (1.0-3.0, default=1.0).
            beta: Brightness control (0-100, default=0).

        Returns:
            The adjusted image (NumPy array).
        """
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

def polarize_frame_hsv(frame, angle=0, strength=1.0):
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

def apply_temporal_filter(prev_frame, cur_frame, alpha=0.95):
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
    filtered_frame = cv2.addWeighted(current_frame_float, alpha, filtered_frame, 1 - alpha, 0)

    # Convert back to uint8 for display
    return cv2.convertScaleAbs(filtered_frame)

def apply_perlin_noise(frame, perlin_noise, amplitude=1.0, frequency=1.0, octaves=1):
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