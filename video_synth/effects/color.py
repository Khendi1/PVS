import cv2
import numpy as np
import logging

from effects.base import EffectBase
from effects.enums import HSV
from common import Widget

log = logging.getLogger(__name__)

class Color(EffectBase):

    def __init__(self, params, group):
        subgroup = self.__class__.__name__
        self.params = params
        self.group = group

        self.hue_shift = params.add("hue_shift",
                                    min=0, max=180, default=0,
                                    subgroup=subgroup, group=group)
        self.sat_shift = params.add("sat_shift",
                                    min=0, max=255, default=0,
                                    subgroup=subgroup, group=group)
        self.val_shift = params.add("val_shift",
                                    min=0, max=255, default=0,
                                    subgroup=subgroup, group=group)

        self.levels_per_channel = params.add("posterize_levels",
                                             min=0, max=100, default=0.0,
                                             subgroup=subgroup, group=group)
        self.num_hues = params.add("num_hues",
                                   min=2, max=10, default=8,
                                   subgroup=subgroup, group=group)

        self.val_threshold = params.add("val_threshold",
                                        min=0, max=255, default=0,
                                        subgroup=subgroup, group=group)
        self.val_hue_shift = params.add("val_hue_shift",
                                        min=0, max=255, default=0,
                                        subgroup=subgroup, group=group)

        self.solarize_threshold = params.add("solarize_threshold",
                                             min=0, max=100, default=0.0,
                                             subgroup=subgroup, group=group)
        self.hue_invert_angle = params.add("hue_invert_angle",
                                           min=0, max=360, default=0,
                                           subgroup=subgroup, group=group)
        self.hue_invert_strength = params.add("hue_invert_strength",
                                              min=0.0, max=1.0, default=0.0,
                                              subgroup=subgroup, group=group)

        self.contrast = params.add("contrast",
                                   min=0.5, max=3.0, default=1.0,
                                   subgroup=subgroup, group=group)
        self.brightness = params.add("brightness",
                                     min=0, max=100, default=0,
                                     subgroup=subgroup, group=group)
        self.gamma = params.add("gamma",
                                min=0.1, max=3.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.highlight_compression = params.add("highlight_compression",
                                                min=0.0, max=1.0, default=0.0,
                                                subgroup=subgroup, group=group)

    def _shift_hue(self, hue: int):
        """
        Shifts the hue of an image by a specified amount, wrapping aroung in necessary.
        """
        return (hue + self.hue_shift.value) % 180

    def _shift_sat(self, sat: int):
        """
        Shifts the saturation of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(sat + self.sat_shift.value, 0, 255)

    def _shift_val(self, val: int):
        """
        Shifts the value of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(val + self.val_shift.value, 0, 255)

    def _shift_hsv(self, hsv: list):
        """
        Shifts the hue, saturation, and value of an image by specified amounts.
        """
        return [
            self._shift_hue(hsv[HSV.H]),
            self._shift_sat(hsv[HSV.S]),
            self._shift_val(hsv[HSV.V]),
        ]

    def _val_threshold_hue_shift(self, hsv: list):
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
        hsv[HSV.H.value][mask] = (
            hsv[HSV.H.value][mask] + self.val_hue_shift.value
        ) % 180

        return hsv

    def modify_hsv(self, image: np.ndarray):
        if (self.hue_shift.value == 0 and
            self.sat_shift.value == 0 and
            self.val_shift.value == 0 and
            self.val_threshold.value == 0):
            return image
        
        is_float = image.dtype == np.float32
        if is_float:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image
            
        hsv_image = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v] # Create a list of channels for _shift_hsv
        
        # Apply the hue, saturation, and value shifts to the image.
        hsv = self._shift_hsv(hsv)
        hsv = self._val_threshold_hue_shift(hsv)

        # Merge the modified channels and convert back to BGR color space.
        result_uint8 = cv2.cvtColor(
            cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value])),
            cv2.COLOR_HSV2BGR,
        )

        if is_float:
            return result_uint8.astype(np.float32)
        else:
            return result_uint8

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
            # log.debug("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)

        # Calculate the step size for quantization, round
        step = 256 // self.levels_per_channel.value
        half_step = step // 2

        # Method 2: Quantization with rounding (often looks better)
        posterized_frame = ((frame + half_step) // step) * step

        # Ensure values stay within 0-255.
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
            # log.debug("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)  # Converts to uint8, scales if needed

        frame = np.where(frame > self.solarize_threshold.value, 255 - frame, frame)

        return frame.astype(np.uint8)

    def adjust_brightness_contrast(self, image):
        """
        Adjusts the brightness and contrast of an image using float precision.
        """
        if self.contrast.value == 1.0 and self.brightness.value == 0:
            return image

        # Ensure we are working with floats for this calculation
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Perform the calculation in float32.
        # Clipping will be handled once at the end of the get_frames function.
        # return image * self.contrast.value + self.brightness.value
        return cv2.convertScaleAbs(
                image, alpha=self.contrast.value, beta=self.brightness.value
            ).astype(np.float32)

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
        shifted_hue = (
            hue_channel + hue_shift * self.hue_invert_strength.value
        ) % 180  # Wrap around
        hsv_frame[:, :, 0] = shifted_hue

        # Convert back to BGR
        polarized_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return polarized_frame

    def adjust_gamma(self, image: np.ndarray):
        """
        Applies gamma correction to the image.
        """
        gamma = self.gamma.value
        if gamma == 1.0:
            return image

        # Ensure we are working with floats
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize to [0, 1], apply gamma, then scale back to [0, 255]
        # The 1e-6 avoids division by zero for black pixels.
        return np.power(image / 255.0, gamma) * 255.0


    def tonemap(self, image: np.ndarray):
        """
        Applies Reinhard tonemapping to compress highlights and prevent clipping.
        """
        strength = self.highlight_compression.value
        if strength == 0.0:
            return image  # PERFORMANCE: Return unchanged when disabled

        # Map the [0,1] strength slider to the intensity parameter of the tonemapper.
        # Negative intensity values decrease brightness.
        tonemap_intensity = (strength * -8.0)

        tonemapper = cv2.createTonemapReinhard(gamma=2.2, intensity=tonemap_intensity, light_adapt=0.0, color_adapt=0.0)
        
        # The tonemapper expects a float32 BGR image, which is what we have.
        # It processes the image and returns it mapped to the [0, 1] range.
        ldr_image = tonemapper.process(image)
        
        # Scale the result back up to the [0, 255] range.
        return ldr_image * 255.0
