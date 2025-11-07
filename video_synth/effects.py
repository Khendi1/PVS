"""
See notes on Program Architecture in README.md

Classes stored here:
    - enumeration classes for effect modes, etc.
    - EffectBase singleton class; extended by each Effect subclass (EX: Color, Feedback, etc.)
    - Effect subclasses; require Control Structures as args, instantiates and adds Control Objects (Parameters, Toggles) to approp structures
    - EffectManager to init, manage, and sequence Effect subclasses

Contribution guide:
    - each new effect class should extend the EffectsBase class
    - each new effect class should be initialized in EffectManager/init()
    - each new effect class should expose 2 kinds of public functions,
      and all private classes should be appended with '_'
        - effect methods: these will be automatically added to the sequencer
        - gui methods: these will not be automatically added or called by the gui; 
                       these should have the word 'gui' in the method name for easy filtering
                       these must be manually called in gui.py/create_trackbars()
"""

import math
import random
from collections import deque
from enum import IntEnum, Enum, auto
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from noise import pnoise2
from gui_elements import TrackbarRow, Toggle, RadioButtonRow
import logging
from patterns3 import Patterns  
from param import ParamTable
from enum import IntEnum, Enum, auto

log = logging.getLogger(__name__)


"""This section stores all local custom enum classes"""

class LumaMode(IntEnum):
    WHITE = auto()
    BLACK = auto()

class NoiseType(IntEnum):
    NONE = 0
    GAUSSIAN = auto()
    POISSON = auto()
    SALT_AND_PEPPER = auto()
    SPECKLE = auto()
    SPARSE = auto()
    RANDOM = auto()

class WarpType(IntEnum):
    NONE = 0
    SINE = auto()
    RADIAL = auto()
    FRACTAL = auto()
    PERLIN = auto()
    WARP0 = auto()

"""Enumeration of blur modes"""
class BlurType(IntEnum):
    NONE = 0
    GAUSSIAN = auto()
    MEDIAN = auto()
    BOX = auto()
    BILATERAL = auto()

"""Enumeration of sharpening modes"""
class SharpenType(IntEnum):
    NONE = 0
    SHARPEN = auto()
    UNSHARP_MASK = auto()

"""Enum to access hsv tuple indicies"""
class HSV(IntEnum):
    H = 0
    S = 1
    V = 2

class ReflectionMode(Enum):
    """Enumeration for different image reflection modes."""

    NONE = 0  # No reflection
    HORIZONTAL = auto()  # Reflect across the Y-axis (flip horizontally)
    VERTICAL = auto() # Reflect across the X-axis (flip vertically)
    BOTH = auto()  # Reflect across both X and Y axes (flip horizontally and vertically)
    QUAD_SYMMETRY = auto()  # Reflect across both axes with quadrants (not implemented)
    SPLIT = auto()  # Reflect left half onto right half
    KALEIDOSCOPE = auto()

    def __str__(self):
        return self.name.replace("_", " ").title()


class Shape(IntEnum):

    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2
    LINE = 3
    DIAMOND = 4
    NONE = 5

"""
End Enum classes, begin effects classes
"""

""" 
EffectBase is a singleton base class to give all individual effects 
classes a common interface in the EffectsManager
"""
class EffectBase:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.name = cls.__name__
        return cls._instance

    def create_gui_panel(self):
        log.info(f"Creating GUI panel for {self.__name__}")


class EffectManager:
    """
    A central class to aggregate all singleton effect objects and simplify dependencies, arg len
    """
    def __init__(self):
        pass

    def init(self, params, toggles, width, height):
        """ This method is separate so the params and frame dims may be initialized"""
        self.params = params
        self.toggles = toggles
        
        self.feedback = Feedback(params, width, height)
        self.color = Color(params)
        self.pixels = Pixels(params, width, height)
        self.shapes = ShapeGenerator(params, width, height)
        self.patterns = Patterns(params, width, height)
        self.reflector = Reflector(params, )                    
        self.sync = Sync(params) 
        self.warp = Warp(params, width, height)
        self.glitch = Glitch(params, toggles)
        self.ptz = PTZ(params, width, height)

        self._all_services = [
            self.feedback,
            self.color,
            self.pixels,
            self.shapes,
            self.patterns,
            self.reflector,
            self.sync,
            self.warp,
            self.glitch,
            self.ptz,
        ]

        self.class_with_methods, self.all_methods = self.get_effect_methods()


    def get_effect_methods(self):
        """
        Collects all unique public method names from a list of effect objects.
        Any public method will be included in the list
        """

        class_with_methods = {}
        methods = []

        for obj in self._all_services:
            # Get all attributes of the current object
            all_attributes = dir(obj)
            
            # Filter for public methods and add them to the set
            public_methods = [
                getattr(obj, attr) for attr in all_attributes 
                if not attr.startswith('_') and 'create_gui_panel' not in attr and callable(getattr(obj, attr))
            ]
            methods+=public_methods
            class_with_methods[type(obj).__name__] = public_methods
        
        # Feedback should not be included in sequencer, so remove from structures here
        cleaned_methods = [m for m in methods if m not in class_with_methods['Feedback']]
        del class_with_methods['Feedback']

        return class_with_methods, cleaned_methods


    def adjust_sequence(self, from_idx, to_idx):
        self.all_methods.insert(to_idx, self.all_methods.pop(from_idx))

    """ 
    obsolete after implementing the effects sequencer.
    default sequence of effects for future reference; 
    """
    def default_effect_sequence(self, frame):
        frame = self.patterns.generate_pattern_frame(frame)
        frame = self.ptz.shift_frame(frame)
        frame = self.sync.sync(frame)
        frame = self.reflector.apply_reflection(frame)
        frame = self.color.polarize_frame_hsv(frame)
        frame = self.color.modify_hsv(frame)
        frame = self.color.adjust_brightness_contrast(frame)
        frame = self.noise.apply_noise(frame)
        frame = self.color.solarize_image(frame)
        frame = self.color.posterize(frame)
        frame = self.pixels.gaussian_blur(frame)
        frame = self.pixels.sharpen_frame(frame)
        frame = self.glitch.apply_glitch_effects(frame)        

        # TODO: test these effects, test ordering
        # frame = color limit_hues_kmeans(frame)
        # frame = fx.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = fx.apply_perlin_noise
        # warp_frame = fx.warp_frame(frame)

        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = fx.lissajous_pattern(frame, t)

        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

        return frame

    def apply_effects(self, frame, frame_count):
        """ 
        Applies a sequence of visual effects to the input frame based on current parameters.
        Each effect is modular and can be enabled/disabled via the GUI.
        The order of effects can be adjusted to achieve different visual styles.
        
        Returns the modified frame.
        """
        
        if frame_count % (self.params.val('frame_skip')+1) == 0: 
            
            # frame = self.default_effect_sequence(frame)

            for method in self.all_methods:
                frame = method(frame)
                # log.debug(F"{str(method.__name__)}  ->  {type(frame).__name__}")

        return frame

    
    def modify_frames(self, dry_frame, wet_frame, prev_frame, frame_count):

        # Blend the current dry frame with the previous wet frame using the alpha param
        if self.toggles.val("effects_first") == True:         
            wet_frame = self.apply_effects(wet_frame, frame_count)
            wet_frame = cv2.addWeighted(dry_frame, 1 - self.feedback.alpha.value, wet_frame, self.feedback.alpha.value, 0)
        else:
            wet_frame = cv2.addWeighted(dry_frame, 1 - self.feedback.alpha.value, wet_frame, self.feedback.alpha.value, 0)
            wet_frame = self.apply_effects(wet_frame, frame_count) 

        # Apply feedback effects
        wet_frame = self.feedback.apply_temporal_filter(prev_frame, wet_frame)
        wet_frame = self.feedback.avg_frame_buffer(wet_frame)
        wet_frame = self.feedback.nth_frame_feedback(wet_frame)
        wet_frame = self.feedback.apply_luma_feedback(prev_frame, wet_frame)
        prev_frame = wet_frame
        prev_frame = self.ptz._shift_prev_frame(prev_frame)
        # prev_frame = effects.feedback.scale_frame(wet_frame)

        return prev_frame, wet_frame


class Color(EffectBase):

    def __init__(self, params):
        self.params = params

        self.hue_shift = params.add("hue_shift", 0, 180, 0)
        self.sat_shift = params.add("sat_shift", 0, 255, 0)
        self.val_shift = params.add("val_shift", 0, 255, 0)

        self.levels_per_channel = params.add("posterize_levels", 0, 100, 0.0)
        self.num_hues = params.add("num_hues", 2, 10, 8)

        self.val_threshold = params.add("val_threshold", 0, 255, 0)
        self.val_hue_shift = params.add("val_hue_shift", 0, 255, 0)

        self.solarize_threshold = params.add("solarize_threshold", 0, 100, 0.0)
        self.hue_invert_angle = params.add("hue_invert_angle", 0, 360, 0)
        self.hue_invert_strength = params.add("hue_invert_strength", 0.0, 1.0, 0.0)

        self.contrast = params.add("contrast", 0.5, 3.0, 1.0)
        self.brightness = params.add("brightness", 0, 100, 0)

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
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v]

        # Apply the hue, saturation, and value shifts to the image.
        hsv = self._shift_hsv(hsv)
        hsv = self._val_threshold_hue_shift(hsv)

        # Merge the modified channels and convert back to BGR color space.
        return cv2.cvtColor(
            cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value])),
            cv2.COLOR_HSV2BGR,
        )

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
            log.debug("Warning: Image not in uint8 format. Converting...")
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
            log.debug("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)  # Converts to uint8, scales if needed

        frame = np.where(frame > self.solarize_threshold.value, 255 - frame, frame)

        return frame.astype(np.uint8)

    def adjust_brightness_contrast(self, image):
        """
        Adjusts the brightness and contrast of an image.

        Args:
            image: The input image (NumPy array).

        Returns:
            The adjusted image (NumPy array).
        """
        adjusted_image = cv2.convertScaleAbs(
            image, alpha=self.contrast.value, beta=self.brightness.value
        )
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
        shifted_hue = (
            hue_channel + hue_shift * self.hue_invert_strength.value
        ) % 180  # Wrap around
        hsv_frame[:, :, 0] = shifted_hue

        # Convert back to BGR
        polarized_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return polarized_frame

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):

        with dpg.collapsing_header(label=f"\tColor", tag="color") as h:
            dpg.bind_item_theme(h, theme)

            hue = TrackbarRow("Hue Shift", self.params["hue_shift"], default_font_id)

            sat = TrackbarRow(
                "Sat Shift", self.params.get("sat_shift"), default_font_id
            )

            val = TrackbarRow(
                "Val Shift", self.params.get("val_shift"), default_font_id
            )

            contrast = TrackbarRow(
                "Contrast", self.params.get("contrast"), default_font_id
            )

            brightness = TrackbarRow(
                "Brighness", self.params.get("brightness"), default_font_id
            )

            val_threshold = TrackbarRow(
                "Val Threshold", self.params.get("val_threshold"), default_font_id
            )

            val_hue_shift = TrackbarRow(
                "Hue Shift for Val", self.params.get("val_hue_shift"), default_font_id
            )

            hue_invert_angle = TrackbarRow(
                "Hue Invert Angle", self.params.get("hue_invert_angle"), default_font_id
            )

            hue_invert_strength = TrackbarRow(
                "Hue Invert Strength",
                self.params.get("hue_invert_strength"),
                default_font_id,
            )

            posterize = TrackbarRow(
                "Posterize Levels", self.params.get("posterize_levels"), default_font_id
            )

            solarize = TrackbarRow(
                "Solarize Threshold", self.params.get("solarize_threshold"), default_font_id
            )

        dpg.bind_item_font("color", global_font_id)


class Pixels(EffectBase):

    def __init__(self, params, image_width: int, image_height: int, noise_type=NoiseType.NONE):
        self.params = params
        self.image_width = image_width
        self.image_height = image_height

        self.sharpen_type = params.add(
            "sharpen_type",
            SharpenType.NONE.value,
            len(SharpenType),
            SharpenType.NONE.value,
        )
        self.sharpen_intensity = params.add("sharpen_intensity", 4.0, 8.0, 4.0)

        self.blur_type = params.add(
            "blur_type", 0, len(BlurType)-1, 1
        )
        self.blur_kernel_size = params.add("blur_kernel_size", 1, 100, 1)


        if not isinstance(noise_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")

        self._noise_type = params.add(
            "noise_type", NoiseType.NONE.value, NoiseType.RANDOM.value, noise_type.value
        )
        self._noise_intensity = params.add("noise_intensity", 0.0, 1.0, 0.1)

    @property
    def noise_type(self) -> NoiseType:
        """Get the current noise type."""
        return self._noise_type.value

    @noise_type.setter
    def noise_type(self, new_type: NoiseType):
        """Set the noise type."""
        if not isinstance(new_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")
        self._noise_type.value = new_type
        log.debug(f"Noise type set to: {self._noise_type.value}")

    @property
    def noise_intensity(self) -> float:
        """Get the current noise intensity."""
        return self._noise_intensity.value

    @noise_intensity.setter
    def noise_intensity(self, new_intensity: float):
        """Set the noise intensity."""
        if not (0.0 <= new_intensity <= 1.0):
            log.warning("Warning: noise_intensity should ideally be between 0.0 and 1.0.")
        self._noise_intensity.value = new_intensity
        log.debug(f"Noise intensity set to: {self._noise_intensity}")

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the currently set noise type to the input image.

        Args:
            image (np.ndarray): The input image (NumPy array).
                                Expected to be in BGR format for color images
                                or grayscale for single-channel images, with
                                pixel values typically 0-255 (uint8).

        Returns:
            np.ndarray: The image with noise applied.
        """
        # Ensure image is a float type for calculations, then convert back to uint8
        # Make a copy to avoid modifying the original image
        noisy_image = image.astype(np.float32)

        # Dispatch to the appropriate noise function based on noise_type
        if self._noise_type.value == NoiseType.NONE.value:
            return self._apply_none_noise(noisy_image)
        elif self._noise_type.value == NoiseType.GAUSSIAN.value:
            return self._apply_gaussian_noise(noisy_image)
        elif self._noise_type.value == NoiseType.POISSON.value:
            return self._apply_poisson_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SALT_AND_PEPPER.value:
            return self._apply_salt_and_pepper_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SPECKLE.value:
            return self._apply_speckle_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SPARSE.value:
            return self._apply_sparse_noise(noisy_image)
        elif self._noise_type.value == NoiseType.RANDOM.value:
            return self._apply_random_noise(noisy_image)
        else:
            log.warning(
                f"Unknown noise type: {self._noise_type.value}. Returning original image."
            )
            return image.copy()  # Return original if type is unknown

    def _apply_none_noise(self, image: np.ndarray) -> np.ndarray:
        """Returns the image without applying any noise."""
        return image.astype(np.uint8)

    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian (normal distribution) noise to the image.
        Intensity controls the standard deviation of the noise.
        """
        mean = 0
        # Standard deviation scales with intensity, up to a max of ~50 for uint8 range
        std_dev = self._noise_intensity * 50
        gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + gaussian_noise
        # Clip values to 0-255 range and convert back to uint8
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Poisson noise to the image.
        For typical 0-255 images, this is simulated by adding noise
        proportional to the pixel intensity.
        Intensity controls the scaling factor for the Poisson distribution.
        """
        # Scale to a range suitable for Poisson (e.g., 0-100), Then add noise and scale back
        scaled_image = (
            image / 255.0 * 100.0
        )  # Scale to 0-100 for better Poisson distribution
        poisson_noise = np.random.poisson(
            scaled_image * self._noise_intensity.value * 2
        ).astype(np.float32)
        noisy_image = image + (
            poisson_noise / 100.0 * 255.0
        )  # Scale noise back to 0-255 range
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_salt_and_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Salt & Pepper noise to the image.
        Intensity controls the proportion of pixels affected.
        """
        amount = self._noise_intensity  # Proportion of pixels to affect
        s_vs_p = 0.5  # Ratio of salt vs. pepper (0.5 means equal)

        # Apply salt noise (white pixels)
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        image[tuple(coords)] = 255

        # Apply pepper noise (black pixels)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        image[tuple(coords)] = 0
        return image.astype(np.uint8)

    def _apply_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Speckle (multiplicative) noise to the image.
        noisy_image = image + image * noise
        Intensity controls the standard deviation of the noise.
        """
        mean = 0
        std_dev = self._noise_intensity * 0.5  # Scale std_dev for multiplicative noise
        speckle_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + image * speckle_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_sparse_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies sparse noise by randomly selecting a percentage of pixels
        and setting them to a random value within the 0-255 range.
        Intensity controls the proportion of pixels affected.
        """
        amount = self._noise_intensity  # Proportion of pixels to affect

        num_pixels_to_affect = np.ceil(
            amount * image.size / image.shape[-1] if image.ndim == 3 else image.size
        ).astype(int)

        # Get image dimensions
        height, width = image.shape[0], image.shape[1]

        # Generate random (y, x) coordinates for the pixels to affect
        random_y = np.random.randint(0, height, num_pixels_to_affect)
        random_x = np.random.randint(0, width, num_pixels_to_affect)

        # Apply random values to the selected pixels
        for i in range(num_pixels_to_affect):
            y, x = random_y[i], random_x[i]
            if image.ndim == 3:  # Color image (e.g., BGR)
                # Assign a list of 3 random values to the (y, x) pixel
                image[y, x] = [random.randint(0, 255) for _ in range(image.shape[2])]
            else:  # Grayscale image
                # Assign a single random value to the (y, x) pixel
                image[y, x] = random.randint(0, 255)

        return image.astype(np.uint8)

    def _apply_random_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies uniform random noise to every pixel.
        Intensity controls the maximum range of the random noise added.
        """
        # Generate random noise in the range [-intensity*127.5, intensity*127.5]
        noise_range = (self._noise_intensity * 127.5)  
        random_noise = np.random.uniform(-noise_range, noise_range, image.shape).astype(
            np.float32
        )
        noisy_image = image + random_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def gaussian_blur(self, frame: np.ndarray):
        mode = self.blur_type.value
        if mode == BlurType.NONE:
            pass
        elif mode == BlurType.GAUSSIAN:
            # TODO: apply snapping to odd kernel size here rather than in the GUI callback, as other blur type *MAY* permit other sizes
            frame = cv2.GaussianBlur(
                frame, (self.blur_kernel_size.value, self.blur_kernel_size.value), 0
            )
        elif mode == BlurType.MEDIAN:
            frame = cv2.medianBlur(frame, self.blur_kernel_size.value)
        elif mode == BlurType.BOX:
            frame = cv2.blur(
                frame, (self.blur_kernel_size.value, self.blur_kernel_size.value)
            )
        elif mode == BlurType.BILATERAL:
            frame = cv2.bilateralFilter(frame, self.blur_kernel_size.value, 75, 75)

        return frame

    def sharpen_frame(self, frame: np.ndarray):

        if self.sharpen_intensity.value <= self.sharpen_intensity.min + 0.01:
            return frame

        sharpening_kernel = np.array(
            [[0, -1, 0], [-1, self.sharpen_intensity.value, -1], [0, -1, 0]]
        )

        # Apply the kernel to the frame using cv2.filter2D
        # -1 indicates that the output image will have the same depth (data type) as the input image.
        sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)

        return sharpened_frame
    

    def create_gui_panel(self, default_font_id, theme):

        with dpg.collapsing_header(label=f"\tPixels", tag="pixels") as h:
            dpg.bind_item_theme(h, theme)
            
            TrackbarRow(
                "Blur Kernel", 
                self.params.get("blur_kernel_size"), 
                default_font_id
            )

            RadioButtonRow(
                label="Blur Type", 
                cls=BlurType,
                param=self.params.get("blur_type"), 
                font=default_font_id
            )

            TrackbarRow(
                "Sharpen Amount", 
                self.params.get("sharpen_intensity"), 
                default_font_id
            )


            RadioButtonRow(
                "Noise Type", 
                NoiseType, 
                self.params.get("noise_type"), 
                default_font_id
            )

            noise_intensity = TrackbarRow(
                "Noise Intensity", 
                self.params.get("noise_intensity"), 
                default_font_id
            )


class Sync(EffectBase):

    def __init__(self, params):
        self.params = params
        self.x_sync_freq = params.add("x_sync_freq", 0.1, 100.0, 1.0)
        self.x_sync_amp = params.add("x_sync_amp", -200, 200, 0.0)
        self.x_sync_speed = params.add("x_sync_speed", 5.0, 10.0, 9.0)
        self.y_sync_freq = params.add("y_sync_freq", 0.1, 100.0, 1.0)
        self.y_sync_amp = params.add("y_sync_amp", -200, 200, 00.0)
        self.y_sync_speed = params.add("y_sync_speed", 5.0, 10.0, 9.0)

    def sync(self, frame: np.ndarray):
        """
        Applies a raster wobble effect to the frame using sine waves on both X and Y axes.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        warped = np.zeros_like(frame)

        # X-axis wobble (horizontal shift per row)
        for y in range(height):
            shift_x = int(
                self.x_sync_amp.value
                * np.sin(
                    y / self.x_sync_freq.value
                    + cv2.getTickCount() / (10**self.x_sync_speed.value)
                )
            )
            warped[y] = np.roll(frame[y], shift_x, axis=0)

        # Y-axis wobble (vertical shift per column)
        warped_y = np.zeros_like(warped)
        for x in range(width):
            shift_y = int(
                self.y_sync_amp.value
                * np.sin(
                    x / self.y_sync_freq.value
                    + cv2.getTickCount() / (10**self.y_sync_speed.value)
                )
            )
            warped_y[:, x] = np.roll(warped[:, x], shift_y, axis=0)

        warped_y = cv2.cvtColor(warped_y, cv2.COLOR_RGB2BGR)
        return warped_y

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\tSync", tag="sync") as h:
            dpg.bind_item_theme(h, theme)

            x_sync_speed = TrackbarRow(
                "X Sync Speed", self.params.get("x_sync_speed"), default_font_id
            )

            x_sync_freq = TrackbarRow(
                "X Sync Freq", self.params.get("x_sync_freq"), default_font_id
            )

            x_sync_amp = TrackbarRow(
                "X Sync Amp", self.params.get("x_sync_amp"), default_font_id
            )

            x_sync_speed = TrackbarRow(
                "Y Sync Speed", self.params.get("y_sync_speed"), default_font_id
            )

            x_sync_freq = TrackbarRow(
                "Y Sync Freq",
                self.params.get("y_sync_freq"),
                default_font_id,
            )

            x_sync_amp = TrackbarRow(
                "Y Sync Amp", self.params.get("y_sync_amp"), default_font_id
            )

        dpg.bind_item_font("sync", global_font_id)


PERLIN_SCALE=1700 #???
class Warp(EffectBase):

    def __init__(self, params, image_width: int, image_height: int):
        self.params = params
        self.width = image_width
        self.height = image_height
        self.warp_type = params.add(
            "warp_type", WarpType.NONE.value, WarpType.WARP0.value, WarpType.NONE.value
        )
        self.warp_angle_amt = params.add("warp_angle_amt", 0, 360, 30)
        self.warp_radius_amt = params.add("warp_radius_amt", 0, 360, 30)
        self.warp_speed = params.add("warp_speed", 0, 100, 10)
        self.warp_use_fractal = params.add("warp_use_fractal", 0, 1, 0)
        self.warp_octaves = params.add("warp_octaves", 1, 8, 4)
        self.warp_gain = params.add("warp_gain", 0.0, 1.0, 0.5)
        self.warp_lacunarity = params.add("warp_lacunarity", 1.0, 4.0, 2.0)
        self.x_speed = params.add("x_speed", 0.0, 100.0, 1.0)
        self.x_size = params.add("x_size", 0.25, 100.0, 20.0)
        self.y_speed = params.add("y_speed", 0.0, 10.0, 1.0)
        self.y_size = params.add("y_size", 0.0, 100.0, 10.0)

        self.t = 0

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

    def _generate_fractal_flow(
        self, t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity
    ):
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
                    noise_x += amplitude * pnoise2(
                        nx * frequency, ny * frequency, base=int(t)
                    )
                    noise_y += amplitude * pnoise2(
                        (nx + 1000) * frequency, (ny + 1000) * frequency, base=int(t)
                    )
                    amplitude *= gain
                    frequency *= lacunarity

                fx[y, x] = amp_x * noise_x
                fy[y, x] = amp_y * noise_y
        return fx, fy

    def _polar_warp(self, frame, t, angle_amt, radius_amt, speed):
        cx, cy = self.width / 2, self.height / 2
        y, x = np.indices((self.height, self.width), dtype=np.float32)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        a = np.arctan2(dy, dx)

        # Modify radius and angle
        r_mod = r + np.sin(a * 5 + t * speed) * radius_amt
        a_mod = a + np.cos(r * 0.02 + t * speed) * (angle_amt * np.pi / 180)

        # Back to Cartesian
        map_x = (r_mod * np.cos(a_mod) + cx).astype(np.float32)
        map_y = (r_mod * np.sin(a_mod) + cy).astype(np.float32)

        return cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _first_warp(self, frame: np.ndarray):
        # self.height, self.width = frame.shape[:2]
        # frame = cv2.resize(frame, (self.width, self.height))

        # Create meshgrid for warping effect
        x_indices, y_indices = np.meshgrid(
            np.arange(self.width), np.arange(self.height)
        )

        # Calculate warped indices using sine function
        time = cv2.getTickCount() / cv2.getTickFrequency()
        x_warp = x_indices + self.x_size.value * np.sin(
            y_indices / 20.0 + time * self.x_speed.value
        )
        y_warp = y_indices + self.y_size.value * np.sin(
            x_indices / 20.0 + time * self.y_speed.value
        )

        # Bound indices within valid range
        x_warp = np.clip(x_warp, 0, self.width - 1).astype(np.float32)
        y_warp = np.clip(y_warp, 0, self.height - 1).astype(np.float32)

        # Remap frame using warped indices
        frame = cv2.remap(
            frame, x_warp, y_warp, interpolation=cv2.INTER_LINEAR
        )

        return frame

    def warp(self, frame):
        self.t += 0.1

        if self.warp_type.value == WarpType.NONE:  # No warp
            return frame
        
        elif self.warp_type.value == WarpType.SINE:  # Sine warp
            fx = np.sin(np.linspace(0, np.pi * 2, self.width)[None, :] + self.t) * self.x_size.value
            fy = np.cos(np.linspace(0, np.pi * 2, self.height)[:, None] + self.t) * self.y_size.value
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        elif self.warp_type.value == WarpType.FRACTAL:
            if self.warp_use_fractal.value > 0: #TODO: CHANGE TO TOGGLE
                fx, fy = self._generate_fractal_flow(
                    self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value, octaves, gain, lacunarity
                )
            else:
                fx, fy = self._generate_fractal_flow(
                    self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value, 1, 1.0, 1.0
                )  # fallback: single octave
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            return cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        elif self.warp_type.value == WarpType.RADIAL:
            return self._polar_warp(frame, 
                                    self.t, 
                                    self.warp_angle_amt.value, 
                                    self.warp_radius_amt.value, 
                                    self.warp_speed.value)

        elif self.warp_type.value == WarpType.PERLIN:  # Perlin warp
            fx, fy = self._generate_perlin_flow(self.t, self.x_size.value, self.y_size.value, self.x_speed.value, self.y_speed.value)
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            map_x = (map_x + fx).astype(np.float32)
            map_y = (map_y + fy).astype(np.float32)
            f = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            return f
        elif self.warp_type.value == WarpType.WARP0:
            return self._first_warp(frame)
        else:
            return frame

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):

        with dpg.collapsing_header(label=f"\tWarp", tag="warp") as h:
            dpg.bind_item_theme(h, theme)

            RadioButtonRow(
                "Warp Type",
                WarpType,
                self.params.get("warp_type"),
                default_font_id,
            )

            TrackbarRow(
                "Warp Angle Amt", 
                self.params.get("warp_angle_amt"), 
                default_font_id
            )

            TrackbarRow(
                "Warp Radius Amt", 
                self.params.get("warp_radius_amt"), 
                default_font_id
            )

            TrackbarRow(
                "Warp Speed", 
                self.params.get("warp_speed"), 
                default_font_id
            )

            TrackbarRow(
                "Warp Use Fractal", 
                self.params.get("warp_use_fractal"), 
                default_font_id
            )

            TrackbarRow(
                "Warp Octaves", 
                self.params.get("warp_octaves"), 
                default_font_id
            )

            warp_gain = TrackbarRow(
                "Warp Gain", 
                self.params.get("warp_gain"), 
                default_font_id
            )

            warp_lacunarity = TrackbarRow(
                "Warp Lacunarity", 
                self.params.get("warp_lacunarity"), 
                default_font_id
            )

            x_speed = TrackbarRow(
                "X Speed", 
                self.params.get("x_speed"), default_font_id
            )

            y_speed = TrackbarRow(
                "Y Speed", 
                self.params.get("y_speed"), 
                default_font_id
            )

            x_size = TrackbarRow(
                "X Size", 
                self.params.get("x_size"), 
                default_font_id
            )

            y_size = TrackbarRow(
                "Y Size", 
                self.params.get("y_size"), 
                default_font_id
            )


class Reflector(EffectBase):
    """
    A class to apply reflection transformations to image frames from a stream.
    """

    def __init__(self, params, mode: ReflectionMode = ReflectionMode.NONE):
        """
        Initializes the Reflector with a specified reflection mode.

        Args:
            mode (ReflectionMode): The reflection mode to apply. Defaults to NONE.
        """
        self.params = params
        if not isinstance(mode, ReflectionMode):
            raise ValueError("mode must be an instance of ReflectionMode Enum.")
        self._mode = params.add(
            "reflection_mode", 0, len(ReflectionMode) - 1, ReflectionMode.NONE.value
        )
        self.width = None 
        self.height = None
        self.num_axis = 3

    @property
    def mode(self) -> ReflectionMode:
        """Get the current reflection mode."""
        return self._mode

    @mode.setter
    def mode(self, new_mode: ReflectionMode):
        """Set the reflection mode."""
        if not isinstance(new_mode, ReflectionMode):
            raise ValueError("new_mode must be an instance of ReflectionMode Enum.")
        self._mode = new_mode

    def apply_reflection(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the currently set reflection mode to the input image frame.

        Args:
            frame (np.ndarray): The input image frame (NumPy array).

        Returns:
            np.ndarray: The reflected image frame.
        """

        # init frame dims
        if self.width is None or self.height is None:
            self.height,self.width, _ = frame.shape

        if self._mode.value == ReflectionMode.NONE.value:
            return frame
        elif self._mode.value == ReflectionMode.HORIZONTAL.value:
            return cv2.flip(frame, 1)
        elif self._mode.value == ReflectionMode.VERTICAL.value:
            return cv2.flip(frame, 0)
        elif self._mode.value == ReflectionMode.BOTH.value:
            return cv2.flip(frame, -1)
        elif self._mode.value == ReflectionMode.QUAD_SYMMETRY.value:
            return self._apply_quad_symmetry(frame)
        elif (self._mode.value == ReflectionMode.SPLIT.value): 
            w_half = self.width // 2
            output_frame = frame.copy()
            left_half = frame[:, :w_half]
            # Reflect left half horizontally
            reflected_left = cv2.flip(left_half, 1)
            # Place reflected left half onto right half
            output_frame[:, w_half:] = reflected_left[:, : self.width - w_half]
            return output_frame
        elif self._mode.value == ReflectionMode.KALEIDOSCOPE.value:
            return self._apply_kaleidoscope(frame)


    def _apply_kaleidoscope(self, frame):
        """Applies the kaleidoscope effect to a single frame."""
        
        segments = self.segments.value
        zoom = self.zoom.value
        rotation = self.rotation.value
        center_x_rel = None
        center_y_rel = None
        border_mode = None

        # Calculate geometric parameters
        K = max(2, segments)
        center = (int(self.width * center_x_rel), int(self.height * center_y_rel))
        angle_offset = rotation
        
        # Map to Polar Coordinates (for tiling in angle)
        max_radius = min(self.width - center[0], center[0], self.height - center[1], center[1])
        polar_h = int(max_radius * zoom * 2) # Height (radius) scaled by zoom
        polar_w = 360 * 10 # Width (angle) in pixels (arbitrarily large for smooth rotation/tiling)

        # Use warpPolar to map the frame to polar coordinates
        polar_img = cv2.warpPolar(
            frame, (polar_w, polar_h), center, max_radius, 
            cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR
        )
        # flags: WARP_FILL_OUTLIERS fills unmapped pixels (like outside the circle) with black
        
        # Calculate the size of one segment (angular slice)
        tile_w = polar_w // K
        tile = polar_img[:, :tile_w]
        
        # Create the full tiled/mirrored image
        tiled_img = np.zeros((polar_h, polar_w, 3), dtype=frame.dtype)
        for i in range(K):
            # Mirror every other tile for true kaleidoscope effect
            t = cv2.flip(tile, 1) if i % 2 == 1 else tile
            # Place the tile
            tiled_img[:, i*tile_w:(i+1)*tile_w] = t

        # Apply rotation by shifting the image horizontally (wrap around)
        shift_pixels = int(polar_w * (angle_offset / 360.0))
        shifted_tiled_img = np.roll(tiled_img, shift_pixels, axis=1)

        # Map back to Cartesian Coordinates
        kaleido_frame = cv2.warpPolar(
            shifted_tiled_img, (self.width, self.height), center, max_radius, 
            cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )

        # Handle Border Modes for the 'background' (outside the circular kaleidoscope area)
        if border_mode == cv2.BORDER_WRAP:
            pass # For this simple example, we'll keep the effect centered.

        return kaleido_frame

    def _apply_quad_symmetry(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies a 4-quadrant kaleidoscope-like reflection to the image.
        It takes the top-left quadrant and reflects it to fill the other three.
        """
        height, width = frame.shape[:2]

        # Calculate half dimensions
        h_half = height // 2
        w_half = width // 2

        # Ensure dimensions are at least 2x2 for quadrants to exist
        if h_half == 0 or w_half == 0:
            log.debug("Warning: Image too small for quad symmetry. Returning original.")
            return frame.copy()

        # Extract the top-left quadrant
        top_left_quadrant = frame[
            0:h_half, 0:w_half
        ].copy()

        # Create reflected versions
        top_right_quadrant = cv2.flip(top_left_quadrant, 1)  # Flip horizontally
        bottom_left_quadrant = cv2.flip(top_left_quadrant, 0)  # Flip vertically
        bottom_right_quadrant = cv2.flip(
            top_left_quadrant, -1
        )  # Flip both horizontally and vertically

        # Create a new canvas to assemble the reflected parts
        output_frame = np.zeros_like(frame)

        # Place the quadrants into the output frame
        output_frame[0:h_half, 0:w_half] = top_left_quadrant
        output_frame[0:h_half, w_half:width] = top_right_quadrant
        output_frame[h_half:height, 0:w_half] = bottom_left_quadrant
        output_frame[h_half:height, w_half:width] = bottom_right_quadrant

        return output_frame

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\tReflector", tag="reflector") as h:
            dpg.bind_item_theme(h, theme)
            reflection_mode = TrackbarRow(
                "Reflection Mode", self.params.get("reflection_mode"), default_font_id
            )

            # reflection_strength = TrackbarRow(
            #     "Reflection Strength",
            #     self.params.get("reflection_strength"),
            #     default_font_id)

        dpg.bind_item_font("reflector", global_font_id)


class PTZ(EffectBase):
    def __init__(self, params, image_width: int, image_height: int):
        self.params = params
        self.height = image_height
        self.width = image_width

        self.x_shift = params.add(
            "x_shift", -image_width, image_width, 0, family="Pan"
        )  # min/max depends on image size
        self.y_shift = params.add(
            "y_shift", -image_height, image_height, 0, family="Pan"
        )  # min/max depends on image size
        self.zoom = params.add("zoom", 0.75, 3, 1.0, family="Pan")
        self.r_shift = params.add("r_shift", -360, 360, 0.0, family="Pan")
        
        
        self.prev_x_shift = params.add(
            "prev_x_shift", -image_width, image_width, 0, family="Pan"
        )  # min/max depends on image size
        self.prev_y_shift = params.add(
            "prev_y_shift", -image_height, image_height, 0, family="Pan"
        )  # min/max depends on image size
        self.prev_zoom = params.add("prev_zoom", 0.75, 3, 1.0, family="Pan")
        self.prev_r_shift = params.add("prev_r_shift", -360, 360, 0.0, family="Pan")

        self.prev_cx = params.add(
            "prev_cx", -image_width/2, image_width/2, 0, family="Pan"
        )
        self.prev_cy = params.add(
            "prev_cy", -image_height/2, image_height/2, 0, family="Pan"
        )
        self.polar_x = params.add("polar_x", -image_width // 2, image_width // 2, 0)
        self.polar_y = params.add("polar_y", -image_height // 2, image_height // 2, 0)
        self.polar_radius = params.add("polar_radius", 0.1, 100, 1.0)

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
        # (height, width) = frame.shape[:2]
        center = (self.width / 2, self.height / 2)

        # Create a new array with the same shape and data type as the original frame
        shifted_frame = np.zeros_like(frame)

        # Create the mapping arrays for the indices.
        x_map = (np.arange(self.width) - self.x_shift.value) % self.width
        y_map = (np.arange(self.height) - self.y_shift.value) % self.height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        # Use cv2.getRotationMatrix2D to get the rotation matrix
        M = cv2.getRotationMatrix2D(
            center, self.r_shift.value, self.zoom.value
        )  # 1.0 is the scale

        # Perform the rotation using cv2.warpAffine
        rotated_frame = cv2.warpAffine(shifted_frame, M, (self.width, self.height))

        return rotated_frame
    
    def _shift_prev_frame(self, frame: np.ndarray):
        '''DUPLICATE CODE :('''
        # (height, width) = frame.shape[:2]
        center = (self.width / 2 +self.prev_cx.value, self.height / 2+self.prev_cy.value)

        # Create a new array with the same shape and data type as the original frame
        shifted_frame = np.zeros_like(frame)

        # Create the mapping arrays for the indices.
        x_map = (np.arange(self.width) - self.prev_x_shift.value) % self.width
        y_map = (np.arange(self.height) - self.prev_y_shift.value) % self.height

        # Use advanced indexing to shift the entire image at once
        shifted_frame = frame[y_map[:, np.newaxis], x_map]

        # Use cv2.getRotationMatrix2D to get the rotation matrix
        M = cv2.getRotationMatrix2D(
            center, self.prev_r_shift.value, self.prev_zoom.value
        )  # 1.0 is the scale

        # Perform the rotation using cv2.warpAffine
        rotated_frame = cv2.warpAffine(shifted_frame, M, (self.width, self.height))

        return rotated_frame


    def _polar_transform(self, frame: np.ndarray):
        """
        Transforms an image with horizontal bars into an image with concentric circles
        using a polar coordinate transform.
        """
        height, width = frame.shape[:2]
        center = (width // 2 + self.polar_x.value, height // 2 + self.polar_y.value)
        max_radius = np.sqrt(
            (width // self.polar_radius.value) ** 2 + (height // self.polar_radius.value) ** 2
        )

        #    The flags parameter is important:
        #    cv2.INTER_LINEAR:  Bilinear interpolation (good quality)
        #    cv2.WARP_FILL_OUTLIERS:  Fills in any missing pixels
        #
        return cv2.warpPolar(
            frame,
            (width, height),  # Output size (can be different from input)
            center,
            max_radius,
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,  # or +WARP_POLAR_LOG
        )

    def _on_button_click(self, sender, app_data, user_data):
        log.info(f"Toggle clicked: {user_data}, {app_data}, {sender}")
        # Perform action based on button click
        if enable_polar_transform == True:
            enable_polar_transform = False
        else:
            enable_polar_transform = True

    def create_gui_panel(
        self, default_font_id=None, global_font_id=None, theme=None
    ):
        width=550
        height=600
        with dpg.collapsing_header(label=f"\tPan", tag="pan") as h:
            dpg.bind_item_theme(h, theme)
            x_shift = TrackbarRow(
                "X Shift", self.params.get("x_shift"), default_font_id
            )
            y_shift = TrackbarRow(
                "Y Shift", self.params.get("y_shift"), default_font_id
            )
            r_shift = TrackbarRow(
                "R Shift", self.params.get("r_shift"), default_font_id
            )
            zoom = TrackbarRow("Zoom", self.params.get("zoom"), default_font_id)

            prev_x_shift = TrackbarRow(
                "Prev X Shift", self.prev_x_shift, default_font_id
            )
            y_shift = TrackbarRow(
                "Prev Y Shift", self.prev_y_shift, default_font_id
            )
            r_shift = TrackbarRow(
                "Prev R Shift", self.prev_r_shift, default_font_id
            )
            zoom = TrackbarRow("Prev Zoom", self.prev_zoom, default_font_id)
            prev_center_x = TrackbarRow("prev_center_x", self.prev_cx, default_font_id)
            prev_center_y = TrackbarRow("prev_center_y", self.prev_cy, default_font_id)

    
            enable_polar_transform_button = Toggle(
                "Enable Polar Transform", "enable_polar_transform"
            )
            dpg.add_button(
                label=enable_polar_transform_button.label,
                tag="enable_polar_transform",
                callback=self._on_button_click,
                user_data=enable_polar_transform_button.tag,
                width=width,
            )
            dpg.bind_item_font(enable_polar_transform_button.tag, default_font_id)

            polar_x = TrackbarRow(
                "Polar Center X", self.params.get("polar_x"), default_font_id
            )
            polar_y = TrackbarRow(
                "Polar Center Y", self.params.get("polar_y"), default_font_id
            )
            polar_radius = TrackbarRow(
                "Polar radius", self.params.get("polar_radius"), default_font_id
            )

        dpg.bind_item_font("pan", global_font_id)


class Feedback(EffectBase):

    def __init__(self, params: ParamTable, image_width: int, image_height: int):
        self.params = params
        self.height = image_height
        self.width = image_width

        family = self.__class__.__name__

        self.frame_skip = params.add("frame_skip", 0, 10, 0, family)
        self.alpha = params.add("alpha", 0.0, 1.0, 0.0, family)
        self.temporal_filter = params.add("temporal_filter", 0, 1.0, 0.0, family)
        self.feedback_luma_threshold = params.add("feedback_luma_threshold", 0, 255, 0, family)
        self.luma_select_mode = params.add(
            "luma_select_mode", LumaMode.WHITE.value, LumaMode.BLACK.value, LumaMode.WHITE.value, family
        )
        self.buffer_select = params.add("buffer_frame_select", -1, 20, -1, family)
        self.buffer_frame_blend = params.add("buffer_frame_blend", 0.0, 1.0, 0.0, family)

        self.prev_frame_scale = params.add("prev_frame_scale", 90, 110, 100, family)

        self.max_buffer_size = 30
        self.buffer_size = params.add("buffer_size", 0, self.max_buffer_size, 0, family)
        # this should probably be initialized in reset() to avoid issues with reloading config
        self.frame_buffer = deque(maxlen=self.max_buffer_size)


    def scale_frame(self, frame):

        target_height, target_width, _ = frame.shape
        resized_frame = cv2.resize(frame, None, fx=self.prev_frame_scale.value, fy=self.prev_frame_scale.value)

        # Calculate the padding needed to center the resized frame
        res_h, res_w, _ = resized_frame.shape
        pad_h = (target_height - res_h) // 2
        pad_w = (target_width - res_w) // 2
        
        # Handle any potential remaining pixel due to integer division
        pad_h_extra = target_height - (res_h + 2 * pad_h)
        pad_w_extra = target_width - (res_w + 2 * pad_w)
        
        if res_h > target_height or res_w > target_width:
            return resized_frame[:target_height, :target_width]

        # Use the target dimensions for the canvas size
        padded_frame = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
        
        # Place the resized frame into the center of the padded canvas
        padded_frame[pad_h : pad_h + res_h, 
                    pad_w : pad_w + res_w] = resized_frame
                    
        # Apply single extra pixel padding if necessary
        if pad_h_extra > 0:
            padded_frame = padded_frame[pad_h_extra//2 : target_height - (pad_h_extra - pad_h_extra//2), :]
        if pad_w_extra > 0:
            padded_frame = padded_frame[:, pad_w_extra//2 : target_width - (pad_w_extra - pad_w_extra//2)]

        return padded_frame


    def nth_frame_feedback(self, frame):

        if self.buffer_select.value == -1 or len(self.frame_buffer) < self.buffer_select.value:
            return frame
                
        nth_frame = self.frame_buffer[self.buffer_select.value]
        
        frame = cv2.addWeighted(
            frame.astype(np.float32),
            1 - self.buffer_frame_blend.value,
            nth_frame.astype(np.float32),
            self.buffer_frame_blend.value,
            0,
        )       

        return cv2.convertScaleAbs(frame)


    def avg_frame_buffer(self, frame):

        # important: update buffer even if we aren't doing anything with it
        self.frame_buffer.append(frame.astype(np.float32))

        # averaging with single previous frame already accomplished, so return
        if self.buffer_size.value <= 1:
            return frame

        # If the buffer is not yet full, return the original frame.
        if len(self.frame_buffer) < self.buffer_size.value:
            log.debug(f"Buffering frames: {len(self.frame_buffer)}/{self.buffer_size.value}")
            return frame

        sliced_deque = np.array(self.frame_buffer)[:self.buffer_size.value]

        avg_frame = np.mean(sliced_deque, axis=0)


        # Convert the averaged frame back to the correct data type (uint8)
        return np.clip(avg_frame, 0, 255).astype(np.uint8)


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
        filtered_frame = cv2.addWeighted(
            current_frame_float,
            1 - self.temporal_filter.value,
            filtered_frame,
            self.temporal_filter.value,
            0,
        )

        # Convert back to uint8 for display
        return cv2.convertScaleAbs(filtered_frame)


    def apply_luma_feedback(self, prev_frame, cur_frame):
        # The mask will determine which parts of the current frame are kept
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        if self.luma_select_mode.value == LumaMode.BLACK.value:
            # Use THRESH_BINARY_INV to key out DARK areas (Luma is low)
            # Pixels with Luma < threshold become white (255) in the mask, meaning they are KEPT
            ret, mask = cv2.threshold(
                gray, self.feedback_luma_threshold.value, 255, cv2.THRESH_BINARY_INV
            )
        elif self.luma_select_mode.value == LumaMode.WHITE.value:
            ret, mask = cv2.threshold(
                gray, self.feedback_luma_threshold.value, 255, cv2.THRESH_BINARY
            )

        # Keep the keyed-out (bright) parts of the current frame
        fg = cv2.bitwise_and(cur_frame, cur_frame, mask=mask)

        # Invert the mask to find the areas *not* keyed out (the dark areas)
        mask_inv = cv2.bitwise_not(mask)

        # Use the inverted mask to "cut a hole" in the previous frame
        bg = cv2.bitwise_and(prev_frame, prev_frame, mask=mask_inv)

        # Combine the new foreground (fg) with the previous frame's background (bg)
        return cv2.add(fg, bg)


    # TODO: implement
    def apply_perlin_noise(
        self, frame, perlin_noise, amplitude=1.0, frequency=1.0, octaves=1
    ):
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
                noise[y, x] = (
                    perlin_noise([x / frequency, y / frequency], octaves=octaves)
                    * amplitude
                )

        # Normalize the noise to [0, 255]
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert the noise to a 3-channel image
        noise_colored = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

        # Blend the original frame with the noise
        noisy_frame = cv2.addWeighted(frame, 1.0, noise_colored, 0.5, 0)

        return noisy_frame


    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):

        with dpg.collapsing_header(label=f"\tFeedback", tag="effects") as h:
            dpg.bind_item_theme(h, theme)
            TrackbarRow(
                "Temporal Filter", 
                self.params.get("temporal_filter"), 
                default_font_id
            )

            TrackbarRow(
                "Feedback", 
                self.alpha, 
                default_font_id
            )

            TrackbarRow(
                "Luma Feedback Threshold",
                self.feedback_luma_threshold,
                default_font_id,
            )

            luma_mode = RadioButtonRow(
                "Luma Mode", LumaMode, self.luma_select_mode, default_font_id
            )

            frame_buffer_size = TrackbarRow(
                "Frame Buffer Size", self.buffer_size, default_font_id
            )

            frame_buffer_select = TrackbarRow(
                "Frame Buffer Frame Select", self.buffer_select, default_font_id
            )

            frame_select_feedback = TrackbarRow(
                "Frame Select Feedback", self.buffer_frame_blend, default_font_id
            )

            frame_skip = TrackbarRow(
                "Frame Skip", self.frame_skip, default_font_id
            )

            num_hues = TrackbarRow(
                "Num Hues", self.params.get("num_hues"), default_font_id
            )

        dpg.bind_item_font("effects", global_font_id)


class Lissajous(EffectBase):
    def __init__(self, params):
        self.params = params
        self.lissajous_A = params.add("lissajous_A", 0, 100, 50)
        self.lissajous_B = params.add("lissajous_B", 0, 100, 50)
        self.lissajous_a = params.add("lissajous_a", 0, 100, 50)
        self.lissajous_b = params.add("lissajous_b", 0, 100, 50)
        self.lissajous_delta = params.add("lissajous_delta", 0, 360, 0)
    
    # TODO: make this more engaging
    def lissajous_pattern(self, frame, t):
        center_x, center_y = self.width // 2, self.height // 2
        for i in range(1000):
            x = int(
                center_x
                + self.lissajous_A.value
                * math.sin(
                    self.lissajous_a.value * t
                    + i * 0.01
                    + self.lissajous_delta.value * math.pi / 180
                )
            )
            y = int(
                center_y
                + self.lissajous_B.value
                * math.sin(self.lissajous_b.value * t + i * 0.01)
            )
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        return frame
    
    def temp_create_gui_panel(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tLissajous", tag="Lissajous"):

            lissajous_A = TrackbarRow(
                "Lissajous A", self.params.get("lissajous_A"), default_font_id
            )

            lissajous_B = TrackbarRow(
                "Lissajous B", self.params.get("lissajous_B"), default_font_id
            )

            lissajous_a = TrackbarRow(
                "Lissajous a", self.params.get("lissajous_a"), default_font_id
            )

            lissajous_b = TrackbarRow(
                "Lissajous b", self.params.get("lissajous_b"), default_font_id
            )

            lissajous_delta = TrackbarRow(
                "Lissajous Delta", self.params.get("lissajous_delta"), default_font_id
            )

        dpg.bind_item_font("Lissajous", global_font_id)


class ImageNoiser(EffectBase):

    def __init__(
        self, params, noise_type: NoiseType = NoiseType.NONE
    ):
        """
        Initializes the ImageNoiser with a default noise type and intensity.

        Args:
            noise_type (NoiseType): The type of noise to apply. Defaults to NONE.
            noise_intensity (float): The intensity of the noise, typically between 0.0 and 1.0.
                                     Its interpretation varies by noise type. Defaults to 0.1.
        """
        self.params = params

        if not isinstance(noise_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")

        self._noise_type = params.add(
            "noise_type", NoiseType.NONE.value, NoiseType.RANDOM.value, noise_type.value
        )
        self._noise_intensity = params.add("noise_intensity", 0.0, 1.0, 0.1)

    @property
    def noise_type(self) -> NoiseType:
        """Get the current noise type."""
        return self._noise_type.value

    @noise_type.setter
    def noise_type(self, new_type: NoiseType):
        """Set the noise type."""
        if not isinstance(new_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")
        self._noise_type.value = new_type
        log.debug(f"Noise type set to: {self._noise_type.value}")

    @property
    def noise_intensity(self) -> float:
        """Get the current noise intensity."""
        return self._noise_intensity.value

    @noise_intensity.setter
    def noise_intensity(self, new_intensity: float):
        """Set the noise intensity."""
        if not (0.0 <= new_intensity <= 1.0):
            log.warning("Warning: noise_intensity should ideally be between 0.0 and 1.0.")
        self._noise_intensity.value = new_intensity
        log.debug(f"Noise intensity set to: {self._noise_intensity}")

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the currently set noise type to the input image.

        Args:
            image (np.ndarray): The input image (NumPy array).
                                Expected to be in BGR format for color images
                                or grayscale for single-channel images, with
                                pixel values typically 0-255 (uint8).

        Returns:
            np.ndarray: The image with noise applied.
        """
        # Ensure image is a float type for calculations, then convert back to uint8
        # Make a copy to avoid modifying the original image
        noisy_image = image.astype(np.float32)

        # Dispatch to the appropriate noise function based on noise_type
        if self._noise_type.value == NoiseType.NONE.value:
            return self._apply_none_noise(noisy_image)
        elif self._noise_type.value == NoiseType.GAUSSIAN.value:
            return self._apply_gaussian_noise(noisy_image)
        elif self._noise_type.value == NoiseType.POISSON.value:
            return self._apply_poisson_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SALT_AND_PEPPER.value:
            return self._apply_salt_and_pepper_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SPECKLE.value:
            return self._apply_speckle_noise(noisy_image)
        elif self._noise_type.value == NoiseType.SPARSE.value:
            return self._apply_sparse_noise(noisy_image)
        elif self._noise_type.value == NoiseType.RANDOM.value:
            return self._apply_random_noise(noisy_image)
        else:
            log.warning(
                f"Unknown noise type: {self._noise_type.value}. Returning original image."
            )
            return image.copy()  # Return original if type is unknown

    def _apply_none_noise(self, image: np.ndarray) -> np.ndarray:
        """Returns the image without applying any noise."""
        return image.astype(np.uint8)

    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian (normal distribution) noise to the image.
        Intensity controls the standard deviation of the noise.
        """
        mean = 0
        # Standard deviation scales with intensity, up to a max of ~50 for uint8 range
        std_dev = self._noise_intensity * 50
        gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + gaussian_noise
        # Clip values to 0-255 range and convert back to uint8
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Poisson noise to the image.
        For typical 0-255 images, this is simulated by adding noise
        proportional to the pixel intensity.
        Intensity controls the scaling factor for the Poisson distribution.
        """
        # Scale to a range suitable for Poisson (e.g., 0-100), Then add noise and scale back
        scaled_image = (
            image / 255.0 * 100.0
        )  # Scale to 0-100 for better Poisson distribution
        poisson_noise = np.random.poisson(
            scaled_image * self._noise_intensity.value * 2
        ).astype(np.float32)
        noisy_image = image + (
            poisson_noise / 100.0 * 255.0
        )  # Scale noise back to 0-255 range
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_salt_and_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Salt & Pepper noise to the image.
        Intensity controls the proportion of pixels affected.
        """
        amount = self._noise_intensity  # Proportion of pixels to affect
        s_vs_p = 0.5  # Ratio of salt vs. pepper (0.5 means equal)

        # Apply salt noise (white pixels)
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        image[tuple(coords)] = 255

        # Apply pepper noise (black pixels)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        image[tuple(coords)] = 0
        return image.astype(np.uint8)

    def _apply_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Speckle (multiplicative) noise to the image.
        noisy_image = image + image * noise
        Intensity controls the standard deviation of the noise.
        """
        mean = 0
        std_dev = self._noise_intensity * 0.5  # Scale std_dev for multiplicative noise
        speckle_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + image * speckle_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_sparse_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies sparse noise by randomly selecting a percentage of pixels
        and setting them to a random value within the 0-255 range.
        Intensity controls the proportion of pixels affected.
        """
        amount = self._noise_intensity  # Proportion of pixels to affect

        num_pixels_to_affect = np.ceil(
            amount * image.size / image.shape[-1] if image.ndim == 3 else image.size
        ).astype(int)

        # Get image dimensions
        height, width = image.shape[0], image.shape[1]

        # Generate random (y, x) coordinates for the pixels to affect
        random_y = np.random.randint(0, height, num_pixels_to_affect)
        random_x = np.random.randint(0, width, num_pixels_to_affect)

        # Apply random values to the selected pixels
        for i in range(num_pixels_to_affect):
            y, x = random_y[i], random_x[i]
            if image.ndim == 3:  # Color image (e.g., BGR)
                # Assign a list of 3 random values to the (y, x) pixel
                image[y, x] = [random.randint(0, 255) for _ in range(image.shape[2])]
            else:  # Grayscale image
                # Assign a single random value to the (y, x) pixel
                image[y, x] = random.randint(0, 255)

        return image.astype(np.uint8)

    def _apply_random_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies uniform random noise to every pixel.
        Intensity controls the maximum range of the random noise added.
        """
        # Generate random noise in the range [-intensity*127.5, intensity*127.5]
        noise_range = (self._noise_intensity * 127.5)  
        random_noise = np.random.uniform(-noise_range, noise_range, image.shape).astype(
            np.float32
        )
        noisy_image = image + random_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def create_gui_panel(self, default_font_id=None, global_font_id=None,theme=None):
        with dpg.collapsing_header(label=f"\tNoiser", tag="noiser") as h:
            dpg.bind_item_theme(h, theme)

            noise_type = RadioButtonRow(
                "Noise Type", NoiseType, self.params.get("noise_type"), default_font_id
            )

            noise_intensity = TrackbarRow(
                "Noise Intensity", self.params.get("noise_intensity"), default_font_id
            )

            # noise_speed = TrackbarRow(
            #     "Noise Speed",
            #     self.params.get("noise_speed"),
            #     default_font_id)

            # noise_freq_x = TrackbarRow(
            #     "Noise Freq X",
            #     self.params.get("noise_freq_x"),
            #     default_font_id)

            # noise_freq_y = TrackbarRow(
            #     "Noise Freq Y",
            #     self.params.get("noise_freq_y"),
            #     default_font_id)

        dpg.bind_item_font("noiser", global_font_id)


class Glitch(EffectBase):   

    def __init__(self, params, toggles):
        self.params = params
        self.glitch_phase_start_frame = 0
        self.glitch_cycle_start_frame = 0

        # State for horizontal scroll freeze glitch
        self.current_fixed_y_end = None
        self.current_growth_duration = None
        self.last_scroll_glitch_reset_frame = (
            0  # Marks when the fixed_y_end and growth_duration were last set
        )
        self.glitch_phase_start_frame_shift = 0
        self.glitch_phase_start_frame_color = 0
        self.glitch_phase_start_frame_block = 0

        self.enable_pixel_shift = toggles.add(
            "Enable Pixel Shift Glitch", "enable_pixel_shift", False
        )
        self.enable_color_split = toggles.add(
            "Enable Color Split Glitch", "enable_color_split", False
        )
        self.enable_block_corruption = toggles.add(
            "Enable Block Corruption Glitch", "enable_block_corruption", False
        )
        self.enable_random_rectangles = toggles.add(
            "Enable Random Rectangles Glitch", "enable_random_rectangles", False
        )
        self.enable_horizontal_scroll_freeze = toggles.add(
            "Enable Horizontal Scroll Freeze Glitch",
            "enable_horizontal_scroll_freeze",
            False,
        )

        self.glitch_duration_frames = params.add("glitch_duration_frames", 1, 300, 60)
        self.glitch_intensity_max = params.add("glitch_intensity_max", 0, 100, 50)
        self.glitch_block_size_max = params.add("glitch_block_size_max", 0, 200, 60)
        self.band_div = params.add("glitch_band_div", 1, 10, 5)
        self.num_glitches = params.add("num_glitches", 0, 100, 0)
        self.glitch_size = params.add("glitch_size", 1, 100, 0)

        self.frame_count = 0

    def _create_buttons(self, gui):
        dpg.add_button(
            label=self.enable_pixel_shift.label,
            callback=self.enable_pixel_shift.toggle,
            parent=gui,
        )
        dpg.add_button(
            label=self.enable_color_split.label,
            callback=self.enable_color_split.toggle,
            parent=gui,
        )
        dpg.add_button(
            label=self.enable_block_corruption.label,
            callback=self.enable_block_corruption.toggle,
            parent=gui,
        )
        dpg.add_button(
            label=self.enable_random_rectangles.label,
            callback=self.enable_random_rectangles.toggle,
            parent=gui,
        )
        dpg.add_button(
            label=self.enable_horizontal_scroll_freeze.label,
            callback=self.enable_horizontal_scroll_freeze.toggle,
            parent=gui,
        )

    def _reset(self):
        self.glitch_phase_start_frame = 0
        self.glitch_cycle_start_frame = 0

    # TODO: implement, rename
    def _glitch_image(self, image: np.ndarray):
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

    def _apply_evolving_pixel_shift(self, frame, frame_num, glitch_phase_start_frame):
        height, width, _ = frame.shape
        current_phase_frame = (
            frame_num - glitch_phase_start_frame
        ) % self.glitch_duration_frames.value

        # Calculate shift amount and direction based on current_phase_frame
        # Use a sinusoidal or linear progression for smooth evolution
        progress = current_phase_frame / self.glitch_duration_frames.value
        shift_amount_x = int(
            self.glitch_intensity_max.value * np.sin(progress * np.pi * 2)
        )  # oscillates
        shift_amount_y = int(
            self.glitch_intensity_max.value * np.cos(progress * np.pi * 2)
        )

        glitched_frame = np.copy(frame)

        # Apply shift to different horizontal/vertical bands
        band_height = height // self.band_div.value
        band_width = width // self.band_div.value

        for i in range(self.band_div.value):
            y_start = i * band_height
            y_end = min((i + 1) * band_height, height)

            # Shift horizontal bands
            if i % 2 == 0:  # Even bands shift right
                shifted_band = np.roll(frame[y_start:y_end, :], shift_amount_x, axis=1)
            else:  # Odd bands shift left
                shifted_band = np.roll(frame[y_start:y_end, :], -shift_amount_x, axis=1)
            glitched_frame[y_start:y_end, :] = shifted_band

        for i in range(self.band_div.value):
            x_start = i * band_width
            x_end = min((i + 1) * band_width, width)

            # Shift vertical bands
            if i % 2 == 0:  # Even bands shift down
                shifted_band = np.roll(
                    glitched_frame[:, x_start:x_end], shift_amount_y, axis=0
                )
            else:  # Odd bands shift up
                shifted_band = np.roll(
                    glitched_frame[:, x_start:x_end], -shift_amount_y, axis=0
                )
            glitched_frame[:, x_start:x_end] = shifted_band

        return glitched_frame

    def _apply_gradual_color_split(self, frame, frame_num, glitch_phase_start_frame):
        """
        Gradually separates and then merges color channels.
        Optimized: Uses cv2.split and np.roll which are fast matrix operations.
        """
        height, width, _ = frame.shape
        current_phase_frame = (
            frame_num - glitch_phase_start_frame
        ) % self.glitch_duration_frames.value

        # Calculate separation amount
        # Increases then decreases over the phase
        progress = current_phase_frame / self.glitch_duration_frames.value
        separation_amount = int(
            self.glitch_intensity_max.value * np.sin(progress * np.pi)
        )  # Peaks at mid-phase

        b, g, r = cv2.split(frame)
        glitched_frame = np.zeros_like(frame)

        # Shift channels independently
        # Red channel shifts right, Green stays, Blue channel shifts left
        glitched_frame[:, :, 2] = np.roll(r, separation_amount, axis=1)  # Red
        glitched_frame[:, :, 1] = g  # Green (no shift)
        glitched_frame[:, :, 0] = np.roll(b, -separation_amount, axis=1)  # Blue

        return glitched_frame

    def _apply_morphing_block_corruption(
        self, frame, frame_num, glitch_phase_start_frame
    ):
        """
        Corrupts random blocks, with the size and number of corrupted blocks
        morphing over time.
        Optimized: Operations within each block are vectorized. The loop is for
        applying effects to multiple distinct random blocks.
        """
        height, width, _ = frame.shape
        current_phase_frame = (
            frame_num - glitch_phase_start_frame
        ) % self.glitch_duration_frames.value

        glitched_frame = np.copy(frame)

        # Calculate corruption intensity and block size based on progress
        progress = current_phase_frame / self.glitch_duration_frames.value
        # Intensity: starts low, peaks, then goes low again
        corruption_intensity = int(255 * np.sin(progress * np.pi))
        # Block size: starts small, grows, then shrinks
        current_block_size = int(
            self.glitch_block_size_max.value * np.sin(progress * np.pi)
        )
        if current_block_size < 10:  # Ensure minimum block size
            current_block_size = 10

        # Number of blocks to corrupt: more blocks when intensity is high
        num_blocks = int(10 + 20 * np.sin(progress * np.pi))

        for _ in range(num_blocks):
            # Random top-left corner for the block
            # Ensure block fits within frame
            x = random.randint(0, max(0, width - current_block_size))
            y = random.randint(0, max(0, height - current_block_size))

            # Get the region of interest
            # Ensure slice does not go out of bounds
            x_end = min(x + current_block_size, width)
            y_end = min(y + current_block_size, height)

            # Only proceed if the block has valid dimensions
            if x_end > x and y_end > y:
                roi = glitched_frame[y:y_end, x:x_end]

                # Apply different corruption methods based on a random choice
                corruption_type = random.choice(["solid_color", "noise", "pixelate"])

                if corruption_type == "solid_color":
                    # Fill with a random color based on intensity
                    color = [
                        random.randint(0, corruption_intensity),
                        random.randint(0, corruption_intensity),
                        random.randint(0, corruption_intensity),
                    ]
                    roi[:, :] = color
                elif corruption_type == "noise":
                    # Add random noise. Ensure noise shape matches ROI.
                    # Ensure the 'high' value for randint is always greater than 'low'.
                    # If corruption_intensity // 2 is 0, make the upper bound 1 to avoid ValueError.
                    low_bound = -corruption_intensity // 2
                    high_bound = corruption_intensity // 2
                    if (
                        high_bound <= low_bound
                    ):  # This happens if corruption_intensity is 0 or 1
                        high_bound = low_bound + 1  # Ensure high > low

                    noise = np.random.randint(
                        low_bound, high_bound, roi.shape, dtype=np.int16
                    )
                    # Apply noise and clip values to 0-255 range
                    roi_with_noise = np.clip(
                        roi.astype(np.int16) + noise, 0, 255
                    ).astype(np.uint8)
                    glitched_frame[y:y_end, x:x_end] = roi_with_noise
                elif corruption_type == "pixelate":
                    # Simple pixelation by resizing down and up
                    # Ensure current_block_size is large enough for pixelation
                    if current_block_size > 5:
                        # Calculate new dimensions for pixelation (e.g., divide by 5)
                        pixel_width = max(1, (x_end - x) // 5)
                        pixel_height = max(1, (y_end - y) // 5)

                        small_roi = cv2.resize(
                            roi,
                            (pixel_width, pixel_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        pixelated_roi = cv2.resize(
                            small_roi,
                            (x_end - x, y_end - y),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        glitched_frame[y:y_end, x:x_end] = pixelated_roi

        return glitched_frame

    def _apply_horizontal_scroll_freeze_glitch(
        self, frame, frame_num, glitch_cycle_start_frame, fixed_y_end, growth_duration
    ):
        """
        Applies a horizontal scrolling glitch effect that freezes and distorts a band.
        The bottom of the bar is fixed, and the top grows for a random duration.
        Each row within the bar has a random side-to-side shift.
        """
        height, width, _ = frame.shape
        glitched_frame = np.copy(frame)

        # Calculate current progress within the growth cycle
        current_frame_in_cycle = frame_num - glitch_cycle_start_frame

        # Growth progress, clamped to 1.0
        growth_progress = min(1.0, current_frame_in_cycle / growth_duration)

        # Determine the glitch band height based on growth progress
        # Starts small (e.g., 10 pixels) and grows up to self.glitch_block_size_max.value (60)
        glitch_band_height = int(
            10 + (self.glitch_block_size_max.value - 10) * growth_progress
        )
        glitch_band_height = max(1, glitch_band_height)  # Ensure minimum height of 1

        # The bottom of the bar is fixed at fixed_y_end
        y_end = fixed_y_end
        # The top of the bar moves upwards as it grows
        y_start = max(0, y_end - glitch_band_height)

        # Ensure y_start and y_end are valid indices
        y_start = min(y_start, height - 1)
        y_end = min(y_end, height)
        if (
            y_start >= y_end
        ):  # Handle cases where band becomes invalid (e.g., too small or out of bounds)
            return glitched_frame  # Return original if band is invalid

        # Get the ROI from the original frame (or glitched_frame if applying sequentially)
        roi = glitched_frame[y_start:y_end, :]

        # Apply random horizontal shift to each row within the band
        # The maximum random shift scales with growth progress
        max_row_shift = int(self.glitch_intensity_max.value * growth_progress)
        if max_row_shift < 1:  # Ensure a minimum shift range
            max_row_shift = 1

        randomly_shifted_roi = np.copy(roi)
        for r_idx in range(roi.shape[0]):  # Iterate through each row in the ROI
            row_shift = random.randint(-max_row_shift, max_row_shift)
            randomly_shifted_roi[r_idx, :] = np.roll(
                roi[r_idx, :], row_shift, axis=0
            )  # axis=0 for 1D array roll

        # Now, use this randomly_shifted_roi for further processing (noise, etc.)
        shifted_roi = randomly_shifted_roi

        # Add some random noise to the shifted band for more distortion
        noise_intensity = int(
            100 * np.sin(growth_progress * np.pi)
        )  # Noise intensity varies

        low_bound_noise = -noise_intensity // 2
        high_bound_noise = noise_intensity // 2
        if high_bound_noise <= low_bound_noise:
            high_bound_noise = low_bound_noise + 1

        noise = np.random.randint(
            low_bound_noise, high_bound_noise, shifted_roi.shape, dtype=np.int16
        )
        corrupted_roi = np.clip(shifted_roi.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        # Replace the band in the glitched frame with the corrupted ROI
        glitched_frame[y_start:y_end, :] = corrupted_roi

        return glitched_frame

    def scanline_distortion(self, frame):
        return frame

    def apply_glitch_effects(self, frame):

        # Get frame dimensions for dynamic calculations
        height, width, _ = frame.shape
        
        # Reset phase start frame if a new phase begins for existing effects
        if self.frame_count % self.glitch_duration_frames.value == 0:
            self.glitch_phase_start_frame_shift = self.frame_count
        if (
            self.frame_count % (self.glitch_duration_frames.value * 1.5) == 0
        ):  # Slightly different cycle for color
            self.glitch_phase_start_frame_color = self.frame_count
        if (
            self.frame_count % (self.glitch_duration_frames.value * 0.75) == 0
        ):  # Faster cycle for blocks
            self.glitch_phase_start_frame_blocks = self.frame_count

        # Check if it's time to reset the horizontal scroll glitch state
        # This will happen on the very first frame or when the current growth duration is over
        if (
            self.frame_count == 0
            or (self.frame_count - self.last_scroll_glitch_reset_frame)
            >= self.current_growth_duration
        ):
            self.last_scroll_glitch_reset_frame = self.frame_count
            self.current_fixed_y_end = random.randint(height // 4, height)
            # Choose a random duration for the growth phase
            self.current_growth_duration = random.randint(
                self.glitch_duration_frames.value // 2,
                self.glitch_duration_frames.value * 2,
            )  # Random period for growth

        # Apply effects to the current frame
        if self.enable_random_rectangles.value:
            frame = self._glitch_image(frame)
        if self.enable_pixel_shift.value:
            frame = self._apply_evolving_pixel_shift(
                frame, self.frame_count, self.glitch_phase_start_frame_shift
            )
        if self.enable_color_split.value:
            frame = self._apply_gradual_color_split(
                frame, self.frame_count, self.glitch_phase_start_frame_color
            )
        if self.enable_block_corruption.value:
            frame = self._apply_morphing_block_corruption(
                frame, self.frame_count, self.glitch_phase_start_frame_blocks
            )
        if self.enable_horizontal_scroll_freeze.value:
            frame = self._apply_horizontal_scroll_freeze_glitch(
                frame,
                self.frame_count,
                self.last_scroll_glitch_reset_frame,
                self.current_fixed_y_end,
                self.current_growth_duration,
            )

        self.frame_count += 1
        return frame

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\tGlitch", tag="glitch") as h:
            dpg.bind_item_theme(h, theme)
            glitch_intensity = TrackbarRow(
                "Glitch Intensity", self.params.get("glitch_intensity_max"), default_font_id
            )

            glitch_duration = TrackbarRow(
                "Glitch Duration", self.params.get("glitch_duration_frames"), default_font_id
            )

            glitch_block_size = TrackbarRow(
                "Glitch Block Size",
                self.params.get("glitch_block_size_max"),
                default_font_id,
            )

            glitch_band_div = TrackbarRow(
                "Glitch Band Divisor", self.params.get("glitch_band_div"), default_font_id
            )

            num_glitches = TrackbarRow(
                "Glitch Qty", self.params.get("num_glitches"), default_font_id
            )

            glitch_size = TrackbarRow(
                "Glitch Size", self.params.get("glitch_size"), default_font_id
            )

            self._create_buttons("glitch")

        dpg.bind_item_font("glitch", global_font_id)


class ShapeGenerator:

    def __init__(self, params, width, height, shape_x_shift=0, shape_y_shift=0):
        self.params = params
        self.shape_x_shift = params.add("shape_x_shift", -width, width, shape_x_shift)  # Allow negative shifts
        self.shape_y_shift = params.add("shape_y_shift", -height, height, shape_y_shift)
        self.center_x = width // 2
        self.center_y = height // 2
        self.width = width
        self.height = height
        
        self.shape_type = params.add("shape_type", 0, len(Shape)-1, Shape.RECTANGLE)
        
        self.line_h = params.add("line_hue", 0, 179, 0)  # Hue range for OpenCV is 0-
        self.line_s = params.add("line_sat", 0, 255, 255)  # Saturation range
        self.line_v = params.add("line_val", 0, 255, 255)  # Value range

        self.line_hsv = [params.val("line_hue"), params.val("line_val"), params.val("line_sat")]  # H, S, V (Red) - will be converted to BGR
        self.line_weight = params.add("line_weight", 1, 20, 2)  # Thickness of the shape outline, must be integer
        self.line_opacity = params.add("line_opacity", 0.0, 1.0, 0.66)  # Opacity of the shape outline
        
        self.size_multiplier = params.add("size_multiplier", 0.1, 10.0, 0.9)  # Scale factor for shape size
        self.aspect_ratio = params.add("aspect_ratio", 0.1, 10.0, 1.0)  # Scale factor for shape size
        self.rotation_angle = params.add("rotation_angle", 0, 360, 0)  # Rotation angle in degrees
        
        self.multiply_grid_x = params.add("multiply_grid_x", 1, 10, 2)  # Number of shapes in X direction
        self.multiply_grid_y = params.add("multiply_grid_y", 1, 10, 2)  # Number of shapes in Y direction
        self.grid_pitch_x = params.add("grid_pitch_x", min=0, max=width, default_val=100)  # Distance between shapes in X direction
        self.grid_pitch_y = params.add("grid_pitch_y", min=0, max=height, default_val=100)  # Distance between shapes in Y direction
        
        self.fill_enabled = True  # Toggle fill on/off
        self.fill_h = params.add("fill_hue", 0, 179, 120)  # Hue for fill color
        self.fill_s = params.add("fill_sat", 0, 255, 100)  # Saturation for fill color
        self.fill_v = params.add("fill_val", 0, 255, 255)  # Value for fill color
        self.fill_hsv = [self.fill_h.value, self.fill_s.value, self.fill_v.value]  # H, S, V (Blue) - will be converted to BGR
        self.fill_opacity = params.add("fill_opacity", 0.0, 1.0, 0.25)
        self.fill_color = self._hsv_to_bgr(self.fill_hsv)
        self.line_color = self._hsv_to_bgr(self.line_hsv)

        self.convas_rotation = params.add("canvas_rotation", 0, 360, 0)  # Rotation angle in degrees
        
        

    def _draw_rectangle(self, canvas, center_x, center_y,):
        """ Draw a rotated rectangle on the canvas """

        rect_width = int(50 * self.size_multiplier.value * self.aspect_ratio.value)
        rect_height = int(50 * self.size_multiplier.value)

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.value, 1)

        # Define the rectangle's corners before rotation
        pts = np.array([
            [center_x - rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y + rect_height // 2],
            [center_x - rect_width // 2, center_y + rect_height // 2]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Apply the rotation
        rotated_pts = cv2.transform(pts, M)
        rotated_pts_int = np.int32(rotated_pts)

        if self.fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], self.fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.value)
        
        return canvas

    def _draw_circle(self, canvas, center_x, center_y):
        """ Draw a circle on the canvas """

        radius = int(30 * self.size_multiplier.value)
        if self.fill_enabled:
            cv2.circle(canvas, (center_x, center_y), radius, self.fill_color, -1) # -1 for fill
        cv2.circle(canvas, (center_x, center_y), radius, self.line_color, self.line_weight.value)

        return canvas
    
    def _draw_triangle(self, canvas, center_x, center_y):
        """ Draw a rotated triangle on the canvas """

        side_length = int(60 * self.size_multiplier.value)

        # Vertices for an equilateral triangle centered at (0,0)
        p1_x = 0
        p1_y = -side_length // 2
        p2_x = -int(side_length * np.sqrt(3) / 4)
        p2_y = side_length // 4
        p3_x = int(side_length * np.sqrt(3) / 4)
        p3_y = side_length // 4

        pts = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]], dtype=np.float32)

        # Translate to the current center
        pts[:, 0] += center_x
        pts[:, 1] += center_y

        # Apply rotation
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.value, 1)
        rotated_pts = cv2.transform(pts.reshape(-1, 1, 2), M)
        rotated_pts_int = np.int32(rotated_pts)

        if self.fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], self.fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.value)

        return canvas
    
    def _draw_line(self, canvas, center_x, center_y):
        length = int(50 * self.size_multiplier.value)
        start_point = (center_x - length // 2, center_y)
        end_point = (center_x + length // 2, center_y)

        if self.fill_enabled:
            cv2.line(canvas, start_point, end_point, self.fill_color, self.line_weight.value)
        cv2.line(canvas, start_point, end_point, self.line_color, self.line_weight.value)

        return canvas

    def _draw_shape_on_canvas(self, canvas, center_x, center_y):
        """ Draw the selected shape on the canvas at the specified center coordinates """

        # Ensure coordinates are within bounds to prevent errors
        # Note: These checks are for safety but may clip shapes if they go way off screen
        center_x = max(0, min(canvas.shape[1], center_x))
        center_y = max(0, min(canvas.shape[0], center_y))

        if self.shape_type.value == Shape.NONE:
            pass
        elif self.shape_type.value == Shape.RECTANGLE:
            canvas = self._draw_rectangle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.CIRCLE:
            canvas = self._draw_circle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.TRIANGLE:
            canvas = self._draw_triangle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.LINE:
            canvas = self._draw_line(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.DIAMOND:
            pass
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type.value}. Must be 'rectangle', 'circle', 'triangle', or 'line'.")
        
        return canvas

    def _blend_rgba_overlay(self, background, overlay_rgba):
        """
        Blend a 4-channel BGRA overlay onto a 3-channel BGR background using the alpha channel.
        Args:
            background (np.ndarray): 3-channel BGR background image.
            overlay_rgba (np.ndarray): 4-channel BGRA overlay image.
        Returns:
            np.ndarray: Blended 3-channel BGR image.
        """
        # Ensure background is float for calculation, then convert back to uint8
        background_float = background.astype(np.float32)

        # Normalize alpha channel (0-255 to 0.0-1.0)
        alpha = overlay_rgba[:, :, 3] / 255.0
        alpha_rgb = np.stack([alpha, alpha, alpha], axis=2) # Convert alpha to 3 channels for element-wise multiplication

        # Blend calculation: (foreground * alpha) + (background * (1 - alpha))
        # Note: overlay_rgba[:,:,:3] extracts the BGR channels of the overlay
        blended_image = (overlay_rgba[:,:,:3].astype(np.float32) * alpha_rgb) + \
                        (background_float * (1.0 - alpha_rgb))

        # Clip values to 0-255 and convert back to uint8
        return np.clip(blended_image, 0, 255).astype(np.uint8)

    # TODO: create a common HSV to BGR conversion function
    def _hsv_to_bgr(self, hsv):
        """ Convert HSV color to BGR color for OpenCV drawing functions """

        hsv_np = np.uint8([[hsv]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]

        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    
    # TODO: make public for testing or after fix
    # TODO: fix bug where shape hue affects the entire frame hue
    def _draw_shapes_on_frame(self, frame):
        """ Draw shapes on the given frame based on current parameters """

        base_center_x, base_center_y = self.width // 2 + self.shape_x_shift.value, self.height // 2 + self.shape_y_shift.value

        # Create separate 3-channel (BGR) canvases for lines and fills
        temp_line_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        temp_fill_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.line_color = self._hsv_to_bgr([self.line_h.value, self.line_s.value, self.line_v.value])
        self.fill_color = self._hsv_to_bgr([self.fill_h.value, self.fill_s.value, self.fill_v.value])

        # Grid Multiplication Mode
        for row in range(self.multiply_grid_y.value):
            for col in range(self.multiply_grid_x.value):
                # Calculate center for each shape in the grid
                current_center_x = base_center_x + col * self.grid_pitch_x.value - (self.multiply_grid_x.value - 1) * self.grid_pitch_x.value // 2
                current_center_y = base_center_y + row * self.grid_pitch_y.value - (self.multiply_grid_y.value - 1) * self.grid_pitch_y.value // 2

                self._draw_shape_on_canvas(temp_line_canvas, current_center_x, current_center_y)

            # If fill is enabled, apply its opacity in a separate blend
            if self.fill_enabled:
                self._draw_shape_on_canvas(temp_fill_canvas, current_center_x, current_center_y)

        # Line Overlay (BGRA)
        line_overlay_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        line_overlay_rgba[:,:,:3] = temp_line_canvas # Copy BGR data
        # Create an alpha mask: pixels are opaque (255) where sum of BGR is > 0 (not black)
        # Then scale by line_opacity
        line_alpha_mask = (temp_line_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.line_opacity * 255)
        line_overlay_rgba[:,:,3] = line_alpha_mask

        # Fill Overlay (BGRA)
        if self.fill_enabled:
            fill_overlay_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            fill_overlay_rgba[:,:,:3] = temp_fill_canvas # Copy BGR data
            # Create an alpha mask for fills
            fill_alpha_mask = (temp_fill_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.fill_opacity * 255)
            fill_overlay_rgba[:,:,3] = fill_alpha_mask

        # blend the layers using our custom self._blend_rgba_overlay method
        if self.fill_enabled:
            frame = self._blend_rgba_overlay(frame, fill_overlay_rgba)

        frame = self._blend_rgba_overlay(frame, line_overlay_rgba)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        return frame
    

    def create_gui_panel(self, default_font_id=None, global_font_id=None, theme=None):
        with dpg.collapsing_header(label=f"\tShape Generator", tag="shape_generator") as h:
            dpg.bind_item_theme(h, theme)
            shape = TrackbarRow("Shape Type",self.shape_type, default_font_id)
            
            canvas_rotation = TrackbarRow(
                "Canvas Rotation", self.convas_rotation, default_font_id
            )

            size_multiplier = TrackbarRow(
                "Size Multiplier", self.size_multiplier, default_font_id
            )
            
            aspect_ratio = TrackbarRow(
                "Aspect Ratio", self.aspect_ratio, default_font_id
            )
            
            rotation = TrackbarRow(
                "Rotation", self.rotation_angle, default_font_id
            )
            
            multiply_grid_x = TrackbarRow(
                "Multiply Grid X", self.multiply_grid_x, default_font_id
            )
            
            multiply_grid_y = TrackbarRow(
                "Multiply Grid Y", self.multiply_grid_y, default_font_id
            )
            
            grid_pitch_x = TrackbarRow(
                "Grid Pitch X", 
                self.params.get("grid_pitch_x"), 
                default_font_id)
            
            grid_pitch_y = TrackbarRow(
                "Grid Pitch Y", 
                self.params.get("grid_pitch_y"), 
                default_font_id)
            
            shape_y_shift = TrackbarRow(
                "Shape Y Shift", 
                self.params.get("shape_y_shift"), 
                default_font_id)
            
            shape_x_shift = TrackbarRow(
                "Shape X Shift", 
                self.params.get("shape_x_shift"), 
                default_font_id)

            with dpg.collapsing_header(label=f"\tLine Generator", tag="line_generator"):

                line_hue = TrackbarRow("Line Hue", self.params.get("line_hue"), default_font_id)
                
                line_sat = TrackbarRow(
                    "Line Sat", 
                    self.params.get("line_sat"), 
                        default_font_id)
                
                line_val = TrackbarRow(
                    "Line Val", 
                    self.params.get("line_val"), 
                        default_font_id)
                
                line_weight = TrackbarRow(
                    "Line Width", 
                    self.params.get("line_weight"), 
                        default_font_id)
                
                line_opacity = TrackbarRow(
                    "Line Opacity", 
                    self.params.get("line_opacity"), 
                        default_font_id)
            dpg.bind_item_font("line_generator", global_font_id)

            with dpg.collapsing_header(label=f"\tFill Generator", tag="fill_generator"):
                fill_hue = TrackbarRow(
                    "Fill Hue", 
                    self.params.get("fill_hue"), 
                        default_font_id)
                
                fill_sat = TrackbarRow(
                    "Fill Sat", 
                    self.params.get("fill_sat"), 
                        default_font_id)
                
                fill_val = TrackbarRow(
                    "Fill Val", 
                    self.params.get("fill_val"), 
                        default_font_id)
                
                fill_opacity = TrackbarRow(
                    "Fill Opacity", 
                    self.params.get("fill_opacity"), 
                        default_font_id)
            dpg.bind_item_font("fill_generator", global_font_id)
        dpg.bind_item_font("shape_generator", global_font_id)
