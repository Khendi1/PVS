import cv2
import numpy as np
import random
import logging

from effects.base import EffectBase
from effects.enums import SharpenType, BlurType, NoiseType
from common import Widget

log = logging.getLogger(__name__)

class Pixels(EffectBase):

    def __init__(self, params, image_width: int, image_height: int, group=None):
        subgroup = self.__class__.__name__
        self.params = params
        self.image_width = image_width
        self.image_height = image_height

        self.sharpen_type = params.add("sharpen_type",
                                        min=0, max=len(SharpenType)-1, default=0,
                                        group=group, subgroup=subgroup,
                                        type=Widget.DROPDOWN, options=SharpenType)
        self.sharpen_intensity = params.add("sharpen_intensity",
                                            min=1.0, max=8.0, default=4.0,
                                            subgroup=subgroup, group=group)
        self.mask_blur = params.add("mask_blur",
                                    min=1, max=10, default=5,
                                    subgroup=subgroup, group=group)
        self.k_size = params.add("k_size",
                                 min=0, max=11, default=3,
                                 subgroup=subgroup, group=group)

        self.blur_type = params.add("blur_type",
                                    min=0, max=len(BlurType)-1, default=0,
                                    group=group, subgroup=subgroup,
                                    type=Widget.DROPDOWN, options=BlurType)
        self.blur_kernel_size = params.add("blur_kernel_size",
                                           min=1, max=100, default=1,
                                           subgroup=subgroup, group=group)

        self.noise_type = params.add("noise_type",
                                     min=NoiseType.NONE.value, max=NoiseType.RANDOM.value, default=NoiseType.NONE.value,
                                     group=group, subgroup=subgroup,
                                     type=Widget.DROPDOWN, options=NoiseType)
        self.noise_intensity = params.add("noise_intensity",
                                          min=0.0, max=1.0, default=0.1,
                                          subgroup=subgroup, group=group)

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
        # PERFORMANCE: Skip early if no noise (most common case)
        if self.noise_type.value == NoiseType.NONE.value:
            return image

        # Ensure image is a float type for calculations, then convert back to uint8
        # Make a copy to avoid modifying the original image
        noisy_image = image.astype(np.float32)

        # Dispatch to the appropriate noise function based on noise_type
        if self.noise_type.value == NoiseType.GAUSSIAN.value:
            return self._apply_gaussian_noise(noisy_image)
        elif self.noise_type.value == NoiseType.POISSON.value:
            return self._apply_poisson_noise(noisy_image)
        elif self.noise_type.value == NoiseType.SALT_AND_PEPPER.value:
            return self._apply_salt_and_pepper_noise(noisy_image)
        elif self.noise_type.value == NoiseType.SPECKLE.value:
            return self._apply_speckle_noise(noisy_image)
        elif self.noise_type.value == NoiseType.SPARSE.value:
            return self._apply_sparse_noise(noisy_image)
        elif self.noise_type.value == NoiseType.RANDOM.value:
            return self._apply_random_noise(noisy_image)
        else:
            log.warning(
                f"Unknown noise type: {self.noise_type.value}. Returning original image."
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
        std_dev = self.noise_intensity * 50
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
        scaled_image = (image / 255.0 * 100.0)
        poisson_noise = np.random.poisson(
            scaled_image * self.noise_intensity.value * 2
        ).astype(np.float32)
        noisy_image = image + (poisson_noise / 100.0 * 255.0)  # Scale noise back to 0-255 range
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_salt_and_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Salt & Pepper noise to the image.
        Intensity controls the proportion of pixels affected.
        """
        amount = self.noise_intensity  # Proportion of pixels to affect
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
        std_dev = self.noise_intensity * 0.5  # Scale std_dev for multiplicative noise
        speckle_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + image * speckle_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_sparse_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies sparse noise by randomly selecting a percentage of pixels
        and setting them to a random value within the 0-255 range.
        Intensity controls the proportion of pixels affected.
        OPTIMIZED: Vectorized to avoid Python loops.
        """
        amount = self.noise_intensity.value  # Proportion of pixels to affect

        num_pixels_to_affect = int(np.ceil(
            amount * image.size / image.shape[-1] if image.ndim == 3 else image.size
        ))

        # Get image dimensions
        height, width = image.shape[0], image.shape[1]

        # Generate random (y, x) coordinates for the pixels to affect (VECTORIZED)
        random_y = np.random.randint(0, height, num_pixels_to_affect)
        random_x = np.random.randint(0, width, num_pixels_to_affect)

        # Apply random values to the selected pixels (VECTORIZED - no Python loop!)
        if image.ndim == 3:  # Color image (e.g., BGR)
            # Generate random values for all pixels at once
            random_values = np.random.randint(0, 256, (num_pixels_to_affect, image.shape[2]), dtype=np.uint8)
            image[random_y, random_x] = random_values
        else:  # Grayscale image
            random_values = np.random.randint(0, 256, num_pixels_to_affect, dtype=np.uint8)
            image[random_y, random_x] = random_values

        return image.astype(np.uint8)

    def _apply_random_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies uniform random noise to every pixel.
        Intensity controls the maximum range of the random noise added.
        """
        # Generate random noise in the range [-intensity*127.5, intensity*127.5]
        noise_range = (self.noise_intensity * 127.5)  
        random_noise = np.random.uniform(-noise_range, noise_range, image.shape).astype(
            np.float32
        )
        noisy_image = image + random_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def blur(self, frame: np.ndarray):
        mode = BlurType(self.blur_type.value)
        if mode == BlurType.NONE:
            return frame

        ksize = self.blur_kernel_size.value

        # PERFORMANCE: Skip blur if kernel size is 1 (no effect)
        if ksize <= 1:
            return frame

        # PERFORMANCE: All OpenCV blur functions work best with uint8, so convert once if needed
        needs_conversion = frame.dtype != np.uint8
        if needs_conversion:
            frame_uint8 = frame.astype(np.uint8)
        else:
            frame_uint8 = frame

        match mode:
            case BlurType.GAUSSIAN:
                result = cv2.GaussianBlur(frame_uint8, (ksize, ksize), 0)
            case BlurType.MEDIAN:
                result = cv2.medianBlur(frame_uint8, ksize)
            case BlurType.BOX:
                result = cv2.blur(frame_uint8, (ksize, ksize))
            case BlurType.BILATERAL:
                result = cv2.bilateralFilter(frame_uint8, ksize, 75, 75)
            case _:
                result = frame_uint8

        # Convert back to original dtype if needed
        if needs_conversion:
            return result.astype(frame.dtype)
        return result

    def _kernel_sharpening(self, image, strength=1.0):
        """Sharpening using a custom convolution kernel (Filter Mode)."""
        base_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=np.float32)
        
        scaled_kernel = base_kernel * strength + (1 - strength) * np.array([[0, 0, 0],
                                                                            [0, 1, 0],
                                                                            [0, 0, 0]], dtype=np.float32)

        sharpened = cv2.filter2D(image, -1, scaled_kernel)
        return sharpened

    def _unsharp_masking(self, image, blur_sigma=5, amount=1.5):
        """Explicit Unsharp Masking technique (controlled by blur and amount)."""
        blurred = cv2.GaussianBlur(image, (0, 0), blur_sigma) 
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def _laplacian_sharpening(self, image, ksize=3):
        """Sharpening using the Laplacian edge detection filter."""
        img_float = image.astype(np.float32)
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=ksize)
        sharpened = img_float - laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def sharpen_frame(self, frame: np.ndarray):

        sharpened_frame = frame
        match SharpenType(self.sharpen_type.value):
            case SharpenType.NONE:
                return frame
            case SharpenType.KERNEL:
                sharpened_frame = self._kernel_sharpening(frame, self.sharpen_intensity.value)
            case SharpenType.UNSHARP:
                sharpened_frame = self._unsharp_masking(frame, self.mask_blur.value, self.sharpen_intensity.value)
            case SharpenType.LAPLACIAN:
                sharpened_frame = self._laplacian_sharpening(frame, self.k_size.value)
            case SharpenType.TEST:
                    if self.sharpen_intensity.value <= self.sharpen_intensity.min + 0.01:
                        return frame
                    sharpening_kernel = np.array(
                        [[0, -1, 0], [-1, self.sharpen_intensity.value, -1], [0, -1, 0]])
                    # Apply the kernel to the frame using cv2.filter2D
                    sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)
        return sharpened_frame
