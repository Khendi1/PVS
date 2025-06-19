import cv2
import numpy as np
from enum import Enum
from config import params

# Define NoiseType enum - assuming it exists in your original code
class NoiseType(Enum):
    NONE = 0
    GAUSSIAN = 1
    SALT_AND_PEPPER = 2
    POISSON = 3
    SPECKLE = 4

class ImageNoiser:
    def __init__(self, noise_type: NoiseType = NoiseType.NONE):
        self.noise_intensity = params.add("noise_intensity", 0.0, 1.0, 0.5)
        self.noise_type = params.add("noise_type", 0, 4, 0) # 0=None, 1=Gaussian, 2=Salt&Pepper, 3=Poisson, 4=Speckle

    def noisy(self, image):
        """
        Applies various types of noise to a given image.

        Args:
            image (numpy.ndarray): The input image (expected to be a cv2 image,
                                   typically a numpy array of uint8 type).

        Returns:
            numpy.ndarray: The noisy image, formatted as a cv2 image (numpy array
                           of uint8 type with values clamped between 0 and 255).
        """
        if self.noise_type == NoiseType.NONE:
            return image

        # Ensure image is in a float format for noise addition to avoid overflow
        # and allow negative values during intermediate calculations,
        # then convert back to uint8 and clip.
        # Use float64 for better precision during calculations
        image_float = image.astype(np.float64) 

        if self.noise_type == NoiseType.GAUSSIAN:
            row, col, ch = image.shape
            mean = 0
            # --- INCREASED NOISE VARIANCE FOR VISIBILITY ---
            var = 100 # Increased from 0.1, sigma will be 10 (sqrt(100))
            sigma = var**0.5
            
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            
            noisy_image = image_float + gauss
            
            # Clip values to 0-255 range and convert back to uint8
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return noisy_image

        elif self.noise_type == NoiseType.SALT_AND_PEPPER:
            # --- INCREASED NOISE AMOUNT FOR VISIBILITY ---
            amount = 0.02 # Increased from 0.004 (2% of pixels will be affected)
            s_vs_p = 0.5  # Ratio of salt vs. pepper noise
            
            out = np.copy(image) # Create a copy to modify

            # Salt mode (white pixels)
            num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
            # Random coordinates for salt pixels
            coords_salt = [np.random.randint(0, dim, num_salt) for dim in image.shape]
            out[tuple(coords_salt)] = 255 # Set to white (assuming 255 for uint8)

            # Pepper mode (black pixels)
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
            # Random coordinates for pepper pixels
            coords_pepper = [np.random.randint(0, dim, num_pepper) for dim in image.shape]
            out[tuple(coords_pepper)] = 0 # Set to black (assuming 0 for uint8)
            
            return out

        elif self.noise_type == NoiseType.POISSON:
            # --- REFINED POISSON NOISE FOR VISIBILITY ---
            # Poisson noise strength depends on pixel intensity.
            # Convert to 0-1 range for a more controlled Poisson application
            # and then scale back.
            
            # This factor controls the "granularity" of the Poisson distribution.
            # A lower factor makes the noise more noticeable.
            # Think of it as inverse of the number of "events" per pixel value.
            # Example: for an 8-bit image, 255 events would be very fine noise.
            # Let's use a smaller value to make it more apparent.
            intensity_factor = 50.0 # Increased from implicit, adjust for desired strength

            # Normalize image to 0-1 range (approximate "rates" for Poisson)
            normalized_image = image_float / 255.0
            
            # Apply Poisson noise: mean and variance of Poisson are equal to lambda.
            # We scale the normalized image by intensity_factor, apply Poisson,
            # then scale back.
            noisy_normalized_image = np.random.poisson(normalized_image * intensity_factor) / intensity_factor
            
            # Convert back to 0-255 range
            noisy_image = noisy_normalized_image * 255.0
            
            # Clip values to 0-255 range and convert back to uint8
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return noisy_image

        elif self.noise_type == NoiseType.SPECKLE:
            row, col, ch = image.shape
            # --- ADDED SCALING FACTOR TO GAUSSIAN NOISE FOR VISIBILITY ---
            # np.random.randn has std dev 1. Multiplying by 0.5 reduces its magnitude.
            # This means `image_float * gauss` will have values typically between
            # -1.5*image_float and 1.5*image_float.
            gauss = np.random.randn(row, col, ch) * 0.5 
            
            # Apply speckle noise: noisy = image + image * noise
            noisy_image = image_float + image_float * gauss
            
            # Clip values to 0-255 range and convert back to uint8
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return noisy_image

        # If an unknown noise type is somehow passed, return the original image
        return image