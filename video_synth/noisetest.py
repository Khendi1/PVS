import numpy as np
import cv2
from enum import IntEnum
import random
import time
from config import params

# Define an Enum for different noise types
class NoiseType(IntEnum):
    NONE = 0
    GAUSSIAN = 1
    POISSON = 2
    SALT_AND_PEPPER = 3
    SPECKLE = 4
    SPARSE = 5
    RANDOM = 6

class ImageNoiser:
    """
    A class to apply various types of noise to an image.
    """
    def __init__(self, noise_type: NoiseType = NoiseType.NONE, noise_intensity: float = 0.1):
        """
        Initializes the ImageNoiser with a default noise type and intensity.

        Args:
            noise_type (NoiseType): The type of noise to apply. Defaults to NONE.
            noise_intensity (float): The intensity of the noise, typically between 0.0 and 1.0.
                                     Its interpretation varies by noise type. Defaults to 0.1.
        """
        if not isinstance(noise_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")
        if not (0.0 <= noise_intensity <= 1.0):
            print("Warning: noise_intensity should ideally be between 0.0 and 1.0.")

        # self._noise_type = noise_type
        # self._noise_intensity = noise_intensity
        self._noise_type = params.add("noise_type", NoiseType.NONE.value, NoiseType.RANDOM.value, noise_type.value)
        self._noise_intensity = params.add("noise_intensity", 0.0, 1.0, noise_intensity)
        print(f"ImageNoiser initialized with type: {self._noise_type.value}, intensity: {self._noise_intensity}")

    @property
    def noise_type(self) -> NoiseType:
        """Get the current noise type."""
        return self._noise_type.val()

    @noise_type.setter
    def noise_type(self, new_type: NoiseType):
        """Set the noise type."""
        if not isinstance(new_type, NoiseType):
            raise ValueError("noise_type must be an instance of NoiseType Enum.")
        self._noise_type.set_value(new_type)
        print(f"Noise type set to: {self._noise_type.val()}")

    @property
    def noise_intensity(self) -> float:
        """Get the current noise intensity."""
        return self._noise_intensity.value

    @noise_intensity.setter
    def noise_intensity(self, new_intensity: float):
        """Set the noise intensity."""
        if not (0.0 <= new_intensity <= 1.0):
            print("Warning: noise_intensity should ideally be between 0.0 and 1.0.")
        self._noise_intensity.set_value(new_intensity)
        print(f"Noise intensity set to: {self._noise_intensity}")

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
            print(f"Unknown noise type: {self._noise_type.value}. Returning original image.")
            return image.copy() # Return original if type is unknown

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
        # Scale image to a range suitable for Poisson (e.g., 0-100)
        # Then add Poisson noise and scale back
        scaled_image = image / 255.0 * 100.0 # Scale to 0-100 for better Poisson distribution
        # Lambda for Poisson distribution is often related to the pixel intensity
        # Use a base lambda and add intensity-scaled value
        poisson_noise = np.random.poisson(scaled_image * self._noise_intensity.value * 2).astype(np.float32)
        noisy_image = image + (poisson_noise / 100.0 * 255.0) # Scale noise back to 0-255 range
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_salt_and_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Salt & Pepper noise to the image.
        Intensity controls the proportion of pixels affected.
        """
        noisy_image = image.copy()
        amount = self._noise_intensity # Proportion of pixels to affect
        s_vs_p = 0.5 # Ratio of salt vs. pepper (0.5 means equal)

        # Apply salt noise (white pixels)
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[tuple(coords)] = 255

        # Apply pepper noise (black pixels)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        return noisy_image.astype(np.uint8)

    def _apply_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Speckle (multiplicative) noise to the image.
        noisy_image = image + image * noise
        Intensity controls the standard deviation of the noise.
        """
        mean = 0
        std_dev = self._noise_intensity * 0.5 # Scale std_dev for multiplicative noise
        speckle_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + image * speckle_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _apply_sparse_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies sparse noise by randomly selecting a percentage of pixels
        and setting them to a random value within the 0-255 range.
        Intensity controls the proportion of pixels affected.
        """
        noisy_image = image.copy()
        amount = self._noise_intensity # Proportion of pixels to affect

        num_pixels_to_affect = np.ceil(amount * image.size / image.shape[-1] if image.ndim == 3 else image.size).astype(int)

        # Get image dimensions
        height, width = image.shape[0], image.shape[1]

        # Generate random (y, x) coordinates for the pixels to affect
        random_y = np.random.randint(0, height, num_pixels_to_affect)
        random_x = np.random.randint(0, width, num_pixels_to_affect)

        # Apply random values to the selected pixels
        for i in range(num_pixels_to_affect):
            y, x = random_y[i], random_x[i]
            if image.ndim == 3: # Color image (e.g., BGR)
                # Assign a list of 3 random values to the (y, x) pixel
                noisy_image[y, x] = [random.randint(0, 255) for _ in range(image.shape[2])]
            else: # Grayscale image
                # Assign a single random value to the (y, x) pixel
                noisy_image[y, x] = random.randint(0, 255)
                
        return noisy_image.astype(np.uint8)

    def _apply_random_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies uniform random noise to every pixel.
        Intensity controls the maximum range of the random noise added.
        """
        # Generate random noise in the range [-intensity*127.5, intensity*127.5]
        # Then add it to the image
        noise_range = self._noise_intensity * 127.5 # Max deviation from original pixel value
        random_noise = np.random.uniform(-noise_range, noise_range, image.shape).astype(np.float32)
        noisy_image = image + random_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

def main():
    """
    Main function to demonstrate the ImageNoiser class.
    It loads a sample image, applies different noise types, and displays them.
    """
    print("--- Image Noiser Demonstration ---")

    # Create a dummy image if no file is found
    try:
        # Attempt to load a sample image. Replace 'sample_image.jpg' with your image path.
        # If you don't have one, it will create a simple grayscale square.
        image_path = 'sample_image.jpg' # You can change this path
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Could not load '{image_path}'. Creating a dummy grayscale image.")
            # Create a simple grayscale square image (200x200)
            original_image = np.zeros((200, 200), dtype=np.uint8)
            cv2.rectangle(original_image, (50, 50), (150, 150), 127, -1) # Gray square
            cv2.circle(original_image, (100, 100), 30, 200, -1) # Lighter circle
            # Convert to 3 channels (BGR) as most noise functions expect this for consistency
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            print(f"Loaded image: {image_path}")

    except Exception as e:
        print(f"An error occurred while loading image: {e}. Creating a dummy grayscale image.")
        original_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(original_image, (50, 50), (150, 150), 127, -1)
        cv2.circle(original_image, (100, 100), 30, 200, -1)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Original Image", original_image)
    cv2.waitKey(1) # Display for a short moment

    noiser = ImageNoiser()

    # Demonstrate different noise types
    noise_types_to_test = [
        NoiseType.NONE,
        NoiseType.GAUSSIAN,
        NoiseType.POISSON,
        NoiseType.SALT_AND_PEPPER,
        NoiseType.SPECKLE,
        NoiseType.SPARSE,
        NoiseType.RANDOM
    ]

    intensities_to_test = [0.05, 0.15, 0.3] # Different intensities for demonstration

    for noise_type in noise_types_to_test:
        for intensity in intensities_to_test:
            noiser.noise_type = noise_type
            noiser.noise_intensity = intensity

            # Apply noise
            noisy_image = noiser.apply_noise(original_image)

            # Display the noisy image
            window_name = f"{noise_type.value} Noise (Intensity: {intensity:.2f})"
            cv2.imshow(window_name, noisy_image)
            cv2.waitKey(1000) # Display each noisy image for 1 second

    print("\nDemonstration complete. Press any key to close all windows.")
    cv2.waitKey(0) # Wait for a key press to close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
