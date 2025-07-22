import cv2
import numpy as np
from enum import Enum
import time
from config import params

# Define an Enum for different reflection modes
class ReflectionMode(Enum):
    """Enumeration for different image reflection modes."""
    NONE = 0        # No reflection
    HORIZONTAL = 1  # Reflect across the Y-axis (flip horizontally)
    VERTICAL = 2    # Reflect across the X-axis (flip vertically)
    BOTH = 3       # Reflect across both X and Y axes (flip horizontally and vertically)
    QUAD_SYMMETRY = 4  # Reflect across both axes with quadrants (not implemented)
    SPLIT = 5      # Reflect left half onto right half

    def __str__(self):
        return self.name.replace('_', ' ').title()

class Reflector:
    """
    A class to apply reflection transformations to image frames from a stream.
    """
    def __init__(self, mode: ReflectionMode = ReflectionMode.NONE):
        """
        Initializes the StreamReflector with a specified reflection mode.

        Args:
            mode (ReflectionMode): The reflection mode to apply. Defaults to NONE.
        """
        if not isinstance(mode, ReflectionMode):
            raise ValueError("mode must be an instance of ReflectionMode Enum.")
        self._mode = params.add("reflection_mode", 0, len(ReflectionMode) - 1, ReflectionMode.NONE.value)
        self.num_axis = 3
        print(f"StreamReflector initialized with mode: {self._mode}")

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
        print(f"Reflection mode set to: {self._mode}")

    def apply_reflection(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the currently set reflection mode to the input image frame.

        Args:
            frame (np.ndarray): The input image frame (NumPy array).

        Returns:
            np.ndarray: The reflected image frame.
        """
        if self._mode.value == ReflectionMode.NONE.value:
            return frame.copy()  # Return a copy to ensure original isn't modified
        elif self._mode.value == ReflectionMode.HORIZONTAL.value:
            # cv2.flip(src, flipCode): flipCode > 0 for horizontal, 0 for vertical, < 0 for both
            return cv2.flip(frame, 1)
        elif self._mode.value == ReflectionMode.VERTICAL.value:
            return cv2.flip(frame, 0)
        elif self._mode.value == ReflectionMode.BOTH.value:
            return cv2.flip(frame, -1)
        elif self._mode.value == ReflectionMode.QUAD_SYMMETRY.value:
            return self._apply_quad_symmetry(frame)
        elif self._mode.value == ReflectionMode.SPLIT.value:  # New mode: reflect left half onto right half
            height, width = frame.shape[:2]
            w_half = width // 2
            output_frame = frame.copy()
            left_half = frame[:, :w_half]
            # Reflect left half horizontally
            reflected_left = cv2.flip(left_half, 1)
            # Place reflected left half onto right half
            output_frame[:, w_half:] = reflected_left[:, :width - w_half]
            return output_frame
        else:
            print(f"Warning: Unknown reflection mode: {self._mode}. Returning original frame.")
            return frame.copy()

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
            print("Warning: Image too small for quad symmetry. Returning original.")
            return frame.copy()

        # Extract the top-left quadrant
        top_left_quadrant = frame[0:h_half, 0:w_half].copy() # .copy() is important to avoid modifying original

        # Create reflected versions
        top_right_quadrant = cv2.flip(top_left_quadrant, 1)  # Flip horizontally
        bottom_left_quadrant = cv2.flip(top_left_quadrant, 0) # Flip vertically
        bottom_right_quadrant = cv2.flip(top_left_quadrant, -1) # Flip both horizontally and vertically

        # Create a new canvas to assemble the reflected parts
        # Ensure the output frame matches the original dimensions, even if they were odd
        output_frame = np.zeros_like(frame)

        # Place the quadrants into the output frame
        output_frame[0:h_half, 0:w_half] = top_left_quadrant
        output_frame[0:h_half, w_half:width] = top_right_quadrant
        output_frame[h_half:height, 0:w_half] = bottom_left_quadrant
        output_frame[h_half:height, w_half:width] = bottom_right_quadrant

        return output_frame

def generate_dummy_frame(width: int, height: int, frame_time: float) -> np.ndarray:
    """
    Generates a simple animated dummy frame for demonstration.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background color animation
    bg_color_r = int((np.sin(frame_time * 0.5) * 0.5 + 0.5) * 255)
    bg_color_g = int((np.sin(frame_time * 0.7) * 0.5 + 0.5) * 255)
    bg_color_b = int((np.sin(frame_time * 0.9) * 0.5 + 0.5) * 255)
    frame[:, :, 0] = bg_color_b # Blue
    frame[:, :, 1] = bg_color_g # Green
    frame[:, :, 2] = bg_color_r # Red

    # Moving circle
    circle_radius = 50
    circle_x = int(width / 2 + np.sin(frame_time * 1.5) * (width / 2 - circle_radius - 10))
    circle_y = int(height / 2 + np.cos(frame_time * 1.2) * (height / 2 - circle_radius - 10))
    cv2.circle(frame, (circle_x, circle_y), circle_radius, (255, 255, 255), -1) # White circle

    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Mode: {StreamReflector.mode_instance.mode.name}" if 'mode_instance' in globals() else "Mode: N/A"
    cv2.putText(frame, text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def main():
    """
    Main function to demonstrate the StreamReflector class with a dummy video stream.
    """
    global mode_instance # Make it global so generate_dummy_frame can access it for text

    print("--- Stream Reflector Demonstration ---")

    # Define frame dimensions
    WIDTH, HEIGHT = 640, 480
    FPS = 30

    # Create a StreamReflector instance
    reflector = StreamReflector(mode=ReflectionMode.NONE)
    mode_instance = reflector # Assign to global for text overlay

    # OpenCV window setup
    cv2.namedWindow("Original Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Reflected Stream", cv2.WINDOW_AUTOSIZE)

    start_time = time.time()
    running = True

    print("\n--- Controls ---")
    print("Press '0' for NONE reflection")
    print("Press '1' for HORIZONTAL reflection")
    print("Press '2' for VERTICAL reflection")
    print("Press '3' for BOTH (Horizontal & Vertical) reflection")
    print("Press 'Q' to quit")
    print("----------------")

    while running:
        current_time = time.time()
        # Generate a dummy frame (simulating a video stream)
        original_frame = generate_dummy_frame(WIDTH, HEIGHT, current_time)

        # Apply reflection using the StreamReflector
        reflected_frame = reflector.apply_reflection(original_frame)

        # Display frames
        cv2.imshow("Original Stream", original_frame)
        cv2.imshow("Reflected Stream", reflected_frame)

        # Handle key presses
        key = cv2.waitKey(int(1000 / FPS)) & 0xFF # Wait for a short duration

        if key == ord('q') or key == ord('Q'):
            running = False
        elif key == ord('0'):
            reflector.mode = ReflectionMode.NONE
        elif key == ord('1'):
            reflector.mode = ReflectionMode.HORIZONTAL
        elif key == ord('2'):
            reflector.mode = ReflectionMode.VERTICAL
        elif key == ord('3'):
            reflector.mode = ReflectionMode.BOTH
    
    cv2.destroyAllWindows()
    print("Stream Reflector demonstration stopped.")

if __name__ == "__main__":
    main()
