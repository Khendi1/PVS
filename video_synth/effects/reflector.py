import cv2
import numpy as np
import logging

from effects.base import EffectBase
from effects.enums import ReflectionMode
from common import Widget

log = logging.getLogger(__name__)

class Reflector(EffectBase):
    """
    A class to apply reflection transformations to image frames from a stream.
    """

    def __init__(self, params, mode: ReflectionMode = ReflectionMode.NONE, group=None):
        """
        Initializes the Reflector with a specified reflection mode.

        Args:
            mode (ReflectionMode): The reflection mode to apply. Defaults to NONE.
        """
        subgroup = self.__class__.__name__
        self.params = params
        if not isinstance(mode, ReflectionMode):
            raise ValueError("mode must be an instance of ReflectionMode Enum.")
        self._mode = params.add("reflection_mode",
                                min=0, max=len(ReflectionMode) - 1, default=ReflectionMode.NONE.value,
                                group=group, subgroup=subgroup,
                                type=Widget.DROPDOWN, options=ReflectionMode)
        self.segments = params.add("reflector_segments",
                                   min=0, max=10, default=0,
                                   subgroup=subgroup, group=group)
        self.zoom = params.add("reflector_z",
                               min=0.5, max=2, default=1.0,
                               subgroup=subgroup, group=group)
        self.rotation = params.add("reflector_r",
                                   min=-360, max=360, default=0.0,
                                   subgroup=subgroup, group=group)
        self.width = None 
        self.height = None

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
            self.height, self.width, _ = frame.shape

        match ReflectionMode(self._mode.value):
            case ReflectionMode.NONE:
                return frame

            case ReflectionMode.HORIZONTAL:
                return cv2.flip(frame, 1)

            case ReflectionMode.VERTICAL:
                return cv2.flip(frame, 0)

            case ReflectionMode.BOTH:
                return cv2.flip(frame, -1)

            case ReflectionMode.QUAD_SYMMETRY:
                return self._apply_quad_symmetry(frame)

            case ReflectionMode.SPLIT:
                w_half = self.width // 2
                output_frame = frame.copy()
                left_half = frame[:, :w_half]
                reflected_left = cv2.flip(left_half, 1)
                output_frame[:, w_half:] = reflected_left[:, :self.width - w_half]
                return output_frame

            case ReflectionMode.KALEIDOSCOPE:
                return self._apply_kaleidoscope(frame)

            case _:
                return frame


    def _apply_kaleidoscope(self, frame):
        """Applies the kaleidoscope effect to a single frame."""
        
        segments = self.segments.value
        zoom = self.zoom.value
        center_x_rel = 1
        center_y_rel = 1
        border_mode = None

        # Calculate geometric parameters
        K = max(2, segments)
        center = (int(self.width * center_x_rel), int(self.height * center_y_rel))
        
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
        shift_pixels = int(polar_w * (self.rotation.value / 360.0))
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
