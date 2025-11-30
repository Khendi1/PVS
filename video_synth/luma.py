import cv2
from enum import IntEnum, auto
import numpy as np

class LumaMode(IntEnum):
    WHITE = auto()
    BLACK = auto()

def luma_key(frame1: np.ndarray, frame2: np.ndarray, mode: LumaMode, threshold: int):
        # The mask will determine which parts of the current frame are kept
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if mode == LumaMode.BLACK.value:
            # Use THRESH_BINARY_INV to key out DARK areas (Luma is low)
            # Pixels with Luma < threshold become white (255) in the mask, meaning they are KEPT
            ret, mask = cv2.threshold(
                gray, threshold, 255, cv2.THRESH_BINARY_INV
            )
        elif mode == LumaMode.WHITE.value:
            ret, mask = cv2.threshold(
                gray, threshold, 255, cv2.THRESH_BINARY
            )

        smoothed_mask = cv2.GaussianBlur(mask, (1, 1), 0)

        # Keep the keyed-out (bright) parts of the current frame
        fg = cv2.bitwise_and(frame1, frame1, mask=smoothed_mask)

        # Invert the mask to find the areas *not* keyed out (the dark areas)
        mask_inv = cv2.bitwise_not(smoothed_mask)

        # Use the inverted mask to "cut a hole" in the previous frame
        bg = cv2.bitwise_and(frame2, frame2, mask=mask_inv)

        # Combine the new foreground (fg) with the previous frame's background (bg)
        return cv2.add(fg, bg)