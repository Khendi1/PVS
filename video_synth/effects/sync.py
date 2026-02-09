import cv2
import numpy as np

from effects.base import EffectBase

class Sync(EffectBase):

    def __init__(self, params, group):
        subgroup = self.__class__.__name__
        self.params = params
        self.x_sync_freq = params.add("x_sync_freq",
                                       min=0.1, max=100.0, default=1.0,
                                       subgroup=subgroup, group=group)
        self.x_sync_amp = params.add("x_sync_amp",
                                     min=-200, max=200, default=0.0,
                                     subgroup=subgroup, group=group)
        self.x_sync_speed = params.add("x_sync_speed",
                                       min=5.0, max=10.0, default=9.0,
                                       subgroup=subgroup, group=group)
        self.y_sync_freq = params.add("y_sync_freq",
                                      min=0.1, max=100.0, default=1.0,
                                      subgroup=subgroup, group=group)
        self.y_sync_amp = params.add("y_sync_amp",
                                     min=-200, max=200, default=0.0,
                                     subgroup=subgroup, group=group)
        self.y_sync_speed = params.add("y_sync_speed",
                                       min=5.0, max=10.0, default=9.0,
                                       subgroup=subgroup, group=group)

    def sync(self, frame: np.ndarray):
        """
        Applies a raster wobble effect to the frame using sine waves on both X and Y axes.
        Optimized with vectorized NumPy operations instead of Python loops.
        Color space conversions removed - operates directly on BGR (channel order doesn't matter for shifting).
        """
        if self.x_sync_amp.value == 0 and self.y_sync_amp.value == 0:
            return frame

        height, width = frame.shape[:2]
        tick = cv2.getTickCount()

        # X-axis wobble (horizontal shift per row) - VECTORIZED
        if self.x_sync_amp.value != 0:
            # Compute shifts for all rows at once
            y_indices = np.arange(height)
            shifts_x = (self.x_sync_amp.value
                       * np.sin(y_indices / self.x_sync_freq.value
                                + tick / (10**self.x_sync_speed.value))).astype(int)

            # Create column index array and apply shifts
            col_indices = (np.arange(width)[None, :] - shifts_x[:, None]) % width
            warped = frame[y_indices[:, None], col_indices]
        else:
            warped = frame

        # Y-axis wobble (vertical shift per column) - VECTORIZED
        if self.y_sync_amp.value != 0:
            # Compute shifts for all columns at once
            x_indices = np.arange(width)
            shifts_y = (self.y_sync_amp.value
                       * np.sin(x_indices / self.y_sync_freq.value
                                + tick / (10**self.y_sync_speed.value))).astype(int)

            # Create row index array and apply shifts
            row_indices = (np.arange(height)[:, None] - shifts_y[None, :]) % height
            warped = warped[row_indices, x_indices[None, :]]

        return warped
