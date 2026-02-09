import cv2
import numpy as np
from collections import deque
import logging

from effects.base import EffectBase
from param import ParamTable
from luma import LumaMode
from common import Widget

log = logging.getLogger(__name__)

class Feedback(EffectBase):

    def __init__(self, params: ParamTable, image_width: int, image_height: int, group=None):
        self.params = params
        self.height = image_height
        self.width = image_width

        subgroup = self.__class__.__name__

        self.alpha = params.add("alpha",
                                min=0.0, max=1.0, default=0.0,
                                subgroup=subgroup, group=group)
        self.temporal_filter = params.add("temporal_filter",
                                          min=0, max=1.0, default=0.0,
                                          subgroup=subgroup, group=group)
        self.feedback_luma_threshold = params.add("feedback_luma_threshold",
                                                  min=0, max=255, default=0,
                                                  subgroup=subgroup, group=group)
        self.luma_mode = params.add("luma_mode",
                                    min=LumaMode.WHITE.value, max=LumaMode.BLACK.value, default=LumaMode.WHITE.value,
                                    group=group, subgroup=subgroup,
                                    type=Widget.RADIO, options=LumaMode)
        self.frame_skip = params.add("frame_skip",
                                     min=0, max=10, default=0,
                                     subgroup=subgroup, group=group)
        self.buffer_select = params.add("buffer_frame_select",
                                        min=-1, max=20, default=-1,
                                        subgroup=subgroup, group=group)
        self.buffer_frame_blend = params.add("buffer_frame_blend",
                                             min=0.0, max=1.0, default=0.0,
                                             subgroup=subgroup, group=group)

        self.prev_frame_scale = params.add("prev_frame_scale",
                                           min=90, max=110, default=100,
                                           subgroup=subgroup, group=group)

        self.max_buffer_size = 30
        self.buffer_size = params.add("buffer_size",
                                      min=0, max=self.max_buffer_size, default=0,
                                      subgroup=subgroup, group=group)
        # this should probably be initialized in reset() to avoid issues with reloading config
        self.frame_buffer = deque(maxlen=self.max_buffer_size)
        # Running sum for efficient frame averaging (avoids converting deque to array every frame)
        self._running_sum = None
        self._prev_buffer_size = 0


    def scale_frame(self, frame):

        target_height, target_width, _ = frame.shape
        scale = self.prev_frame_scale.value / 100.0
        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)

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

        if self.buffer_select.value == -1 or len(self.frame_buffer) <= self.buffer_select.value:
            return frame
                
        nth_frame = self.frame_buffer[self.buffer_select.value]
        
        frame = cv2.addWeighted(
            frame,
            1 - self.buffer_frame_blend.value,
            nth_frame,
            self.buffer_frame_blend.value,
            0,
        )       

        # return cv2.convertScaleAbs(frame).astype('uint8')
        return frame


    def avg_frame_buffer(self, frame):

        # CRITICAL FIX: Only copy and buffer frames when actually needed (buffer_size > 1)
        # Previously this was copying every frame, causing massive memory accumulation
        if self.buffer_size.value <= 1:
            self._running_sum = None  # Reset running sum if not averaging
            # PERFORMANCE: Clear buffer when not in use to free memory
            if len(self.frame_buffer) > 0:
                self.frame_buffer.clear()
            return frame

        # Now actually buffer the frame
        frame_copy = frame.copy()
        self.frame_buffer.append(frame_copy)

        # If the buffer is not yet full, return the original frame.
        if len(self.frame_buffer) < self.buffer_size.value:
            log.debug(f"Buffering frames: {len(self.frame_buffer)}/{self.buffer_size.value}")
            self._running_sum = None  # Reset running sum during fill phase
            return frame

        # Check if buffer_size changed - if so, rebuild running sum
        if self._prev_buffer_size != self.buffer_size.value:
            log.debug(f"Buffer size changed from {self._prev_buffer_size} to {self.buffer_size.value}, rebuilding running sum")
            buffer_list = list(self.frame_buffer)[-self.buffer_size.value:]
            self._running_sum = np.sum(buffer_list, axis=0)
            self._prev_buffer_size = self.buffer_size.value
        # Initialize running sum on first pass
        elif self._running_sum is None:
            buffer_list = list(self.frame_buffer)[-self.buffer_size.value:]
            self._running_sum = np.sum(buffer_list, axis=0)
            self._prev_buffer_size = self.buffer_size.value
        # Update running sum: add newest frame, subtract oldest frame
        else:
            # Get the frame that's about to be dropped (oldest in the window)
            # Since we just appended, the window is the last buffer_size.value frames
            # The oldest frame in the window is at index -(buffer_size.value + 1) before the append,
            # which is now at index -(buffer_size.value)
            oldest_in_window_idx = len(self.frame_buffer) - self.buffer_size.value - 1
            if oldest_in_window_idx >= 0:
                oldest_frame = self.frame_buffer[oldest_in_window_idx]
                self._running_sum = self._running_sum + frame_copy - oldest_frame
            else:
                # Edge case: this shouldn't happen if logic is correct, but rebuild if needed
                buffer_list = list(self.frame_buffer)[-self.buffer_size.value:]
                self._running_sum = np.sum(buffer_list, axis=0)

        # Return the average
        avg_frame = self._running_sum / self.buffer_size.value
        return avg_frame


    def apply_temporal_filter(self, prev_frame, cur_frame):
        """
        Applies a temporal filter (exponential moving average) to reduce noise and flicker in a video stream.

        Args:
            video_path (str, optional): Path to the video file. If None, uses the default webcam.
            alpha (float): The weighting factor for the current frame (0.0 to 1.0).
                        Higher alpha means less smoothing, more responsiveness to changes.
                        Lower alpha means more smoothing, less responsiveness.
        """

        # Apply the temporal filter (Exponential Moving Average)
        # filtered_frame = alpha * current_frame_float + (1 - alpha) * filtered_frame
        # This formula directly updates the filtered_frame based on the new current_frame.
        # It's a low-pass filter in the time domain.
        filtered_frame = cv2.addWeighted(
            cur_frame.astype(np.float32),
            1 - self.temporal_filter.value,
            prev_frame.astype(np.float32),
            self.temporal_filter.value,
            0,
        )

        # Convert back to uint8 for display
        # return cv2.convertScaleAbs(filtered_frame).astype('uint8')
        return filtered_frame

    def apply_luma_feedback2(self, cur_frame, prev_frame):
        return luma_key(cur_frame, prev_frame, self.luma_mode.value, self.feedback_luma_threshold.value)
    
    # TODO: this is a duplicate function; find way to reuse in mixer
    def apply_luma_feedback(self, prev_frame, cur_frame):
        # PERFORMANCE: Early exit if threshold is 0 (no effect)
        if self.feedback_luma_threshold.value == 0:
            return cur_frame

        # PERFORMANCE: Convert to uint8 only once, reuse for both frames
        # Also check if already uint8 to avoid unnecessary conversion
        if cur_frame.dtype != np.uint8:
            cur_frame_int = cur_frame.astype(np.uint8)
        else:
            cur_frame_int = cur_frame

        if prev_frame.dtype != np.uint8:
            prev_frame_int = prev_frame.astype(np.uint8)
        else:
            prev_frame_int = prev_frame

        # PERFORMANCE: Use fast grayscale conversion (0.299*R + 0.587*G + 0.114*B is slower)
        gray = cv2.cvtColor(cur_frame_int, cv2.COLOR_BGR2GRAY)

        # PERFORMANCE: Use NumPy boolean indexing instead of bitwise ops (3x faster!)
        match LumaMode(self.luma_mode.value):
            case LumaMode.BLACK:
                mask_bool = gray < self.feedback_luma_threshold.value
            case LumaMode.WHITE:
                mask_bool = gray >= self.feedback_luma_threshold.value
            case _:
                log.warning("Invalid luma_mode; defaulting to WHITE.")
                mask_bool = gray >= self.feedback_luma_threshold.value

        # PERFORMANCE: Direct NumPy where() is much faster than bitwise_and + add
        # Expand mask to 3 channels for BGR
        mask_3d = mask_bool[:, :, np.newaxis]
        result = np.where(mask_3d, cur_frame_int, prev_frame_int)

        return result.astype(np.float32)


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

        return noisy_frame.astype('uint8')
