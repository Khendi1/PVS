import cv2
import numpy as np
import random

from effects.base import EffectBase
from common import Widget, Toggle
from effects.enums import BlendModes

class Glitch(EffectBase):   

    def __init__(self, params, group=None):
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

        # General glitch parameters
        subgroup_general = "Glitch_General"

        self.enable_pixel_shift = params.add("enable_pixel_shift",
                                              group=group, subgroup=subgroup_general,
                                              type=Widget.TOGGLE)
        self.enable_color_split = params.add("enable_color_split",
                                             group=group, subgroup=subgroup_general,
                                             type=Widget.TOGGLE)
        self.enable_block_corruption = params.add("enable_block_corruption",
                                                  group=group, subgroup=subgroup_general,
                                                  type=Widget.TOGGLE)
        self.enable_random_rectangles = params.add("enable_random_rectangles",
                                                   group=group, subgroup=subgroup_general,
                                                   type=Widget.TOGGLE)
        self.enable_horizontal_scroll_freeze = params.add("enable_horizontal_scroll_freeze",
                                                          group=group, subgroup=subgroup_general,
                                                          type=Widget.TOGGLE)

        self.glitch_duration_frames = params.add("glitch_duration_frames",
                                                 min=1, max=300, default=60,
                                                 subgroup=subgroup_general, group=group)
        self.glitch_intensity_max = params.add("glitch_intensity_max",
                                               min=0, max=100, default=50,
                                               subgroup=subgroup_general, group=group)
        self.glitch_block_size_max = params.add("glitch_block_size_max",
                                                min=0, max=200, default=60,
                                                subgroup=subgroup_general, group=group)
        self.band_div = params.add("glitch_band_div",
                                   min=1, max=10, default=5,
                                   subgroup=subgroup_general, group=group)
        self.num_glitches = params.add("num_glitches",
                                       min=0, max=100, default=0,
                                       group=group, subgroup=subgroup_general)
        self.glitch_size = params.add("glitch_size",
                                      min=1, max=100, default=0,
                                      group=group, subgroup=subgroup_general)

        # Slitscan parameters
        subgroup_slitscan = "Glitch_Slitscan"

        self.enable_slitscan = params.add("enable_slitscan",
                                          group=group, subgroup=subgroup_slitscan,
                                          type=Widget.TOGGLE)
        self.ss_dir = params.add("slitscan_direction",
                                 min=0, max=1, default=0,
                                 group=group, subgroup=subgroup_slitscan,
                                 type=Widget.TOGGLE)
        self.ss_slice_width = params.add("slitscan_slice_width",
                                         min=1, max=50, default=5,
                                         group=group, subgroup=subgroup_slitscan)
        self.ss_time_offset = params.add("slitscan_time_offset",
                                         min=1, max=60, default=10,
                                         group=group, subgroup=subgroup_slitscan)
        self.ss_speed = params.add("slitscan_speed",
                                   min=0.1, max=10.0, default=1.0,
                                   group=group, subgroup=subgroup_slitscan)
        self.ss_reverse = params.add("slitscan_reverse",
                                     min=0, max=1, default=0,
                                     group=group, subgroup=subgroup_slitscan,
                                     type=Widget.TOGGLE)
        self.ss_buffer_size = params.add("slitscan_buffer_size",
                                         min=10, max=120, default=60,
                                         group=group, subgroup=subgroup_slitscan)
        self.ss_blend_mode = params.add("slitscan_blend_mode",
                                        min=0, max=2, default=0,
                                        group=group, subgroup=subgroup_slitscan,
                                        type=Widget.DROPDOWN, options=BlendModes)
        self.ss_blend_alpha = params.add("slitscan_blend_alpha",
                                         min=0.0, max=1.0, default=1.0,
                                         group=group, subgroup=subgroup_slitscan,
                                         type=Widget.SLIDER)
        self.ss_position_offset = params.add("slitscan_position_offset",
                                             min=-100, max=100, default=0,
                                             group=group, subgroup=subgroup_slitscan)
        self.ss_wobble_amount = params.add("slitscan_wobble_amount",
                                           min=0, max=50, default=0,
                                           group=group, subgroup=subgroup_slitscan)
        self.ss_wobble_freq = params.add("slitscan_wobble_freq",
                                         min=0.1, max=10.0, default=1.0,
                                         group=group, subgroup=subgroup_slitscan)

        # Echo/Stutter parameters
        subgroup_echo = "Glitch_Echo"

        self.enable_echo = params.add("enable_echo",
                                      group=group, subgroup=subgroup_echo,
                                      type=Widget.TOGGLE)
        self.echo_probability = params.add("echo_probability",
                                           min=0.0, max=1.0, default=0.1,
                                           group=group, subgroup=subgroup_echo)
        self.echo_buffer_size = params.add("echo_buffer_size",
                                           min=5, max=60, default=30,
                                           group=group, subgroup=subgroup_echo)
        self.echo_freeze_min = params.add("echo_freeze_min",
                                          min=1, max=30, default=2,
                                          group=group, subgroup=subgroup_echo)
        self.echo_freeze_max = params.add("echo_freeze_max",
                                          min=2, max=60, default=10,
                                          group=group, subgroup=subgroup_echo)
        self.echo_blend_amount = params.add("echo_blend_amount",
                                            min=0.0, max=1.0, default=1.0,
                                            group=group, subgroup=subgroup_echo)

        # Slitscan frame buffer
        self.ss_buffer = []
        self.ss_position = 0

        # Echo/Stutter frame buffer and state
        self.echo_buffer = []
        self._echo_frozen_frame = None
        self._echo_freeze_counter = 0

        self.frame_count = 0

    def _create_buttons(self, gui):
        dpg.add_button(
            label=self.enable_pixel_shift.label,
            callback=self.enable_pixel_shift.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_color_split.label,
            callback=self.enable_color_split.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_block_corruption.label,
            callback=self.enable_block_corruption.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_random_rectangles.label,
            callback=self.enable_random_rectangles.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_horizontal_scroll_freeze.label,
            callback=self.enable_horizontal_scroll_freeze.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_slitscan.label,
            callback=self.enable_slitscan.toggle,
            group=gui,
        )
        dpg.add_button(
            label=self.enable_echo.label,
            callback=self.enable_echo.toggle,
            group=gui,
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

    def _apply_slitscan(self, frame):
        """
        Applies a slitscan effect by compositing slices from different frames in the buffer.

        Slitscan works by taking horizontal or vertical slices from frames at different points
        in time and combining them into a single output frame, creating time-displacement effects.
        """
        height, width, _ = frame.shape

        # Add current frame to buffer
        self.ss_buffer.append(frame.copy())

        # Maintain buffer size
        max_buffer_size = int(self.ss_buffer_size.value)
        if len(self.ss_buffer) > max_buffer_size:
            self.ss_buffer.pop(0)

        # If buffer isn't full yet, return original frame
        if len(self.ss_buffer) < 2:
            return frame

        # Create output frame
        output = np.zeros_like(frame)

        # Get parameters
        direction = int(self.ss_dir.value)  # 0=horizontal, 1=vertical
        slice_width = int(self.ss_slice_width.value)
        time_offset = int(self.ss_time_offset.value)
        speed = self.ss_speed.value
        reverse = bool(self.ss_reverse.value)
        blend_mode = int(self.ss_blend_mode.value)
        blend_alpha = self.ss_blend_alpha.value
        position_offset = int(self.ss_position_offset.value)
        wobble_amount = int(self.ss_wobble_amount.value)
        wobble_freq = self.ss_wobble_freq.value

        # Update scanning position
        self.ss_position += speed

        buffer_len = len(self.ss_buffer)

        if direction == 0:  # Horizontal slitscan
            num_slices = height // slice_width + 1

            for i in range(num_slices):
                # Calculate slice position with wobble
                slice_y = int(i * slice_width)
                if wobble_amount > 0:
                    wobble_offset = int(wobble_amount * np.sin(self.ss_position * wobble_freq * 0.1 + i * 0.5))
                    slice_y = max(0, min(height - slice_width, slice_y + wobble_offset))

                slice_y_end = min(slice_y + slice_width, height)

                # Calculate which frame to pull this slice from
                if reverse:
                    time_index = (i * time_offset + int(self.ss_position) + position_offset) % buffer_len
                else:
                    time_index = (num_slices - i - 1) * time_offset + int(self.ss_position) + position_offset
                    time_index = time_index % buffer_len

                # Clamp to buffer bounds
                time_index = max(0, min(buffer_len - 1, time_index))

                # Get slice from historical frame
                source_frame = self.ss_buffer[time_index]
                slice_data = source_frame[slice_y:slice_y_end, :]

                # Apply blend mode
                if blend_mode == 0:  # Replace
                    output[slice_y:slice_y_end, :] = slice_data
                elif blend_mode == 1:  # Blend
                    output[slice_y:slice_y_end, :] = cv2.addWeighted(
                        output[slice_y:slice_y_end, :], 1 - blend_alpha,
                        slice_data, blend_alpha, 0
                    )
                elif blend_mode == 2:  # Additive
                    output[slice_y:slice_y_end, :] = np.clip(
                        output[slice_y:slice_y_end, :].astype(np.int16) +
                        (slice_data.astype(np.int16) * blend_alpha).astype(np.int16),
                        0, 255
                    ).astype(np.uint8)

        else:  # Vertical slitscan
            num_slices = width // slice_width + 1

            for i in range(num_slices):
                # Calculate slice position with wobble
                slice_x = int(i * slice_width)
                if wobble_amount > 0:
                    wobble_offset = int(wobble_amount * np.sin(self.ss_position * wobble_freq * 0.1 + i * 0.5))
                    slice_x = max(0, min(width - slice_width, slice_x + wobble_offset))

                slice_x_end = min(slice_x + slice_width, width)

                # Calculate which frame to pull this slice from
                if reverse:
                    time_index = (i * time_offset + int(self.ss_position) + position_offset) % buffer_len
                else:
                    time_index = (num_slices - i - 1) * time_offset + int(self.ss_position) + position_offset
                    time_index = time_index % buffer_len

                # Clamp to buffer bounds
                time_index = max(0, min(buffer_len - 1, time_index))

                # Get slice from historical frame
                source_frame = self.ss_buffer[time_index]
                slice_data = source_frame[:, slice_x:slice_x_end]

                # Apply blend mode
                if blend_mode == 0:  # Replace
                    output[:, slice_x:slice_x_end] = slice_data
                elif blend_mode == 1:  # Blend
                    output[:, slice_x:slice_x_end] = cv2.addWeighted(
                        output[:, slice_x:slice_x_end], 1 - blend_alpha,
                        slice_data, blend_alpha, 0
                    )
                elif blend_mode == 2:  # Additive
                    output[:, slice_x:slice_x_end] = np.clip(
                        output[:, slice_x:slice_x_end].astype(np.int16) +
                        (slice_data.astype(np.int16) * blend_alpha).astype(np.int16),
                        0, 255
                    ).astype(np.uint8)

        return output

    def _apply_echo_stutter(self, frame):
        """
        Creates echo/stutter effects by randomly freezing and replaying frames from buffer.

        This effect maintains a buffer of recent frames and occasionally freezes playback,
        replaying a frame from the buffer for a random duration, creating rhythmic stuttering
        and temporal glitching effects.
        """
        # Add current frame to buffer
        self.echo_buffer.append(frame.copy())

        # Maintain buffer size
        max_buffer_size = int(self.echo_buffer_size.value)
        if len(self.echo_buffer) > max_buffer_size:
            self.echo_buffer.pop(0)

        # If buffer isn't full yet, return original frame
        if len(self.echo_buffer) < 2:
            return frame

        # Check if we should trigger a new freeze
        if self._echo_freeze_counter <= 0 and np.random.random() < self.echo_probability.value:
            # Grab a random frame from the buffer
            buffer_depth = len(self.echo_buffer)
            random_idx = np.random.randint(0, buffer_depth)
            self._echo_frozen_frame = self.echo_buffer[random_idx].copy()

            # Determine how long to hold this frame
            freeze_min = int(self.echo_freeze_min.value)
            freeze_max = int(self.echo_freeze_max.value)
            # Ensure max is always >= min
            if freeze_max < freeze_min:
                freeze_max = freeze_min
            self._echo_freeze_counter = np.random.randint(freeze_min, freeze_max + 1)

        # If we're in a freeze, return the frozen frame (optionally blended)
        if self._echo_freeze_counter > 0:
            self._echo_freeze_counter -= 1

            if self._echo_frozen_frame is not None:
                # Blend frozen frame with current frame based on blend_amount
                blend_amount = self.echo_blend_amount.value
                if blend_amount >= 0.99:  # Full replacement
                    return self._echo_frozen_frame
                else:  # Partial blend
                    return cv2.addWeighted(
                        frame.astype(np.float32), 1 - blend_amount,
                        self._echo_frozen_frame.astype(np.float32), blend_amount,
                        0
                    ).astype(frame.dtype)

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
        if self.enable_slitscan.value:
            frame = self._apply_slitscan(frame)
        if self.enable_echo.value:
            frame = self._apply_echo_stutter(frame)

        self.frame_count += 1
        return frame
