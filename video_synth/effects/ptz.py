import cv2
import numpy as np

from effects.base import EffectBase

class PTZ(EffectBase):
    def __init__(self, params, image_width: int, image_height: int, group=None):
        subgroup = self.__class__.__name__
        self.params = params
        self.height = image_height
        self.width = image_width

        self.x_shift = params.add("x_shift",
                                   min=-image_width, max=image_width, default=0,
                                   subgroup=subgroup, group=group)  # min/max depends on image size
        self.y_shift = params.add("y_shift",
                                  min=-image_height, max=image_height, default=0,
                                  subgroup=subgroup, group=group)  # min/max depends on image size
        self.zoom = params.add("zoom",
                               min=0.75, max=3, default=1.0,
                               subgroup=subgroup, group=group)
        self.r_shift = params.add("r_shift",
                                  min=-360, max=360, default=0.0,
                                  subgroup=subgroup, group=group)


        self.prev_x_shift = params.add("prev_x_shift",
                                       min=-image_width, max=image_width, default=0,
                                       subgroup=subgroup, group=group)  # min/max depends on image size
        self.prev_y_shift = params.add("prev_y_shift",
                                       min=-image_height, max=image_height, default=0,
                                       subgroup=subgroup, group=group)  # min/max depends on image size
        self.prev_zoom = params.add("prev_zoom",
                                    min=0.75, max=3, default=1.0,
                                    subgroup=subgroup, group=group)
        self.prev_r_shift = params.add("prev_r_shift",
                                       min=-360, max=360, default=0.0,
                                       subgroup=subgroup, group=group)

        self.prev_cx = params.add("prev_cx",
                                  min=-image_width/2, max=image_width/2, default=0,
                                  subgroup=subgroup, group=group)
        self.prev_cy = params.add("prev_cy",
                                  min=-image_height/2, max=image_height/2, default=0,
                                  subgroup=subgroup, group=group)
        self.polar_x = params.add("polar_x",
                                  min=-image_width // 2, max=image_width // 2, default=0,
                                  subgroup=subgroup, group=group)
        self.polar_y = params.add("polar_y",
                                  min=-image_height // 2, max=image_height // 2, default=0,
                                  subgroup=subgroup, group=group)
        self.polar_radius = params.add("polar_radius",
                                       min=0.1, max=100, default=1.0,
                                       subgroup=subgroup, group=group)

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
        # If all parameters are at their default values, do nothing.
        if (self.x_shift.value == 0 and
            self.y_shift.value == 0 and
            self.r_shift.value == 0.0 and
            self.zoom.value == 1.0):
            return frame

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
        if (self.prev_x_shift.value == 0 and
            self.prev_y_shift.value == 0 and
            self.prev_r_shift.value == 0.0 and
            self.prev_zoom.value == 1.0 and
            self.prev_cx.value == 0 and
            self.prev_cy.value == 0):
            return frame

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
