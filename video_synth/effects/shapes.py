import cv2
import numpy as np

from effects.base import EffectBase
from effects.enums import Shape
from common import Widget

class Shapes:

    def __init__(self, params, width, height, shape_x_shift=0, shape_y_shift=0, group=None):
        self.params = params
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        subgroup = self.__class__.__name__

        
        self.shape_type = params.add("shape_type",
                                      min=0, max=len(Shape)-1, default=Shape.RECTANGLE.value,
                                      group=group, subgroup=subgroup,
                                      type=Widget.DROPDOWN, options=Shape)  # Shape type enum

        self.line_h = params.add("line_hue",
                                 min=0, max=179, default=0,
                                 subgroup=subgroup, group=group)  # Hue range for OpenCV is 0-
        self.line_s = params.add("line_sat",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)  # Saturation range
        self.line_v = params.add("line_val",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)  # Value range

        self.line_hsv = [self.line_h, self.line_s, self.line_v]  # H, S, V (Red) - will be converted to BGR
        self.line_weight = params.add("line_weight",
                                      min=1, max=20, default=2,
                                      subgroup=subgroup, group=group)  # Thickness of the shape outline, must be integer
        self.line_opacity = params.add("line_opacity",
                                       min=0.0, max=1.0, default=0.66,
                                       subgroup=subgroup, group=group)  # Opacity of the shape outline
        self.line_color = self._hsv_to_bgr(self.line_hsv)

        self.size_multiplier = params.add("size_multiplier",
                                          min=0.1, max=10.0, default=0.9,
                                          subgroup=subgroup, group=group)  # Scale factor for shape size
        self.aspect_ratio = params.add("aspect_ratio",
                                       min=0.1, max=10.0, default=1.0,
                                       subgroup=subgroup, group=group)  # Scale factor for shape size
        self.rotation_angle = params.add("rotation_angle",
                                         min=0, max=360, default=0,
                                         subgroup=subgroup, group=group)  # Rotation angle in degrees

        self.shape_x_shift = params.add("shape_x_shift",
                                        min=-width, max=width, default=shape_x_shift,
                                        subgroup=subgroup, group=group)  # Allow negative shifts
        self.shape_y_shift = params.add("shape_y_shift",
                                        min=-height, max=height, default=shape_y_shift,
                                        subgroup=subgroup, group=group)

        self.multiply_grid_x = params.add("multiply_grid_x",
                                          min=1, max=10, default=2,
                                          subgroup=subgroup, group=group)  # Number of shapes in X direction
        self.multiply_grid_y = params.add("multiply_grid_y",
                                          min=1, max=10, default=2,
                                          subgroup=subgroup, group=group)  # Number of shapes in Y direction
        self.grid_pitch_x = params.add("grid_pitch_x",
                                       min=0, max=width, default=100,
                                       subgroup=subgroup, group=group)  # Distance between shapes in X direction
        self.grid_pitch_y = params.add("grid_pitch_y",
                                       min=0, max=height, default=100,
                                       subgroup=subgroup, group=group)  # Distance between shapes in Y direction

        self.fill_enabled = True  # Toggle fill on/off
        self.fill_h = params.add("fill_hue",
                                 min=0, max=179, default=120,
                                 subgroup=subgroup, group=group)  # Hue for fill color
        self.fill_s = params.add("fill_sat",
                                 min=0, max=255, default=100,
                                 subgroup=subgroup, group=group)  # Saturation for fill color
        self.fill_v = params.add("fill_val",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)  # Value for fill color
        self.fill_hsv = [self.fill_h.value, self.fill_s.value, self.fill_v.value]  # H, S, V (Blue) - will be converted to BGR
        self.fill_opacity = params.add("fill_opacity",
                                       min=0.0, max=1.0, default=0.25,
                                       subgroup=subgroup, group=group)
        self.fill_color = self._hsv_to_bgr(self.fill_hsv)

        self.convas_rotation = params.add("canvas_rotation",
                                          min=0, max=360, default=0,
                                          subgroup=subgroup, group=group)  # Rotation angle in degrees
        
        

    def _draw_rectangle(self, canvas, center_x, center_y,):
        """ Draw a rotated rectangle on the canvas """

        rect_width = int(50 * self.size_multiplier.value * self.aspect_ratio.value)
        rect_height = int(50 * self.size_multiplier.value)

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.value, 1)

        # Define the rectangle's corners before rotation
        pts = np.array([
            [center_x - rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y + rect_height // 2],
            [center_x - rect_width // 2, center_y + rect_height // 2]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Apply the rotation
        rotated_pts = cv2.transform(pts, M)
        rotated_pts_int = np.int32(rotated_pts)

        if self.fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], self.fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.value)
        
        return canvas

    def _draw_circle(self, canvas, center_x, center_y):
        """ Draw a circle on the canvas """

        radius = int(30 * self.size_multiplier.value)
        if self.fill_enabled:
            cv2.circle(canvas, (center_x, center_y), radius, self.fill_color, -1) # -1 for fill
        cv2.circle(canvas, (center_x, center_y), radius, self.line_color, self.line_weight.value)

        return canvas
    
    def _draw_triangle(self, canvas, center_x, center_y):
        """ Draw a rotated triangle on the canvas """

        side_length = int(60 * self.size_multiplier.value)

        # Vertices for an equilateral triangle centered at (0,0)
        p1_x = 0
        p1_y = -side_length // 2
        p2_x = -int(side_length * np.sqrt(3) / 4)
        p2_y = side_length // 4
        p3_x = int(side_length * np.sqrt(3) / 4)
        p3_y = side_length // 4

        pts = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]], dtype=np.float32)

        # Translate to the current center
        pts[:, 0] += center_x
        pts[:, 1] += center_y

        # Apply rotation
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.value, 1)
        rotated_pts = cv2.transform(pts.reshape(-1, 1, 2), M)
        rotated_pts_int = np.int32(rotated_pts)

        if self.fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], self.fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.value)

        return canvas
    
    def _draw_line(self, canvas, center_x, center_y):
        length = int(50 * self.size_multiplier.value)
        start_point = (center_x - length // 2, center_y)
        end_point = (center_x + length // 2, center_y)

        if self.fill_enabled:
            cv2.line(canvas, start_point, end_point, self.fill_color, self.line_weight.value)
        cv2.line(canvas, start_point, end_point, self.line_color, self.line_weight.value)

        return canvas

    def _draw_shape_on_canvas(self, canvas, center_x, center_y):
        """ Draw the selected shape on the canvas at the specified center coordinates """

        # Ensure coordinates are within bounds to prevent errors
        # Note: These checks are for safety but may clip shapes if they go way off screen
        center_x = max(0, min(canvas.shape[1], center_x))
        center_y = max(0, min(canvas.shape[0], center_y))

        if self.shape_type.value == Shape.NONE:
            pass
        elif self.shape_type.value == Shape.RECTANGLE:
            canvas = self._draw_rectangle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.CIRCLE:
            canvas = self._draw_circle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.TRIANGLE:
            canvas = self._draw_triangle(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.LINE:
            canvas = self._draw_line(canvas, center_x, center_y)
        elif self.shape_type.value == Shape.DIAMOND:
            pass
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type.value}. Must be 'rectangle', 'circle', 'triangle', or 'line'.")
        
        return canvas

    def _blend_rgba_overlay(self, background, overlay_rgba):
        """
        Blend a 4-channel BGRA overlay onto a 3-channel BGR background using the alpha channel.
        Args:
            background (np.ndarray): 3-channel BGR background image.
            overlay_rgba (np.ndarray): 4-channel BGRA overlay image.
        Returns:
            np.ndarray: Blended 3-channel BGR image.
        """
        # Ensure background is float for calculation, then convert back to uint8
        background_float = background.astype(np.float32)

        # Normalize alpha channel (0-255 to 0.0-1.0)
        alpha = overlay_rgba[:, :, 3] / 255.0
        alpha_rgb = np.stack([alpha, alpha, alpha], axis=2) # Convert alpha to 3 channels for element-wise multiplication

        # Blend calculation: (foreground * alpha) + (background * (1 - alpha))
        # Note: overlay_rgba[:,:,:3] extracts the BGR channels of the overlay
        blended_image = (overlay_rgba[:,:,:3].astype(np.float32) * alpha_rgb) + \
                        (background_float * (1.0 - alpha_rgb))

        # Clip values to 0-255 and convert back to uint8
        return np.clip(blended_image, 0, 255).astype(np.uint8)

    # TODO: create a common HSV to BGR conversion function
    def _hsv_to_bgr(self, hsv):
        """ Convert HSV color to BGR color for OpenCV drawing functions """

        hsv_np = np.uint8([[hsv]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]

        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    
    # TODO: make public for testing or after fix
    # TODO: fix bug where shape hue affects the entire frame hue
    def _draw_shapes_on_frame(self, frame):
        """ Draw shapes on the given frame based on current parameters """

        base_center_x, base_center_y = self.width // 2 + self.shape_x_shift.value, self.height // 2 + self.shape_y_shift.value

        # Create separate 3-channel (BGR) canvases for lines and fills
        temp_line_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        temp_fill_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.line_color = self._hsv_to_bgr([self.line_h.value, self.line_s.value, self.line_v.value])
        self.fill_color = self._hsv_to_bgr([self.fill_h.value, self.fill_s.value, self.fill_v.value])

        # Grid Multiplication Mode
        for row in range(self.multiply_grid_y.value):
            for col in range(self.multiply_grid_x.value):
                # Calculate center for each shape in the grid
                current_center_x = base_center_x + col * self.grid_pitch_x.value - (self.multiply_grid_x.value - 1) * self.grid_pitch_x.value // 2
                current_center_y = base_center_y + row * self.grid_pitch_y.value - (self.multiply_grid_y.value - 1) * self.grid_pitch_y.value // 2

                self._draw_shape_on_canvas(temp_line_canvas, current_center_x, current_center_y)

            # If fill is enabled, apply its opacity in a separate blend
            if self.fill_enabled:
                self._draw_shape_on_canvas(temp_fill_canvas, current_center_x, current_center_y)

        # Line Overlay (BGRA)
        line_overlay_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        line_overlay_rgba[:,:,:3] = temp_line_canvas # Copy BGR data
        # Create an alpha mask: pixels are opaque (255) where sum of BGR is > 0 (not black)
        # Then scale by line_opacity
        line_alpha_mask = (temp_line_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.line_opacity * 255)
        line_overlay_rgba[:,:,3] = line_alpha_mask

        # Fill Overlay (BGRA)
        if self.fill_enabled:
            fill_overlay_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            fill_overlay_rgba[:,:,:3] = temp_fill_canvas # Copy BGR data
            # Create an alpha mask for fills
            fill_alpha_mask = (temp_fill_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.fill_opacity * 255)
            fill_overlay_rgba[:,:,3] = fill_alpha_mask

        # blend the layers using our custom self._blend_rgba_overlay method
        if self.fill_enabled:
            frame = self._blend_rgba_overlay(frame, fill_overlay_rgba)

        frame = self._blend_rgba_overlay(frame, line_overlay_rgba)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        return frame
