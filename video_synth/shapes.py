import cv2
import numpy as np
from config import params, image_width, image_height
from enum import IntEnum
from sliders import TrackbarRow
import dearpygui.dearpygui as dpg

class Shape(IntEnum):

    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2
    LINE = 3
    DIAMOND = 4
    NONE = 5

class ShapeGenerator:

    def __init__(self, width, height, shape_x_shift=0, shape_y_shift=0):
        
        self.shape_x_shift = params.add("shape_x_shift", -width, width, shape_x_shift)  # Allow negative shifts
        self.shape_y_shift = params.add("shape_y_shift", -height, height, shape_y_shift)
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.shape_type = params.add("shape_type", 0, len(Shape)-1, Shape.RECTANGLE)
        
        self.line_h = params.add("line_hue", 0, 179, 0)  # Hue range for OpenCV is 0-
        self.line_s = params.add("line_sat", 0, 255, 255)  # Saturation range
        self.line_v = params.add("line_val", 0, 255, 255)  # Value range
        self.line_hsv = [params.val("line_hue"), params.val("line_val"), params.val("line_sat")]  # H, S, V (Red) - will be converted to BGR
        self.line_weight = params.add("line_weight", 1, 20, 2)  # Thickness of the shape outline, must be integer
        self.line_opacity = params.add("line_opacity", 0.0, 1.0, 0.66)  # Opacity of the shape outline
        
        self.size_multiplier = params.add("size_multiplier", 0.1, 10.0, 0.9)  # Scale factor for shape size
        self.aspect_ratio = params.add("aspect_ratio", 0.1, 10.0, 1.0)  # Scale factor for shape size
        self.rotation_angle = params.add("rotation_angle", 0, 360, 0)  # Rotation angle in degrees
        
        self.multiply_grid_x = params.add("multiply_grid_x", 1, 10, 2)  # Number of shapes in X direction
        self.multiply_grid_y = params.add("multiply_grid_y", 1, 10, 2)  # Number of shapes in Y direction
        self.grid_pitch_x = params.add("grid_pitch_x", min=0, max=width, default_val=100)  # Distance between shapes in X direction
        self.grid_pitch_y = params.add("grid_pitch_y", min=0, max=height, default_val=100)  # Distance between shapes in Y direction
        
        self.fill_enabled = True  # Toggle fill on/off
        self.fill_h = params.add("fill_hue", 0, 179, 120)  # Hue for fill color
        self.fill_s = params.add("fill_sat", 0, 255, 100)  # Saturation for fill color
        self.fill_v = params.add("fill_val", 0, 255, 255)  # Value for fill color
        self.fill_hsv = [self.fill_h.val(), self.fill_s.val(), self.fill_v.val()]  # H, S, V (Blue) - will be converted to BGR
        self.fill_opacity = params.add("fill_opacity", 0.0, 1.0, 0.25)
        self.fill_color = self.hsv_to_bgr(self.fill_hsv)
        self.line_color = self.hsv_to_bgr(self.line_hsv)

        self.convas_rotation = params.add("canvas_rotation", 0, 360, 0)  # Rotation angle in degrees
        
        

    def draw_rectangle(self, canvas, center_x, center_y,):
        """ Draw a rotated rectangle on the canvas """

        rect_width = int(50 * self.size_multiplier.val() * self.aspect_ratio.val())
        rect_height = int(50 * self.size_multiplier.val())

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.val(), 1)

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
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.val())
        
        return canvas

    def draw_circle(self, canvas, center_x, center_y):
        """ Draw a circle on the canvas """

        radius = int(30 * self.size_multiplier.val())
        if self.fill_enabled:
            cv2.circle(canvas, (center_x, center_y), radius, self.fill_color, -1) # -1 for fill
        cv2.circle(canvas, (center_x, center_y), radius, self.line_color, self.line_weight.val())

        return canvas
    
    def draw_triangle(self, canvas, center_x, center_y):
        """ Draw a rotated triangle on the canvas """

        side_length = int(60 * self.size_multiplier.val())

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
        M = cv2.getRotationMatrix2D((center_x, center_y), self.rotation_angle.val(), 1)
        rotated_pts = cv2.transform(pts.reshape(-1, 1, 2), M)
        rotated_pts_int = np.int32(rotated_pts)

        if self.fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], self.fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, self.line_color, self.line_weight.val())

        return canvas
    
    def draw_line(self, canvas, center_x, center_y):
        length = int(50 * self.size_multiplier.val())
        start_point = (center_x - length // 2, center_y)
        end_point = (center_x + length // 2, center_y)

        if self.fill_enabled:
            cv2.line(canvas, start_point, end_point, self.fill_color, self.line_weight.val())
        cv2.line(canvas, start_point, end_point, self.line_color, self.line_weight.val())

        return canvas

    def draw_shape_on_canvas(self, canvas, center_x, center_y):
        """ Draw the selected shape on the canvas at the specified center coordinates """

        # Ensure coordinates are within bounds to prevent errors
        # Note: These checks are for safety but may clip shapes if they go way off screen
        center_x = max(0, min(canvas.shape[1], center_x))
        center_y = max(0, min(canvas.shape[0], center_y))

        if self.shape_type.val() == Shape.NONE:
            pass
        elif self.shape_type.val() == Shape.RECTANGLE:
            canvas = self.draw_rectangle(canvas, center_x, center_y)
        elif self.shape_type.val() == Shape.CIRCLE:
            canvas = self.draw_circle(canvas, center_x, center_y)
        elif self.shape_type.val() == Shape.TRIANGLE:
            canvas = self.draw_triangle(canvas, center_x, center_y)
        elif self.shape_type.val() == Shape.LINE:
            canvas = self.draw_line(canvas, center_x, center_y)
        elif self.shape_type.val() == Shape.DIAMOND:
            pass
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type.val()}. Must be 'rectangle', 'circle', 'triangle', or 'line'.")
        
        return canvas

    def blend_rgba_overlay(self, background, overlay_rgba):
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
    def hsv_to_bgr(self, hsv):
        """ Convert HSV color to BGR color for OpenCV drawing functions """

        hsv_np = np.uint8([[hsv]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]

        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    
    def draw_shapes_on_frame(self, frame, width, height):
        """ Draw shapes on the given frame based on current parameters """

        base_center_x, base_center_y = width // 2 + self.shape_x_shift.val(), height // 2 + self.shape_y_shift.val()

        # Create separate 3-channel (BGR) canvases for lines and fills
        temp_line_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        temp_fill_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.line_color = self.hsv_to_bgr([self.line_h.val(), self.line_s.val(), self.line_v.val()])
        self.fill_color = self.hsv_to_bgr([self.fill_h.val(), self.fill_s.val(), self.fill_v.val()])

        # Grid Multiplication Mode
        for row in range(self.multiply_grid_y.val()):
            for col in range(self.multiply_grid_x.val()):
                # Calculate center for each shape in the grid
                current_center_x = base_center_x + col * self.grid_pitch_x.val() - (self.multiply_grid_x.val() - 1) * self.grid_pitch_x.val() // 2
                current_center_y = base_center_y + row * self.grid_pitch_y.val() - (self.multiply_grid_y.val() - 1) * self.grid_pitch_y.val() // 2

                self.draw_shape_on_canvas(temp_line_canvas, current_center_x, current_center_y)

            # If fill is enabled, apply its opacity in a separate blend
            if self.fill_enabled:
                self.draw_shape_on_canvas(temp_fill_canvas, current_center_x, current_center_y)

        # Line Overlay (BGRA)
        line_overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        line_overlay_rgba[:,:,:3] = temp_line_canvas # Copy BGR data
        # Create an alpha mask: pixels are opaque (255) where sum of BGR is > 0 (not black)
        # Then scale by line_opacity
        line_alpha_mask = (temp_line_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.line_opacity * 255)
        line_overlay_rgba[:,:,3] = line_alpha_mask

        # Fill Overlay (BGRA)
        if self.fill_enabled:
            fill_overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            fill_overlay_rgba[:,:,:3] = temp_fill_canvas # Copy BGR data
            # Create an alpha mask for fills
            fill_alpha_mask = (temp_fill_canvas.sum(axis=2) > 0).astype(np.uint8) * int(self.fill_opacity * 255)
            fill_overlay_rgba[:,:,3] = fill_alpha_mask

        # blend the layers using our custom self.blend_rgba_overlay method
        if self.fill_enabled:
            frame = self.blend_rgba_overlay(frame, fill_overlay_rgba)

        frame = self.blend_rgba_overlay(frame, line_overlay_rgba)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        return frame
    

    def create_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tShape Generator", tag="shape_generator"):
            shape_slider = TrackbarRow(
                "Shape Type",
                params.get("shape_type"),
                default_font_id)
            
            canvas_rotation_slider = TrackbarRow(
                "Canvas Rotation", 
                params.get("canvas_rotation"), 
                default_font_id)

            size_multiplier_slider = TrackbarRow(
                "Size Multiplier", 
                params.get("size_multiplier"), 
                default_font_id)
            
            aspect_ratio_slider = TrackbarRow(
                "Aspect Ratio", 
                params.get("aspect_ratio"), 
                default_font_id)
            
            rotation_slider = TrackbarRow(
                "Rotation", 
                params.get("rotation_angle"), 
                default_font_id)
            
            multiply_grid_x_slider = TrackbarRow(
                "Multiply Grid X", 
                params.get("multiply_grid_x"), 
                default_font_id)
            
            multiply_grid_y_slider = TrackbarRow(
                "Multiply Grid Y", 
                params.get("multiply_grid_y"), 
                default_font_id)
            
            grid_pitch_x_slider = TrackbarRow(
                "Grid Pitch X", 
                params.get("grid_pitch_x"), 
                default_font_id)
            
            grid_pitch_y_slider = TrackbarRow(
                "Grid Pitch Y", 
                params.get("grid_pitch_y"), 
                default_font_id)
            
            shape_y_shift_slider = TrackbarRow(
                "Shape Y Shift", 
                params.get("shape_y_shift"), 
                default_font_id)
            
            shape_x_shift_slider = TrackbarRow(
                "Shape X Shift", 
                params.get("shape_x_shift"), 
                default_font_id)

            with dpg.collapsing_header(label=f"\Line Generator", tag="line_generator"):

                line_hue_slider = TrackbarRow(
                    "Line Hue", 
                    params.get("line_hue"), 
                    default_font_id)
                
                line_sat_slider = TrackbarRow(
                    "Line Sat", 
                    params.get("line_sat"), 
                        default_font_id)
                
                line_val_slider = TrackbarRow(
                    "Line Val", 
                    params.get("line_val"), 
                        default_font_id)
                
                line_weight_slider = TrackbarRow(
                    "Line Width", 
                    params.get("line_weight"), 
                        default_font_id)
                
                line_opacity_slider = TrackbarRow(
                    "Line Opacity", 
                    params.get("line_opacity"), 
                        default_font_id)
            dpg.bind_item_font("line_generator", global_font_id)

            with dpg.collapsing_header(label=f"\tFill Generator", tag="fill_generator"):
                fill_hue_slider = TrackbarRow(
                    "Fill Hue", 
                    params.get("fill_hue"), 
                        default_font_id)
                
                fill_sat_slider = TrackbarRow(
                    "Fill Sat", 
                    params.get("fill_sat"), 
                        default_font_id)
                
                fill_val_slider = TrackbarRow(
                    "Fill Val", 
                    params.get("fill_val"), 
                        default_font_id)
                
                fill_opacity_slider = TrackbarRow(
                    "Fill Opacity", 
                    params.get("fill_opacity"), 
                        default_font_id)
            dpg.bind_item_font("fill_generator", global_font_id)
        dpg.bind_item_font("shape_generator", global_font_id)
