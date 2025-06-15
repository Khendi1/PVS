import cv2
import numpy as np
from param import Param
import config as p

class ShapeGenerator:
    def __init__(self, width, height, start_shift_x=0, start_shift_y=0):
        self.start_shift_x = p.add_param("start_shift_x", -width, width, start_shift_x)  # Allow negative shifts
        self.start_shift_y = p.add_param("start_shift_y", -height, height, start_shift_y)
        self.center_x = width // 2
        self.center_y = height // 2
        self.fill_enabled = True  # Toggle fill on/off
        self.shape_type = 'rectangle'  # 'rectangle', 'circle', 'triangle', 'line'
        self.line_h = p.add_param("line_hue", 0, 179, 0)  # Hue range for OpenCV is 0-
        self.line_s = p.add_param("line_saturation", 0, 255, 255)  # Saturation range
        self.line_v = p.add_param("line_value", 0, 255, 255)  # Value range
        self.line_hsv = [self.line_h.value, self.line_v.value, self.line_s.value]  # H, S, V (Red) - will be converted to BGR
        self.size_multiplier = p.add_param("size_multiplier", 0.1, 10.0, 1.0)  # Scale factor for shape size
        self.aspect_ratio = p.add_param("aspect_ratio", 0.1, 10.0, 1.0)  # Scale factor for shape size
        self.rotation_angle = p.add_param("rotation_angle", 0, 360, 0)  # Rotation angle in degrees
        self.multiply_grid_x = p.add_param("multiply_grid_x", 1, 10, 2)  # Number of shapes in X direction
        self.multiply_grid_y = p.add_param("multiply_grid_y", 1, 10, 2)  # Number of shapes in Y direction
        self.grid_pitch_x = p.add_param("grid_pitch_x", min_val=10, max_val=200, default_val=50)  # Distance between shapes in X direction
        self.grid_pitch_y = p.add_param("grid_pitch_y", min_val=10, max_val=200, default_val=50)  # Distance between shapes in Y direction
        self.line_weight = p.add_param("line_weight", 1, 20, 5)  # Thickness of the shape outline, must be integer
        self.line_opacity = p.add_param("line_opacity", 0.0, 1.0, 1.0)  # Opacity of the shape outline
        self.fill_h = p.add_param("fill_hue", 0, 179, 120)  # Hue for fill color
        self.fill_s = p.add_param("fill_saturation", 0, 255, 255)  # Saturation for fill color
        self.fill_v = p.add_param("fill_value", 0, 255, 255)  # Value for fill color
        self.fill_hsv = [self.fill_h.value, self.fill_s.value, self.fill_v.value]  # H, S, V (Blue) - will be converted to BGR
        self.fill_opacity = p.add_param("fill_opacity", 0.0, 1.0, 0.5)
        self.fill_color = self.hsv_to_bgr(self.fill_hsv)
        self.line_color = self.hsv_to_bgr(self.line_hsv)

        print("\n\n\nPress 'q' to quit.")
        print("\n--- General Controls ---")
        print(" '1': Rectangle")
        print(" '2': Circle")
        print(" '3': Triangle")
        print(" '4': Line")
        print(" '+': Increase size")
        print(" '-': Decrease size")
        print(" 'r': Rotate clockwise")
        print(" 'R': Rotate counter-clockwise")
        print("\n--- Position Shift ---")
        print(" 'w': Shift Up")
        print(" 's': Shift Down")
        print(" 'a': Shift Left")
        print(" 'd': Shift Right")
        print("\n--- Line Properties ---")
        print(" '[': Decrease line weight")
        print(" ']': Increase line weight")
        print(" 'h': Line Hue +")
        print(" 'H': Line Hue -")
        print(" 'j': Line Saturation +")
        print(" 'J': Line Saturation -")
        print(" 'k': Line Value +")
        print(" 'K': Line Value -")
        print(" 'o': Line Opacity +")
        print(" 'O': Line Opacity -")
        print("\n--- Fill Properties ---")
        print(" 'f': Toggle Fill (On/Off)")
        print(" 'g': Fill Hue +")
        print(" 'G': Fill Hue -")
        print(" 'i': Fill Saturation +")
        print(" 'I': Fill Saturation -")
        print(" 'l': Fill Value +")
        print(" 'L': Fill Value -")
        print(" 'p': Fill Opacity +")
        print(" 'P': Fill Opacity -")
        print("\n--- Grid Multiplication ---")
        print(" 'u': Increase Grid X count")
        print(" 'U': Decrease Grid X count")
        print(" 'v': Increase Grid Y count")
        print(" 'V': Decrease Grid Y count")
        print(" 'c': Increase Grid X pitch")
        print(" 'C': Decrease Grid X pitch")
        print(" 'z': Increase Grid Y pitch")
        print(" 'Z': Decrease Grid Y pitch")

    def draw_rectangle(self, canvas, center_x, center_y,):

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

    def draw_circle(self, canvas, center_x, center_y):
        radius = int(30 * self.size_multiplier.value)
        if self.fill_enabled:
            cv2.circle(canvas, (center_x, center_y), radius, self.fill_color, -1) # -1 for fill
        cv2.circle(canvas, (center_x, center_y), radius, self.line_color, self.line_weight.value)

        return canvas
    
    def draw_triangle(self, canvas, center_x, center_y):
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
    
    def draw_line(self, canvas, center_x, center_y):
        length = int(50 * self.size_multiplier.value)
        start_point = (center_x - length // 2, center_y)
        end_point = (center_x + length // 2, center_y)

        if self.fill_enabled:
            cv2.line(canvas, start_point, end_point, self.fill_color, self.line_weight.value)
        cv2.line(canvas, start_point, end_point, self.line_color, self.line_weight.value)

        return canvas

    def draw_shape_on_canvas(self, canvas, center_x, center_y):

        # Ensure coordinates are within bounds to prevent errors
        # Note: These checks are for safety but may clip shapes if they go way off screen
        center_x = max(0, min(canvas.shape[1], center_x))
        center_y = max(0, min(canvas.shape[0], center_y))

        if self.shape_type == 'rectangle':
            canvas = self.draw_rectangle(canvas, center_x, center_y)
        elif self.shape_type == 'circle':
            canvas = self.draw_circle(canvas, center_x, center_y)
        elif self.shape_type == 'triangle':
            canvas = self.draw_triangle(canvas, center_x, center_y)
        elif self.shape_type == 'line':
            canvas = self.draw_line(canvas, center_x, center_y)
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type}. Must be 'rectangle', 'circle', 'triangle', or 'line'.")
        
        return canvas

    # Helper function to blend a 4-channel BGRA overlay onto a 3-channel BGR background
    def blend_rgba_overlay(self, background, overlay_rgba):
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

    def hsv_to_bgr(self, hsv):
        # Convert H, S, V to BGR
        hsv_np = np.uint8([[hsv]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    
    def draw_shapes_on_frame(self, frame, width, height):
        base_center_x, base_center_y = width // 2 + self.start_shift_x.value, height // 2 + self.start_shift_y.value

        # # Create a completely transparent overlay to draw all shapes onto
        # # This is crucial for accumulated drawing before final blending
        # overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # Create separate 3-channel (BGR) canvases for lines and fills
        # These will temporarily hold the shapes on a black background
        temp_line_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        temp_fill_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.line_color = self.hsv_to_bgr([self.line_h.value, self.line_s.value, self.line_v.value])
        self.fill_color = self.hsv_to_bgr([self.fill_h.value, self.fill_s.value, self.fill_v.value])

        # Grid Multiplication Mode
        for row in range(self.multiply_grid_y.value):
            for col in range(self.multiply_grid_x.value):
                # Calculate center for each shape in the grid
                # The subtraction ensures the grid is centered relative to base_center_x/y
                current_center_x = base_center_x + col * self.grid_pitch_x.value - (self.multiply_grid_x.value - 1) * self.grid_pitch_x.value // 2
                current_center_y = base_center_y + row * self.grid_pitch_y.value - (self.multiply_grid_y.value - 1) * self.grid_pitch_y.value // 2

                self.draw_shape_on_canvas(temp_line_canvas, current_center_x, current_center_y)

        # # Apply the overlay onto the original frame with the specified opacity  
        # # The line and fill opacities are blended on the overlay itself in draw_shape_on_canvas
        # # Now, blend the entire overlay (with all shapes drawn) onto the frame
        # cv2.addWeighted(overlay, self.line_opacity, frame, 1.0 - self.line_opacity, 0, frame)
        # If fill is enabled, apply its opacity in a separate blend
        # This will make filled areas appear with their opacity
            if self.fill_enabled:
                    # line_weight=0 and line_color=(0,0,0) ensures only fills are drawn
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

        # 3. Blend the layers onto the display_frame using our custom self.blend_rgba_overlay function.
        #    Order matters if shapes overlap: typically fills then lines for outlining.
        if self.fill_enabled:
            frame = self.blend_rgba_overlay(frame, fill_overlay_rgba)

        frame = self.blend_rgba_overlay(frame, line_overlay_rgba)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        return frame

    def keyboard_controls(self, key):
        """
        Handle keyboard controls for shape manipulation.
        """
        if key == ord('1'):
            self.shape_type = 'rectangle'
            # print("Shape set to rectangle")
        elif key == ord('2'):
            self.shape_type = 'circle'
            # print("Shape set to circle")
        elif key == ord('3'):
            self.shape_type = 'triangle'
            # print("Shape set to triangle")
        elif key == ord('4'):
            self.shape_type = 'line'
            # print("Shape set to line")
        elif key == ord('+'):
            self.size_multiplier.value += 0.1
            # print(f"Size multiplier increased to {self.size_multiplier}")
        elif key == ord('-'):
            self.size_multiplier.value = max(0.1, self.size_multiplier.value - 0.1)
            # print(f"Size multiplier decreased to {self.size_multiplier}")
        elif key == ord('r'):
            self.rotation_angle.value = (self.rotation_angle.value + 10) % 360
            # print(f"Rotation angle set to {self.rotation_angle} degrees")
        elif key == ord('R'):
            self.rotation_angle.value = (self.rotation_angle.value - 10) % 360
            # print(f"Rotation angle set to {self.rotation_angle} degrees")
        # Position Shift
        elif key == ord('w'):
            self.start_shift_y.value -= 10
            # print(f"Shift Y position increased to {self.start_shift_y}")
        elif key == ord('s'):
           self.start_shift_y.value += 10
           # print(f"Shift Y position decreased to {self.start_shift_y}")
        elif key == ord('a'):
            self.start_shift_x.value -= 10
            # print(f"Shift X position increased to {self.start_shift_x}")
        elif key == ord('d'):
            self.start_shift_x.value += 10
            # print(f"Shift X position decreased to {self.start_shift_x}")
        # Line Properties
        elif key == ord('['):
            self.line_weight.value = max(1, self.line_weight - 1)
            # print(f"Line weight decreased to {self.line_weight}")
        elif key == ord(']'):
            self.line_weight.value += 1
            # print(f"Line weight increased to {self.line_weight}")
        elif key == ord('h'):
            self.line_hsv[0] = (self.line_hsv[0] + 5) % 180 # Hue is 0-179
            # print(f"Line hue increased to {self.line_hsv[0]}")
        elif key == ord('H'):
            self.line_hsv[0] = (self.line_hsv[0] - 5) % 180
            # print(f"Line hue decreased to {self.line_hsv[0]}")
        elif key == ord('j'):
            self.line_hsv[1] = min(255, self.line_hsv[1] + 5)
            # print(f"Line saturation increased to {self.line_hsv[1]}")
        elif key == ord('J'):
            self.line_hsv[1] = max(0, self.line_hsv[1] - 5)
            # print(f"Line saturation decreased to {self.line_hsv[1]}")
        elif key == ord('k'):
            self.line_hsv[2] = min(255, self.line_hsv[2] + 5)
            # print(f"Line value increased to {self.line_hsv[2]}")
        elif key == ord('K'):
            self.line_hsv[2] = max(0, self.line_hsv[2] - 5)
            # print(f"Line value decreased to {self.line_hsv[2]}")
        elif key == ord('o'):
            self.line_opacity.value = min(1.0, self.line_opacity.value + 0.05)
            # print(f"Line opacity increased to {self.line_opacity}")
        elif key == ord('O'):
            self.line_opacity.value = max(0.0, self.line_opacity.value - 0.05)
            # print(f"Line opacity decreased to {self.line_opacity}")
        # Fill Properties
        elif key == ord('f'):
            self.fill_enabled = not self.fill_enabled
        elif key == ord('g'):
            self.fill_hsv[0] = (self.fill_hsv[0] + 5) % 180
            # print(f"Fill hue increased to {self.fill_hsv[0]}")
        elif key == ord('G'):
            self.fill_hsv[0] = (self.fill_hsv[0] - 5) % 180
            # print(f"Fill hue decreased to {self.fill_hsv[0]}")
        elif key == ord('i'):
            self.fill_hsv[1] = min(255, self.fill_hsv[1] + 5)
            # print(f"Fill hue increased to {self.fill_hsv[1]}")
        elif key == ord('I'):
            self.fill_hsv[1] = max(0, self.fill_hsv[1] - 5)
            # print(f"Fill hue decreased to {self.fill_hsv[1]}")
        elif key == ord('l'):
            self.fill_hsv[2] = min(255, self.fill_hsv[2] + 5)
            # print(f"Fill hue increased to {self.fill_hsv[2]}")
        elif key == ord('L'):
            self.fill_hsv[2] = max(0, self.fill_hsv[2] - 5)
            # print(f"Fill hue decreased to {self.fill_hsv[2]}")
        elif key == ord('p'):
            self.fill_opacity = min(1.0, self.fill_opacity + 0.05)
            # print(f"Fill opacity increased to {self.fill_opacity}")
        elif key == ord('P'):
            self.fill_opacity = max(0.0, self.fill_opacity - 0.05)
            # print(f"Fill opacity decreased to {self.fill_opacity}")
        # Grid Mode Controls
        elif key == ord('u'):
            self.multiply_grid_x.value += 1
            # print(f"Grid X multiplier increased to {self.multiply_grid_x}")
        elif key == ord('U'):
            self.multiply_grid_x.value = max(1, self.multiply_grid_x.value - 1)
            # print(f"Grid X multiplier decreased to {self.multiply_grid_x}")
        elif key == ord('v'):
            self.multiply_grid_y.value += 1
            # print(f"Grid Y multiplier increased to {self.multiply_grid_y}")
        elif key == ord('V'):
            self.multiply_grid_y.value = max(1, self.multiply_grid_y.value - 1)
            # print(f"Grid Y multiplier decreased to {self.multiply_grid_y}")
        elif key == ord('c'):
            self.grid_pitch_x.value += 10
            # print(f"Grid X pitch increased to {self.grid_pitch_x}")
        elif key == ord('C'):
            self.grid_pitch_x.value -= 10
            # print(f"Grid X pitch decreased to {self.grid_pitch_x}")
        elif key == ord('z'):
            self.grid_pitch_y.value += 10
            # print(f"Grid Y pitch increased to {self.grid_pitch_y}")
        elif key == ord('Z'):
            self.grid_pitch_y.value -= 10
            # print(f"Grid Y pitch decreased to {self.grid_pitch_y}")