import cv2
import numpy as np

# Helper function to blend a 4-channel BGRA overlay onto a 3-channel BGR background
def blend_rgba_overlay(background, overlay_rgba):
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


def draw_shapes_on_video():
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Initial shape parameters
    shape_type = 'rectangle'  # 'rectangle', 'circle', 'triangle', 'line'

    # Line properties
    line_hsv = [0, 255, 255]  # H, S, V (Red) - will be converted to BGR
    line_opacity = 1.0       # 0.0 to 1.0
    line_weight = 2

    # Fill properties
    fill_enabled = False
    fill_hsv = [120, 255, 255] # H, S, V (Blue) - will be converted to BGR
    fill_opacity = 0.5       # 0.0 to 1.0

    shape_size_multiplier = 1.0
    aspect_ratio = 1.0  # For rectangles/ellipses, width/height ratio
    rotation_angle = 0  # Degrees

    # Starting position shift
    start_shift_x = 0
    start_shift_y = 0

    # Grid multiplication mode
    multiply_grid_x = 1
    multiply_grid_y = 1
    grid_pitch_x = 100
    grid_pitch_y = 100

    print("Press 'q' to quit.")
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

    def hsv_to_bgr(hsv):
        # Convert H, S, V to BGR
        hsv_np = np.uint8([[hsv]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        height, width, _ = frame.shape
        base_center_x, base_center_y = width // 2 + start_shift_x, height // 2 + start_shift_y

        # Create separate 3-channel (BGR) canvases for lines and fills
        # These will temporarily hold the shapes on a black background
        temp_line_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        temp_fill_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        line_bgr_color = hsv_to_bgr(line_hsv)
        fill_bgr_color = hsv_to_bgr(fill_hsv)

        # Grid Multiplication Mode: Draw all shapes onto their respective temporary canvases
        for row in range(multiply_grid_y):
            for col in range(multiply_grid_x):
                current_center_x = base_center_x + col * grid_pitch_x - (multiply_grid_x - 1) * grid_pitch_x // 2
                current_center_y = base_center_y + row * grid_pitch_y - (multiply_grid_y - 1) * grid_pitch_y // 2

                # Draw lines onto temp_line_canvas
                # fill_enabled=False and fill_color=(0,0,0) ensures only lines are drawn
                draw_shape_on_canvas(temp_line_canvas, shape_type, current_center_x, current_center_y,
                                     shape_size_multiplier, aspect_ratio, rotation_angle,
                                     line_bgr_color, line_weight, False, (0,0,0))

                # Draw fills onto temp_fill_canvas if enabled
                if fill_enabled:
                    # line_weight=0 and line_color=(0,0,0) ensures only fills are drawn
                    draw_shape_on_canvas(temp_fill_canvas, shape_type, current_center_x, current_center_y,
                                         shape_size_multiplier, aspect_ratio, rotation_angle,
                                         (0,0,0), 0, True, fill_bgr_color)

        # --- Blending Strategy ---
        # 1. Start with a copy of the original frame as our display output
        display_frame = frame.copy()

        # 2. Create 4-channel BGRA overlays from our temporary 3-channel canvases
        #    and set their alpha channels based on their content and desired opacity.

        # Line Overlay (BGRA)
        line_overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        line_overlay_rgba[:,:,:3] = temp_line_canvas # Copy BGR data
        # Create an alpha mask: pixels are opaque (255) where sum of BGR is > 0 (not black)
        # Then scale by line_opacity
        line_alpha_mask = (temp_line_canvas.sum(axis=2) > 0).astype(np.uint8) * int(line_opacity * 255)
        line_overlay_rgba[:,:,3] = line_alpha_mask

        # Fill Overlay (BGRA)
        if fill_enabled:
            fill_overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            fill_overlay_rgba[:,:,:3] = temp_fill_canvas # Copy BGR data
            # Create an alpha mask for fills
            fill_alpha_mask = (temp_fill_canvas.sum(axis=2) > 0).astype(np.uint8) * int(fill_opacity * 255)
            fill_overlay_rgba[:,:,3] = fill_alpha_mask

        # 3. Blend the layers onto the display_frame using our custom blend_rgba_overlay function.
        #    Order matters if shapes overlap: typically fills then lines for outlining.
        if fill_enabled:
            display_frame = blend_rgba_overlay(display_frame, fill_overlay_rgba)

        display_frame = blend_rgba_overlay(display_frame, line_overlay_rgba)

        # Show the final blended frame
        cv2.imshow('Shape Generator', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        # General Controls
        elif key == ord('1'):
            shape_type = 'rectangle'
        elif key == ord('2'):
            shape_type = 'circle'
        elif key == ord('3'):
            shape_type = 'triangle'
        elif key == ord('4'):
            shape_type = 'line'
        elif key == ord('+'):
            shape_size_multiplier += 0.1
        elif key == ord('-'):
            shape_size_multiplier = max(0.1, shape_size_multiplier - 0.1)
        elif key == ord('r'):
            rotation_angle = (rotation_angle + 10) % 360
        elif key == ord('R'):
            rotation_angle = (rotation_angle - 10) % 360
        # Position Shift
        elif key == ord('w'):
            start_shift_y -= 10
        elif key == ord('s'):
            start_shift_y += 10
        elif key == ord('a'):
            start_shift_x -= 10
        elif key == ord('d'):
            start_shift_x += 10
        # Line Properties
        elif key == ord('['):
            line_weight = max(1, line_weight - 1)
        elif key == ord(']'):
            line_weight += 1
        elif key == ord('h'):
            line_hsv[0] = (line_hsv[0] + 5) % 180 # Hue is 0-179
        elif key == ord('H'):
            line_hsv[0] = (line_hsv[0] - 5) % 180
        elif key == ord('j'):
            line_hsv[1] = min(255, line_hsv[1] + 5)
        elif key == ord('J'):
            line_hsv[1] = max(0, line_hsv[1] - 5)
        elif key == ord('k'):
            line_hsv[2] = min(255, line_hsv[2] + 5)
        elif key == ord('K'):
            line_hsv[2] = max(0, line_hsv[2] - 5)
        elif key == ord('o'):
            line_opacity = min(1.0, line_opacity + 0.05)
        elif key == ord('O'):
            line_opacity = max(0.0, line_opacity - 0.05)
        # Fill Properties
        elif key == ord('f'):
            fill_enabled = not fill_enabled
        elif key == ord('g'):
            fill_hsv[0] = (fill_hsv[0] + 5) % 180
        elif key == ord('G'):
            fill_hsv[0] = (fill_hsv[0] - 5) % 180
        elif key == ord('i'):
            fill_hsv[1] = min(255, fill_hsv[1] + 5)
        elif key == ord('I'):
            fill_hsv[1] = max(0, fill_hsv[1] - 5)
        elif key == ord('l'):
            fill_hsv[2] = min(255, fill_hsv[2] + 5)
        elif key == ord('L'):
            fill_hsv[2] = max(0, fill_hsv[2] - 5)
        elif key == ord('p'):
            fill_opacity = min(1.0, fill_opacity + 0.05)
        elif key == ord('P'):
            fill_opacity = max(0.0, fill_opacity - 0.05)
        # Grid Mode Controls
        elif key == ord('u'):
            multiply_grid_x += 1
        elif key == ord('U'):
            multiply_grid_x = max(1, multiply_grid_x - 1)
        elif key == ord('v'):
            multiply_grid_y += 1
        elif key == ord('V'):
            multiply_grid_y = max(1, multiply_grid_y - 1)
        elif key == ord('c'):
            grid_pitch_x += 10
        elif key == ord('C'):
            grid_pitch_x = max(10, grid_pitch_x - 10) # Prevent 0 pitch
        elif key == ord('z'):
            grid_pitch_y += 10
        elif key == ord('Z'):
            grid_pitch_y = max(10, grid_pitch_y - 10) # Prevent 0 pitch

    cap.release()
    cv2.destroyAllWindows()

# draw_shape_on_canvas remains unchanged, it draws opaque shapes onto a given BGR canvas
def draw_shape_on_canvas(canvas, shape_type, center_x, center_y,
               size_multiplier, aspect_ratio, rotation_angle,
               line_color, line_weight, fill_enabled, fill_color):

    # Ensure coordinates are within bounds to prevent errors
    center_x = max(0, min(canvas.shape[1], center_x))
    center_y = max(0, min(canvas.shape[0], center_y))

    if shape_type == 'rectangle':
        rect_width = int(50 * size_multiplier * aspect_ratio)
        rect_height = int(50 * size_multiplier)

        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1)
        pts = np.array([
            [center_x - rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y - rect_height // 2],
            [center_x + rect_width // 2, center_y + rect_height // 2],
            [center_x - rect_width // 2, center_y + rect_height // 2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        rotated_pts = cv2.transform(pts, M)
        rotated_pts_int = np.int32(rotated_pts)

        if fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, line_color, line_weight)

    elif shape_type == 'circle':
        radius = int(30 * size_multiplier)
        if fill_enabled:
            cv2.circle(canvas, (center_x, center_y), radius, fill_color, -1)
        cv2.circle(canvas, (center_x, center_y), radius, line_color, line_weight)

    elif shape_type == 'triangle':
        side_length = int(60 * size_multiplier)

        p1_x = 0
        p1_y = -side_length // 2
        p2_x = -int(side_length * np.sqrt(3) / 4)
        p2_y = side_length // 4
        p3_x = int(side_length * np.sqrt(3) / 4)
        p3_y = side_length // 4

        pts = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]], dtype=np.float32)

        pts[:, 0] += center_x
        pts[:, 1] += center_y

        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1)
        rotated_pts = cv2.transform(pts.reshape(-1, 1, 2), M)
        rotated_pts_int = np.int32(rotated_pts)

        if fill_enabled:
            cv2.fillPoly(canvas, [rotated_pts_int], fill_color)
        cv2.polylines(canvas, [rotated_pts_int], True, line_color, line_weight)

    elif shape_type == 'line':
        line_length = int(80 * size_multiplier)

        p1_x = center_x - line_length // 2
        p1_y = center_y
        p2_x = center_x + line_length // 2
        p2_y = center_y

        pts = np.array([[p1_x, p1_y], [p2_x, p2_y]], dtype=np.float32).reshape(-1, 1, 2)
        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1)
        rotated_pts = cv2.transform(pts, M)

        cv2.line(canvas, tuple(np.int32(rotated_pts[0][0])), tuple(np.int32(rotated_pts[1][0])), line_color, line_weight)

if __name__ == "__main__":
    draw_shapes_on_video()