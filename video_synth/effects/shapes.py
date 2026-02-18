import cv2
import numpy as np

from effects.enums import Shape
from common import Widget


class Shapes:

    def __init__(self, params, width, height, shape_x_shift=0, shape_y_shift=0, group=None):
        self.params = params
        self.width = width
        self.height = height
        subgroup = self.__class__.__name__

        self.shape_type = params.add("shape_type",
                                      min=0, max=len(Shape)-1, default=Shape.NONE.value,
                                      group=group, subgroup=subgroup,
                                      type=Widget.DROPDOWN, options=Shape)

        self.line_h = params.add("line_hue", min=0, max=179, default=0,
                                 subgroup=subgroup, group=group)
        self.line_s = params.add("line_sat", min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.line_v = params.add("line_val", min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.line_weight = params.add("line_weight", min=1, max=20, default=2,
                                      subgroup=subgroup, group=group)
        self.line_opacity = params.add("line_opacity", min=0.0, max=1.0, default=0.66,
                                       subgroup=subgroup, group=group)

        self.size_multiplier = params.add("size_multiplier", min=0.1, max=10.0, default=0.9,
                                          subgroup=subgroup, group=group)
        self.aspect_ratio = params.add("aspect_ratio", min=0.1, max=10.0, default=1.0,
                                       subgroup=subgroup, group=group)
        self.rotation_angle = params.add("rotation_angle", min=0, max=360, default=0,
                                         subgroup=subgroup, group=group)

        self.shape_x_shift = params.add("shape_x_shift", min=-width, max=width, default=shape_x_shift,
                                        subgroup=subgroup, group=group)
        self.shape_y_shift = params.add("shape_y_shift", min=-height, max=height, default=shape_y_shift,
                                        subgroup=subgroup, group=group)

        self.multiply_grid_x = params.add("multiply_grid_x", min=1, max=10, default=2,
                                          subgroup=subgroup, group=group)
        self.multiply_grid_y = params.add("multiply_grid_y", min=1, max=10, default=2,
                                          subgroup=subgroup, group=group)
        self.grid_pitch_x = params.add("grid_pitch_x", min=0, max=width, default=100,
                                       subgroup=subgroup, group=group)
        self.grid_pitch_y = params.add("grid_pitch_y", min=0, max=height, default=100,
                                       subgroup=subgroup, group=group)

        self.fill_h = params.add("fill_hue", min=0, max=179, default=120,
                                 subgroup=subgroup, group=group)
        self.fill_s = params.add("fill_sat", min=0, max=255, default=100,
                                 subgroup=subgroup, group=group)
        self.fill_v = params.add("fill_val", min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.fill_opacity = params.add("fill_opacity", min=0.0, max=1.0, default=0.25,
                                       subgroup=subgroup, group=group)

        self.canvas_rotation = params.add("canvas_rotation", min=0, max=360, default=0,
                                          subgroup=subgroup, group=group)

        self._draw_fn = {
            Shape.RECTANGLE.value: self._draw_rectangle,
            Shape.CIRCLE.value: self._draw_circle,
            Shape.TRIANGLE.value: self._draw_triangle,
            Shape.LINE.value: self._draw_line,
            Shape.DIAMOND.value: self._draw_diamond,
        }

    def _hsv_to_bgr(self, h, s, v):
        hsv_np = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def _draw_rectangle(self, canvas, cx, cy, color, thickness):
        w = int(50 * self.size_multiplier.value * self.aspect_ratio.value)
        h = int(50 * self.size_multiplier.value)
        M = cv2.getRotationMatrix2D((cx, cy), self.rotation_angle.value, 1)
        pts = np.array([
            [cx - w // 2, cy - h // 2],
            [cx + w // 2, cy - h // 2],
            [cx + w // 2, cy + h // 2],
            [cx - w // 2, cy + h // 2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        rotated = np.int32(cv2.transform(pts, M))
        if thickness == -1:
            cv2.fillPoly(canvas, [rotated], color)
        else:
            cv2.polylines(canvas, [rotated], True, color, thickness)

    def _draw_circle(self, canvas, cx, cy, color, thickness):
        radius = int(30 * self.size_multiplier.value)
        cv2.circle(canvas, (cx, cy), radius, color, thickness)

    def _draw_triangle(self, canvas, cx, cy, color, thickness):
        side = int(60 * self.size_multiplier.value)
        tri_h = int(side * np.sqrt(3) / 4)
        pts = np.array([
            [0, -side // 2],
            [-tri_h, side // 4],
            [tri_h, side // 4]
        ], dtype=np.float32)
        pts[:, 0] += cx
        pts[:, 1] += cy
        M = cv2.getRotationMatrix2D((cx, cy), self.rotation_angle.value, 1)
        rotated = np.int32(cv2.transform(pts.reshape(-1, 1, 2), M))
        if thickness == -1:
            cv2.fillPoly(canvas, [rotated], color)
        else:
            cv2.polylines(canvas, [rotated], True, color, thickness)

    def _draw_line(self, canvas, cx, cy, color, thickness):
        if thickness < 0:
            return
        length = int(50 * self.size_multiplier.value)
        start = (cx - length // 2, cy)
        end = (cx + length // 2, cy)
        cv2.line(canvas, start, end, color, thickness)

    def _draw_diamond(self, canvas, cx, cy, color, thickness):
        size = int(40 * self.size_multiplier.value)
        w = int(size * self.aspect_ratio.value)
        pts = np.array([
            [cx, cy - size],
            [cx + w, cy],
            [cx, cy + size],
            [cx - w, cy]
        ], dtype=np.float32).reshape(-1, 1, 2)
        M = cv2.getRotationMatrix2D((cx, cy), self.rotation_angle.value, 1)
        rotated = np.int32(cv2.transform(pts, M))
        if thickness == -1:
            cv2.fillPoly(canvas, [rotated], color)
        else:
            cv2.polylines(canvas, [rotated], True, color, thickness)

    def _draw_shape(self, canvas, cx, cy, color, thickness):
        cx = max(0, min(canvas.shape[1], cx))
        cy = max(0, min(canvas.shape[0], cy))
        draw_fn = self._draw_fn.get(self.shape_type.value)
        if draw_fn:
            draw_fn(canvas, cx, cy, color, thickness)

    def _blend_overlay(self, frame, canvas, opacity):
        mask = canvas.any(axis=2)
        if not mask.any():
            return frame
        result = frame.copy()
        result[mask] = (frame[mask].astype(np.float32) * (1.0 - opacity) +
                        canvas[mask].astype(np.float32) * opacity).astype(np.uint8)
        return result

    def draw_shapes(self, frame):
        if self.shape_type.value == Shape.NONE.value:
            return frame

        line_color = self._hsv_to_bgr(self.line_h.value, self.line_s.value, self.line_v.value)
        fill_color = self._hsv_to_bgr(self.fill_h.value, self.fill_s.value, self.fill_v.value)

        base_cx = self.width // 2 + self.shape_x_shift.value
        base_cy = self.height // 2 + self.shape_y_shift.value

        fill_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        line_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        grid_x = self.multiply_grid_x.value
        grid_y = self.multiply_grid_y.value

        for row in range(grid_y):
            for col in range(grid_x):
                cx = base_cx + col * self.grid_pitch_x.value - (grid_x - 1) * self.grid_pitch_x.value // 2
                cy = base_cy + row * self.grid_pitch_y.value - (grid_y - 1) * self.grid_pitch_y.value // 2
                self._draw_shape(fill_canvas, cx, cy, fill_color, -1)
                self._draw_shape(line_canvas, cx, cy, line_color, self.line_weight.value)

        frame = self._blend_overlay(frame, fill_canvas, self.fill_opacity.value)
        frame = self._blend_overlay(frame, line_canvas, self.line_opacity.value)
        return frame
