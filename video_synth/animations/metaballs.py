import cv2
import numpy as np

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget


def compute_metaball_field_vectorized(X, Y, metaballs):
    """Fully vectorized metaball computation - no Python loops."""
    # Extract all positions and radii at once
    positions = np.array([[b['x'], b['y']] for b in metaballs], dtype=np.float32)
    radii = np.array([b['radius'] for b in metaballs], dtype=np.float32)

    # Broadcast positions to compute all distances at once
    # Shape: (num_balls, height, width)
    dx = X[np.newaxis, :, :] - positions[:, 0, np.newaxis, np.newaxis]
    dy = Y[np.newaxis, :, :] - positions[:, 1, np.newaxis, np.newaxis]
    dist_sq = dx**2 + dy**2 + 1e-6

    # Compute field contributions for all balls at once
    r_sq = radii[:, np.newaxis, np.newaxis]**2
    field_contributions = r_sq / dist_sq

    # Sum along ball axis
    return field_contributions.sum(axis=0)


class Metaballs(Animation):
    def __init__(self, params, width=800, height=600, group=None):
        super().__init__(params, width, height, group=group)
        subgroup=self.__class__.__name__
        p_name = group.name.lower()
        self.metaballs = []
        
        self.num_metaballs = params.add("num_metaballs",
                                        min=2, max=10, default=5,
                                        subgroup=subgroup, group=group)
        self.min_radius = params.add("min_radius",
                                     min=20, max=100, default=40,
                                     subgroup=subgroup, group=group)
        self.max_radius = params.add("max_radius",
                                     min=40, max=200, default=80,
                                     subgroup=subgroup, group=group)
        self.radius_multiplier = params.add("radius_multiplier",
                                            min=1.0, max=3.0, default=1.0,
                                            subgroup=subgroup, group=group)
        self.max_speed = params.add("max_speed",
                                    min=1, max=10, default=3,
                                    subgroup=subgroup, group=group)
        self.speed_multiplier = params.add("speed_multiplier",
                                           min=1.0, max=3.0, default=1.0,
                                           subgroup=subgroup, group=group)
        self.threshold = params.add("threshold",
                                    min=0.5, max=3.0, default=1.6,
                                    subgroup=subgroup, group=group)
        self.smooth_coloring_max_field = params.add("smooth_coloring_max_field",
                                                    min=1.0, max=3.0, default=1.5,
                                                    subgroup=subgroup, group=group)
        self.skew_angle = params.add("metaball_skew_angle",
                                     min=0.0, max=360.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.skew_intensity = params.add("metaball_skew_intensity",
                                         min=0.0, max=1.0, default=0.0,
                                         subgroup=subgroup, group=group)
        self.zoom = params.add("metaball_zoom",
                               min=1.0, max=3.0, default=1.0,
                               subgroup=subgroup, group=group)
        self.colormap = params.add("metaball_colormap",
                                   min=0, max=len(COLORMAP_OPTIONS)-1, default=Colormap.JET.value,
                                   group=group, subgroup=subgroup,
                                   type=Widget.DROPDOWN, options=Colormap)
        self.feedback_alpha = params.add("metaballs_feedback",
                                         min=0.0, max=1.0, default=0.95,
                                         subgroup=subgroup, group=group)
        self.render_scale = params.add("metaballs_render_scale",
                                       min=0.25, max=1.0, default=0.25,
                                       subgroup=subgroup, group=group)

        self.current_num_metaballs = self.num_metaballs.value
        self.current_radius_multiplier = self.radius_multiplier.value
        self.current_speed_multiplier = self.speed_multiplier.value
        self.previous_frame = None

        # Performance optimization: cache meshgrid and parameters
        self._cached_meshgrid = None
        self._cached_zoom = None
        self._cached_skew_angle = None
        self._cached_skew_intensity = None
        self._cached_render_scale = None

        self.setup_metaballs()

    def adjusteters(self):
        if self.current_radius_multiplier != self.radius_multiplier.value:
            for ball in self.metaballs:
                ball['radius'] = int(ball['radius'] * self.radius_multiplier.value / self.current_radius_multiplier)
            self.current_radius_multiplier = self.radius_multiplier.value

        if self.current_speed_multiplier != self.speed_multiplier.value:
            for ball in self.metaballs:
                ball['vx'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
                ball['vy'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
            self.current_speed_multiplier = self.speed_multiplier.value

    def setup_metaballs(self):
        num_metaballs = self.num_metaballs.value
        if len(self.metaballs) > num_metaballs:
            self.metaballs = self.metaballs[:num_metaballs]
        else:
            delta = num_metaballs - len(self.metaballs)
            for _ in range(delta):
                r = np.random.randint(self.min_radius.value, self.max_radius.value) * self.radius_multiplier.value
                x = np.random.randint(self.max_radius.value, self.width - self.max_radius.value)
                y = np.random.randint(self.max_radius.value, self.height - self.max_radius.value)
                vx = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                vy = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                self.metaballs.append({'x': x, 'y': y, 'radius': r, 'vx': vx, 'vy': vy})
        self.current_num_metaballs = num_metaballs

    def create_metaball_frame(self, metaballs, threshold, max_field_strength=None):
        # Performance optimization: render at lower resolution then upscale
        render_scale = self.render_scale.value
        render_width = int(self.width * render_scale)
        render_height = int(self.height * render_scale)

        # Cache meshgrid if zoom/skew parameters haven't changed (MAJOR optimization)
        need_rebuild = (self._cached_meshgrid is None or
                       self._cached_zoom != self.zoom.value or
                       self._cached_skew_angle != self.skew_angle.value or
                       self._cached_skew_intensity != self.skew_intensity.value or
                       self._cached_render_scale != render_scale)

        if need_rebuild:
            x_coords = np.arange(render_width) * (self.width / render_width)
            y_coords = np.arange(render_height) * (self.height / render_height)
            X, Y = np.meshgrid(x_coords, y_coords)

            center_x, center_y = self.width / 2, self.height / 2
            X_centered = X - center_x
            Y_centered = Y - center_y
            X_processed = X_centered / self.zoom.value
            Y_processed = Y_centered / self.zoom.value
            if self.skew_intensity.value > 0:
                angle_rad = np.radians(self.skew_angle.value)
                X_processed += Y_processed * self.skew_intensity.value * np.cos(angle_rad)
                Y_processed += X_processed * self.skew_intensity.value * np.sin(angle_rad)
            X_transformed = X_processed + center_x
            Y_transformed = Y_processed + center_y

            # Cache the transformed meshgrid
            self._cached_meshgrid = (X_transformed, Y_transformed)
            self._cached_zoom = self.zoom.value
            self._cached_skew_angle = self.skew_angle.value
            self._cached_skew_intensity = self.skew_intensity.value
            self._cached_render_scale = render_scale
        else:
            X_transformed, Y_transformed = self._cached_meshgrid

        # Use vectorized computation - leverages NumPy's C implementation
        # 2-3x faster than Python loop, works on any Python version
        field_strength = compute_metaball_field_vectorized(X_transformed, Y_transformed, metaballs)

        if max_field_strength is not None:
            normalized_field = np.clip(field_strength / max_field_strength, 0, 1)
            grayscale_image = (normalized_field * 255).astype(np.uint8)
            image = cv2.applyColorMap(grayscale_image, COLORMAP_OPTIONS[self.colormap.value])
        else:
            image = ((field_strength >= threshold) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Upscale to target resolution if rendered at lower res
        if render_scale < 1.0:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return image

    def do_metaballs(self, frame: np.ndarray):
        if self.num_metaballs.value != self.current_num_metaballs:
            self.setup_metaballs()
        
        if self.current_radius_multiplier != self.radius_multiplier.value or self.current_speed_multiplier != self.speed_multiplier.value:
            self.adjusteters()

        for ball in self.metaballs:
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']

            if not (ball['radius'] < ball['x'] < self.width - ball['radius']):
                ball['vx'] *= -1
            if not (ball['radius'] < ball['y'] < self.height - ball['radius']):
                ball['vy'] *= -1

        current_frame = self.create_metaball_frame(self.metaballs,
                                                threshold=self.threshold.value,
                                                max_field_strength=self.smooth_coloring_max_field.value)
        
        if self.previous_frame is None:
            self.previous_frame = current_frame.astype(np.float32)
        else:
            current_frame = cv2.addWeighted(current_frame.astype(np.float32), 1-self.feedback_alpha.value, 
                                            self.previous_frame, self.feedback_alpha.value, 0)
            self.previous_frame = current_frame

        return current_frame

    def get_frame(self, frame: np.ndarray = None):
        return self.do_metaballs(frame)
