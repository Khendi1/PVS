import cv2
import numpy as np

from animations.base import Animation
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget, Toggle

class Voronoi(Animation):
    """
    Voronoi Relaxation - points iteratively move toward cell centroids,
    creating organic, cell-like tessellations that breathe and flow.
    """
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Point parameters
        self.num_points = params.add("voronoi_num_points",
                                     min=5, max=200, default=50,
                                     subgroup=subgroup, group=group)
        self.relaxation_speed = params.add("voronoi_relax_speed",
                                           min=0.01, max=0.5, default=0.1,
                                           subgroup=subgroup, group=group)
        self.jitter = params.add("voronoi_jitter",
                                 min=0.0, max=5.0, default=0.5,
                                 subgroup=subgroup, group=group)

        # Visual parameters
        self.show_edges = params.add("voronoi_show_edges",
                                     min=0, max=1, default=1,
                                     subgroup=subgroup, group=group,
                                     type=Widget.TOGGLE)
        self.show_points = params.add("voronoi_show_points",
                                      min=0, max=1, default=1,
                                      subgroup=subgroup, group=group,
                                      type=Widget.TOGGLE)
        self.fill_cells = params.add("voronoi_fill_cells",
                                     min=0, max=1, default=1,
                                     subgroup=subgroup, group=group,
                                     type=Widget.TOGGLE)
        self.edge_thickness = params.add("voronoi_edge_thickness",
                                         min=1, max=5, default=2,
                                         subgroup=subgroup, group=group)
        self.point_size = params.add("voronoi_point_size",
                                     min=2, max=10, default=5,
                                     subgroup=subgroup, group=group)

        # Color parameters
        self.edge_r = params.add("voronoi_edge_r",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.edge_g = params.add("voronoi_edge_g",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.edge_b = params.add("voronoi_edge_b",
                                 min=0, max=255, default=255,
                                 subgroup=subgroup, group=group)
        self.colormap = params.add("voronoi_colormap",
                                   min=0, max=len(COLORMAP_OPTIONS)-1, default=0,
                                   subgroup=subgroup, group=group,
                                   type=Widget.DROPDOWN, options=Colormap)

        # Animation
        self.color_cycle_speed = params.add("voronoi_color_speed",
                                            min=0.0, max=2.0, default=0.2,
                                            subgroup=subgroup, group=group)

        self.time = 0.0
        self._init_points()
        self.prev_num_points = self.num_points.value

    def _init_points(self):
        """Initialize Voronoi seed points."""
        n = int(self.num_points.value)
        # Random initial positions with margin
        margin = 20
        self.points = np.random.uniform(
            [margin, margin],
            [self.width - margin, self.height - margin],
            (n, 2)
        ).astype(np.float32)
        # Generate random colors for each cell
        self.cell_colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)

    def _compute_voronoi_image(self):
        """Compute Voronoi diagram using distance transform."""
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:self.height, :self.width]

        # Calculate distance from each pixel to each point
        # Use broadcasting: points shape (n, 2), coords shape (h, w)
        n = len(self.points)
        min_dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
        cell_indices = np.zeros((self.height, self.width), dtype=np.int32)

        for i, (px, py) in enumerate(self.points):
            dist = (x_coords - px) ** 2 + (y_coords - py) ** 2
            mask = dist < min_dist
            min_dist = np.where(mask, dist, min_dist)
            cell_indices = np.where(mask, i, cell_indices)

        return cell_indices, np.sqrt(min_dist)

    def _compute_centroids(self, cell_indices):
        """Compute centroid of each Voronoi cell."""
        n = len(self.points)
        centroids = np.zeros((n, 2), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:self.height, :self.width]

        for i in range(n):
            mask = cell_indices == i
            if np.any(mask):
                centroids[i, 0] = np.mean(x_coords[mask])
                centroids[i, 1] = np.mean(y_coords[mask])
                counts[i] = np.sum(mask)
            else:
                # Keep current position if cell is empty
                centroids[i] = self.points[i]

        return centroids

    def _detect_edges(self, cell_indices):
        """Detect edges between Voronoi cells."""
        # Shift and compare to detect boundaries
        edges = np.zeros((self.height, self.width), dtype=bool)

        # Compare with shifted versions
        if self.height > 1:
            edges[:-1, :] |= cell_indices[:-1, :] != cell_indices[1:, :]
        if self.width > 1:
            edges[:, :-1] |= cell_indices[:, :-1] != cell_indices[:, 1:]

        return edges

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:

        # Check for point count change
        if self.num_points.value != self.prev_num_points:
            self._init_points()
            self.prev_num_points = self.num_points.value

        self.time += 0.016

        # Add jitter to points
        jitter_amount = self.jitter.value
        if jitter_amount > 0:
            jitter = np.random.normal(0, jitter_amount, self.points.shape).astype(np.float32)
            jittered_points = self.points + jitter
        else:
            jittered_points = self.points.copy()

        # Temporarily use jittered points for rendering
        original_points = self.points.copy()
        self.points = jittered_points

        # Compute Voronoi diagram
        cell_indices, distances = self._compute_voronoi_image()

        # Create output frame
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fill cells with colors
        if self.fill_cells.value:
            # Color offset for animation
            color_offset = int(self.time * self.color_cycle_speed.value * 50) % 256

            for i in range(len(self.points)):
                mask = cell_indices == i
                # Cycle colors over time
                color_idx = (i * 5 + color_offset) % 256
                # Use colormap for coloring
                base_color = cv2.applyColorMap(
                    np.array([[color_idx]], dtype=np.uint8),
                    COLORMAP_OPTIONS[int(self.colormap.value)]
                )[0, 0]
                pattern[mask] = base_color

        # Restore original points for relaxation
        self.points = original_points

        # Draw edges
        if self.show_edges.value:
            edges = self._detect_edges(cell_indices)
            edge_color = (int(self.edge_b.value), int(self.edge_g.value), int(self.edge_r.value))
            thickness = int(self.edge_thickness.value)

            if thickness == 1:
                pattern[edges] = edge_color
            else:
                # Dilate edges for thicker lines
                kernel = np.ones((thickness, thickness), np.uint8)
                edges_thick = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)
                pattern[edges_thick > 0] = edge_color

        # Draw points
        if self.show_points.value:
            point_size = int(self.point_size.value)
            for px, py in self.points:
                cv2.circle(pattern, (int(px), int(py)), point_size, (255, 255, 255), -1)
                cv2.circle(pattern, (int(px), int(py)), point_size, (0, 0, 0), 1)

        # Lloyd's relaxation - move points toward cell centroids
        centroids = self._compute_centroids(cell_indices)
        relax_speed = self.relaxation_speed.value
        self.points += (centroids - self.points) * relax_speed

        # Keep points within bounds
        margin = 10
        self.points[:, 0] = np.clip(self.points[:, 0], margin, self.width - margin)
        self.points[:, 1] = np.clip(self.points[:, 1], margin, self.height - margin)

        return pattern
