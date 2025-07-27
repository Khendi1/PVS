import numpy as np
from config import params

class ReactionDiffusionSimulator:
    def __init__(self, width=200, height=200, da=1.0, db=0.5, feed=0.055, kill=0.062):
        self.width = width
        self.height = height
        self.da = params.add("da", 0, 2.0, da)
        self.db = params.add("db", 0, 2.0, db)
        self.feed = params.add("feed", 0, 0.1, feed)
        self.kill = params.add("kill", 0, 0.1, kill)
        self.dt = 1.0
        self.iterations_per_frame = 20
        self.current_A = np.ones((height, width), dtype=np.float32)
        self.current_B = np.zeros((height, width), dtype=np.float32)
        self.next_A = np.copy(self.current_A)
        self.next_B = np.copy(self.current_B)
        self.seed_size = 20
        self.center_x, self.center_y = width // 2, height // 2
        self.initialize_seed()

# Feed rate (f): How much chemical A is added to the system
# Kill rate (k): How much chemical B is removed from the system
# Example patterns:
#   - Worms: f=0.055, k=0.062
#   - Spots: f=0.035, k=0.065
#   - Maze: f=0.029, k=0.057
#   - Coral: f=0.054, k=0.063

    def initialize_seed(self):
        # Seed the center of the grid with a small square of chemical B
        # This initial perturbation is necessary to kickstart pattern formation
        center_x, center_y = self.width // 2, self.height // 2
        self.current_B[center_y - self.seed_size // 2 : center_y + self.seed_size // 2,
                center_x - self.seed_size // 2 : center_x + self.seed_size // 2] = 1.0
        self.current_A[center_y - self.seed_size // 2 : center_y + self.seed_size // 2,
                center_x - self.seed_size // 2 : center_x + self.seed_size // 2] = 0.0

    # --- Helper Function for Laplacian Calculation ---
    def calculate_laplacian(self, grid, i, j):
        """
        Calculates the 5-point discrete Laplacian for a cell (i, j) on the grid.
        Uses periodic (wrap-around) boundary conditions.
        """
        # Get values of neighbors, wrapping around the edges
        val_center = grid[i, j]
        val_up = grid[(i - 1 + self.height) % self.height, j]
        val_down = grid[(i + 1) % self.height, j]
        val_left = grid[i, (j - 1 + self.width) % self.width]
        val_right = grid[i, (j + 1) % self.width]

        # Laplacian formula: sum of neighbors - 4 * center
        laplacian = val_up + val_down + val_left + val_right - 4 * val_center
        return laplacian

    def update_simulation(self):
        """
        Performs one step of the Gray-Scott reaction-diffusion simulation using
        NumPy's vectorized operations for efficiency.
        """

        # Calculate Laplacians using array slicing and rolling for periodic boundary conditions
        # This is equivalent to applying a convolution kernel for the Laplacian
        lap_A = (
            np.roll(self.current_A, 1, axis=0) +  # Up
            np.roll(self.current_A, -1, axis=0) + # Down
            np.roll(self.current_A, 1, axis=1) +  # Left
            np.roll(self.current_A, -1, axis=1) - # Right
            4 * self.current_A
        )

        lap_B = (
            np.roll(self.current_B, 1, axis=0) +  # Up
            np.roll(self.current_B, -1, axis=0) + # Down
            np.roll(self.current_B, 1, axis=1) +  # Left
            np.roll(self.current_B, -1, axis=1) - # Right
            4 * self.current_B
        )

        # Apply Gray-Scott equations using vectorized operations
        # dA/dt = Da * Laplacian(A) - A * B*B + f * (1 - A)
        # dB/dt = Db * Laplacian(B) + A * B*B - (k + f) * B
        diff_A = self.da * lap_A - self.current_A * self.current_B**2 + self.feed * (1 - self.current_A)
        diff_B = self.db * lap_B + self.current_A * self.current_B**2 - (self.kill + self.feed) * self.current_B

        next_A = np.clip(self.current_A + diff_A * self.dt, 0.0, 1.0)
        next_B = np.clip(self.current_B + diff_B * self.dt, 0.0, 1.0)

        # Swap current and next states for the next iteration
        self.current_A, next_A = next_A, self.current_A
        self.current_B, next_B = next_B, self.current_B

    def run(self):
        """
        Runs the simulation for a specified number of iterations.
        """
        for _ in range(self.iterations_per_frame):
            self.update_simulation()

        # Create an image for display
        # We'll visualize chemical B, mapping its concentration to grayscale intensity
        # (1 - B) makes areas with more B appear darker, which often looks better for patterns
        display_image = (1 - self.current_B) * 255
        display_image = display_image.astype(np.uint8) # Convert to 8-bit for OpenCV

