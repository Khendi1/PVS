import numpy as np
from config import params # Assuming 'config' module and 'params' object are correctly set up
import cv2
import random # Import the random module

class ReactionDiffusionSimulator:

    def __init__(self, width=500, height=500, da=1.0, db=0.5, feed=0.055, kill=0.062, randomize_seed=False, max_seed_size=50, num_seeds=15):
        self.width = width
        self.height = height
        self.da = params.add("da", 0, 2.0, da)
        self.db = params.add("db", 0, 2.0, db)

        example_patterns = {
            "worms": (0.055, 0.062),
            "spots": (0.035, 0.065),
            "maze": (0.029, 0.057),
            "coral": (0.054, 0.063)
        }
        self.pattern = example_patterns.get("coral", (feed, kill))

        # Feed rate (f): How much chemical A is added to the system
        self.feed = params.add("feed", 0, 0.1, feed)
        # Kill rate (k): How much chemical B is removed from the system
        self.kill = params.add("kill", 0, 0.1, kill)
        # Time step for the simulation. Smaller values increase stability but require more iterations.
        self.dt = 0.25  # Reduced from 1.0 for better stability
        # Number of simulation steps per displayed frame. Increased to compensate for smaller dt.
        self.iterations_per_frame = params.add("iterations_per_frame", 5, 100, 50)
        self.current_A = np.ones((height, width), dtype=np.float32)
        self.current_B = np.zeros((height, width), dtype=np.float32)
        self.next_A = np.copy(self.current_A)
        self.next_B = np.copy(self.current_B)
        
        # New parameters for seed randomization
        self.randomize_seed = randomize_seed
        self.max_seed_size = max_seed_size
        self.num_seeds = num_seeds # New parameter for number of seeds

        self.initialize_seed()

    def initialize_seed(self):
        """
        Seeds the grid with chemical B, either at a fixed center
        or with multiple random sizes and locations, and removes chemical A from those areas.
        This initial perturbation is necessary to kickstart pattern formation.
        """
        # Reset the grid before seeding to ensure a clean start for new seeds
        self.current_A.fill(1.0)
        self.current_B.fill(0.0)

        if self.randomize_seed:
            for _ in range(self.num_seeds): # Loop for multiple seeds
                # Randomize seed size
                seed_size = random.randint(5, self.max_seed_size)
                
                # Randomize seed location, ensuring it's within bounds
                # The seed_size // 2 offset keeps the square fully within the grid
                center_x = random.randint(seed_size // 2, self.width - seed_size // 2 - 1)
                center_y = random.randint(seed_size // 2, self.height - seed_size // 2 - 1)
                
                # Apply the seed
                self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
                self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0
        else:
            # Use fixed seed size and location if not randomizing (single seed at center)
            seed_size = 20
            center_x, center_y = self.width // 2, self.height // 2
            
            # Apply the single seed
            self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
            self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0


    def update_simulation(self):
        """
        Performs one step of the Gray-Scott reaction-diffusion simulation using
        NumPy's vectorized operations for efficiency.
        """
        # Calculate Laplacians using array slicing and rolling for periodic boundary conditions
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
        diff_A = self.da.value * lap_A - self.current_A * self.current_B**2 + self.feed.value * (1 - self.current_A)
        diff_B = self.db.value * lap_B + self.current_A * self.current_B**2 - (self.kill.value + self.feed.value) * self.current_B

        # Update next state arrays and clip values to stay within [0, 1] range
        self.next_A = np.clip(self.current_A + diff_A * self.dt, 0.0, 1.0)
        self.next_B = np.clip(self.current_B + diff_B * self.dt, 0.0, 1.0)

        # Swap current and next states for the next iteration
        temp_A = self.current_A
        temp_B = self.current_B
        self.current_A = self.next_A
        self.current_B = self.next_B
        self.next_A = temp_A 
        self.next_B = temp_B 


    def run(self):
        """
        Runs the simulation for a specified number of iterations and returns the display image.
        """
        for _ in range(self.iterations_per_frame.value):
            self.update_simulation()

        # Hue (H): Map chemical A concentration to hue (0-179 for OpenCV)
        # A higher concentration of A can correspond to one end of the spectrum,
        # and a lower concentration to another.
        # We'll use a range that gives a nice gradient, e.g., 0 to 120 (blue to green/yellow)
        hue = (self.current_A * 120).astype(np.uint8) 

        # Saturation (S): Map chemical B concentration to saturation (0-255)
        # Areas with more B will be more saturated (vibrant)
        saturation = (self.current_B * 255).astype(np.uint8)

        # Value (V): Map overall activity or a combination to brightness (0-255)
        # Here, we'll use a combination of A and B to ensure brightness.
        value = ((self.current_A + self.current_B) / 2 * 255).astype(np.uint8)

        hsv_image = cv2.merge([hue, saturation, value])
        
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":

    # Initialize the simulator with randomization enabled
    # You can set randomize_seed=False to go back to the fixed center seed
    # max_seed_size controls the upper limit for random seed dimensions
    simulator = ReactionDiffusionSimulator(randomize_seed=True, max_seed_size=80) 
    print("Starting Reaction-Diffusion Simulation. Press 'ESC' to quit.")
    cv2.namedWindow("Reaction-Diffusion Pattern", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reaction-Diffusion Pattern", 600, 600) # Set initial window size

    while True:
        frame = simulator.run()
        # Resize the frame for better viewing, maintaining aspect ratio if needed
        frame_resized = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Reaction-Diffusion Pattern", frame_resized)
        
        key = cv2.waitKey(1)
        if key == 27:   # ESC key to exit
            break
    cv2.destroyAllWindows()
