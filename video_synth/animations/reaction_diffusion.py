import cv2
import numpy as np
import random
from animations.base import Animation

class ReactionDiffusion(Animation):

    def __init__(self, params, width=500, height=500, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        da=1.0
        db=0.5
        feed=0.055
        kill=0.062
        randomize_seed=False
        max_seed_size=50
        num_seeds=15

        self.da = params.add("da",
                             min=0, max=2.0, default=da,
                             subgroup=subgroup, group=group)
        self.db = params.add("db",
                             min=0, max=2.0, default=db,
                             subgroup=subgroup, group=group)

        self.feed = params.add("feed",
                               min=0, max=0.1, default=feed,
                               subgroup=subgroup, group=group)
        self.kill = params.add("kill",
                               min=0, max=0.1, default=kill,
                               subgroup=subgroup, group=group)
        self.iterations_per_frame = params.add("iterations_per_frame",
                                               min=5, max=100, default=50,
                                               subgroup=subgroup, group=group)
        
        self.dt = 0.15
        self.current_A = np.ones((height, width), dtype=np.float32)
        self.current_B = np.zeros((height, width), dtype=np.float32)
        self.next_A = np.copy(self.current_A)
        self.next_B = np.copy(self.current_B)
        
        self.randomize_seed = randomize_seed
        self.max_seed_size = max_seed_size
        self.num_seeds = num_seeds 

        self.initialize_seed()

    def initialize_seed(self):
        self.current_A.fill(1.0)
        self.current_B.fill(0.0)

        if self.randomize_seed:
            for _ in range(self.num_seeds):
                seed_size = random.randint(5, self.max_seed_size)
                center_x = random.randint(seed_size // 2, self.width - seed_size // 2 - 1)
                center_y = random.randint(seed_size // 2, self.height - seed_size // 2 - 1)
                
                self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
                self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                               center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0
        else:
            seed_size = 20
            center_x, center_y = self.width // 2, self.height // 2
            
            self.current_B[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 1.0
            self.current_A[center_y - seed_size // 2 : center_y + seed_size // 2,
                           center_x - seed_size // 2 : center_x + seed_size // 2] = 0.0


    def update_simulation(self):
        lap_A = (
            np.roll(self.current_A, 1, axis=0) +
            np.roll(self.current_A, -1, axis=0) +
            np.roll(self.current_A, 1, axis=1) +
            np.roll(self.current_A, -1, axis=1) -
            4 * self.current_A
        )

        lap_B = (
            np.roll(self.current_B, 1, axis=0) +
            np.roll(self.current_B, -1, axis=0) +
            np.roll(self.current_B, 1, axis=1) +
            np.roll(self.current_B, -1, axis=1) -
            4 * self.current_B
        )

        diff_A = self.da.value * lap_A - self.current_A * self.current_B**2 + self.feed.value * (1 - self.current_A)
        diff_B = self.db.value * lap_B + self.current_A * self.current_B**2 - (self.kill.value + self.feed.value) * self.current_B

        self.next_A = np.clip(self.current_A + diff_A * self.dt, 0.0, 1.0)
        self.next_B = np.clip(self.current_B + diff_B * self.dt, 0.0, 1.0)

        self.current_A, self.current_B = self.next_A, self.next_B


    def run(self):
        for _ in range(self.iterations_per_frame.value):
            self.update_simulation()

        hue = (self.current_A * 120).astype(np.uint8) 
        saturation = (self.current_B * 255).astype(np.uint8)
        value = ((self.current_A + self.current_B) / 2 * 255).astype(np.uint8)
        hsv_image = cv2.merge([hue, saturation, value])
        
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def get_frame(self, frame):
        return self.run()
