import cv2
import numpy as np
import logging

from animations.base import Animation
from common import Widget, Toggle

log = logging.getLogger(__name__)

class Physarum(Animation):
    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        subgroup = self.__class__.__name__

        # Simulation parameters
        self.num_agents = params.add("phys_num_agents",
                                     min=1000, max=10000, default=1000,
                                     subgroup=subgroup, group=group)
        self.sensor_angle_spacing = params.add("phys_sensor_angle_spacing",
                                               min=0.0, max=np.pi/2, default=np.pi/8,
                                               subgroup=subgroup, group=group) # Radians
        self.sensor_distance = params.add("phys_sensor_distance",
                                          min=1, max=20, default=9,
                                          subgroup=subgroup, group=group)
        self.turn_angle = params.add("phys_turn_angle",
                                     min=0.0, max=np.pi/2, default=np.pi/4,
                                     subgroup=subgroup, group=group) # Radians
        self.step_distance = params.add("phys_step_distance",
                                        min=1, max=10, default=1,
                                        subgroup=subgroup, group=group)
        self.decay_factor = params.add("phys_decay_factor",
                                       min=0.0, max=1.0, default=0.1,
                                       subgroup=subgroup, group=group)
        self.diffuse_factor = params.add("phys_diffuse_factor",
                                         min=0.0, max=1.0, default=0.5,
                                         subgroup=subgroup, group=group)
        self.deposit_amount = params.add("phys_deposit_amount",
                                         min=0.1, max=5.0, default=1.0,
                                         subgroup=subgroup, group=group)
        self.grid_resolution_scale = params.add("phys_grid_res_scale",
                                                min=0.1, max=1.0, default=0.5,
                                                subgroup=subgroup, group=group)
        self.wrap_around = params.add("phys_wrap_around",
                                      min=0, max=1, default=1,
                                      group=group, subgroup=subgroup,
                                      type=Widget.RADIO, options=Toggle) # Boolean as int

        # Color parameters
        self.trail_r = params.add("phys_trail_r",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.trail_g = params.add("phys_trail_g",
                                  min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.trail_b = params.add("phys_trail_b",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_r = params.add("phys_agent_r",
                                  min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.agent_g = params.add("phys_agent_g",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_b = params.add("phys_agent_b",
                                  min=0, max=255, default=0,
                                  subgroup=subgroup, group=group)
        self.agent_size = params.add("phys_agent_size",
                                     min=1, max=5, default=1,
                                     subgroup=subgroup, group=group)


        self.grid_width = int(self.width * self.grid_resolution_scale.value)
        self.grid_height = int(self.height * self.grid_resolution_scale.value)

        self.trail_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.agents = self._initialize_agents()

        # Store previous parameter values to detect changes for re-initialization
        self.prev_num_agents = self.num_agents.value
        self.prev_grid_resolution_scale = self.grid_resolution_scale.value

    def _reinitialize_simulation(self):
        log.debug("Reinitializing Physarum simulation due to parameter change.")
        self.grid_width = int(self.width * self.grid_resolution_scale.value)
        self.grid_height = int(self.height * self.grid_resolution_scale.value)
        self.trail_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.agents = self._initialize_agents()

    def _initialize_agents(self):
        # Agents: [x, y, angle]
        agents = np.zeros((self.num_agents.value, 3), dtype=np.float32)
        agents[:, 0] = np.random.uniform(0, self.grid_width, self.num_agents.value)   # x
        agents[:, 1] = np.random.uniform(0, self.grid_height, self.num_agents.value)  # y
        agents[:, 2] = np.random.uniform(0, 2 * np.pi, self.num_agents.value)         # angle
        return agents

    def _sense(self, agent_x, agent_y, agent_angle):
        # Sensor angles for front-left, front, front-right
        angle_f = agent_angle
        angle_r = agent_angle + self.sensor_angle_spacing.value
        angle_l = agent_angle - self.sensor_angle_spacing.value

        # Sensor positions
        dist = self.sensor_distance.value
        xf, yf = agent_x + dist * np.cos(angle_f), agent_y + dist * np.sin(angle_f)
        xr, yr = agent_x + dist * np.cos(angle_r), agent_y + dist * np.sin(angle_r)
        xl, yl = agent_x + dist * np.cos(angle_l), agent_y + dist * np.sin(angle_l)

        # Ensure sensor positions are within bounds or wrap around
        xf, yf = self._get_safe_coords(xf, yf)
        xr, yr = self._get_safe_coords(xr, yr)
        xl, yl = self._get_safe_coords(xl, yl)

        # Read trail map at sensor positions
        # Ensure coordinates are integers before indexing
        return (
            self.trail_map[int(yf), int(xf)],
            self.trail_map[int(yr), int(xr)],
            self.trail_map[int(yl), int(yl)]
        )

    def _get_safe_coords(self, x, y):
        if self.wrap_around.value:
            x = x % self.grid_width
            y = y % self.grid_height
        else:
            x = np.clip(x, 0, self.grid_width - 1)
            y = np.clip(y, 0, self.grid_height - 1)
        return x, y

    def _move_agents(self):
        new_angles = np.copy(self.agents[:, 2]) # Start with current angles

        for i, agent in enumerate(self.agents):
            x, y, angle = agent
            val_f, val_r, val_l = self._sense(x, y, angle)

            # Decision making based on sensed values
            if val_f > val_l and val_f > val_r:
                # Move forward (no change to angle)
                pass
            elif val_l > val_f and val_l > val_r:
                # Turn left
                new_angles[i] -= self.turn_angle.value
            elif val_r > val_f and val_r > val_l:
                # Turn right
                new_angles[i] += self.turn_angle.value
            else:
                # Random turn if no clear direction
                new_angles[i] += np.random.uniform(-self.turn_angle.value, self.turn_angle.value)

        self.agents[:, 2] = new_angles % (2 * np.pi) # Update angles and wrap around 2pi

        # Update positions
        step = self.step_distance.value
        self.agents[:, 0] += step * np.cos(self.agents[:, 2])
        self.agents[:, 1] += step * np.sin(self.agents[:, 2])

        # Apply boundary conditions
        if self.wrap_around.value:
            self.agents[:, 0] = self.agents[:, 0] % self.grid_width
            self.agents[:, 1] = self.agents[:, 1] % self.grid_height
        else:
            self.agents[:, 0] = np.clip(self.agents[:, 0], 0, self.grid_width - 1)
            self.agents[:, 1] = np.clip(self.agents[:, 1], 0, self.grid_height - 1)

    def _deposit_trails(self):
        # Deposit chemical at agent's current position
        for agent in self.agents:
            x, y = int(agent[0]), int(agent[1])
            # Ensure y, x are within bounds after potential clipping in _get_safe_coords
            y = np.clip(y, 0, self.grid_height - 1)
            x = np.clip(x, 0, self.grid_width - 1)
            self.trail_map[y, x] += self.deposit_amount.value
            self.trail_map[y, x] = np.clip(self.trail_map[y, x], 0.0, 1.0) # Clip to max trail value

    def _diffuse_and_decay(self):
        # Decay
        self.trail_map *= (1.0 - self.decay_factor.value)

        # Diffusion (simple blur)
        if self.diffuse_factor.value > 0:
            kernel_size = 3 # For a simple blur
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.trail_map = cv2.GaussianBlur(self.trail_map, (kernel_size, kernel_size), self.diffuse_factor.value)

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:

        # Check for re-initialization based on parameters
        current_num_agents = self.num_agents.value
        current_grid_res_scale = self.grid_resolution_scale.value

        if (current_num_agents != self.prev_num_agents or
            current_grid_res_scale != self.prev_grid_resolution_scale):
            
            self._reinitialize_simulation()
            # Update previous values
            self.prev_num_agents = current_num_agents
            self.prev_grid_resolution_scale = current_grid_res_scale

        # Update simulation
        self._move_agents()
        self._deposit_trails()
        self._diffuse_and_decay()

        # Render to frame
        if frame is None:
            output_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            output_frame = frame.copy()

        # Scale trail_map to full frame size
        display_trail_map = cv2.resize(self.trail_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Color the trails
        trail_color_bgr = (self.trail_b.value, self.trail_g.value, self.trail_r.value)
        colored_trails = np.zeros_like(output_frame, dtype=np.uint8)
        colored_trails[:,:,0] = (display_trail_map * trail_color_bgr[0]).astype(np.uint8)
        colored_trails[:,:,1] = (display_trail_map * trail_color_bgr[1]).astype(np.uint8)
        colored_trails[:,:,2] = (display_trail_map * trail_color_bgr[2]).astype(np.uint8)

        # Blend trails onto the frame
        output_frame = cv2.addWeighted(output_frame, 1.0, colored_trails, 1.0, 0)

        # Draw agents
        agent_color_bgr = (self.agent_b.value, self.agent_g.value, self.agent_r.value)
        agent_radius = self.agent_size.value
        for agent in self.agents:
            # Map agent coordinates from simulation grid to display grid
            display_x = int(agent[0] / self.grid_resolution_scale.value)
            display_y = int(agent[1] / self.grid_resolution_scale.value)
            cv2.circle(output_frame, (display_x, display_y), agent_radius, agent_color_bgr, -1)
        
        return output_frame
