import cv2
import numpy as np
import time
from config import params
from gui import TrackbarRow, TrackbarCallback
from buttons import Button
import dearpygui.dearpygui as dpg
from param import Param

class LavaLampSynth:
    def __init__(self, width=800, height=600):
        """
        Initializes the LavaLampSynth with given dimensions.
        """
        self.width = width
        self.height = height
        self.metaballs = []
        self.num_metaballs = 5        # Number of metaballs
        self.min_radius = 40          # Minimum radiuWSs of a metaball
        self.max_radius = 80          # Maximum radius of a metaball
        self.max_speed = 3            # Maximum pixels per frame movement
        self.threshold = 1.6          # Threshold for metaball field strength (adjust for different blob shapes)
        
        # Adjusted smooth_coloring_max_field for a better gradient.
        # This value should be high enough to encompass the range of field strengths
        # you want to map to a gradient, preventing premature clipping to pure white.
        # It's scaled by max_radius to better fit the r^2/dist_sq field function.
        self.smooth_coloring_max_field = 1.5 # Experiment with this multiplier (e.g., 1.0 to 3.0)

        # Feedback effect parameters
        self.feedback_alpha = 0.950  # Weight of the new frame (0.0 - 1.0)
        self.previous_frame = None  # Stores the previous frame for feedback

        self.num_metaballs = params.add("num_metaballs", 2, 10, self.num_metaballs)
        self.current_num_metaballs = self.num_metaballs.value

        self.min_radius = params.add("min_radius", 20, 100, 40)
        self.max_radius = params.add("max_radius", 40, 200, 80)
        self.radius_multplier = params.add("radius_multiplier", 1.0, 3.0, 1.0)
        self.current_radius_multiplier = self.radius_multplier.value

        self.max_speed = params.add("max_speed", 1, 10, self.max_speed)
        self.speed_multiplier = params.add("speed_multiplier", 1.0, 3.0, 1.0)
        self.current_speed_multiplier = self.speed_multiplier.value

        self.threshold = params.add("threshold", 0.5, 3.0, self.threshold)
        self.smooth_coloring_max_field = params.add("smooth_coloring_max_field", 1.0, 3.0, self.smooth_coloring_max_field)

        self.skew_angle = params.add("metaball_skew_angle", 0.0, 360.0, 0.0)  # Angle to skew the metaballs
        self.skew_intensity = params.add("metaball_skew_intensity", 0.0, 1.0, 0.0)  # Intensity of the skew effect

        self.zoom = params.add("metaball_zoom", 1.0, 3.0, 1.0)  # Zoom level for the metaballs

        self.hue = params.add("metaball_hue", 0.0, 255.0, 0.0)  # Hue shift for the metaballs
        self.saturation = params.add("metaball_saturation", 0.0, 255.0, 255.0)  # Saturation for the metaballs
        self.value = params.add("metaball_value", 0.0, 255.0, 255.0)  # Value for the metaballs
        
        # apply feedback to the metaball frame
        self.feedback_alpha = params.add("metaballs_feedback", 0.0, 1.0, self.feedback_alpha)
        # amount to blend the metaball frame wisth the input frame
        self.frame_blend = params.add("metaballs_frame_blend", 0.0, 1.0, 0.5)

        self.setup_metaballs()


    def adjust_parameters(self):
        """
        Adjusts the parameters based on the current values in the config.
        """
        if self.current_num_metaballs != self.num_metaballs.value:
            self.setup_metaballs()
            self.current_num_metaballs = self.num_metaballs.value

        if self.current_radius_multiplier != self.radius_multplier.value:
            for ball in self.metaballs:
                ball['radius'] = int(ball['radius'] * self.radius_multplier.value / self.current_radius_multiplier)
            self.current_radius_multiplier = self.radius_multplier.value

        if self.current_speed_multiplier != self.speed_multiplier.value:
            for ball in self.metaballs:
                ball['vx'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
                ball['vy'] *= (self.speed_multiplier.value / self.current_speed_multiplier)
            self.current_speed_multiplier = self.speed_multiplier.value


    def setup_metaballs(self):
        """
        Initializes the positions, radii, and velocities of the metaballs.
        """
        num_metaballs = self.num_metaballs.value
        if len(self.metaballs) > num_metaballs:
            # If reducing the number of metaballs, truncate the list
            self.metaballs = self.metaballs[:num_metaballs]
            self.current_num_metaballs = num_metaballs
        else:
            # If increasing the number of metaballs, append new ones
            # Random initial position within the frame, ensuring balls start within bounds
            delta = num_metaballs - len(self.metaballs)
            for i in range(delta):
                x = np.random.randint(self.max_radius.value, self.width - self.max_radius.value)
                y = np.random.randint(self.max_radius.value, self.height - self.max_radius.value)
                # Random radius within the defined range
                r = np.random.randint(self.min_radius.value, self.max_radius.value) * self.radius_multplier.value
                # Random velocity components
                vx = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                vy = np.random.uniform(-self.max_speed.value, self.max_speed.value) * self.speed_multiplier.value
                self.metaballs.append({'x': x, 'y': y, 'radius': r, 'vx': vx, 'vy': vy})

    def create_metaball_frame(self, metaballs, threshold, max_field_strength=None):
        """
        Generates a single frame with metaball blobs based on their properties.

        Args:
            metaballs (list): A list of dictionaries, where each dictionary
                              represents a metaball with 'x', 'y', and 'radius' keys.
            threshold (float): The field strength value above which pixels are colored.
            max_field_strength (float, optional): If provided, the field strength will be
                                                  normalized by this value for smoother coloring.
                                                  If None, a simple binary coloring is used.

        Returns:
            numpy.ndarray: An 8-bit grayscale or BGR image (frame) representing the metaballs.
        """
        # Create a grid of pixel coordinates for efficient calculation using NumPy
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Initialize a 2D array to store the total field strength for each pixel
        field_strength = np.zeros((self.height, self.width), dtype=np.float32)

        # Iterate through each metaball and add its contribution to the field
        for ball in metaballs:
            mx, my, r = ball['x'], ball['y'], ball['radius']

            # Calculate squared distance from each pixel to the metaball center
            # Adding a small epsilon (1e-6) to avoid division by zero if a pixel
            # is exactly at the metaball's center.
            dist_sq = (X - mx)**2 + (Y - my)**2 + 1e-6

            # Add the field contribution of this metaball.
            # The field strength decreases with the squared distance from the center.
            field_strength += (r**2) / dist_sq

        # --- Coloring the frame based on field strength ---
        if max_field_strength is not None:
            # Normalize the field strength to a 0-1 range based on max_field_strength.
            # This allows for a gradient effect, like a real lava lamp.
            # np.clip ensures values stay within 0 and 1 before scaling to 0-255.
            normalized_field = np.clip(field_strength / max_field_strength, 0, 1)

            # Map normalized field strength to a grayscale value (0-255)
            grayscale_image = (normalized_field * 255).astype(np.uint8)

            # Apply a colormap to create vibrant colors
            # cv2.COLORMAP_JET is a good general-purpose colormap.
            # You can try others like cv2.COLORMAP_HSV, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_TURBO
            image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)
        else:
            # Simple binary thresholding: pixels above threshold are white (255), others black (0)
            image = (field_strength >= threshold) * 255
            image = image.astype(np.uint8)
            # Convert grayscale to BGR to match the output type of the colormap branch
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

        return image

    def do_metaballs(self, frame: np.ndarray):
        """
        Updates metaball positions and generates the current frame, applying feedback.
        """
        if frame is not None:
            # Update metaball positions for the next frame

            if self.num_metaballs.value != len(self.metaballs):
                self.setup_metaballs()
            
            if self.current_radius_multiplier != self.radius_multplier.value or self.current_speed_multiplier != self.speed_multiplier.value:
                self.adjust_parameters()

            for ball in self.metaballs:
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']

                # Simple bouncing off the window edges
                # Check horizontal bounds
                if ball['x'] - ball['radius'] < 0:
                    ball['x'] = ball['radius']
                    ball['vx'] *= -1
                elif ball['x'] + ball['radius'] > self.width:
                    ball['x'] = self.width - ball['radius']
                    ball['vx'] *= -1

                # Check vertical bounds
                if ball['y'] - ball['radius'] < 0:
                    ball['y'] = ball['radius']
                    ball['vy'] *= -1
                elif ball['y'] + ball['radius'] > self.height:
                    ball['y'] = self.height - ball['radius']
                    ball['vy'] *= -1

            # Generate the current frame using the class's configuration
            current_frame = self.create_metaball_frame(self.metaballs,
                                                    threshold=self.threshold.value,
                                                    max_field_strength=self.smooth_coloring_max_field.value)
            
            # Apply feedback effect
            if self.previous_frame is None:
                # If it's the first frame, just use the current frame
                self.previous_frame = current_frame
            else:
                # Blend the current frame with the previous frame
                # alpha * current_frame + beta * previous_frame + gamma
                current_frame = cv2.addWeighted(current_frame, 1-self.feedback_alpha.value, 
                                                self.previous_frame, self.feedback_alpha.value, 0)
                self.previous_frame = current_frame # Store this blended frame for the next iteration

            return cv2.addWeighted(current_frame, 1 - self.frame_blend.value,
                                            frame, self.frame_blend.value, 0) if frame is not None else current_frame

        return frame

    # def metaballs_sliders(self, default_font_id=None, global_font_id=None):
    #     with dpg.collapsing_header(label=f"\tMetaballs", tag="metaballs"):
    #         num_metaballs_slider = TrackbarRow(
    #             "Num Metaballs",
    #             params.get("num_metaballs"),
    #             TrackbarCallback(params.get("num_metaballs"), "num_metaballs").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         min_radius_slider = TrackbarRow(
    #             "Min Radius",
    #             params.get("min_radius"),
    #             TrackbarCallback(params.get("min_radius"), "min_radius").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         max_radius_slider = TrackbarRow(
    #             "Max Radius",
    #             params.get("max_radius"),
    #             TrackbarCallback(params.get("max_radius"), "max_radius").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         max_speed_slider = TrackbarRow(
    #             "Max Speed",
    #             params.get("max_speed"),
    #             TrackbarCallback(params.get("max_speed"), "max_speed").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         threshold_slider = TrackbarRow(
    #             "Threshold",
    #             params.get("threshold"),
    #             TrackbarCallback(params.get("threshold"), "threshold").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         smooth_coloring_max_field_slider = TrackbarRow(
    #             "Smooth Coloring Max Field",
    #             params.get("smooth_coloring_max_field"),
    #             TrackbarCallback(params.get("smooth_coloring_max_field"), "smooth_coloring_max_field").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)
            
    #         feedback_alpha_slider = TrackbarRow(
    #             "Feedback Alpha",
    #             params.get("metaballs_feedback"),
    #             TrackbarCallback(params.get("metaballs_feedback"), "metaballs_feedback").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)

    #         frame_blend_slider = TrackbarRow(
    #             "Frame Blend",
    #             params.get("metaballs_frame_blend"),
    #             TrackbarCallback(params.get("metaballs_frame_blend"), "metaballs_frame_blend").__call__,
    #             self.reset_slider_callback,
    #             default_font_id)

    #     dpg.bind_item_font("metaballs", global_font_id)

