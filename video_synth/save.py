import dearpygui.dearpygui as dpg
from datetime import datetime
import yaml
import os
import numpy as np
import random
from gui_elements import Button

class SaveController:
    """
    A class to handle the save, load, and randomize buttons in the GUI.
    It provides methods to save current parameter values, load next/previous values,
    and randomize parameter values.
    """
    def __init__(self, params, width, height, 
                yaml_filename: str = 'saved_values.yaml',
                save_dir_name: str = 'save'):
        self.params = params
        self.index = 0
        self.width = width
        self.height = height
        self.save_dir_path = self.init_save_dir(save_dir_name, yaml_filename)
        self.yaml_file_path = os.path.join(self.save_dir_path, yaml_filename)
        self.save = Button("Save", "save")
        self.fwd = Button("Load Next", "load_next")
        self.prev = Button("Load Prev", "load_prev")
        self.rand = Button("Load Random", "load_rand")


    def init_save_dir(self, save_dir_name: str, yaml_filename: str):
        
        # Get the directory of the current script and go up one level to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Construct the path to the save dir and create it if it doesn't exist
        save_dir_path = os.path.join(project_root, save_dir_name)
        os.makedirs(save_dir_path, exist_ok=True)

        return save_dir_path


    def on_fwd_button_click(self):

        print(f"Forward button clicked!")

        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = (self.index + 1) % len(saved_values.all())
            self.load_param_vals(saved_values)
            
        except Exception as e:
            print(f"Error loading values: {e}")


    def on_prev_button_click(self):

        print(f"Prev button clicked!")

        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = (self.index - 1) % len(saved_values[0])
            self.load_param_vals(saved_values)

        except Exception as e:
            print(f"Error loading values: {e}")


    def on_rand_button_click(self):
        print(f"Random button clicked!")
    
        # get values from saved_values.yaml
        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = random.randint(0, len(saved_values[0]) - 1)
            self.load_param_vals(saved_values)
            
        except Exception as e:
            print(f"Error loading values: {e}")


    def load_param_vals(self, saved_values):
        d = saved_values[0][self.index]
        print(f"loaded values at index {self.index}: {d}\n\n")
        for param_name in params.keys():
            for tag in d.keys():
                if tag == param_name:
                    params[param_name].set(d[tag])
                    dpg.set_value(param_name, d[tag])


    def save2(self):
        
        current_yaml_data = {}

        # Read existing YAML data if the file exists
        if os.path.exists(self.yaml_file_path):
            try:
                with open(self.yaml_file_path, 'r') as f:
                    current_yaml_data = yaml.safe_load(f)
                    if current_yaml_data is None: # Handle empty YAML file
                        current_yaml_data = {}
                print(f"Successfully loaded existing YAML from: {self.yaml_file_path}")
                print(f"Current YAML data: {current_yaml_data}")
            except yaml.YAMLError as e:
                print(f"Error loading YAML file {self.yaml_file_path}: {e}")
                return
            except Exception as e:
                print(f"An unexpected error occurred while reading {self.yaml_file_path}: {e}")
                return
        else:
            print(f"YAML file not found at {self.yaml_file_path}. A new one will be created.")


    def on_save_button_click(self, frame: np.ndarray):
        self.save2(self)
        
        # TODO: determine best way to grap a frame for saving w/o having to pass into method
        # cv2.imwrite(f"{date_time_str}.jpg", frame)

    def create_save_buttons(self):
        """
        Creates the save, load, and randomize buttons in the GUI.
        """
        width = self.width - 20

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.save.label, callback=self.on_save_button_click, user_data=self.save.tag, width=width//3)
            dpg.add_button(label=self.fwd.label, callback=self.on_fwd_button_click, user_data=self.fwd.tag, width=width//3)
            dpg.add_button(label=self.prev.label, callback=self.on_prev_button_click, user_data=self.prev.tag, width=width//3)

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.rand.label, callback=self.on_rand_button_click, user_data=self.rand.tag, width=width//3)


class Timeline:
    def __init__(self, sliders):
        self.labels = [s.labels for s in sliders]
        self.current_state = [s.value for s in sliders]
        self.prev_state = self.current_state

    def update(self, sliders=None, buttons=None):
        if sliders is not None:
            self.labels = [s.labels for s in sliders]
            self
            self.current_state = [s.value for s in sliders]
        self.prev_state = self.current_state
        self.current_state = [s.value for s in sliders]
