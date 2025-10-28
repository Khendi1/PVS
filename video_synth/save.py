import dearpygui.dearpygui as dpg
from datetime import datetime
import yaml
import os
import numpy as np
import random
from gui_elements import Toggle
from param import Param, ParamTable
import logging

log = logging.getLogger(__name__)

class SaveController:
    """
    A class to handle the enable the saving and loading of patches.

    It provides methods to save the current patch values to a local yaml file,
    load next/previous patch, and select random saved patches.
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
        self.save = Toggle("Save", "save")
        self.fwd = Toggle("Load Next", "load_next")
        self.prev = Toggle("Load Prev", "load_prev")
        self.rand = Toggle("Load Random", "load_rand")


    def init_save_dir(self, save_dir_name: str, yaml_filename: str):
        
        # Get the directory of the current script and go up one level to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Construct the path to the save dir and create it if it doesn't exist
        save_dir_path = os.path.join(project_root, save_dir_name)
        os.makedirs(save_dir_path, exist_ok=True)

        return save_dir_path


    def on_save_button_click(self, app_data, user_data):
        ROOT_KEY = 'entries'
        current_yaml_data = {}

        # Read existing YAML data if the file exists
        if os.path.exists(self.yaml_file_path):
            try:
                with open(self.yaml_file_path, 'r') as f:
                    current_yaml_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                log.exception(f"Could not read existing YAML data: {e}. Starting with empty data.")
                current_yaml_data = {}
        else:
            log.warning(f"File {self.yaml_file_path} does not exist. Creating a new file.")

        entries_list = current_yaml_data.get(ROOT_KEY, [])
        
        # error check
        if not isinstance(entries_list, list):
            entries_list = []

        new_data = self.get_values()
        entries_list.append(new_data)
        current_yaml_data[ROOT_KEY] = entries_list

        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(current_yaml_data, file, default_flow_style=False, sort_keys=False)
            

    def load_button_callback(self, sender, app_data, user_data):

        # Read existing YAML data if the file exists
        if os.path.exists(self.yaml_file_path):
            try:
                with open(self.yaml_file_path, 'r') as f:
                    saved_values = list(yaml.safe_load_all(f))

                print(f"beofre {self.index}, len {len(saved_values[0]['entries'])}")
                if user_data == self.fwd.tag:
                    self.index = (self.index + 1) % len(saved_values[0]['entries'])
                elif user_data == self.prev.tag:
                    self.index = (self.index - 1) % len(saved_values[0]['entries'])
                else:
                    self.index = random.randint(0, len(saved_values[0]['entries']) - 1)
                print(f"after {self.index}")

                parsed_values = saved_values[0]['entries'][self.index]
                self.load_param_vals(parsed_values)

                log.info(f"Successfully loaded save index ({self.index}) from: {self.yaml_file_path}")

            except yaml.YAMLError as e:
                print(f"Error loading YAML file {self.yaml_file_path}: {e}")
                return
            except Exception as e:
                print(f"An unexpected error occurred while reading {self.yaml_file_path}: {e}")
                return
        else:
            print(f"YAML file not found at {self.yaml_file_path}. A new one will be created.")


    def load_param_vals(self, parsed_param_dict):
        for param_name in self.params.keys():
            for tag in parsed_param_dict.keys():
                if tag == param_name:
                    self.params[param_name].value = parsed_param_dict[tag]
                    if dpg.does_item_exist(tag):
                        # item_type = dpg.get_item_info(tag)['type']
                        # log.debug(item_type)
                        # log.debug(f"found {tag}, setting to {parsed_param_dict[tag]}, type:{type(parsed_param_dict[tag])}")
                        dpg.set_value(tag, parsed_param_dict[tag])
                    else:
                        # log.warning(f"{tag} widget does not exists")
                        pass


    def get_values(self):
        """ Parse values from ParamTable Param objects, return as dict of param names and values"""
        new_data = {}
        for param_name, param_obj in self.params.items():
            if isinstance(param_obj, Param):
                new_data[param_name] = param_obj.value
            else:
                log.warning(f"{param_name} is not a Param object, cannot save value")
        print(new_data)
        return new_data


    def create_save_buttons(self):
        """
        Creates the save, load, and randomize buttons in the GUI.
        """
        width = self.width - 20

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.save.label, callback=self.on_save_button_click, user_data=self.save.tag, width=width//3)
            dpg.add_button(label=self.fwd.label, callback=self.load_button_callback, user_data=self.fwd.tag, width=width//3)
            dpg.add_button(label=self.prev.label, callback=self.load_button_callback, user_data=self.prev.tag, width=width//3)

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.rand.label, callback=self.load_button_callback, user_data=self.rand.tag, width=width//3)


""" Goal: store changes to paramters to support undo/redo buttons """
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
