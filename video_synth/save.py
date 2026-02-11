import yaml
import os
import random
from param import Param
import logging

log = logging.getLogger(__name__)

class SaveController:
    """
    A class to handle the saving and loading of patches.
    Accepts a dict of ParamTables keyed by name (e.g. {"src_1": table, "mixer": table, ...}).
    """
    def __init__(self, param_tables: dict, yaml_filename='saved_values.yaml', save_dir_name='save'):
        self.param_tables = param_tables
        self.index = 0
        self.save_dir_path = self.init_save_dir(save_dir_name)
        self.yaml_file_path = os.path.join(self.save_dir_path, yaml_filename)
        self.patch_loaded_callback = None

    def init_save_dir(self, save_dir_name: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        save_dir_path = os.path.join(project_root, save_dir_name)
        os.makedirs(save_dir_path, exist_ok=True)
        return save_dir_path

    def save_patch(self):
        ROOT_KEY = 'entries'
        current_yaml_data = {}

        if os.path.exists(self.yaml_file_path):
            try:
                with open(self.yaml_file_path, 'r') as f:
                    current_yaml_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                log.exception(f"Could not read existing YAML data: {e}. Starting with empty data.")
                current_yaml_data = {}
        else:
            log.warning(f"File {self.yaml_file_path} does not exist. Creating a new file.")

        if current_yaml_data is None:
            current_yaml_data = {}
            
        entries_list = current_yaml_data.get(ROOT_KEY, [])
        
        if not isinstance(entries_list, list):
            entries_list = []

        new_data = self.get_values()
        entries_list.append(new_data)
        current_yaml_data[ROOT_KEY] = entries_list

        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(current_yaml_data, file, default_flow_style=False, sort_keys=False)
        log.info(f"Successfully saved patch to: {self.yaml_file_path}")

    def load_patch(self, direction):
        if not os.path.exists(self.yaml_file_path):
            log.exception(f"YAML file not found at {self.yaml_file_path}.")
            return

        try:
            with open(self.yaml_file_path, 'r') as f:
                saved_values = yaml.safe_load(f)

            if not saved_values or 'entries' not in saved_values or not saved_values['entries']:
                log.warning("No patches found in the YAML file.")
                return

            entries = saved_values['entries']
            num_entries = len(entries)

            if direction == 'next':
                self.index = (self.index + 1) % num_entries
            elif direction == 'prev':
                self.index = (self.index - 1) % num_entries
            elif direction == 'random':
                self.index = random.randint(0, num_entries - 1)

            parsed_values = entries[self.index]
            self.load_param_vals(parsed_values)

            log.info(f"Successfully loaded save index ({self.index}) from: {self.yaml_file_path}")
            if self.patch_loaded_callback:
                self.patch_loaded_callback()

        except yaml.YAMLError as e:
            log.exception(f"Error loading YAML file {self.yaml_file_path}: {e}")
        except Exception as e:
            log.exception(f"An unexpected error occurred while reading {self.yaml_file_path}: {e}")

    def load_next_patch(self):
        self.load_patch('next')

    def load_prev_patch(self):
        self.load_patch('prev')

    def load_random_patch(self):
        self.load_patch('random')

    def load_param_vals(self, parsed_param_dict):
        for param_name, value in parsed_param_dict.items():
            param = self._find_param(param_name)
            if param is None:
                continue
            if param.options and hasattr(param.options, '__members__') and isinstance(value, str):
                try:
                    enum_member = param.options[value]
                    param.value = enum_member
                except KeyError:
                    log.warning(f"Could not find enum member '{value}' for param '{param_name}'. Using default.")
            else:
                param.value = value

    def _find_param(self, param_name):
        """Search all param tables for a param by name."""
        for table in self.param_tables.values():
            if param_name in table:
                return table[param_name]
        return None

    def get_values(self):
        new_data = {}
        for table_name, table in self.param_tables.items():
            for param_name, param_obj in table.items():
                if isinstance(param_obj, Param):
                    if hasattr(param_obj.value, 'name'):
                        new_data[param_name] = param_obj.value.name
                    else:
                        new_data[param_name] = param_obj.value
                else:
                    log.warning(f"{param_name} is not a Param object, cannot save value")
        log.debug(new_data)
        return new_data