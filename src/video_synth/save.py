# Video Synth — real-time collaborative visual art synthesizer.
# Copyright (C) 2026 Kyle Henderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import yaml
import os
import random
import threading
import time
import numbers
from param import Param
import logging

log = logging.getLogger(__name__)

LOCK_FILENAME = 'session.lock'
AUTOSAVE_FILENAME = 'autosave.yaml'

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

        # --- Morph / interpolation state ---
        self._morph_lock = threading.Lock()
        self._morph_thread = None
        self._morph_stop = None  # threading.Event for the active morph, if any

    def init_save_dir(self, save_dir_name: str):
        import sys as _sys
        if getattr(_sys, 'frozen', False):
            # Running inside a PyInstaller bundle: use the bundle root (dist/VideoSynth/)
            project_root = _sys._MEIPASS
        else:
            # save.py lives at <root>/src/video_synth/, so the project root
            # (which holds save/) is two levels up.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
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

    def get_patch_entry(self, index):
        """
        Read a single patch entry (a flat {param_name: value} dict) from
        saved_values.yaml by integer index.

        Returns None if the file is missing/empty or the index is out of range.
        """
        if not os.path.exists(self.yaml_file_path):
            log.warning(f"YAML file not found at {self.yaml_file_path}.")
            return None
        try:
            with open(self.yaml_file_path, 'r') as f:
                saved_values = yaml.safe_load(f)
        except yaml.YAMLError as e:
            log.exception(f"Error loading YAML file {self.yaml_file_path}: {e}")
            return None

        if not saved_values or 'entries' not in saved_values or not saved_values['entries']:
            log.warning("No patches found in the YAML file.")
            return None

        entries = saved_values['entries']
        if not isinstance(index, int) or index < 0 or index >= len(entries):
            return None
        return entries[index]

    # --- Patch morph / interpolation ---

    def stop_morph(self):
        """Signal any running morph thread to stop and wait briefly for it."""
        with self._morph_lock:
            stop = self._morph_stop
            thread = self._morph_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def morph_to(self, index, duration, hz=60.0):
        """
        Linearly interpolate all numeric param values from their current values
        to the target patch (by integer index) over ``duration`` seconds in a
        background daemon thread.

        Only params that exist in the target entry, resolve to a real Param, and
        are numeric on both ends are lerped. Non-numeric / enum / string params
        are snapped to the target value once, at the end of the morph.

        If a morph is already running it is cancelled before this one starts.
        ``duration <= 0`` applies the target instantly. Returns True if a valid
        target was found and the morph was started/applied, else False.
        """
        target = self.get_patch_entry(index)
        if target is None:
            return False

        # Split target keys into numeric lerp candidates and snap-only keys.
        lerp_items = []   # list of (Param, start_value, end_value)
        snap_values = {}  # {param_name: value}
        for name, tgt_value in target.items():
            param = self._find_param(name)
            if param is None:
                continue
            cur_value = param.value
            if (isinstance(tgt_value, numbers.Number) and not isinstance(tgt_value, bool)
                    and isinstance(cur_value, numbers.Number) and not isinstance(cur_value, bool)):
                lerp_items.append((param, float(cur_value), float(tgt_value)))
            else:
                snap_values[name] = tgt_value

        # Cancel any in-flight morph before starting a new one.
        self.stop_morph()

        if duration is None or duration <= 0:
            for param, _start, end in lerp_items:
                param.value = end
            if snap_values:
                self.load_param_vals(snap_values)
            if self.patch_loaded_callback:
                self.patch_loaded_callback()
            with self._morph_lock:
                self.index = index
            return True

        stop = threading.Event()
        thread = threading.Thread(
            target=self._run_morph,
            args=(index, float(duration), float(hz), lerp_items, snap_values, stop),
            name="patch-morph",
            daemon=True,
        )
        with self._morph_lock:
            self._morph_stop = stop
            self._morph_thread = thread
        thread.start()
        return True

    def _run_morph(self, index, duration, hz, lerp_items, snap_values, stop):
        step = 1.0 / hz if hz > 0 else 1.0 / 60.0
        start_time = time.monotonic()
        try:
            while not stop.is_set():
                elapsed = time.monotonic() - start_time
                t = elapsed / duration
                if t >= 1.0:
                    break
                for param, start, end in lerp_items:
                    param.value = start + (end - start) * t
                time.sleep(step)

            if stop.is_set():
                return  # superseded by a newer morph; don't snap to target

            # Reached the end: snap everything to the exact target values.
            for param, _start, end in lerp_items:
                param.value = end
            if snap_values:
                self.load_param_vals(snap_values)
            with self._morph_lock:
                self.index = index
            if self.patch_loaded_callback:
                self.patch_loaded_callback()
            log.info(f"Morph to patch index {index} complete.")
        finally:
            with self._morph_lock:
                if self._morph_stop is stop:
                    self._morph_stop = None
                    self._morph_thread = None

    def load_param_vals(self, parsed_param_dict):
        for param_name, value in parsed_param_dict.items():
            param = self._find_param(param_name)
            if param is None:
                continue
            if param.options and hasattr(param.options, '__members__') and isinstance(value, str):
                try:
                    enum_member = param.options[value]
                    param.value = enum_member.value
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

    # --- Crash Recovery ---

    @property
    def _lock_path(self):
        return os.path.join(self.save_dir_path, LOCK_FILENAME)

    @property
    def _autosave_path(self):
        return os.path.join(self.save_dir_path, AUTOSAVE_FILENAME)

    def write_lock(self):
        """Write a session lock file. Its presence on next startup means the previous session crashed."""
        try:
            with open(self._lock_path, 'w') as f:
                import time
                f.write(str(time.time()))
        except OSError as e:
            log.warning(f"Could not write session lock file: {e}")

    def clear_lock(self):
        """Remove the session lock file on clean exit."""
        try:
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        except OSError as e:
            log.warning(f"Could not remove session lock file: {e}")

    def check_crash(self):
        """Return (crashed, has_autosave). crashed=True if lock file exists from prior run."""
        crashed = os.path.exists(self._lock_path)
        has_autosave = os.path.exists(self._autosave_path)
        return crashed, has_autosave

    def autosave(self):
        """Save all current param values to autosave.yaml (flat dict, not entries list)."""
        try:
            data = self.get_values()
            with open(self._autosave_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            log.debug("Autosaved params to autosave.yaml")
        except Exception as e:
            log.warning(f"Autosave failed: {e}")

    def recover_from_autosave(self):
        """Load param values from autosave.yaml."""
        if not os.path.exists(self._autosave_path):
            log.warning("No autosave file found, cannot recover.")
            return False
        try:
            with open(self._autosave_path, 'r') as f:
                data = yaml.safe_load(f)
            if data:
                self.load_param_vals(data)
                log.info("Recovered param state from autosave.yaml")
                return True
        except Exception as e:
            log.warning(f"Could not load autosave: {e}")
        return False