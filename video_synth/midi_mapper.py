"""
Generic MIDI controller mapping module with learn mode and YAML persistence.

This module runs alongside the existing hard-coded MIDI controller classes in midi.py.
It allows any MIDI controller to be mapped to any parameter via a learn mode,
and persists those mappings to a YAML file for automatic reload on startup.

Parameters are addressed by qualified keys: "group_name/param_name"
(e.g. "Src 1 Effects/hue_shift") to avoid collisions between groups.

Usage:
    param_tables = {"Src 1 Effects": src1_table, "Mixer": mixer_table, ...}
    mapper = MidiMapper(param_tables)
    mapper.start()
    mapper.start_learn("Src 1 Effects/hue_shift")
    mapper.stop()
"""

import mido
import yaml
import threading
import logging
from pathlib import Path

from midi import MidiProcessor, ControllerBase
from param import ParamTable

log = logging.getLogger(__name__)

MAPPINGS_FILENAME = "midi_mappings.yaml"
SEPARATOR = "/"


class MidiMapperController(ControllerBase):
    """
    A generic MIDI controller that routes CC messages to parameters
    based on a configurable mapping dictionary.

    Mappings use qualified keys ("group/param_name") to uniquely identify
    parameters across multiple groups that may share param names.
    """

    def __init__(self, port_name, param_tables, mappings=None):
        self.port_name = port_name
        self.param_tables = param_tables  # {"group_name": (ParamTable, group_filter|None)}
        # {cc_number (int): qualified_key (str "group/param")}
        self.mappings = mappings if mappings is not None else {}
        # One MidiProcessor per CC for independent smoothing state
        self._processors = {}

    def _get_processor(self, cc):
        if cc not in self._processors:
            self._processors[cc] = MidiProcessor(
                min_midi=0, max_midi=127,
                base_smoothing=0.05, acceleration_factor=0.05
            )
        return self._processors[cc]

    def _resolve_param(self, qualified_key):
        """Resolve a qualified key to (ParamTable, param_name, Param) or None."""
        if SEPARATOR not in qualified_key:
            return None
        group_name, param_name = qualified_key.split(SEPARATOR, 1)
        entry = self.param_tables.get(group_name)
        if entry is None:
            return None
        table = entry[0]  # (ParamTable, group_filter)
        if param_name not in table:
            return None
        return table, param_name, table[param_name]

    def set_values(self, control, value):
        qualified_key = self.mappings.get(control)
        if qualified_key is None:
            return

        resolved = self._resolve_param(qualified_key)
        if resolved is None:
            log.warning(f"Mapped param '{qualified_key}' not found for CC {control}")
            return

        table, param_name, param = resolved
        processor = self._get_processor(control)
        smoothed = processor.process_message(control, value, param.min, param.max)
        table.set(param_name, smoothed)

    def add_mapping(self, cc, qualified_key):
        self.mappings[cc] = qualified_key
        log.info(f"Mapped CC {cc} -> '{qualified_key}' on '{self.port_name}'")

    def remove_mapping(self, cc):
        removed = self.mappings.pop(cc, None)
        if removed:
            self._processors.pop(cc, None)
            log.info(f"Removed mapping CC {cc} (was '{removed}') on '{self.port_name}'")


class MidiMapper:
    """
    Manages MIDI learn mode, mapping persistence, and per-port listener threads.

    Accepts a dict of named ParamTables to preserve parameter identity across
    groups that share param names (e.g. hue_shift in src_1, src_2, post).
    """

    def __init__(self, param_tables):
        """
        Args:
            param_tables: dict of {"group_label": ParamTable, ...}
                e.g. {"Src 1 Effects": src1_effects.params, "Mixer": mixer.params}
        """
        self.param_tables = param_tables

        # Per-port controllers: {"port_name": MidiMapperController}
        self.controllers = {}

        # Thread management
        self._threads = []
        self._stop_event = threading.Event()

        # Learn mode state (thread-safe via lock)
        self._learn_lock = threading.Lock()
        self._learning = False
        self._learn_target = None  # qualified key "group/param_name"

        # Resolve save directory
        script_dir = Path(__file__).parent
        self._save_dir = script_dir / ".." / "save"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_path = self._save_dir / MAPPINGS_FILENAME

    def get_all_qualified_keys(self):
        """
        Returns a dict of {group_name: [param_name, ...]} for building the GUI.
        Filters params by their group attribute when a group_filter is provided.
        """
        result = {}
        for group_name, (table, group_filter) in self.param_tables.items():
            if group_filter is not None:
                result[group_name] = sorted(
                    name for name, param in table.items()
                    if param.group == group_filter
                )
            else:
                result[group_name] = sorted(table.keys())
        return result

    def start(self):
        """Scan MIDI ports, load saved mappings, and start listener threads."""
        saved_mappings = self._load_mappings_from_yaml()

        try:
            input_ports = mido.get_input_names()
        except Exception as e:
            log.exception(f"Failed to scan MIDI ports: {e}")
            return

        if not input_ports:
            log.info("No MIDI input ports found.")
            return

        log.info(f"Found {len(input_ports)} MIDI input port(s)")

        for port_name in input_ports:
            port_mappings = saved_mappings.get(port_name, {})

            controller = MidiMapperController(
                port_name=port_name,
                param_tables=self.param_tables,
                mappings=port_mappings
            )
            self.controllers[port_name] = controller

            if port_mappings:
                log.info(f"Loaded {len(port_mappings)} mapping(s) for '{port_name}'")

            thread = threading.Thread(
                target=self._listener_thread,
                args=(port_name, controller),
                daemon=True
            )
            thread.start()
            self._threads.append(thread)

        log.info(f"Started {len(self._threads)} listener thread(s)")

    def stop(self):
        """Stop all listener threads."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5)
            if thread.is_alive():
                log.warning(f"Listener thread did not terminate gracefully")
        self._threads.clear()
        log.info("All listener threads stopped")

    def start_learn(self, qualified_key):
        """
        Enter learn mode. The next CC message received from any port
        will be mapped to the given qualified key ("group/param_name").

        Returns:
            True if learn mode was started, False if param doesn't exist.
        """
        if SEPARATOR not in qualified_key:
            log.warning(f"Cannot learn - invalid key format '{qualified_key}' (expected 'group/param')")
            return False

        group_name, param_name = qualified_key.split(SEPARATOR, 1)
        entry = self.param_tables.get(group_name)
        if entry is None or param_name not in entry[0]:
            log.warning(f"Cannot learn - param '{qualified_key}' not found")
            return False

        with self._learn_lock:
            self._learning = True
            self._learn_target = qualified_key

        log.info(f"Learn mode ON - waiting for CC to map to '{qualified_key}'")
        return True

    def cancel_learn(self):
        """Cancel learn mode without creating a mapping."""
        with self._learn_lock:
            was_learning = self._learning
            self._learning = False
            self._learn_target = None

        if was_learning:
            log.info("Learn mode cancelled")

    def get_mappings(self):
        """
        Returns the current mappings for all ports.
        Returns:
            dict: {"port_name": {cc: qualified_key, ...}, ...}
        """
        result = {}
        for port_name, controller in self.controllers.items():
            if controller.mappings:
                result[port_name] = dict(controller.mappings)
        return result

    def get_learn_state(self):
        """
        Returns the current learn mode state.
        Returns:
            dict: {"learning": bool, "target": str or None}
        """
        with self._learn_lock:
            return {
                "learning": self._learning,
                "target": self._learn_target
            }

    def clear_mapping(self, port_name, cc):
        """Remove a single mapping."""
        controller = self.controllers.get(port_name)
        if controller:
            controller.remove_mapping(cc)
            self.save_mappings()

    def clear_all_mappings(self, port_name=None):
        """Clear all mappings, optionally for a specific port only."""
        if port_name:
            controller = self.controllers.get(port_name)
            if controller:
                controller.mappings.clear()
                controller._processors.clear()
                log.info(f"Cleared all mappings for '{port_name}'")
        else:
            for name, controller in self.controllers.items():
                controller.mappings.clear()
                controller._processors.clear()
            log.info("Cleared all mappings for all ports")

        self.save_mappings()

    def save_mappings(self):
        """Persist current mappings to YAML."""
        data = {"ports": {}}

        for port_name, controller in self.controllers.items():
            if controller.mappings:
                data["ports"][port_name] = {
                    "mappings": {int(cc): key for cc, key in controller.mappings.items()}
                }

        try:
            with open(self._yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            log.info(f"Saved mappings to {self._yaml_path}")
        except Exception as e:
            log.exception(f"Failed to save mappings: {e}")

    def _load_mappings_from_yaml(self):
        """
        Load mappings from YAML file.
        Returns:
            dict: {"port_name": {cc_int: qualified_key, ...}, ...}
        """
        if not self._yaml_path.exists():
            return {}

        try:
            with open(self._yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data or "ports" not in data:
                return {}

            result = {}
            for port_name, port_data in data["ports"].items():
                raw_mappings = port_data.get("mappings", {})
                result[port_name] = {int(cc): key for cc, key in raw_mappings.items()}

            return result

        except yaml.YAMLError as e:
            log.exception(f"Failed to parse {self._yaml_path}: {e}")
            return {}
        except Exception as e:
            log.exception(f"Failed to load mappings: {e}")
            return {}

    def _listener_thread(self, port_name, controller):
        """
        Listener thread for a single MIDI port.
        Handles learn mode interception before normal routing.
        """
        try:
            with mido.open_input(port_name) as inport:
                log.info(f"Listening on '{port_name}'")

                for msg in inport:
                    if self._stop_event.is_set():
                        break

                    if not hasattr(msg, 'control'):
                        continue

                    cc = msg.control
                    value = msg.value

                    # Check learn mode (thread-safe)
                    with self._learn_lock:
                        if self._learning and self._learn_target:
                            controller.add_mapping(cc, self._learn_target)
                            log.info(
                                f"Learned CC {cc} on '{port_name}' "
                                f"-> '{self._learn_target}'"
                            )
                            self._learning = False
                            self._learn_target = None
                            self.save_mappings()
                            continue

                    # Normal routing
                    controller.set_values(cc, value)

        except ValueError as e:
            log.warning(f"Could not open port '{port_name}': {e}")
        except Exception as e:
            log.warning(f"Error on port '{port_name}': {e}")
        finally:
            log.info(f"Listener thread for '{port_name}' terminated")
