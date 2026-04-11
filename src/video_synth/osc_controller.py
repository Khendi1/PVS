"""
Open Sound Control (OSC) controller module with learn mode and YAML persistence.

Allows any OSC-capable application (TouchOSC, SuperCollider, Max/MSP, DAWs, etc.)
to control video synth parameters over UDP. Supports learn mode for mapping
OSC addresses to parameters, with persistent YAML storage.

Parameters are addressed by qualified keys: "group_name/param_name"
(e.g. "Src 1 Effects/hue_shift") to avoid collisions between groups.

Usage:
    param_tables = {"Src 1 Effects": (src1_table, group_filter), "Mixer": (mixer_table, None), ...}
    osc = OSCController(param_tables, host="0.0.0.0", port=9000)
    osc.start()
    osc.start_learn("Src 1 Effects/hue_shift")
    osc.stop()
"""

import yaml
import threading
import logging
from pathlib import Path
from pythonosc import dispatcher, osc_server

log = logging.getLogger(__name__)

MAPPINGS_FILENAME = "osc_mappings.yaml"
SEPARATOR = "/"


class OSCController:
    """
    Manages OSC server, learn mode, mapping persistence, and message routing.

    Accepts the same param_tables dict format as MidiMapper:
        {"group_label": (ParamTable, group_filter|None), ...}
    """

    def __init__(self, param_tables, host="0.0.0.0", port=9000):
        self.param_tables = param_tables
        self.host = host
        self.port = port

        # {osc_address (str): qualified_key (str "group/param")}
        self.mappings = {}

        # Server state
        self._server = None
        self._server_thread = None

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
        """Load saved mappings, create dispatcher, and start the OSC server thread."""
        if self._server is not None:
            log.warning("OSC server already running")
            return

        self._load_mappings_from_yaml()

        disp = dispatcher.Dispatcher()
        disp.set_default_handler(self._on_osc_message)

        try:
            self._server = osc_server.ThreadingOSCUDPServer(
                (self.host, self.port), disp
            )
        except OSError as e:
            log.error(f"Failed to start OSC server on {self.host}:{self.port}: {e}")
            return

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._server_thread.start()
        log.info(f"OSC server listening on {self.host}:{self.port}")

        if self.mappings:
            log.info(f"Loaded {len(self.mappings)} OSC mapping(s)")

    def stop(self):
        """Stop the OSC server."""
        if self._server is not None:
            self._server.shutdown()
            if self._server_thread is not None:
                self._server_thread.join(timeout=5)
            self._server = None
            self._server_thread = None
            log.info("OSC server stopped")

    def _on_osc_message(self, address, *args):
        """
        Default handler for all incoming OSC messages.
        Routes through learn mode or normal mapping resolution.
        """
        if not args:
            return

        value = args[0]
        if not isinstance(value, (int, float)):
            return

        # Check learn mode (thread-safe)
        with self._learn_lock:
            if self._learning and self._learn_target:
                self.mappings[address] = self._learn_target
                log.info(f"Learned OSC '{address}' -> '{self._learn_target}'")
                self._learning = False
                self._learn_target = None
                self.save_mappings()
                return

        # Normal routing
        qualified_key = self.mappings.get(address)
        if qualified_key is None:
            return

        resolved = self._resolve_param(qualified_key)
        if resolved is None:
            log.warning(f"Mapped param '{qualified_key}' not found for OSC '{address}'")
            return

        table, param_name, param = resolved

        # Map OSC value [0.0, 1.0] to param's [min, max] range
        mapped = param.min + float(value) * (param.max - param.min)
        table.set(param_name, mapped)

    def _resolve_param(self, qualified_key):
        """Resolve a qualified key to (ParamTable, param_name, Param) or None."""
        if SEPARATOR not in qualified_key:
            return None
        # Split on first separator only (group names don't contain /)
        group_name, param_name = qualified_key.split(SEPARATOR, 1)
        entry = self.param_tables.get(group_name)
        if entry is None:
            return None
        table = entry[0]  # (ParamTable, group_filter)
        if param_name not in table:
            return None
        return table, param_name, table[param_name]

    # --- Learn Mode ---

    def start_learn(self, qualified_key):
        """
        Enter learn mode. The next OSC message received will be mapped
        to the given qualified key ("group/param_name").

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

        log.info(f"OSC learn mode ON - waiting for message to map to '{qualified_key}'")
        return True

    def cancel_learn(self):
        """Cancel learn mode without creating a mapping."""
        with self._learn_lock:
            was_learning = self._learning
            self._learning = False
            self._learn_target = None

        if was_learning:
            log.info("OSC learn mode cancelled")

    def get_learn_state(self):
        """Returns the current learn mode state."""
        with self._learn_lock:
            return {
                "learning": self._learning,
                "target": self._learn_target
            }

    # --- Mapping Management ---

    def get_mappings(self):
        """Returns the current mappings: {osc_address: qualified_key, ...}"""
        return dict(self.mappings)

    def clear_mapping(self, address):
        """Remove a single mapping by OSC address."""
        removed = self.mappings.pop(address, None)
        if removed:
            log.info(f"Removed OSC mapping '{address}' (was '{removed}')")
            self.save_mappings()

    def clear_all_mappings(self):
        """Clear all mappings."""
        self.mappings.clear()
        self.save_mappings()
        log.info("Cleared all OSC mappings")

    # --- Persistence ---

    def save_mappings(self):
        """Persist current mappings to YAML."""
        data = {"mappings": {addr: key for addr, key in self.mappings.items()}}

        try:
            with open(self._yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            log.info(f"Saved OSC mappings to {self._yaml_path}")
        except Exception as e:
            log.exception(f"Failed to save OSC mappings: {e}")

    def _load_mappings_from_yaml(self):
        """Load mappings from YAML file."""
        if not self._yaml_path.exists():
            return

        try:
            with open(self._yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data or "mappings" not in data:
                return

            self.mappings = {str(addr): key for addr, key in data["mappings"].items()}

        except yaml.YAMLError as e:
            log.exception(f"Failed to parse {self._yaml_path}: {e}")
        except Exception as e:
            log.exception(f"Failed to load OSC mappings: {e}")
