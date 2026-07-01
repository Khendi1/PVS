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

"""
Parameter history for undo/redo.

Maintains a bounded ring buffer of parameter snapshots so recent state changes
can be undone (Ctrl+Z) and redone. A snapshot is a flat dict of param name →
value; enum/dropdown values are stored by their member name so they can be
restored the same way SaveController.load_param_vals does.
"""

import logging
import threading
from collections import deque

log = logging.getLogger(__name__)


class ParamHistory:
    """
    Ring buffer of the last N parameter states with undo/redo support.

    Thread-safe: API requests and the video loop mutate params from different
    threads, so snapshot/undo/redo are all guarded by a single lock.
    """

    def __init__(self, params, capacity: int = 50):
        """
        Args:
            params: The combined ParamTable holding every Param.
            capacity (int): Maximum number of undo states to retain.
        """
        self.params = params
        self.capacity = max(1, int(capacity))
        self._undo = deque(maxlen=self.capacity)
        self._redo = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    # --- Snapshot capture / restore -----------------------------------------

    def _capture(self) -> dict:
        """Return a flat dict of the current value of every param.

        Enum/dropdown values are stored by their `.name` so restore can look
        them up on the param's `options` enum, matching SaveController.
        """
        snap = {}
        for name, param in self.params.params.items():
            value = param.value
            if hasattr(value, 'name'):
                snap[name] = value.name
            else:
                snap[name] = value
        return snap

    def _apply(self, snapshot: dict):
        """Apply a snapshot to the params.

        Only touches params that still exist. Enum/string values are resolved
        via the param's options enum the same way SaveController.load_param_vals
        does, so dropdown params restore correctly.
        """
        for name, value in snapshot.items():
            if name not in self.params.params:
                continue
            param = self.params.params[name]
            if param.options and hasattr(param.options, '__members__') and isinstance(value, str):
                try:
                    enum_member = param.options[value]
                    param.value = enum_member.value
                except KeyError:
                    log.warning(f"Could not find enum member '{value}' for param '{name}'. Skipping.")
            else:
                param.value = value

    # --- Public API ----------------------------------------------------------

    def snapshot(self):
        """Capture the current param state and push it onto the undo stack.

        Pushing a new snapshot clears the redo stack. A snapshot identical to
        the most recent one is skipped to avoid redundant no-op entries.
        """
        with self._lock:
            state = self._capture()
            if self._undo and self._undo[-1] == state:
                return
            self._undo.append(state)
            self._redo.clear()

    def undo(self) -> bool:
        """Revert to the previous param state.

        Saves the current state onto the redo stack, pops the undo stack, and
        applies that snapshot. Returns True if a state was applied.
        """
        with self._lock:
            if not self._undo:
                return False
            self._redo.append(self._capture())
            state = self._undo.pop()
            self._apply(state)
            return True

    def redo(self) -> bool:
        """Re-apply the most recently undone param state.

        Saves the current state onto the undo stack, pops the redo stack, and
        applies that snapshot. Returns True if a state was applied.
        """
        with self._lock:
            if not self._redo:
                return False
            self._undo.append(self._capture())
            state = self._redo.pop()
            self._apply(state)
            return True

    def counts(self) -> dict:
        """Return the number of available undo and redo states."""
        with self._lock:
            return {"undo": len(self._undo), "redo": len(self._redo)}
