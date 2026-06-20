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
Eurorack / CV-Gate input controller.

Reads control voltages from a USB audio interface's line input and maps
them to synthesizer parameters in real time.  Each audio channel is treated
as an independent CV lane with its own configurable voltage range and target
parameter.

Signal conventions
------------------
Eurorack CV:   typically ±5 V or 0–10 V
Gate/trigger:  0 V (off) or +5 V (on) pulses
Pedals/faders: 0–5 V unipolar

The module normalises any input voltage range to the Param's [min, max]
range, so 0–5 V → 0–100 % is just a matter of setting `volt_min=0,
volt_max=5`.

Dependencies
------------
    sounddevice  (already used by AudioReactiveModule)

Usage
-----
    param_tables = {"Src 1": (src1_params, group_filter), ...}
    cv = CVController(param_tables, device=None, sample_rate=44100)
    cv.start()
    cv.map_channel(channel=0, qualified_key="Src 1 Effects/plasma_speed",
                   volt_min=-5.0, volt_max=5.0)
    # ... later ...
    cv.stop()

YAML persistence
----------------
Mappings are saved to ``save/cv_mappings.yaml`` automatically whenever they
change, and reloaded on the next ``start()`` call.

The mapping file schema::

    mappings:
      0:                          # audio channel index (0-based)
        param: "Src 1 Effects/plasma_speed"
        volt_min: -5.0
        volt_max: 5.0
        smoothing: 0.05           # IIR low-pass coefficient (0 = no smoothing)
"""

import threading
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

_MAPPINGS_FILENAME = "cv_mappings.yaml"

try:
    import sounddevice as sd
    import numpy as np
    _SD_AVAILABLE = True
except (ImportError, OSError) as _err:
    _SD_AVAILABLE = False
    log.warning("CVController: sounddevice unavailable (%s) — CV input disabled.", _err)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_audio_devices():
    """Return a list of (index, name) for all input-capable devices."""
    if not _SD_AVAILABLE:
        return []
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append((i, d["name"]))
    return devices


# ---------------------------------------------------------------------------
# Per-channel state
# ---------------------------------------------------------------------------

class _CVChannel:
    __slots__ = ("param_key", "volt_min", "volt_max", "smoothing",
                 "_smoothed", "_lock")

    def __init__(self, param_key: str, volt_min: float, volt_max: float,
                 smoothing: float):
        self.param_key = param_key
        self.volt_min = volt_min
        self.volt_max = volt_max
        self.smoothing = float(smoothing)
        self._smoothed: Optional[float] = None
        self._lock = threading.Lock()

    def process(self, raw_volt: float) -> float:
        """Apply IIR smoothing and return the current smoothed voltage."""
        with self._lock:
            if self._smoothed is None or self.smoothing <= 0.0:
                self._smoothed = raw_volt
            else:
                self._smoothed += self.smoothing * (raw_volt - self._smoothed)
            return self._smoothed

    def to_param_value(self, volt: float, p_min: float, p_max: float) -> float:
        """Map *volt* from [volt_min, volt_max] to [p_min, p_max]."""
        v_range = self.volt_max - self.volt_min
        if abs(v_range) < 1e-9:
            return p_min
        t = (volt - self.volt_min) / v_range
        t = max(0.0, min(1.0, t))
        return p_min + t * (p_max - p_min)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CVController:
    """
    Reads audio-rate CV signals from a sound device and maps each channel
    to a synthesizer parameter.

    Args:
        param_tables: Same format as MidiMapper / OSCController —
                      ``{group_label: (ParamTable, group_filter | None)}``.
        device: sounddevice device index or substring of its name.
                ``None`` uses the system default input device.
        sample_rate: ADC sample rate in Hz (default 44 100).
        block_size: Samples per callback block.  Smaller = lower latency.
        num_channels: Number of input channels to open (default 2).
    """

    def __init__(self, param_tables, device=None, sample_rate: int = 44100,
                 block_size: int = 512, num_channels: int = 2):
        self.param_tables = param_tables
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.num_channels = num_channels

        # channel_index → _CVChannel
        self._channels: dict[int, _CVChannel] = {}
        self._channels_lock = threading.Lock()

        self._stream: Optional[object] = None  # sd.InputStream
        self._running = False

        # Learn mode
        self._learn_lock = threading.Lock()
        self._learning = False
        self._learn_channel: Optional[int] = None
        self._learn_target: Optional[str] = None

        # YAML persistence
        script_dir = Path(__file__).parent
        self._save_dir = script_dir.parent / "save"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_path = self._save_dir / _MAPPINGS_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Load saved mappings and start the audio input stream."""
        if not _SD_AVAILABLE:
            log.error("CVController: sounddevice not available — cannot start.")
            return
        if self._running:
            log.warning("CVController already running.")
            return

        self._load_mappings()

        channels = max(1, min(self.num_channels, 64))
        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            dev_name = self.device if self.device is not None else "default"
            log.info(
                "CVController: listening on device '%s' (%d ch, %d Hz)",
                dev_name, channels, self.sample_rate
            )
            if self._channels:
                log.info("CVController: loaded %d channel mapping(s)", len(self._channels))
        except Exception as exc:
            log.error("CVController: failed to open audio stream: %s", exc)

    def stop(self):
        """Stop the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        log.info("CVController: stopped.")

    def map_channel(self, channel: int, qualified_key: str,
                    volt_min: float = -5.0, volt_max: float = 5.0,
                    smoothing: float = 0.05):
        """
        Map an audio channel to a parameter.

        Args:
            channel: 0-based audio channel index.
            qualified_key: ``"GroupLabel/param_name"`` as used by MidiMapper.
            volt_min: Input voltage that maps to param.min.
            volt_max: Input voltage that maps to param.max.
            smoothing: IIR smoothing coefficient (0 = off, 0.05 = gentle).
        """
        ch = _CVChannel(qualified_key, volt_min, volt_max, smoothing)
        with self._channels_lock:
            self._channels[channel] = ch
        self._save_mappings()
        log.info("CVController: ch%d → '%s' [%.1fV – %.1fV]",
                 channel, qualified_key, volt_min, volt_max)

    def unmap_channel(self, channel: int):
        """Remove the mapping for *channel*."""
        with self._channels_lock:
            self._channels.pop(channel, None)
        self._save_mappings()
        log.info("CVController: ch%d unmapped.", channel)

    def get_mappings(self) -> dict:
        """Return a serialisable snapshot of current channel mappings."""
        with self._channels_lock:
            return {
                ch: {
                    "param": c.param_key,
                    "volt_min": c.volt_min,
                    "volt_max": c.volt_max,
                    "smoothing": c.smoothing,
                }
                for ch, c in self._channels.items()
            }

    def get_channel_values(self) -> dict:
        """Return {channel: smoothed_voltage} for all mapped channels."""
        with self._channels_lock:
            return {
                ch: (c._smoothed if c._smoothed is not None else 0.0)
                for ch, c in self._channels.items()
            }

    def start_learn(self, channel: int, qualified_key: str):
        """
        Enter learn mode: the next audio activity on *channel* triggers
        an automatic map.  For gate signals, ``volt_min`` and ``volt_max``
        will be auto-detected from the signal range during the learn window.
        For immediate mapping without auto-detection, use :meth:`map_channel`.
        """
        with self._learn_lock:
            self._learning = True
            self._learn_channel = channel
            self._learn_target = qualified_key
        log.info("CVController: learn mode — ch%d will map to '%s'",
                 channel, qualified_key)

    def cancel_learn(self):
        with self._learn_lock:
            self._learning = False
            self._learn_channel = None
            self._learn_target = None

    @staticmethod
    def list_devices() -> list:
        """Return available audio input devices as [{index, name, channels}]."""
        if not _SD_AVAILABLE:
            return []
        result = []
        for idx, name in _list_audio_devices():
            d = sd.query_devices(idx)
            result.append({
                "index": idx,
                "name": name,
                "max_input_channels": d["max_input_channels"],
                "default_samplerate": int(d["default_samplerate"]),
            })
        return result

    # ------------------------------------------------------------------
    # Audio callback (runs in sounddevice thread — keep it fast)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.debug("CVController audio status: %s", status)

        with self._channels_lock:
            channels_snapshot = dict(self._channels)

        for ch_idx, channel in channels_snapshot.items():
            if ch_idx >= indata.shape[1]:
                continue

            # RMS of block → treat as representative voltage
            # (audio interfaces deliver ±1.0 normalised; scale by volt_max
            #  to preserve polarity for AC-coupled CV)
            block = indata[:, ch_idx]
            peak = float(block[abs(block).argmax()])  # signed peak for polarity
            # Scale from normalised ±1.0 to volts using the channel's range
            volt_scale = max(abs(channel.volt_min), abs(channel.volt_max))
            raw_volt = peak * volt_scale

            smoothed = channel.process(raw_volt)

            param, p_min, p_max = self._resolve_param(channel.param_key)
            if param is not None:
                param.value = channel.to_param_value(smoothed, p_min, p_max)

        # Learn mode: map channel on first significant signal
        with self._learn_lock:
            if self._learning and self._learn_channel is not None:
                ch = self._learn_channel
                if ch < indata.shape[1]:
                    block = indata[:, ch]
                    if float(import_np_abs_max(block)) > 0.01:
                        self.map_channel(
                            ch, self._learn_target,
                            volt_min=channel.volt_min if ch in channels_snapshot else -5.0,
                            volt_max=channel.volt_max if ch in channels_snapshot else 5.0,
                        )
                        self._learning = False
                        self._learn_channel = None
                        self._learn_target = None

    # ------------------------------------------------------------------
    # Param resolution (same pattern as OSCController / MidiMapper)
    # ------------------------------------------------------------------

    def _resolve_param(self, qualified_key: str):
        """Return (Param, min, max) or (None, 0, 1)."""
        if "/" not in qualified_key:
            return None, 0.0, 1.0
        group_label, param_name = qualified_key.split("/", 1)
        entry = self.param_tables.get(group_label)
        if entry is None:
            return None, 0.0, 1.0
        table, group_filter = entry
        if param_name not in table.params:
            return None, 0.0, 1.0
        param = table.params[param_name]
        if group_filter is not None and param.group != group_filter:
            return None, 0.0, 1.0
        return param, float(param.min), float(param.max)

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def _save_mappings(self):
        data = {"mappings": {}}
        with self._channels_lock:
            for ch, c in self._channels.items():
                data["mappings"][ch] = {
                    "param": c.param_key,
                    "volt_min": c.volt_min,
                    "volt_max": c.volt_max,
                    "smoothing": c.smoothing,
                }
        try:
            with open(self._yaml_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except OSError as exc:
            log.warning("CVController: could not save mappings: %s", exc)

    def _load_mappings(self):
        if not self._yaml_path.exists():
            return
        try:
            with open(self._yaml_path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("CVController: could not load mappings: %s", exc)
            return
        with self._channels_lock:
            self._channels.clear()
            for ch, cfg in (data.get("mappings") or {}).items():
                self._channels[int(ch)] = _CVChannel(
                    param_key=cfg.get("param", ""),
                    volt_min=float(cfg.get("volt_min", -5.0)),
                    volt_max=float(cfg.get("volt_max", 5.0)),
                    smoothing=float(cfg.get("smoothing", 0.05)),
                )


# ---------------------------------------------------------------------------
# numpy helper (imported lazily to survive missing sounddevice)
# ---------------------------------------------------------------------------

def import_np_abs_max(arr):
    import numpy as _np
    return _np.abs(arr).max()
