"""
Audio reactive module for the video synthesizer.

Provides real-time FFT analysis of audio input (microphone/line-in) and allows
linking frequency band energies to any parameter, analogous to the LFO system.

Classes:
    AudioBand - A single frequency band link to a parameter (analogous to LFO)
    AudioReactiveModule - Manages audio capture and all AudioBand instances (analogous to OscBank)
"""

import threading
import logging
import numpy as np
from param import Param, ParamTable
from common import Groups, Widget

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    log.warning(f"sounddevice not available: {e}. Audio reactive features will be disabled.")


BAND_NAMES = ['Bass', 'Low Mid', 'Mid', 'High Mid', 'Treble']

# Frequency ranges for each band (Hz)
BAND_RANGES = [
    (20, 250),      # Bass
    (250, 500),     # Low Mid
    (500, 2000),    # Mid
    (2000, 6000),   # High Mid
    (6000, 20000),  # Treble
]

NUM_BANDS = len(BAND_NAMES)


class AudioBand:
    """
    Represents a single audio frequency band linked to a parameter.
    Analogous to LFO in lfo.py.

    Each AudioBand tracks a specific frequency band's energy and maps it
    to a linked parameter's range using an envelope follower for smoothing.
    """

    def __init__(self, params: ParamTable, name: str, band_index: int = 0):
        self.name = name
        self.band_index = band_index
        self.linked_param = None
        self.smoothed_value = 0.0

        group = None
        subgroup = None

        self.band_select = params.add(
            f"{name}_band", min=0, max=NUM_BANDS - 1, default=band_index,
            group=group, subgroup=subgroup, type=Widget.DROPDOWN,
            options={bn: i for i, bn in enumerate(BAND_NAMES)}
        )
        self.sensitivity = params.add(
            f"{name}_sensitivity", min=0.0, max=5.0, default=1.0,
            group=group, subgroup=subgroup
        )
        self.attack = params.add(
            f"{name}_attack", min=0.0, max=1.0, default=0.3,
            group=group, subgroup=subgroup
        )
        self.decay = params.add(
            f"{name}_decay", min=0.0, max=1.0, default=0.1,
            group=group, subgroup=subgroup
        )
        self.cutoff_min = params.add(
            f"{name}_cutoff_min", min=-100.0, max=100.0, default=-100.0,
            group=group, subgroup=subgroup
        )
        self.cutoff_max = params.add(
            f"{name}_cutoff_max", min=-100.0, max=100.0, default=100.0,
            group=group, subgroup=subgroup
        )

    def link_param(self, param: Param):
        """Link this audio band to a parameter, adjusting cutoff ranges."""
        self.linked_param = param
        self.cutoff_min.max = param.max
        self.cutoff_min.min = param.min
        self.cutoff_min.value = param.min
        self.cutoff_max.max = param.max
        self.cutoff_max.min = param.min
        self.cutoff_max.value = param.max

    def unlink_param(self):
        """Unlink from the current parameter."""
        if self.linked_param:
            self.linked_param.linked_audio_band = None
        self.linked_param = None

    def update(self, band_energies: list):
        """
        Update the linked parameter based on current audio energy.

        Args:
            band_energies: List of 5 float energy values, one per frequency band.
        """
        if self.linked_param is None:
            return

        band_idx = int(self.band_select.value)
        if band_idx < 0 or band_idx >= len(band_energies):
            return

        raw_energy = band_energies[band_idx]

        # Apply sensitivity
        energy = raw_energy * self.sensitivity.value

        # Clamp energy to 0-1 range before envelope follower
        energy = min(1.0, max(0.0, energy))

        # Envelope follower: fast attack, slow decay
        attack = self.attack.value
        decay = self.decay.value
        if energy > self.smoothed_value:
            self.smoothed_value += attack * (energy - self.smoothed_value)
        else:
            self.smoothed_value += decay * (energy - self.smoothed_value)

        # Map smoothed 0-1 value to the linked parameter's range
        param = self.linked_param
        mapped = param.min + self.smoothed_value * (param.max - param.min)

        # Apply cutoff clamping
        mapped = np.clip(mapped, self.cutoff_min.value, self.cutoff_max.value)

        param.value = mapped


class AudioReactiveModule:
    """
    Manages audio capture and FFT analysis for audio-reactive parameter control.
    Analogous to OscBank in lfo.py.

    Uses sounddevice for audio input and provides per-frame FFT band energy
    analysis that AudioBand instances consume to modulate linked parameters.
    """

    def __init__(self, params: ParamTable, sample_rate=44100, block_size=2048):
        self.params = params
        self.bands = []
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.band_energies = [0.0] * NUM_BANDS
        self.available = SOUNDDEVICE_AVAILABLE

        # Double-buffer for thread-safe audio transfer
        self._write_buffer = np.zeros(block_size, dtype=np.float32)
        self._read_buffer = np.zeros(block_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._has_new_data = False

        # Pre-compute FFT helpers
        self._window = np.hanning(block_size).astype(np.float32)
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)

        # Pre-compute bin index ranges for each band
        self._band_slices = []
        for low_hz, high_hz in BAND_RANGES:
            low_bin = int(np.searchsorted(self._freqs, low_hz))
            high_bin = int(np.searchsorted(self._freqs, high_hz))
            self._band_slices.append((low_bin, max(low_bin + 1, high_bin)))

        self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback - runs on audio thread."""
        if status:
            log.debug(f"Audio callback status: {status}")
        # Take mono mix of input
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._buffer_lock:
            samples_to_copy = min(len(mono), self.block_size)
            self._write_buffer[:samples_to_copy] = mono[:samples_to_copy]
            self._has_new_data = True

    def start(self):
        """Start audio capture stream."""
        if not self.available:
            log.warning("Audio reactive module unavailable - sounddevice not installed.")
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
            )
            self._stream.start()
            log.info(f"Audio reactive module started (sample_rate={self.sample_rate}, block_size={self.block_size})")
            return True
        except Exception as e:
            log.error(f"Failed to start audio stream: {e}")
            self.available = False
            return False

    def stop(self):
        """Stop audio capture stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning(f"Error stopping audio stream: {e}")
            self._stream = None
            log.info("Audio reactive module stopped.")

    def analyze(self):
        """
        Perform FFT analysis on the latest audio buffer and update all linked bands.
        Call this once per frame from the video loop.
        """
        if not self.available or self._stream is None:
            return

        # Swap buffers under lock (minimal lock time)
        with self._buffer_lock:
            if not self._has_new_data:
                # No new audio data since last frame, still update bands with old energies
                for band in self.bands:
                    band.update(self.band_energies)
                return
            self._read_buffer, self._write_buffer = self._write_buffer, self._read_buffer
            self._has_new_data = False

        # FFT analysis (outside lock)
        windowed = self._read_buffer * self._window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Normalize spectrum
        spectrum = spectrum / (self.block_size / 2)

        # Compute energy per band
        for i, (lo, hi) in enumerate(self._band_slices):
            if lo < hi and hi <= len(spectrum):
                self.band_energies[i] = float(np.mean(spectrum[lo:hi]))
            else:
                self.band_energies[i] = 0.0

        # Update all linked bands
        for band in self.bands:
            band.update(self.band_energies)

    def add_band(self, name: str, band_index: int = 0):
        """Create and register a new AudioBand."""
        band = AudioBand(self.params, name, band_index)
        self.bands.append(band)
        return band

    def remove_band(self, band: AudioBand):
        """Remove an AudioBand and clean up its parameters."""
        param_names_to_remove = [
            f"{band.name}_band",
            f"{band.name}_sensitivity",
            f"{band.name}_attack",
            f"{band.name}_decay",
            f"{band.name}_cutoff_min",
            f"{band.name}_cutoff_max",
        ]
        for param_name in param_names_to_remove:
            if param_name in self.params.params:
                del self.params.params[param_name]

        self.bands.remove(band)

    def update(self):
        """Alias for analyze() to match OscBank.update() interface."""
        self.analyze()
