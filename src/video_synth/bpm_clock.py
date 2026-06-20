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
Global BPM Clock — MIDI clock input and tap-tempo with LFO quantisation.

The BPMClock class:
- Listens for MIDI clock messages (24 ticks per quarter note) from any port
  whose name matches a configurable substring.
- Maintains a running BPM estimate using a ring buffer of inter-tick intervals.
- Exposes a monotonic beat_phase (0.0–1.0) that LFOs in tempo-sync mode use
  instead of their own time counter, so they lock perfectly to the beat.
- Supports tap-tempo via record_tap() — averages the last 4 taps.
- Can be driven externally (no MIDI device) by setting bpm directly.

Integration with LFO
--------------------
After creating BPMClock, pass it to OscBank or individual LFOs.  When an
LFO's ``tempo_sync`` param is non-zero, its oscillator uses:

    effective_frequency = bpm_clock.bpm / 60.0 * note_division

where ``note_division`` maps to standard musical values:
    0 = free (normal Hz-based, no sync)
    1 = 1/1   (whole note)
    2 = 1/2   (half note)
    3 = 1/4   (quarter note)
    4 = 1/8   (eighth note)
    5 = 1/16  (sixteenth note)
    6 = 1/32  (thirty-second note)

Usage
-----
    clock = BPMClock()
    clock.start(port_name="Arturia")   # auto-detect port containing "Arturia"
    # or:
    clock.bpm = 120.0                  # manual BPM without MIDI

    lfo = LFO(..., bpm_clock=clock)
"""

import threading
import time
import logging
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

# MIDI clock sends 24 ticks per quarter note
_TICKS_PER_BEAT = 24

# Note division name → multiplier relative to quarter note (beats per cycle)
NOTE_DIVISIONS = {
    "FREE":  0.0,
    "1/1":   4.0,   # whole note = 4 beats
    "1/2":   2.0,
    "1/4":   1.0,   # quarter note = 1 beat
    "1/8":   0.5,
    "1/16":  0.25,
    "1/32":  0.125,
}

NOTE_DIVISION_NAMES = list(NOTE_DIVISIONS.keys())


class BPMClock:
    """
    Singleton-style global BPM clock.

    Thread-safe: all public attributes and methods are safe to call from any
    thread.  The MIDI listener runs on its own daemon thread.
    """

    def __init__(self, default_bpm: float = 120.0):
        self._lock = threading.Lock()

        # BPM state
        self._bpm: float = default_bpm
        self._tick_times: deque = deque(maxlen=_TICKS_PER_BEAT * 2)  # 2 beats of history
        self._last_tick_time: Optional[float] = None

        # Beat phase: monotonically increasing fraction of a beat (0.0–1.0)
        # Derived from wall time using current BPM.
        self._phase_ref_time: float = time.monotonic()
        self._phase_ref_beat: float = 0.0

        # Tap tempo
        self._taps: deque = deque(maxlen=8)

        # MIDI thread
        self._midi_thread: Optional[threading.Thread] = None
        self._midi_port = None
        self._running = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm

    @bpm.setter
    def bpm(self, value: float):
        value = max(20.0, min(300.0, float(value)))
        with self._lock:
            self._bpm = value
            # Reset phase reference so the phase is continuous
            self._phase_ref_time = time.monotonic()
            self._phase_ref_beat = self._beat_phase_unsafe()
        log.info("BPMClock: BPM set to %.1f", value)

    @property
    def beat_phase(self) -> float:
        """Current beat phase in [0.0, 1.0).  Updates at the current BPM."""
        with self._lock:
            return self._beat_phase_unsafe() % 1.0

    @property
    def beat_count(self) -> float:
        """Total beats elapsed since last BPM reference reset."""
        with self._lock:
            return self._beat_phase_unsafe()

    def _beat_phase_unsafe(self) -> float:
        """Must be called with self._lock held."""
        elapsed = time.monotonic() - self._phase_ref_time
        return self._phase_ref_beat + elapsed * (self._bpm / 60.0)

    def frequency_for_division(self, division_name: str) -> float:
        """
        Return the LFO frequency (Hz) for a note division at the current BPM.

        Args:
            division_name: One of NOTE_DIVISION_NAMES ("FREE", "1/4", etc.)
        Returns:
            Frequency in Hz, or 0.0 for FREE.
        """
        mult = NOTE_DIVISIONS.get(division_name, 0.0)
        if mult == 0.0:
            return 0.0
        with self._lock:
            beats_per_cycle = mult  # e.g. 0.5 beats per cycle for 1/8 note
            # freq = beats_per_cycle * (bpm / 60) — but division is beats per cycle,
            # so freq = (bpm/60) / beats_per_cycle
            # Wait — 1/4 note = 1 beat → freq = bpm/60 Hz (one cycle per beat) ✓
            # 1/2 note = 2 beats → freq = bpm/120 Hz ✓
            # 1/8 note = 0.5 beats → freq = bpm/30 Hz ✓
            return (self._bpm / 60.0) / mult

    # ------------------------------------------------------------------
    # Tap tempo
    # ------------------------------------------------------------------

    def record_tap(self):
        """Record a tap.  After ≥2 taps, BPM is updated from the average interval."""
        now = time.monotonic()
        self._taps.append(now)
        if len(self._taps) >= 2:
            intervals = [self._taps[i] - self._taps[i - 1]
                         for i in range(1, len(self._taps))]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                self.bpm = 60.0 / avg_interval

    def reset_taps(self):
        self._taps.clear()

    # ------------------------------------------------------------------
    # MIDI clock input
    # ------------------------------------------------------------------

    def start(self, port_name: Optional[str] = None):
        """
        Start listening for MIDI clock on the first port matching *port_name*.

        Args:
            port_name: Substring to match against available port names.
                       None = use the first available input port.
        """
        if self._running:
            log.warning("BPMClock already running.")
            return

        try:
            import mido
        except ImportError:
            log.warning("BPMClock: mido not installed — MIDI clock input disabled.")
            return

        available = mido.get_input_names()
        if not available:
            log.info("BPMClock: no MIDI input ports found.")
            return

        chosen = None
        if port_name is None:
            chosen = available[0]
        else:
            for name in available:
                if port_name.lower() in name.lower():
                    chosen = name
                    break

        if chosen is None:
            log.info("BPMClock: no MIDI port matching '%s' — clock disabled.", port_name)
            return

        self._running = True
        self._midi_thread = threading.Thread(
            target=self._midi_loop, args=(chosen,), daemon=True
        )
        self._midi_thread.start()
        log.info("BPMClock: listening for MIDI clock on '%s'", chosen)

    def stop(self):
        self._running = False
        if self._midi_port is not None:
            try:
                self._midi_port.close()
            except Exception:
                pass
            self._midi_port = None
        log.info("BPMClock: stopped.")

    def _midi_loop(self, port_name: str):
        try:
            import mido
            with mido.open_input(port_name) as port:
                self._midi_port = port
                for msg in port:
                    if not self._running:
                        break
                    self._handle_midi(msg)
        except Exception as exc:
            log.error("BPMClock MIDI thread error: %s", exc)
        finally:
            self._running = False

    def _handle_midi(self, msg):
        """Process a single mido Message."""
        if msg.type == "clock":
            now = time.monotonic()
            with self._lock:
                self._tick_times.append(now)
                if len(self._tick_times) >= 2:
                    # Use the last N ticks to estimate BPM
                    n = min(len(self._tick_times), _TICKS_PER_BEAT)
                    recent = list(self._tick_times)[-n:]
                    interval = (recent[-1] - recent[0]) / (len(recent) - 1)
                    if interval > 0:
                        new_bpm = 60.0 / (interval * _TICKS_PER_BEAT)
                        # Clamp to reasonable range
                        if 20.0 <= new_bpm <= 300.0:
                            self._bpm = new_bpm
                            # Sync phase reference on every beat boundary
                            # (every 24 ticks)
                            if len(self._tick_times) % _TICKS_PER_BEAT == 0:
                                self._phase_ref_time = now
                                self._phase_ref_beat = round(self._beat_phase_unsafe())

        elif msg.type == "start":
            with self._lock:
                self._tick_times.clear()
                self._phase_ref_time = time.monotonic()
                self._phase_ref_beat = 0.0
            log.debug("BPMClock: MIDI Start received — phase reset.")

        elif msg.type == "stop":
            log.debug("BPMClock: MIDI Stop received.")

        elif msg.type == "songpos":
            # Sync beat counter to song position pointer
            # Song position is in MIDI beats (6 clock ticks each)
            midi_beats = msg.pos
            quarter_notes = midi_beats / 4.0
            with self._lock:
                self._phase_ref_time = time.monotonic()
                self._phase_ref_beat = quarter_notes
            log.debug("BPMClock: Song Position → %.1f beats", quarter_notes)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a serialisable status snapshot for the API."""
        with self._lock:
            phase = self._beat_phase_unsafe() % 1.0
            return {
                "bpm": round(self._bpm, 2),
                "beat_phase": round(phase, 4),
                "beat_count": round(self._beat_phase_unsafe(), 2),
                "midi_active": self._running,
                "note_divisions": NOTE_DIVISION_NAMES,
            }
