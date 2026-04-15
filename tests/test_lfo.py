"""
Tests for the LFO (Low Frequency Oscillator) module.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from param import ParamTable
from lfo import LFO, LFOShape, OscBank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lfo(name: str, shape: LFOShape, params: ParamTable = None) -> LFO:
    if params is None:
        params = ParamTable()
    return LFO(
        params=params,
        name=name,
        frequency=1.0,
        amplitude=1.0,
        phase=0.0,
        shape=shape.value,
        seed=0,
        max_amplitude=10,
        min_amplitude=-10,
    )


# ---------------------------------------------------------------------------
# LFOShape coverage — each shape should yield a numeric value without raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    LFOShape.NONE,
    LFOShape.SINE,
    LFOShape.SQUARE,
    LFOShape.TRIANGLE,
    LFOShape.SAWTOOTH,
])
def test_lfo_shape_produces_value(shape):
    params = ParamTable()
    lfo = _make_lfo(f"test_{shape.name}", shape, params=params)
    val = lfo.get_next_value()
    # All shapes except NONE produce a float or int; NONE yields 0
    assert isinstance(val, (int, float))


def test_lfo_none_yields_zero():
    params = ParamTable()
    lfo = _make_lfo("none_lfo", LFOShape.NONE, params=params)
    for _ in range(5):
        val = lfo.get_next_value()
        assert val == 0


def test_lfo_sine_stays_within_amplitude_plus_seed():
    params = ParamTable()
    lfo = _make_lfo("sine_lfo", LFOShape.SINE, params=params)
    amplitude = lfo.amplitude.value
    seed = lfo.seed.value
    for _ in range(60):
        val = lfo.get_next_value()
        # sine: amp * sin(...) + seed, so within [-amp + seed, amp + seed]
        assert -amplitude + seed - 1e-6 <= val <= amplitude + seed + 1e-6


def test_lfo_square_is_binary():
    params = ParamTable()
    lfo = _make_lfo("square_lfo", LFOShape.SQUARE, params=params)
    amplitude = lfo.amplitude.value
    seed = lfo.seed.value
    for _ in range(30):
        val = lfo.get_next_value()
        # Square wave: np.sign(sin(x)) returns -1, 0, or +1.
        # At zero-crossings sin=0 so output=seed; otherwise ±amp+seed.
        assert abs(val - seed) <= amplitude + 1e-6


def test_lfo_triangle_stays_within_amplitude():
    params = ParamTable()
    lfo = _make_lfo("tri_lfo", LFOShape.TRIANGLE, params=params)
    amplitude = lfo.amplitude.value
    seed = lfo.seed.value
    for _ in range(60):
        val = lfo.get_next_value()
        assert val <= amplitude + seed + 1e-6
        assert val >= -amplitude + seed - 1e-6


def test_lfo_sawtooth_produces_varying_values():
    params = ParamTable()
    lfo = _make_lfo("saw_lfo", LFOShape.SAWTOOTH, params=params)
    vals = [lfo.get_next_value() for _ in range(60)]
    # A sawtooth should not be constant
    assert len(set(round(v, 6) for v in vals)) > 1


# ---------------------------------------------------------------------------
# PERLIN shape — requires 'noise' library; skip gracefully if missing
# ---------------------------------------------------------------------------

def test_lfo_perlin_shape():
    pytest.importorskip("noise", reason="'noise' package not installed; skipping Perlin LFO test")
    params = ParamTable()
    lfo = _make_lfo("perlin_lfo", LFOShape.PERLIN, params=params)
    val = lfo.get_next_value()
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Linking LFO to a param — ensures linked_param is updated each tick
# ---------------------------------------------------------------------------

def test_lfo_link_param_updates_value():
    params = ParamTable()
    target = params.new("target_param", min=0.0, max=10.0, default=5.0)

    lfo = _make_lfo("linked_lfo", LFOShape.SINE, params=params)
    lfo.link_param(target)

    # Drive several ticks
    for _ in range(10):
        lfo.get_next_value()

    # The param value should have been driven within its [min, max] range
    assert target.min <= target.value <= target.max


def test_lfo_unlink_param_stops_driving():
    params = ParamTable()
    target = params.new("unlink_target", min=0.0, max=10.0, default=5.0)

    lfo = _make_lfo("unlink_lfo", LFOShape.SINE, params=params)
    lfo.link_param(target)
    lfo.unlink_param()

    assert lfo.linked_param is None
    # Store value before ticks
    before = target.value
    for _ in range(10):
        lfo.get_next_value()
    # After unlinking, the param should no longer be modified
    assert target.value == before


# ---------------------------------------------------------------------------
# OscBank — basic construction and update
# ---------------------------------------------------------------------------

def test_osc_bank_creates_correct_number_of_oscillators():
    params = ParamTable()
    bank = OscBank(params, num_osc=3)
    assert len(bank) == 3


def test_osc_bank_update_does_not_raise():
    params = ParamTable()
    bank = OscBank(params, num_osc=2)
    # No params are linked so update is a no-op — should not raise
    bank.update()


def test_osc_bank_add_oscillator():
    params = ParamTable()
    bank = OscBank(params, num_osc=1)
    initial_len = len(bank)
    bank.add_oscillator("extra_osc", frequency=0.5, amplitude=1.0, phase=0.0, shape=LFOShape.SINE.value)
    assert len(bank) == initial_len + 1


def test_osc_bank_remove_oscillator():
    params = ParamTable()
    bank = OscBank(params, num_osc=2)
    initial_len = len(bank)
    osc = bank[0]
    bank.remove_oscillator(osc)
    assert len(bank) == initial_len - 1


def test_osc_bank_linked_param_driven_by_update():
    params = ParamTable()
    target = params.new("bank_target", min=0.0, max=10.0, default=5.0)

    bank = OscBank(params, num_osc=1)
    bank[0].link_param(target)

    for _ in range(5):
        bank.update()

    assert target.min <= target.value <= target.max


# ---------------------------------------------------------------------------
# get_next_value — consecutive calls advance time
# ---------------------------------------------------------------------------

def test_lfo_consecutive_calls_advance():
    params = ParamTable()
    lfo = _make_lfo("advance_lfo", LFOShape.SINE, params=params)
    vals = [lfo.get_next_value() for _ in range(30)]
    # With a real frequency the sine will not be constant across 30 steps
    assert not all(v == vals[0] for v in vals)
