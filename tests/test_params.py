"""
Tests for ParamTable and Param classes.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from param import ParamTable, Param
from common import Groups


# ---------------------------------------------------------------------------
# ParamTable.new() — basic creation
# ---------------------------------------------------------------------------

def test_new_creates_param_with_correct_min_max_default():
    table = ParamTable()
    p = table.new("my_param", min=0, max=100, default=42)
    assert p.min == 0
    assert p.max == 100
    assert p.default == 42
    assert p.value == 42


def test_new_int_default_produces_int_value():
    table = ParamTable()
    p = table.new("int_param", min=0, max=10, default=5)
    assert isinstance(p.value, int)
    assert p.value == 5


def test_new_float_default_produces_float_value():
    table = ParamTable()
    p = table.new("float_param", min=0.0, max=1.0, default=0.5)
    assert isinstance(p.value, float)
    assert p.value == pytest.approx(0.5)


def test_new_raises_on_duplicate_name():
    table = ParamTable()
    table.new("dup", min=0, max=1, default=0)
    with pytest.raises(ValueError, match="already exists"):
        table.new("dup", min=0, max=1, default=0)


# ---------------------------------------------------------------------------
# Two params with same name in different tables don't collide
# ---------------------------------------------------------------------------

def test_same_name_different_tables():
    t1 = ParamTable(group="GroupA")
    t2 = ParamTable(group="GroupB")
    p1 = t1.new("shared_name", min=0, max=10, default=1)
    p2 = t2.new("shared_name", min=0, max=20, default=15)
    assert p1.value == 1
    assert p2.value == 15


# ---------------------------------------------------------------------------
# Value clamping — out-of-range assignments are silently clamped (not raised)
# ---------------------------------------------------------------------------

def test_value_above_max_is_clamped():
    table = ParamTable()
    p = table.new("clamp_test", min=0, max=10, default=5)
    p.value = 999
    assert p.value == 10


def test_value_below_min_is_clamped():
    table = ParamTable()
    p = table.new("clamp_low", min=5, max=20, default=10)
    p.value = -100
    assert p.value == 5


def test_value_at_boundary_is_accepted():
    table = ParamTable()
    p = table.new("boundary", min=0.0, max=1.0, default=0.5)
    p.value = 0.0
    assert p.value == pytest.approx(0.0)
    p.value = 1.0
    assert p.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# set() — convenience method on ParamTable
# ---------------------------------------------------------------------------

def test_table_set_updates_value():
    table = ParamTable()
    table.new("settable", min=0, max=100, default=0)
    table.set("settable", 77)
    assert table.val("settable") == 77


def test_table_set_returns_clamped_value():
    table = ParamTable()
    table.new("clamped_set", min=0, max=10, default=5)
    result = table.set("clamped_set", 999)
    assert result == 10


def test_table_set_raises_for_unknown_param():
    table = ParamTable()
    with pytest.raises(ValueError):
        table.set("nonexistent", 5)


# ---------------------------------------------------------------------------
# val() / get() / contains / indexing
# ---------------------------------------------------------------------------

def test_val_returns_current_value():
    table = ParamTable()
    table.new("v", min=0, max=1, default=0)
    table.set("v", 1)
    assert table.val("v") == 1


def test_get_returns_param_object():
    table = ParamTable()
    p = table.new("g", min=0, max=1, default=0)
    assert table.get("g") is p


def test_contains_operator():
    table = ParamTable()
    table.new("exists", min=0, max=1, default=0)
    assert "exists" in table
    assert "missing" not in table


def test_getitem_by_string():
    table = ParamTable()
    p = table.new("key", min=0, max=1, default=0)
    assert table["key"] is p


def test_getitem_by_index():
    table = ParamTable()
    p0 = table.new("first", min=0, max=1, default=0)
    p1 = table.new("second", min=0, max=1, default=1)
    assert table[0] is p0
    assert table[1] is p1


def test_getitem_raises_on_missing_key():
    table = ParamTable()
    with pytest.raises(KeyError):
        _ = table["no_such_param"]


# ---------------------------------------------------------------------------
# Iteration via .items() / .values() / .keys()
# ---------------------------------------------------------------------------

def test_param_table_can_be_iterated_via_items():
    table = ParamTable()
    table.new("a", min=0, max=1, default=0)
    table.new("b", min=0, max=1, default=1)
    names = [name for name, _ in table.items()]
    assert "a" in names
    assert "b" in names


def test_param_table_keys():
    table = ParamTable()
    table.new("x", min=0, max=1, default=0)
    assert "x" in table.keys()


def test_param_table_values_are_param_objects():
    table = ParamTable()
    table.new("val_test", min=0, max=1, default=0)
    for v in table.values():
        assert isinstance(v, Param)


# ---------------------------------------------------------------------------
# Param arithmetic dunder methods
# ---------------------------------------------------------------------------

def test_param_add_number():
    table = ParamTable()
    p = table.new("add", min=0, max=100, default=10)
    assert p + 5 == 15


def test_param_sub_number():
    table = ParamTable()
    p = table.new("sub", min=0, max=100, default=10)
    assert p - 3 == 7


def test_param_mul_number():
    table = ParamTable()
    p = table.new("mul", min=0, max=100, default=4)
    assert p * 3 == 12


def test_param_div_number():
    table = ParamTable()
    p = table.new("div", min=0.0, max=100.0, default=10.0)
    assert p / 2 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# reset() and randomize()
# ---------------------------------------------------------------------------

def test_param_reset_returns_to_default():
    table = ParamTable()
    p = table.new("reset_me", min=0, max=100, default=50)
    p.value = 99
    p.reset()
    assert p.value == 50


def test_param_randomize_stays_in_range():
    table = ParamTable()
    p = table.new("rand", min=10, max=20, default=15)
    for _ in range(20):
        p.randomize()
        assert p.min <= p.value <= p.max
