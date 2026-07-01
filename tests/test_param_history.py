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
Tests for ParamHistory undo/redo ring buffer.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from param import ParamTable
from param_history import ParamHistory


def _make_table():
    """A small ParamTable with a couple of numeric params."""
    table = ParamTable(group="Test")
    table.new("alpha", min=0, max=100, default=10)
    table.new("beta", min=0.0, max=1.0, default=0.5)
    return table


# ---------------------------------------------------------------------------
# undo restores the previous state
# ---------------------------------------------------------------------------

def test_undo_restores_previous_values():
    table = _make_table()
    hist = ParamHistory(table)

    # Snapshot the starting state, then change a value.
    hist.snapshot()
    table.set("alpha", 42)
    assert table.val("alpha") == 42

    assert hist.undo() is True
    assert table.val("alpha") == 10  # back to the snapshotted value


def test_undo_returns_false_when_empty():
    table = _make_table()
    hist = ParamHistory(table)
    assert hist.undo() is False


# ---------------------------------------------------------------------------
# redo re-applies an undone state
# ---------------------------------------------------------------------------

def test_redo_reapplies_value():
    table = _make_table()
    hist = ParamHistory(table)

    hist.snapshot()
    table.set("alpha", 42)

    hist.undo()
    assert table.val("alpha") == 10

    assert hist.redo() is True
    assert table.val("alpha") == 42


def test_redo_returns_false_when_empty():
    table = _make_table()
    hist = ParamHistory(table)
    hist.snapshot()
    assert hist.redo() is False


# ---------------------------------------------------------------------------
# A new snapshot clears the redo stack
# ---------------------------------------------------------------------------

def test_new_snapshot_clears_redo():
    table = _make_table()
    hist = ParamHistory(table)

    hist.snapshot()
    table.set("alpha", 42)
    hist.undo()
    assert hist.counts()["redo"] == 1

    # A fresh snapshot should drop the redo stack.
    hist.snapshot()
    assert hist.counts()["redo"] == 0
    assert hist.redo() is False


# ---------------------------------------------------------------------------
# Multi-step undo/redo across several params
# ---------------------------------------------------------------------------

def test_multi_step_undo_redo():
    table = _make_table()
    hist = ParamHistory(table)

    hist.snapshot()               # state: alpha=10, beta=0.5
    table.set("alpha", 30)
    hist.snapshot()               # state: alpha=30, beta=0.5
    table.set("beta", 0.9)

    assert table.val("alpha") == 30
    assert table.val("beta") == pytest.approx(0.9)

    hist.undo()                   # -> alpha=30, beta=0.5
    assert table.val("beta") == pytest.approx(0.5)
    assert table.val("alpha") == 30

    hist.undo()                   # -> alpha=10, beta=0.5
    assert table.val("alpha") == 10

    hist.redo()                   # -> alpha=30
    assert table.val("alpha") == 30


# ---------------------------------------------------------------------------
# Ring buffer respects capacity N
# ---------------------------------------------------------------------------

def test_ring_buffer_respects_capacity():
    table = _make_table()
    hist = ParamHistory(table, capacity=3)

    # Push more snapshots than capacity; each must differ to be recorded.
    for i in range(10):
        table.set("alpha", i)
        hist.snapshot()

    assert hist.counts()["undo"] == 3

    # Only the 3 most recent states remain undoable.
    table.set("alpha", 99)
    assert hist.undo() is True
    assert hist.undo() is True
    assert hist.undo() is True
    assert hist.undo() is False


# ---------------------------------------------------------------------------
# Redundant no-op snapshots are skipped
# ---------------------------------------------------------------------------

def test_duplicate_snapshot_is_skipped():
    table = _make_table()
    hist = ParamHistory(table)

    hist.snapshot()
    hist.snapshot()  # identical state — should be a no-op
    assert hist.counts()["undo"] == 1
