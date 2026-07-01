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
Tests for the patch morph / interpolation feature (SaveController.morph_to).

Builds a SaveController with in-memory ParamTables and a temp saved_values.yaml
containing a couple of entries, then exercises the lerp behaviour.
"""
import sys
import time
from pathlib import Path

import yaml
import pytest

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from param import ParamTable
from save import SaveController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def param_tables():
    pt = ParamTable()
    pt.new("brightness", min=0, max=255, default=0)
    pt.new("contrast", min=0.0, max=3.0, default=1.0)
    return {"main": pt}


@pytest.fixture
def controller(param_tables, tmp_path):
    """A SaveController pointed at a temp saved_values.yaml with two entries."""
    entries = [
        {"brightness": 0, "contrast": 1.0},      # index 0 (start)
        {"brightness": 200, "contrast": 3.0},    # index 1 (target)
    ]
    yaml_path = tmp_path / "saved_values.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"entries": entries}, f)

    sc = SaveController(param_tables, yaml_filename="saved_values.yaml")
    # Point the controller at our temp file instead of the repo save/ dir.
    sc.yaml_file_path = str(yaml_path)
    return sc


# ---------------------------------------------------------------------------
# get_patch_entry
# ---------------------------------------------------------------------------

def test_get_patch_entry_valid_index(controller):
    entry = controller.get_patch_entry(1)
    assert entry == {"brightness": 200, "contrast": 3.0}


def test_get_patch_entry_out_of_range(controller):
    assert controller.get_patch_entry(5) is None
    assert controller.get_patch_entry(-1) is None


# ---------------------------------------------------------------------------
# morph_to
# ---------------------------------------------------------------------------

def test_morph_zero_duration_snaps_instantly(controller, param_tables):
    pt = param_tables["main"]
    assert controller.morph_to(1, duration=0) is True
    # No thread; values applied synchronously.
    assert pt.val("brightness") == 200
    assert pt.val("contrast") == pytest.approx(3.0)


def test_morph_reaches_target(controller, param_tables):
    pt = param_tables["main"]
    assert controller.morph_to(1, duration=0.2) is True
    # Wait for the background morph thread to finish.
    controller._morph_thread.join(timeout=5.0)
    assert pt.val("brightness") == 200
    assert pt.val("contrast") == pytest.approx(3.0)


def test_morph_passes_through_intermediate_value(controller, param_tables):
    pt = param_tables["main"]
    # contrast is a float param, so intermediate lerp values are preserved.
    controller.morph_to(1, duration=0.5)
    # Sample partway through: value should be strictly between start and end.
    time.sleep(0.15)
    mid = pt.val("contrast")
    controller._morph_thread.join(timeout=5.0)
    assert 1.0 < mid < 3.0, f"expected an intermediate contrast value, got {mid}"
    assert pt.val("contrast") == pytest.approx(3.0)


def test_morph_invalid_index_returns_false(controller, param_tables):
    pt = param_tables["main"]
    assert controller.morph_to(99, duration=0.1) is False
    # Params untouched.
    assert pt.val("brightness") == 0


def test_new_morph_cancels_previous(controller, param_tables):
    pt = param_tables["main"]
    controller.morph_to(1, duration=5.0)  # slow morph
    first_thread = controller._morph_thread
    # Immediately start a fast morph to the same target; should cancel the first.
    controller.morph_to(1, duration=0.1)
    first_thread.join(timeout=5.0)
    assert not first_thread.is_alive()
    controller._morph_thread.join(timeout=5.0)
    assert pt.val("brightness") == 200
