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
Tests for MIDI mapping export / import (share midi_mappings.yaml via UI).

Exercises both the MidiMapper reload/import methods directly and the
FastAPI /midi/export and /midi/import endpoints via TestClient. No real
MIDI hardware or ports are opened.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# api.py -> lfo -> noise is fine, but guard moderngl just like test_api.py.
sys.modules.setdefault("moderngl", MagicMock())

import yaml
import pytest

from param import ParamTable
from midi_mapper import MidiMapper, MidiMapperController
from api import APIServer

pytest.importorskip("httpx", reason="httpx required for TestClient; install with pip install httpx")
from fastapi.testclient import TestClient


PORT_NAME = "Test Controller 1"
SAMPLE_YAML = f"""ports:
  {PORT_NAME}:
    mappings:
      1: Src 1 Effects/brightness
      7: Mixer/contrast
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def param_tables():
    src1 = ParamTable(group="Src 1 Effects")
    src1.new("brightness", min=0, max=255, default=128)
    mixer = ParamTable(group="Mixer")
    mixer.new("contrast", min=0.5, max=3.0, default=1.0)
    # MidiMapper expects {"group_label": (ParamTable, group_filter|None)}
    return {
        "Src 1 Effects": (src1, None),
        "Mixer": (mixer, None),
    }


@pytest.fixture
def mapper(param_tables, tmp_path):
    m = MidiMapper(param_tables)
    # Redirect persistence into a temp dir so the real save/ is untouched.
    m._save_dir = tmp_path
    m._yaml_path = tmp_path / "midi_mappings.yaml"
    # Register a controller for a fake port so reload can apply live.
    m.controllers[PORT_NAME] = MidiMapperController(
        port_name=PORT_NAME, param_tables=param_tables, mappings={}
    )
    return m


# ---------------------------------------------------------------------------
# MidiMapper.import_mappings / reload_mappings (direct)
# ---------------------------------------------------------------------------

def test_import_mappings_applies_live(mapper):
    loaded = mapper.import_mappings(SAMPLE_YAML)

    assert PORT_NAME in loaded
    assert loaded[PORT_NAME] == {1: "Src 1 Effects/brightness", 7: "Mixer/contrast"}

    # The running controller now reflects the imported mappings.
    controller = mapper.controllers[PORT_NAME]
    assert controller.mappings == {1: "Src 1 Effects/brightness", 7: "Mixer/contrast"}

    # get_mappings() reflects the imported state.
    assert mapper.get_mappings()[PORT_NAME][1] == "Src 1 Effects/brightness"


def test_import_mappings_writes_yaml(mapper):
    mapper.import_mappings(SAMPLE_YAML)
    assert mapper._yaml_path.exists()
    data = yaml.safe_load(mapper._yaml_path.read_text())
    assert data["ports"][PORT_NAME]["mappings"][1] == "Src 1 Effects/brightness"


def test_import_mappings_malformed_yaml_raises(mapper):
    with pytest.raises(ValueError):
        mapper.import_mappings("ports: [unterminated")


def test_import_mappings_wrong_structure_raises(mapper):
    # 'ports' must be a mapping, not a list.
    with pytest.raises(ValueError):
        mapper.import_mappings("ports:\n  - just_a_list_item\n")


def test_reload_mappings_picks_up_disk_changes(mapper):
    mapper._yaml_path.write_text(SAMPLE_YAML)
    loaded = mapper.reload_mappings()
    assert loaded[PORT_NAME] == {1: "Src 1 Effects/brightness", 7: "Mixer/contrast"}
    assert mapper.controllers[PORT_NAME].mappings[7] == "Mixer/contrast"


def test_export_mappings_yaml_roundtrip(mapper):
    mapper.import_mappings(SAMPLE_YAML)
    text = mapper.export_mappings_yaml()
    data = yaml.safe_load(text)
    assert data["ports"][PORT_NAME]["mappings"][1] == "Src 1 Effects/brightness"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(param_tables, tmp_path):
    params = ParamTable()
    params.new("brightness", min=0, max=255, default=128)

    m = MidiMapper(param_tables)
    m._save_dir = tmp_path
    m._yaml_path = tmp_path / "midi_mappings.yaml"
    m.controllers[PORT_NAME] = MidiMapperController(
        port_name=PORT_NAME, param_tables=param_tables, mappings={}
    )

    api = APIServer(
        params=params,
        mixer=None,
        save_controller=None,
        midi_mapper=m,
        host="127.0.0.1",
        port=8010,
    )
    api._mapper = m  # keep a handle for assertions
    return TestClient(api.app)


def test_import_endpoint_applies_and_export_returns_it(client):
    r = client.post("/midi/import", json={"mappings": SAMPLE_YAML})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["ports"] == 1
    assert body["mappings"] == 2

    # Export should now return the imported mappings as YAML.
    r2 = client.get("/midi/export")
    assert r2.status_code == 200
    assert "attachment" in r2.headers.get("content-disposition", "")
    data = yaml.safe_load(r2.text)
    assert data["ports"][PORT_NAME]["mappings"][1] == "Src 1 Effects/brightness"


def test_import_endpoint_malformed_returns_400(client):
    r = client.post("/midi/import", json={"mappings": "ports: [unterminated"})
    assert r.status_code == 400


def test_export_endpoint_503_when_no_mapper(param_tables):
    params = ParamTable()
    params.new("brightness", min=0, max=255, default=128)
    api = APIServer(
        params=params,
        mixer=None,
        save_controller=None,
        midi_mapper=None,
        host="127.0.0.1",
        port=8011,
    )
    c = TestClient(api.app)
    assert c.get("/midi/export").status_code == 503
    assert c.post("/midi/import", json={"mappings": "ports: {}"}).status_code == 503
