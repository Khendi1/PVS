"""
Tests for the FastAPI server (api.py).

Heavy dependencies (Mixer, SaveController, AudioReactiveModule, uvicorn) are
mocked so no real hardware is needed.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "video_synth"))

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch moderngl before any animation imports touch it
# (api.py imports lfo which imports noise; those are fine.
#  But if api.py or its imports pull in animations/shaders the ctx creation
#  would fail on headless CI — guard just in case.)
# ---------------------------------------------------------------------------
sys.modules.setdefault("moderngl", MagicMock())

from param import ParamTable
from api import APIServer

# httpx is required by fastapi's TestClient
pytest.importorskip("httpx", reason="httpx required for TestClient; install with pip install httpx")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    pt = ParamTable()
    pt.new("brightness", min=0, max=255, default=128)
    pt.new("contrast", min=0.5, max=3.0, default=1.0)
    return pt


@pytest.fixture
def mock_mixer(params):
    mixer = MagicMock()
    # Provide a real numpy frame so /snapshot can encode it
    mixer.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return mixer


@pytest.fixture
def mock_save_controller():
    sc = MagicMock()
    sc.load_next_patch = MagicMock()
    sc.load_prev_patch = MagicMock()
    sc.load_random_patch = MagicMock()
    sc.save_patch = MagicMock()
    return sc


@pytest.fixture
def server(params, mock_mixer, mock_save_controller):
    api = APIServer(
        params=params,
        mixer=mock_mixer,
        save_controller=mock_save_controller,
        host="127.0.0.1",
        port=8001,
    )
    return api


@pytest.fixture
def client(server):
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

def test_get_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Video Synth API"


def test_get_root_lists_endpoints(client):
    response = client.get("/")
    data = response.json()
    assert "endpoints" in data
    endpoints = data["endpoints"]
    assert "GET /params" in endpoints
    assert "PUT /params/{name}" in endpoints


# ---------------------------------------------------------------------------
# GET /params
# ---------------------------------------------------------------------------

def test_get_params_returns_200(client):
    response = client.get("/params")
    assert response.status_code == 200


def test_get_params_returns_list(client, params):
    response = client.get("/params")
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(params.params)


def test_get_params_contain_expected_fields(client):
    response = client.get("/params")
    data = response.json()
    assert len(data) > 0
    first = data[0]
    for field in ("name", "value", "min", "max", "default", "group", "subgroup", "type"):
        assert field in first, f"Missing field '{field}' in param response"


# ---------------------------------------------------------------------------
# GET /params/{name}
# ---------------------------------------------------------------------------

def test_get_param_by_name(client):
    response = client.get("/params/brightness")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "brightness"
    assert data["min"] == 0
    assert data["max"] == 255
    assert data["default"] == 128


def test_get_param_not_found(client):
    response = client.get("/params/no_such_param")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# PUT /params/{name}
# ---------------------------------------------------------------------------

def test_put_param_valid_value(client, params):
    response = client.put("/params/brightness", json={"value": 200})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["value"] == 200
    # Verify the actual param was updated
    assert params.val("brightness") == 200


def test_put_param_at_boundary_min(client, params):
    response = client.put("/params/brightness", json={"value": 0})
    assert response.status_code == 200
    assert response.json()["value"] == 0


def test_put_param_at_boundary_max(client, params):
    response = client.put("/params/brightness", json={"value": 255})
    assert response.status_code == 200
    assert response.json()["value"] == 255


def test_put_param_value_above_max_returns_400(client):
    response = client.put("/params/brightness", json={"value": 999})
    assert response.status_code == 400


def test_put_param_value_below_min_returns_400(client):
    response = client.put("/params/brightness", json={"value": -1})
    assert response.status_code == 400


def test_put_param_not_found_returns_404(client):
    response = client.put("/params/nonexistent", json={"value": 10})
    assert response.status_code == 404


def test_put_float_param_valid(client, params):
    response = client.put("/params/contrast", json={"value": 2.0})
    assert response.status_code == 200
    assert abs(response.json()["value"] - 2.0) < 1e-6


def test_put_float_param_out_of_range(client):
    response = client.put("/params/contrast", json={"value": 10.0})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /params/reset/{name}
# ---------------------------------------------------------------------------

def test_reset_param_returns_200(client, params):
    # First set it to a non-default value
    params.set("brightness", 200)
    response = client.post("/params/reset/brightness")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    # Should have been reset to default (128)
    assert body["value"] == 128


def test_reset_param_not_found_returns_404(client):
    response = client.post("/params/reset/no_param")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /snapshot
# ---------------------------------------------------------------------------

def test_get_snapshot_returns_200(client):
    response = client.get("/snapshot")
    assert response.status_code == 200


def test_get_snapshot_content_type_is_jpeg(client):
    response = client.get("/snapshot")
    assert response.status_code == 200
    assert "image/jpeg" in response.headers.get("content-type", "")


def test_get_snapshot_returns_503_when_no_frame(params, mock_save_controller):
    api = APIServer(
        params=params,
        mixer=None,   # no mixer → no frame
        save_controller=mock_save_controller,
        host="127.0.0.1",
        port=8002,
    )
    c = TestClient(api.app)
    response = c.get("/snapshot")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Patch endpoints
# ---------------------------------------------------------------------------

def test_patch_next_returns_200(client, mock_save_controller):
    response = client.post("/patch/next")
    assert response.status_code == 200
    mock_save_controller.load_next_patch.assert_called_once()


def test_patch_random_returns_200(client, mock_save_controller):
    response = client.post("/patch/random")
    assert response.status_code == 200
    mock_save_controller.load_random_patch.assert_called_once()


def test_patch_prev_returns_200(client, mock_save_controller):
    response = client.post("/patch/prev")
    assert response.status_code == 200
    mock_save_controller.load_prev_patch.assert_called_once()


def test_patch_save_returns_200(client, mock_save_controller):
    response = client.post("/patch/save")
    assert response.status_code == 200
    mock_save_controller.save_patch.assert_called_once()


def test_patch_next_returns_503_when_no_save_controller(params):
    api = APIServer(
        params=params,
        mixer=None,
        save_controller=None,
        host="127.0.0.1",
        port=8003,
    )
    c = TestClient(api.app)
    response = c.post("/patch/next")
    assert response.status_code == 503


def test_patch_random_returns_503_when_no_save_controller(params):
    api = APIServer(
        params=params,
        mixer=None,
        save_controller=None,
        host="127.0.0.1",
        port=8004,
    )
    c = TestClient(api.app)
    response = c.post("/patch/random")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# LFO endpoints — basic list when empty
# ---------------------------------------------------------------------------

def test_get_lfo_list_empty(client):
    response = client.get("/lfo")
    assert response.status_code == 200
    assert response.json() == []


def test_get_lfo_by_name_not_found(client):
    response = client.get("/lfo/brightness")
    assert response.status_code == 404
