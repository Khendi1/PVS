"""
Smoke tests for every Animation subclass.

Each test instantiates the class with (params, width=640, height=480, group=group)
and calls get_frame(dummy_frame) once, asserting a valid numpy array is returned.

Animations that require GPU (moderngl) context are skipped with an explicit reason.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
for _p in [str(_SRC), str(_SRC / "video_synth")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest
from param import ParamTable
from common import Groups

WIDTH = 640
HEIGHT = 480


class _Group:
    """Minimal group stub that satisfies .name access in all animation __init__ methods."""
    def __init__(self, name: str = "SRC_1_EFFECTS"):
        self.name = name


@pytest.fixture
def params():
    return ParamTable(group="Test")


@pytest.fixture
def group():
    return _Group("SRC_1_EFFECTS")


@pytest.fixture
def dummy_frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helper to assert a valid frame was returned
# ---------------------------------------------------------------------------

def _assert_valid_frame(result):
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert result.ndim in (2, 3), f"Expected 2D or 3D array, got ndim={result.ndim}"
    if result.ndim == 3:
        assert result.shape[0] == HEIGHT
        assert result.shape[1] == WIDTH
    else:
        assert result.shape[0] == HEIGHT
        assert result.shape[1] == WIDTH


# ---------------------------------------------------------------------------
# Individual tests per animation class
# ---------------------------------------------------------------------------

def test_metaballs(params, group, dummy_frame):
    from animations.metaballs import Metaballs
    anim = Metaballs(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_plasma(params, group, dummy_frame):
    from animations.plasma import Plasma
    anim = Plasma(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_reaction_diffusion(params, group, dummy_frame):
    from animations.reaction_diffusion import ReactionDiffusion
    anim = ReactionDiffusion(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_moire(params, group, dummy_frame):
    from animations.moire import Moire
    anim = Moire(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_dla(params, group, dummy_frame):
    from animations.dla import DLA
    anim = DLA(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_physarum(params, group, dummy_frame):
    from animations.physarum import Physarum
    # Use minimal agents count to keep test fast
    anim = Physarum(params, width=WIDTH, height=HEIGHT, group=group)
    # Override num_agents to a small value for speed
    anim.num_agents.value = 10
    anim.agents = anim._initialize_agents()
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_strange_attractor(params, group, dummy_frame):
    from animations.strange_attractor import StrangeAttractor
    anim = StrangeAttractor(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_chladni(params, group, dummy_frame):
    from animations.chladni import Chladni
    anim = Chladni(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_voronoi(params, group, dummy_frame):
    from animations.voronoi import Voronoi
    anim = Voronoi(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_oscillator_grid(params, group, dummy_frame):
    from animations.oscillator_grid import OscillatorGrid
    anim = OscillatorGrid(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_drift_field(params, group, dummy_frame):
    from animations.drift_field import DriftField
    anim = DriftField(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_harmonic_interference(params, group, dummy_frame):
    from animations.harmonic_interference import HarmonicInterference
    anim = HarmonicInterference(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_lenia(params, group, dummy_frame):
    from animations.lenia import Lenia
    anim = Lenia(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_fractal_zoom(params, group, dummy_frame):
    from animations.fractal_zoom import FractalZoom
    anim = FractalZoom(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


def test_perlin_noise(params, group, dummy_frame):
    from animations.perlin_noise import PerlinNoise
    anim = PerlinNoise(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


@pytest.mark.skip(reason="Shaders requires a GPU/moderngl OpenGL context unavailable in CI")
def test_shaders(params, group, dummy_frame):
    from animations.shaders import Shaders
    anim = Shaders(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


@pytest.mark.skip(reason="Shaders2 requires a GPU/moderngl OpenGL context unavailable in CI")
def test_shaders2(params, group, dummy_frame):
    from animations.shaders2 import Shaders2
    anim = Shaders2(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)


@pytest.mark.skip(reason="Shaders3 requires a GPU/moderngl OpenGL context unavailable in CI")
def test_shaders3(params, group, dummy_frame):
    from animations.shaders3 import Shaders3
    anim = Shaders3(params, width=WIDTH, height=HEIGHT, group=group)
    result = anim.get_frame(dummy_frame)
    _assert_valid_frame(result)
