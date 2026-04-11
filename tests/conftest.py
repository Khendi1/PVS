"""
Shared pytest fixtures for the video_synth test suite.
"""
import sys
import os
from pathlib import Path

# src/ layout: add src/ (for `import video_synth`) and src/video_synth/
# (for the bare internal imports like `from api import APIServer`).
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


@pytest.fixture
def width():
    return WIDTH


@pytest.fixture
def height():
    return HEIGHT


@pytest.fixture
def params():
    """Return a fresh ParamTable for each test."""
    return ParamTable(group="Test")


@pytest.fixture
def dummy_frame():
    """Return a black 480x640x3 uint8 numpy array."""
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class _Group:
    """Minimal group stub: just needs a .name attribute."""
    def __init__(self, name: str = "SRC_1_EFFECTS"):
        self.name = name


@pytest.fixture
def group():
    """Return a lightweight group stub compatible with all animations/effects."""
    return _Group("SRC_1_EFFECTS")
