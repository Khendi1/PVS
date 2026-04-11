"""
Smoke tests for every effect class.

Each test instantiates the class with appropriate arguments and calls its
primary processing method, asserting that a valid numpy array of shape
(480, 640, 3) is returned (or at least a numpy array, shape preserved).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "video_synth"))

import numpy as np
import pytest
from param import ParamTable

WIDTH = 640
HEIGHT = 480


class _Group:
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


def _assert_valid_bgr(result, height=HEIGHT, width=WIDTH):
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert result.ndim == 3
    assert result.shape[0] == height
    assert result.shape[1] == width
    assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------

def test_color_modify_hsv(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    # With default values (all shifts == 0) it returns unchanged input
    result = fx.modify_hsv(dummy_frame)
    _assert_valid_bgr(result)


def test_color_posterize(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    # Default levels == 0 → early return, but result must still be ndarray
    result = fx.posterize(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_adjust_brightness_contrast(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    result = fx.adjust_brightness_contrast(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_adjust_gamma(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    result = fx.adjust_gamma(dummy_frame.astype(np.float32))
    assert isinstance(result, np.ndarray)


def test_color_color_cycle(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    # Enable the effect so it exercises the real code path
    fx.color_cycle_speed.value = 1.0
    result = fx.color_cycle(dummy_frame)
    _assert_valid_bgr(result)


def test_color_channel_mix(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    # Modify one channel mix value so the identity-exit is skipped
    fx.ch_mix_rg.value = 0.1
    result = fx.channel_mix(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_color_bitcrush(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.color_bitcrush.value = 4
    result = fx.color_bitcrush(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_duotone(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.duotone_strength.value = 0.5
    result = fx.duotone(dummy_frame)
    _assert_valid_bgr(result)


def test_color_channel_isolate(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.ch_r.value = 0.5
    result = fx.channel_isolate(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_chromatic_aberration(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.chroma_ab_x.value = 5
    fx.chroma_ab_y.value = 5
    result = fx.chromatic_aberration(dummy_frame)
    _assert_valid_bgr(result)


def test_color_temperature(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.color_temp.value = 0.5
    result = fx.color_temperature(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_color_false_color(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.false_color_strength.value = 1.0
    result = fx.false_color(dummy_frame)
    _assert_valid_bgr(result)


def test_color_invert(params, group, dummy_frame):
    from effects.color import Color
    fx = Color(params, group=group)
    fx.invert_strength.value = 1.0
    result = fx.invert(dummy_frame)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def test_sync_passthrough_when_amp_zero(params, group, dummy_frame):
    from effects.sync import Sync
    fx = Sync(params, group=group)
    # Both amps default to 0.0 — should return the frame unchanged
    result = fx.sync(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == dummy_frame.shape


def test_sync_applies_effect(params, group, dummy_frame):
    from effects.sync import Sync
    fx = Sync(params, group=group)
    fx.x_sync_amp.value = 5.0
    result = fx.sync(dummy_frame)
    _assert_valid_bgr(result)


# ---------------------------------------------------------------------------
# PTZ
# ---------------------------------------------------------------------------

def test_ptz_shift_frame(params, group, dummy_frame):
    from effects.ptz import PTZ
    fx = PTZ(params, image_width=WIDTH, image_height=HEIGHT, group=group)
    result = fx.shift_frame(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Warp
# ---------------------------------------------------------------------------

def test_warp_default(params, group, dummy_frame):
    from effects.warp import Warp
    fx = Warp(params, image_width=WIDTH, image_height=HEIGHT, group=group)
    result = fx.warp(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == HEIGHT
    assert result.shape[1] == WIDTH


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------

def test_pixels_apply_noise(params, group, dummy_frame):
    from effects.pixels import Pixels
    fx = Pixels(params, image_width=WIDTH, image_height=HEIGHT, group=group)
    result = fx.apply_noise(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# ImageNoiser
# ---------------------------------------------------------------------------

def test_image_noiser_apply_noise_none_type(params, group, dummy_frame):
    from effects.image_noiser import ImageNoiser
    from effects.enums import NoiseType
    fx = ImageNoiser(params, noise_type=NoiseType.NONE, group=group)
    result = fx.apply_noise(dummy_frame)
    assert isinstance(result, np.ndarray)


def test_image_noiser_apply_noise_gaussian(params, group, dummy_frame):
    from effects.image_noiser import ImageNoiser
    from effects.enums import NoiseType
    # Use a fresh params table to avoid duplicate param names from the first test
    p2 = ParamTable(group="Test2")
    fx = ImageNoiser(p2, noise_type=NoiseType.NONE, group=group)
    # Set noise_type param directly to GAUSSIAN to exercise the gaussian path
    fx.noise_type.value = NoiseType.GAUSSIAN.value
    result = fx.apply_noise(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

def test_reflector_apply_reflection_none(params, group, dummy_frame):
    from effects.reflector import Reflector
    from effects.enums import ReflectionMode
    fx = Reflector(params, mode=ReflectionMode.NONE, group=group)
    result = fx.apply_reflection(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_reflector_apply_reflection_horizontal(params, group, dummy_frame):
    from effects.reflector import Reflector
    from effects.enums import ReflectionMode
    # Use a separate ParamTable to avoid duplicate param name collision
    p2 = ParamTable(group="Test2")
    fx = Reflector(p2, mode=ReflectionMode.HORIZONTAL, group=group)
    fx._mode.value = ReflectionMode.HORIZONTAL.value
    result = fx.apply_reflection(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_shapes_draw_shapes_none(params, group, dummy_frame):
    from effects.shapes import Shapes
    from effects.enums import Shape
    fx = Shapes(params, width=WIDTH, height=HEIGHT, group=group)
    # Default shape is NONE — draw_shapes should return frame untouched or with shape
    result = fx.draw_shapes(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_shapes_draw_shapes_circle(params, group, dummy_frame):
    from effects.shapes import Shapes
    from effects.enums import Shape
    fx = Shapes(params, width=WIDTH, height=HEIGHT, group=group)
    fx.shape_type.value = Shape.CIRCLE.value
    result = fx.draw_shapes(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Lissajous
# ---------------------------------------------------------------------------

def test_lissajous_pattern(params, group, dummy_frame):
    from effects.lissajous import Lissajous
    fx = Lissajous(params, width=WIDTH, height=HEIGHT, group=group)
    result = fx.lissajous_pattern(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------

def test_erosion_apply_erosion_passthrough(params, group, dummy_frame):
    from effects.erosion import Erosion
    fx = Erosion(params, width=WIDTH, height=HEIGHT, group=group)
    # Default strength is 0.0 — should return input unchanged
    result = fx.apply_erosion(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_erosion_apply_erosion_active(params, group, dummy_frame):
    from effects.erosion import Erosion
    fx = Erosion(params, width=WIDTH, height=HEIGHT, group=group)
    fx.erosion_strength.value = 0.5
    result = fx.apply_erosion(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == (HEIGHT, WIDTH, 3)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def test_feedback_apply_temporal_filter(params, group, dummy_frame):
    from effects.feedback import Feedback
    fx = Feedback(params, image_width=WIDTH, image_height=HEIGHT, group=group)
    prev = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cur = dummy_frame.copy()
    result = fx.apply_temporal_filter(prev, cur)
    assert isinstance(result, np.ndarray)


def test_feedback_apply_luma_feedback(params, group, dummy_frame):
    from effects.feedback import Feedback
    fx = Feedback(params, image_width=WIDTH, image_height=HEIGHT, group=group)
    prev = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result = fx.apply_luma_feedback(prev, dummy_frame)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Glitch
# ---------------------------------------------------------------------------

def test_glitch_apply_glitch_effects_default(params, group, dummy_frame):
    from effects.glitch import Glitch
    fx = Glitch(params, group=group)
    # All effects are disabled by default — should return the frame with no crash
    result = fx.apply_glitch_effects(dummy_frame)
    assert isinstance(result, np.ndarray)
