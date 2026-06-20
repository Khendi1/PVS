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
Text & Typography Engine animation source.

Renders animated text onto a black canvas using Pillow (PIL) for high-quality
font rendering.  The text can scroll horizontally or vertically, pulse in
opacity/scale, cycle colours, and be updated at runtime via the `set_message`
method (called by the API / web UI ticker input).

Parameters exposed to the param system
---------------------------------------
text_scroll_speed   — pixels per frame (positive = left, negative = right)
text_scroll_axis    — 0 = horizontal, 1 = vertical
text_font_size      — point size
text_brightness     — overall brightness multiplier 0–1
text_r / g / b      — base text colour channels 0–255
text_pulse_speed    — opacity pulse rate (0 = disabled)
text_pulse_depth    — how much opacity oscillates (0–1)
text_bg_alpha       — background fill opacity (0 = transparent, 1 = solid black)
text_letter_spacing — extra pixels between characters (−10 to +50)
text_line_spacing   — extra pixels between lines (−10 to +50)
"""

import time
import logging
import threading
import numpy as np
import cv2

from animations.base import Animation
from common import Widget

log = logging.getLogger(__name__)

# Default built-in messages shown in round-robin when no custom text is set.
_DEFAULT_MESSAGES = [
    "VIDEO SYNTH",
    "LIVE AV",
    "◈ ◈ ◈",
    "∿ SIGNAL ∿",
]

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    log.warning("TextEngine: Pillow not installed — text frames will be black. pip install Pillow")


def _get_font(size: int):
    """Try to load a monospace font at *size* pt; fall back to Pillow default."""
    if not _PIL_AVAILABLE:
        return None
    candidates = [
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
        "LiberationMono-Regular.ttf",
        "Courier New.ttf",
        "cour.ttf",      # Windows
        "Menlo.ttc",     # macOS
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    try:
        return ImageFont.load_default(size=max(10, size))
    except TypeError:
        return ImageFont.load_default()


class TextEngine(Animation):
    """Animated text/ticker animation source."""

    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__

        # Runtime-mutable message (thread-safe via lock)
        self._message_lock = threading.Lock()
        self._custom_message: str | None = None
        self._default_index = 0
        self._last_default_change = time.monotonic()
        self._default_cycle_seconds = 8.0

        # Scroll state
        self._scroll_pos = 0.0
        self._last_time = time.monotonic()

        # --- Params ---
        self.scroll_speed = params.new("text_scroll_speed",
                                       min=-20.0, max=20.0, default=2.0,
                                       subgroup=subgroup, group=group)
        self.scroll_axis = params.new("text_scroll_axis",
                                      min=0, max=1, default=0,
                                      subgroup=subgroup, group=group,
                                      type=Widget.DROPDOWN,
                                      options=["Horizontal", "Vertical"])
        self.font_size = params.new("text_font_size",
                                    min=8, max=256, default=48,
                                    subgroup=subgroup, group=group)
        self.brightness = params.new("text_brightness",
                                     min=0.0, max=1.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.color_r = params.new("text_r", min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.color_g = params.new("text_g", min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.color_b = params.new("text_b", min=0, max=255, default=255,
                                  subgroup=subgroup, group=group)
        self.pulse_speed = params.new("text_pulse_speed",
                                      min=0.0, max=5.0, default=0.0,
                                      subgroup=subgroup, group=group)
        self.pulse_depth = params.new("text_pulse_depth",
                                      min=0.0, max=1.0, default=0.5,
                                      subgroup=subgroup, group=group)
        self.bg_alpha = params.new("text_bg_alpha",
                                   min=0.0, max=1.0, default=0.0,
                                   subgroup=subgroup, group=group)
        self.letter_spacing = params.new("text_letter_spacing",
                                         min=-10, max=50, default=0,
                                         subgroup=subgroup, group=group)
        self.line_spacing = params.new("text_line_spacing",
                                       min=-10, max=50, default=4,
                                       subgroup=subgroup, group=group)

        # Cache to avoid re-rendering static text every frame
        self._cache_key = None
        self._cached_text_img = None

    # ------------------------------------------------------------------
    # Public API — called by REST endpoint / web UI
    # ------------------------------------------------------------------

    def set_message(self, text: str | None):
        """Set a custom message.  Pass None to revert to built-in rotation."""
        with self._message_lock:
            self._custom_message = text
            self._scroll_pos = 0.0

    def get_message(self) -> str:
        with self._message_lock:
            if self._custom_message is not None:
                return self._custom_message
        now = time.monotonic()
        if now - self._last_default_change > self._default_cycle_seconds:
            self._default_index = (self._default_index + 1) % len(_DEFAULT_MESSAGES)
            self._last_default_change = now
        return _DEFAULT_MESSAGES[self._default_index]

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_text_image(self, text: str, font_size: int, color: tuple,
                            letter_spacing: int, line_spacing: int) -> np.ndarray:
        """Render *text* onto a transparent RGBA canvas sized to the text.

        Returns an RGBA uint8 ndarray.
        """
        if not _PIL_AVAILABLE:
            return np.zeros((self.height, self.width, 4), dtype=np.uint8)

        font = _get_font(font_size)
        lines = text.split("\n")

        # Measure each line with letter spacing applied
        tmp = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(tmp)
        line_heights = []
        line_widths = []
        for line in lines:
            if letter_spacing != 0 and line:
                # Render char by char to apply custom letter spacing
                w = sum(draw.textlength(ch, font=font) for ch in line)
                w += letter_spacing * max(0, len(line) - 1)
            else:
                w = draw.textlength(line, font=font)
            bbox = font.getbbox(line or " ")
            h = bbox[3] - bbox[1]
            line_widths.append(int(w))
            line_heights.append(h)

        total_w = max(line_widths) if line_widths else self.width
        row_h = max(line_heights) if line_heights else font_size
        gap = row_h + int(line_spacing)
        total_h = gap * len(lines) - int(line_spacing)

        canvas = Image.new("RGBA", (max(1, total_w), max(1, total_h)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        y = 0
        for line, lw in zip(lines, line_widths):
            if letter_spacing != 0 and line:
                x = 0
                for ch in line:
                    draw.text((x, y), ch, font=font, fill=color)
                    x += int(draw.textlength(ch, font=font)) + letter_spacing
            else:
                draw.text((0, y), line, font=font, fill=color)
            y += gap

        return np.array(canvas, dtype=np.uint8)

    def _composite(self, text_rgba: np.ndarray, brightness: float,
                   opacity: float, bg_alpha: float) -> np.ndarray:
        """Composite the text RGBA image into a BGR output frame."""
        out = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Background fill
        if bg_alpha > 0.0:
            out[:] = bg_alpha * 255.0

        if text_rgba is None or text_rgba.size == 0:
            return np.clip(out, 0, 255).astype(np.uint8)

        th, tw = text_rgba.shape[:2]
        if th == 0 or tw == 0:
            return np.clip(out, 0, 255).astype(np.uint8)

        # Tile the text image so it fills the canvas (for scrolling)
        # We need enough tiles to cover width+tw (horizontal) or height+th (vertical)
        if self.scroll_axis.value == 0:  # horizontal
            reps_x = (self.width + tw - 1) // tw + 2
            reps_y = max(1, (self.height + th - 1) // th)
        else:  # vertical
            reps_x = max(1, (self.width + tw - 1) // tw)
            reps_y = (self.height + th - 1) // th + 2

        tiled = np.tile(text_rgba, (reps_y, reps_x, 1))

        # Apply scroll offset
        sp = int(self._scroll_pos)
        if self.scroll_axis.value == 0:
            offset_x = sp % tw if tw > 0 else 0
            offset_y = (self.height - th) // 2
            offset_y = max(0, min(offset_y, tiled.shape[0] - self.height))
        else:
            offset_x = (self.width - tw) // 2
            offset_x = max(0, min(offset_x, tiled.shape[1] - self.width))
            offset_y = sp % th if th > 0 else 0

        crop = tiled[offset_y:offset_y + self.height, offset_x:offset_x + self.width]

        # Pad if crop is smaller than frame (edge case for very large fonts)
        if crop.shape[0] < self.height or crop.shape[1] < self.width:
            pad = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            h_c = min(crop.shape[0], self.height)
            w_c = min(crop.shape[1], self.width)
            pad[:h_c, :w_c] = crop[:h_c, :w_c]
            crop = pad

        # Alpha composite text over background
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0 * opacity * brightness
        rgb = crop[:, :, :3].astype(np.float32)
        # BGR order (OpenCV)
        out[:, :, 0] += alpha[:, :, 0] * rgb[:, :, 2]
        out[:, :, 1] += alpha[:, :, 0] * rgb[:, :, 1]
        out[:, :, 2] += alpha[:, :, 0] * rgb[:, :, 0]

        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Animation interface
    # ------------------------------------------------------------------

    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        message = self.get_message()
        font_size = int(self.font_size.value)
        letter_spacing = int(self.letter_spacing.value)
        line_spacing = int(self.line_spacing.value)

        r = int(self.color_r.value)
        g = int(self.color_g.value)
        b = int(self.color_b.value)
        color = (r, g, b, 255)

        # Re-render text only when style params change
        cache_key = (message, font_size, r, g, b, letter_spacing, line_spacing)
        if cache_key != self._cache_key:
            self._cached_text_img = self._render_text_image(
                message, font_size, color, letter_spacing, line_spacing)
            self._cache_key = cache_key
            self._scroll_pos = 0.0

        # Advance scroll
        self._scroll_pos += self.scroll_speed.value * dt * 30.0  # normalise to 30 fps

        # Pulse opacity
        brightness = float(self.brightness.value)
        pulse_speed = float(self.pulse_speed.value)
        if pulse_speed > 0.0:
            depth = float(self.pulse_depth.value)
            osc = (np.sin(now * pulse_speed * 2 * np.pi) + 1.0) / 2.0  # 0–1
            opacity = 1.0 - depth + depth * osc
        else:
            opacity = 1.0

        return self._composite(
            self._cached_text_img,
            brightness=brightness,
            opacity=opacity,
            bg_alpha=float(self.bg_alpha.value),
        )
