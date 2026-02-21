"""
OBS filter parameter definitions and update logic.
Defines GUI-controllable parameters for OBS source filters (Color Correction, Chroma Key,
Sharpen, Source Transform) and sends updates to OBS via WebSocket.
"""

import time
import logging
from enum import IntEnum
from param import ParamTable
from common import Groups, Widget, Toggle

log = logging.getLogger(__name__)


class ChromaKeyColor(IntEnum):
    GREEN = 0
    BLUE = 1
    MAGENTA = 2
    CUSTOM = 3


CHROMA_KEY_COLOR_NAMES = {
    ChromaKeyColor.GREEN: "green",
    ChromaKeyColor.BLUE: "blue",
    ChromaKeyColor.MAGENTA: "magenta",
    ChromaKeyColor.CUSTOM: "custom",
}


# OBS filter kind identifiers
FILTER_COLOR_CORRECTION = "color_filter_v2"
FILTER_CHROMA_KEY = "chroma_key_filter_v2"
FILTER_SHARPEN = "sharpness_filter_v2"

# Filter names as they appear in OBS
FILTER_NAME_CC = "VS Color Correction"
FILTER_NAME_CK = "VS Chroma Key"
FILTER_NAME_SHARPEN = "VS Sharpen"


class OBSFilters:
    """
    Manages OBS filter parameters and sends updates to OBS via WebSocket.
    Not an effect class — does not process video frames.
    """

    def __init__(self, obs_controller=None):
        self.obs = obs_controller
        self.group = Groups.OBS
        self.params = ParamTable(group="OBS")

        self._last_sent = {}
        self._last_update_time = 0
        self._update_interval = 0.1  # Max 10 updates/sec
        self._filters_created = set()

        self._init_connection_params()
        self._init_color_correction_params()
        self._init_chroma_key_params()
        self._init_sharpen_params()
        self._init_transform_params()

    def _init_connection_params(self):
        subgroup = "Connection"
        self.enabled = self.params.add("obs_enabled",
                                       min=0, max=1, default=0,
                                       subgroup=subgroup, group=self.group,
                                       type=Widget.TOGGLE)
        self.source_name = self.params.add("obs_source_name",
                                           min=0, max=0, default=0,
                                           subgroup=subgroup, group=self.group)

    def _init_color_correction_params(self):
        subgroup = "Color Correction"
        self.cc_enabled = self.params.add("obs_cc_enabled",
                                          min=0, max=1, default=0,
                                          subgroup=subgroup, group=self.group,
                                          type=Widget.TOGGLE)
        self.cc_hue_shift = self.params.add("obs_cc_hue_shift",
                                            min=-180.0, max=180.0, default=0.0,
                                            subgroup=subgroup, group=self.group)
        self.cc_saturation = self.params.add("obs_cc_saturation",
                                             min=-1.0, max=5.0, default=0.0,
                                             subgroup=subgroup, group=self.group)
        self.cc_contrast = self.params.add("obs_cc_contrast",
                                           min=-2.0, max=2.0, default=0.0,
                                           subgroup=subgroup, group=self.group)
        self.cc_brightness = self.params.add("obs_cc_brightness",
                                             min=-1.0, max=1.0, default=0.0,
                                             subgroup=subgroup, group=self.group)
        self.cc_gamma = self.params.add("obs_cc_gamma",
                                        min=-3.0, max=3.0, default=0.0,
                                        subgroup=subgroup, group=self.group)
        self.cc_opacity = self.params.add("obs_cc_opacity",
                                          min=0, max=100, default=100,
                                          subgroup=subgroup, group=self.group)

    def _init_chroma_key_params(self):
        subgroup = "Chroma Key"
        self.ck_enabled = self.params.add("obs_ck_enabled",
                                          min=0, max=1, default=0,
                                          subgroup=subgroup, group=self.group,
                                          type=Widget.TOGGLE)
        self.ck_key_color_type = self.params.add("obs_ck_key_color_type",
                                                  min=0, max=len(ChromaKeyColor) - 1, default=0,
                                                  subgroup=subgroup, group=self.group,
                                                  type=Widget.DROPDOWN, options=ChromaKeyColor)
        self.ck_similarity = self.params.add("obs_ck_similarity",
                                             min=1, max=1000, default=400,
                                             subgroup=subgroup, group=self.group)
        self.ck_smoothness = self.params.add("obs_ck_smoothness",
                                             min=1, max=1000, default=50,
                                             subgroup=subgroup, group=self.group)
        self.ck_spill = self.params.add("obs_ck_spill",
                                        min=1, max=1000, default=50,
                                        subgroup=subgroup, group=self.group)

    def _init_sharpen_params(self):
        subgroup = "Sharpen"
        self.sharpen_enabled = self.params.add("obs_sharpen_enabled",
                                               min=0, max=1, default=0,
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.TOGGLE)
        self.sharpen_sharpness = self.params.add("obs_sharpen_sharpness",
                                                 min=0.0, max=1.0, default=0.0,
                                                 subgroup=subgroup, group=self.group)

    def _init_transform_params(self):
        subgroup = "Transform"
        self.transform_enabled = self.params.add("obs_transform_enabled",
                                                 min=0, max=1, default=0,
                                                 subgroup=subgroup, group=self.group,
                                                 type=Widget.TOGGLE)
        self.pos_x = self.params.add("obs_pos_x",
                                     min=0.0, max=3840.0, default=0.0,
                                     subgroup=subgroup, group=self.group)
        self.pos_y = self.params.add("obs_pos_y",
                                     min=0.0, max=2160.0, default=0.0,
                                     subgroup=subgroup, group=self.group)
        self.rotation = self.params.add("obs_rotation",
                                        min=0.0, max=360.0, default=0.0,
                                        subgroup=subgroup, group=self.group)
        self.scale_x = self.params.add("obs_scale_x",
                                       min=0.0, max=5.0, default=1.0,
                                       subgroup=subgroup, group=self.group)
        self.scale_y = self.params.add("obs_scale_y",
                                       min=0.0, max=5.0, default=1.0,
                                       subgroup=subgroup, group=self.group)
        self.crop_top = self.params.add("obs_crop_top",
                                        min=0, max=1920, default=0,
                                        subgroup=subgroup, group=self.group)
        self.crop_bottom = self.params.add("obs_crop_bottom",
                                           min=0, max=1920, default=0,
                                           subgroup=subgroup, group=self.group)
        self.crop_left = self.params.add("obs_crop_left",
                                         min=0, max=1920, default=0,
                                         subgroup=subgroup, group=self.group)
        self.crop_right = self.params.add("obs_crop_right",
                                          min=0, max=1920, default=0,
                                          subgroup=subgroup, group=self.group)

    def _get_source_name(self):
        """Get the current source name. Returns None if not set."""
        val = self.source_name.value
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    def _ensure_filter_exists(self, source_name, filter_name, filter_kind):
        """Create filter on source if it doesn't exist yet."""
        key = f"{source_name}:{filter_name}"
        if key in self._filters_created:
            return True

        existing = self.obs.get_source_filters(source_name)
        for f in existing:
            if f.get('filterName') == filter_name:
                self._filters_created.add(key)
                return True

        if self.obs.create_filter(source_name, filter_name, filter_kind):
            self._filters_created.add(key)
            return True
        return False

    def _has_changed(self, key, value):
        """Check if a value differs from last sent."""
        if key not in self._last_sent or self._last_sent[key] != value:
            self._last_sent[key] = value
            return True
        return False

    def _update_color_correction(self, source_name):
        if not self.cc_enabled.value:
            return

        settings = {
            "hue_shift": float(self.cc_hue_shift.value),
            "saturation": float(self.cc_saturation.value),
            "contrast": float(self.cc_contrast.value),
            "brightness": float(self.cc_brightness.value),
            "gamma": float(self.cc_gamma.value),
            "opacity": int(self.cc_opacity.value),
        }

        # Only send if something changed
        cache_key = f"cc:{source_name}"
        settings_tuple = tuple(settings.values())
        if not self._has_changed(cache_key, settings_tuple):
            return

        if self._ensure_filter_exists(source_name, FILTER_NAME_CC, FILTER_COLOR_CORRECTION):
            self.obs.set_filter_settings(source_name, FILTER_NAME_CC, settings)

    def _update_chroma_key(self, source_name):
        if not self.ck_enabled.value:
            return

        color_type_val = int(self.ck_key_color_type.value)
        if hasattr(self.ck_key_color_type.value, 'value'):
            color_type_val = self.ck_key_color_type.value.value
        color_name = CHROMA_KEY_COLOR_NAMES.get(color_type_val, "green")

        settings = {
            "key_color_type": color_name,
            "similarity": int(self.ck_similarity.value),
            "smoothness": int(self.ck_smoothness.value),
            "spill": int(self.ck_spill.value),
        }

        cache_key = f"ck:{source_name}"
        settings_tuple = tuple(settings.values())
        if not self._has_changed(cache_key, settings_tuple):
            return

        if self._ensure_filter_exists(source_name, FILTER_NAME_CK, FILTER_CHROMA_KEY):
            self.obs.set_filter_settings(source_name, FILTER_NAME_CK, settings)

    def _update_sharpen(self, source_name):
        if not self.sharpen_enabled.value:
            return

        settings = {
            "sharpness": float(self.sharpen_sharpness.value),
        }

        cache_key = f"sharpen:{source_name}"
        settings_tuple = tuple(settings.values())
        if not self._has_changed(cache_key, settings_tuple):
            return

        if self._ensure_filter_exists(source_name, FILTER_NAME_SHARPEN, FILTER_SHARPEN):
            self.obs.set_filter_settings(source_name, FILTER_NAME_SHARPEN, settings)

    def _update_transform(self, source_name):
        if not self.transform_enabled.value:
            return

        transform = {
            "positionX": float(self.pos_x.value),
            "positionY": float(self.pos_y.value),
            "rotation": float(self.rotation.value),
            "scaleX": float(self.scale_x.value),
            "scaleY": float(self.scale_y.value),
            "cropTop": int(self.crop_top.value),
            "cropBottom": int(self.crop_bottom.value),
            "cropLeft": int(self.crop_left.value),
            "cropRight": int(self.crop_right.value),
        }

        cache_key = f"transform:{source_name}"
        transform_tuple = tuple(transform.values())
        if not self._has_changed(cache_key, transform_tuple):
            return

        # Transform requires scene name and item ID — get current scene
        scenes = self.obs.get_scenes()
        if not scenes:
            return
        current_scene = scenes[0] if scenes else None
        if current_scene:
            item_id = self.obs.get_scene_item_id(current_scene, source_name)
            if item_id is not None:
                self.obs.set_source_transform(current_scene, item_id, transform)

    def update(self):
        """
        Called each frame from the video loop.
        Sends changed filter settings to OBS, throttled to ~10 updates/sec.
        """
        if not self.enabled.value:
            return
        if self.obs is None or not self.obs.connected:
            return

        now = time.perf_counter()
        if now - self._last_update_time < self._update_interval:
            return
        self._last_update_time = now

        source_name = self._get_source_name()
        if source_name is None:
            return

        self._update_color_correction(source_name)
        self._update_chroma_key(source_name)
        self._update_sharpen(source_name)
        self._update_transform(source_name)
