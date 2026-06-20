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
The Mixer object is used for managing and mixing video sources into a single frame
A single mixed frame is retrieved every iteration of the main loop.
"""

import cv2
import numpy as np
from enum import Enum, IntEnum, auto
from animations.enums import *
from animations.metaballs import Metaballs
from animations.plasma import Plasma
from animations.reaction_diffusion import ReactionDiffusion
from animations.moire import Moire
from animations.shaders import Shaders
from animations.shaders2 import Shaders2
from animations.strange_attractor import StrangeAttractor
from animations.physarum import Physarum
from animations.dla import DLA
from animations.chladni import Chladni
from animations.voronoi import Voronoi
from animations.drift_field import DriftField
from animations.lenia import Lenia
from animations.fractal_zoom import FractalZoom
from animations.oscillator_grid import OscillatorGrid
from animations.harmonic_interference import HarmonicInterference
from animations.perlin_noise import PerlinNoise
from animations.shaders3 import Shaders3
from animations.text_engine import TextEngine
from animations.screen_capture import ScreenCapture
import os
from pathlib import Path
from param import ParamTable
from luma import LumaMode
from luma import *
from common import *
import concurrent.futures 


log = logging.getLogger(__name__)

class FileSource(Enum):
    VIDEO = auto()
    IMAGE = auto()


class MixModes(IntEnum):
    ALPHA_BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2


class ImageSource:
    """Serves a static image as a frame source, matching the animation get_frame interface."""
    def __init__(self, image_path, width, height):
        img = cv2.imread(image_path)
        if img is not None:
            self.image = cv2.resize(img, (width, height)).astype(np.float32)
        else:
            log.error(f"Could not load image: {image_path}")
            self.image = np.zeros((height, width, 3), dtype=np.float32)

    def get_frame(self, frame=None):
        return self.image.copy()


class Mixer:
    """Mixer is used to init sources, get frames from each, and blend them"""

    def __init__(self, effects, num_devices, width=640, height=480):

        self.group = Groups.MIXER
        subgroup = self.group

        self.width = width
        self.height = height

        self.src_1_effects = effects[MixerSource.SRC_1]
        self.src_2_effects = effects[MixerSource.SRC_2]
        self.post_effects = effects[MixerSource.POST]

        self.params = ParamTable(group="Mixer")

        # cap variables to store video capture objects or animation instances
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip = False

        # current frame for API snapshot endpoint
        self.current_frame = None

        # search for available video capture devices
        num_devices_int = int(num_devices)  # Param object -> int
        found_devices = self._detect_devices(max_index=num_devices_int)

        # Only add devices that were actually detected
        self.sources = {}
        # Map device names to their cv2 capture index
        self._device_indices = {}
        for dev_index in found_devices:
            name = f"DEVICE_{dev_index}"
            self.sources[name] = len(self.sources)
            self._device_indices[name] = dev_index

        # Virtual camera is always available as a dedicated source
        self.sources["VIRTUAL_CAM"] = len(self.sources)

        # add file sources to sources dict
        for file_source in FileSource:
            self.sources[file_source.name] = len(self.sources)
        # add animation sources to sources dict
        for anim_source in AnimSource:
            self.sources[anim_source.name] = len(self.sources)
        
        srcs_str = ", ".join([f"\n\t{k}: {v}" for k,v in self.sources.items()])
        log.info(f"Sources: {srcs_str}")

        # --- Build animation arg dict for convenient passing ---
        
        anim_args = {
            "group": Groups.SRC_1_ANIMATIONS,
            "params": self.src_1_effects.params,
            "width": self.width,
            "height": self.height
        }
        self.src_1_animations = {
            AnimSource.METABALLS.name: Metaballs(**anim_args),
            AnimSource.PLASMA.name: Plasma(**anim_args),
            AnimSource.REACTION_DIFFUSION.name: ReactionDiffusion(**anim_args),
            AnimSource.MOIRE.name: Moire(**anim_args, oscs=self.src_1_effects.oscs),
            AnimSource.SHADERS.name: Shaders(**anim_args),
            AnimSource.SHADERS_2.name: Shaders2(**anim_args),
            AnimSource.STRANGE_ATTRACTOR.name: StrangeAttractor(**anim_args),
            AnimSource.PHYSARUM.name: Physarum(**anim_args),
            AnimSource.DLA.name: DLA(**anim_args),
            AnimSource.CHLADNI.name: Chladni(**anim_args),
            AnimSource.VORONOI.name: Voronoi(**anim_args),
            AnimSource.DRIFT_FIELD.name: DriftField(**anim_args),
            AnimSource.LENIA.name: Lenia(**anim_args),
            AnimSource.FRACTAL_ZOOM.name: FractalZoom(**anim_args),
            AnimSource.OSCILLATOR_GRID.name: OscillatorGrid(**anim_args),
            AnimSource.HARMONIC_INTERFERENCE.name: HarmonicInterference(**anim_args),
            AnimSource.PERLIN_NOISE.name: PerlinNoise(**anim_args),
            AnimSource.SHADERS_3.name: Shaders3(**anim_args),
            AnimSource.TEXT_ENGINE.name: TextEngine(**anim_args),
            AnimSource.SCREEN_CAPTURE.name: ScreenCapture(**anim_args),
        }

        anim_args["group"] = Groups.SRC_2_ANIMATIONS
        anim_args["params"] = self.src_2_effects.params
        self.src_2_animations = {
            AnimSource.METABALLS.name: Metaballs(**anim_args),
            AnimSource.PLASMA.name: Plasma(**anim_args),
            AnimSource.REACTION_DIFFUSION.name: ReactionDiffusion(**anim_args),
            AnimSource.MOIRE.name: Moire(**anim_args, oscs=self.src_2_effects.oscs),
            AnimSource.SHADERS.name: Shaders(**anim_args),
            AnimSource.SHADERS_2.name: Shaders2(**anim_args),
            AnimSource.STRANGE_ATTRACTOR.name: StrangeAttractor(**anim_args),
            AnimSource.PHYSARUM.name: Physarum(**anim_args),
            AnimSource.DLA.name: DLA(**anim_args),
            AnimSource.CHLADNI.name: Chladni(**anim_args),
            AnimSource.VORONOI.name: Voronoi(**anim_args),
            AnimSource.DRIFT_FIELD.name: DriftField(**anim_args),
            AnimSource.LENIA.name: Lenia(**anim_args),
            AnimSource.FRACTAL_ZOOM.name: FractalZoom(**anim_args),
            AnimSource.OSCILLATOR_GRID.name: OscillatorGrid(**anim_args),
            AnimSource.HARMONIC_INTERFERENCE.name: HarmonicInterference(**anim_args),
            AnimSource.PERLIN_NOISE.name: PerlinNoise(**anim_args),
            AnimSource.SHADERS_3.name: Shaders3(**anim_args),
            AnimSource.TEXT_ENGINE.name: TextEngine(**anim_args),
            AnimSource.SCREEN_CAPTURE.name: ScreenCapture(**anim_args),
        }

        # --- Configure file sources ---
        try:
            self.video_samples = os.listdir(self._find_dir("samples"))
        except (FileNotFoundError, OSError):
            self.video_samples = []

        try:
            self.images = os.listdir(self._find_dir("images"))
        except (FileNotFoundError, OSError):
            self.images = []

        # --- Source Params ---

        # initialize source 1 to use the first hardware device available
        # safely default to metaballs if no devices found
        default_src1 = f"DEVICE_{found_devices[0]}" if found_devices else AnimSource.METABALLS.name
        self.selected_source1 = self.params.new("source_1",
                                                      min=0, max=len(self.sources), default=default_src1,
                                                      subgroup=subgroup, group=self.group,
                                                      type=Widget.DROPDOWN, options=list(self.sources.keys()),
                                                      info="Selects the animation/source for layer 1")

        # init source 2 to metaballs
        self.selected_source2 = self.params.new("source_2",
                                                      min=0, max=len(self.sources), default=AnimSource.METABALLS.name,
                                                      subgroup=subgroup, group=self.group,
                                                      type=Widget.DROPDOWN, options=list(self.sources.keys()),
                                                      info="Selects the animation/source for layer 2")

        # --- File selection params ---
        video_options = self.video_samples if self.video_samples else ["(none)"]
        image_options = self.images if self.images else ["(none)"]

        self.video_file_src1 = self.params.new("video_file_src1",
                                               min=0, max=len(video_options),
                                               default=video_options[0],
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.DROPDOWN, options=video_options,
                                               info="Selects a video file as source 1")
        self.video_file_src2 = self.params.new("video_file_src2",
                                               min=0, max=len(video_options),
                                               default=video_options[0],
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.DROPDOWN, options=video_options,
                                               info="Selects a video file as source 2")
        self.image_file_src1 = self.params.new("image_file_src1",
                                               min=0, max=len(image_options),
                                               default=image_options[0],
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.DROPDOWN, options=image_options,
                                               info="Selects an image file as source 1")
        self.image_file_src2 = self.params.new("image_file_src2",
                                               min=0, max=len(image_options),
                                               default=image_options[0],
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.DROPDOWN, options=image_options,
                                               info="Selects an image file as source 2")

        # --- Video playback controls (only meaningful for VIDEO file sources) ---

        self.video_pause_src1 = self.params.new("video_pause_src1",
                                                min=0, max=1, default=0,
                                                subgroup=subgroup, group=self.group,
                                                type=Widget.TOGGLE,
                                                info="Pause/resume playback of the source 1 video file")
        self.video_pause_src2 = self.params.new("video_pause_src2",
                                                min=0, max=1, default=0,
                                                subgroup=subgroup, group=self.group,
                                                type=Widget.TOGGLE,
                                                info="Pause/resume playback of the source 2 video file")
        self.video_scrub_src1 = self.params.new("video_scrub_src1",
                                                min=0.0, max=100.0, default=0.0,
                                                subgroup=subgroup, group=self.group,
                                                info="Scrub position in the source 1 video file (0–100%)")
        self.video_scrub_src2 = self.params.new("video_scrub_src2",
                                                min=0.0, max=100.0, default=0.0,
                                                subgroup=subgroup, group=self.group,
                                                info="Scrub position in the source 2 video file (0–100%)")

        # --- Parameters for blending and keying ---

        self.blend_mode = self.params.new("blend_mode",
                                               min=0, max=2, default=0,
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.RADIO, options=MixModes,
                                               info="How the two sources are combined (alpha, luma key, chroma key)")
        self.luma_threshold = self.params.new("luma_threshold",
                                                   min=0, max=255, default=128,
                                                   subgroup=subgroup, group=self.group,
                                                   info="Brightness level that defines the luma key boundary (0–255)")
        self.luma_selection = self.params.new("luma_selection",
                                                   min=LumaMode.WHITE.value, max=LumaMode.BLACK.value, default=LumaMode.WHITE.value,
                                                   subgroup=subgroup, group=self.group,
                                                   type=Widget.RADIO, options=LumaMode,
                                                   info="Whether to key out bright or dark pixels")
        self.luma_blur = self.params.new("luma_blur",
                                              min=1, max=51, default=1,
                                              subgroup=subgroup, group=self.group,
                                              info="Feathers the edges of the luma key mask")
        self.upper_hue = self.params.new("upper_hue",
                                              min=0, max=179, default=80,
                                              subgroup=subgroup, group=self.group,
                                              info="Upper bound of the chroma key hue range")
        self.upper_saturation = self.params.new("upper_sat",
                                                     min=0, max=255, default=255,
                                                     subgroup=subgroup, group=self.group,
                                                     info="Upper bound of the chroma key saturation range")
        self.upper_value = self.params.new("upper_val",
                                                min=0, max=255, default=255,
                                                subgroup=subgroup, group=self.group,
                                                info="Upper bound of the chroma key value range")
        self.lower_hue = self.params.new("lower_hue",
                                              min=0, max=179, default=0,
                                              subgroup=subgroup, group=self.group,
                                              info="Lower bound of the chroma key hue range")
        self.lower_saturation = self.params.new("lower_sat",
                                                     min=0, max=255, default=100,
                                                     subgroup=subgroup, group=self.group,
                                                     info="Lower bound of the chroma key saturation range")
        self.lower_value = self.params.new("lower_val",
                                                min=0, max=255, default=100,
                                                subgroup=subgroup, group=self.group,
                                                info="Lower bound of the chroma key value range")

        self.alpha_blend = self.params.new("alpha_blend",
                                                min=0.0, max=1.0, default=0.5,
                                                subgroup=subgroup, group=self.group,
                                                info="Mix ratio between source 1 and source 2 (0 = all S1, 1 = all S2)")

        self.swap = self.params.new("swap_sources",
                                         min=0, max=1, default=0,
                                         subgroup=subgroup, group=self.group,
                                         type=Widget.TOGGLE,
                                         info="Toggle to swap source 1 and source 2")

        # --- Initialize previous and wet frames for each source ---

        self.src_1_prev = None
        self.src_2_prev = None
        self.post_prev = None

        self.src_1_wet = None
        self.src_2_wet = None
        self.post_wet = None

        self.src_1_count = 0
        self.src_2_count = 0
        self.post_count = 0

        # --- Video playback state (per source) ---
        # Last decoded raw video frame, replayed while a video source is paused.
        self._last_video_frame = {MixerSource.SRC_1: None, MixerSource.SRC_2: None}
        # Last scrub value applied to each capture, used to seek only on change.
        self._scrub_last = {MixerSource.SRC_1: None, MixerSource.SRC_2: None}

        # --- Start video sources ---
        self.start_video(self.selected_source1.value, MixerSource.SRC_1)
        self.start_video(self.selected_source2.value, MixerSource.SRC_2)

        # Cached frames for when sources timeout
        self._cached_frame1 = None
        self._cached_frame2 = None
        self._first_frame_received = False

        # Performance controls
        self.blackout = False
        self.freeze = False
        self._frozen_frame = None


    # find dir one level up from current working directory
    def _find_dir(self, dir_name: str, file_name: str = None):
        # __file__ is src/video_synth/mixer.py — go up two levels to project root
        project_root = Path(__file__).parent.parent.parent

        video_path_object = (
            project_root / dir_name / file_name
            if file_name is not None
            else project_root / dir_name
        )

        return str(video_path_object.resolve().as_posix())


    def _detect_devices(self, max_index: int):
        """Scan device indices and return list of indices that responded."""
        log.info(f"Scanning for video capture devices (indices 0-{max_index - 1})")
        found = []
        for index in range(int(max_index)):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        log.info(f"Found video capture device at index {index}")
                        found.append(index)
                    cap.release()
                else:
                    log.debug(f"No capture device at index {index}")
            except Exception as e:
                log.error(f"Error while checking device at index {index}: {e}")
        log.info(f"Device scan complete: {len(found)} device(s) found")
        return found


    def _find_virtualcam_index(self):
        """Find the cv2 capture index for the pyvirtualcam device.

        Scans indices beyond the detected physical devices to find
        the virtual camera output loopback.
        """
        for index in range(MAX_DEVICES + 5):
            if index in self._device_indices.values():
                continue  # skip known physical devices
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        log.info(f"Found virtual camera at index {index}")
                        return index
            except Exception:
                continue
        return None


    def _open_cv2_capture(self, cap, source_name, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        log.info(f"Opening cv2 VideoCapture for source_{index}: {source_name}")

        if source_name in self._device_indices:
            # Physical device — use the actual cv2 device index
            source_val = self._device_indices[source_name]
        elif source_name == "VIRTUAL_CAM":
            # Virtual camera — find it by scanning for the pyvirtualcam device
            source_val = self._find_virtualcam_index()
            if source_val is None:
                log.warning("Virtual camera device not found")
                return cap
        elif source_name == FileSource.VIDEO.name:
            file_param = self.video_file_src1 if index == MixerSource.SRC_1 else self.video_file_src2
            if file_param.value and file_param.value != "(none)":
                source_val = self._find_dir("samples", file_param.value)
            else:
                log.warning(f"No video file selected for source {index}")
                return cap
        else:
            source_val = self.sources[source_name]

        cap = cv2.VideoCapture(source_val)

        # Reduce camera buffer to 1 frame to minimize latency and prevent stale frames
        if isinstance(source_val, int):  # Only for real camera devices (not files)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.live_caps.append(cap)
        self.skip = True

        if not cap.isOpened():
            log.error(f"Could not open video source {index}: {source_val}")
        return cap


    def _open_image_source(self, cap, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()
        file_param = self.image_file_src1 if index == MixerSource.SRC_1 else self.image_file_src2
        if file_param.value and file_param.value != "(none)":
            file_path = self._find_dir("images", file_param.value)
            log.info(f"Loading image source for source_{index}: {file_path}")
            return ImageSource(file_path, self.width, self.height)
        else:
            log.warning(f"No image file selected for source {index}")
            return cap


    def _open_animation(self, cap, source_name, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()
        animations = self.src_1_animations if index == MixerSource.SRC_1 else self.src_2_animations
        return animations[source_name]


    def _alpha_blend(self, frame1, frame2):
        alpha = self.alpha_blend.value

        return cv2.addWeighted(
            frame1.astype(np.float32), 1 - alpha, frame2.astype(np.float32), alpha, 0
        )


    def _luma_key(self, frame1, frame2):
        return luma_key(
            frame1, frame2, self.luma_selection.value, self.luma_threshold.value,
            self.luma_blur.value
        ).astype(np.float32)


    def _chroma_key(self, frame1, frame2, lower=(0, 100, 0), upper=(80, 255, 80)):
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask == 255, frame2, frame1)
        return result.astype(np.float32)


    def _get_pause_param(self, source_index):
        return self.video_pause_src1 if source_index == MixerSource.SRC_1 else self.video_pause_src2

    def _get_scrub_param(self, source_index):
        return self.video_scrub_src1 if source_index == MixerSource.SRC_1 else self.video_scrub_src2

    def _apply_scrub(self, cap, source_index):
        """Seek a video capture when its scrub param has changed since the last applied value."""
        scrub = self._get_scrub_param(source_index)
        last = self._scrub_last[source_index]
        if last is not None and abs(scrub.value - last) < 1e-6:
            return  # scrub unchanged — let playback advance normally

        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total and total > 0:
            target = int((scrub.value / 100.0) * total)
            target = max(0, min(target, int(total) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            # Drop the paused freeze-frame so the new position is shown even while paused
            self._last_video_frame[source_index] = None
        self._scrub_last[source_index] = scrub.value

    def _process_single_source(self, source_index):
        ret, frame = False, None

        cap = self.cap1 if source_index == MixerSource.SRC_1 else self.cap2
        selected_source = self.selected_source1 if source_index == MixerSource.SRC_1 else self.selected_source2

        # Read frame
        if not isinstance(cap, cv2.VideoCapture):
            frame = cap.get_frame(frame)
            ret = True
        else:
            # CRITICAL FIX: For live cameras, grab+retrieve is faster than read() and helps flush buffers
            # This ensures we get the latest frame, not a stale buffered one
            is_live_camera = isinstance(selected_source.value, str) and (
                selected_source.value.startswith("DEVICE_") or selected_source.value == "VIRTUAL_CAM"
            )
            is_video_file = selected_source.value == FileSource.VIDEO.name

            # Video file playback controls: scrub (seek on change) then pause (replay last frame)
            if is_video_file:
                self._apply_scrub(cap, source_index)
                if self._get_pause_param(source_index).value and self._last_video_frame[source_index] is not None:
                    ret, frame = True, self._last_video_frame[source_index].copy()

            if frame is None:
                if is_live_camera:
                    # Grab() is non-blocking, retrieve() decodes - this is faster for live sources
                    # Also helps flush any stale frames from the internal buffer
                    ret = cap.grab()
                    if ret:
                        ret, frame = cap.retrieve()
                else:
                    ret, frame = cap.read()

                if not ret:
                    if is_video_file:
                        log.info(f"Video end reached for source {source_index}. Looping back to start.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    else:
                        log.error(f"Source {source_index} '{selected_source.value}' read failed")

                # Cache the latest raw video frame for pause/freeze replay
                if is_video_file and ret and frame is not None:
                    self._last_video_frame[source_index] = frame.copy()

        # Apply effects if frame is successfully read
        if ret and frame is not None:
            # Normalise to mixer resolution before effects so cached wet/prev frames never mismatch
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            wet_frame = self.src_1_wet if source_index == MixerSource.SRC_1 else self.src_2_wet
            prev_frame = self.src_1_prev if source_index == MixerSource.SRC_1 else self.src_2_prev
            count = self.src_1_count if source_index == MixerSource.SRC_1 else self.src_2_count
            effects_manager = self.src_1_effects if source_index == MixerSource.SRC_1 else self.src_2_effects

            if wet_frame is None:
                wet_frame = np.zeros_like(frame)
            if prev_frame is None:
                prev_frame = frame.copy()

            count += 1
            prev_frame, wet_frame = effects_manager.get_frames(frame, wet_frame, prev_frame, count)
            frame = wet_frame

            # Update instance variables
            if source_index == MixerSource.SRC_1:
                self.src_1_wet = wet_frame
                self.src_1_prev = prev_frame
                self.src_1_count = count
            else:
                self.src_2_wet = wet_frame
                self.src_2_prev = prev_frame
                self.src_2_count = count

        return ret, frame


    def start_video(self, source_name, index):
        log.info(f"Requested mixer source {index}: {source_name}")

        # Reset playback state so the new source plays from the current scrub position
        # and the paused freeze-frame from a previous clip is discarded.
        self._last_video_frame[index] = None
        self._scrub_last[index] = None

        if index == MixerSource.SRC_1:
            self.src_1_wet = None
            self.src_1_prev = None
            self.src_1_effects.reset_feedback_buffer()
        elif index == MixerSource.SRC_2:
            self.src_2_wet = None
            self.src_2_prev = None
            self.src_2_effects.reset_feedback_buffer()

        # Check if source is an animation by looking in the animation dictionaries
        is_animation = source_name in self.src_1_animations or source_name in self.src_2_animations
        is_image = source_name == FileSource.IMAGE.name

        if is_animation:
            log.info(f"Starting mixer source {index}: animation {source_name}")
            if index == MixerSource.SRC_1:
                self.cap1 = self._open_animation(self.cap1, source_name, index)
            elif index == MixerSource.SRC_2:
                self.cap2 = self._open_animation(self.cap2, source_name, index)
        elif is_image:
            log.info(f"Starting mixer source {index}: image source")
            if index == MixerSource.SRC_1:
                self.cap1 = self._open_image_source(self.cap1, index)
            elif index == MixerSource.SRC_2:
                self.cap2 = self._open_image_source(self.cap2, index)
        else:  # handle cv2 sources (devices, video files)
            log.info(f"Starting mixer source {index}: cv2 source {source_name}")
            if index == MixerSource.SRC_1:
                self.cap1 = self._open_cv2_capture(self.cap1, source_name, index)
            elif index == MixerSource.SRC_2:
                self.cap2 = self._open_cv2_capture(self.cap2, source_name, index)


    def get_frame(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._process_single_source, MixerSource.SRC_1)
            future2 = executor.submit(self._process_single_source, MixerSource.SRC_2)

            # Wait longer for first frame (sources need initialization time)
            # After that, use aggressive timeout to maintain smooth frame rate
            if not self._first_frame_received:
                FRAME_TIMEOUT = None  # Wait indefinitely for first frame
            else:
                FRAME_TIMEOUT = 0.05  # 50ms timeout for smooth playback

            # Try to get SRC_1 with timeout
            try:
                ret1, frame1 = future1.result(timeout=FRAME_TIMEOUT)
                if frame1 is not None:
                    self._cached_frame1 = frame1  # Cache successful frame
            except concurrent.futures.TimeoutError:
                # Source too slow - use cached frame to maintain smooth playback
                ret1 = True if self._cached_frame1 is not None else False
                frame1 = self._cached_frame1 if self._cached_frame1 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
                log.debug("SRC_1 timeout - using cached frame")

            # Try to get SRC_2 with timeout
            try:
                ret2, frame2 = future2.result(timeout=FRAME_TIMEOUT)
                if frame2 is not None:
                    self._cached_frame2 = frame2  # Cache successful frame
            except concurrent.futures.TimeoutError:
                ret2 = True if self._cached_frame2 is not None else False
                frame2 = self._cached_frame2 if self._cached_frame2 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
                log.debug("SRC_2 timeout - using cached frame")

            # Mark that we've successfully received first frames
            if not self._first_frame_received and ret1 and ret2:
                self._first_frame_received = True
                log.info("First frames received - enabling 50ms timeouts for smooth playback")

        # Fall back to black frames for any failed source rather than returning None,
        # so the virtual cam and other outputs always receive a frame.
        if not ret1 or frame1 is None:
            log.error("Could not retrieve frames from both sources.")
            frame1 = self._cached_frame1 if self._cached_frame1 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.skip = True
        if not ret2 or frame2 is None:
            log.error("Could not retrieve frames from both sources.")
            frame2 = self._cached_frame2 if self._cached_frame2 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.skip = True

        # Ensure frames are the same size for mixing
        if frame1.shape[0] != self.height or frame1.shape[1] != self.width:
             frame1 = cv2.resize(frame1, (self.width, self.height))
        if frame2.shape[0] != self.height or frame2.shape[1] != self.width:
             frame2 = cv2.resize(frame2, (self.width, self.height))

        if self.swap.value == True:
            temp = frame1.copy()
            frame1 = frame2
            frame2 = temp

        if self.blackout:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.freeze and self._frozen_frame is not None:
            return self._frozen_frame.copy()

        if self.blend_mode.value == MixModes.LUMA_KEY.value:
            result = self._luma_key(frame1, frame2)
        elif self.blend_mode.value == MixModes.CHROMA_KEY.value:
            lower = (self.lower_hue.value, self.lower_saturation.value, self.lower_value.value)
            upper = (self.upper_hue.value, self.upper_saturation.value, self.upper_value.value)
            result = self._chroma_key(frame1, frame2, lower=lower, upper=upper)
        else:
            result = self._alpha_blend(frame1, frame2)

        self._frozen_frame = result.copy()
        return result
