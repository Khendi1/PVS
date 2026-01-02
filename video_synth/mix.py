"""
The Mixer object is used for managing and mixing video sources into a single frame
A single mixed frame is retrieved every iteration of the main loop.
"""

import cv2
import numpy as np
from enum import IntEnum, auto
from animations import *
import os
from pathlib import Path
from effects import LumaMode, EffectManager
from luma import *
from config import ParentClass, SourceIndex, WidgetType
import concurrent.futures 


log = logging.getLogger(__name__)


class MixModes(IntEnum):
    ALPHA_BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2


class Mixer:
    """Mixer is used to init sources, get frames from each, and blend them"""

    def __init__(self, effects, num_devices, width=640, height=480):

        self.parent = ParentClass.MIXER
        subclass = self.__class__.__name__

        self.width = width
        self.height = height

        self.src_1_effects = effects[SourceIndex.SRC_1]
        self.src_2_effects = effects[SourceIndex.SRC_2]
        self.post_effects = effects[SourceIndex.POST]

        self.post_params = self.post_effects.params
        self.post_toggles = self.post_effects.toggles

        # cap variables to store video capture objects or animation instances
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip = False

        self.sources = {
            "VIDEO_FILE": "VIDEO_FILE",
            "IMAGE_FILE": "IMAGE_FILE",
            "METABALLS_ANIM": "METABALLS_ANIM",
            "PLASMA_ANIM": "PLASMA_ANIM",
            "REACTION_DIFFUSION_ANIM": "REACTION_DIFFUSION_ANIM",
            "MOIRE_ANIM": "MOIRE_ANIM",
            "SHADER_ANIM": "SHADER_ANIM",
        }
        self.sources.update({f"DEVICE_{i}": i for i in range(num_devices)})
        
        log.info(f"Sources: {self.sources}")

        self._detect_devices(max_index=num_devices)

        # --------------------------------------------------------------------
        parent = ParentClass.SRC_1_ANIMATIONS
        self.src_1_animations = {
            "METABALLS_ANIM": Metaballs(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            "PLASMA_ANIM": Plasma(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            "REACTION_DIFFUSION_ANIM": ReactionDiffusion(self.src_1_effects.params, self.src_1_effects.toggles, self.width, self.height, parent=parent),
            "MOIRE_ANIM": Moire(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            "SHADER_ANIM": ShaderVisualizer(self.src_1_effects.params, self.src_1_effects.toggles, self.width, self.height, parent=parent)
        }

        parent = ParentClass.SRC_2_ANIMATIONS
        self.src_2_animations = {
            "METABALLS_ANIM": Metaballs(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            "PLASMA_ANIM": Plasma(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            "REACTION_DIFFUSION_ANIM": ReactionDiffusion(self.src_2_effects.params, self.src_2_effects.toggles, self.width, self.height, parent=parent),
            "MOIRE_ANIM": Moire(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            "SHADER_ANIM": ShaderVisualizer(self.src_2_effects.params, self.src_2_effects.toggles, self.width, self.height, parent=parent)
        }
        # --------------------------------------------------------------------

        # --- Configure file sources ---
        self.video_samples = os.listdir(self._find_dir("samples"))
        self.video_file_name1 = self.video_samples[0] if len(self.video_samples) > 0 else None
        self.video_file_name2 = self.video_samples[0] if len(self.video_samples) > 0 else None

        self.images = os.listdir(self._find_dir("images"))
        self.default_image_file_path = self.images[0] if len(self.images) > 0 else None
        self.image_file_name1 = self.default_image_file_path
        self.image_file_name2 = self.default_image_file_path

        # --- Source Params ---

        # initialize source 1 to use the first hardware device available
        self.selected_source1 = self.post_params.add(
            "source_1", 0, 0, "DEVICE_0", subclass, self.parent, 
            WidgetType.DROPDOWN, list(self.sources.keys())
        )

        # init source 2 to metaballs
        self.selected_source2 = self.post_params.add(
            "source_2", 0, 0, "METABALLS_ANIM", subclass, self.parent, 
            WidgetType.DROPDOWN, list(self.sources.keys())
        )

        # --- Parameters for blending and keying ---

        self.blend_mode = self.post_params.add("blend_mode", 0, 2, 0, subclass, self.parent, WidgetType.RADIO, MixModes)
        self.luma_threshold = self.post_params.add("luma_threshold", 0, 255, 128, subclass, self.parent)
        self.luma_selection = self.post_params.add("luma_selection", LumaMode.WHITE.value, LumaMode.BLACK.value, LumaMode.WHITE.value, subclass, self.parent, WidgetType.RADIO, LumaMode)
        self.upper_hue = self.post_params.add("upper_hue", 0, 179, 80, subclass, self.parent)
        self.upper_saturation = self.post_params.add("upper_sat", 0, 255, 255, subclass, self.parent)
        self.upper_value = self.post_params.add("upper_val", 0, 255, 255, subclass, self.parent)
        self.lower_hue = self.post_params.add("lower_hue", 0, 179, 0, subclass, self.parent)  
        self.lower_saturation = self.post_params.add("lower_sat", 0, 255, 100, subclass, self.parent)
        self.lower_value = self.post_params.add("lower_val", 0, 255, 100, subclass, self.parent)
        self.alpha_blend = self.post_params.add("alpha_blend", 0.0, 1.0, 0.5, subclass, self.parent)
        self.swap = self.post_toggles.add("Swap Frames", "swap", False)

        self.src_1_prev = None
        self.src_2_prev = None
        self.post_prev = None

        self.src_1_wet = None
        self.src_2_wet = None
        self.post_wet = None

        self.src_1_count = 0
        self.src_2_count = 0
        self.post_count = 0

        self.start_video(self.selected_source1.value, SourceIndex.SRC_1)
        self.start_video(self.selected_source2.value, SourceIndex.SRC_2)

    # find dir one level up from current working directory
    def _find_dir(self, dir_name: str, file_name: str = None):

        script_dir = Path(__file__).parent

        video_path_object = (
            script_dir / ".." / dir_name / file_name
            if file_name is not None
            else script_dir / ".." / dir_name
        )

        return str(video_path_object.resolve().as_posix())


    def _detect_devices(self, max_index):
        log.info(f"Attempting to find video capture sources ({max_index})")
        for index in range(max_index):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        log.info(f"Found video capture device at index {index}")
                        self.sources[f"DEVICE_{index}"] = index
                    cap.release()
                else:
                    log.warning(f"Failed to find capture device at index {index}")
            except Exception as e:
                pass


    def _failback_source(self):
        # TODO: implement a fallback camera source if the selected source fails, move to mixer class
        pass


    def _open_cv2_capture(self, cap, source_name, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        log.info(f"Opening cv2 VideoCapture for source_{index}: {source_name}")
        
        source_val = self.sources[source_name]

        if source_name == "VIDEO_FILE":
            file_name = self.video_file_name1 if index == 1 else self.video_file_name2
            source_val = self._find_dir("samples", file_name)
        elif source_name == "IMAGE_FILE":
            file_name = self.image_file_name1 if index == 1 else self.image_file_name2
            source_val = self._find_dir("images", file_name)

        cap = cv2.VideoCapture(source_val)
        self.live_caps.append(cap)
        self.skip = True

        if not cap.isOpened():
            log.error(f"Could not open live video source {index}.")
            # cap = self._failback_source()
        return cap


    def _open_animation(self, cap, source_name, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()
        animations = self.src_1_animations if index == 1 else self.src_2_animations
        return animations[source_name]


    def _alpha_blend(self, frame1, frame2):
        alpha = self.alpha_blend.value

        return cv2.addWeighted(
            frame1.astype(np.float32), 1 - alpha, frame2.astype(np.float32), alpha, 0
        )


    def _luma_key(self, frame1, frame2):
        return luma_key(
            frame1, frame2, self.luma_selection.value, self.luma_threshold.value
        ).astype(np.float32)


    def _chroma_key(self, frame1, frame2, lower=(0, 100, 0), upper=(80, 255, 80)):
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask == 255, frame2, frame1)
        return result.astype(np.float32)


    def _process_single_source(self, source_index):
        ret, frame = False, None

        cap = self.cap1 if source_index == SourceIndex.SRC_1 else self.cap2
        selected_source = self.selected_source1 if source_index == SourceIndex.SRC_1 else self.selected_source2
        
        # Read frame
        if not isinstance(cap, cv2.VideoCapture):
            frame = cap.get_frame(frame)
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                if selected_source.value == "VIDEO_FILE":
                    log.info(f"Video end reached for source {source_index}. Looping back to start.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                else:
                    log.error(f"Source {source_index} '{selected_source.value}' read failed")

        # Apply effects if frame is successfully read
        if ret and frame is not None:
            wet_frame = self.src_1_wet if source_index == SourceIndex.SRC_1 else self.src_2_wet
            prev_frame = self.src_1_prev if source_index == SourceIndex.SRC_1 else self.src_2_prev
            count = self.src_1_count if source_index == SourceIndex.SRC_1 else self.src_2_count
            effects_manager = self.src_1_effects if source_index == SourceIndex.SRC_1 else self.src_2_effects

            if wet_frame is None:
                wet_frame = np.zeros_like(frame)
            if prev_frame is None:
                prev_frame = frame.copy()

            count += 1
            prev_frame, wet_frame = effects_manager.modify_frames(frame, wet_frame, prev_frame, count)
            frame = wet_frame

            # Update instance variables
            if source_index == SourceIndex.SRC_1:
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

        if "ANIM" not in source_name:  # handle cv2 sources
            log.info(f"Starting mixer source {index}: with cv2 source name: {source_name}")
            if index == SourceIndex.SRC_1:
                self.cap1 = self._open_cv2_capture(self.cap1, source_name, index)
            elif index == SourceIndex.SRC_2:
                self.cap2 = self._open_cv2_capture(self.cap2, source_name, index)
        else:  # handle animation sources
            log.info(f"Starting mixer source {index}: with animation source name: {source_name}")
            if index == SourceIndex.SRC_1:
                self.cap1 = self._open_animation(self.cap1, source_name, index)
            elif index == SourceIndex.SRC_2:
                self.cap2 = self._open_animation(self.cap2, source_name, index)


    def get_mixed_frame(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._process_single_source, SourceIndex.SRC_1)
            future2 = executor.submit(self._process_single_source, SourceIndex.SRC_2)

            ret1, frame1 = future1.result()
            ret2, frame2 = future2.result()

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            if frame1.shape[0] != self.height or frame1.shape[1] != self.width:
                 frame1 = cv2.resize(frame1, (self.width, self.height))
            if frame2.shape[0] != self.height or frame2.shape[1] != self.width:
                 frame2 = cv2.resize(frame2, (self.width, self.height))

            if self.swap.value == True:
                temp = frame1.copy()
                frame1 = frame2
                frame2 = temp

            if self.blend_mode.value == MixModes.LUMA_KEY.value:
                return self._luma_key(frame1,frame2)
            elif self.blend_mode.value == MixModes.CHROMA_KEY.value:
                lower = (self.lower_hue.value, self.lower_saturation.value, self.lower_value.value)
                upper = (self.upper_hue.value, self.upper_saturation.value, self.upper_value.value)
                return self._chroma_key(frame1, frame2, lower=lower, upper=upper)
            else:
                return self._alpha_blend(frame1, frame2)
        else:
            log.error("Could not retrieve frames from both sources.")
            return None
