"""
The Mixer object is used for managing and mixing video sources into a single frame
A single mixed frame is retrieved every iteration of the main loop.
"""

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from enum import IntEnum, Enum, auto
from animations import *
import os
from pathlib import Path
from effects import LumaMode, EffectManager
from param import ParamTable
from luma import *
from config import ParentClass, SourceIndex, WidgetType
import concurrent.futures # Added for parallel processing

log = logging.getLogger(__name__)

class MixModes(IntEnum):
    ALPHA_BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2

class MixSources(Enum):
    """ 
    The MixSources Enum class is used to standardize strings 

    For cv2 sources (devices, images, video files), there is an A and B enum
    so that different they can be used on source 1 and source 2 simultaneously

    Note that if you want to mix two video files, the sources must be set to 
    """
    DEVICE_1 = 0
    DEVICE_2 = auto()
    VIDEO_FILE_1 = auto()
    VIDEO_FILE_2 = auto()
    IMAGE_FILE_1 = auto()
    IMAGE_FILE_2 = auto()
    METABALLS_ANIM = auto()
    PLASMA_ANIM = auto()
    REACTION_DIFFUSION_ANIM = auto()
    MOIRE_ANIM = auto()
    SHADER_ANIM = auto()

DEVICE_SOURCE_NAMES = [member for member in MixSources if "DEVICE" in member.name]
FILE_SOURCE_NAMES = [member for member in MixSources if "FILE" in member.name]
ANIMATED_SOURCE_NAMES = [member for member in MixSources if "ANIM" in member.name]

class Mixer:
    """Mixer is used to init sources, get frames from each, and blend them"""


    def __init__(self, effects, num_devices, width=640, height=480):

        self.parent = ParentClass.MIXER
        subclass = self.__class__.__name__

        self.src_1_effects = effects[SourceIndex.SRC_1]
        self.src_2_effects = effects[SourceIndex.SRC_2]
        self.post_effects = effects[SourceIndex.POST]

        self.post_params = self.post_effects.params
        self.post_toggles = self.post_effects.toggles
        
        self.width = width
        self.height = height

        # cap variables to store video capture objects or animation instances
        # e.g. self.cap1 can be a cv2.VideoCapture or Metaballs instance
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip = False

        self.sources = {}   # dict for storing device/animation name and index

        # add valid cv2 video device indicies to source dict
        self.cv2_max_devices = num_devices
        self.detect_devices(max_index=self.cv2_max_devices)

        # file source indicies begin at cv2_max_devices+1
        for src in FILE_SOURCE_NAMES:
            self.cv2_max_devices += 1
            self.sources[src.name] = self.cv2_max_devices

        # animation source indicies begin at cv2_max_devices+4
        i = 0
        for src in ANIMATED_SOURCE_NAMES:
            i+=1
            self.sources[src.name] = self.cv2_max_devices+i

# --------------------------------------------------------------------
        parent = ParentClass.SRC_1_ANIMATIONS
        self.src_1_animations = {
            MixSources.METABALLS_ANIM.name: Metaballs(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.PLASMA_ANIM.name: Plasma(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.REACTION_DIFFUSION_ANIM.name: ReactionDiffusion(self.src_1_effects.params, self.src_1_effects.toggles, self.width, self.height, parent=parent),
            MixSources.MOIRE_ANIM.name: Moire(self.src_1_effects.params, self.src_1_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.SHADER_ANIM.name: ShaderVisualizer(self.src_1_effects.params, self.src_1_effects.toggles, self.width, self.height, parent=parent)
        }

        parent = ParentClass.SRC_2_ANIMATIONS
        self.src_2_animations = {
            MixSources.METABALLS_ANIM.name: Metaballs(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.PLASMA_ANIM.name: Plasma(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.REACTION_DIFFUSION_ANIM.name: ReactionDiffusion(self.src_2_effects.params, self.src_2_effects.toggles, self.width, self.height, parent=parent),
            MixSources.MOIRE_ANIM.name: Moire(self.src_2_effects.params, self.src_2_effects.toggles, width=self.width, height=self.height, parent=parent),
            MixSources.SHADER_ANIM.name: ShaderVisualizer(self.src_2_effects.params, self.src_2_effects.toggles, self.width, self.height, parent=parent)
        }
# --------------------------------------------------------------------

        self.device_sources = [k for k,v in self.sources.items() if v <= self.cv2_max_devices-(len(FILE_SOURCE_NAMES)-1)]

        # --- Configure file sources ---
        self.video_samples = os.listdir(self.find_dir("samples"))
        self.images = os.listdir(self.find_dir("images"))

        # default file paths for video and image files. The actual path can be changed in the GUI
        self.default_image_file_path = self.images[0] if len(self.images) > 0 else None

        self.video_file_name1 = self.video_samples[0] if len(self.video_samples) > 0 else None
        self.video_file_name2 = self.video_samples[0] if len(self.video_samples) > 0 else None

        self.image_file_name1 = self.default_image_file_path
        self.image_file_name2 = self.default_image_file_path

        # --- Source Params ---

        # initialize source 1 to use the first hardware device available (probably webcam if on laptop)
        self.selected_source1 = self.post_params.add(
            "source1", 0, max(self.sources.values()), self.sources[self.device_sources[0]], subclass, self.parent, WidgetType.DROPDOWN, list(self.sources.keys())
        )
        # init source 2 to metaballs
        self.selected_source2 = self.post_params.add(
            "source2", 0, max(self.sources.values()), self.sources[MixSources.METABALLS_ANIM.name], subclass, self.parent, WidgetType.DROPDOWN, list(self.sources.keys())
        )

        # --- Parameters for blending and keying ---

        self.blend_mode = self.post_params.add("blend_mode", 0, 2, 0, subclass, self.parent)

        # Luma keying threshold and selection mode
        self.luma_threshold = self.post_params.add("luma_threshold", 0, 255, 128, subclass, self.parent)
        self.luma_selection = self.post_params.add("luma_selection", LumaMode.WHITE.value, 
                                         LumaMode.BLACK.value, LumaMode.WHITE.value, subclass, self.parent)

        # Chroma key upper and lower HSV bounds
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

        # a frame must next be obtained from the capture object or animation instance
        # before the mixer can blend or key between the two sources.
        self.start_video(self.selected_source1.value, 1)
        self.start_video(self.selected_source2.value, 2)
        
        # show source k-v pairs for debugging
        log.debug(f'Sources: {self.sources}')

    # find dir one level up from current working directory
    def find_dir(self, dir_name: str, file_name: str = None):

        script_dir = Path(__file__).parent

        video_path_object = (
            script_dir / ".." / dir_name / file_name
            if file_name is not None
            else script_dir / ".." / dir_name
        )

        return str(video_path_object.resolve().as_posix())


    def detect_devices(self, max_index):

        log.info(f"Attempting to find video capture sources ({max_index})")
        for index in range(max_index):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)

                # Try to read a frame to confirm the device is open and working
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        log.info(f"Found video capture device at index {index}")
                        self.sources[f'{MixSources.DEVICE_1.name}_{index}'] = index
                        self.sources[f'{MixSources.DEVICE_2.name}_{index}'] = index
                    cap.release()
                else:
                    log.warning(f"Failed to find capture device at index {index}")
            except Exception as e:
                pass


    def failback_camera(self):
        # TODO: implement a fallback camera source if the selected source fails, move to mixer class
        pass


    def open_cv2_capture(self, cap, source, index):
        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        # Initialize new capture
        if source == self.sources[MixSources.VIDEO_FILE_1.name] or source == self.sources[MixSources.VIDEO_FILE_2.name]:
            file_name = self.video_file_name1 if index == 1 else self.video_file_name2
            source = self.find_dir("samples", file_name)
        elif source == self.sources[MixSources.IMAGE_FILE_1.name] or source == self.sources[MixSources.IMAGE_FILE_2.name]:
            file_name = self.image_file_name1 if index == 1 else self.image_file_name2
            source = self.find_dir("images", file_name)

        cap = cv2.VideoCapture(source)

        # Add to list of live captures for proper release on exit
        self.live_caps.append(cap)

        # Skip the first frame to allow camera to adjust
        self.skip = True

        if not cap.isOpened():
            log.error(f"Could not open live video source {index}.")
            cap = self.failback_camera

        return cap


    def open_animation(self, cap, source_val, index):

        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        animations = self.src_1_animations if index == 1 else self.src_2_animations
        
        source = None
        # get key using value since all animation values are unique
        for k,v in self.sources.items():
            if source_val == v:
                return animations[k]


    def start_video(self, source, index):
        """
        Initializes a video capture object based on the source value.
        """

        # handle live sources (webcams, capture cards, files)
        if source <= self.cv2_max_devices:
            log.info(
                f"Starting mixer source {index}: with cv2 source value: {source}"
            )
            if index == 1:
                self.cap1 = self.open_cv2_capture(self.cap1, source, index)
            elif index == 2:
                self.cap2 = self.open_cv2_capture(self.cap2, source, index)
        else:  # handle animation sources
            log.info(
                f"Starting mixer source {index}: with animation source value: {source}"
            )
            if index == 1:
                self.cap1 = self.open_animation(self.cap1, source, index)
            if index == 2:
                self.cap2 = self.open_animation(self.cap2, source, index)

    def blend(self, frame1, frame2):
        alpha = self.alpha_blend.value

        return cv2.addWeighted(frame1.astype(np.float32), 1-alpha, frame2.astype(np.float32), alpha, 0)


    def _luma_key(self, frame1, frame2):
        return luma_key(frame1, frame2, self.luma_selection.value, self.luma_threshold.value).astype(np.float32)


    def chroma_key(self, frame1, frame2, lower=(0, 100, 0), upper=(80, 255, 80)):
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask == 255, frame2, frame1)
        return result.astype(np.float32)

    def _process_single_source(self, source_index):
        ret, frame = False, None
        
        cap = self.cap1 if source_index == 1 else self.cap2
        selected_source = self.selected_source1 if source_index == 1 else self.selected_source2
        video_file_name = self.video_file_name1 if source_index == 1 else self.video_file_name2
        image_file_name = self.image_file_name1 if source_index == 1 else self.image_file_name2

        # Read frame
        if not isinstance(cap, cv2.VideoCapture):
            frame = cap.get_frame()
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                if selected_source.value == self.sources.get(MixSources.VIDEO_FILE_1.name if source_index == 1 else MixSources.VIDEO_FILE_2.name):
                    log.info(f"Video end reached for source {source_index}. Looping back to start.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                else:
                    log.error(f"Source {source_index} '{selected_source.value}' read failed")
        
        # Apply effects if frame is successfully read
        if ret and frame is not None:
            wet_frame = self.src_1_wet if source_index == 1 else self.src_2_wet
            prev_frame = self.src_1_prev if source_index == 1 else self.src_2_prev
            count = self.src_1_count if source_index == 1 else self.src_2_count
            effects_manager = self.src_1_effects if source_index == 1 else self.src_2_effects

            if wet_frame is None:
                wet_frame = np.zeros_like(frame)
            if prev_frame is None:
                prev_frame = frame.copy()
            
            count += 1
            prev_frame, wet_frame = effects_manager.modify_frames(
                frame, wet_frame, prev_frame, count
            )
            frame = wet_frame
            
            # Update instance variables
            if source_index == 1:
                self.src_1_wet = wet_frame
                self.src_1_prev = prev_frame
                self.src_1_count = count
            else:
                self.src_2_wet = wet_frame
                self.src_2_prev = prev_frame
                self.src_2_count = count
                
        return ret, frame


    def get_mixed_frame(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._process_single_source, 1)
            future2 = executor.submit(self._process_single_source, 2)

            ret1, frame1 = future1.result()
            ret2, frame2 = future2.result()

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            # Use self.width and self.height which are initialized once
            if frame1.shape[0] != self.height or frame1.shape[1] != self.width:
                 frame1 = cv2.resize(frame1, (self.width, self.height))
            if frame2.shape[0] != self.height or frame2.shape[1] != self.width:
                 frame2 = cv2.resize(frame2, (self.width, self.height))

            if self.swap.value == True:
                temp = frame1.copy()
                frame1 = frame2
                frame2 = temp

            # For luma_key, you can pass threshold as needed
            if self.blend_mode.value == MixModes.LUMA_KEY.value:
                return self._luma_key(frame1,frame2)
            elif self.blend_mode.value == MixModes.CHROMA_KEY.value:
                lower = (
                    self.lower_hue.value,
                    self.lower_saturation.value,
                    self.lower_value.value,
                )
                upper = (
                    self.upper_hue.value,
                    self.upper_saturation.value,
                    self.upper_value.value,
                )
                return self.chroma_key(frame1, frame2, lower=lower, upper=upper)

            else:
                return self.blend(frame1, frame2)
        else:
            log.error("Could not retrieve frames from both sources.")
            return None