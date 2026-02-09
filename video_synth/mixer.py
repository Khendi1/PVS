"""
The Mixer object is used for managing and mixing video sources into a single frame
A single mixed frame is retrieved every iteration of the main loop.
"""

import cv2
import numpy as np
from enum import Enum, IntEnum, auto
from animations import *
import os
from pathlib import Path
from effects import LumaMode, EffectManager
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

        self.post_params = self.post_effects.params

        # cap variables to store video capture objects or animation instances
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip = False

        # search for available video capture devices
        num_devices = self._detect_devices(max_index=num_devices)
        # add devices to sources dict if found
        self.sources = {f"DEVICE_{i}": i for i in range(num_devices+1)}
        i = num_devices+1
        # add file sources to sources dict
        for file_source in FileSource:
            self.sources[file_source.name] = i
            i += 1
        # add animation sources to sources dict
        for anim_source in AnimSource:
            self.sources[anim_source.name] = i  
            i += 1
        
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
            AnimSource.MOIRE.name: Moire(**anim_args),
            AnimSource.SHADERS.name: Shaders(**anim_args),
            AnimSource.STRANGE_ATTRACTOR.name: StrangeAttractor(**anim_args),
            AnimSource.PHYSARUM.name: Physarum(**anim_args),
            AnimSource.DLA.name: DLA(**anim_args),
            AnimSource.CHLADNI.name: Chladni(**anim_args),
            AnimSource.VORONOI.name: Voronoi(**anim_args)
        }

        anim_args["group"] = Groups.SRC_2_ANIMATIONS
        anim_args["params"] = self.src_2_effects.params
        self.src_2_animations = {
            AnimSource.METABALLS.name: Metaballs(**anim_args),
            AnimSource.PLASMA.name: Plasma(**anim_args),
            AnimSource.REACTION_DIFFUSION.name: ReactionDiffusion(**anim_args),
            AnimSource.MOIRE.name: Moire(**anim_args),
            AnimSource.SHADERS.name: Shaders(**anim_args),
            AnimSource.STRANGE_ATTRACTOR.name: StrangeAttractor(**anim_args),
            AnimSource.PHYSARUM.name: Physarum(**anim_args),
            AnimSource.DLA.name: DLA(**anim_args),
            AnimSource.CHLADNI.name: Chladni(**anim_args),
            AnimSource.VORONOI.name: Voronoi(**anim_args)
        }

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
        # safely default to metaballs if no devices found
        default_src1 = "DEVICE_0" if num_devices > 0 else AnimSource.METABALLS.name
        self.selected_source1 = self.post_params.add("source_1",
                                                      min=0, max=len(self.sources), default=default_src1,
                                                      subgroup=subgroup, group=self.group,
                                                      type=Widget.DROPDOWN, options=list(self.sources.keys()))

        # init source 2 to metaballs
        self.selected_source2 = self.post_params.add("source_2",
                                                      min=0, max=len(self.sources), default=AnimSource.METABALLS.name,
                                                      subgroup=subgroup, group=self.group,
                                                      type=Widget.DROPDOWN, options=list(self.sources.keys()))

        # --- Parameters for blending and keying ---

        self.blend_mode = self.post_params.add("blend_mode",
                                               min=0, max=2, default=0,
                                               subgroup=subgroup, group=self.group,
                                               type=Widget.RADIO, options=MixModes)
        self.luma_threshold = self.post_params.add("luma_threshold",
                                                   min=0, max=255, default=128,
                                                   subgroup=subgroup, group=self.group)
        self.luma_selection = self.post_params.add("luma_selection",
                                                   min=LumaMode.WHITE.value, max=LumaMode.BLACK.value, default=LumaMode.WHITE.value,
                                                   subgroup=subgroup, group=self.group,
                                                   type=Widget.RADIO, options=LumaMode)
        self.upper_hue = self.post_params.add("upper_hue",
                                              min=0, max=179, default=80,
                                              subgroup=subgroup, group=self.group)
        self.upper_saturation = self.post_params.add("upper_sat",
                                                     min=0, max=255, default=255,
                                                     subgroup=subgroup, group=self.group)
        self.upper_value = self.post_params.add("upper_val",
                                                min=0, max=255, default=255,
                                                subgroup=subgroup, group=self.group)
        self.lower_hue = self.post_params.add("lower_hue",
                                              min=0, max=179, default=0,
                                              subgroup=subgroup, group=self.group)
        self.lower_saturation = self.post_params.add("lower_sat",
                                                     min=0, max=255, default=100,
                                                     subgroup=subgroup, group=self.group)
        self.lower_value = self.post_params.add("lower_val",
                                                min=0, max=255, default=100,
                                                subgroup=subgroup, group=self.group)
                                                
        self.alpha_blend = self.post_params.add("alpha_blend",
                                                min=0.0, max=1.0, default=0.5,
                                                subgroup=subgroup, group=self.group)
                                                
        self.swap = self.post_params.add("swap_sources",
                                         min=0, max=1, default=0,
                                         subgroup=subgroup, group=self.group,
                                         type=Widget.RADIO, options=Toggle)

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

        # --- Start video sources ---
        self.start_video(self.selected_source1.value, MixerSource.SRC_1)
        self.start_video(self.selected_source2.value, MixerSource.SRC_2)

        # Cached frames for when sources timeout
        self._cached_frame1 = None
        self._cached_frame2 = None
        self._first_frame_received = False


    # find dir one level up from current working directory
    def _find_dir(self, dir_name: str, file_name: str = None):

        script_dir = Path(__file__).parent

        video_path_object = (
            script_dir / ".." / dir_name / file_name
            if file_name is not None
            else script_dir / ".." / dir_name
        )

        return str(video_path_object.resolve().as_posix())


    def _detect_devices(self, max_index: int):
        log.info(f"Attempting to find video capture sources ({max_index})")
        num_success = 0
        for index in range(int(max_index)):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        log.info(f"Found video capture device at index {index}")
                        num_success += 1
                    cap.release()
                else:
                    print('\n\nNone found\n\n')
                    log.warning(f"Failed to find capture device at index {index}")
            except Exception as e:
                log.error(f"Error while checking device at index {index}: {e}")
                # pass
            return num_success


    def _open_cv2_capture(self, cap, source_name, index):
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        log.info(f"Opening cv2 VideoCapture for source_{index}: {source_name}")
        
        source_val = self.sources[source_name]

        if source_name == "VIDEO_FILE":
            file_name = self.video_file_name1 if index == MixerSource.SRC_1 else self.video_file_name2
            source_val = self._find_dir("samples", file_name)
        elif source_name == "IMAGE_FILE":
            file_name = self.image_file_name1 if index == MixerSource.SRC_1 else self.image_file_name2
            source_val = self._find_dir("images", file_name)

        cap = cv2.VideoCapture(source_val)

        # CRITICAL FIX: Reduce camera buffer to 1 frame to minimize latency and prevent stale frames
        # This dramatically improves responsiveness and reduces blocking time
        if isinstance(source_val, int):  # Only for real camera devices (not files)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.live_caps.append(cap)
        self.skip = True

        if not cap.isOpened():
            log.error(f"Could not open live video source {index}.")
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

        cap = self.cap1 if source_index == MixerSource.SRC_1 else self.cap2
        selected_source = self.selected_source1 if source_index == MixerSource.SRC_1 else self.selected_source2
        
        # Read frame
        if not isinstance(cap, cv2.VideoCapture):
            frame = cap.get_frame(frame)
            ret = True
        else:
            # CRITICAL FIX: For live cameras, grab+retrieve is faster than read() and helps flush buffers
            # This ensures we get the latest frame, not a stale buffered one
            is_live_camera = isinstance(selected_source.value, str) and selected_source.value.startswith("DEVICE_")

            if is_live_camera:
                # Grab() is non-blocking, retrieve() decodes - this is faster for live sources
                # Also helps flush any stale frames from the internal buffer
                ret = cap.grab()
                if ret:
                    ret, frame = cap.retrieve()
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

        if is_animation:  # handle animation sources
            log.info(f"Starting mixer source {index}: with animation source name: {source_name}")
            if index == MixerSource.SRC_1:
                self.cap1 = self._open_animation(self.cap1, source_name, index)
            elif index == MixerSource.SRC_2:
                self.cap2 = self._open_animation(self.cap2, source_name, index)
        else:  # handle cv2 sources (devices, video files, image files)
            log.info(f"Starting mixer source {index}: with cv2 source name: {source_name}")
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
