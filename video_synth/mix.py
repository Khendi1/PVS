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
from gui_elements import TrackbarRow, RadioButtonRow, Toggle
from effects import LumaMode


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


    def __init__(self, params, toggles, num_devices):

        self.params = params

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

        # Dictionary of available animation sources. These differ from captured sources
        # in that they generate frames algorithmically rather than capturing from a device or file.
        self.animation_sources = {
            MixSources.METABALLS_ANIM.name: Metaballs(params, width=640, height=480),
            MixSources.PLASMA_ANIM.name: Plasma(params, width=640, height=480),
            MixSources.REACTION_DIFFUSION_ANIM.name: ReactionDiffusionSimulator(params, 640, 480),
        }

        self.device_sources = [k for k,v in self.sources.items() if v <= self.cv2_max_devices-(len(FILE_SOURCE_NAMES)-1)]
        # self.file_sources = 

        # --- Configure file sources ---
        self.video_samples = os.listdir(self.find_dir("samples"))
        self.images = os.listdir(self.find_dir("images"))

        # default file paths for video and image files. The actual path can be changed in the GUI
        self.default_video_file_path = self.video_samples[0] if len(self.video_samples) > 0 else None
        self.default_image_file_path = self.images[0] if len(self.images) > 0 else None

        self.video_file_name1 = self.default_video_file_path
        self.video_file_name2 = self.default_video_file_path

        self.image_file_name1 = self.default_image_file_path
        self.image_file_name2 = self.default_image_file_path

        # --- Source Params ---

        # initialize source 1 to use the first hardware device available (probably webcam if on laptop)
        self.selected_source1 = params.add(
            "source1", 0, max(self.sources.values()) - 1, self.sources[self.device_sources[0]]
        )
        # init source 2 to metaballs
        self.selected_source2 = params.add(
            "source2", 0, max(self.sources.values()) - 1, self.sources[MixSources.METABALLS_ANIM.name]
        )

        # --- Parameters for blending and keying ---

        self.blend_mode = params.add("blend_mode", 0, 2, 0)

        # Luma keying threshold and selection mode
        self.luma_threshold = params.add("luma_threshold", 0, 255, 128)
        self.luma_selection = params.add("luma_selection", 0, 1, 0)

        # Chroma key upper and lower HSV bounds
        self.upper_hue = params.add("upper_hue", 0, 179, 80)
        self.upper_saturation = params.add("upper_sat", 0, 255, 255)
        self.upper_value = params.add("upper_val", 0, 255, 255)
        self.lower_hue = params.add("lower_hue", 0, 179, 0)  
        self.lower_saturation = params.add("lower_sat", 0, 255, 100)
        self.lower_value = params.add("lower_val", 0, 255, 100)

        self.alpha_blend = params.add("alpha_blend", 0.0, 1.0, 0.5)

        self.swap = toggles.add("Swap Frames", "swap", False)

        # a frame must next be obtained from the capture object or animation instance
        # before the mixer can blend or key between the two sources.
        self.start_video(self.selected_source1.value, 1)
        self.start_video(self.selected_source2.value, 2)

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


    def open_animation(self, cap, source_val):

        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()
        
        source = None
        # get key using value since all animation values are unique
        for k,v in self.sources.items():
            if source_val == v:
                return self.animation_sources[k]


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
                self.cap1 = self.open_cv2_capture(self.cap1, source, 1)
            elif index == 2:
                self.cap2 = self.open_cv2_capture(self.cap2, source, 2)
        else:  # handle animation sources
            log.info(
                f"Starting mixer source {index}: with animation source value: {source}"
            )
            if index == 1:
                self.cap1 = self.open_animation(self.cap1, source)
            if index == 2:
                self.cap2 = self.open_animation(self.cap2, source)


    def select_source1_callback(self, sender, app_data):

        log.info(
            f"source1 callback app_data: {app_data}/{self.sources[app_data]}, \
            selected_source1: {self.selected_source1.value}"
        )

        source_index = self.sources[app_data]
    
        # Abort if the same source is selected
        if (
            source_index == self.selected_source1.value
            or source_index == self.selected_source2.value
        ):
            return

        self.selected_source1.value = source_index
        
        self.start_video(self.selected_source1.value, 1)


    def select_source2_callback(self, sender, app_data):
        
        log.info(
            f"source2 callback app_data: {app_data}/{self.sources[app_data]}, \
            selected_source2: {self.selected_source2.value}"
        )

        source_index = self.sources[app_data]
    
        # Abort if the same source is selected
        if (
            source_index == self.selected_source1.value
            or source_index == self.selected_source2.value
        ):
            return

        self.selected_source2.value = source_index
        
        self.start_video(self.selected_source2.value, 2)


    def blend(self, frame1, frame2):
        alpha = self.alpha_blend.value
        return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)


    def luma_key(self, frame1, frame2):
        """Mixes two frames using luma keying (brightness threshold)."""

        # The mask will determine which parts of the current frame are kept
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mask = None
        if self.luma_selection.value == LumaMode.BLACK.value:
            # Use THRESH_BINARY_INV to key out DARK areas (Luma is low)
            # Pixels with Luma < threshold become white (255) in the mask, meaning they are KEPT
            ret, mask = cv2.threshold(
                gray, self.luma_threshold.value, 255, cv2.THRESH_BINARY_INV
            )
        elif self.luma_selection.value == LumaMode.WHITE.value:
            ret, mask = cv2.threshold(
                gray, self.luma_threshold.value, 255, cv2.THRESH_BINARY
            )

        # Keep the keyed-out (bright) parts of the current frame
        fg = cv2.bitwise_and(frame1, frame1, mask=mask)

        # Invert the mask to find the areas *not* keyed out (the dark areas)
        mask_inv = cv2.bitwise_not(mask)

        # Use the inverted mask to "cut a hole" in the previous frame
        bg = cv2.bitwise_and(frame2, frame2, mask=mask_inv)

        # Combine the new foreground (fg) with the previous frame's background (bg)
        return cv2.add(fg, bg)


    def chroma_key(self, frame1, frame2, lower=(0, 100, 0), upper=(80, 255, 80)):
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask == 255, frame2, frame1)
        return result


    def get_frame(self):
        ret1, frame1 = False, None
        ret2, frame2 = False, None

        # Read from source 1
        if not isinstance(self.cap1, cv2.VideoCapture):
            frame1 = self.cap1.get_frame(frame1)
            ret1 = True
        else:
            ret1, frame1 = self.cap1.read()
            # TODO: retry a couple times to let devices connect before changing src
            if not ret1:
                log.error(
                    f"Source 1 '{self.selected_source1.value}' read failed"
                )
                # self.cap1.release()
                # self.cap1 = self.failback_camera()

        # Read from source 2
        if not isinstance(self.cap2, cv2.VideoCapture):
            frame2 = self.cap2.get_frame(frame2)
            ret2 = True
        else:
            ret2, frame2 = self.cap2.read()
            if not ret2:
                log.error(
                    f"Source 2 '{self.selected_source2.value}' read failed, attempting to reopen."
                )
                # self.cap2.release()
                # self.cap2 = self.failback_camera()

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            if self.swap.value == True:
                temp = frame1.copy()
                frame1 = frame2
                frame2 = temp

            # For luma_key, you can pass threshold as needed
            if self.blend_mode.value == MixModes.LUMA_KEY.value:
                return self.luma_key(frame1,frame2)
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


    def get_hsv_for_cv2(self, dpg_rgba_value):
        """Converts Dear PyGui's [R, G, B, A] (0.0-1.0) to OpenCV's [H, S, V] (0-180, 0-255, 0-255)."""
        
        r, g, b, a = dpg_rgba_value
        bgr_float = np.array([b, g, r])
        bgr_255 = (bgr_float * 255).astype(np.uint8)
        bgr_image = bgr_255.reshape(1, 1, 3)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_image[0, 0]

        return h, s, v


    def color_picker_callback(self, sender, app_data, user_data):
        """Callback to read RGBA from the color picker and convert it to cv2-compatible HSV."""
        rgba = dpg.get_value(sender)

        # Convert DPG's RGBA to cv2's HSV
        h, s, v = self.get_hsv_for_cv2(rgba)

        log.info(f"Color Picker: {user_data}: H={h}, S={s}, V={v}")
        if 'upper' in user_data:
            self.upper_hue.value = h
            self.upper_saturation.value = s
            self.upper_value.value = v
        elif 'lower' in user_data:
            self.lower_hue.value = h
            self.lower_saturation.value = s
            self.lower_value.value = v

    def _blend_mode_select_callback(self, sender, app_data, user_data):
        if MixModes.ALPHA_BLEND.name in app_data:
            self.blend_mode.value = 0
            dpg.configure_item("alpha_blend", show=True)
            dpg.configure_item("alpha_blend_reset", show=False)            

            dpg.configure_item("upper_chroma", show=False)
            dpg.configure_item("lower_chroma", show=False)
            dpg.configure_item("luma_threshold", show=False)
            dpg.configure_item("luma_threshold_reset", show=False)
            dpg.configure_item("luma_selection", show=False)
            dpg.configure_item("luma_selection_reset", show=False)
        elif MixModes.LUMA_KEY.name in app_data:
            self.blend_mode.value = MixModes.LUMA_KEY.value
            dpg.configure_item("luma_threshold", show=True)
            dpg.configure_item("luma_threshold_reset", show=True)
            dpg.configure_item("luma_selection", show=True)
            dpg.configure_item("luma_selection_reset", show=True)

            dpg.configure_item("alpha_blend", show=False)
            dpg.configure_item("alpha_blend_reset", show=False)                        
            dpg.configure_item("upper_chroma", show=False)
            dpg.configure_item("lower_chroma", show=False)
        elif MixModes.CHROMA_KEY.name in app_data:
            self.blend_mode.value = MixModes.CHROMA_KEY.value
            dpg.configure_item("upper_chroma", show=True)
            dpg.configure_item("lower_chroma", show=True)
            
            dpg.configure_item("alpha_blend", show=False)
            dpg.configure_item("alpha_blend_reset", show=False)            
            dpg.configure_item("luma_threshold", show=False)
            dpg.configure_item("luma_threshold_reset", show=False)
            dpg.configure_item("luma_selection", show=False)
            dpg.configure_item("luma_selection_reset", show=False)

    def _file_select_callback(self, sender, app_data, user_data):
        file_name, file_extension = os.path.splitext(app_data['file_path_name'])


        if "1" in user_data:
            self.video_file_name1 = file_name
            if ".mp4" in file_extension:
                pass
            else:
                pass
        elif "2" in user_data:
            self.video_file_name2 = file_name
            if ".mp4" in file_extension:
                pass
            else:
                pass

    def create_gui_panel(self, theme):
        with dpg.collapsing_header(label=f"\tMixer", tag="mixer") as h:
            dpg.bind_item_theme(h, theme)
            
            # Get list of srcs without DEVICE_2, X_FILE_2
            sources = [src for src in self.sources.keys() if 'E_2' not in src]
            
            dpg.add_button(
                label=self.swap.label,
                callback=self.swap.toggle,
                parent="mixer",
            )

            with dpg.group(horizontal=True):
                dpg.add_text("Source 1")
                dpg.add_combo(sources, default_value="DEVICE_1_0", tag="source_1", callback=self.select_source1_callback)
                with dpg.file_dialog(label="Source File Dialog 1", width=300, height=400, show=False, callback=lambda s, a, u : print(s, a, u), tag="filedialog1", user_data="filedialog1"):
                    # dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                    dpg.add_file_extension(".mp4", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".jpeg", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".jpg", color=(255, 255, 0, 255))

                dpg.add_button(label="File 1", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))

            # dpg.add_combo(self.video_samples+self.images, default_value=self.default_video_file_path, tag="source_1_file", callback=self.select_source1_file)
            # dpg.add_input_text(label="Video File Path 1", tag="file_path_source_1", default_value=mixer.default_video_file_path)
            
            #get only sources for source 2; hacky shortcut to avoid selecting 'device_1' and 'x_file_1'
            sources = [src for src in self.sources.keys() if 'E_1' not in src]

            with dpg.group(horizontal=True):
                dpg.add_text("Source 2")
                dpg.add_combo(sources, default_value="METABALLS_ANIM", tag="source_2", callback=self.select_source2_callback)
                with dpg.file_dialog(label="Source File Dialog 2", width=300, height=400, show=False, callback=lambda s, a, u : print(s, a, u), tag="filedialog2", user_data="filedialog2"):
                    # dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                    dpg.add_file_extension(".mp4", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".jpeg", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".jpg", color=(255, 255, 0, 255))

                dpg.add_button(label="File 2", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))


            RadioButtonRow(
                "Blend Mode",
                MixModes,
                self.blend_mode,
                None,
                callback = self._blend_mode_select_callback
            )

            TrackbarRow(
                "Alpha Blend",
                self.params.get("alpha_blend"),
                None
            )
            
            with dpg.group(horizontal=True):
                dpg.add_color_picker(
                    (255, 0, 255, 200), 
                    label="Upper Chroma", 
                    width=200, 
                    tag='upper_chroma', 
                    user_data='upper_chroma', 
                    callback=self.color_picker_callback, 
                    display_hex=False, 
                    display_rgb=False, 
                    display_hsv=True
                )
                dpg.add_color_picker(
                    (255, 0, 255, 200), 
                    label="Lower Chroma", 
                    width=200, 
                    tag='lower_chroma', 
                    user_data='lower_chroma', 
                    callback=self.color_picker_callback, 
                    display_hex=False, 
                    display_rgb=False, 
                    display_hsv=True
                )
            
            luma_threshold_slider = TrackbarRow(
                "Luma Threshold",
                self.params.get("luma_threshold"),
                None) # fix defulat font_id=None
            
            luma_selection_slider = TrackbarRow(
                "Luma Selection",
                self.params.get("luma_selection"),
                None) # fix defulat font_id=None
            
    def select_source1_file(self, sender, app_data):
        self.video_file_name1 = app_data

    def select_source2_file(self, sender, app_data):
        self.video_file_name2 = app_data