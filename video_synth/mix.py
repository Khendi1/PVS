import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from enum import IntEnum, Enum, auto
from animations import *
import os
from pathlib import Path
from gui_elements import TrackbarRow


class MixModes(IntEnum):
    BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2

""" 
The MixSources Enum class is used to standardize stings 

For cv2 sources (devices, images, video files), there is an A and B enum
so that different they can be used on source 1 and source 2 simultaneously

Note that if you want to mix two video files, the sources must be set to 
"""
class MixSources(Enum):
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

    def __init__(self, params):

        self.params = params

        # cap variables to store video capture objects or animation instances
        # e.g. self.cap1 can be a cv2.VideoCapture or Metaballs instance
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip1 = False
        self.skip2 = False

        # --- Configure sources ---
        self.sources = {}   # dict for storing device/animation name and index

        # add valid cv2 video device indicies to source dict
        self.cv2_max_devices = 10
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
        self.luma_selection = params.add("luma_selection", 0, 1, 1)

        # Chroma key upper and lower HSV bounds
        self.upper_hue = params.add("upper_hue", 0, 179, 80)
        self.upper_saturation = params.add("upper_sat", 0, 255, 255)
        self.upper_value = params.add("upper_val", 0, 255, 255)
        self.lower_hue = params.add("lower_hue", 0, 179, 0)  
        self.lower_saturation = params.add("lower_sat", 0, 255, 100)
        self.lower_value = params.add("lower_val", 0, 255, 100)

        # amount to blend the metaball frame wisth the input frame
        self.frame_blend = params.add("frame_blend", 0.0, 1.0, 0.5)

        # a frame must next be obtained from the capture object or animation instance
        # before the mixer can blend or key between the two sources.
        self.start_video(self.selected_source1.value, 1)
        self.start_video(self.selected_source2.value, 2)


    def find_dir(self, dir_name: str, file_name: str = None):

        script_dir = Path(__file__).parent

        video_path_object = (
            script_dir / ".." / dir_name / file_name
            if file_name is not None
            else script_dir / ".." / dir_name
        )

        return str(video_path_object.resolve().as_posix())


    def detect_devices(self, max_index):
        for index in range(max_index):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)

            # Try to read a frame to confirm the device is open and working
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.sources[f'{MixSources.DEVICE_1.name}_{index}'] = index
                    self.sources[f'{MixSources.DEVICE_2.name}_{index}'] = index
                cap.release()


    def failback_camera(self):
        # TODO: implement a fallback camera source if the selected source fails, move to mixer class
        pass


    def open_cv2_capture(self, cap, source, index):
        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        # Initialize new capture
        if source == self.sources[MixSources.VIDEO_FILE_1.name] or source == self.sources[MixSources.VIDEO_FILE_2.name]:
            if index == 1:
                source = self.find_dir("samples", self.video_file_name1)
            else:
                source = self.find_dir("samples", self.video_file_name2)
        elif source == self.sources[MixSources.IMAGE_FILE_1.name] or source == self.sources[MixSources.IMAGE_FILE_2.name]:
            if index == 1:
                source = self.find_dir("images", self.image_file_name1)
            else:
                source = self.find_dir("images", self.image_file_name2)

        cap = cv2.VideoCapture(source)

        # Add to list of live captures for proper release on exit
        self.live_caps.append(cap)

        # Skip the first frame to allow camera to adjust
        self.skip1 = True

        if not cap.isOpened():
            print("Error: Could not open live video source.")
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
            print(
                f"Starting mixer source {index}: with cv2 source value: {source}"
            )
            if index == 1:
                self.cap1 = self.open_cv2_capture(self.cap1, source, 1)
            elif index == 2:
                self.cap2 = self.open_cv2_capture(self.cap2, source, 2)
        else:  # handle animation sources
            print(
                f"Starting mixer source {index}: with animation source value: {source}"
            )
            if index == 1:
                self.cap1 = self.open_animation(self.cap1, source)
            if index == 2:
                self.cap2 = self.open_animation(self.cap2, source)


    def select_source1_callback(self, sender, app_data):

        print(
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
        
        print(
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
        alpha = self.frame_blend.value
        return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)


    def luma_key(self, frame1, frame2, threshold=128, white=True):
        """Mixes two frames using luma keying (brightness threshold)."""
        # Convert frame1 to grayscale to get luma
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Check if we want to key out white or black areas
        if white:
            # Where mask is white, use frame1; else use frame2
            result = np.where(mask == 255, frame1, frame2)
        else:
            # Where mask is black, use frame1; else use frame2
            result = np.where(mask == 0, frame2, frame1)
        return result


    def chroma_key(self, frame1, frame2, lower=(0, 100, 0), upper=(80, 255, 80)):
        """Placeholder for chroma keying (green screen)."""
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
                print(
                    f"Error: Source 1 '{self.selected_source1.value}' read failed"
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
                print(
                    f"Error: Source 2 '{self.selected_source2.value}' read failed, attempting to reopen."
                )
                # self.cap2.release()
                # self.cap2 = self.failback_camera()

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            # For luma_key, you can pass threshold as needed
            if self.blend_mode.value == MixModes.LUMA_KEY.value:
                return self.luma_key(
                    frame1,
                    frame2,
                    threshold=self.luma_threshold.value,
                    white=self.luma_selection.value,
                )
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
            print("Error: Could not retrieve frames from both sources.")
            return None


    def mix_panel(self):
        with dpg.collapsing_header(label=f"\tMixer", tag="mixer"):

            # Get list of srcs without DEVICE_2, X_FILE_2
            sources = [src for src in self.sources.keys() if 'E_2' not in src]
            
            dpg.add_text("Video Source 1")
            dpg.add_combo(sources, default_value="DEVICE_1_0", tag="source_1", callback=self.select_source1_callback)
            dpg.add_combo(self.video_samples+self.images, default_value=self.default_video_file_path, tag="source_1_file", callback=self.select_source1_file)
            # dpg.add_input_text(label="Video File Path 1", tag="file_path_source_1", default_value=mixer.default_video_file_path)
            
            sources = [src for src in self.sources.keys() if 'E_1' not in src]

            dpg.add_text("Video Source 2")
            dpg.add_combo(sources, default_value="METABALLS_ANIM", tag="source_2", callback=self.select_source2_callback)
            # dpg.add_input_text(label="Video File Path 2", tag="file_path_source_2", default_value=mixer.default_video_file_path)
            dpg.add_combo(self.video_samples+self.images, default_value=self.default_video_file_path, tag="source_2_file", callback=self.select_source2_file)
            
            dpg.add_spacer(height=10)

            dpg.add_text("Mixer")
            # dpg.add_slider_float(label="Blending", default_value=alpha, min_value=0.0, max_value=1.0, callback=alpha_callback, format="%.2f")
            
            blend_mode_slider = TrackbarRow("Blend Mode", self.params.get("blend_mode"), None)
            
            frame_blend_slider = TrackbarRow(
                "Frame Blend",
                self.params.get("frame_blend"),
                None) # fix defulat font_id=None
            
            upper_hue_key_slider = TrackbarRow(
                "Upper Hue Key",
                self.params.get("upper_hue"),
                None) # fix defulat font_id=None
            
            lower_hue_key_slider = TrackbarRow(
                "Lower Hue Key",
                self.params.get("lower_hue"),
                None) # fix defulat font_id=None
            
            upper_sat_slider = TrackbarRow(
                "Upper Sat Key",
                self.params.get("upper_sat"),
                None) # fix defulat font_id=None
            
            lower_sat_slider = TrackbarRow(
                "Lower Sat Key",
                self.params.get("lower_sat"),
                None) # fix defulat font_id=None
            
            upper_val_slider = TrackbarRow(
                "Upper Val Key",
                self.params.get("upper_val"),
                None) # fix defulat font_id=None
            
            lower_val_slider = TrackbarRow(
                "Lower Val Key",
                self.params.get("lower_val"),
                None) # fix defulat font_id=None
            
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