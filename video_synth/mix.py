import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from config import *
from enum import IntEnum, Enum, auto
from animations import *
import os
from pathlib import Path


class MixModes(IntEnum):
    BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2


class MixSources(Enum):
    DEVICE = 0
    VIDEO_FILE = auto()
    IMAGE_FILE = auto()
    METABALLS = auto()
    PLASMA = auto()
    REACTION_DIFFUSION = auto()
    MOIRE = auto()
    SHADER = auto()


FILE_SOURCES = [MixSources.VIDEO_FILE, MixSources.IMAGE_FILE]

ANIMATED_SOURCES = [
    MixSources.METABALLS,
    MixSources.PLASMA,
    MixSources.REACTION_DIFFUSION,
    MixSources.MOIRE,
    MixSources.SHADER,
]


class Mixer:
    def __init__(self):

        # cap variables to store video capture objects or animation instances
        # e.g. self.cap1 can be a cv2.VideoCapture or Metaballs instance
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip1 = False
        self.skip2 = False

        # dict for storing device/animation name and index
        self.sources = {}

        # add valid cv2 video device indicies to source dict
        self.cv2_max_devices = 10
        self.detect_devices(max_index=self.cv2_max_devices)

        # file source indicies begin at cv2_max_devices+1
        self.cv2_max_devices+=1
        self.sources[MixSources.VIDEO_FILE.name] = self.cv2_max_devices
        self.cv2_max_devices+=1
        self.sources[MixSources.IMAGE_FILE.name] = self.cv2_max_devices

        # animation source indicies begin at cv2_max_devices+1
        i = 0
        for src in ANIMATED_SOURCES:
            i+=1
            self.sources[src.name] = self.cv2_max_devices+i

        # default file paths for video and image files. The actual path can be changed in the GUI
        self.default_video_file_path = str(
            self.find_samples_dir("Big_Buck_Bunny_1080_10s_2MB.mp4")
        )
        self.default_image_file_path = "test_image.jpg"

        self.video_file_name = self.default_video_file_path
        self.image_file_name = self.default_image_file_path

        # Dictionary of available animation sources. These differ from captured sources
        # in that they generate frames algorithmically rather than capturing from a device or file.
        self.animation_sources = {
            MixSources.METABALLS.name: Metaballs(width=640, height=480),
            MixSources.PLASMA.name: Plasma(width=640, height=480),
            MixSources.REACTION_DIFFUSION.name: ReactionDiffusionSimulator(640, 480),
        }


        self.device_sources = [k for k,v in self.sources.items() if v <= self.cv2_max_devices-2]
        self.file_sources = [self.sources[MixSources.VIDEO_FILE.name], self.sources[MixSources.IMAGE_FILE.name]]



        self.selected_source1 = params.add(
            "source1", 0, max(self.sources.values()) - 1, self.sources[self.device_sources[0]]
        )
        self.selected_source2 = params.add(
            "source2", 0, max(self.sources.values()) - 1, self.sources[MixSources.METABALLS.name]
        )

        # --- Parameters for blending and keying ---

        self.blend_mode = params.add(
            "blend_mode", 0, 2, 0
        )  # 0: blend, 1: luma key, 2: chroma key

        # Luma keying threshold and selection mode
        self.luma_threshold = params.add(
            "luma_threshold", 0, 255, 128
        )  # Threshold for luma keying
        self.luma_selection = params.add(
            "luma_selection", 0, 1, 1
        )  # 0 for black, 1 for white

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

    def find_samples_dir(self, file_name: str = None):

        script_dir = Path(__file__).parent

        video_path_object = (
            script_dir / ".." / "samples" / file_name
            if file_name is not None
            else script_dir / ".." / "samples"
        )

        return str(video_path_object.resolve().as_posix())

    def detect_devices(self, max_index):
        for index in range(max_index):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)

            # Try to read a frame to confirm the device is open and working
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.sources[f'{MixSources.DEVICE.name}_{index}'] = index
                cap.release()

    def list_sample_files(self):
        # TODO: list files in sample dir
        pass

    def failback_camera(self):
        # TODO: implement a fallback camera source if the selected source fails, move to mixer class
        pass

    def open_cv2_capture(self, cap, source):
        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()

        # Initialize new capture
        if source == self.sources[MixSources.VIDEO_FILE.name]:
            source = self.video_file_name  # video file name should be set in callback
        elif source == self.sources[MixSources.IMAGE_FILE.name]:
            source = self.image_file_name
        
        cap = cv2.VideoCapture(source)

        # if not os.path.exists(source):
        #     print(f"File not found: {source}")
        #     return


        # Add to list of live captures for proper release on exit
        self.live_caps.append(cap)

        # Skip the first frame to allow camera to adjust
        self.skip1 = True

        if not cap.isOpened():
            print("Error: Could not open live video source for source 1.")
            # TODO: Handle error as needed (e.g., fallback to default source)

        return cap

    def open_animation(self, cap, source_val):
        # Release previous capture if it exists and is not an animation
        if cap and isinstance(cap, cv2.VideoCapture):
            cap.release()
        
        source = None
        for k,v in self.sources.items():
            if source_val == v:
                return self.animation_sources[k]


    def start_video(self, source, index):
        """
        Initializes a video capture object based on the source value.
        """

        print(
            f"Starting source {index}: source value: {source}"
        )

        # handle live sources (webcams, capture cards, files)
        if source <= self.cv2_max_devices:
            if index == 1:
                self.cap1 = self.open_cv2_capture(self.cap1, source)
            elif index == 2:
                self.cap2 = self.open_cv2_capture(self.cap2, source)
        else:  # handle animation sources
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

        # Show/hide file path input based on selection
        if app_data == MixSources.VIDEO_FILE.name:
            dpg.show_item("file_path_source_1")
        else:
            dpg.hide_item("file_path_source_1")
        
        self.start_video(self.selected_source1.value, 1)

    def select_source2_callback(self, sender, app_data):
        
        print(
            f"source2 callback app_data: {app_data}/{self.sources[app_data]}, \
            selected_source2: {self.selected_source2.value}"
        )

        source_index = self.sources[app_data]
        print("source index", source_index)
    
        # Abort if the same source is selected
        if (
            source_index == self.selected_source1.value
            or source_index == self.selected_source2.value
        ):
            return

        self.selected_source2.value = source_index

        # Show/hide file path input based on selection
        if app_data == MixSources.VIDEO_FILE.name:
            dpg.show_item("file_path_source_2")
        else:
            dpg.hide_item("file_path_source_2")
        
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

    def mix_sources(self):
        ret1, frame1 = False, None
        ret2, frame2 = False, None

        # Read from source 1
        if not isinstance(self.cap1, cv2.VideoCapture):
            frame1 = self.cap1.get_frame(frame1)
            ret1 = True
        else:
            ret1, frame1 = self.cap1.read()
            if not ret1:  # If reading fails, release and try to re-open
                print(
                    f"Error: Source 1 '{self.selected_source1.value}' read failed, attempting to reopen."
                )
                self.cap1.release()
                self.cap1 = self.failback_camera()

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
                self.cap2.release()
                self.cap2 = self.failback_camera()

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            # print(f"mode: {mode}, luma_threshold: {self.luma_threshold.value}, luma_selection: {self.luma_selection.value}")

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
            # print(
            #     f"Source 1 {MixSources(self.selected_source1.value).name} ret: {ret1},\n"
            #     f"Source 2 {MixSources(self.selected_source2.value).name} ret: {ret2}"
            # )
            # TODO: implement fallback source or error handling
            return None
