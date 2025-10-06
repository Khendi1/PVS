import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from config import *
from enum import IntEnum
from animations import *

class MixModes(IntEnum):
    BLEND = 0
    LUMA_KEY = 1
    CHROMA_KEY = 2

class MixSources(IntEnum):
    WEBCAM = 0
    HDMI_CAPTURE = 1
    COMPOSITE_CAPTURE = 2
    VGA_CAPTURE = 3
    VIDEO_FILE = 4
    IMAGE_FILE = 5
    METABALLS = 6
    PLASMA = 7
    REACTION_DIFFUSION = 8
    MOIRE = 9
    SHADER = 10  

LIVE_SOURCE_VALUES = [MixSources.WEBCAM.value, MixSources.HDMI_CAPTURE.value, MixSources.COMPOSITE_CAPTURE.value, MixSources.VGA_CAPTURE.value]
FILE_SOURCE_VALUES = [MixSources.VIDEO_FILE.value, MixSources.IMAGE_FILE.value]
ANIMATED_SOURCE_VALUES = [MixSources.METABALLS.value, MixSources.PLASMA.value, MixSources.REACTION_DIFFUSION.value, MixSources.MOIRE.value, MixSources.SHADER]

class Mixer():
    def __init__(self):

        # cap variables to store video capture objects or animation instances
        # e.g. self.cap1 can be a cv2.VideoCapture or LavaLampSynth instance
        # a frame must next be obtained from the capture object or animation instance
        # before the mixer can blend or key between the two sources. 
        # The mthod for obtaining a frame depends on the type of source.
        self.cap1 = None
        self.cap2 = None

        # list to track live cv2.VideoCapture objects for proper release on exit
        self.live_caps = []

        # flags to skip the first frame after starting a new video source
        self.skip1 = False
        self.skip2 = False

        # default file paths for video and image files. The actual path can be changed in the GUI
        self.default_video_file_path = "Big_Buck_Bunny_1080_10s_2MB.mp4"
        self.default_image_file_path = "test_image.jpg"

        # Dictionary of available animation sources. These differ from captured sources 
        # in that they generate frames algorithmically rather than capturing from a device or file.
        self.animation_sources = {
            "Metaballs": LavaLampSynth(width=640, height=480), #800x600 default size
            # "Plasma": Plasma(),
            # "Moire": MoirePatternGenerator(size=(800, 600)),
            # "Shader": 
            "ReactionDiffusion": ReactionDiffusionSimulator(800, 600)
        }

        self.selected_source1 = params.add("source1", 0, len(MixSources)-1, MixSources.WEBCAM.value) 
        self.selected_source2 = params.add("source2", 0, len(MixSources)-1, MixSources.METABALLS.value)

        # --- Parameters for blending and keying ---

        self.blend_mode = params.add("blend_mode", 0, 2, 0)  # 0: blend, 1: luma key, 2: chroma key

        # Luma keying threshold and selection mode
        self.luma_threshold = params.add("luma_threshold", 0, 255, 128)  # Threshold for luma keying
        self.luma_selection = params.add("luma_selection", 0, 1, 1)  # 0 for black, 1 for white

        # Chroma key upper and lower HSV bounds
        self.upper_hue = params.add("upper_hue", 0, 179, 80)  
        self.upper_saturation = params.add("upper_sat", 0, 255, 255)
        self.upper_value = params.add("upper_val", 0, 255, 255)
        self.lower_hue = params.add("lower_hue", 0, 179, 0)  # Lower hue for chroma keying
        self.lower_saturation = params.add("lower_sat", 0, 255, 100)  # Lower saturation for chroma keying
        self.lower_value = params.add("lower_val", 0, 255, 100)  # Lower value for chroma keying

        # amount to blend the metaball frame wisth the input frame
        self.frame_blend = params.add("frame_blend", 0.0, 1.0, 0.5)

        self.start_video(self.selected_source1.value, 1)
        self.start_video(self.selected_source2.value, 2)

    #TODO: make this more modular
    def start_video(self, source, index):
        """
        Initializes a video capture object based on the source name.
        """

        print(f"Starting source {index}: {MixSources(source).name}, source value: {source}")

        if source == MixSources.WEBCAM.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(0)
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print("Error: Could not open webcam for source 1.")
                self.skip1 = True
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(0)
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print("Error: Could not open webcam for source 2.")

        elif source == MixSources.COMPOSITE_CAPTURE.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(2) # May need to adjust index based on system
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print("Error: Could not open capture device for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(2) # May need to adjust index based on system
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print("Error: Could not open capture device for source 2.")
        
        elif source == MixSources.VGA_CAPTURE.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(3) # May need to adjust index based on system
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print("Error: Could not open capture device for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(3) # May need to adjust index based on system
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print("Error: Could not open capture device for source 2.")

        elif source == MixSources.HDMI_CAPTURE.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(1)
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print("Error: Could not open capture device for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(1)
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print("Error: Could not open capture device for source 2.")

        elif source == MixSources.METABALLS.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Metaballs"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Metaballs"]

        elif source == MixSources.PLASMA.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Plasma"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Plasma"]

        elif source == MixSources.REACTION_DIFFUSION.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["ReactionDiffusion"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["ReactionDiffusion"]
        
        elif source == MixSources.MOIRE.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Moire"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Moire"]
        
        elif source == MixSources.SHADER.value:
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Shader"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Shader"]

        elif source == MixSources.IMAGE_FILE.value:
            file_path = dpg.get_value(f"file_path_source_{index}")
            if not file_path:  # If field is empty, use default
                file_path = "test_image.jpg"
                dpg.set_value(f"file_path_source_{index}", file_path)
            
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                # self.cap1 = load_opencv_image_as_texture(file_path)
                if self.cap1 == 0:
                    print(f"Error: Could not load image '{file_path}' for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                # self.cap2 = load_opencv_image_as_texture(file_path)
                if self.cap2 == 0:
                    print(f"Error: Could not load image '{file_path}' for source 2.")

        elif source == MixSources.VIDEO_FILE.value:
            file_path = dpg.get_value(f"file_path_source_{index}")
            if not file_path:  # If field is empty, use default
                file_path = self.default_video_file_path
                dpg.set_value(f"file_path_source_{index}", file_path)
            print(f"Loading video file '{file_path}' for source {index}.")
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(file_path)

                new_width = 640
                new_height = 360

                # Attempt to set the properties
                self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
                self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)


                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print(f"Error: Could not open video file '{file_path}' for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(file_path)
                new_width = 640
                new_height = 360

                # Attempt to set the properties
                self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
                self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print(f"Error: Could not open video file '{file_path}' for source 2.")

        else:
            print(f"Error: Unknown source '{source}' for source {index}.")

    def select_source1_callback(self, sender, app_data):

        print(f"app_data: {app_data}/{MixSources[app_data].value}, selected_source1: {self.selected_source1.value}")
        # Abort if the same source is selected
        if app_data == self.selected_source1.value or app_data == self.selected_source2.value:
            return
        
        self.selected_source1.value = MixSources[app_data].value
        print(f"Source 1 selected: {MixSources(self.selected_source1.value).name}")
        # Show/hide file path input based on selection
        if app_data == MixSources.VIDEO_FILE.name:
            dpg.show_item("file_path_source_1")
            # If switching to video file, try to load it immediately
            self.start_video(self.selected_source1.value, 1)
        else:
            dpg.hide_item("file_path_source_1")
            self.start_video(self.selected_source1.value, 1)

    def select_source2_callback(self, sender, app_data):
        """
        Callback for the second dropdown menu.
        """
        self.selected_source2.value = MixSources[app_data].value
        print(app_data)
        print(f"Selected source 2: {MixSources(self.selected_source2.value).name}")
        # Show/hide file path input based on selection
        if app_data == MixSources.VIDEO_FILE.name:
            dpg.show_item("file_path_source_2")
            # If switching to video file, try to load it immediately
            self.start_video(self.selected_source2.value, 2)
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
        if self.selected_source1.value == MixSources.METABALLS.value and isinstance(self.cap1, LavaLampSynth):
            frame1 = self.cap1.do_metaballs(np.zeros((self.cap1.height, self.cap1.width, 3), dtype=np.uint8))
            ret1 = True
        elif self.selected_source1.value == MixSources.PLASMA.value:
            frame1 = self.cap1.generate_plasma_effect(image_width, image_height)
            ret1 = True
        elif self.selected_source1.value in (LIVE_SOURCE_VALUES + FILE_SOURCE_VALUES) and self.cap1 and self.cap1.isOpened():
            ret1, frame1 = self.cap1.read()
            image_height, image_width = frame1.shape[:2]

            if not ret1: # If reading fails, release and try to re-open
                print(f"Error: Source 1 '{self.selected_source1.value}' read failed, attempting to reopen.")
                self.cap1.release()
                self.cap1 = cv2.VideoCapture(self.selected_source1.value if self.selected_source1.value == "Webcam" else dpg.get_value("file_path_source_1"))
                if not self.cap1.isOpened(): print("Error: Failed to reopen source 1.")

        # Read from source 2
        if isinstance(self.cap2, LavaLampSynth):
            frame2 = self.cap2.do_metaballs(np.zeros((600, 800, 3), dtype=np.uint8))
            # frame2 = self.cap2.do_metaballs(np.zeros((image_height, image_width, 3), dtype=np.uint8))
            ret2 = True
        elif self.selected_source2.value == MixSources.PLASMA.value:
            frame2 = self.cap2.generate_plasma_effect(image_width, image_height)
            ret2 = True
        elif self.selected_source2.value in (LIVE_SOURCE_VALUES + FILE_SOURCE_VALUES) and self.cap2 and self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if not ret2: # If reading fails, release and try to re-open
                print(f"Error: Source 2 '{self.selected_source2.value}' read failed, attempting to reopen.")
                self.cap2.release()
                self.cap2 = cv2.VideoCapture(self.selected_source2.value if self.selected_source2.value == MixSources.WEBCAM.value else dpg.get_value("file_path_source_2"))
                if not self.cap2.isOpened(): print("Error: Failed to reopen source 2.")

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            # print(f"mode: {mode}, luma_threshold: {self.luma_threshold.value}, luma_selection: {self.luma_selection.value}")

            # For luma_key, you can pass threshold as needed
            if self.blend_mode.value == MixModes.LUMA_KEY.value:
                return self.luma_key(frame1, frame2, 
                                    threshold=self.luma_threshold.value,
                                    white=self.luma_selection.value)
            elif self.blend_mode.value == MixModes.CHROMA_KEY.value:
                lower = (self.lower_hue.value, self.lower_saturation.value, self.lower_value.value)
                upper = (self.upper_hue.value, self.upper_saturation.value, self.upper_value.value)
                return self.chroma_key(frame1, frame2, lower=lower, upper=upper)

            else:
                return self.blend(frame1, frame2)
        else:
            print("Error: Could not retrieve frames from both sources.")
            print( f"Source 1 {MixSources(self.selected_source1.value).name} ret: {ret1},\n"
                  f"Source 2 {MixSources(self.selected_source2.value).name} ret: {ret2}" )
            #TODO: implement fallback source or error handling 
            return None