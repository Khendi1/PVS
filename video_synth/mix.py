import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from config import *
from enum import IntEnum
from animations import *

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

class Mixer():
    def __init__(self):

        # cap variables to store video capture objects or animation instances
        # e.g. self.cap1 can be a cv2.VideoCapture or LavaLampSynth instance
        # a frame must be obtained from the capture object or animation instance
        # before the mixer can blend or key between the two sources. The mthod for
        # obtaining a frame depends on the type of source.
        self.cap1 = None
        self.cap2 = None

        self.live_caps = []
        self.skip1 = False
        self.skip2 = False

        self.default_video_file_path = "Big_Buck_Bunny_1080_10s_2MB.mp4"
        self.default_image_file_path = "test_image.jpg"

        # Dictionary of available animation sources. These differ from captured sources 
        # in that they generate frames algorithmically rather than capturing from a device or file.
        self.animation_sources = {
            "Metaballs": LavaLampSynth(width=800, height=600),
            # "Plasma": Plasma(),
            # "Moire": MoirePatternGenerator(size=(800, 600)),
            # "Shader": 
            "ReactionDiffusion": ReactionDiffusionSimulator(800, 600)
        }

        self.video_sources = [
            "Webcam", 
            "HDMI Capture", 
            "Composite Capture", 
            "VGA Capture", 
            "Video File", 
            "Image File"]
        self.video_sources += list(self.animation_sources.keys())
        
        # self.selected_source1 = params.add("source1", 0, len(self.video_sources)-1, 0) #, self.video_sources, callback=self.select_source1_callback)
        # self.selected_source2 = params.add("source2", 0, len(self.video_sources)-1, 1) #, self.video_sources, callback=self.select_source2_callback)
        self.selected_source1 = "Webcam"
        self.selected_source2 = "Metaballs"

        # --- Parameters for blending and keying ---

        # Luma keying threshold and selection mode
        self.luma_threshold = params.add("luma_threshold", 0, 255, 128)  # Threshold for luma keying
        self.luma_selection = params.add("luma_selection", 0, 1, 1)  # 0 for black, 1 for white

        # Chroma key upper and lower HSV bounds
        self.upper_hue = params.add("upper_hue", 0, 179, 80)  
        self.upper_saturation = params.add("upper_saturation", 0, 255, 255)
        self.upper_value = params.add("upper_value", 0, 255, 255)
        self.lower_hue = params.add("lower_hue", 0, 179, 0)  # Lower hue for chroma keying
        self.lower_saturation = params.add("lower_saturation", 0, 255, 100)  # Lower saturation for chroma keying
        self.lower_value = params.add("lower_value", 0, 255, 100)  # Lower value for chroma keying

        # amount to blend the metaball frame wisth the input frame
        self.frame_blend = params.add("frame_blend", 0.0, 1.0, 0.5)


    #TODO: make this more modular
    def start_video(self, source, index):
        """
        Initializes a video capture object based on the source name.
        """
        if source == "Webcam":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(0)
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print("Error: Could not open webcam for source 1.")
                # frame = self.cap1.read()
                skip1 = True
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(0)
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print("Error: Could not open webcam for source 2.")

        elif source == "Composite Capture":
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
        
        elif source == "VGA Capture":
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

        elif source == "HDMI Capture":
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

        elif source == "Metaballs":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Metaballs"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Metaballs"]

        elif source == "Plasma":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Plasma"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Plasma"]

        elif source == "ReactionDiffusion":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["ReactionDiffusion"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["ReactionDiffusion"]
        
        elif source == "Moire":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Moire"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Moire"]
        
        elif source == "Shader":
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = self.animation_sources["Shader"]
            if index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = self.animation_sources["Shader"]

        elif source == "Image File":
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

        elif source == "Video File":
            file_path = dpg.get_value(f"file_path_source_{index}")
            if not file_path:  # If field is empty, use default
                file_path = self.default_video_file_path
                dpg.set_value(f"file_path_source_{index}", file_path)
            
            if index == 1:
                if self.cap1 and not isinstance(self.cap1, LavaLampSynth): # or plasma, reddif,...
                    self.cap1.release()
                self.cap1 = cv2.VideoCapture(file_path)
                self.live_caps.append(self.cap1)
                if not self.cap1.isOpened():
                    print(f"Error: Could not open video file '{file_path}' for source 1.")
            elif index == 2:
                if self.cap2 and not isinstance(self.cap2, LavaLampSynth): # or plasma, reddif,...
                    self.cap2.release()
                self.cap2 = cv2.VideoCapture(file_path)
                self.live_caps.append(self.cap2)
                if not self.cap2.isOpened():
                    print(f"Error: Could not open video file '{file_path}' for source 2.")

    def select_source1_callback(self, sender, app_data):
        global selected_source1
        
        # Abort if the same source is selected
        if app_data == selected_source1 or app_data == selected_source2:
            return
        
        selected_source1 = app_data
        print(f"Source 1 selected: {selected_source1}")
        # Show/hide file path input based on selection
        if app_data == "Video File":
            dpg.show_item("file_path_source_1")
            # If switching to video file, try to load it immediately
            self.start_video(selected_source1, 1)
        else:
            dpg.hide_item("file_path_source_1")
            self.start_video(selected_source1, 1)

    def select_source2_callback(self, sender, app_data):
        """
        Callback for the second dropdown menu.
        """
        global selected_source2
        selected_source2 = app_data
        print(f"Selected source 2: {selected_source2}")
        # Show/hide file path input based on selection
        if app_data == "Video File":
            dpg.show_item("file_path_source_2")
            # If switching to video file, try to load it immediately
            self.start_video(selected_source2, 2)
        else:
            dpg.hide_item("file_path_source_2")
            self.start_video(selected_source2, 2)

    def mix_sources_old(self):
        ret1, frame1 = False, None
        ret2, frame2 = False, None

        # Read from source 1
        if selected_source1 == "Metaballs" and isinstance(self.cap1, LavaLampSynth):
            frame1 = self.cap1.do_metaballs(np.zeros((self.cap1.height, self.cap1.width, 3), dtype=np.uint8))
            ret1 = True
        elif selected_source1 == "Plasma":
            frame1 = generate_plasma_effect(image_width, image_height)
            ret1 = True
        elif selected_source1 == "Webcam" or selected_source1 == "Video File" or selected_source1 == "Capture" and self.cap1 and self.cap1.isOpened():
            ret1, frame1 = self.cap1.read()
            image_height, image_width = frame1.shape[:2]
            if not ret1: # If reading fails, release and try to re-open
                self.cap1.release()
                self.cap1 = cv2.VideoCapture(selected_source1 if selected_source1 == "Webcam" else dpg.get_value("file_path_source_1"))
                if not self.cap1.isOpened(): print("Error: Failed to reopen source 1.")

        # Read from source 2
        if isinstance(self.cap2, LavaLampSynth):
            frame2 = self.cap2.do_metaballs(np.zeros((image_height, image_width, 3), dtype=np.uint8))
            ret2 = True
        elif selected_source2 == "Plasma":
            frame2 = generate_plasma_effect(image_width, image_height)
            ret2 = True
        
        elif selected_source2 in ["Webcam", "Video File", "Capture"] and self.cap2 and self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if not ret2: # If reading fails, release and try to re-open
                self.cap2.release()
                self.cap2 = cv2.VideoCapture(selected_source2 if selected_source2 == "Webcam" else dpg.get_value("file_path_source_2"))
                if not self.cap2.isOpened(): print("Error: Failed to reopen source 2.")

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for blending
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            # Blend the two frames using the alpha value from the slider
            alpha = self.frame_blend.value
            blended_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
            return blended_frame

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

    def mix_sources(self, mode="blend"):
        ret1, frame1 = False, None
        ret2, frame2 = False, None


        # Read from source 1
        if self.selected_source1 == "Metaballs" and isinstance(self.cap1, LavaLampSynth):
            frame1 = self.cap1.do_metaballs(np.zeros((self.cap1.height, self.cap1.width, 3), dtype=np.uint8))
            ret1 = True
        elif self.selected_source1 == "Plasma":
            frame1 = generate_plasma_effect(image_width, image_height)
            ret1 = True
        elif self.selected_source1 == "Webcam" or self.selected_source1 == "Video File" or self.selected_source1 == "Capture" and self.cap1 and self.cap1.isOpened():
            ret1, frame1 = self.cap1.read()
            image_height, image_width = frame1.shape[:2]

            if not ret1: # If reading fails, release and try to re-open
                print(f"Error: Source 1 '{self.selected_source1}' read failed, attempting to reopen.")
                self.cap1.release()
                self.cap1 = cv2.VideoCapture(self.selected_source1 if self.selected_source1 == "Webcam" else dpg.get_value("file_path_source_1"))
                if not self.cap1.isOpened(): print("Error: Failed to reopen source 1.")

        # Read from source 2
        if isinstance(self.cap2, LavaLampSynth):
            frame2 = self.cap2.do_metaballs(np.zeros((600, 800, 3), dtype=np.uint8))
            # frame2 = self.cap2.do_metaballs(np.zeros((image_height, image_width, 3), dtype=np.uint8))
            ret2 = True
        elif self.selected_source2 == "Plasma":
            frame2 = generate_plasma_effect(image_width, image_height)
            ret2 = True
        
        elif self.selected_source2 in ["Webcam", "Video File", "Capture"] and self.cap2 and self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if not ret2: # If reading fails, release and try to re-open
                print(f"Error: Source 2 '{self.selected_source2}' read failed, attempting to reopen.")
                self.cap2.release()
                self.cap2 = cv2.VideoCapture(self.selected_source2 if self.selected_source2 == "Webcam" else dpg.get_value("file_path_source_2"))
                if not self.cap2.isOpened(): print("Error: Failed to reopen source 2.")

        # Process and display frames
        if ret1 and ret2:
            # Ensure frames are the same size for mixing
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

            mixing_modes = {
                "blend": self.blend,
                "luma_key": self.luma_key,
                "chroma_key": self.chroma_key,  # Not yet used, but ready for future extension
            }

            if mode in mixing_modes:
                # For luma_key, you can pass threshold as needed
                if mode == "luma_key":
                    return mixing_modes[mode](frame1, frame2, 
                                              threshold=self.luma_threshold.value, 
                                              white=self.luma_selection.value)
                elif mode == "chroma_key":
                    lower = (self.lower_hue.value, self.lower_saturation.value, self.lower_value.value)
                    upper = (self.upper_hue.value, self.upper_saturation.value, self.upper_value.value)
                    return mixing_modes[mode](frame1, frame2,
                                              lower=lower,
                                              upper=upper)
                else:
                    return mixing_modes[mode](frame1, frame2, )
            else:
                print(f"Unknown mixing mode: {mode}. Defaulting to blend.")
                return self.blend(frame1, frame2)
        else:
            print("Error: Could not retrieve frames from both sources.")
            print( f"Source 1 ret: {ret1}, Source 2 ret: {ret2}" )
            return None