from config import *
from gui_elements import TrackbarRow, TrackbarCallback
import dearpygui.dearpygui as dpg
from generators import PerlinNoise, Interp, Oscillator
from save import SaveButtons
from buttons import Button
import random
import datetime
import yaml

class Interface:

    def __init__(self, panel_width=550, panel_height=420):
        self.sliders = []
        self.buttons = []
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.slider_dict = None
        self.default_font_id = None
        self.global_font_id = None
        # TODO: debug automatically building params sliders
        # only creates last param in list
        # self.panels = self.build_panels_dict(params.all())

    def reset_slider_callback(self, sender, app_data, user_data):
        param = params.get(str(user_data))
        if param is None:
            print(f"Slider or param not found for {user_data}")
            return
        print(f"Got reset callback for {user_data}; setting to default value {param.default_val}")
        param.reset()
        dpg.set_value(user_data, param.value)

    def on_toggle_button_click(self, sender, app_data, user_data):
        print(f"test: {user_data}")
        for tag, button in toggles.items():
            if user_data == tag:
                print("test2")
                toggles.toggle(tag)

    def on_button_click(self, sender, app_data, user_data):
        print(f"Button clicked: {user_data}, {app_data}, {sender}")
        # Perform action based on button click
        if user_data == "save":
            self.on_save_button_click()
        elif user_data == "reset_all":
            self.reset_values()
        elif user_data == "random":
            self.randomize_values()
        elif user_data == "enable_polar_transform":
            if enable_polar_transform == True:
                enable_polar_transform = False
            else:
                enable_polar_transform = True
        
    def reset_values(self):
        for param in params.values:
            param.set(param.default_val)
            dpg.set_value(param.name, param.value)

    def randomize_values(self):
        for param in params.all():
            param.randomize()
            dpg.set_value(param.name, param.value)

    def create_buttons(self, width, height):
        reset_button = Button("Reset all", 'reset_all')
        random_button = Button("Random", 'random')

        width -= 20

        with dpg.group(horizontal=True):
            dpg.add_button(label=reset_button.label, callback=self.on_button_click, user_data=reset_button.tag, width=width//3)
            dpg.add_button(label=random_button.label, tag=random_button.tag, callback=self.on_button_click, user_data=random_button.tag, width=width//3)
            toggles["effects_first"].create()

    def resize_buttons(self, sender, app_data):
        # Get the current width of the window
        window_width = dpg.get_item_width("Controls")
        
        # Set each button to half the window width (minus a small padding if you want)
        half_width = window_width // 2
        dpg.set_item_width(sender, half_width)

    def listbox_cb(self, sender, app_data):
        """
        Callback function for the listbox.  Prints the selected items.

        Args:
            sender: The sender of the event (the listbox).
            app_data: A dictionary containing the selected items.  For a listbox,
                    it's  { 'items': [index1, index2, ...] }
        """
        print("Sender:", sender)
        print("App Data:", app_data)
        for i in range(NUM_OSCILLATORS):
            if f"osc{i}" in sender:
                param = None
                for tag, param in params.items():
                    if tag == app_data:
                        osc_bank[i].linked_param = param
                        break

    def create_control_window(self, width=550, height=600):

        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars(width, height)
            # self.create_trackbar_panels_for_param()
            # self.save_buttons = SaveButtons(width, height).create_save_buttons()
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)

    def create_trackbars(self, width, height):

        with dpg.font_registry():
            global_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 18) # Larger font size for the header
            default_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14) # Default font size for other items
        dpg.bind_font(default_font_id)

        with dpg.collapsing_header(label=f"\tHSV", tag="hsv"):
            
            hue_slider = TrackbarRow(
                "Hue Shift", 
                params["hue_shift"], 
                TrackbarCallback(params.get("hue_shift"), "hue_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            sat_slider = TrackbarRow(
                "Sat Shift", 
                params.get("sat_shift"), 
                TrackbarCallback(params.get("sat_shift"), "sat_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            val_slider = TrackbarRow(
                "Val Shift", 
                params.get("val_shift"), 
                TrackbarCallback(params.get("val_shift"), "val_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            contrast_slider = TrackbarRow(
                "Contrast", 
                params.get("contrast"), 
                TrackbarCallback(params.get("contrast"), "contrast").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            brightness_slider = TrackbarRow(
                "Brighness", 
                params.get("brightness"), 
                TrackbarCallback(params.get("brightness"), "brightness").__call__, 
                self.reset_slider_callback, 
                default_font_id)

            val_threshold_slider = TrackbarRow(
                "Val Threshold", 
                params.get("val_threshold"), 
                TrackbarCallback(params.get("val_threshold"), "val_threshold").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            val_hue_shift_slider = TrackbarRow(
                "Hue Shift for Val", 
                params.get("val_hue_shift"), 
                TrackbarCallback(params.get("val_hue_shift"), "val_hue_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)      
            
            hue_invert_angle_slider = TrackbarRow(
                "Hue Invert Angle",
                params.get("hue_invert_angle"),
                TrackbarCallback(params.get("hue_invert_angle"), "hue_invert_angle").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            hue_invert_strength_slider = TrackbarRow(
                "Hue Invert Strength",
                params.get("hue_invert_strength"),
                TrackbarCallback(params.get("hue_invert_strength"), "hue_invert_strength").__call__,
                self.reset_slider_callback,
                default_font_id)
        dpg.bind_item_font("hsv", global_font_id)
    
        with dpg.collapsing_header(label=f"\tEffects", tag="effects"):
            temporal_filter_slider = TrackbarRow(
                "Temporal Filter", 
                params.get("temporal_filter"), 
                TrackbarCallback(params.get("temporal_filter"), "temporal_filter").__call__, 
                self.reset_slider_callback, 
                default_font_id)
        
            alpha_slider = TrackbarRow(
                "Feedback", 
                params.get("alpha"), 
                TrackbarCallback(params.get("alpha"), "alpha").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            blur_kernel_slider = TrackbarRow(
                "Blur Kernel", 
                params.get("blur_kernel_size"), 
                TrackbarCallback(params.get("blur_kernel_size"), "blur_kernel_size").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            blur_type_slider = TrackbarRow(
                "Blur Type",
                params.get("blur_type"),
                TrackbarCallback(params.get("blur_type"), "blur_type").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            # noise_intensity_slider = TrackbarRow(
            #     "Noise Intensity",
            #     params.get("noise_intensity"),
            #     TrackbarCallback(params.get("noise_intensity"), "noise_intensity").__call__,
            #     self.reset_slider_callback,
            #     default_font_id)

            # noise_type_slider = TrackbarRow(
            #     "Noise Type",
            #     params.get("noise_type"),
            #     TrackbarCallback(params.get("noise_type"), "noise_type").__call__,
            #     self.reset_slider_callback,
            #     default_font_id)

            num_glitches_slider = TrackbarRow(
                "Glitch Qty", 
                params.get("num_glitches"), 
                TrackbarCallback(params.get("num_glitches"), "num_glitches").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            glitch_size_slider = TrackbarRow(
                "Glitch Size", 
                params.get("glitch_size"), 
                TrackbarCallback(params.get("glitch_size"), "glitch_size").__call__, 
                self.reset_slider_callback, 
                default_font_id)
        dpg.bind_item_font("effects", global_font_id)

        with dpg.collapsing_header(label=f"\tKeying", tag="keying"):
            key_upper_hue_slider = TrackbarRow(
                "Key Upper Hue",
                params.get("key_upper_hue"),
                TrackbarCallback(params.get("key_upper_hue"), "key_upper_hue").__call__,
                self.reset_slider_callback,
                default_font_id)
            key_lower_hue_slider = TrackbarRow(
                "Key Lower Hue",
                params.get("key_lower_hue"),
                TrackbarCallback(params.get("key_lower_hue"), "key_lower_hue").__call__,
                self.reset_slider_callback,
                default_font_id)
            key_upper_sat_slider = TrackbarRow(
                "Key Upper Sat",
                params.get("key_upper_sat"),
                TrackbarCallback(params.get("key_upper_sat"), "key_upper_sat").__call__,
                self.reset_slider_callback,
                default_font_id)
            key_lower_sat_slider = TrackbarRow(
                "Key Lower Sat",
                params.get("key_lower_sat"),
                TrackbarCallback(params.get("key_lower_sat"), "key_lower_sat").__call__,
                self.reset_slider_callback,
                default_font_id)
            key_upper_val_slider = TrackbarRow(
                "Key Upper Val",
                params.get("key_upper_val"),
                TrackbarCallback(params.get("key_upper_val"), "key_upper_val").__call__,
                self.reset_slider_callback,
                default_font_id)
            key_lower_val_slider = TrackbarRow(
                "Key Lower Val",
                params.get("key_lower_val"),
                TrackbarCallback(params.get("key_lower_val"), "key_lower_val").__call__,
                self.reset_slider_callback,
                default_font_id)
        dpg.bind_item_font("keying", global_font_id)

        with dpg.collapsing_header(label=f"\tPan", tag="pan"):
            
            x_shift_slider = TrackbarRow(
                "X Shift", 
                params.get("x_shift"), 
                TrackbarCallback(params.get("x_shift"), "x_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)         

            y_shift_slider = TrackbarRow(
                "Y Shift", 
                params.get("y_shift"), 
                TrackbarCallback(params.get("y_shift"), "y_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            r_shift_slider = TrackbarRow(
                "R Shift", 
                params.get("r_shift"), 
                TrackbarCallback(params.get("r_shift"), "r_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            zoom_slider = TrackbarRow(
                "Zoom",
                params.get("zoom"),
                TrackbarCallback(params.get("zoom"), "zoom").__call__,
                self.reset_slider_callback,
                default_font_id)

            enable_polar_transform_button = Button("Enable Polar Transform", "enable_polar_transform")
            dpg.add_button(label=enable_polar_transform_button.label, tag="enable_polar_transform", callback=self.on_button_click, user_data=enable_polar_transform_button.tag, width=width)
            dpg.bind_item_font(enable_polar_transform_button.tag, default_font_id)
            polar_x_slider = TrackbarRow(
                "Polar Center X", 
                params.get("polar_x"), 
                TrackbarCallback(params.get("polar_x"), "polar_x").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            polar_y_slider = TrackbarRow(
                "Polar Center Y", 
                params.get("polar_y"), 
                TrackbarCallback(params.get("polar_y"), "polar_y").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            polar_radius_slider = TrackbarRow(
                "Polar radius", 
                params.get("polar_radius"), 
                TrackbarCallback(params.get("polar_radius"), "polar_radius").__call__, 
                self.reset_slider_callback, 
                default_font_id)
        dpg.bind_item_font("pan", global_font_id)

        with dpg.collapsing_header(label=f"\tSync", tag="sync"):
                x_sync_speed_slider = TrackbarRow(
                    "X Sync Speed",
                    params.get("x_sync_speed"),
                    TrackbarCallback(params.get("x_sync_speed"), "x_sync_speed").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_sync_freq_slider = TrackbarRow(
                    "X Sync Freq",
                    params.get("x_sync_freq"),
                    TrackbarCallback(params.get("x_sync_freq"), "x_sync_freq").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_sync_amp_slider = TrackbarRow(
                    "X Sync Amp",
                    params.get("x_sync_amp"),
                    TrackbarCallback(params.get("x_sync_amp"), "x_sync_amp").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_sync_speed_slider = TrackbarRow(
                    "Y Sync Speed",
                    params.get("y_sync_speed"),
                    TrackbarCallback(params.get("y_sync_speed"), "y_sync_speed").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_sync_freq_slider = TrackbarRow(
                    "Y Sync Freq",
                    params.get("y_sync_freq"),
                    TrackbarCallback(params.get("y_sync_freq"), "y_sync_freq").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_sync_amp_slider = TrackbarRow(
                    "Y Sync Amp",
                    params.get("y_sync_amp"),
                    TrackbarCallback(params.get("y_sync_amp"), "y_sync_amp").__call__,
                    self.reset_slider_callback,
                    default_font_id)
        dpg.bind_item_font("sync", global_font_id)

        with dpg.collapsing_header(label=f"\tNoiser", tag="noiser"):
            noise_type_slider = TrackbarRow(
                "Noise Type", 
                params.get("noise_type"), 
                TrackbarCallback(params.get("noise_type"), "noise_type").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            noise_intensity_slider = TrackbarRow(
                "Noise Intensity", 
                params.get("noise_intensity"), 
                TrackbarCallback(params.get("noise_intensity"), "noise_intensity").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            # noise_speed_slider = TrackbarRow(
            #     "Noise Speed", 
            #     params.get("noise_speed"), 
            #     TrackbarCallback(params.get("noise_speed"), "noise_speed").__call__, 
            #     self.reset_slider_callback, 
            #     default_font_id)
            
            # noise_freq_x_slider = TrackbarRow(
            #     "Noise Freq X", 
            #     params.get("noise_freq_x"), 
            #     TrackbarCallback(params.get("noise_freq_x"), "noise_freq_x").__call__, 
            #     self.reset_slider_callback, 
            #     default_font_id)
            
            # noise_freq_y_slider = TrackbarRow(
            #     "Noise Freq Y", 
            #     params.get("noise_freq_y"), 
            #     TrackbarCallback(params.get("noise_freq_y"), "noise_freq_y").__call__, 
            #     self.reset_slider_callback, 
            #     default_font_id)
        dpg.bind_item_font("noiser", global_font_id)

        with dpg.collapsing_header(label=f"\tNoise Generator", tag="noise_generator"):
            
            perlin_amplitude_slider = TrackbarRow(
                "Perlin Amplitude", 
                params.get("perlin_amplitude"), 
                TrackbarCallback(params.get("perlin_amplitude"), "perlin_amplitude").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            perlin_frequency_slider = TrackbarRow(
                "Perlin Frequency", 
                params.get("perlin_frequency"), 
                TrackbarCallback(params.get("perlin_frequency"), "perlin_frequency").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            perlin_octaves_slider = TrackbarRow(
                "Perlin Octaves", 
                params.get("perlin_octaves"), 
                TrackbarCallback(params.get("perlin_octaves"), "perlin_octaves").__call__, 
                self.reset_slider_callback, 
                default_font_id)
        dpg.bind_item_font("noise_generator", global_font_id)

        with dpg.collapsing_header(label=f"\tShape Generator", tag="shape_generator"):
            shape_slider = TrackbarRow(
                "Shape Type",
                params.get("shape_type"),
                TrackbarCallback(params.get("shape_type"), "shape_type").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            canvas_rotation_slider = TrackbarRow(
                "Canvas Rotation", 
                params.get("canvas_rotation"), 
                TrackbarCallback(params.get("canvas_rotation"), "canvas_rotation").__call__, 
                self.reset_slider_callback, 
                default_font_id)

            size_multiplier_slider = TrackbarRow(
                "Size Multiplier", 
                params.get("size_multiplier"), 
                TrackbarCallback(params.get("size_multiplier"), "size_multiplier").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            aspect_ratio_slider = TrackbarRow(
                "Aspect Ratio", 
                params.get("aspect_ratio"), 
                TrackbarCallback(params.get("aspect_ratio"), "aspect_ratio").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            rotation_slider = TrackbarRow(
                "Rotation", 
                params.get("rotation_angle"), 
                TrackbarCallback(params.get("rotation_angle"), "rotation_angle").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            multiply_grid_x_slider = TrackbarRow(
                "Multiply Grid X", 
                params.get("multiply_grid_x"), 
                TrackbarCallback(params.get("multiply_grid_x"), "multiply_grid_x").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            multiply_grid_y_slider = TrackbarRow(
                "Multiply Grid Y", 
                params.get("multiply_grid_y"), 
                TrackbarCallback(params.get("multiply_grid_y"), "multiply_grid_y").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            grid_pitch_x_slider = TrackbarRow(
                "Grid Pitch X", 
                params.get("grid_pitch_x"), 
                TrackbarCallback(params.get("grid_pitch_x"), "grid_pitch_x").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            grid_pitch_y_slider = TrackbarRow(
                "Grid Pitch Y", 
                params.get("grid_pitch_y"), 
                TrackbarCallback(params.get("grid_pitch_y"), "grid_pitch_y").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            shape_y_shift_slider = TrackbarRow(
                "Shape Y Shift", 
                params.get("shape_y_shift"), 
                TrackbarCallback(params.get("shape_y_shift"), "shape_y_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            shape_x_shift_slider = TrackbarRow(
                "Shape X Shift", 
                params.get("shape_x_shift"), 
                TrackbarCallback(params.get("shape_x_shift"), "shape_x_shift").__call__, 
                self.reset_slider_callback, 
                default_font_id)

            with dpg.collapsing_header(label=f"\Line Generator", tag="line_generator"):

                line_hue_slider = TrackbarRow(
                    "Line Hue", 
                    params.get("line_hue"), 
                    TrackbarCallback(params.get("line_hue"), "line_hue").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                line_sat_slider = TrackbarRow(
                    "Line Sat", 
                    params.get("line_sat"), 
                    TrackbarCallback(params.get("line_sat"), "line_sat").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                line_val_slider = TrackbarRow(
                    "Line Val", 
                    params.get("line_val"), 
                    TrackbarCallback(params.get("line_val"), "line_val").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                line_weight_slider = TrackbarRow(
                    "Line Width", 
                    params.get("line_weight"), 
                    TrackbarCallback(params.get("line_weight"), "line_weight").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                line_opacity_slider = TrackbarRow(
                    "Line Opacity", 
                    params.get("line_opacity"), 
                    TrackbarCallback(params.get("line_opacity"), "line_opacity").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
            dpg.bind_item_font("line_generator", global_font_id)

            with dpg.collapsing_header(label=f"\tFill Generator", tag="fill_generator"):
                fill_hue_slider = TrackbarRow(
                    "Fill Hue", 
                    params.get("fill_hue"), 
                    TrackbarCallback(params.get("fill_hue"), "fill_hue").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                fill_sat_slider = TrackbarRow(
                    "Fill Sat", 
                    params.get("fill_sat"), 
                    TrackbarCallback(params.get("fill_sat"), "fill_sat").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                fill_val_slider = TrackbarRow(
                    "Fill Val", 
                    params.get("fill_val"), 
                    TrackbarCallback(params.get("fill_val"), "fill_val").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
                
                fill_opacity_slider = TrackbarRow(
                    "Fill Opacity", 
                    params.get("fill_opacity"), 
                    TrackbarCallback(params.get("fill_opacity"), "fill_opacity").__call__, 
                    self.reset_slider_callback, 
                    default_font_id)
            dpg.bind_item_font("fill_generator", global_font_id)
        dpg.bind_item_font("shape_generator", global_font_id)

        with dpg.collapsing_header(label=f"\tPattern Generator", tag="pattern_generator"):
            pattern_type_slider = TrackbarRow(
                "Pattern Type",
                params.get("pattern_type"),
                TrackbarCallback(params.get("pattern_type"), "pattern_type").__call__,
                self.reset_slider_callback,
                default_font_id)
            speed_slider = TrackbarRow(
                "Speed",
                params.get("pattern_speed"),
                TrackbarCallback(params.get("pattern_speed"), "pattern_speed").__call__,
                self.reset_slider_callback,
                default_font_id)
            angle_amt_slider = TrackbarRow(
                "Angle Amt",
                params.get("pattern_angle_amt"),
                TrackbarCallback(params.get("pattern_angle_amt"), "pattern_angle_amt").__call__,
                self.reset_slider_callback,
                default_font_id)
            radius_amt_slider = TrackbarRow(
                "Radius Amt",
                params.get("pattern_radius_amt"),
                TrackbarCallback(params.get("pattern_radius_amt"), "pattern_radius_amt").__call__,
                self.reset_slider_callback,
                default_font_id)
            use_fractal_slider = TrackbarRow(
                "Use Fractal",
                params.get("pattern_use_fractal"),
                TrackbarCallback(params.get("pattern_use_fractal"), "pattern_use_fractal").__call__,
                self.reset_slider_callback,
                default_font_id)
            octaves_slider = TrackbarRow(
                "Octaves",
                params.get("pattern_octaves"),
                TrackbarCallback(params.get("pattern_octaves"), "pattern_octaves").__call__,
                self.reset_slider_callback,
                default_font_id)
            gain_slider = TrackbarRow(
                "Gain",
                params.get("pattern_gain"),
                TrackbarCallback(params.get("pattern_gain"), "pattern_gain").__call__,
                self.reset_slider_callback,
                default_font_id)
            lacunarity_slider = TrackbarRow(
                "Lacunarity",
                params.get("pattern_lacunarity"),
                TrackbarCallback(params.get("pattern_lacunarity"), "pattern_lacunarity").__call__,
                self.reset_slider_callback,
                default_font_id)
        dpg.bind_item_font("pattern_generator", global_font_id)

        with dpg.collapsing_header(label=f"\tLissajous", tag="Lissajous"):
                lissajous_A_slider = TrackbarRow(
                    "Lissajous A",
                    params.get("lissajous_A"),
                    TrackbarCallback(params.get("lissajous_A"), "lissajous_A").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                lissajous_B_slider = TrackbarRow(
                    "Lissajous B",
                    params.get("lissajous_B"),
                    TrackbarCallback(params.get("lissajous_B"), "lissajous_B").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                lissajous_a_slider = TrackbarRow(
                    "Lissajous a",
                    params.get("lissajous_a"),
                    TrackbarCallback(params.get("lissajous_a"), "lissajous_a").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                lissajous_b_slider = TrackbarRow(
                    "Lissajous b",
                    params.get("lissajous_b"),
                    TrackbarCallback(params.get("lissajous_b"), "lissajous_b").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                lissajous_delta_slider = TrackbarRow(
                    "Lissajous Delta",
                    params.get("lissajous_delta"),
                    TrackbarCallback(params.get("lissajous_delta"), "lissajous_delta").__call__,
                    self.reset_slider_callback,
                    default_font_id)
        dpg.bind_item_font("Lissajous", global_font_id)

        with dpg.collapsing_header(label=f"\tTest", tag="test"):
                frame_skip_slider = TrackbarRow(
                    "Frame Skip",
                    params.get("frame_skip"),
                    TrackbarCallback(params.get("frame_skip"), "frame_skip").__call__,
                    self.reset_slider_callback,
                    default_font_id)

                warp_type_slider = TrackbarRow(
                    "Warp Type",
                    params.get("warp_type"),
                    TrackbarCallback(params.get("warp_type"), "warp_type").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_angle_amt_slider = TrackbarRow(
                    "Warp Angle Amt",
                    params.get("warp_angle_amt"),
                    TrackbarCallback(params.get("warp_angle_amt"), "warp_angle_amt").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_radius_amt_slider = TrackbarRow(
                    "Warp Radius Amt",
                    params.get("warp_radius_amt"),
                    TrackbarCallback(params.get("warp_radius_amt"), "warp_radius_amt").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_speed_slider = TrackbarRow(
                    "Warp Speed",
                    params.get("warp_speed"),
                    TrackbarCallback(params.get("warp_speed"), "warp_speed").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_use_fractal_slider = TrackbarRow(
                    "Warp Use Fractal",
                    params.get("warp_use_fractal"),
                    TrackbarCallback(params.get("warp_use_fractal"), "warp_use_fractal").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_octaves_slider = TrackbarRow(
                    "Warp Octaves",
                    params.get("warp_octaves"),
                    TrackbarCallback(params.get("warp_octaves"), "warp_octaves").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_gain_slider = TrackbarRow(
                    "Warp Gain",
                    params.get("warp_gain"),
                    TrackbarCallback(params.get("warp_gain"), "warp_gain").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                warp_lacunarity_slider = TrackbarRow(
                    "Warp Lacunarity",
                    params.get("warp_lacunarity"),
                    TrackbarCallback(params.get("warp_lacunarity"), "warp_lacunarity").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_speed_slider = TrackbarRow(
                    "X Speed",
                    params.get("x_speed"),
                    TrackbarCallback(params.get("x_speed"), "x_speed").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                y_speed_slider = TrackbarRow(
                    "Y Speed",
                    params.get("y_speed"),
                    TrackbarCallback(params.get("y_speed"), "y_speed").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                x_size_slider = TrackbarRow(
                    "X Size",
                    params.get("x_size"),
                    TrackbarCallback(params.get("x_size"), "x_size").__call__,
                    self.reset_slider_callback,
                    default_font_id)
                y_size_slider = TrackbarRow(
                    "Y Size",
                    params.get("y_size"),
                    TrackbarCallback(params.get("y_size"), "y_size").__call__,
                    self.reset_slider_callback,
                    default_font_id)

        osc_freq_sliders = []
        osc_amp_sliders = []
        osc_phase_sliders = []
        osc_seed_sliders = []
        osc_shape_sliders = []
        for i in range(NUM_OSCILLATORS):
            print(f"Creating sliders for oscillator {i}")
            with dpg.collapsing_header(label=f"\tOscillator {i}", tag=f"osc{i}"):
                osc_shape_sliders.append(TrackbarRow(
                    f"Osc {i} Shape", 
                    osc_bank[i].shape, 
                    TrackbarCallback(osc_bank[i].shape, f"osc{i}_shape").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                osc_freq_sliders.append(TrackbarRow(
                    f"Osc {i} Freq", 
                    osc_bank[i].frequency, 
                    TrackbarCallback(osc_bank[i].frequency, f"osc{i}_frequency").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_amp_sliders.append(TrackbarRow(
                    f"Osc {i} Amp", 
                    osc_bank[i].amplitude, 
                    TrackbarCallback(osc_bank[i].amplitude, f"osc{i}_amplitude").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_phase_sliders.append(TrackbarRow(
                    f"Osc {i} Phase", 
                    osc_bank[i].phase, 
                    TrackbarCallback(osc_bank[i].phase, f"osc{i}_phase").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_seed_sliders.append(TrackbarRow(
                    f"Osc {i} Seed", 
                    osc_bank[i].seed, 
                    TrackbarCallback(osc_bank[i].seed, f"osc{i}_seed").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                # Create a list of items for the listbox
                items = list(params.keys())

                # Create the listbox
                dpg.add_combo(items=items,
                                label="Select Parameter",
                                tag=f"osc{i}_combobox",
                                default_value=None,
                                callback=self.listbox_cb)

            dpg.bind_item_font(f"osc{i}", global_font_id)

    def build_panels_dict(self, params):
        panels = {}
        for p in params.values():
            if p.name not in panels.keys():
                panels[str(p.family)] = []
            panels[str(p.family)].append(p.name)
                
        return panels

    def create_trackbar_panels_for_param(self):
        with dpg.font_registry():
            global_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 18) # Larger font size for the header
            default_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14) # Default font size for other items
        
        # Bind the default font to the whole window (optional, but good practice)
        dpg.bind_font(self.default_font_id)
        li = []
        for panel_name, panel_params in self.panels.items():
            with dpg.collapsing_header(label=f"\t{panel_name}", tag=f"{panel_name}"):
                for p in panel_params:
                    li.append(TrackbarRow(
                    p, 
                    params.get(p), 
                    TrackbarCallback(params.get(p), p).__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
            dpg.bind_item_font(panel_name, global_font_id)



