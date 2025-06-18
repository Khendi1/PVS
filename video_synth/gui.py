import config as c
from config import NUM_OSCILLATORS, params, toggles
from gui_elements import TrackbarRow, Button, TrackbarRow
import dearpygui.dearpygui as dpg
from generators import PerlinNoise, Interp, Oscillator
import random
import datetime
import yaml

#  Trackbar Callback #############################################################
class TrackbarCallback:
    """
    A callable class instance used as a callback for Dear PyGui trackbars.
    It updates a specified Param object's value and an associated text item.
    """
    def __init__(self, target_param_obj, display_text_tag=None):
        """
        Initializes the callback instance.

        Args:
            target_param_obj (Param): The Param object whose 'value' attribute
                                      this trackbar will control.
            display_text_tag (str, optional): The tag of a dpg.add_text item
                                            to update with the current value.
        """
        self.target_param = target_param_obj
        self.display_text_tag = display_text_tag

    def __call__(self, sender, app_data):
        """
        This method is invoked when the trackbar's value changes.
        sender: The tag/ID of the trackbar that triggered the callback.
        app_data: The new value of the trackbar.
        """
        # Update the Param object's value
        if self.display_text_tag == "blur_kernel_size":
            params.set("blur_kernel_size", max(1, app_data | 1))
        else:
            params.set(self.target_param.name, app_data)
        dpg.set_value(sender, app_data)

#  Interface ##################################################################### 
class Interface:
    def __init__(self, panel_width=550, panel_height=420):
        self.sliders = []
        self.buttons = []
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.slider_dict = None
        # self.panels = self.build_panels_dict(params.all())
        self.default_font_id = None
        # with dpg.font_registry():
        #     global_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 18) # Larger font size for the header
        #     self.default_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14) # Default font size for other items
        
        # # Bind the default font to the whole window (optional, but good practice)
        # dpg.bind_font(self.default_font_id)

    def reset_slider_callback(self, sender, app_data, user_data):
        param = params.get(str(user_data))
        if param is None:
            print(f"Slider or param not found for {user_data}")
            return
        print(f"Got reset callback for {user_data}; setting to default value {param.default_val}")
        param.reset()
        dpg.set_value(user_data, param.value)

    # Button Callbacks ###########################################################

    def on_save_button_click(self):
        date_time_str = datetime.now().strftime("%m-%d-%Y %H-%M")
        print(f"Saving values at {date_time_str}")
        
        data = {}
        for k, v in params.all().items():
            print(f"{k}: {v.value}")
            data[k] = v.value
        
        # Append the data to the YAML file
        with open("saved_values.yaml", "a") as f:
            yaml.dump([data], f, default_flow_style=False)
        
        # Optionally, save the modified image
        cv2.imwrite(f"{date_time_str}.jpg", feedback_frame)

    def on_fwd_button_click(self):

        fwd = self.get_button_by_tag("load_next")
        prev = self.get_button_by_tag("load_prev")
        print(f"Forward button clicked!")

        # get values from saved_values.yaml
        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            fwd.index = (fwd.index + 1) % len(saved_values[0])
            prev.index = fwd.index
            d = saved_values[0][fwd.index]
            print(f"loaded values at index {fwd.index}: {d}\n\n")
            
            for s in self.sliders:
                for tag in d.keys():
                    if tag == s.tag:
                        s.value = d[tag]
                        dpg.set_value(s.tag, s.value)
            
        except Exception as e:
            print(f"Error loading values: {e}")

    def on_prev_button_click(self):

        # fwd = get_button_by_tag("load_next")
        # b = get_button_by_tag("load_prev")
        print(f"Prev button clicked!")

        # get values from saved_values.yaml
        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            b.index = (b.index - 1) % len(saved_values[0])
            fwd.index = b.index
            d = saved_values[0][b.index]
            print(f"loaded values at index {b.index}: {d}\n\n")
            
            for s in self.sliders:
                for tag in d.keys():
                    if tag == s.tag:
                        s.value = d[tag]
                        dpg.set_value(s.tag, s.value)
            
        except Exception as e:
            print(f"Error loading values: {e}")

    def on_rand_button_click(self):

        # fwd = get_button_by_tag("load_next")
        # prev = get_button_by_tag("load_prev")
        # rand = get_button_by_tag("load_rand")
        print(f"Random button clicked!")
    
        # get values from saved_values.yaml
        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            rand.index = random.randint(0, len(saved_values[0]) - 1)
            fwd.index = rand.index
            prev.index = rand.index
            d = saved_values[0][rand.index]
            print(f"loaded values at index {rand.index}: {d}\n\n")
            
            for s in self.sliders:
                for tag in d.keys():
                    if tag == s.tag:
                        s.value = d[tag]
                        dpg.set_value(s.tag, s.value)
            
        except Exception as e:
            print(f"Error loading values: {e}")

    def on_button_click(self, sender, app_data, user_data):
        print(f"Button clicked: {user_data}, {app_data}, {sender}")
        # Perform action based on button click
        if user_data == "save":
            self.on_save_button_click()
        elif user_data == "reset_all":
            self.reset_values()
        elif user_data == "random":
            self.randomize_values()
        elif user_data == "load_next":
           self.on_fwd_button_click()
        elif user_data == "load_prev":
            self.on_prev_button_click()
        elif user_data == "load_rand":
            self.on_rand_button_click()
        elif user_data == "interp":
            pass
        elif user_data == "fade":
            pass
        elif user_data == "enable_polar_transform":
            if c.enable_polar_transform == True:
                c.enable_polar_transform = False
            else:
                c.enable_polar_transform = True

    # Button methods #############################################################

    def reset_values(self):
        for s in self.sliders:
            s.value = s.min_value
            if s.tag == "x_shift" or s.tag == "y_shift":
                s.value = 0
            dpg.set_value(s.tag, s.value)

    def randomize_values(self):
        # TODO: use param.randomize() method
        for p in params.all():
            c.randomize()
            # dpg.set_value(s.tag, s.value)

    def create_buttons(self, width, height):

        save_button = Button("Save", "save")
        reset_button = Button("Reset all", 'reset_all')
        random_button = Button("Random", 'random')
        load_next_button = Button("Load >>", 'load_next')
        load_rand_button = Button("Load ??", "load_rand")
        load_prev_button = Button("Load <<", "load_prev")

        self.buttons = [save_button, reset_button, random_button, load_next_button, load_rand_button, load_prev_button]

        width -= 20

        # with dpg.group(horizontal=True):
            
        with dpg.group(horizontal=True):
            dpg.add_button(label=save_button.label, callback=self.on_button_click, user_data=save_button.tag, width=width//3)
            dpg.add_button(label=reset_button.label, callback=self.on_button_click, user_data=reset_button.tag, width=width//3)
            dpg.add_button(label=random_button.label, tag=random_button.tag, callback=self.on_button_click, user_data=random_button.tag, width=width//3)

        with dpg.group(horizontal=True):
            dpg.add_button(label=load_prev_button.label, callback=self.on_button_click, user_data=load_prev_button.tag, width=width//3)
            dpg.add_button(label=load_rand_button.label, callback=self.on_button_click, user_data=load_rand_button.tag, width=width//3)    
            dpg.add_button(label=load_next_button.label, callback=self.on_button_click, user_data=load_next_button.tag, width=width//3)

        # future buttons: load image, reload image, max feedback, undo, redo, save image

    def resize_buttons(self, sender, app_data):
        # Get the current width of the window
        window_width = dpg.get_item_width("Controls")
        
        # Set each button to half the window width (minus a small padding if you want)
        half_width = window_width // 2
        dpg.set_item_width(sender, half_width)

    # Listbox callback #############################################################

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
                for tag, param in params.all().items():
                    if tag == app_data:
                        c.osc_bank[i].linked_param = param
                        break

    # Create the control window and features #######################################
    def create_control_window(self, width=550, height=600):
        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars(width, height)
            # self.create_trackbar_panels_for_param()
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
        
        # Bind the default font to the whole window (optional, but good practice)
        dpg.bind_font(default_font_id)

        with dpg.collapsing_header(label=f"\tHSV", tag="hsv"):
            hue_slider = TrackbarRow(
                "Hue Shift", 
                params.get("hue_shift"), 
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
            
            temporal_filter_slider = TrackbarRow(
                "Temporal Filter", 
                params.get("temporal_filter"), 
                TrackbarCallback(params.get("temporal_filter"), "temporal_filter").__call__, 
                self.reset_slider_callback, 
                default_font_id)
        dpg.bind_item_font("hsv", global_font_id)

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
            # polar_coord_mode_slider = TrackbarRow("Polar Coord Mode", c.polar_coord_mode, self.polar_x_cb, self.reset_slider_callback, default_font_id)
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

        dpg.bind_item_font("pan", global_font_id)

        with dpg.collapsing_header(label=f"\tShape Generator", tag="shape_generator"):
            shape_slider = TrackbarRow(
                "Shape Type",
                params.get("shape_type"),
                TrackbarCallback(params.get("shape_type"), "shape_type").__call__,
                self.reset_slider_callback,
                default_font_id)

            line_hue_slider = TrackbarRow(
                "Line Hue", 
                params.get("line_hue"), 
                TrackbarCallback(params.get("line_hue"), "line_hue").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            line_saturation_slider = TrackbarRow(
                "Line Sat", 
                params.get("line_saturation"), 
                TrackbarCallback(params.get("line_saturation"), "line_saturation").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            line_value_slider = TrackbarRow(
                "Line Val", 
                params.get("line_value"), 
                TrackbarCallback(params.get("line_value"), "line_value").__call__, 
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
            
            fill_hue_slider = TrackbarRow(
                "Fill Hue", 
                params.get("fill_hue"), 
                TrackbarCallback(params.get("fill_hue"), "fill_hue").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            fill_saturation_slider = TrackbarRow(
                "Fill Sat", 
                params.get("fill_saturation"), 
                TrackbarCallback(params.get("fill_saturation"), "fill_saturation").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            fill_value_slider = TrackbarRow(
                "Fill Val", 
                params.get("fill_value"), 
                TrackbarCallback(params.get("fill_value"), "fill_value").__call__, 
                self.reset_slider_callback, 
                default_font_id)
            
            fill_opacity_slider = TrackbarRow(
                "Fill Opacity", 
                params.get("fill_opacity"), 
                TrackbarCallback(params.get("fill_opacity"), "fill_opacity").__call__, 
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
            
            dpg.bind_item_font("shape_generator", global_font_id)

        osc_freq_sliders = []
        osc_amp_sliders = []
        osc_phase_sliders = []
        osc_seed_sliders = []
        osc_shape_sliders = []
        for i in range(NUM_OSCILLATORS):
            with dpg.collapsing_header(label=f"\tOscillator {i}", tag=f"osc{i}"):
                osc_shape_sliders.append(TrackbarRow(
                    f"Osc {i} Shape", 
                    c.osc_bank[i].shape, 
                    TrackbarCallback(c.osc_bank[i].shape, f"osc{i}_shape").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                osc_freq_sliders.append(TrackbarRow(
                    f"Osc {i} Freq", 
                    c.osc_bank[i].frequency, 
                    TrackbarCallback(c.osc_bank[i].frequency, f"osc{i}_frequency").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_amp_sliders.append(TrackbarRow(
                    f"Osc {i} Amp", 
                    c.osc_bank[i].amplitude, 
                    TrackbarCallback(c.osc_bank[i].amplitude, f"osc{i}_amplitude").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_phase_sliders.append(TrackbarRow(
                    f"Osc {i} Phase", 
                    c.osc_bank[i].phase, 
                    TrackbarCallback(c.osc_bank[i].phase, f"osc{i}_phase").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_seed_sliders.append(TrackbarRow(
                    f"Osc {i} Seed", 
                    c.osc_bank[i].seed, 
                    TrackbarCallback(c.osc_bank[i].seed, f"osc{i}_seed").__call__, 
                    self.reset_slider_callback, 
                    default_font_id))
                
                # Create a list of items for the listbox
                items = list(params.all().keys())

                # Create the listbox
                dpg.add_combo(items=items,
                                label="Select Parameter",
                                tag=f"osc{i}_combobox",
                                default_value=None,
                                callback=self.listbox_cb)

            dpg.bind_item_font(f"osc{i}", global_font_id)

    def build_panels_dict(self, params):
        panels = {}
        for p in params.all().values():
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



