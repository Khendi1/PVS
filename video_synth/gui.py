import config as p
from gui_elements import TrackbarRow, Button, TrackbarRow
import dearpygui.dearpygui as dpg
from generators import PerlinNoise, Interp, Oscillator
import random

class Interface:
    def __init__(self, width=550, height=420):
        self.sliders = []
        self.buttons = []
        self.slider_dict = None


    def get_button_by_tag(tag):
        for b in self.buttons:
            if b.tag == tag:
                return b

    def reset_slider_callback(self, sender, app_data, user_data):
        print(f"Got reset callback for {user_data}")
        s = self.slider_dict[str(user_data)]
        s.value = s.min_value
        dpg.set_value(user_data, s.min_value)

    def x_shift_cb(self, sender, app_data):
        self.slider_dict["x_shift"].value = app_data
        p.params["x_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def y_shift_cb(self, sender, app_data):
        self.slider_dict["y_shift"].value = app_data
        p.params["y_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def r_shift_cb(self, sender, app_data):
        self.slider_dict["r_shift"].value = app_data
        p.params["r_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def hue_shift_cb(self, sender, app_data):
        self.slider_dict["hue_shift"].value = app_data
        p.params["hue_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def sat_shift_cb(self, sender, app_data):
        self.slider_dict["sat_shift"].value = app_data
        p.params["sat_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def val_shift_cb(self, sender, app_data):
        self.slider_dict["val_shift"].value = app_data
        p.params["val_shift"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def alpha_cb(self, sender, app_data):
        self.slider_dict["alpha"].value = app_data
        p.alpha = app_data
        p.params["alpha"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def num_glitches_cb(self, sender, app_data):
        self.slider_dict["num_glitches"].value = app_data
        p.params["num_glitches"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def glitch_size_cb(self, sender, app_data):
        self.slider_dict["glitch_size"].value = app_data
        p.params["glitch_size"].set_value(app_data)
        dpg.set_value(sender, app_data)

    def blur_kernel_size_cb(self, sender, app_data):
        self.slider_dict["blur_kernel_size"].value = app_data
        p.params["blur_kernel_size"].set_value(app_data)
        p.params["blur_kernel_size"].value = max(1, p.params["blur_kernel_size"].value | 1)
        dpg.set_value(sender, app_data)    

    def val_threshold_cb(self, sender, app_data):
        self.slider_dict["val_threshold"].value = app_data
        p.val_threshold = app_data
        dpg.set_value(sender, app_data)

    def val_hue_shift_cb(self, sender, app_data):
        self.slider_dict["val_hue_shift"].value = app_data
        p.val_threshold = app_data
        dpg.set_value(sender, app_data)

    def perlin_frequency_cb(self, sender, app_data):
        self.slider_dict["perlin_frequency"].value = app_data
        p.params["perlin_frequency"].set_value(app_data)
        p.perlin_noise.frequency = app_data
        dpg.set_value(sender, app_data)

    def perlin_amplitude_cb(self, sender, app_data):
        self.slider_dict["perlin_amplitude"].value = app_data
        p.params["perlin_amplitude"].set_value(app_data)
        p.perlin_noise.amplitude = app_data
        dpg.set_value(sender, app_data)

    def perlin_octaves_cb(self, sender, app_data):
        self.slider_dict["perlin_octaves"].value = app_data
        p.params["perlin_octaves"].set_value(app_data)
        p.perlin_noise.ovtaves = app_data
        dpg.set_value(sender, app_data)

    def amplitude_cb(self, sender, app_data):
        for i in range(4):
            if f"osc{i}" in sender:
                self.slider_dict[f"osc{i}_amplitude"].value = app_data
                p.osc_bank[i].amplitude.set_value(app_data)                             
                dpg.set_value(sender, app_data)

    def frequency_cb(self, sender, app_data):
        for i in range(4):
            if f"osc{i}" in sender:
                self.slider_dict[f"osc{i}_frequency"].value = app_data
                p.osc_bank[i].frequency.set_value(app_data)                                 
                dpg.set_value(sender, app_data)
    
    def phase_cb(self, sender, app_data):
        for i in range(4):
            if f"osc{i}" in sender:
                self.slider_dict[f"osc{i}_phase"].value = app_data
                p.osc_bank[i].phase.set_value(app_data)                              
                dpg.set_value(sender, app_data)
    
    def seed_cb(self, sender, app_data):
        for i in range(4):
            if f"osc{i}" in sender:
                self.slider_dict[f"osc{i}_seed"].value = app_data
                p.osc_bank[i].seed.set_value(app_data)  
                dpg.set_value(sender, app_data)
    
    def shape_cb(self, sender, app_data):
        for i in range(4):
            if f"osc{i}" in sender:
                self.slider_dict[f"osc{i}_shape"].value = app_data
                p.osc_bank[i].shape.set_value(app_data)                               
                dpg.set_value(sender, app_data)

    def polar_x_cb(self, sender, app_data):
        self.slider_dict["polar_x"].value = app_data
        p.params["polar_x"].set_value(app_data)                                   
        dpg.set_value(sender, app_data)

    def polar_y_cb(self, sender, app_data):
        self.slider_dict["polar_y"].value = app_data
        p.params["polar_y"].set_value(app_data)                                
        dpg.set_value(sender, app_data)

    def polar_radius_cb(self, sender, app_data):
        self.slider_dict["polar_radius"].value = app_data
        p.params["polar_radius"].set_value(app_data)                                    
        dpg.set_value(sender, app_data)

    def contrast_cb(self, sender, app_data):
        self.slider_dict["contrast"].value = app_data
        p.params["contrast"].set_value(app_data)                                    
        dpg.set_value(sender, app_data)

    def brightness_cb(self, sender, app_data):
        self.slider_dict["brightness"].value = app_data
        p.params["brightness"].set_value(app_data)                                    
        dpg.set_value(sender, app_data)

    ##############################################################33
    ##############################################################33
    ##############################################################33

    def on_save_button_click(self):
        date_time_str = datetime.now().strftime("%m-%d-%Y %H-%M")
        print(f"Saving values at {date_time_str}")
        
        data = {}
        for k, v in p.params.items():
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

        fwd = get_button_by_tag("load_next")
        b = get_button_by_tag("load_prev")
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

        fwd = get_button_by_tag("load_next")
        prev = get_button_by_tag("load_prev")
        rand = get_button_by_tag("load_rand")
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

    def reset_values(self):
        for s in self.sliders:
            s.value = s.min_value
            if s.tag == "x_shift" or s.tag == "y_shift":
                s.value = 0
            dpg.set_value(s.tag, s.value)

    def randomize_values(self):
        for s in self.sliders:
            if s.tag == "blur_kernel":
                s.value = max(1, random.randint(1, s.max_value) | 1)
            elif s.tag == "x_shift":
                s.value = random.randint(-image_width, image_width)
            elif s.tag == "y_shift":
                s.value = random.randint(-image_height, image_height)
            elif s.tag == "glitch_size":
                s.value = random.randint(1, s.max_value)
            elif s.tag == 'feedback':
                s.value = random.uniform(0.0, 1.0)
            else:
                s.value = random.randint(s.min_value, s.max_value)
            dpg.set_value(s.tag, s.value)

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
        selected_indices = app_data['items']
        items = dpg.get_item_configuration(sender)['items'] #get the items
        selected_items = [items[i] for i in selected_indices]

        print("Selected Items:", selected_items) # print the selected items


    def create_trackbars(self, width, height):

        with dpg.font_registry():
            # 2. Add a custom font with a specific size
            # You can use a system font or provide a path to a .ttf file
            # 'C:/Windows/Fonts/arial.ttf' is a common path on Windows
            # For cross-platform, you might need to find a suitable font or
            # bundle one with your application.
            # Here, we'll use the default font family but specify a larger size.
            # If you want a specific font, replace "C:/Windows/Fonts/arial.ttf"
            # with the path to your font file.
            global_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 18) # Larger font size for the header
            default_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14) # Default font size for other items
        
        # Bind the default font to the whole window (optional, but good practice)
        dpg.bind_font(default_font_id)

        with dpg.collapsing_header(label=f"HSV", tag="hsv"):
            hue_slider = TrackbarRow("Hue Shift", p.params["hue_shift"], self.hue_shift_cb, self.reset_slider_callback, default_font_id)
            sat_slider = TrackbarRow("Sat Shift", p.params["sat_shift"], self.sat_shift_cb, self.reset_slider_callback, default_font_id)
            val_slider = TrackbarRow("Val Shift", p.params["val_shift"], self.val_shift_cb, self.reset_slider_callback, default_font_id)
            contrast_slider = TrackbarRow("Contrast", p.params["contrast"], self.contrast_cb, self.reset_slider_callback, default_font_id)
            brightness_slider = TrackbarRow("Brighness", p.params["brightness"], self.brightness_cb, self.reset_slider_callback, default_font_id)

        with dpg.collapsing_header(label=f"Feedback, Blur, Glitch", tag="feedback_blur_glitch"):
            alpha_slider = TrackbarRow("Feedback", p.params["alpha"], self.alpha_cb, self.reset_slider_callback, default_font_id)
            blur_kernel_slider = TrackbarRow("Blur Kernel", p.params["blur_kernel_size"], self.blur_kernel_size_cb, self.reset_slider_callback, default_font_id)
            num_glitches_slider = TrackbarRow("Glitch Qty", p.params["num_glitches"], self.num_glitches_cb, self.reset_slider_callback, default_font_id)
            glitch_size_slider = TrackbarRow("Glitch Size", p.params["glitch_size"], self.glitch_size_cb, self.reset_slider_callback, default_font_id)
        
        with dpg.collapsing_header(label=f"Val Threshold", tag="val_threshold_header"):
            val_threshold_slider = TrackbarRow("Val Threshold", p.params["val_threshold"], self.val_threshold_cb, self.reset_slider_callback, default_font_id)
            val_hue_shift_slider = TrackbarRow("Hue Shift for Val", p.params["val_hue_shift"], self.val_hue_shift_cb, self.reset_slider_callback, default_font_id)
        
        with dpg.collapsing_header(label=f"Pan", tag="pan"):
            x_shift_slider = TrackbarRow("X Shift", p.params["x_shift"], self.y_shift_cb, self.reset_slider_callback, default_font_id)
            y_shift_slider = TrackbarRow("Y Shift", p.params["y_shift"], self.x_shift_cb, self.reset_slider_callback, default_font_id)
            r_shift_slider = TrackbarRow("R Shift", p.params["r_shift"], self.r_shift_cb, self.reset_slider_callback, default_font_id)

        with dpg.collapsing_header(label=f"Noise Generator", tag="noise_generator"):
            perlin_amplitude_slider = TrackbarRow("Perlin Amplitude", p.params["perlin_amplitude"], self.perlin_frequency_cb, self.reset_slider_callback, default_font_id)
            perlin_frequency_slider = TrackbarRow("Perlin Frequency", p.params["perlin_frequency"], self.perlin_amplitude_cb, self.reset_slider_callback, default_font_id)
            perlin_octaves_slider = TrackbarRow("Perlin Octaves", p.params["perlin_octaves"], self.perlin_octaves_cb, self.reset_slider_callback, default_font_id)
        
        with dpg.collapsing_header(label=f"Polar Transform", tag="polar_transform"):
            enable_polar_transform_button = Button("Enable Polar Transform", "enable_polar_transform")
            # polar_coord_mode_slider = TrackbarRow("Polar Coord Mode", p.polar_coord_mode, self.polar_x_cb, self.reset_slider_callback, default_font_id)
            dpg.add_button(label=enable_polar_transform_button.label, tag="enable_polar_transform", callback=self.on_button_click, user_data=enable_polar_transform_button.tag, width=width)
            dpg.bind_item_font(enable_polar_transform_button.tag, default_font_id)
            polar_x_slider = TrackbarRow("Polar Center X", p.params["polar_x"], self.polar_x_cb, self.reset_slider_callback, default_font_id)
            polar_y_slider = TrackbarRow("Polar Center Y", p.params["polar_y"], self.polar_y_cb, self.reset_slider_callback, default_font_id)
            polar_radius_slider = TrackbarRow("Polar radius", p.params["polar_radius"], self.polar_radius_cb, self.reset_slider_callback, default_font_id)

        dpg.bind_item_font("hsv", global_font_id)
        dpg.bind_item_font("feedback_blur_glitch", global_font_id)
        dpg.bind_item_font("val_threshold_header", global_font_id)
        dpg.bind_item_font("noise_generator", global_font_id)
        dpg.bind_item_font("pan", global_font_id)
        dpg.bind_item_font("polar_transform", global_font_id)

        osc_freq_sliders = []
        osc_amp_sliders = []
        osc_phase_sliders = []
        osc_seed_sliders = []
        osc_shape_sliders = []
        for i in range(4):
            with dpg.collapsing_header(label=f"Osc{i}", tag=f"osc{i}"):
                osc_shape_sliders.append(TrackbarRow(f"Osc {i} Shape", p.osc_bank[i].shape, self.shape_cb, self.reset_slider_callback, default_font_id))
                osc_freq_sliders.append(TrackbarRow(f"Osc {i} Freq", p.osc_bank[i].frequency, self.frequency_cb, self.reset_slider_callback, default_font_id))
                osc_amp_sliders.append(TrackbarRow(f"Osc {i} Amp", p.osc_bank[i].amplitude, self.amplitude_cb, self.reset_slider_callback, default_font_id))
                osc_phase_sliders.append(TrackbarRow(f"Osc {i} Phase", p.osc_bank[i].phase, self.phase_cb, self.reset_slider_callback, default_font_id))
                osc_seed_sliders.append(TrackbarRow(f"Osc {i} Seed", p.osc_bank[i].seed, self.seed_cb, self.reset_slider_callback, default_font_id))
                                # Create a list of items for the listbox
                items = ["None", "Hue", "Sat", "Val", "Feedback", "Glitch Size", "Glitch Qty", "Pan X", "Pan Y", "R Shift", "Blur Kernel", "Val Threshold", "Val Hue Shift", "Perlin Amplitude", "Perlin Frequency", "Perlin Octaves", "Polar X", "Polar Y", "Polar Radius"]

                # Create the listbox
                dpg.add_combo(items=items,
                                label="Select Parameter",
                                tag=f"osc{i}_combobox",
                                default_value=None,
                                callback=self.listbox_cb)

            dpg.bind_item_font(f"osc{i}", global_font_id)

        self.sliders = [hue_slider, sat_slider, val_slider, alpha_slider, num_glitches_slider, glitch_size_slider, 
                val_threshold_slider, val_hue_shift_slider, blur_kernel_slider, x_shift_slider, y_shift_slider, r_shift_slider , 
                perlin_amplitude_slider, perlin_frequency_slider, perlin_octaves_slider, polar_radius_slider, brightness_slider, contrast_slider]

        self.slider_dict = {
            "hue_shift": hue_slider,
            "sat_shift": hue_slider,
            "val_shift": hue_slider,
            "alpha": alpha_slider,
            "num_glitches": num_glitches_slider,
            "glitch_size": glitch_size_slider,
            "val_threshold": val_threshold_slider,
            "val_hue_shift": val_hue_shift_slider, 
            "blur_kernel_size": blur_kernel_slider, 
            "x_shift": x_shift_slider, 
            "y_shift": y_shift_slider, 
            "r_shift": r_shift_slider,
            "perlin_amplitude": perlin_amplitude_slider, 
            "perlin_frequency": perlin_frequency_slider, 
            "perlin_octaves": perlin_octaves_slider,
            "polar_x": polar_x_slider,
            "polar_y": polar_y_slider,
            "polar_radius": polar_radius_slider,
            "brightness": brightness_slider,
            "contrast": contrast_slider
        }

        for i in range(4):
            self.slider_dict[f"osc{i}_frequency"] = osc_freq_sliders[i]
            self.slider_dict[f"osc{i}_amplitude"] = osc_amp_sliders[i]
            self.slider_dict[f"osc{i}_phase"] = osc_phase_sliders[i]
            self.slider_dict[f"osc{i}_seed"] = osc_seed_sliders[i]
            self.slider_dict[f"osc{i}_shape"] = osc_shape_sliders[i]
            self.sliders.append(osc_freq_sliders[i])
            self.sliders.append(osc_amp_sliders[i])
            self.sliders.append(osc_phase_sliders[i])
            self.sliders.append(osc_seed_sliders[i])
            self.sliders.append(osc_shape_sliders[i])

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
            if p.enable_polar_transform == True:
                p.enable_polar_transform = False
            else:
                p.enable_polar_transform = True

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
        # dpg.set_item_width("button2", half_width)

    def create_control_window(self, width=550, height=600):
        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):

            self.create_trackbars(width, height)
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)
