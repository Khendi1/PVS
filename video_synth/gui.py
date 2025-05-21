import vars
from gui_elements import SliderRow, Button
import dearpygui.dearpygui as dpg

class Interface:
    def __init__(self, width=550, height=420):

        self.sliders = []
        self.buttons = []

        self.slider_dict = None
        print("test4")
        # self.create_trackbars()
        print("test5")
        # self.create_buttons()

    def get_button_by_tag(tag):
        for b in self.buttons:
            if b.tag == tag:
                return b

    def reset_slider_callback(self, sender, app_data, user_data):
        print(f"Got reset callback for {user_data}")
        s = self.slider_dict[user_data]
        s.value = s.min_value
        dpg.set_value(user_data, s.min_value)

    def x_shift_cb(self, sender, app_data):
        self.slider_dict["x_shift"].value = app_data
        vars.x_shift = app_data
        dpg.set_value(sender, app_data)

    def y_shift_cb(self, sender, app_data):
        self.slider_dict["y_shift"].value = app_data
        vars.y_shift = app_data
        dpg.set_value(sender, app_data)

    def r_shift_cb(self, sender, app_data):
        self.slider_dict["r_shift"].value = app_data
        vars.r_shift = app_data
        dpg.set_value(sender, app_data)


    def hue_shift_cb(self, sender, app_data):
        self.slider_dict["hue_shift"].value = app_data
        vars.hue_shift = app_data
        dpg.set_value(sender, app_data)

    def sat_shift_cb(self, sender, app_data):
        self.slider_dict["sat_shift"].value = app_data
        vars.sat_shift = app_data
        dpg.set_value(sender, app_data)

    def val_shift_cb(self, sender, app_data):
        self.slider_dict["val_shift"].value = app_data
        vars.val_shift = app_data
        dpg.set_value(sender, app_data)

    def alpha_cb(self, sender, app_data):
        self.slider_dict["alpha"].value = app_data
        vars.alpha = app_data
        dpg.set_value(sender, app_data)

    def num_glitches_cb(self, sender, app_data):
        self.slider_dict["num_glitches"].value = app_data
        vars.num_glitches = app_data
        dpg.set_value(sender, app_data)

    def glitch_size_cb(self, sender, app_data):
        self.slider_dict["glitch_size"].value = app_data
        vars.glitch_size = app_data
        dpg.set_value(sender, app_data)

    def blur_kernel_size_cb(self, sender, app_data):
        self.slider_dict["blur_kernel_size"].value = app_data
        vars.blur_kernel_size = app_data
        dpg.set_value(sender, app_data)    

    def val_threshold_cb(self, sender, app_data):
        self.slider_dict["val_threshold"].value = app_data
        vars.val_threshold = app_data
        dpg.set_value(sender, app_data)

    def val_hue_shift_cb(self, sender, app_data):
        self.slider_dict["val_hue_shift"].value = app_data
        vars.val_threshold = app_data
        dpg.set_value(sender, app_data)

    def perlin_frequency_cb(self, sender, app_data):
        self.slider_dict["perlin_frequency"].value = app_data
        vars.perlin_frequency = app_data
        dpg.set_value(sender, app_data)

    def perlin_amplitude_cb(self, sender, app_data):
        self.slider_dict["perlin_amplitude"].value = app_data
        vars.perlin_amplitude = app_data
        dpg.set_value(sender, app_data)

    def perlin_octaves_cb(self, sender, app_data):
        self.slider_dict["perlin_octaves"].value = app_data
        vars.perlin_octaves = app_data
        dpg.set_value(sender, app_data)

    # def interp_mode_cb(self, sender, app_data):
    #     self.slider_dict[""].value = app_data
    #     vars.interp_mode = app_data
    #     dpg.set_value(sender, app_data)

    # def osc_mode_cb(self, sender, app_data):
    #     self.slider_dict[""].value = app_data
    #     vars.osc_mode = app_data
    #     dpg.set_value(sender, app_data)

    def amplitude_cb(self, sender, app_data):
        self.slider_dict["amplitude"].value = app_data
        vars.amplitude = app_data
        dpg.set_value(sender, app_data)

    def frequency_cb(self, sender, app_data):
        self.slider_dict["frequency"].value = app_data
        vars.frequency = app_data                                    
        dpg.set_value(sender, app_data)

    def polar_x_cb(self, sender, app_data):
        self.slider_dict["polar_x"].value = app_data
        vars.polar_x = app_data                                    
        dpg.set_value(sender, app_data)

    def polar_y_cb(self, sender, app_data):
        self.slider_dict["polar_y"].value = app_data
        vars.polar_y = app_data                                    
        dpg.set_value(sender, app_data)

    def on_save_button_click(self):
        date_time_str = datetime.now().strftime("%m-%d-%Y %H-%M")
        print(f"Saving values at {date_time_str}")
        
        # Prepare the data to save
        data = {
            "timestamp": date_time_str,
            "hue_shift": vars.hue_shift,
            "sat_shift": vars.sat_shift,
            "val_shift": vars.val_shift,
            "alpha": vars.alpha,
            "num_glitches": vars.num_glitches,
            "glitch_size": vars.glitch_size,
            "val_threshold": vars.val_threshold,
            "val_hue_shift": vars.val_hue_shift,
            "blur_kernel_size": vars.blur_kernel_size,
            "x_shift": vars.x_shift,
            "y_shift": vars.y_shift,
        }
        
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
            
            for s in sliders:
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
            
            for s in sliders:
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
            
            for s in sliders:
                for tag in d.keys():
                    if tag == s.tag:
                        s.value = d[tag]
                        dpg.set_value(s.tag, s.value)
            
        except Exception as e:
            print(f"Error loading values: {e}")

    def reset_values(self):
        global sliders
        for s in sliders:
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

    def create_trackbars(self):

        hue_slider = SliderRow("Hue Shift", "hue_shift", vars.hue_shift, 0, 180, self.hue_shift_cb, 'int', self.reset_slider_callback)
        sat_slider = SliderRow("Sat Shift", "sat_shift", vars.sat_shift, 0, 255, self.sat_shift_cb, 'int', self.reset_slider_callback)
        val_slider = SliderRow("Val Shift", "val_shift", vars.val_shift, 0, 255, self.val_shift_cb, 'int', self.reset_slider_callback)
        alpha_slider = SliderRow("Feedback", "feedback", vars.alpha, 0.0, 1.0, self.alpha_cb, 'float', self.reset_slider_callback)
        num_glitches_slider = SliderRow("Glitch Qty", "glitch_qty", vars.num_glitches, 0, 100, self.num_glitches_cb, 'int', self.reset_slider_callback)
        glitch_size_slider = SliderRow("Glitch Size", "glitch_size", vars.glitch_size, 1, 100, self.glitch_size_cb, 'int', self.reset_slider_callback)
        val_threshold_slider = SliderRow("Val Threshold", "val_threshold", vars.val_threshold, 0, 255, self.val_threshold_cb, 'int', self.reset_slider_callback)
        val_hue_shift_slider = SliderRow("Hue Shift for Val", "hue_val_shift", vars.val_hue_shift, 0, 180, self.val_hue_shift_cb, 'int', self.reset_slider_callback)
        blur_kernel_slider = SliderRow("Blur Kernel", "blur_kernel", vars.blur_kernel_size, 0, 100, self.blur_kernel_size_cb, 'int', self.reset_slider_callback)
        x_shift_slider = SliderRow("X Shift", "x_shift", vars.x_shift, -vars.image_width//2, vars.image_width//2, self.y_shift_cb, 'int', self.reset_slider_callback)
        y_shift_slider = SliderRow("Y Shift", "y_shift", vars.y_shift, -vars.image_height//2, vars.image_height//2, self.x_shift_cb, 'int', self.reset_slider_callback)
        r_shift_slider = SliderRow("R Shift", "r_shift", vars.r_shift, -369, 360, self.r_shift_cb, 'int', self.reset_slider_callback)

        perlin_amplitude_slider = SliderRow("Perlin Amplitude", "perlin_amplitude", vars.perlin_amplitude, 0, 100, self.perlin_frequency_cb, 'int', self.reset_slider_callback)
        perlin_frequency_slider = SliderRow("Perlin Frequency", "perlin_frequency", vars.perlin_frequency, 0, 100, self.perlin_amplitude_cb, 'int', self.reset_slider_callback)
        perlin_octaves_slider = SliderRow("Perlin Octaves", "perlin_octaves", vars.perlin_octaves, 0, 100, self.perlin_octaves_cb, 'int', self.reset_slider_callback)
        
        polar_x_slider = SliderRow("Polar Center X", "polar_x", vars.polar_x, -vars.image_width//2, vars.image_width//2, self.polar_x_cb, 'int', self.reset_slider_callback)
        polar_y_slider = SliderRow("Polar Center Y", "polar_y", vars.polar_y, -vars.image_height//2, vars.image_height//2, self.polar_y_cb, 'int', self.reset_slider_callback)

        self.sliders = [hue_slider, sat_slider, val_slider, alpha_slider, num_glitches_slider, glitch_size_slider, 
                val_threshold_slider, val_hue_shift_slider, blur_kernel_slider, x_shift_slider, y_shift_slider,r_shift_slider , perlin_amplitude_slider, perlin_frequency_slider, perlin_octaves_slider]

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
            "perlin_amplitud": perlin_amplitude_slider, 
            "perlin_frequency": perlin_frequency_slider, 
            "perlin_octaves": perlin_octaves_slider,
            "polar_x": polar_x_slider,
            "polar_y": polar_y_slider
        }

    def on_button_click(self, sender, app_data, user_data):
        print(f"Button clicked: {user_data}, {app_data}, {sender}")
        # Perform action based on button click
        if user_data == "save":
            on_save_button_click()
        elif user_data == "reset_all":
            reset_values()
        elif user_data == "random":
            randomize_values()
        elif user_data == "load_next":
            on_fwd_button_click()
        elif user_data == "load_prev":
            on_prev_button_click()
        elif user_data == "load_rand":
            on_rand_button_click()
        elif user_data == "interp":
            pass
        elif user_data == "fade":
            pass

    def create_buttons(self, width, height):

        save_button = Button("Save", "save")
        reset_button = Button("Reset all", 'reset_all')
        random_button = Button("Random", 'random')
        load_next_button = Button("Load >>", 'load_next')
        load_rand_button = Button("Load ??", "load_rand")
        load_prev_button = Button("Load <<", "load_prev")

        self.buttons = [save_button, reset_button, random_button, load_next_button, load_rand_button, load_prev_button]

        width -= 20
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

    def create_control_window(self, width=550, height=420):
        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars()
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)
