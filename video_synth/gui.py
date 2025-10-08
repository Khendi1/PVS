from config import *
import dearpygui.dearpygui as dpg
from save import SaveController
from buttons import Buttons, Button
from mix import *
# from fx import *
from shared_objects import fx, FX
from sliders import TrackbarRow, TrackbarCallback, TrackbarRow2

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

    # TODO: this has been moved to the trackbar class; remove after moving all create slider functions to their respective classes
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
        # TODO: I don't like this, but it works for now
        if user_data == "save":
            self.saver.save2()
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
        for param in params.params:
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
            # toggles["save"].create()
            dpg.add_button(label="Save", tag="save", callback=self.on_button_click, user_data="save", width=width//3)


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


    def create_control_window(self, width=550, height=600, mixer=None):

        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars(width, height, mixer)
            # self.create_trackbar_panels_for_param()
            self.saver = SaveController(width, height).create_save_buttons()
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)


    def mix_panel(self, mixer):
        with dpg.collapsing_header(label=f"\tMixer", tag="mixer"):
            dpg.add_text("Video Source 1")
            dpg.add_combo(list(MixSources.__members__.keys()), default_value="INTERNAL_WEBCAM", tag="source_1", callback=mixer.select_source1_callback)
            # Initially hide the input text for file path 1 as webcam is default
            dpg.add_input_text(label="Video File Path 1", tag="file_path_source_1", default_value=mixer.default_video_file_path, show=False)
            
            dpg.add_text("Video Source 2")
            dpg.add_combo(list(MixSources.__members__.keys()), default_value="METABALLS", tag="source_2", callback=mixer.select_source2_callback)
            dpg.add_input_text(label="Video File Path 2", tag="file_path_source_2", default_value=mixer.default_video_file_path)
            dpg.add_spacer(height=10)

            dpg.add_text("Mixer")
            # dpg.add_slider_float(label="Blending", default_value=alpha, min_value=0.0, max_value=1.0, callback=alpha_callback, format="%.2f")
            
            blend_mode_slider = TrackbarRow(
                "Blend Mode",
                params.get("blend_mode"),
                TrackbarCallback(params.get("blend_mode"), "blend_mode").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            frame_blend_slider = TrackbarRow(
                "Frame Blend",
                params.get("frame_blend"),
                TrackbarCallback(params.get("frame_blend"), "frame_blend").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            upper_hue_key_slider = TrackbarRow(
                "Upper Hue Key",
                params.get("upper_hue"),
                TrackbarCallback(params.get("upper_hue"), "upper_hue").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            lower_hue_key_slider = TrackbarRow(
                "Lower Hue Key",
                params.get("lower_hue"),
                TrackbarCallback(params.get("lower_hue"), "lower_hue").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            upper_sat_slider = TrackbarRow(
                "Upper Sat Key",
                params.get("upper_sat"),
                TrackbarCallback(params.get("upper_sat"), "upper_sat").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            lower_sat_slider = TrackbarRow(
                "Lower Sat Key",
                params.get("lower_sat"),
                TrackbarCallback(params.get("lower_sat"), "lower_sat").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            upper_val_slider = TrackbarRow(
                "Upper Val Key",
                params.get("upper_val"),
                TrackbarCallback(params.get("upper_val"), "upper_val").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            lower_val_slider = TrackbarRow(
                "Lower Val Key",
                params.get("lower_val"),
                TrackbarCallback(params.get("lower_val"), "lower_val").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            luma_threshold_slider = TrackbarRow(
                "Luma Threshold",
                params.get("luma_threshold"),
                TrackbarCallback(params.get("luma_threshold"), "luma_threshold").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
            
            luma_selection_slider = TrackbarRow(
                "Luma Selection",
                params.get("luma_selection"),
                TrackbarCallback(params.get("luma_selection"), "luma_selection").__call__,
                self.reset_slider_callback,
                None) # fix defulat font_id=None
        

    def metaballs_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tMetaballs", tag="metaballs"):
            num_metaballs_slider = TrackbarRow(
                "Num Metaballs",
                params.get("num_metaballs"),
                TrackbarCallback(params.get("num_metaballs"), "num_metaballs").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            min_radius_slider = TrackbarRow(
                "Min Radius",
                params.get("min_radius"),
                TrackbarCallback(params.get("min_radius"), "min_radius").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            max_radius_slider = TrackbarRow(
                "Max Radius",
                params.get("max_radius"),
                TrackbarCallback(params.get("max_radius"), "max_radius").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            max_speed_slider = TrackbarRow(
                "Max Speed",
                params.get("max_speed"),
                TrackbarCallback(params.get("max_speed"), "max_speed").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            threshold_slider = TrackbarRow(
                "Threshold",
                params.get("threshold"),
                TrackbarCallback(params.get("threshold"), "threshold").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            smooth_coloring_max_field_slider = TrackbarRow(
                "Smooth Coloring Max Field",
                params.get("smooth_coloring_max_field"),
                TrackbarCallback(params.get("smooth_coloring_max_field"), "smooth_coloring_max_field").__call__,
                self.reset_slider_callback,
                default_font_id)
            
            feedback_alpha_slider = TrackbarRow(
                "Feedback Alpha",
                params.get("metaballs_feedback"),
                TrackbarCallback(params.get("metaballs_feedback"), "metaballs_feedback").__call__,
                self.reset_slider_callback,
                default_font_id)

        dpg.bind_item_font("metaballs", global_font_id)


    def perlin_generator_sliders(self, default_font_id=None, global_font_id=None):
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
    

    def plasma_sliders(self, default_font_id=None, global_font_id=None):
        plasma_freq_sliders = []
        plasma_amp_sliders = []
        plasma_phase_sliders = []
        plasma_seed_sliders = []
        plasma_shape_sliders = []
        plasma_params = [
            "plasma_speed",
            "plasma_distance",
            "plasma_color_speed",
            "plasma_flow_speed",
        ]
        with dpg.collapsing_header(label=f"\plasma Oscillator", tag="plasma_oscillator"):
            for i in range(len(plasma_params)):
                with dpg.collapsing_header(label=f"\t{plasma_params[i]}", tag=f"{plasma_params[i]}"):
                    plasma_shape_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Shape", 
                        params.get(f"{plasma_params[i]}_shape"), 
                        TrackbarCallback(params.get(f"{plasma_params[i]}_shape"), f"{plasma_params[i]} _shape").__call__, 
                        self.reset_slider_callback, 
                        default_font_id))
                    plasma_freq_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Freq", 
                        params.get(f"{plasma_params[i]}_frequency"), 
                        TrackbarCallback(params.get(f"{plasma_params[i]}_frequency"), f"{plasma_params[i]}_frequency").__call__, 
                        self.reset_slider_callback, 
                        default_font_id))
                    
                    plasma_amp_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Amp", 
                        params.get(f"{plasma_params[i]}_amplitude"), 
                        TrackbarCallback(params.get(f"{plasma_params[i]}_amplitude"), f"{plasma_params[i]}_amplitude").__call__, 
                        self.reset_slider_callback, 
                        default_font_id))
                    
                    plasma_phase_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Phase", 
                        params.get(f"{plasma_params[i]}_phase"), 
                        TrackbarCallback(params.get(f"{plasma_params[i]}_phase"), f"{plasma_params[i]}_phase").__call__, 
                        self.reset_slider_callback, 
                        default_font_id))
                    
                    plasma_seed_sliders.append(TrackbarRow(
                        f"{plasma_params[i]} Seed", 
                        params.get(f"{plasma_params[i]}_seed"), 
                        TrackbarCallback(params.get(f"{plasma_params[i]}_seed"), f"{plasma_params[i]}_seed").__call__, 
                        self.reset_slider_callback, 
                        default_font_id))
                dpg.bind_item_font(f"{plasma_params[i]}", global_font_id)
        dpg.bind_item_font("plasma_oscillator", global_font_id)

    
    def osc_sliders(self, default_font_id=None, global_font_id=None):
        
        osc_freq_sliders = []
        osc_amp_sliders = []
        osc_phase_sliders = []
        osc_seed_sliders = []
        osc_shape_sliders = []
        osc_noise_octaves = []
        osc_noise_persistence = []
        osc_noise_lacunarity = []
        osc_noise_repeat = []
        osc_noise_base = []
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
                
                osc_noise_octaves.append(TrackbarRow(
                    f"Osc {i} Noise Octaves",
                    osc_bank[i].noise_octaves,
                    TrackbarCallback(osc_bank[i].noise_octaves, f"osc{i}_noise_octaves").__call__,
                    self.reset_slider_callback, 
                    default_font_id))
                
                osc_noise_persistence.append(TrackbarRow(
                    f"Osc {i} Noise Persistence",
                    osc_bank[i].noise_persistence,
                    TrackbarCallback(osc_bank[i].noise_persistence, f"osc{i}_noise_persistence").__call__,
                    self.reset_slider_callback,
                    default_font_id))
                
                osc_noise_lacunarity.append(TrackbarRow(
                    f"Osc {i} Noise Lacunarity",
                    osc_bank[i].noise_lacunarity,
                    TrackbarCallback(osc_bank[i].noise_lacunarity, f"osc{i}_noise_lacunarity").__call__,
                    self.reset_slider_callback,
                    default_font_id))
                
                osc_noise_repeat.append(TrackbarRow(
                    f"Osc {i} Noise Repeat",
                    osc_bank[i].noise_repeat,
                    TrackbarCallback(osc_bank[i].noise_repeat, f"osc{i}_noise_repeat").__call__,
                    self.reset_slider_callback,
                    default_font_id))
                
                osc_noise_base.append(TrackbarRow(
                    f"Osc {i} Noise Base",
                    osc_bank[i].noise_base,
                    TrackbarCallback(osc_bank[i].noise_base, f"osc{i}_noise_base").__call__,
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


    def reaction_diffusion_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tReaction Diffusion", tag="reaction_diffusion"):
            rd_diffusion_rate_a_slider = TrackbarRow(
                "Diffusion Rate A",
                params.get("da"),
                TrackbarCallback(params.get("da"), "da").__call__,
                self.reset_slider_callback,
                default_font_id)
            rd_diffusion_rate_b_slider = TrackbarRow(
                "Diffusion Rate B",
                params.get("db"),
                TrackbarCallback(params.get("db"), "db").__call__,
                self.reset_slider_callback,
                default_font_id)
            rd_feed_rate_slider = TrackbarRow(
                "Feed Rate",
                params.get("feed"),
                TrackbarCallback(params.get("feed"), "feed").__call__,
                self.reset_slider_callback,
                default_font_id)
            rd_kill_rate_slider = TrackbarRow(
                "Kill Rate",
                params.get("kill"),
                TrackbarCallback(params.get("kill"), "kill").__call__,
                self.reset_slider_callback,
                default_font_id)
        dpg.bind_item_font("reaction_diffusion", global_font_id)


    def moire_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tMoire", tag="moire"):
            pass
        dpg.bind_item_font("moire", global_font_id)


    def test_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tTest", tag="test"):
            pass
        dpg.bind_item_font("test", global_font_id)


    def shader_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tShader", tag="shader"):
            pass
        dpg.bind_item_font("shader", global_font_id)


    def create_trackbars(self, width, height, mixer):

        with dpg.font_registry():
            global_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 18) # Larger font size for the header
            default_font_id = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14) # Default font size for other items
        dpg.bind_font(default_font_id)

        self.test_sliders(default_font_id, global_font_id)
        self.mix_panel(mixer)

        # TODO: initialize using for loop over dict keys
        fx[FX.COLOR].create_sliders(default_font_id, global_font_id)
        fx[FX.BASIC].create_sliders(default_font_id, global_font_id)
        fx[FX.GLITCH].create_sliders(default_font_id, global_font_id)
        fx[FX.REFLECTOR].create_sliders(default_font_id, global_font_id)
        fx[FX.PTZ].create_sliders(default_font_id, global_font_id)
        fx[FX.SYNC].create_sliders(default_font_id, global_font_id)
        fx[FX.PATTERNS].create_sliders(default_font_id, global_font_id)
        fx[FX.NOISE].create_sliders(default_font_id, global_font_id)
        fx[FX.WARP].create_sliders(default_font_id, global_font_id)
        # fx[FX.PIXELS].create_sliders(default_font_id, global_font_id)
        fx[FX.SHAPES].create_sliders(default_font_id, global_font_id)
        # TODO: fix temp sliders
        fx[FX.BASIC].temp_create_sliders(default_font_id, global_font_id)


        self.metaballs_sliders(default_font_id, global_font_id)
        self.osc_sliders(default_font_id, global_font_id)
        
        # self.moire_sliders(default_font_id, global_font_id)
        # self.plasma_sliders(default_font_id, global_font_id)
        # self.reaction_diffusion_sliders(default_font_id, global_font_id)
        # self.perlin_generator_sliders(default_font_id, global_font_id)
        # self.shape_generator_sliders(default_font_id, global_font_id)
        # self.lissajous_sliders(default_font_id, global_font_id)
        # self.perlin_generator_sliders(default_font_id, global_font_id)


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

