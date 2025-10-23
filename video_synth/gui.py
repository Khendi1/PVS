import dearpygui.dearpygui as dpg
from save import SaveController
from mix import *
from globals import effects
from gui_elements import TrackbarRow, ButtonsTable, Toggle
import logging

log = logging.getLogger(__name__)

class Interface:

    def __init__(self, params, osc_bank, toggles, panel_width=550, panel_height=420):
        self.sliders = []
        self.buttons = []
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.slider_dict = None
        self.default_font_id = None
        self.global_font_id = None
        self.params = params
        self.osc_bank = osc_bank
        self.toggles = toggles
        # TODO: debug automatically building params sliders
        # only creates last param in list
        # self.panels = self.build_panels_dict(params.all())

    # TODO: this has been moved to the trackbar class; remove after moving all create slider functions to their respective classes
    def reset_slider_callback(self, sender, app_data, user_data):
        param = self.params.get(str(user_data))
        if param is None:
            print(f"Slider or param not found for {user_data}")
            return
        print(f"Got reset callback for {user_data}; setting to default value {param.default_val}")
        param.reset()
        dpg.set_value(user_data, param.value)


    def on_toggle_button_click(self, sender, app_data, user_data):
        print(f"test: {user_data}")
        for tag, button in self.toggles.items():
            if user_data == tag:
                print("test2")
                self.toggles.toggle(tag)


    def on_button_click(self, sender, app_data, user_data):
        print(f"Toggle clicked: {user_data}, {app_data}, {sender}")
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
        print(self.params)
        for param in self.params.params.values():
            print(param.name)
            param.reset()
            dpg.set_value(param.name, param.value)


    def randomize_values(self):
        for param in self.params.all():
            param.randomize()
            dpg.set_value(param.name, param.value)


    def create_buttons(self, width, height):
        reset_button = Toggle("Reset all", 'reset_all')
        random_button = Toggle("Random", 'random')

        width -= 20

        with dpg.group(horizontal=True):
            dpg.add_button(label=reset_button.label, callback=self.on_button_click, user_data=reset_button.tag, width=width//3)
            dpg.add_button(label=random_button.label, tag=random_button.tag, callback=self.on_button_click, user_data=random_button.tag, width=width//3)
            self.toggles["effects_first"].create()
            # self.toggles["save"].create()
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
        for i in range(self.osc_bank.len):
            if f"osc{i}" in sender:
                param = None
                for tag, param in self.params.items():
                    if tag == app_data:
                        self.osc_bank[i].linked_param = param
                        break


    def create_control_window(self, params, width=550, height=600, mixer=None):

        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars(width, height, mixer)
            # self.create_trackbar_panels_for_param()
            self.saver = SaveController(self.params, width, height).create_save_buttons()
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)


    def create_panels_from_list(self, source_obj_list):
        for obj in source_obj_list:
            panel = obj.create_gui_panel()
     

    def perlin_generator_sliders(self, default_font_id=None, global_font_id=None):
        with dpg.collapsing_header(label=f"\tNoise Generator", tag="noise_generator"):
            
            perlin_amplitude_slider = TrackbarRow(
                "Perlin Amplitude", 
                self.params.get("perlin_amplitude"), 
                default_font_id)
            
            perlin_frequency_slider = TrackbarRow(
                "Perlin Frequency", 
                self.params.get("perlin_frequency"), 
                default_font_id)
            
            perlin_octaves_slider = TrackbarRow(
                "Perlin Octaves", 
                self.params.get("perlin_octaves"), 
                default_font_id)
            
        dpg.bind_item_font("noise_generator", global_font_id)
    
    
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

        for i in range(self.osc_bank.len):
            
            with dpg.collapsing_header(label=f"\tOscillator {i}", tag=f"osc{i}"):
                osc_shape_sliders.append(TrackbarRow(
                    f"Osc {i} Shape", 
                    self.osc_bank[i].shape, 
                    default_font_id))
                
                osc_freq_sliders.append(TrackbarRow(
                    f"Osc {i} Freq", 
                    self.osc_bank[i].frequency, 
                    default_font_id))
                
                osc_amp_sliders.append(TrackbarRow(
                    f"Osc {i} Amp", 
                    self.osc_bank[i].amplitude, 
                    default_font_id))
                
                osc_phase_sliders.append(TrackbarRow(
                    f"Osc {i} Phase", 
                    self.osc_bank[i].phase,
                    default_font_id))
                
                osc_seed_sliders.append(TrackbarRow(
                    f"Osc {i} Seed", 
                    self.osc_bank[i].seed, 
                    default_font_id))
                
                osc_noise_octaves.append(TrackbarRow(
                    f"Osc {i} Noise Octaves",
                    self.osc_bank[i].noise_octaves,
                    default_font_id))
                
                osc_noise_persistence.append(TrackbarRow(
                    f"Osc {i} Noise Persistence",
                    self.osc_bank[i].noise_persistence,
                    default_font_id))
                
                osc_noise_lacunarity.append(TrackbarRow(
                    f"Osc {i} Noise Lacunarity",
                    self.osc_bank[i].noise_lacunarity,
                    default_font_id))
                
                osc_noise_repeat.append(TrackbarRow(
                    f"Osc {i} Noise Repeat",
                    self.osc_bank[i].noise_repeat,
                    default_font_id))
                
                osc_noise_base.append(TrackbarRow(
                    f"Osc {i} Noise Base",
                    self.osc_bank[i].noise_base,
                    default_font_id))
                
                # Create a list of items for the listbox
                items = list(self.params.keys())

                # Create the listbox
                dpg.add_combo(items=items,
                                label="Select Parameter",
                                tag=f"osc{i}_combobox",
                                default_value=None,
                                callback=self.listbox_cb)

            dpg.bind_item_font(f"osc{i}", global_font_id)


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

        default_font_id, global_font_id = None, None

        self.test_sliders(default_font_id, global_font_id)
        mixer.mix_panel()

        effects.color.create_gui_panel(default_font_id, global_font_id)
        effects.feedback.create_gui_panel(default_font_id,global_font_id)
        effects.glitch.create_gui_panel(default_font_id,global_font_id)
        effects.reflector.create_gui_panel(default_font_id,global_font_id)
        effects.ptz.create_gui_panel(default_font_id,global_font_id)
        effects.sync.create_gui_panel(default_font_id,global_font_id)
        effects.noise.create_gui_panel(default_font_id,global_font_id)
        effects.shapes.create_gui_panel(default_font_id,global_font_id)
        # fx_dict[FX.PATTERNS].create_gui_panel(default_font_id, global_font_id)
        # fx_dict[FX.WARP].create_gui_panel(default_font_id, global_font_id)
        # fx_dict[FX.PIXELS].create_gui_panel(default_font_id, global_font_id)
        # fx_dict[FX.LISSAJOUS].create_gui_panel(default_font_id, global_font_id)

        self.create_panels_from_list(mixer.animation_sources.values())
        self.osc_sliders(default_font_id, global_font_id)
        
        # self.perlin_generator_sliders(default_font_id, global_font_id)
        # self.lissajous_sliders(default_font_id, global_font_id)

    # under test; not currently used
    def build_panels_dict(self, params):
        panels = {}
        for p in self.params.values():
            if p.name not in panels.keys():
                panels[str(p.family)] = []
            panels[str(p.family)].append(p.name)
                
        return panels

    # under test; not currently used
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
                    self.params.get(p), 
                    default_font_id))
                
            dpg.bind_item_font(panel_name, global_font_id)