import dearpygui.dearpygui as dpg
from save import SaveController
from mix import *
from config import effects
from gui_elements import TrackbarRow, ButtonsTable, Toggle
import logging

log = logging.getLogger(__name__)

class Interface:

    def __init__(self, params, osc_bank, toggles, panel_width=550, panel_height=420):
        self.params = params
        self.osc_bank = osc_bank
        self.toggles = toggles

        self.panel_width = panel_width
        self.panel_height = panel_height

        self.default_font_id = None
        self.global_font_id = None
        
        # TODO: debug automatically building params sliders
        # only creates last param in list
        # self.panels = self.build_panels_dict(params.all())


    def on_button_click(self, sender, app_data, user_data):
        log.info(f"Toggle clicked: {user_data}, {app_data}, {sender}")
        # Perform action based on button click
        # TODO: I don't like this, but it works for now
        if user_data == "reset_all":
            self.reset_values()
        elif user_data == "random":
            self.randomize_values()
        elif user_data == "enable_polar_transform":
            if enable_polar_transform == True:
                enable_polar_transform = False
            else:
                enable_polar_transform = True


    def reset_values(self):
        for param in self.params.values():
            param.reset()

            if dpg.does_item_exist(param.name):
                try:
                    # Get the item's command/type (checkbox, slider, etc)
                    info = dpg.get_item_info(param.name).get('command')
                    final_val = param.default_val
                    
                    if info == 'add_checkbox' or info == 'add_radio_button':
                        # Checkboxes/radio buttons require a Python bool
                        final_val = bool(param.name)
                        
                    elif info == 'add_input_int':
                        # Input int requires an integer
                        final_val = int(param.name)
                        
                    elif info == 'add_input_float' or info == 'add_slider_float':
                        # Float widgets require a float
                        final_val = float(param.name)
                        
                    # set the converted value
                    dpg.set_value(param.name, final_val)
                    
                except ValueError as e:
                    # Catches internal Python conversion errors (e.g., float("abc"))
                    log.error(f"Type Conversion Error for {param.name}: {e}")
                except Exception as e:
                    # Catches the DPG exception [1008] and allows the loop to continue
                    log.error(f"dpg.set_value failed for {param.name}: {e}")


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


    def resize_buttons(self, sender, app_data):
        # Get the current width of the window

        # TODO: "Controls" should not be hard coded as such
        window_width = dpg.get_item_width("Controls")
        
        # Set each button to half the window width (minus a small padding if you want)
        half_width = window_width // 2
        dpg.set_item_width(sender, half_width)


    def create_control_window(self, params, width=600, height=700, mixer=None, osc_bank=None):

        dpg.create_context()

        with dpg.window(tag="Controls", label="Controls", width=width, height=height):
            self.create_trackbars(width, height, mixer, osc_bank)
            # self.create_trackbar_panels_for_param()
            self.saver = SaveController(self.params, width, height).create_save_buttons()
            self.create_buttons(width, height)
            # dpg.set_viewport_resize_callback(resize_buttons)

        dpg.create_viewport(title='Controls', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Controls", True)


    def create_panels_from_list(self, source_obj_list, theme):
        for obj in source_obj_list:
            panel = obj.create_gui_panel(theme=theme)
     

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
    

    def create_control_theme(self):
        """Creates and returns a Maroon-colored collapsing header theme."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                # Header Background (Maroon: R=128, G=0, B=0)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (128, 0, 0, 100))
                # Header Hovered (Slightly lighter Maroon)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (150, 0, 0, 255))
                # Header Text (White)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        return theme
    
    def create_effect_theme(self):
        """Creates and returns a Dark Green-colored collapsing header theme."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                # Header Background (Dark Green: R=0, G=100, B=0)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 100, 0, 100))
                # Header Hovered (Slightly lighter Green)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 140, 0, 255))
                # Header Text (White)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        return theme

    def create_animation_theme(self):
        """Creates and returns a Dark Blue-colored collapsing header theme."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                # Header Background (Dark Blue: R=0, G=0, B=128)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 0, 128, 100))
                # Header Hovered (Slightly lighter Blue)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 0, 160, 255))
                # Header Text (White)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        return theme

    def create_trackbars(self, width, height, mixer, osc_bank):

        default_font_id, global_font_id = None, None
        effects_theme = self.create_effect_theme()
        animation_theme = self.create_animation_theme()
        control_theme = self.create_control_theme()

        mixer.create_gui_panel(control_theme)

        effects.reflector.create_gui_panel(default_font_id,global_font_id,effects_theme)
        effects.color.create_gui_panel(default_font_id, global_font_id,effects_theme)
        effects.feedback.create_gui_panel(default_font_id,global_font_id,effects_theme)
        effects.pixels.create_gui_panel(default_font_id,effects_theme)
        effects.ptz.create_gui_panel(default_font_id,global_font_id,effects_theme)
        effects.sync.create_gui_panel(default_font_id,global_font_id,effects_theme)
        effects.glitch.create_gui_panel(default_font_id,global_font_id,effects_theme)
        effects.shapes.create_gui_panel(default_font_id, global_font_id,effects_theme)
        effects.warp.create_gui_panel(default_font_id, global_font_id,effects_theme)
        effects.patterns.create_gui_panel(default_font_id, global_font_id)
        # self.perlin_generator_sliders(default_font_id, global_font_id)
        # self.lissajous_sliders(default_font_id, global_font_id)
        
        self.create_panels_from_list(mixer.animation_sources.values(), animation_theme)
        
        osc_bank.create_gui_panel(default_font_id, global_font_id, control_theme)
    

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