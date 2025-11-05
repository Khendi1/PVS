import dearpygui.dearpygui as dpg
import logging
from enum import Enum

log = logging.getLogger(__name__)

RESET_BUTTON_WIDTH = 50
SLIDER_WIDTH = -100

""" A helper method to return a dictionary from an enumeration. Primarily used to pass options arg to RadioButtonRow"""
def dict_from_enum(cls: Enum):
    return {member.name: member.value for member in cls}

class TrackbarRow:
    """
    A TrackbarRow is a 
    """


    def __init__(self, label, param, font):

        self.label = label
        self.tag = param.name
        self.callback = TrackbarCallback(param, param.name).__call__
        self.slider = None
        self.button = None
        self.value = param.default_val
        self.type = type(param.default_val).__name__
        self.param = param
        self.font = font
        self.create()

    def create(self):
        """Slider and reset buttion creation logic"""
        with dpg.group(horizontal=True):
            
            self.button = dpg.add_button(label="Reset", 
                                         callback=self.reset_slider_callback, 
                                         width=50, 
                                         tag=self.tag + "_reset", 
                                         user_data=self.tag)
            
            if self.type == 'float':
                self.slider = dpg.add_slider_float(label=self.label, tag=self.tag, 
                                                   default_value=self.param.default_val, 
                                                   min_value=self.param.min, 
                                                   max_value=self.param.max, 
                                                   callback=self.callback, 
                                                   width=SLIDER_WIDTH)
            else:
                self.slider = dpg.add_slider_int(label=self.label, 
                                                 tag=self.tag, 
                                                 default_value=self.param.default_val, 
                                                 min_value=self.param.min, 
                                                 max_value=self.param.max, 
                                                 callback=self.callback, 
                                                 width=SLIDER_WIDTH)
            
            dpg.bind_item_font(self.tag, self.font)
            dpg.bind_item_font(self.tag + "_reset", self.font)

    def reset_slider_callback(self, sender, app_data, user_data):
        param = self.param
        log.info(f"Got reset callback for {user_data}; setting to default value {param.default_val}")
        if param is None:
            log.warning(f"Slider or param not found for {user_data}")
            return
        param.reset()
        dpg.set_value(user_data, param.value)

class TrackbarCallback:
    """
    A callable class instance used as a callback for Dear PyGui trackbars.
    This prevents having to write a callback for each individual trackbar
    It updates a specified Param object's value and an associated text item.
    """
    def __init__(self, param_obj, display_text_tag=None):
        """
        Initializes the callback instance.
        Args:
            param_obj (Param): The Param object whose 'value' attribute
                                      this trackbar will control.
            display_text_tag (str, optional): The tag of a dpg.add_text item
                                            to update with the current value.
        """
        self.param = param_obj
        self.display_text_tag = display_text_tag

    def __call__(self, sender, app_data):
        """
        This method is invoked when the trackbar's value changes.
        Args: 
            sender: The tag/ID of the trackbar that triggered the callback.
            app_data: The new value of the trackbar.
        """
        # Update the Param object's value
        self.param.value = app_data
        dpg.set_value(sender, app_data)

class RadioButtonRow:
    """ 
    This class creates a row of radio buttions. It can be used in a similar way to th TrackbarRow,
    with the difference being in the labeling of discrete states with a semantic label.
    
    Args:
        options: 
    """
    def __init__(self, label: str = None, cls: Enum = None, param=None, font=None):
        
        self.label = label
        self.options = dict_from_enum(cls)
        self.param = param
        self.font = font

        dpg.add_radio_button(
            list(self.options.keys()), 
            callback=self.callback, 
            horizontal=True,
            user_data=self.param.name,
        )
    

    def callback(self, sender, app_data, user_data):
        self.param.value = self.options[app_data]


class Toggle:
    def __init__(self, label, tag, default_val=False, font=None):
        self.label = label
        self.tag = tag
        self.user_data = tag
        self.callback = None
        self.value = default_val
    
    def val(self):
        return self.value
    
    def toggle(self):
        self.value = not self.value

    def on_toggle_button_click(self, sender, app_data, user_data):
        self.toggle()

    def create(self):
        dpg.add_button(label=self.label, tag=self.tag, callback=self.on_toggle_button_click, user_data=self.tag)


class ButtonsTable:
    """
    A singleton class to create and store Toggle instances
    """
    def __init__(self):
        self.buttons = {
            "effects_first": Toggle("Effects First", "effects_first", default_val=False),
            "shapes": Toggle("Shapes", "shapes", default_val=False)
        }

    def __getitem__(self, key):
        """
        This method is called when an item is accessed using obj[key].
        """
        if isinstance(key, str):
            return self.buttons[key]
        elif isinstance(key, int):
            # Example: allow indexing by integer for specific keys
            keys_list = list(self.buttons.keys())
            if 0 <= key < len(keys_list):
                return self.buttons[keys_list[key]]
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("Key must be a string or an integer")

    def add(self, label, tag, default_val=False):
        self.buttons[tag] = Toggle(label=label, tag=tag, default_val=default_val)
        return self.buttons[tag]
        
    def val(self, tag):
        if tag in self.buttons:
            return self.buttons[tag].val
        else:
            log.info(f"Toggle with tag '{tag}' not found.")
        return None
    
    def get(self, tag):
        return self.buttons[tag]
    
    def toggle(self, tag):
        if tag in self.buttons:
            self.buttons[tag].toggle()
            log.info(f'Toggling {tag} to {self.buttons[tag].val}')
    
    def items(self):
        return self.buttons.items()
    