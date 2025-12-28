import dearpygui.dearpygui as dpg
import logging
from enum import Enum

log = logging.getLogger(__name__)

RESET_BUTTON_WIDTH = 50
SLIDER_WIDTH = -100

""" A helper method to return a dictionary from an enumeration. Primarily used to pass options arg to RadioButtonRow"""
def dict_from_enum(cls: Enum):
    return {member.name: member.value for member in cls}

class RadioButtonRow:
    """ 
    This class creates a row of radio buttions. It can be used in a similar way to th TrackbarRow,
    with the difference being in the labeling of discrete states with a semantic label.
    
    Args:
        options: 
    """
    def __init__(self, label: str = None, cls: Enum = None, param=None, font=None, callback=None):
        
        self.label = label
        self.options = dict_from_enum(cls)
        self.param = param
        self.font = font

        dpg.add_radio_button(
            list(self.options.keys()), 
            callback=self.callback if callback is None else callback, 
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
    