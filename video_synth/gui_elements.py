import dearpygui.dearpygui as dpg
from config import params

class TrackbarRow:

    def __init__(self, label, param, callback, button_callback, font):

        self.label = label #TODO: method to get label from param name
        self.tag = param.name
        self.callback = callback
        self.button_callback = button_callback
        self.slider = None
        self.button = None
        self.value = param.default_val
        self.type = type(param.default_val).__name__
        self.param = param
        self.font = font
        self.create()

    def create(self):
        with dpg.group(horizontal=True):
            # self.button = dpg.add_button(label="Reset", callback=lambda x: self.reset, width=50)
            self.button = dpg.add_button(label="Reset", callback=self.button_callback, width=50, tag=self.tag + "_reset", user_data=self.tag)
            if self.type == 'float':
                self.slider = dpg.add_slider_float(label=self.label, tag=self.tag, 
                                                   default_value=self.param.default_val, 
                                                   min_value=self.param.min, 
                                                   max_value=self.param.max, 
                                                   callback=self.callback, 
                                                   width=-100)
            else:
                self.slider = dpg.add_slider_int(label=self.label, tag=self.tag, default_value=self.param.default_val, min_value=self.param.min, max_value=self.param.max, callback=self.callback, width=-100)
            dpg.bind_item_font(self.tag, self.font)
            dpg.bind_item_font(self.tag + "_reset", self.font)

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
        Args: 
            sender: The tag/ID of the trackbar that triggered the callback.
            app_data: The new value of the trackbar.
        """
        # Update the Param object's value
        params.set(self.target_param.name, app_data)
        dpg.set_value(sender, app_data)
