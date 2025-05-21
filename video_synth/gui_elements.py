import dearpygui.dearpygui as dpg

class Button:
    def __init__(self, label, tag, max=None):
        self.label = label
        self.tag = tag
        self.user_data = tag
        self.index = 0
        self.max = max
        self.callback = None

    def __plus__(self, value):
        if self.max is not None:
            self.index = (self.index + value) % self.max
        else:
            self.index += value

    def __minus__(self, value):
        if self.max is not None:
            self.index = (self.index + value) % self.max
        else:
            self.index += value

    def __eq__(self, value):
        self.index = value # be careful
    
class SliderRow:
    def __init__(self, label, tag, default_value, min_value, max_value, callback, type, button_callback):
        self.label = label
        self.tag = tag
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.callback = callback
        self.button_callback = button_callback
        self.slider = None
        self.button = None
        self.value = default_value
        self.type = type
        self.create()

    def create(self):
        with dpg.group(horizontal=True):
            # self.button = dpg.add_button(label="Reset", callback=lambda x: self.reset, width=50)
            self.button = dpg.add_button(label="Reset", callback=self.button_callback, width=50, tag=self.tag + "_reset", user_data=self.tag)
            if self.type == 'float':
                self.slider = dpg.add_slider_float(label=self.label, tag=self.tag, default_value=self.default_value, min_value=self.min_value, max_value=self.max_value, callback=self.callback, width=-100)
            else:
                self.slider = dpg.add_slider_int(label=self.label, tag=self.tag, default_value=self.default_value, min_value=self.min_value, max_value=self.max_value, callback=self.callback, width=-100)
