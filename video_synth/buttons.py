
import dearpygui.dearpygui as dpg

class Button:
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
    
class Buttons:
    def __init__(self):
        self.buttons = {
            "invert": Button("Invert", "invert", default_val=False),
            "polar_transform": Button("Polar Transform", "polar_transform", default_val=False),
            "grayscale": Button("Grayscale", "grayscale", default_val=False),
            "edge_detect": Button("Edge Detect", "edge_detect", default_val=False),
            "cartoonify": Button("Cartoonify", "cartoonify", default_val=False),
            "posterize": Button("Posterize", "posterize", default_val=False),
            "sepia": Button("Sepia", "sepia", default_val=False),
            "solarize": Button("Solarize", "solarize", default_val=False),
            "negative": Button("Negative", "negative", default_val=False),
            "swirl": Button("Swirl", "swirl", default_val=False),
            "fisheye": Button("Fisheye", "fisheye", default_val=False),
            "kaleidoscope": Button("Kaleidoscope", "kaleidoscope", default_val=False),
            "pixelate": Button("Pixelate", "pixelate", default_val=False),
            "vignette": Button("Vignette", "vignette", default_val=False),
            "noise_overlay": Button("Noise Overlay", "noise_overlay", default_val=False),
            "effects_first": Button("Effects First", "effects_first", default_val=False),
            "shapes": Button("Shapes", "shapes", default_val=False)
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

    def add_button(self, label, tag, default_val=False):
        self.buttons[tag] = Button(label=label, tag=tag, default_val=default_val)
        
    def val(self, tag):
        if tag in self.buttons:
            return self.buttons[tag].val
        else:
            print(f"Button with tag '{tag}' not found.")
        return None
    
    def get(self, tag):
        return self.buttons[tag]
    
    def toggle(self, tag):
        if tag in self.buttons:
            self.buttons[tag].toggle()
            print(f'Toggling {tag} to {self.buttons[tag].val}')
    
    def items(self):
        return self.buttons.items()