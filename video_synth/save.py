import dearpygui.dearpygui as dpg
from config import params
from datetime import datetime
import yaml

class SaveButtons:
    """
    A class to handle the save, load, and randomize buttons in the GUI.
    It provides methods to save current parameter values, load next/previous values,
    and randomize parameter values.
    """
    def __init__(self, width, height):

        self.index = 0
        self.save = Button("Save", "save")
        self.fwd = Button("Load Next", "load_next")
        self.prev = Button("Load Prev", "load_prev")
        self.rand = Button("Load Random", "load_rand")

    def on_fwd_button_click(self):

        print(f"Forward button clicked!")

        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = (self.index + 1) % len(saved_values.all())
            self.load_param_vals(saved_values)
            
        except Exception as e:
            print(f"Error loading values: {e}")

    def on_prev_button_click(self):

        print(f"Prev button clicked!")

        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = (self.index - 1) % len(saved_values[0])
            self.load_param_vals(saved_values)

        except Exception as e:
            print(f"Error loading values: {e}")

    def on_rand_button_click(self):
        print(f"Random button clicked!")
    
        # get values from saved_values.yaml
        try:
            with open("saved_values.yaml", "r") as f:
                saved_values = list(yaml.safe_load_all(f))

            self.index = random.randint(0, len(saved_values[0]) - 1)
            self.load_param_vals(saved_values)
            
        except Exception as e:
            print(f"Error loading values: {e}")
    
    def load_param_vals(self, saved_values):
        d = saved_values[0][self.index]
        print(f"loaded values at index {self.index}: {d}\n\n")
        for param_name in params.keys():
            for tag in d.keys():
                if tag == param_name:
                    params[param_name].set(d[tag])
                    dpg.set_value(param_name, d[tag])

    def on_save_button_click(self, frame):
        date_time_str = datetime.now().strftime("%m-%d-%Y %H-%M")
        print(f"Saving values at {date_time_str}")
        
        data = {}
        for k, v in params.items():
            print(f"{k}: {v.value}")
            data[k] = v.value
        
        # Append the data to the YAML file
        with open("saved_values.yaml", "a") as f:
            yaml.dump([data], f, default_flow_style=False)
        
        # Optionally, save the modified image
        # TODO: determine best way to grap a frame for saving w/o having to pass
        # cv2.imwrite(f"{date_time_str}.jpg", frame)

    def create_save_buttons(self, width, height):
        """
        Creates the save, load, and randomize buttons in the GUI.
        """
        width -= 20

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.save.label, callback=self.on_button_click, user_data=self.save.tag, width=width//3)
            dpg.add_button(label=self.fwd.label, callback=self.on_button_click, user_data=self.fwd.tag, width=width//3)
            dpg.add_button(label=self.prev.label, callback=self.on_button_click, user_data=self.prev.tag, width=width//3)

        with dpg.group(horizontal=True):
            dpg.add_button(label=self.rand.label, callback=self.on_button_click, user_data=self.rand.tag, width=width//3)
