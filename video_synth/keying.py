from config import params, toggles

class Keying:
    def __init__(self, image_width: int, image_height: int):
        
        # TODO: implement keying parameters
        # This probably needs to be a separate class
        self.key_upper_hue = params.add("key_upper_hue", 0, 180, 0)
        self.key_lower_hue = params.add("key_lower_hue", 0, 180, 0)
        self.key_upper_sat = params.add("key_upper_sat", 0, 255, 255)
        self.key_lower_sat = params.add("key_lower_sat", 0, 255, 0)
        self.key_upper_val = params.add("key_upper_val", 0, 255, 255)
        self.key_lower_val = params.add("key_lower_val", 0, 255, 0)
        # self.key_fuzz = params.add("key_fuzz", 0, 100, 0)
        # self.key_invert = params.add("key_invert", 0, 1, 0)
        # self.key_feather = params.add("key_feather", 0, 100, 0)