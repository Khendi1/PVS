
class Param:
    def __init__(self, name, min_val, max_val, default_val):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val
        print(f"Creating param {self.name} with min: {self.min_val}, max: {self.max_val}, default: {self.default_val}, type: '{str(type(self.default_val))}'")
        self.value = default_val

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        if isinstance(other, Param):
            return self.value + other.value
        elif isinstance(other, (int, float)):
            return self.value + other
        else:
            raise TypeError("Unsupported type for addition")
    
    def __sub__(self, other):
        if isinstance(other, Param):
            return self.value - other.value
        elif isinstance(other, (int, float)):
            return self.value - other
        else:
            raise TypeError("Unsupported type for subtraction")

    def __mul__(self, other):
        if isinstance(other, Param):
            return self.value * other.value
        elif isinstance(other, (int, float)):
            return self.value * other
        else:
            raise TypeError("Unsupported type for multiplication")
    
    def __truediv__(self, other):
        if isinstance(other, Param):
            return self.value / other.value
        elif isinstance(other, (int, float)):
            return self.value / other
        else:
            raise TypeError("Unsupported type for division")
    
    def reset(self):
        self.value = self.default_val
        return self.value
    
    def set_max(self, max_val):
        self.max_val = max_val
        if self.value > max_val:
            self.value = max_val
        return self.value
    
    def set_min(self, min_val):
        self.min_val = min_val
        if self.value < min_val:
            self.value = min_val
        return self.value

    def set_min_max(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        if self.value < min_val:
            self.value = min_val
        elif self.value > max_val:
            self.value = max_val
        return self.value
    
    def set_value(self, value):
        if value < self.min_val:
            self.value = self.min_val
        elif value > self.max_val:
            self.value = self.max_val
        else:
            self.value = value
        
        if isinstance(self.default_val, float):
            self.value = float(self.value)
        elif isinstance(self.default_val, int):
            self.value = int(self.value)
        
        return self.value
    
    def randomize(self):
        import random
        if isinstance(self.default_val, float):
            self.value = random.uniform(self.min_val, self.max_val) 
        else:
            random.randint(self.min_val, self.max_val)
                # for s in self.slider_dict.values():
        #     if s.tag == "blur_kernel": s.value = max(1, random.randint(1, s.max_value) | 1)
        #     elif s.tag == "x_shift": s.value = random.randint(-image_width, image_width)
        #     elif s.tag == "y_shift": s.value = random.randint(-image_height, image_height)
        #     elif s.tag == "glitch_size": s.value = random.randint(1, s.max_value)
        #     elif s.tag == 'feedback': s.value = random.uniform(0.0, 1.0)
        #     else: s.value = random.randint(s.min_value, s.max_value)
        return self.value

