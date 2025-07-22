import random

class ParamTable:
    """
    Manages a collection of Param objects, allowing access and manipulation
    of parameters by name or index.
    """
    def __init__(self):
        """Initializes an empty dictionary to store parameters."""
        self.params = {}
    
    def __getitem__(self, key):
        """
        Allows accessing parameters using dictionary-like syntax (e.g., params['name'])
        or list-like syntax (e.g., params[0]).
        """
        if isinstance(key, str):
            if key in self.params:
                return self.params[key]
            else:
                raise KeyError(f"Parameter '{key}' does not exist.")
        elif isinstance(key, int):
            # Allows indexing by integer for specific keys
            keys_list = list(self.params.keys())
            if 0 <= key < len(keys_list):
                return self.params[keys_list[key]]
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("Key must be a string or an integer")

    def val(self, param_name: str):
        """
        Returns the current value of the Param object with the given name.
        This method is retained for backward compatibility.
        """
        if param_name in self.params:
            # Access the value via the property
            return self.params[param_name].value
        else:
            raise ValueError(f"Parameter '{param_name}' does not exist.")
    
    def get(self, param_name: str, default=None):
        """
        Returns the Param object itself with the given name.
        This method is retained for backward compatibility.
        """
        if param_name in self.params:
            return self.params[param_name]
        elif default is not None:
            return default
        else:
            raise ValueError(f"Parameter '{param_name}' does not exist.")

    def set(self, param_name, value):
        """
        Sets the value of the parameter with the given name.
        This method is retained for backward compatibility.
        It now uses the Param's property setter.
        """
        if param_name in self.params:
            # Assigning to the 'value' property will trigger its setter logic
            self.params[param_name].value = value
            return self.params[param_name].value # Return the potentially clamped/modified value
        else:
            raise ValueError(f"Parameter '{param_name}' does not exist.")
        
    def all(self):
        """Returns a dictionary of all parameters."""
        return self.params
    
    def items(self):
        """Returns a view of the parameter items (name, Param object)."""
        return self.params.items()
    
    def keys(self):
        """Returns a view of the parameter names."""
        return self.params.keys()
    
    def values(self):
        """Returns a view of the Param objects."""
        return self.params.values()
    
    def add(self, name: str, min: int | float, max: int | float, default_val: int | float, family=None):
        """
        Adds a new parameter to the table.
        Args:
            name (str): The unique name of the parameter.
            min (int/float): The minimum allowed value for the parameter.
            max (int/float): The maximum allowed value for the parameter.
            default_val (int/float/bool): The default value for the parameter.
            family (str, optional): An optional family/group name for the parameter.
        Returns:
            Param: The newly created Param object.
        Raises:
            ValueError: If a parameter with the given name already exists.
        """
        if name not in self.params:
            self.params[name] = Param(name, min, max, default_val, family=family)
            return self.params[name]
        else:
            print(self.params) # Debugging print, as in original
            raise ValueError(f"Parameter '{name}' already exists.")
        
class Param:
    """
    Represents a single parameter with a name, min/max bounds, default value,
    and its current value. Includes clamping and type conversion.
    """
    def __init__(self, name, min, max, default_val, family=None):
        """
        Initializes a Param object.
        Args:
            name (str): The name of the parameter.
            min (int/float): The minimum allowed value.
            max (int/float): The maximum allowed value.
            default_val (int/float/bool): The default value.
            family (str, optional): The family/group the parameter belongs to.
        """
        self.name = name
        self.min = min
        self.max = max
        self.default_val = default_val
        self.family = "None" if family is None else family
        
        # Initialize the internal _value attribute using the setter
        # This ensures initial default_val is clamped and type-casted correctly
        self._value = None # Placeholder for the internal value
        self.value = default_val # Assigning to 'value' calls the setter

    @property
    def value(self):
        """
        Getter for the parameter's current value.
        Accessing `param_instance.value` will call this method.
        """
        return self._value

    @value.setter
    def value(self, new_value):
        """
        Setter for the parameter's current value.
        Assigning to `param_instance.value = new_val` will call this method.
        It handles clamping, type casting, and specific logic for certain parameters.
        """
        # 1. Clamp the new_value within min and max
        if new_value < self.min:
            clamped_value = self.min
        elif new_value > self.max:
            clamped_value = self.max
        else:
            clamped_value = new_value
        
        # 2. Type cast the clamped_value to match the default_val's type
        # This ensures consistency (e.g., if default_val is int, value remains int)
        if isinstance(self.default_val, float):
            self._value = float(clamped_value)
        elif isinstance(self.default_val, int):
            self._value = int(clamped_value)
        else:
            # For other types (like bool), directly assign
            self._value = clamped_value

        # 3. Apply specific logic for "blur_kernel_size"
        # Ensures blur kernel size is odd and at least 1
        if self.name == "blur_kernel_size":
            self._value = max(1, int(self._value) | 1) # Bitwise OR with 1 ensures it's odd

    def __str__(self):
        """String representation of the Param object."""
        return f"{self.name}: {self.value}"

    def __int__(self):
        """Allows casting Param object to an integer."""
        return int(self.value)

    def __float__(self):
        """Allows casting Param object to a float."""
        return float(self.value)

    # Arithmetic dunder methods for direct operations with Param objects or numbers
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
            # Handle division by zero for other Param's value
            if other.value == 0:
                raise ZeroDivisionError("Cannot divide by zero (other Param's value is zero)")
            return self.value / other.value
        elif isinstance(other, (int, float)):
            # Handle division by zero for numbers
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return self.value / other
        else:
            raise TypeError("Unsupported type for division")
    
    def reset(self):
        """Resets the parameter's value to its default value."""
        self.value = self.default_val # Assignment calls the setter
        return self.value
    
    def set_max(self, max):
        """Sets a new maximum value for the parameter and clamps the current value if necessary."""
        self.max = max
        # Assigning current value to itself will trigger the setter,
        # which will re-clamp it if it's now above the new max
        self.value = self.value
        return self.value
    
    def set_min(self, min):
        """Sets a new minimum value for the parameter and clamps the current value if necessary."""
        self.min = min
        # Assigning current value to itself will trigger the setter,
        # which will re-clamp it if it's now below the new min
        self.value = self.value
        return self.value

    def randomize(self):
        """Sets the parameter's value to a random value within its min/max range."""
        if isinstance(self.default_val, float):
            self.value = random.uniform(self.min, self.max) 
        elif isinstance(self.default_val, int):
            self.value = random.randint(self.min, self.max)
        else:
            # For other types (e.g., bool), you might need specific randomization logic
            pass
        return self.value
    
    # Retain val() for backward compatibility. It now calls the property getter.
    def val(self):
        """
        Returns the current value of the parameter. This method is retained for backward compatibility.
        It now internally calls the `value` property's getter.
        """
        return self.value # This accesses the @property getter
    
    def min_max(self):
        """Returns a tuple containing the minimum and maximum allowed values."""
        return (self.min, self.max)

# Example Usage (as a module)
if __name__ == "__main__":
    print("--- ParamTable and Param Class Demonstration ---")

    # Create a ParamTable instance
    params = ParamTable()

    # Add some parameters
    print("\nAdding parameters:")
    freq_param = params.add("frequency", 0.1, 10.0, 1.0, family="Oscillator")
    amp_param = params.add("amplitude", 0.0, 1.0, 0.5, family="Oscillator")
    blur_param = params.add("blur_kernel_size", 1, 101, 5, family="ImageFX")
    toggle_param = params.add("effect_on", 0, 1, 1, family="Control") # Using 0/1 for boolean-like behavior

    print("\nCurrent Parameters:")
    for name, param_obj in params.items():
        print(f"  {param_obj}") # Uses Param's __str__

    # Demonstrate accessing values using the new property syntax
    print("\nAccessing values via property:")
    print(f"  Frequency: {freq_param.value}")
    print(f"  Amplitude: {amp_param.value}")
    print(f"  Blur Kernel Size: {blur_param.value}")

    # Demonstrate setting values using the new property syntax
    print("\nSetting values via property:")
    freq_param.value = 5.75
    amp_param.value = 1.2 # This will be clamped to 1.0
    blur_param.value = 10 # This will be clamped to 11 (odd)
    toggle_param.value = 0 # Set to off

    print(f"  New Frequency: {freq_param.value}")
    print(f"  New Amplitude (clamped): {amp_param.value}")
    print(f"  New Blur Kernel Size (clamped to odd): {blur_param.value}")
    print(f"  Effect On: {toggle_param.value}")

    # Demonstrate backward compatibility with val() and set_value()
    print("\nDemonstrating backward compatibility (val() and set() methods):")
    print(f"  Frequency (using val()): {params.val('frequency')}")
    params.set("amplitude", 0.25)
    print(f"  Amplitude (using set()): {params.val('amplitude')}")

    # Demonstrate clamping behavior
    print("\nDemonstrating clamping:")
    freq_param.value = -0.5 # Will be clamped to 0.1
    print(f"  Frequency after setting to -0.5 (clamped): {freq_param.value}")
    blur_param.value = 1000 # Will be clamped to 101 (odd)
    print(f"  Blur Kernel Size after setting to 1000 (clamped): {blur_param.value}")

    # Demonstrate type casting
    print("\nDemonstrating type casting:")
    freq_param.value = 3 # Will be cast to float (default_val was float)
    print(f"  Frequency after setting to int 3 (now float): {freq_param.value} (type: {type(freq_param.value)})")
    blur_param.value = 5.7 # Will be cast to int (default_val was int), then made odd
    print(f"  Blur Kernel Size after setting to float 5.7 (now int and odd): {blur_param.value} (type: {type(blur_param.value)})")

    # Demonstrate reset and randomize
    print("\nDemonstrating reset and randomize:")
    freq_param.reset()
    print(f"  Frequency after reset: {freq_param.value}")
    amp_param.randomize()
    print(f"  Amplitude after randomize: {amp_param.value}")

    # Demonstrate arithmetic operations
    print("\nDemonstrating arithmetic operations:")
    param_sum = freq_param + amp_param
    print(f"  Frequency + Amplitude: {param_sum:.2f}")
    param_prod = blur_param * 2
    print(f"  Blur Kernel Size * 2: {param_prod}")

    # Demonstrate set_max/set_min
    print("\nDemonstrating set_max/set_min:")
    freq_param.set_max(5.0)
    print(f"  Frequency max set to 5.0. Current value: {freq_param.value}")
    freq_param.value = 6.0 # Should be clamped to 5.0
    print(f"  Frequency set to 6.0 (clamped): {freq_param.value}")

    freq_param.set_min(2.0)
    print(f"  Frequency min set to 2.0. Current value: {freq_param.value}")
    freq_param.value = 1.0 # Should be clamped to 2.0
    print(f"  Frequency set to 1.0 (clamped): {freq_param.value}")
