import random
import logging
from config import WidgetType


class ParamTable:
    """
    Manages a collection of Param objects, allowing access and manipulation
    of parameters by name or index.
    """
    def __init__(self):
        """Initializes an empty dictionary to store parameters."""
        self.params = {}

    def __repr__(self):
        return self.params
    
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

    def __delitem__(self, key):
        """
        Enables item deletion using del obj[key]
        """
        del self.params[key]

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
    
    def add(self, name: str, min: int | float, max: int | float, default_val: int | float, subclass=None, parent=None, type=WidgetType.SLIDER, options=None) -> 'Param':
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
            self.params[name] = Param(name, min, max, default_val, family=subclass, parent=parent, type=type, options=options)
            return self.params[name]
        else:
            raise ValueError(f"Parameter '{name}' already exists.")
        
class Param:
    """
    Represents a single parameter with a name, min/max bounds, default value,
    and its current value. Includes clamping and type conversion.
    """
    def __init__(self, name, min, max, default_val, family=None, parent=None, type=WidgetType.SLIDER, options=None):
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
        self.parent = parent
        self.type = type
        self.options = options

        
        # Initialize the internal _value attribute using the setter
        # This ensures initial default_val is clamped and type-casted correctly
        self._value = None # Placeholder for the private/internal value
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
        # Clamp the new_value within min and max
        if new_value < self.min:
            clamped_value = self.min
        elif new_value > self.max:
            clamped_value = self.max
        else:
            clamped_value = new_value
        
        # Type cast the clamped_value to match the default_val's type
        if isinstance(self.default_val, float):
            self._value = float(clamped_value)
        elif isinstance(self.default_val, int):
            self._value = int(clamped_value)
        else:
            # For other types (like bool), directly assign
            self._value = clamped_value

        # Apply specific logic for certain params
        # Ensures blur kernel size is odd and at least 1
        if self.name == "blur_kernel_size":
            self._value = max(1, int(self._value) | 1) # Bitwise OR with 1 ensures it's odd

    def __repr__(self):
        """"""
        return f"Param(name={self.name}, min={self.min}, max={self.max}, default_val={self.default_val}, family=None"

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
            raise TypeError(f"Unsupported type '{type(other)}' for multiplication")
    
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
    
    def min_max(self):
        """Returns a tuple containing the minimum and maximum allowed values."""
        return (self.min, self.max)