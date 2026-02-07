import random
import logging
from common import *


class ParamTable:
    """
    Manages a collection of Param objects, allowing access and manipulation
    of parameters by name or index.
    """
    def __init__(self, group: str = "Global"):
        """Initializes an empty dictionary to store parameters."""
        self.group = group
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
                raise KeyError(f"Parameter '{key}' does not exist in ParamTable.group {self.group}.")
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
    
    def add(self, name: str, min: int | float = 0, max: int | float = 1,
            default: int | float = 0, subgroup=None, group=None, 
            type=Widget.SLIDER, options=None) -> 'Param':
        """
        Add a new Param to the table. Defaults to a slider widget type if not specified.
        Args:
            name (str): The unique name of the parameter.
            min (int/float): The minimum allowed value for the parameter.
            max (int/float): The maximum allowed value for the parameter.
            default (int/float/bool): The default value for the parameter.
            subgroup (str, optional): An optional subgroup/subgroup name for the parameter.
        Returns:
            Param: The newly created Param object.
        Raises:
            ValueError: If a parameter with the given name already exists.
        """
        if name not in self.params:
            self.params[name] = Param(name, min, max, default, group=group, subgroup=subgroup, type=type, options=options)
            return self.params[name]
        else:
            raise ValueError(f"Parameter '{name}' already exists.")
        

class Param:
    """
    Represents a single parameter with a name, min/max bounds, default value,
    and its current value. Includes clamping, type conversion, randomize, and reset methods.
    """
    def __init__(self, name, min=None, max=None, default=None, 
                 subgroup=None, group=None, 
                 type=Widget.SLIDER, options=None):
        """
        Initializes a Param object.
        Args:
            name (str): The name of the parameter. Must be unique within a ParamTable.
            min (int/float): The minimum allowed value. If None, defaults to 0.
            max (int/float): The maximum allowed value. If None, defaults to 1.
            default (int/float/bool): The default value. 
            group (str, optional): The group this parameter belongs to.
            subgroup (str, optional): The subgroup/subgroup the parameter belongs to.
            type (Widget): The type of widget to use for this parameter. Defaults to Widget.SLIDER.
        """
        self.name = name

        self.min = min if min is not None else 0
        self.max = max if max is not None else 1
        self.default = default if default is not None else int(0)

        self.group = group if group is not None else Groups.UNCATEGORIZED.name
        self.subgroup = subgroup if subgroup is not None else self.group

        self.type = type
        self.options = options

        self.linked_oscillator = None

        # Initialize the internal _value attribute using the setter
        # This ensures initial default is clamped and type-casted correctly
        self._value = None # Placeholder for the private/internal value
        self.value = self.default # Assigning to 'value' calls the setter


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
        clamped_value = new_value
        if self.type not in [Widget.DROPDOWN, Widget.RADIO]:
            # Clamp the new_value within min and max
            if new_value < self.min:
                clamped_value = self.min
            elif new_value > self.max:
                clamped_value = self.max
        
        # Type cast the clamped_value to match the default's type
        if isinstance(self.default, float):
            self._value = float(clamped_value)
        elif isinstance(self.default, int) and not isinstance(clamped_value, str):
             self._value = int(clamped_value)
        else:
            # For other types (like bool or string), directly assign
            self._value = clamped_value

        # Apply specific logic for certain params
        # Ensures blur kernel size is odd and at least 1
        if self.name == "blur_kernel_size":
            self._value = max(1, int(self._value) | 1) # Bitwise OR with 1 ensures it's odd


    def __repr__(self):
        """"""
        return f"Param(name={self.name}, min={self.min}, max={self.max}, default={self.default}, subgroup=None"


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
        self.value = self.default # Assignment calls the setter
        return self.value
    

    def randomize(self):
        """Sets the parameter's value to a random value within its min/max range."""
        if isinstance(self.default, float):
            self.value = random.uniform(self.min, self.max) 
        elif isinstance(self.default, int):
            self.value = random.randint(self.min, self.max)
        else:
            # For other types (e.g., bool), you might need specific randomization logic
            pass
        return self.value