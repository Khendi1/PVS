
import random
import math
from enum import Enum
import numpy as np
import noise
from param import Param
import logging

log = logging.getLogger(__name__)


def map_value(value, from_min, from_max, to_min, to_max, round_down=True):
  """
  Maps a value from one range to another.

  Args:
    value: The value to map.
    from_min: The minimum value of the original range.
    from_max: The maximum value of the original range.
    to_min: The minimum value of the target range.
    to_max: The maximum value of the target range.
  Returns:
    The mapped value in the target range, rounded down to the nearest integer.
  """
  # Calculate the proportion of the value within the original range
  proportion = (value - from_min) / (from_max - from_min)

  # Map the proportion to the target range
  mapped_value = to_min + proportion * (to_max - to_min)

  if round_down:
    return math.floor(mapped_value)
  else:
    return mapped_value  # Return the mapped value without rounding

class OscillatorShape(Enum):
    NONE = 0
    SINE = 1
    SQUARE = 2
    TRIANGLE = 3
    SAWTOOTH = 4
    PERLIN = 5

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            value = value.lower()
            if value == "sine":
                return cls.SINE
            elif value == "square":
                return cls.SQUARE
            elif value == "triangle":
                return cls.TRIANGLE
            elif value == "sawtooth":
                return cls.SAWTOOTH
            elif value == "perlin":
                return cls.PERLIN
        return cls(value)

class Oscillator:
    def __init__(self, params, name, frequency, amplitude, phase, shape, seed=0, linked_param_name=None, max_amplitude=100, min_amplitude=-100):
        self.name = name
        self.param_max = max_amplitude
        self.param_min = min_amplitude
        self.frequency = params.add(f"{name}_frequency", 0, 2, frequency)
        self.amplitude = params.add(f"{name}_amplitude", -100, 100, amplitude)
        # self.amplitude = params.add(f"{name}_amplitude", self.param_min, self.param_max, amplitude)
        self.phase = params.add(f"{name}_phase", 0, 360, phase)
        self.seed = params.add(f"{name}_seed", 0, 100, seed) # TODO: Change ambiguous name to something more descriptive
        self.shape = params.add(f"{name}_shape", 0, len(OscillatorShape)-1, shape)
        
        self.noise_octaves = params.add(f"{name}_noise_octaves", 1, 10, 6)
        self.noise_persistence = params.add(f"{name}_noise_persistence", 0.1, 1.0, 0.5)
        self.noise_lacunarity = params.add(f"{name}_noise_lacunarity", 1.0, 2.0, 2.0)
        self.noise_repeat = params.add(f"{name}_noise_repeat", 1, 1000, 100)
        self.noise_base = params.add(f"{name}_noise_base", 0, 1000, 456)

        self.sample_rate = 30
        self.direction = 1 
        self.oscillator = self.create_oscillator()
        self.linked_param = None
        self.value = 0

        # Link to a parameter if provided
        if linked_param_name is not None:
            self.linked_param = params.get(linked_param_name)
            if self.linked_param is None:
                raise ValueError(f"Linked parameter '{linked_param_name}' not found in params.")
            self.link_param(self.linked_param)

    def get_next_value(self, map=False):
        """
        Gets the next value from the current waveform generator.

        Returns:
            The next sample value.
        """
        self.value = next(self.oscillator)
        return self.value
    
    def _scale_value(self, param: Param, value: int | float, in_min=-1.0, in_max=1.0):
        """Scales a value from the noise range to the oscillator's param_min/max."""
        return (value - in_min) * ((param.max - param.min) / (in_max - in_min)) + param.min

    def create_oscillator(self):
        """
        Creates a waveform oscillator using time instead of phase.

        Returns:
            generator: A generator that yields the next sample of the waveform.
        """
        t = 0.0

        while True:
            freq = self.frequency.value
            amp = self.amplitude.value
            phase_offset = self.phase.value
            seed = self.seed.value
            shape = int(self.shape.value)
            direction = self.direction

            # Calculate the argument for the waveform functions
            arg = 2 * np.pi * freq * t + np.deg2rad(phase_offset)

            if shape == 0:  # null wave
                pass
            elif shape == 1:  # sine wave
                sample = amp * np.sin(arg) + seed
                sample *= direction
            elif shape == 2:  # square wave
                sample = amp * np.sign(np.sin(arg)) + seed
            elif shape == 3:  # triangle wave
                sample = amp * (2 / np.pi) * np.arcsin(np.sin(arg)) + seed
                sample *= direction
            elif shape == 4:  # Sawtooth wave
                sample = amp * (2 * (t * freq - np.floor(t * freq + 0.5))) + seed
                sample *= direction
            elif shape == 5:  # Perlin noise             
                
                # noise_input_x = self._time / 10.0 # Adjust divisor for desired speed/smoothness
                noise_input_x = (t * self.frequency.value) # % self.noise_repeat

                sample = noise.pnoise1(
                    noise_input_x,
                    octaves=self.noise_octaves.value,
                    persistence=self.noise_persistence.value,
                    lacunarity=self.noise_lacunarity.value,
                    repeat=self.noise_repeat.value, # This ensures the noise repeats after 'noise_repeat' units
                    base=self.noise_base.value
                )

            else:
                raise ValueError(f"Invalid shape value. Must be 0 (sine), 1 (square), 2 (triangle), or 3 (sawtooth). got shape={shape}")

            if self.linked_param is not None:
                if shape == 5:
                    # TODO: handle perlin noise mapping using the map_value method
                    mapped_sample = self._scale_value(self.linked_param, sample, in_min=-1.0, in_max=1.0) * self.amplitude.value
                elif isinstance(self.linked_param.default_val, float):
                    # mapped_sample = self._scale_value(self.linked_param, sample, in_min=-1.0, in_max=1.0) #* #self.amplitude.value
                    mapped_sample = map_value(round(sample, 5), self.param_min, self.param_max, self.linked_param.min, self.linked_param.max, round_down=False)
                elif isinstance(self.linked_param.default_val, int):
                    mapped_sample = map_value(sample, self.param_min, self.param_max, self.linked_param.min, self.linked_param.max)
        
                self.linked_param.value = mapped_sample
                # print(f'{sample} mapped to {mapped_sample} for linked param {self.linked_param.name}')
            yield sample
            t += 1 / self.sample_rate  # Increment time by sample period 
        
    def link_param(self, param: Param):
        """
        Links the oscillator parameters to a parameter object.

        Args:
            param (Param): The parameter object to link to.
        """
        log.debug(f"Linking {self.name} to {param.name}")
        self.linked_param = param
        self.amplitude.max =  param.max
        self.amplitude.min =  param.min
        self.phase.max = param.max
        self.phase.min = param.min
        self.seed.max = param.max
        self.seed.min = param.min
    
    def unlink_param(self):
        """
        Unlinks the oscillator parameters from the parameter object.
        """
        self.linked_param = None

    # @property
    # def value(self):
    #     """Returns the current calculated value of the oscillator."""
    #     return self._current_value

class OscBank():
    def __init__(self, params, num_osc):
        self.len = num_osc
        self.osc_bank = []
        temp = [self.osc_bank.append(Oscillator(params=params, name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4)) \
                         for i in range(num_osc)]        
        log.info(f"Oscillator bank initialized with {len(self.osc_bank)} oscillators.")

    def update(self):
        for osc in self.osc_bank:
            if osc.linked_param is not None:
                osc.get_next_value()

    def __getitem__(self, index):
        """
        This method allows accessing elements using square bracket notation.
        It delegates the actual access to the internal list.
        """
        return self.osc_bank[index]

    def __len__(self):
        """
        This method allows using len() on the object.
        """
        return len(self.osc_bank)

    def __setitem__(self, index, value):
        """
        This method allows modifying elements using square bracket notation.
        """
        self.osc_bank[index] = value

    def __delitem__(self, index):
        """
        This method allows deleting elements using the del keyword.
        """
        del self.osc_bank[index]