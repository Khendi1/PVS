
import random
import math
from enum import Enum
import numpy as np
from config import params

def map_value(value, from_min, from_max, to_min, to_max):
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

  return math.floor(mapped_value)

class Interp(Enum):
    LINEAR = 1
    COSINE = 2
    CUBIC = 3

class PerlinNoise():
    def __init__(self, 
            seed, amplitude=1, frequency=1, 
            octaves=1, interp=Interp.COSINE, use_fade=False):
        self.seed = random.Random(seed).random()
        self.amplitude = params.add("perlin_amplitude", 0.1, 1000, amplitude) # min/max depends on linked param
        self.frequency = params.add("perlin_frequency", 0.1, 500, frequency)
        self.octaves = params.add("perlin_octaves", 1, 10, octaves)
        self.interp = interp
        self.use_fade = use_fade

        self.mem_x = dict()

    def __noise(self, x):
        # made for improve performance
        if x not in self.mem_x:
            self.mem_x[x] = random.Random(self.seed + x).uniform(-1, 1)
        return self.mem_x[x]

    def __interpolated_noise(self, x):
        prev_x = int(x) # previous integer
        next_x = prev_x + 1 # next integer
        frac_x = x - prev_x # fractional of x

        if self.use_fade:
            frac_x = self.__fade(frac_x)

        # intepolate x
        if self.interp is Interp.LINEAR:
            res = self.__linear_interp(
                self.__noise(prev_x), 
                self.__noise(next_x),
                frac_x)
        elif self.interp is Interp.COSINE:
            res = self.__cosine_interp(
                self.__noise(prev_x), 
                self.__noise(next_x),
                frac_x)
        else:
            res = self.__cubic_interp(
                self.__noise(prev_x - 1), 
                self.__noise(prev_x), 
                self.__noise(next_x),
                self.__noise(next_x + 1),
                frac_x)

        return res

    def get(self, x):
        frequency = self.frequency
        amplitude = self.amplitude
        result = 0
        for _ in range(self.octaves):
            result += self.__interpolated_noise(x * frequency) * amplitude
            frequency *= 2
            amplitude /= 2

        return result


    def __linear_interp(self, a, b, x):
        return a + x * (b - a)


    def __cosine_interp(self, a, b, x):
        x2 = (1 - math.cos(x * math.pi)) / 2
        return a * (1 - x2) + b * x2


    def __cubic_interp(self, v0, v1, v2, v3, x):
        p = (v3 - v2) - (v0 - v1)
        q = (v0 - v1) - p
        r = v2 - v0
        s = v1
        return p * x**3 + q * x**2 + r * x + s


    def __fade(self, x):
        # useful only for linear interpolation
        return (6 * x**5) - (15 * x**4) + (10 * x**3)

class Oscillator:
    def __init__(self, name, frequency, amplitude, phase, shape, seed=0):
        self.name = name
        self.param_max = 100
        self.param_min = 0
        self.frequency = params.add(f"{name}_frequency", 0, 2, frequency)
        self.amplitude = params.add(f"{name}_amplitude", 0, 100, amplitude)
        self.phase = params.add(f"{name}_phase", 0, 360, phase)
        self.seed = params.add(f"{name}_seed", 0, 100, seed)
        self.shape = params.add(f"{name}_shape", 0, 3, shape)
        self.sample_rate = 30
        self.direction = 1
        self.oscillator = self.create_oscillator()
        self.linked_param = None
        self.value = 0

    def get_next_value(self):
        """
        Gets the next value from the current waveform generator.

        Returns:
            The next sample value.
        """
        self.value = next(self.oscillator)
        return self.value
        
    def create_oscillator(self):
        """
        Creates a sine wave oscillator.

        Args:
            frequency (float): The frequency of the sine wave in Hz.
            amplitude (float): The amplitude of the sine wave (peak value).
            sample_rate (int): The number of samples per second.  This doesn't
                            directly apply to this visual modulation, but is here
                            for conceptual completeness and potential future audio
                            integration.

        Returns:
            generator: A generator that yields the next sample of the sine wave.
        """
        phase = 0
        while True:
            # Calculate the next sample of the sine wave
            if int(self.shape) == 0: # Sine wave
                sample = self.get_amplitude() * np.sin(2 * np.pi * self.get_frequency() * self.get_phase()) + self.get_seed()
            elif int(self.shape) == 1: # Square wave
                sample = self.get_amplitude() * np.sign(np.sin(2 * np.pi * self.get_frequency() * self.get_phase())) + self.get_seed()
            elif int(self.shape) == 2: # Triangle wave
                sample = self.get_amplitude() * 2 * np.abs(2 * (self.get_phase() * self.get_frequency() - np.floor(self.get_phase() * self.get_frequency() + 0.5))) - self.get_amplitude()  + self.get_seed()
            elif int(self.shape) == 3: # Sawtooth wave 
                sample = self.get_amplitude() * 2 * (self.get_phase() * self.get_frequency() - np.floor(self.get_phase() * self.get_frequency())) - self.get_amplitude()  + self.get_seed()
                sample *= self.direction
            else:
                raise ValueError(f"Invalid shape value. Must be 0 (sine), 1 (square), 2 (triangle), or 3 (sawtooth). got shape={self.shape}")

            if self.linked_param is not None:
                # sample += self.get_amplitude()/2
                # print(f"mappings from {self.param_min}-{self.param_max} to {self.linked_param.min_val}-{self.linked_param.max_val}")
                mapped_sample = map_value(sample, -self.param_max, self.param_max, self.linked_param.min_val, self.linked_param.max_val)
                self.linked_param.value = mapped_sample
            yield sample
            self.phase.value += 1 / self.sample_rate  # Increment phase.  Not directly time-based here.
        
    def update_params(self, freq=None, amp=None, phase=None, shape=None, direction=None, seed=None):
        if freq is not None:
            self.frequency.value = freq
        if amp is not None:
            self.amplitude.value = amp
        if phase is not None:
            self.phase.value = phase
        if shape is not None:
            self.shape.value = shape
        if self.direction is not None:
            self.direction = direction
        if self.seed is not None:
            self.seed.value = seed
    
    def get_frequency(self):
        return self.frequency.value
    
    def get_amplitude(self):
        return self.amplitude.value

    def get_phase(self):
        return self.phase.value

    def get_shape(self):
        return self.shape.value

    def get_direction(self):
        return self.direction

    def get_seed(self):
        return self.seed.value

    def link_param(self, param):
        """
        Links the oscillator parameters to a parameter object.

        Args:
            param (Param): The parameter object to link to.
        """
        print(f"Linking {self.name} to {param.name}")
        self.linked_param = param
        self.amplitude.max_val =  param.max_val
        self.amplitude.min_val =  param.min_val
        self.phase.max_val = param.max_val
        self.phase.min_val = param.min_val
        self.seed.max_val = param.max_val
        self.seed.min_val = param.min_val
    
    def unlink_param(self):
        """
        Unlinks the oscillator parameters from the parameter object.
        """
        self.linked_param = None
