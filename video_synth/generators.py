
import random
import math
from enum import Enum
import numpy as np
from config import params
import noise

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
    def __init__(self, name, frequency, amplitude, phase, shape, seed=0, linked_param_name=None, max_amplitude=100, min_amplitude=-100):
        self.name = name
        self.param_max = max_amplitude
        self.param_min = min_amplitude
        self.frequency = params.add(f"{name}_frequency", 0, 2, frequency)
        self.amplitude = params.add(f"{name}_amplitude", self.param_min, self.param_max, amplitude)
        self.phase = params.add(f"{name}_phase", 0, 360, phase)
        self.seed = params.add(f"{name}_seed", 0, 100, seed)
        self.shape = params.add(f"{name}_shape", 0, 4, shape) # TODO: use an enum for shape
        
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
    
    def _scale_value(self, param, value, in_min=-1.0, in_max=1.0):
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
            freq = self.get_frequency()
            amp = self.get_amplitude()
            phase_offset = self.get_phase()
            seed = self.get_seed()
            shape = int(self.get_shape())
            direction = self.get_direction()

            # Calculate the argument for the waveform functions
            arg = 2 * np.pi * freq * t + np.deg2rad(phase_offset)

            if shape == 0:  # Sine wave
                sample = amp * np.sin(arg) + seed
                sample *= direction
            elif shape == 1:  # Square wave
                sample = amp * np.sign(np.sin(arg)) + seed
            elif shape == 2:  # Triangle wave
                sample = amp * (2 / np.pi) * np.arcsin(np.sin(arg)) + seed
                sample *= direction
            elif shape == 3:  # Sawtooth wave
                sample = amp * (2 * (t * freq - np.floor(t * freq + 0.5))) + seed
                sample *= direction
            elif shape == 4:  # Perlin noise             
                # For a "smoothly evolving" Perlin noise over time, the 'x' input to pnoise1
                # should continuously increase. To make it "loop" or repeat after a certain period,
                # you can use the repeat parameter in pnoise1 and modulo the input.
                
                # Let's say we want one full "cycle" of the Perlin noise to take
                # a certain amount of time. The input to pnoise1 usually represents
                # a spatial coordinate. For a temporal signal, time becomes that coordinate.
                # The 'repeat' parameter in pnoise1 is key for looping.

                # The effective "speed" of the Perlin noise evolution can be controlled by
                # how fast the input `x` changes.
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
                if shape == 4:
                    mapped_sample = self._scale_value(self.linked_param, sample, in_min=-1.0, in_max=1.0) * self.amplitude.value
                elif isinstance(self.linked_param.default_val, float):
                    mapped_sample = map_value(round(sample, 5), self.param_min, self.param_max, self.linked_param.min, self.linked_param.max, round_down=False)
                elif isinstance(self.linked_param.default_val, int):
                    mapped_sample = map_value(sample, self.param_min, self.param_max, self.linked_param.min, self.linked_param.max)
        
                self.linked_param.value = mapped_sample
                print(f'{sample} mapped to {mapped_sample} for linked param {self.linked_param.name}')
            yield sample
            t += 1 / self.sample_rate  # Increment time by sample period 
        
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