
import random
import math
from enum import Enum, IntEnum
import numpy as np
import noise
from param import Param, ParamTable
import logging
from common import Groups, Widget

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

#   log.info(f'\t{value}->{mapped_value}')

  if round_down:
    return math.floor(mapped_value)
  else:
    return mapped_value  # Return the mapped value without rounding


class LFOShape(Enum):
    NONE = 0
    SINE = 1
    SQUARE = 2
    TRIANGLE = 3
    SAWTOOTH = 4
    PERLIN = 5


class LFO:
    def __init__(self, params: ParamTable, name, frequency, amplitude, phase, shape, seed=0, linked_param_name=None, max_amplitude=100, min_amplitude=-100):
        self.name = name
        group = None
        subgroup = None
        self.param_max = max_amplitude
        self.param_min = min_amplitude
        self.shape = params.add(f"{name}_shape",
                                min=0, max=len(LFOShape)-1, default=shape,
                                group=group, subgroup=subgroup,
                                type=Widget.DROPDOWN, options=LFOShape)
        self.frequency = params.add(f"{name}_frequency",
                                    min=0, max=1, default=float(frequency),
                                    subgroup=subgroup, group=group)
        self.amplitude = params.add(f"{name}_amplitude",
                                    min=min_amplitude, max=max_amplitude, default=amplitude,
                                    subgroup=subgroup, group=group)
        self.phase = params.add(f"{name}_phase",
                                min=0, max=360, default=phase,
                                subgroup=subgroup, group=group)
        self.seed = params.add(f"{name}_seed",
                               min=0, max=100, default=seed,
                               subgroup=subgroup, group=group) # TODO: Change ambiguous name to something more descriptive

        self.noise_octaves = params.add(f"{name}_noise_octaves",
                                        min=1, max=10, default=6,
                                        subgroup=subgroup, group=group)
        self.noise_persistence = params.add(f"{name}_noise_persistence",
                                            min=0.1, max=1.0, default=0.5,
                                            subgroup=subgroup, group=group)
        self.noise_lacunarity = params.add(f"{name}_noise_lacunarity",
                                           min=1.0, max=2.0, default=2.0,
                                           subgroup=subgroup, group=group)
        self.noise_repeat = params.add(f"{name}_noise_repeat",
                                       min=1, max=1000, default=100,
                                       subgroup=subgroup, group=group)
        self.noise_base = params.add(f"{name}_noise_base",
                                     min=0, max=1000, default=456,
                                     subgroup=subgroup, group=group)

        self.cutoff_min = params.add(f"{name}_cutoff_min",
                                      min=min_amplitude, max=max_amplitude, default=min_amplitude,
                                      subgroup=subgroup, group=group)
        self.cutoff_max = params.add(f"{name}_cutoff_max",
                                      min=min_amplitude, max=max_amplitude, default=max_amplitude,
                                      subgroup=subgroup, group=group)

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


            match shape:
                case LFOShape.NONE.value:
                    sample = 0
                case LFOShape.SINE.value:
                    sample = amp * np.sin(arg) + seed
                    sample *= direction
                case LFOShape.SQUARE.value:
                    sample = amp * np.sign(np.sin(arg)) + seed
                case LFOShape.TRIANGLE.value:
                    sample = amp * (2 / np.pi) * np.arcsin(np.sin(arg)) + seed
                    sample *= direction
                case LFOShape.SAWTOOTH.value:
                    sample = amp * (2 * (t * freq - np.floor(t * freq + 0.5))) + seed
                    sample *= direction
                case LFOShape.PERLIN.value:
                    noise_input_x = (t * self.frequency.value) # % self.noise_repeat
                    sample = noise.pnoise1(
                        noise_input_x,
                        octaves=self.noise_octaves.value,
                        persistence=self.noise_persistence.value,
                        lacunarity=self.noise_lacunarity.value,
                        repeat=self.noise_repeat.value, # This ensures the noise repeats after 'noise_repeat' units
                        base=self.noise_base.value
                    )
                    log.debug(sample)
                case _:
                    raise ValueError(f"Invalid shape value. Must be 0 (NONE), 1 (SINE), 2 (SQUARE), 3 (TRIANGLE), 4 (SAWTOOTH), or 5 (PERLIN). got shape={shape}")
        
            if self.linked_param is not None:
                if shape == LFOShape.PERLIN.value:
                    # TODO: handle perlin noise mapping using the map_value method
                    mapped_sample = self._scale_value(self.linked_param, sample, in_min=-1.0, in_max=1.0)
                elif isinstance(self.linked_param.default, float):
                    # mapped_sample = self._scale_value(self.linked_param, sample, in_min=-1.0, in_max=1.0) #* #self.amplitude.value
                    mapped_sample = map_value(round(sample, 5), -amp + seed, amp + seed, self.linked_param.min, self.linked_param.max, round_down=False)
                elif isinstance(self.linked_param.default, int):
                    mapped_sample = map_value(sample, -amp + seed, amp + seed, self.linked_param.min, self.linked_param.max)

                # Apply cutoff clamping
                mapped_sample = np.clip(mapped_sample, self.cutoff_min.value, self.cutoff_max.value)

                self.linked_param.value = mapped_sample
                # log.debug(f'{sample} mapped to {mapped_sample} for {self.linked_param.name}')
            
            t += 1 / self.sample_rate  # Increment time by sample period 
            yield sample
        
    def link_param(self, param: Param):
        """
        Links the oscillator parameters to a parameter object.

        Args:
            param (Param): The parameter object to link to.
        """
        # log.debug(f"Linking {self.name} to {param.name}")
        self.linked_param = param
        self.amplitude.max =  param.max
        self.amplitude.min =  param.min
        self.phase.max = param.max
        self.phase.min = param.min
        self.seed.max = param.max
        self.seed.min = param.min
        self.cutoff_min.max = param.max
        self.cutoff_min.min = param.min
        self.cutoff_min.value = param.min
        self.cutoff_max.max = param.max
        self.cutoff_max.min = param.min
        self.cutoff_max.value = param.max
        self.group = param.group
        self.subgroup = param.subgroup

    def unlink_param(self):
        """
        Unlinks the oscillator parameters from the parameter object.
        """
        if self.linked_param:
            self.linked_param.linked_oscillator = None
        self.linked_param = None


class OscBank():
    def __init__(self, params, num_osc):
        self.len = num_osc
        self.oscillators = []
        self.params = params
        temp = [self.oscillators.append(LFO(params=params, name=f"osc{i}", frequency=0.5, amplitude=1.0, phase=0.0, shape=i%4)) \
                         for i in range(num_osc)]        


    def update(self):
        for osc in self.oscillators:
            if osc.linked_param is not None:
                osc.get_next_value()


    def __getitem__(self, index):
        """
        This method allows accessing elements using square bracket notation.
        It delegates the actual access to the internal list.
        """
        return self.oscillators[index]


    def __len__(self):
        """
        This method allows using len() on the object.
        """
        return len(self.oscillators)


    def __setitem__(self, index, value):
        """
        This method allows modifying elements using square bracket notation.
        """
        self.oscillators[index] = value


    def __delitem__(self, index):
        """
        This method allows deleting elements using the del keyword.
        """
        del self.oscillators[index]


    def add_oscillator(self, name, subgroup=None, group=None, frequency=0.5, amplitude=1.0, phase=0.0, shape=1):
        osc = LFO(params=self.params, name=name, frequency=frequency, amplitude=amplitude, phase=phase, shape=shape)
        self.oscillators.append(osc)
        self.len = len(self.oscillators)
        return osc


    def remove_oscillator(self, osc):
        # When removing an oscillator, we need to clean up its parameters from the main param table
        param_names_to_remove = [
            f"{osc.name}_frequency",
            f"{osc.name}_amplitude",
            f"{osc.name}_phase",
            f"{osc.name}_seed",
            f"{osc.name}_shape",
            f"{osc.name}_noise_octaves",
            f"{osc.name}_noise_persistence",
            f"{osc.name}_noise_lacunarity",
            f"{osc.name}_noise_repeat",
            f"{osc.name}_noise_base",
            f"{osc.name}_cutoff_min",
            f"{osc.name}_cutoff_max"
        ]
        for param_name in param_names_to_remove:
            if param_name in self.params.params:
                del self.params.params[param_name]

        self.oscillators.remove(osc)
        self.len = len(self.oscillators)


    def _shape_callback(self, sender, app_data, user_data):
        for i in range(len(self.oscillators)):
            if str(i) in user_data[:-3]:
                self.oscillators[i].shape.value = LFOShape[app_data].value
