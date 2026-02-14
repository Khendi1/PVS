import mido
import time
import threading
import time
import logging
from abc import ABC, abstractmethod
from param import ParamTable

log = logging.getLogger(__name__)

MIN = 0
MAX = 127

#NOTE: any new midi controllers must expose a class-level name attribute and append to this list
CONTROLLER_NAMES = []


""" Identifies midi controller ports from controller_names"""
def identify_midi_ports(controller_names: list[str], *params: ParamTable):
    
    # Mido uses a default backend (often python-rtmidi) which handles platform differences.    
    controllers = []
    ports_found = False

    log.info(f"Scanning for MIDI input and output ports for {controller_names}...")
    
    try:
        # List all available MIDI input ports. Ex: ["MIDI Mix 0", "SMC Mixer 1"]
        input_ports = mido.get_input_names()
        # Note that the controller_names list should follow a similar naming convention,
        # with the port number omitted

        string = "Found MIDI Input Device Ports:"
        if input_ports:

            # attempt to match found ports with known controller names
            for i, port_name in enumerate(input_ports):
                for name in controller_names:
                    if name in port_name:
                        found_controller = MidiControllerInterface(params, name=name, port_name=port_name)
                        # add to list of controllers so threads can be gracefully stopped on exit
                        controllers.append(found_controller)
                        # build output string with formatting so we only have to log once
                        string += f"\n\t{port_name}: initialized with class {name}"
                    else:
                        string += f"\n\t{port_name}: No matching controller class found"
        
            log.info(string)
            ports_found = True
        else:
            log.info(f"{string}: None")
        
        # List all available MIDI output ports
        output_ports = mido.get_output_names()
        if output_ports:
            string = "Found MIDI Output Devices:"
            for i, name in enumerate(output_ports):
                string += (f"\n\t[{i+1}] {name}")
            ports_found = True
            log.info(string)
            
    except Exception as e:
        log.exception(f"\nAn unexpected error occurred during port scan: {e}")
        return    
    if not ports_found:
        log.warning("No MIDI ports found by the operating system.")

    return controllers


class MidiProcessor:
    """
    A class to handle the processing of MIDI messages,
    including mapping, acceleration, and smoothing.
    """
    def __init__(self, min_midi=0, max_midi=127,
                 base_smoothing=0.05, acceleration_factor=0.05):
        """
        Initializes the MIDI processor with mapping and smoothing parameters.

        Args:
            min_midi (int): Minimum raw MIDI CC value (typically 0).
            max_midi (int): Maximum raw MIDI CC value (typically 127).
            min_output (int/float): Minimum value for the mapped output range.
            max_output (int/float): Maximum value for the mapped output range.
            base_smoothing (float): The base smoothing factor (alpha) for
                                    exponential smoothing. A smaller value (e.g., 0.01)
                                    means more smoothing (slower response) at low speeds.
                                    Should be between 0 and 1.
            acceleration_factor (float): Multiplier for how much input speed
                                         increases the smoothing factor. Higher values
                                         mean more "acceleration" or responsiveness
                                         to rapid movements.
        """
        MIN_midi = min_midi
        MAX_midi = max_midi

        self.base_smoothing = base_smoothing
        self.acceleration_factor = acceleration_factor
        self.time_epsilon = 0.001 # Small value to prevent division by zero in time_delta calculations

        # State variables for processing
        self.current_mapped_value = 0.0
        self.last_midi_value = None
        self.last_message_abs_time = time.time()


    def map_range(self, value, out_min, out_max):
        """
        Maps a value from one numerical range to another.

        Args:
            value (float): The input value to map.
            in_min (float): The minimum of the input range.
            in_max (float): The maximum of the input range.
            out_min (float): The minimum of the output range.
            out_max (float): The maximum of the output range.

        Returns:
            float: The mapped value in the output range.
        """
        return (value - MIN) * (out_max - out_min) / (MAX - MIN) + out_min


    def process_message(self, control, value, min_output=-150, max_output=150):
        """
        Processes an incoming MIDI message. If it's a control change message,
        it applies mapping, acceleration, and smoothing to its value.

        Args:
            msg (mido.Message): The incoming MIDI message.

        Returns:
            float or None: The processed (smoothed and accelerated) value if
                           it's a control change message, otherwise None.
        """
        midi_cc_value = value

        # Calculate actual time delta between messages for speed calculation
        current_abs_time = time.time()
        time_delta = current_abs_time - self.last_message_abs_time
        self.last_message_abs_time = current_abs_time

        # Calculate change in MIDI value
        midi_value_change = 0
        if self.last_midi_value is not None:
            midi_value_change = midi_cc_value - self.last_midi_value
        self.last_midi_value = midi_cc_value

        # Map raw MIDI CC value to the target output range (-150 to 150)
        target_mapped_value = self.map_range(midi_cc_value, min_output, max_output)

        # Calculate dynamic smoothing factor based on input speed
        # Input speed is defined as the absolute change in MIDI value per unit of time.
        # A higher speed (larger abs(midi_value_change) and smaller time_delta)
        # should lead to a higher effective_smoothing_factor (less smoothing, faster response).
        
        effective_time_delta = max(time_delta, self.time_epsilon) # Prevent division by zero
        input_speed = abs(midi_value_change) / effective_time_delta

        # Adjust smoothing factor (alpha): faster input means less smoothing (higher alpha)
        # The dynamic_alpha will range from base_smoothing to 1.0.
        dynamic_alpha = min(1.0, self.base_smoothing + (input_speed * self.acceleration_factor))

        # Apply exponential smoothing to the current mapped value
        # current_mapped_value = alpha * new_value + (1 - alpha) * old_smoothed_value
        self.current_mapped_value = (dynamic_alpha * target_mapped_value) + \
            ((1 - dynamic_alpha) * self.current_mapped_value)

        log.debug(f"MIDI CC {control}: Raw={value}, Target={target_mapped_value:.2f}, "
                f"Smoothed={self.current_mapped_value:.2f}, TimeDelta={time_delta:.4f}, "
                f"InputSpeed={input_speed:.2f}, Alpha={dynamic_alpha:.2f}")
        
        return self.current_mapped_value


class ControllerBase(ABC):
    """ Abstract class to enforce common interface for setting params """
    @abstractmethod
    def set_values(self):
        """Must be implemented by any concrete subgroup."""
        pass


class SMC_Mixer(ControllerBase):
    """
    A class to represent the SMC-Mixer and handle the 
    mapping of encoders, faders, and buttons to pages of parameters.

    mappings:
        encoders:   30-37
        faders:     40-47

        m:          20-27   next mode (blur, noise, fractal, etc.)
        s:          28-35   (DO NOT USE 30-35)
        r:          36-43   (DO NOT USE 36-67, 40-43)
        square:     44-51   (DO NOT USE 44-47)
        
        play        52      save
        pause       53      load random from save
        record      54      load random params

        reverse     55      load previous encoder page
        forward     56      load next encoder page

        previous    57      load previous fader page
        next        58      load next fader page

        up          59      
        down        60      

        left        61
        right       62

    """
    name = "SMC-Mixer"
    CONTROLLER_NAMES.append(name)

    def __init__(self, port_name, params):
        """
        Initializes the SMC_Mixer instance.
        """
        self.params = params

        self.port_name = port_name 
        self.processor = MidiProcessor(min_midi=MIN, max_midi=MAX,
                                       base_smoothing=0.05, acceleration_factor=0.05)

        # each dict entry is a page of parameters
        self.fader_params = {
            0: [ 'frame_blend', 'metaballs_feedback', 'smooth_coloring_max_field', 'threshold', 'radius_multiplier', 'speed_multiplier', 'num_metaballs', 'metaball_zoom'],
            1: ['alpha', 'temporal_filter', 'x_sync_speed', 'x_sync_freq', 'x_sync_amp', 'y_sync_speed', 'y_sync_freq', 'y_sync_amp'],
            2: ['hue_shift', 'sat_shift', 'val_shift', 'val_threshold', 'val_hue_shift', 'contrast', 'hue_invert', 'hue_invert_angle'],
        }
        self.fader_config = self.fader_params.get(0, [])

        self.encoder_params = {
            0: ['frame_blend', 'metaball_skew_angle', 'metaball_skew_intensity', 'metaball_hue', 'metaballs_saturation', 'metaballs_value', 'metaballs_contrast', 'metaballs_brightness'],
            1: ['hue_shift', 'sat_shift', 'val_shift', 'zoom', 'r_shift', 'x_shift', 'y_shift', 'blur_kernel_size', ''],
            2: ['alpha', 'temporal_filter', 'blur_kernel_size', 'zoom', 'r_shift', 'x_shift', 'y_shift', ''],
        }
        self.encoder_config = self.encoder_params.get(0, [])

    def set_values(self, control, value):
        """
        Maps the MIDI control messages to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        
        Args:
            control (int): The MIDI control number.
            value (int): The value of the MIDI control message.
        """
        if control in range(30, 38):
            self._set_encoder_param(control, value)
        elif control in range(40, 48):
            self._set_fader_param(control, value)
        elif control in range(20, 36):
            self._set_button_param(control, value)

    def _set_fader_param(self, control, value):
        """
        Maps the MIDI faders to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        
        index = control % 10 
        param = self.params.get(self.fader_config[index])

        if param is None:
            log.warning(f"No parameter found for control {control} in fader_config.")
            return

        value = self.processor.process_message(control, value, param.min, param.max)

        self.params.set(self.fader_config[index], value)

        log.info(f"{self.fader_config[index]}: {value} (MIDI value: {value})")

    def _set_encoder_param(self, control, value):
        """
        Maps the MIDI encoders to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        index = control % 10 #
        param = self.params.get(self.encoder_config[index])

        if param is None:
            log.warning(f"No parameter found for channel {control} in encoder_config.")
            return
        
        value = self.processor.process_message(control, value, param.min, param.max)
        self.params.set(self.encoder_config[index], value)

        log.info(f"{self.encoder_config[index]}: {value} (MIDI value: {value})")

    def _set_button_param(self, control, value):
        """
        Maps the MIDI buttons to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        log.info("Mapping buttons... (this is a placeholder function)")


class MidiMix(ControllerBase):
    name = "MIDI Mix"
    CONTROLLER_NAMES.append(name)

    def __init__(self, params, port_name):
        """
        Initializes the SMC_Mixer instance.
        """
        self.params = params
        MIN = 0
        MAX = 127

        self.port_name = port_name
        self.processor = MidiProcessor(min_midi=MIN, max_midi=MAX,
                                       base_smoothing=0.05, acceleration_factor=0.05)

        # MIDI CC values
        self.fader_controls = [19, 23, 27, 31, 49, 53, 57, 61, 62] # Faders 1-8
        self.pot_controls = [16, 17, 18, 20, 21, 22, 24, 25, 26, 28,29,30, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60] # Encoders 1-9
        self.button_controls = [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,21, 22, 24]

        # each dict entry is a page of parameters
        self.fader_params = {
            1: ['alpha', 'temporal_filter', 'hue_shift', 'sat_shift', 'val_shift', 'contrast', 'brightness', 'blur_kernel_size', 'frame_blend'], 
            # 2: ['hue_shift', 'sat_shift', 'val_shift', 'val_threshold', 'val_hue_shift', 'contrast', 'hue_invert', 'hue_invert_angle'],
        }
        self.fader_config = self.fader_params.get(1, [])

        self.encoder_params = {
            0: ['sequence', 'pattern_type', 'pattern_mod', 'pattern_r', 'pattern_g', 'pattern_b', "x_sync_amp", 'x_sync_freq', 'x_sync_speed', 'y_sync_amp', 'y_sync_freq', 'y_sync_speed', 'x_shift', 'hue_invert_angle', 'hue_invert_strength', 'y_shift', 'val_threshold', 'val_hue_shift', 'r_shift', 'noise_type', 'noise_intensity', 'zoom', 'reflection_mode', 'blur_type'],
            1: ['hue_shift', 'contrast', 'brightness', 'sat_shift', 'val_threshold', 'val_hue_shift', 'val_shift', 'hue_invert_angle', 'hue_invert_strength', 'x_shift', 'noise_type', 'noise_intensity', 'y_shift', 'blur_type', 'blur_kernel_size', 'zoom', 'x_sync_speed', 'y_sync_speed', 'r_shift', 'x_sync_freq', 'y_sync_freq', 'reflection_mode', 'x_sync_amp', 'y_sync_amp'],
            2: ["x_sync_amp", 'x_sync_freq', 'x_sync_speed', 'y_sync_amp', 'y_sync_freq', 'y_sync_speed', 'x_shift', 'pattern_type', 'pattern_mod', 'y_shift', 'solarize_threshold', 'posterize_levels', 'r_shift', 'hue_invert_angle', 'hue_invert_strength', 'zoom', 'val_threshold', 'val_hue_shift', 'reflection_mode', 'noise_type', 'noise_intensity', 'sequence', 'sharpen_intensity', 'blur_type'] }
        self.encoder_config = self.encoder_params.get(2, [])

        # self._set_button_params = {}

    def set_values(self, control, value):
        """
        Maps the MIDI control messages to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        
        Args:
            control (int): The MIDI control number.
            value (int): The value of the MIDI control message.
        """
        if control in self.fader_controls:
            self._set_fader_param(control, value)
        elif control in self.pot_controls:
            self._set_encoder_param(control, value)
        elif control in self.button_controls:
            self._set_button_param(control, value)

    def _set_fader_param(self, control, value):
        """
        Maps the MIDI faders to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        
        index = self.fader_controls.index(control)

        param = self.params.get(self.fader_config[index])

        if param is None:
            log.warning(f"No parameter found for control {control} in fader_config.")
            return

        value = self.processor.process_message(control, value, param.min, param.max)

        self.params.set(self.fader_config[index], value)

        log.info(f"{self.fader_config[index]}: {value} (MIDI value: {value})")

    def _set_encoder_param(self, control, value):
        """
        Maps the MIDI encoders to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        index = self.pot_controls.index(control) # get the index of the control in the pot_controls list

        param = self.params.get(self.encoder_config[index]) # get the parameter by name using the index

        if param is None:
            log.warning(f"Param '{param.name}' not found for channel {control} in encoder_config.")
            return
        
        value = self.processor.process_message(control, value, param.min, param.max)
        self.params.set(self.encoder_config[index], value)

        log.info(f"{self.encoder_config[index]}: {value} (MIDI value: {value})")

    def _set_button_param(self, control, value):
        """
        Maps the MIDI buttons to the SMC-Mixer.
        This function is a placeholder for actual mapping logic.
        """
        log.info("Mapping buttons... (this is a placeholder function)")


class MidiControllerInterface:
    """
    A generic class to handle MIDI input, process messages, and modify params
    This class runs in a separate thread to continuously listen for MIDI messages
    from the provided controller. See the SMC_Mixer class for specific mappings.
    """
    def __init__(self, params, port_name=None, controller=None, name=None):
        """
        Initializes the MidiController instance.
        
        Args:
            port_name (str): The name of the MIDI input port to open.
        """

        # Select the MIDI controller to use.
        self.controller = controller if controller else self.get_controller_by_name(name, port_name, params)

        # Use the controller's default port name if it exists,
        # otherwise prompt the user to select a port.
        self.port_name = self.controller.port_name if hasattr(self.controller, 'port_name') else self.select_port()

        # create a thread to handle MIDI input
        self.thread_stop = False
        self.thread = threading.Thread(target=self.input_thread_handler)
        self.thread.daemon = True
        self.thread.start()


    def get_controller_by_name(self, name, port_name, params):
        if name == SMC_Mixer.name:
            return SMC_Mixer(params=params, port_name=port_name)
        elif name == MidiMix.name:
            return MidiMix(params=params, port_name=port_name)
        #NOTE: if more


    def select_port(self):
        """
        Selects a MIDI input port by name.

        Args:
            port_name (str): The name of the MIDI input port to select.
        
        Returns:
            mido.Input: The opened MIDI input port.
        """
        
        # Get a list of all available MIDI input port names on the system
        input_ports = mido.get_input_names()

        if not input_ports:
            log.warning("No MIDI input ports found. Please ensure your MIDI device is connected and drivers are installed.")
            return

        log.info("Available MIDI Input Ports:")
        for i, port_name in enumerate(input_ports):
            log.info(f"{i}: {port_name}")

        # Prompt the user to select a MIDI port from the list
        selected_port_index = -1
        while selected_port_index < 0 or selected_port_index >= len(input_ports):
            try:
                choice = input(f"Enter the number of the MIDI input port to use (0-{len(input_ports)-1}): ")
                selected_port_index = int(choice)
            except ValueError:
                log.error("Invalid input. Please enter a number.")
            if selected_port_index < 0 or selected_port_index >= len(input_ports):
                log.error("Invalid port number. Please try again.")

        chosen_port_name = input_ports[selected_port_index]
        log.info(f"Selected port: {chosen_port_name}")
        return chosen_port_name

    def set_values(self, control, value):
        """
        Sets the values for the specified MIDI control number.

        Args:
            control (int): The MIDI control number.
            value (int): The value of the MIDI control message.
        """
        if self.controller is not None:
            self.controller.set_values(control, value)
        else:
            log.info("No controller set to handle MIDI messages.")
        
    def input_thread_handler(self):
        """
        Handles MIDI input in a separate thread.
        Opens the specified MIDI input port and uses the MidiProcessor instance
        to process incoming messages.
        """

        try:
            with mido.open_input(self.port_name) as inport:
                log.info(f"MIDI input thread started for port: {inport.name}")
                log.info("Listening for MIDI messages... (Press Ctrl+C in the main terminal to stop)")

                # Continuously listen for messages until the thread_stop flag is set
                for msg in inport:
                    if self.thread_stop:
                        log.warning(f"Stopping MIDI input for port: {inport.name}\n")
                        break

                    # get the control number from the message
                    control = msg.control if hasattr(msg, 'control') else None
                    if control is None:
                        log.info(f"Received non-control message: {msg}")
                        continue

                    self.set_values(control, msg.value)

        except ValueError as e:
            log.warning(f"Could not open MIDI port '{self.port_name}'. {e}")
            log.info("Ensure the port name is correct and the device is connected ")
        except Exception as e:
            log.warning(f"An unexpected error occurred in the MIDI thread: {e}")
        finally:
            log.warning(f"MIDI input thread for '{self.port_name}' has terminated.")

