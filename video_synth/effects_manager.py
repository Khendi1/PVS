"""
See notes on Program Architecture in README.md

Classes stored here:
    - enumeration classes for effect modes, etc.
    - EffectBase singleton class; extended by each Effect subgroup (EX: Color, Feedback, etc.)
    - Effect subgroupes; require Control Structures as args, instantiates and adds Control Objects (Parameters) to approp structures
    - EffectManager to init, manage, and sequence Effect subgroupes

Contribution guide:
    - each new effect class should extend the EffectsBase class
    - each new effect class should be initialized in EffectManager/init()
    - each new effect class should expose 2 kinds of public functions,
      and all private classes should be appended with '_'
        - effect methods: these will be automatically added to the sequencer
        - gui methods: these will not be automatically added or called by the gui; 
                       these should have the word 'gui' in the method name for easy filtering
                       these must be manually called in gui.py/create_trackbars()
"""
import logging
import time
import cv2
import numpy as np

from param import ParamTable
from lfo import OscBank
from patterns3 import Patterns
from effects.color import Color
from effects.pixels import Pixels
from effects.sync import Sync
from effects.warp import Warp
from effects.reflector import Reflector
from effects.ptz import PTZ
from effects.feedback import Feedback
from effects.glitch import Glitch
from effects.shapes import Shapes

log = logging.getLogger(__name__)

class EffectManager:
    """
    A central class to aggregate all singleton effect objects and simplify dependencies, arg len
    """
    def __init__(self, group, width, height):
        self.group = group
        self.params = ParamTable(group=f"EffectManager_{group}")
        self.oscs = OscBank(self.params, 0)
        
        self.feedback = Feedback(self.params, width, height, self.group)
        self.color = Color(self.params, self.group)
        self.pixels = Pixels(self.params, width, height, group=self.group)
        self.shapes = Shapes(self.params, width, height, group=self.group)
        self.patterns = Patterns(self.params, self.oscs, width, height, self.group)
        self.reflector = Reflector(self.params, group=self.group)                    
        self.sync = Sync(self.params, self.group) 
        self.warp = Warp(self.params, width, height, self.group)
        self.glitch = Glitch(self.params, self.group)
        self.ptz = PTZ(self.params, width, height, self.group)

        self._all_services = [
            self.feedback,
            self.color,
            self.pixels,
            self.shapes,
            self.patterns,
            self.reflector,
            self.sync,
            self.warp,
            self.glitch,
            self.ptz,
        ]

        self.class_with_methods, self.all_methods = self._get_effect_methods()

        # Per-frame performance data (read by video_loop when DEBUG is enabled)
        self.perf_data = {}


    def _get_effect_methods(self):
        """
        Collects all unique public method names from a list of effect objects.
        Any public method will be included in the list
        """

        class_with_methods = {}
        methods = []

        for obj in self._all_services:
            # Get all attributes of the current object
            all_attributes = dir(obj)
            
            # Filter for public methods and add them to the set
            public_methods = [
                getattr(obj, attr) for attr in all_attributes 
                if not attr.startswith('_') and callable(getattr(obj, attr))
            ]
            methods+=public_methods
            class_with_methods[type(obj).__name__] = public_methods
        
        # Feedback should not be included in sequencer, so remove from structures here
        cleaned_methods = [m for m in methods if m not in class_with_methods['Feedback']]
        del class_with_methods['Feedback']

        # log.debug(f"EffectManager found {len(cleaned_methods)} effect methods from {len(class_with_methods)} classes.")
        # log.debug(f"EffectManager methods: {[m.__name__ for m in cleaned_methods]}")

        return class_with_methods, cleaned_methods


    def reset_feedback_buffer(self):
        self.feedback.frame_buffer.clear()


    def adjust_sequence(self, from_idx, to_idx):
        self.all_methods.insert(to_idx, self.all_methods.pop(from_idx))


    """ 
    obsolete after implementing the effects sequencer.
    default sequence of effects for future reference; 
    """
    def _default_effect_sequence(self, frame):
        frame = self.ptz.shift_frame(frame)
        frame = self.patterns.generate_pattern_frame(frame)
        frame = self.sync.sync(frame)
        frame = self.reflector.apply_reflection(frame)
        frame = self.color.polarize_frame_hsv(frame)
        frame = self.color.modify_hsv(frame)
        frame = self.color.adjust_brightness_contrast(frame)
        frame = self.pixels.apply_noise(frame)
        frame = self.color.solarize_image(frame)
        frame = self.color.posterize(frame)
        frame = self.pixels.blur(frame)
        frame = self.pixels.sharpen_frame(frame)
        frame = self.glitch.apply_glitch_effects(frame)        

        # TODO: test these effects, test ordering
        # frame = color limit_hues_kmeans(frame)
        # frame = fx.polar_transform(frame, params.get("polar_x"), params.get("polar_y"), params.get("polar_radius"))
        # frame = fx.apply_perlin_noise
        # warp_frame = fx.warp_frame(frame)

        # frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = fx.lissajous_pattern(frame, t)

        # frame = s.draw_shapes_on_frame(frame, c.image_width, c.image_height)

        return frame


    def _apply_effect_chain(self, frame, frame_count):
        """
        Applies a sequence of visual effects to the input frame based on current parameters.
        Each effect is modular and can be enabled/disabled via the GUI.
        The order of effects can be adjusted to achieve different visual styles.

        Returns the modified frame.
        """

        if frame_count % (self.feedback.frame_skip.value+1) == 0:
            original_frame_for_recovery = None
            slow_effects = []
            effect_timings = {}
            for method in self.all_methods:
                t_start = time.perf_counter()
                processed_frame = method(frame)
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                effect_timings[method.__name__] = elapsed_ms

                if elapsed_ms > 20:
                    slow_effects.append((method.__name__, elapsed_ms))

                if processed_frame is None:
                    log.warning(f"Method {method.__name__} in EffectManager ({self.group}) returned None. Recovering with previous frame.")
                    if original_frame_for_recovery is None:
                        log.error(f"No recovery frame available! Effect chain may be broken.")
                        continue
                    frame = original_frame_for_recovery
                else:
                    frame = processed_frame
                    del processed_frame

            # Store per-effect timings for diagnostics
            self.perf_data['effects'] = effect_timings

            if slow_effects and frame_count % 30 == 0:
                effects_str = ", ".join([f"{name}={t:.1f}ms" for name, t in slow_effects])
                log.warning(f"{self.group.name} slow effects: {effects_str}")

        return frame

    
    def get_frames(self, dry_frame, wet_frame, prev_frame, frame_count):
        perf = self.perf_data

        # Blend the current dry frame with the previous wet frame using the alpha param (feedback)
        t0 = time.perf_counter()
        wet_frame = cv2.addWeighted(dry_frame.astype(np.float32), 1 - self.feedback.alpha.value, wet_frame.astype(np.float32), self.feedback.alpha.value, 0)
        perf['alpha_blend'] = (time.perf_counter() - t0) * 1000

        # Apply effects AFTER blend so they affect the output regardless of alpha
        t0 = time.perf_counter()
        wet_frame = self._apply_effect_chain(wet_frame, frame_count)
        perf['effect_chain'] = (time.perf_counter() - t0) * 1000

        # Apply feedback effects
        t0 = time.perf_counter()
        wet_frame = self.feedback.apply_temporal_filter(prev_frame, wet_frame)
        wet_frame = self.feedback.avg_frame_buffer(wet_frame)
        wet_frame = self.feedback.nth_frame_feedback(wet_frame)
        wet_frame = self.feedback.apply_luma_feedback(prev_frame, wet_frame)
        perf['feedback'] = (time.perf_counter() - t0) * 1000

        prev_frame = wet_frame
        prev_frame = self.ptz._shift_prev_frame(prev_frame)

        return np.clip(prev_frame, 0, 255), np.clip(wet_frame, 0, 255)
