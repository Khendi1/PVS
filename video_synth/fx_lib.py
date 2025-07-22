from fx import Effects
from noiser import ImageNoiser, NoiseType
from shapes import ShapeGenerator
from patterns3 import Patterns
from keying import Keying
from reflactor import Reflector
from metaballs import LavaLampSynth
from config import FPS
from generators import Oscillator

class Effects_Library:
    """
    This class contains methods to apply various effects to video frames.
    It includes methods for shifting frames, applying reflections, modifying HSV,
    adjusting brightness and contrast, sharpening, glitching, adding noise,
    polarizing, blurring, solarizing, and posterizing images.
    """
    
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.basic = Effects(image_width, image_height)
        self.noise = ImageNoiser(NoiseType.NONE)
        self.shapes = ShapeGenerator(image_width, image_height)
        self.patterns = Patterns(image_width, image_height)
        self.key = Keying(image_width, image_height)     # TODO: test this
        self.reflector = Reflector()                    # Initialize the reflector
        self.metaballs = LavaLampSynth(image_width, image_height)  # Initialize the Lava Lamp synth