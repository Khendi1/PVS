from abc import ABC, abstractmethod
import numpy as np

class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, params, width=640, height=480, group=None):
        self.params = params
        self.width = width
        self.height = height
        self.group = group


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("subgroupes should implement this method.")
