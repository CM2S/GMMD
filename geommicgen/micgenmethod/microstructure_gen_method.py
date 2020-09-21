import abc
import numpy as np


class GenerationMethod(abc.ABC):
    @abc.abstractmethod
    def generate_microstructure(self, microstructure_sample):
        pass
