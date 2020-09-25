"""Module for the GenerationMethod abstract class."""

import abc


class GenerationMethod(abc.ABC):
    """This is the abstract class for generation methods."""

    @abc.abstractmethod
    def generate_microstructure(self, microstructure_sample):
        """Generate a microstructure."""
