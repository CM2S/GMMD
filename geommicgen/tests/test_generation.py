"""
Unit tests regarding microstructure generation.
The classe tessted is GenerationMethod
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np


# pylint: disable=import-error
from geommicgen.micgenmethod.microstructure_gen_method import (
    GenerationMethod,
)



class MicGenTest(GenerationMethod):
    def generate_microstructure(self, microstructure_sample):
        pass


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        class MicGenTestIncomp(GenerationMethod):
            pass

        with self.assertRaises(TypeError):

            _ = MicGenTestIncomp()


