"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

# from microstructure.phase import Phase


from geommicgen.micgenmethod.microstructure_gen_method import GenerationMethod


class MicGenTest(GenerationMethod):
    def generate_microstructure(self, microstructure_sample):
        pass


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        with self.assertRaises(TypeError):

            _ = MicGenTest()

    @patch("geommicgen.microstructure.particle_classes.Disk")
    @patch("geommicgen.microstructure.phase.FixedValue")
    def test_generate_particles_number(self, mock_fixed_value, mock_disk):
        gen_method = MicGenTest()
        rve_dims = [1.0, 1.0]
        descriptors = {
            "n": mock_fixed_value("n", 10),
            "vf": mock_fixed_value("vf", 0.1),
        }
        phase = sentinel.phase
        particles = gen_method.generate_particles(
            rve_dims, mock_disk, phase, **descriptors
        )
        self.assertTrue(len(particles), 10)
        for particle in particles:
            self.assertTrue(particle.name, "Disk()")

    @patch("geommicgen.microstructure.particle_classes.Disk")
    @patch("geommicgen.microstructure.phase.FixedValue")
    def test_generate_particles_vf(self, mock_fixed_value, mock_disk):
        gen_method = MicGenTest()
        rve_dims = [1.0, 1.0]
        vf_mock = mock_fixed_value("vf", 0.1)
        vf_mock.value = 0.1
        descriptors = {
            "r": mock_fixed_value("r", 0.1),
            "vf": vf_mock,
        }
        mock_disk.return_value = sentinel.particle
        sentinel.particle.volume = np.pi * 0.1 ** 2
        phase = sentinel.phase
        particles = gen_method.generate_particles(
            rve_dims, mock_disk, phase, **descriptors
        )
        self.assertEqual(len(particles), 4)
        for particle in particles:
            self.assertTrue(particle.name, "Disk()")
