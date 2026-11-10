"""
Unit tests regarding the abstract GenerationMethod class.

These tests target behavior shared by every generation method (both molecular dynamics and
random sequential adsorption), such as setting the simulation box. Behavior specific to a
single generation method belongs in its own dedicated test module
(test_molecular_dynamics_sim.py, test_rand_seq_adsorption.py, ...).
"""
import unittest
from unittest.mock import Mock

# pylint: disable=import-error
from geommicgen.micgenmethod.microstructure_gen_method import (
    GenerationMethod,
)
from geommicgen.microstructure.particleclasses import (
    CylindricalFiber,
)


class MicGenTest(GenerationMethod):
    """Minimal concrete GenerationMethod used to exercise the shared base-class behavior."""

    def generate_microstructure(self, microstructure_sample):
        pass


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        class MicGenTestIncomp(GenerationMethod):
            pass

        with self.assertRaises(TypeError):

            _ = MicGenTestIncomp()

    def test_set_box_cylindrical_fiber_set_box(self):
        """Check if the simulation box is correctly set if there a cylindrical fibers."""

        current_generation_method = MicGenTest()
        rve_dims = [1.0, 2.0, 3.0]
        mock_cylindrical_fiber_1 = Mock()
        mock_cylindrical_fiber_2 = Mock()
        mock_cylindrical_fiber_1.__class__ = CylindricalFiber
        mock_cylindrical_fiber_2.__class__ = CylindricalFiber
        mock_cylindrical_fiber_1.direction_fibers = 0
        mock_cylindrical_fiber_2.direction_fibers = 0
        particles = [mock_cylindrical_fiber_1, mock_cylindrical_fiber_2]
        current_generation_method.set_box(particles, rve_dims)
        self.assertEqual(current_generation_method.box, [2.0, 3.0])

    def test_set_box_other_particles(self):
        """Check if the simulation box is correctly set if there no a cylindrical fibers."""

        current_generation_method = MicGenTest()
        rve_dims = [2.0, 3.0]
        mock_disk = Mock()
        mock_ellipse = Mock()
        particles = [mock_disk, mock_ellipse]
        current_generation_method.set_box(particles, rve_dims)
        self.assertEqual(current_generation_method.box, [2.0, 3.0])
