import unittest
from unittest.mock import sentinel, Mock, patch

# from microstructure.phase import Phase


import numpy as np

from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.tests.helpers import disk_microstructure, sphere_microstructure


class TestMicrostructure(unittest.TestCase):
    """Test class for the Microstructure class."""

    def setUp(self):
        rve_dims = [1.0, 1.0]
        self.microstructure_2D = Microstructure(rve_dims)

        rve_dims = [1.0, 1.0, 1.0]
        self.microstructure_3D = Microstructure(rve_dims)

    def test_init(self):
        rve_dims = [1.0]
        with self.assertRaises(ValueError):
            _ = Microstructure(rve_dims)

        rve_dims = [1.0, 1.0, 1.9, 2]
        with self.assertRaises(ValueError):
            _ = Microstructure(rve_dims)

        rve_dims = [-1.0, 1.0, 1.9]
        with self.assertRaises(ValueError):
            _ = Microstructure(rve_dims)

    def test_add_phase_saved_correctly(self):
        """Check if the phase is saved correctly in the microstructure Dictionary"""
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        self.microstructure_2D.add_phase(matrix_mock)
        self.assertEqual(
            self.microstructure_2D.phases[sentinel.matrix.name], sentinel.matrix
        )

    def test_add_phase_detect_multiple_matrix_phase(self):
        """Check if multiple matrix phases raise an exception"""
        matrix_mock_1 = sentinel.matrix_1
        matrix_mock_1.type = Mock(__name__="Matrix")
        matrix_mock_2 = sentinel.matrix_2
        matrix_mock_2.type = Mock(__name__="Matrix")
        with self.assertRaises(ValueError):
            # Number of RVE dimensions is not compatible with particle type
            self.microstructure_2D.add_phase(matrix_mock_1)
            self.microstructure_2D.add_phase(matrix_mock_2)

    def test_add_phase_save_matrix_phase(self):
        """Check if the name of the matrix phase is properly stored"""
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        self.microstructure_2D.add_phase(matrix_mock)
        self.assertEqual(self.microstructure_2D.matrix_phase, matrix_mock.name)

    def test_add_phase_mic_dim_incompatible_w_particle(self):
        """Check if matrix/particle incompatibility is detected."""
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        disks_mock = sentinel.disks
        disks_mock.type = Mock(__name__="Sphere")
        disks_mock.type.dim = 3
        with self.assertRaises(ValueError):
            # Number of RVE dimensions is not compatible with particle type
            self.microstructure_2D.add_phase(matrix_mock)
            self.microstructure_2D.add_phase(disks_mock)

    def test_add_phase_incompatible_phases(self):
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        cylindricalfiber_mock = sentinel.cylindricalfiber
        cylindricalfiber_mock.type = Mock(__name__="CylindricalFiber")
        spheres_mock = sentinel.disks
        spheres_mock.type = Mock(__name__="Sphere")
        spheres_mock.type.dim = 3
        with self.assertRaises(ValueError):
            # Number of RVE dimensions is not compatible with particle type
            self.microstructure_3D.add_phase(matrix_mock)
            self.microstructure_3D.add_phase(spheres_mock)
            self.microstructure_3D.add_phase(cylindricalfiber_mock)

    # with self.assertRaises(ValueError):
    #     # Number of RVE dimensions is not compatible with particle type
    #     rve_dims = [1.0, 1.0]
    #     descriptors = {
    #         "1": {"Phase_Type": 1},
    #         "2": {"Phase_Type": 4, "r": 0.1, "vf": 0.5},
    #     }
    #     _ = Microstructure(descriptors, rve_dims)
    #
    # with self.assertRaises(ValueError):
    #     # Only one matrix phase can be specified
    #     rve_dims = [1.0, 1.0, 1.0]
    #     descriptors = {
    #         "1": {"Phase_Type": 1},
    #         "2": {"Phase_Type": 1},
    #     }
    #     _ = Microstructure(descriptors, rve_dims)


class TestFromDescriptors(unittest.TestCase):
    """Test class for the microstructure built from the descriptors of a deck."""

    DESCRIPTORS = {"0": {"phase_type": 1}, "1": {"phase_type": 2, "vf": 0.2, "n": 4}}
    # A matrix and a phase of disks, as the input data file reader hands them over

    def test_the_phases_are_declared(self):
        microstructure = Microstructure.from_descriptors([1.0, 1.0], self.DESCRIPTORS)
        self.assertEqual(list(microstructure.phases), ["0", "1"])
        self.assertEqual(microstructure.matrix_phase, "0")
        self.assertEqual(microstructure.phases["1"].type.__name__, "Disk")
        self.assertEqual(microstructure.phases["1"].descriptors["n"].value, 4)
        self.assertEqual(microstructure.particles, [])
        for i_phase in microstructure.phases.values():
            self.assertIs(i_phase.microstructure, microstructure)

    def test_a_deck_without_a_matrix_is_refused(self):
        with self.assertRaises(ValueError) as context:
            Microstructure.from_descriptors([1.0, 1.0], {"1": self.DESCRIPTORS["1"]})
        self.assertIn("matrix", str(context.exception))

    def test_a_deck_with_two_matrices_is_refused(self):
        with self.assertRaises(ValueError):
            Microstructure.from_descriptors(
                [1.0, 1.0], {"0": {"phase_type": 1}, "2": {"phase_type": 1}}
            )

    def test_each_call_builds_a_microstructure_of_its_own(self):
        first = Microstructure.from_descriptors([1.0, 1.0], self.DESCRIPTORS)
        second = Microstructure.from_descriptors([1.0, 1.0], self.DESCRIPTORS)
        self.assertIsNot(first.phases["1"], second.phases["1"])
        # Each sample of a run is generated into a microstructure of its own


class TestInsideParticlePhase(unittest.TestCase):
    """Test class for telling the points inside a particle from the ones outside."""

    def brute_force(self, microstructure, points):
        """Test every point against every particle, with no cell list in between."""
        return [
            int(
                any(
                    i_particle.point_inside(i_point, microstructure.rve_dims)
                    for i_particle in microstructure.particles
                )
            )
            for i_point in points
        ]

    def check(self, microstructure):
        """Check the lookup against the brute force on points all over and beyond."""
        rng = np.random.RandomState(0)
        dim = microstructure.dim
        points = rng.uniform(-0.5, 1.5, size=(2000, dim))
        # Beyond the RVE on both sides, so that the wrapping of the points is exercised
        inside = microstructure.inside_particle_phase(list(points))
        self.assertEqual(inside, self.brute_force(microstructure, points))
        self.assertGreater(sum(inside), 0)
        self.assertLess(sum(inside), len(inside))
        # Both answers occur, so the equality is not two lists of the same constant

    def test_two_dimensional(self):
        self.check(disk_microstructure())
        # One of the disks crosses a face of the RVE

    def test_three_dimensional(self):
        self.check(sphere_microstructure())
        # One of the spheres crosses two faces


if __name__ == "__main__":
    unittest.main()