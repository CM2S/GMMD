import unittest
from unittest.mock import sentinel, Mock, patch

import numpy as np

# from microstructure.phase import Phase


from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.particleclasses.disk import Disk


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

    def test_add_phase_cylindricalfiber_compatible(self):
        """Check that CylindricalFiber phases coexist with each other and the matrix."""
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        cylindricalfiber_mock_1 = sentinel.cylindricalfiber_1
        cylindricalfiber_mock_1.type = Mock(__name__="CylindricalFiber")
        cylindricalfiber_mock_1.type.dim = 3
        cylindricalfiber_mock_2 = sentinel.cylindricalfiber_2
        cylindricalfiber_mock_2.type = Mock(__name__="CylindricalFiber")
        cylindricalfiber_mock_2.type.dim = 3

        self.microstructure_3D.add_phase(matrix_mock)
        self.microstructure_3D.add_phase(cylindricalfiber_mock_1)
        self.microstructure_3D.add_phase(cylindricalfiber_mock_2)

        self.assertIn(cylindricalfiber_mock_2.name, self.microstructure_3D.phases)

    def test_add_phase_cylindricalfiber_incompatible_with_other_particles(self):
        """Check that a CylindricalFiber phase is rejected next to a non-fiber phase."""
        spheres_mock = sentinel.spheres
        spheres_mock.type = Mock(__name__="Sphere")
        spheres_mock.type.dim = 3
        cylindricalfiber_mock = sentinel.cylindricalfiber
        cylindricalfiber_mock.type = Mock(__name__="CylindricalFiber")
        cylindricalfiber_mock.type.dim = 3

        self.microstructure_3D.add_phase(spheres_mock)
        with self.assertRaises(ValueError):
            self.microstructure_3D.add_phase(cylindricalfiber_mock)

    def test_particles_property(self):
        """Check that particles aggregates the particles of every phase."""
        phase_1 = Mock()
        phase_1.particles = [sentinel.particle_1, sentinel.particle_2]
        phase_2 = Mock()
        phase_2.particles = [sentinel.particle_3]
        self.microstructure_2D.phases = {"phase_1": phase_1, "phase_2": phase_2}

        self.assertEqual(
            self.microstructure_2D.particles,
            [sentinel.particle_1, sentinel.particle_2, sentinel.particle_3],
        )

    def test_volume_fraction_property(self):
        """Check that volume_fraction sums the volume fraction of every phase."""
        phase_1 = Mock()
        phase_1.volume_fraction = 0.2
        phase_2 = Mock()
        phase_2.volume_fraction = 0.3
        self.microstructure_2D.phases = {"phase_1": phase_1, "phase_2": phase_2}

        self.assertAlmostEqual(self.microstructure_2D.volume_fraction, 0.5)

    def test_volume_fraction_circ_property(self):
        """Check that volume_fraction_circ sums the circumscribed volume fraction of
        every phase."""
        phase_1 = Mock()
        phase_1.volume_fraction_circ = 0.15
        phase_2 = Mock()
        phase_2.volume_fraction_circ = 0.1
        self.microstructure_2D.phases = {"phase_1": phase_1, "phase_2": phase_2}

        self.assertAlmostEqual(self.microstructure_2D.volume_fraction_circ, 0.25)

    def test_inside_particle_phase(self):
        """Check that points are correctly flagged as inside/outside the particle phase."""
        disk = Disk("FakePhase", {"r": 0.1, "n": 1}, [1.0, 1.0])
        disk.position_center = np.array([0.5, 0.5])
        phase = Mock()
        phase.particles = [disk]
        self.microstructure_2D.phases = {"phase_1": phase}

        pts = [[0.5, 0.5], [0.9, 0.9]]
        self.assertEqual(self.microstructure_2D.inside_particle_phase(pts), [1, 0])


if __name__ == "__main__":
    unittest.main()
