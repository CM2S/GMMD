import unittest
from unittest.mock import sentinel, Mock, patch

# from microstructure.phase import Phase


from geommicgen.microstructure.microstructure import Microstructure


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
        """Check if multiple matrix phases raise an exception"""
        matrix_mock = sentinel.matrix
        matrix_mock.type = Mock(__name__="Matrix")
        disks_mock = sentinel.disks
        disks_mock.type = Mock(__name__="Disk")
        disks_mock.type.dim = 3
        with self.assertRaises(ValueError):
            # Number of RVE dimensions is not compatible with particle type
            self.microstructure_2D.add_phase(matrix_mock)
            self.microstructure_2D.add_phase(disks_mock)

    def test_add_phase_incompatible_phases(self):
        self.assertTrue(False)

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


if __name__ == "__main__":
    unittest.main()
