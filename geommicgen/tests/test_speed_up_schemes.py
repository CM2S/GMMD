"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

from micgenmethod.speed_up_schemes import SpeedUpScheme, CellList


class TestSpeedUpScheme(unittest.TestCase):
    """Class for the unit test regarding the speed up schemes"""

    def test_new_list_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        class SpeedUpSchemeTest(SpeedUpScheme):
            pass

        with self.assertRaises(TypeError):

            _ = SpeedUpSchemeTest()

    def test_cell_list_n_cell_dim(self):
        """Test the property n_cell_dim of cell lists"""
        radii = [0.1, 0.2, 0.25, 0.1, 0.3, 0.2, 0.05]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList()
        current_cell_list.max_radius = 0.3
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        self.assertTrue(all(current_cell_list.n_cell_dim == np.array([1, 3])))

    def test_cell_list_max_radius(self):
        """Test the attribute max_radius of cell lists"""
        radii = [0.1, 0.2, 0.25, 0.1, 0.3, 0.2, 0.05]
        center_positions = [
            np.array([0.2, 0.2]),
            np.array([0.8, 0.25]),
            np.array([0.5, 0.4]),
            np.array([1.0, 0.6]),
            np.array([0.5, 0.55]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=2)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        self.assertTrue(current_cell_list.max_radius == 0.3)

    def test_cell_list_cell_side_length(self):
        """Test the property cell_side_length of cell lists"""
        radii = [0.1, 0.2, 0.25, 0.1, 0.3, 0.2, 0.05]
        center_positions = [
            np.array([0.2, 0.2]),
            np.array([0.8, 0.25]),
            np.array([0.5, 0.4]),
            np.array([1.0, 0.6]),
            np.array([0.5, 0.55]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=2)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)
        self.assertTrue(all(current_cell_list.cell_side_length == np.array([1, 2 / 3])))

    def test_cell_list_new_list_2d(self):
        """Test if the cell is properly constructed for 2 dimensions."""
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.2, 0.2]),
            np.array([0.8, 0.25]),
            np.array([0.5, 0.4]),
            np.array([1.0, 0.6]),
            np.array([0.5, 0.55]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=2)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1.4, 1.0])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)
        correct_cell_list = 12 * [set()]
        correct_cell_list[0] = {0}
        correct_cell_list[2] = {1}
        correct_cell_list[5] = {4, 2}
        correct_cell_list[6] = {3}
        self.assertTrue(current_cell_list.cell_list == correct_cell_list)

    def test_cell_list_new_list_3d(self):
        """Test if the cell is properly constructed 3 dimensions."""
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[0.7, 1, 1.2])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)
        correct_cell_list = 2 * 3 * 4 * [set()]
        correct_cell_list[4] = {0}
        correct_cell_list[17] = {1}
        correct_cell_list[5] = {2, 3}
        correct_cell_list[1] = {4}
        self.assertTrue(current_cell_list.cell_list == correct_cell_list)

    def test_cell_list_new_list_3d_particle_list(self):
        """Test if the cell is properly constructed 3 dimensions."""
        radii = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) / 2
        center_positions = [
            np.array([0.3, 0.25, 0.15]),
            np.array([0.3, 0.45, 0.15]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.2, 0.8, 0.15]),
            np.array([0.2, 0.8, 0.35]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)
        correct_particle_list = [None for _ in range(5)]
        correct_particle_list[0] = [1]
        correct_particle_list[1] = []
        correct_particle_list[2] = [3, 4]
        correct_particle_list[3] = [4]
        correct_particle_list[4] = []
        self.assertTrue(current_cell_list.particle_list == correct_particle_list)

    def test_neighboor_cell_is_bottom(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        print(current_cell_list.n_cell_dim)
        self.assertEqual(6, current_cell_list.neighboor_cell(0, 10, 3, [3, 3, 3]))
        self.assertEqual(7, current_cell_list.neighboor_cell(1, 10, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighboor_cell(2, 10, 3, [3, 3, 3]))
        self.assertEqual(15, current_cell_list.neighboor_cell(9, 10, 3, [3, 3, 3]))
        self.assertEqual(16, current_cell_list.neighboor_cell(10, 10, 3, [3, 3, 3]))
        self.assertEqual(17, current_cell_list.neighboor_cell(11, 10, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighboor_cell(18, 10, 3, [3, 3, 3]))
        self.assertEqual(25, current_cell_list.neighboor_cell(19, 10, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighboor_cell(20, 10, 3, [3, 3, 3]))

    def test_neighboor_cell_at_top(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        print(current_cell_list.n_cell_dim)
        self.assertEqual(0, current_cell_list.neighboor_cell(6, 16, 3, [3, 3, 3]))
        self.assertEqual(1, current_cell_list.neighboor_cell(7, 16, 3, [3, 3, 3]))
        self.assertEqual(2, current_cell_list.neighboor_cell(8, 16, 3, [3, 3, 3]))
        self.assertEqual(9, current_cell_list.neighboor_cell(15, 16, 3, [3, 3, 3]))
        self.assertEqual(10, current_cell_list.neighboor_cell(16, 16, 3, [3, 3, 3]))
        self.assertEqual(11, current_cell_list.neighboor_cell(17, 16, 3, [3, 3, 3]))
        self.assertEqual(18, current_cell_list.neighboor_cell(24, 16, 3, [3, 3, 3]))
        self.assertEqual(19, current_cell_list.neighboor_cell(25, 16, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighboor_cell(26, 16, 3, [3, 3, 3]))

    def test_neighboor_cell_at_right(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        print(current_cell_list.n_cell_dim)
        self.assertEqual(0, current_cell_list.neighboor_cell(2, 14, 3, [3, 3, 3]))
        self.assertEqual(3, current_cell_list.neighboor_cell(5, 14, 3, [3, 3, 3]))
        self.assertEqual(6, current_cell_list.neighboor_cell(8, 14, 3, [3, 3, 3]))
        self.assertEqual(9, current_cell_list.neighboor_cell(11, 14, 3, [3, 3, 3]))
        self.assertEqual(12, current_cell_list.neighboor_cell(14, 14, 3, [3, 3, 3]))
        self.assertEqual(15, current_cell_list.neighboor_cell(17, 14, 3, [3, 3, 3]))
        self.assertEqual(18, current_cell_list.neighboor_cell(20, 14, 3, [3, 3, 3]))
        self.assertEqual(21, current_cell_list.neighboor_cell(23, 14, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighboor_cell(26, 14, 3, [3, 3, 3]))

    def test_neighboor_cell_at_left(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        print(current_cell_list.n_cell_dim)
        self.assertEqual(2, current_cell_list.neighboor_cell(0, 12, 3, [3, 3, 3]))
        self.assertEqual(5, current_cell_list.neighboor_cell(3, 12, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighboor_cell(6, 12, 3, [3, 3, 3]))
        self.assertEqual(11, current_cell_list.neighboor_cell(9, 12, 3, [3, 3, 3]))
        self.assertEqual(14, current_cell_list.neighboor_cell(12, 12, 3, [3, 3, 3]))
        self.assertEqual(17, current_cell_list.neighboor_cell(15, 12, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighboor_cell(18, 12, 3, [3, 3, 3]))
        self.assertEqual(23, current_cell_list.neighboor_cell(21, 12, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighboor_cell(24, 12, 3, [3, 3, 3]))

    def test_neighboor_cell_at_front(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        self.assertEqual(18, current_cell_list.neighboor_cell(0, 4, 3, [3, 3, 3]))
        self.assertEqual(19, current_cell_list.neighboor_cell(1, 4, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighboor_cell(2, 4, 3, [3, 3, 3]))
        self.assertEqual(21, current_cell_list.neighboor_cell(3, 4, 3, [3, 3, 3]))
        self.assertEqual(22, current_cell_list.neighboor_cell(4, 4, 3, [3, 3, 3]))
        self.assertEqual(23, current_cell_list.neighboor_cell(5, 4, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighboor_cell(6, 4, 3, [3, 3, 3]))
        self.assertEqual(25, current_cell_list.neighboor_cell(7, 4, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighboor_cell(8, 4, 3, [3, 3, 3]))

    def test_neighboor_cell_at_back(self):
        radii = [0.1, 0.05, 0.1, 0.1, 0.14]
        center_positions = [
            np.array([0.3, 0.7, 0.1]),
            np.array([0.5, 0.8, 0.7]),
            np.array([0.4, 0.8, 0.15]),
            np.array([0.6, 0.75, 0.2]),
            np.array([0.65, 0.1, 0.2]),
        ]
        particles = [
            Mock(radius=radius, position_center=position, dim=3)
            for radius, position in zip(radii, center_positions)
        ]
        mock_molecular_dynamics_sim = Mock(box=[1, 1, 1])
        current_cell_list = CellList()
        current_cell_list.molecular_dynamics_sim = mock_molecular_dynamics_sim
        current_cell_list.new_list(particles)

        self.assertEqual(0, current_cell_list.neighboor_cell(18, 22, 3, [3, 3, 3]))
        self.assertEqual(1, current_cell_list.neighboor_cell(19, 22, 3, [3, 3, 3]))
        self.assertEqual(2, current_cell_list.neighboor_cell(20, 22, 3, [3, 3, 3]))
        self.assertEqual(3, current_cell_list.neighboor_cell(21, 22, 3, [3, 3, 3]))
        self.assertEqual(4, current_cell_list.neighboor_cell(22, 22, 3, [3, 3, 3]))
        self.assertEqual(5, current_cell_list.neighboor_cell(23, 22, 3, [3, 3, 3]))
        self.assertEqual(6, current_cell_list.neighboor_cell(24, 22, 3, [3, 3, 3]))
        self.assertEqual(7, current_cell_list.neighboor_cell(25, 22, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighboor_cell(26, 22, 3, [3, 3, 3]))
