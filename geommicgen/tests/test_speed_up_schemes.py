"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

from geommicgen.micgenmethod.speed_up_schemes import SpeedUpScheme, CellList


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
        particles = [Mock(radius=radius) for radius in radii]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList(mock_molecular_dynamics_sim, particles)
        self.assertTrue(all(current_cell_list.n_cell_dim == np.array([1, 3])))

    def test_cell_list_max_radius(self):
        """Test the attribute max_radius of cell lists"""
        radii = [0.1, 0.2, 0.25, 0.1, 0.3, 0.2, 0.05]
        particles = [Mock(radius=radius) for radius in radii]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList(mock_molecular_dynamics_sim, particles)
        self.assertTrue(current_cell_list.max_radius == 0.3)

    def test_cell_list_cell_side_length(self):
        """Test the property cell_side_length of cell lists"""
        radii = [0.1, 0.2, 0.25, 0.1, 0.3, 0.2, 0.05]
        particles = [Mock(radius=radius) for radius in radii]
        mock_molecular_dynamics_sim = Mock(box=[1.0, 2.0])
        current_cell_list = CellList(mock_molecular_dynamics_sim, particles)
        self.assertTrue(
            all(current_cell_list.cell_side_length == np.array([1 / 3, 1 / 6]))
        )

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
        current_cell_list = CellList(mock_molecular_dynamics_sim, particles)
        current_cell_list.new_list(particles)
        correct_cell_list = 12 * [set()]
        correct_cell_list[0] = {particles[0]}
        correct_cell_list[2] = {particles[1]}
        correct_cell_list[5] = {particles[4], particles[2]}
        correct_cell_list[6] = {particles[3]}
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
        current_cell_list = CellList(mock_molecular_dynamics_sim, particles)
        current_cell_list.new_list(particles)
        correct_cell_list = 2 * 3 * 4 * [set()]
        correct_cell_list[4] = {particles[0]}
        correct_cell_list[17] = {particles[1]}
        correct_cell_list[5] = {particles[2], particles[3]}
        correct_cell_list[1] = {particles[4]}
        self.assertTrue(current_cell_list.cell_list == correct_cell_list)
