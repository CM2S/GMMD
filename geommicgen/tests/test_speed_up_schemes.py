"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

from geommicgen.postproc.plotfuncs.plotting_functions import (
    plot_particles_3d,
    plot_particles_2d,
)
import pickle
import numpy as np

from geommicgen.micgenmethod.speed_up_schemes.MD_speed_up_schemes import MD_SpeedUpScheme, CellList, VerletList
from geommicgen.micgenmethod.speed_up_schemes.RSA_speed_up_schemes import (
    RSA_SpeedUpScheme,
    CellList as RSACellList,
    Naive as RSANaive,
)
from geommicgen.microstructure.particleclasses import Ellipse, Disk

from geommicgen.micgenmethod.microstructure_gen_method import (
    GenerationMethod,
)
from geommicgen.micgenmethod.molecular_dynamics_sim import (
    MolecularDynamicsSimulation,
)

import numpy as np


class TestMDSpeedUpScheme(unittest.TestCase):
    """Class for the unit tests regarding the molecular dynamics (MD) speed up schemes."""

    def test_new_list_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        class SpeedUpSchemeTest(MD_SpeedUpScheme):
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
        correct_particle_list[0] = {0, 1}
        correct_particle_list[1] = {0, 1}
        correct_particle_list[2] = {2, 3, 4}
        correct_particle_list[3] = {2, 3, 4}
        correct_particle_list[4] = {2, 3, 4}

        self.assertTrue(current_cell_list.particle_list == correct_particle_list)

    def test_neighbor_cell_is_bottom(self):
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

        self.assertEqual(6, current_cell_list.neighbor_cell(0, 10, 3, [3, 3, 3]))
        self.assertEqual(7, current_cell_list.neighbor_cell(1, 10, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighbor_cell(2, 10, 3, [3, 3, 3]))
        self.assertEqual(15, current_cell_list.neighbor_cell(9, 10, 3, [3, 3, 3]))
        self.assertEqual(16, current_cell_list.neighbor_cell(10, 10, 3, [3, 3, 3]))
        self.assertEqual(17, current_cell_list.neighbor_cell(11, 10, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighbor_cell(18, 10, 3, [3, 3, 3]))
        self.assertEqual(25, current_cell_list.neighbor_cell(19, 10, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighbor_cell(20, 10, 3, [3, 3, 3]))

    def test_neighbor_cell_at_top(self):
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

        self.assertEqual(0, current_cell_list.neighbor_cell(6, 16, 3, [3, 3, 3]))
        self.assertEqual(1, current_cell_list.neighbor_cell(7, 16, 3, [3, 3, 3]))
        self.assertEqual(2, current_cell_list.neighbor_cell(8, 16, 3, [3, 3, 3]))
        self.assertEqual(9, current_cell_list.neighbor_cell(15, 16, 3, [3, 3, 3]))
        self.assertEqual(10, current_cell_list.neighbor_cell(16, 16, 3, [3, 3, 3]))
        self.assertEqual(11, current_cell_list.neighbor_cell(17, 16, 3, [3, 3, 3]))
        self.assertEqual(18, current_cell_list.neighbor_cell(24, 16, 3, [3, 3, 3]))
        self.assertEqual(19, current_cell_list.neighbor_cell(25, 16, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighbor_cell(26, 16, 3, [3, 3, 3]))

    def test_neighbor_cell_at_right(self):
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

        self.assertEqual(0, current_cell_list.neighbor_cell(2, 14, 3, [3, 3, 3]))
        self.assertEqual(3, current_cell_list.neighbor_cell(5, 14, 3, [3, 3, 3]))
        self.assertEqual(6, current_cell_list.neighbor_cell(8, 14, 3, [3, 3, 3]))
        self.assertEqual(9, current_cell_list.neighbor_cell(11, 14, 3, [3, 3, 3]))
        self.assertEqual(12, current_cell_list.neighbor_cell(14, 14, 3, [3, 3, 3]))
        self.assertEqual(15, current_cell_list.neighbor_cell(17, 14, 3, [3, 3, 3]))
        self.assertEqual(18, current_cell_list.neighbor_cell(20, 14, 3, [3, 3, 3]))
        self.assertEqual(21, current_cell_list.neighbor_cell(23, 14, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighbor_cell(26, 14, 3, [3, 3, 3]))

    def test_neighbor_cell_at_left(self):
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

        self.assertEqual(2, current_cell_list.neighbor_cell(0, 12, 3, [3, 3, 3]))
        self.assertEqual(5, current_cell_list.neighbor_cell(3, 12, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighbor_cell(6, 12, 3, [3, 3, 3]))
        self.assertEqual(11, current_cell_list.neighbor_cell(9, 12, 3, [3, 3, 3]))
        self.assertEqual(14, current_cell_list.neighbor_cell(12, 12, 3, [3, 3, 3]))
        self.assertEqual(17, current_cell_list.neighbor_cell(15, 12, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighbor_cell(18, 12, 3, [3, 3, 3]))
        self.assertEqual(23, current_cell_list.neighbor_cell(21, 12, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighbor_cell(24, 12, 3, [3, 3, 3]))

    def test_neighbor_cell_at_front(self):
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

        self.assertEqual(18, current_cell_list.neighbor_cell(0, 4, 3, [3, 3, 3]))
        self.assertEqual(19, current_cell_list.neighbor_cell(1, 4, 3, [3, 3, 3]))
        self.assertEqual(20, current_cell_list.neighbor_cell(2, 4, 3, [3, 3, 3]))
        self.assertEqual(21, current_cell_list.neighbor_cell(3, 4, 3, [3, 3, 3]))
        self.assertEqual(22, current_cell_list.neighbor_cell(4, 4, 3, [3, 3, 3]))
        self.assertEqual(23, current_cell_list.neighbor_cell(5, 4, 3, [3, 3, 3]))
        self.assertEqual(24, current_cell_list.neighbor_cell(6, 4, 3, [3, 3, 3]))
        self.assertEqual(25, current_cell_list.neighbor_cell(7, 4, 3, [3, 3, 3]))
        self.assertEqual(26, current_cell_list.neighbor_cell(8, 4, 3, [3, 3, 3]))

    def test_neighbor_cell_at_back(self):
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

        self.assertEqual(0, current_cell_list.neighbor_cell(18, 22, 3, [3, 3, 3]))
        self.assertEqual(1, current_cell_list.neighbor_cell(19, 22, 3, [3, 3, 3]))
        self.assertEqual(2, current_cell_list.neighbor_cell(20, 22, 3, [3, 3, 3]))
        self.assertEqual(3, current_cell_list.neighbor_cell(21, 22, 3, [3, 3, 3]))
        self.assertEqual(4, current_cell_list.neighbor_cell(22, 22, 3, [3, 3, 3]))
        self.assertEqual(5, current_cell_list.neighbor_cell(23, 22, 3, [3, 3, 3]))
        self.assertEqual(6, current_cell_list.neighbor_cell(24, 22, 3, [3, 3, 3]))
        self.assertEqual(7, current_cell_list.neighbor_cell(25, 22, 3, [3, 3, 3]))
        self.assertEqual(8, current_cell_list.neighbor_cell(26, 22, 3, [3, 3, 3]))


class TestMDVerletList(unittest.TestCase):
    """Class for the unit tests regarding the MD Verlet list speed up scheme."""

    def test_intersection_issue_small_large_2(self):
        """Test for the Verlet list with two small ellipses inside a larger one."""

        rve_dims = [1, 1]

        ellipse_1 = Ellipse(
            "1",
            {
                "major_axis": 0.22567583341910252,
                "minor_axis": 0.11283791670955126,
                "angle": -0.16950065246047263,
            },
            rve_dims,
        )
        ellipse_1.position_center = np.array([0.97905861, 0.11558148])
        ellipse_2 = Ellipse(
            "1",
            {
                "major_axis": 0.05,
                "minor_axis": 0.0319719920308712,
                "angle": -0.04285712792441079,
            },
            rve_dims,
        )
        ellipse_2.position_center = np.array([0.04442033, 0.11730827])
        ellipse_3 = Ellipse(
            "1",
            {
                "major_axis": 0.05,
                "minor_axis": 0.0319719920308712,
                "angle": -0.04285712792441079,
            },
            rve_dims,
        )
        ellipse_3.position_center = np.array([0.09517409, 0.09517425])

        particles = [ellipse_1, ellipse_2, ellipse_3]

        verlet_list = VerletList(1.5)
        molecular_dynamics_sim = MolecularDynamicsSimulation(
            0, 500, 0, 0.01, 0, "random", True, ""
        )
        molecular_dynamics_sim.box = rve_dims
        molecular_dynamics_sim.set_speed_up_scheme(verlet_list)

        verlet_list.a_new_verlet_list_has_to_be_computed = True
        verlet_list.new_list(particles)
        self.assertTrue(verlet_list.particle_list[0] == [0, 1, 2])
        self.assertTrue(verlet_list.particle_list[1] == [0, 1, 2])
        self.assertTrue(verlet_list.particle_list[2] == [0, 1, 2])

    def test_intersection_issue_small_large_3(self):

        rve_dims = [1, 1]

        disk_1 = Disk(
            "1",
            {"r": 0.05},
            rve_dims,
        )
        disk_1.position_center = np.array([0.5, 0.5])

        particles = [disk_1]

        verlet_list = VerletList(1.5)
        molecular_dynamics_sim = MolecularDynamicsSimulation(
            0, 500, 0, 0.01, 0, "random", True, ""
        )
        molecular_dynamics_sim.box = rve_dims
        molecular_dynamics_sim.set_speed_up_scheme(verlet_list)

        verlet_list.new_list(particles)
        molecular_dynamics_sim.speed_up_scheme.verlet_neighborhoods[
            0
        ].position_center = np.array([0.7, 0.7])

        verlet_list.new_list(particles)
        self.assertTrue(
            np.all(
                molecular_dynamics_sim.speed_up_scheme.verlet_neighborhoods[
                    0
                ].position_center
                == np.array([0.5, 0.5])
            )
        )


class TestRSASpeedUpScheme(unittest.TestCase):
    """Class for the unit tests regarding the Random Sequential Adsorption (RSA) speed up schemes."""

    # def test_new_list_abstract(self):
    #     """Test if new_list is an abstract method of RSA_SpeedUpScheme."""

    #     class SpeedUpSchemeTest(RSA_SpeedUpScheme):
    #         pass

    #     with self.assertRaises(TypeError):

    #         _ = SpeedUpSchemeTest()

    # def test_box_property_uses_rsa_sim_box(self):
    #     """Test that the box property is read from the rsa_sim attribute."""
    #     cell_list = RSACellList()
    #     cell_list.rsa_sim = Mock(box=[3.0, 4.0])
    #     self.assertEqual(cell_list.box, [3.0, 4.0])

    # def test_box_property_none_without_rsa_sim(self):
    #     """Test that the box property is None when no rsa_sim has been set."""
    #     cell_list = RSACellList()
    #     self.assertIsNone(cell_list.box)

    # def test_update_max_radius_initial(self):
    #     """Test that the first call to update_max_radius uses the particles already in the box."""
    #     particles = [Mock(radius=0.1), Mock(radius=0.3), Mock(radius=0.2)]
    #     trial_particle = Mock(radius=0.15)
    #     cell_list = RSACellList()

    #     was_updated = cell_list.update_max_radius(particles, trial_particle)

    #     self.assertTrue(was_updated)
    #     self.assertEqual(cell_list.max_radius, 0.3)

    def test_update_max_radius_trial_particle_larger(self):
        """Test that max_radius is updated when the trial particle is bigger."""
        cell_list = RSACellList()
        cell_list.max_radius = 0.3
        trial_particle = Mock(radius=0.5)

        was_updated = cell_list.update_max_radius([], trial_particle)

        self.assertTrue(was_updated)
        self.assertEqual(cell_list.max_radius, 0.5)

    def test_update_max_radius_trial_particle_smaller(self):
        """Test that max_radius is left unchanged when the trial particle is smaller."""
        cell_list = RSACellList()
        cell_list.max_radius = 0.3
        trial_particle = Mock(radius=0.2)

        was_updated = cell_list.update_max_radius([], trial_particle)

        self.assertFalse(was_updated)
        self.assertEqual(cell_list.max_radius, 0.3)

    def test_update_n_cell_dim_without_box_raises(self):
        """Test that update_n_cell_dim raises when the simulation box is undefined."""
        cell_list = RSACellList()
        cell_list.max_radius = 0.25

        with self.assertRaises(ValueError):
            cell_list.update_n_cell_dim()

    def test_update_n_cell_dim_and_cell_side_length(self):
        """Test the computation of n_cell_dim and cell_side_length."""
        cell_list = RSACellList()
        cell_list.box=[2,1]
        cell_list.max_radius = 0.25
        cell_list.update_n_cell_dim()
        cell_list.update_cell_side_length()
        self.assertEqual(cell_list.n_cell_dim, [4,2])
        self.assertEqual(cell_list.cell_side_length,[0.5,0.5])

    def test_get_particle_cell_2d(self):
        """Test the cell index computation for a 2D particle."""
        cell_list = RSACellList()
        cell_list.n_cell_dim = [4, 4]
        cell_list.cell_side_length = [0.5, 0.5]
        particle = Mock(position_center=np.array([1.1, 1.1]), dim=2)
        self.assertEqual(cell_list.get_particle_cell(particle), 10)

    def test_get_particle_cell_3d(self):
        """Test the cell index computation for a 3D particle."""
        cell_list = RSACellList()
        cell_list.n_cell_dim = [2, 2, 2]
        cell_list.cell_side_length = [0.5, 0.5, 0.5]
        particle = Mock(position_center=np.array([0.6, 0.6, 0.6]), dim=3)

        self.assertEqual(cell_list.get_particle_cell(particle), 7)

    def test_new_list_builds_cell_list_and_particle_list(self):
        """Test that new_list assigns the particles to cells and finds the trial particle's neighbors."""
        particles = [
            Mock(radius=0.25, position_center=np.array([0.1, 0.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([0.6, 0.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([1.1, 1.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([1.9, 1.9]), dim=2),
        ]
        trial_particle = Mock(radius=0.25, position_center=np.array([0.55, 0.55]), dim=2)
        cell_list = RSACellList()
        cell_list.rsa_sim = Mock(box=[2.0, 2.0])

        cell_list.new_list(particles, trial_particle)

        self.assertEqual(cell_list.max_radius, 0.25)
        correct_cell_list = 16 * [set()]
        correct_cell_list[0] = {0}
        correct_cell_list[1] = {1}
        correct_cell_list[10] = {2}
        correct_cell_list[15] = {3}
        self.assertTrue(cell_list.cell_list == correct_cell_list)
        # Particle 3 (cell 15) is not a neighbor of the trial particle's cell (cell 5).
        self.assertEqual(sorted(cell_list.particle_list), [0, 1, 2])

    def test_new_list_incrementally_adds_accepted_particle(self):
        """Test that a particle accepted between two calls is added to the cell list without a full rebuild."""
        particles = [
            Mock(radius=0.25, position_center=np.array([0.1, 0.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([0.6, 0.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([1.1, 1.1]), dim=2),
            Mock(radius=0.25, position_center=np.array([1.9, 1.9]), dim=2),
        ]
        trial_particle_1 = Mock(radius=0.25, position_center=np.array([0.55, 0.55]), dim=2)
        cell_list = RSACellList()
        cell_list.rsa_sim = Mock(box=[2.0, 2.0])
        cell_list.new_list(particles, trial_particle_1)
        self.assertEqual(sorted(cell_list.particle_list), [0, 1, 2])

        # trial_particle_1 got accepted, so it now belongs to the simulation box.
        particles_with_accepted = particles + [trial_particle_1]
        trial_particle_2 = Mock(radius=0.25, position_center=np.array([0.55, 0.55]), dim=2)

        cell_list.new_list(particles_with_accepted, trial_particle_2)

        # max_radius did not grow, so the cell list should have been updated incrementally.
        self.assertEqual(cell_list.cell_list[5], {4})
        self.assertEqual(sorted(cell_list.particle_list), [0, 1, 2, 4])

    def test_new_list_rebuilds_cell_list_when_max_radius_grows(self):
        """Test that a bigger trial particle triggers a full rebuild of the cell list."""
        particles = [
            Mock(radius=0.1, position_center=np.array([0.1, 0.1]), dim=2),
            Mock(radius=0.1, position_center=np.array([1.9, 1.9]), dim=2),
        ]
        cell_list = RSACellList()
        cell_list.rsa_sim = Mock(box=[2.0, 2.0])
        trial_particle_1 = Mock(radius=0.1, position_center=np.array([1.0, 1.0]), dim=2)
        cell_list.new_list(particles, trial_particle_1)
        self.assertEqual(cell_list.max_radius, 0.1)

        trial_particle_2 = Mock(radius=0.5, position_center=np.array([1.0, 1.0]), dim=2)
        cell_list.new_list(particles, trial_particle_2)

        self.assertEqual(cell_list.max_radius, 0.5)
        self.assertTrue(all(np.array(cell_list.n_cell_dim) == np.array([2, 2])))
        self.assertEqual(len(cell_list.cell_list), 4)
        self.assertEqual(cell_list.cell_list[0], {0})
        self.assertEqual(cell_list.cell_list[3], {1})


def load_a_troublesome_example(previous_mic_path):
    """Load a troblesome example for debugging."""

    with open(previous_mic_path, "rb") as mic:
        info_previous_sample = pickle.load(mic)
        # No need to generate a new microstructure. Using a previous microstructure.
        current_sample = info_previous_sample["microstructure"]
        current_mic_generator = info_previous_sample["generation_method"]
        trouble_pair = []
        trouble_pair_ind = []
        for i_part_ind, i_particle in enumerate(current_sample.particles):
            if (
                0.55 < i_particle.position_center[0] < 0.65
            ) and 0.05 < i_particle.position_center[1] < 0.25:
                trouble_pair.append(i_particle)
                trouble_pair_ind.append(i_part_ind)
                print(vars(i_particle), i_particle.position_center)

        # intersection, overlap_length, _ = trouble_pair[0].intersection_gjk(
        #     trouble_pair[1], [1, 1]
        # )
        # self.assertTrue(intersection)
        # intersection, overlap_length, _ = trouble_pair[0].intersection_gjk(
        #     trouble_pair[2], [1, 1]
        # )
        # self.assertTrue(intersection)
        # print(
        #     current_mic_generator.speed_up_scheme.particle_list[
        #         trouble_pair_ind[0]
        #     ],
        #     current_mic_generator.speed_up_scheme.particle_list[
        #         trouble_pair_ind[1]
        #     ],
        #     current_mic_generator.speed_up_scheme.particle_list[
        #         trouble_pair_ind[2]
        #     ],
        # )
        plot_particles_2d(
            trouble_pair
            + [
                current_mic_generator.speed_up_scheme.verlet_neighborhoods[ind]
                for ind in trouble_pair_ind
            ],
            [1, 1],
            "",
            show=True,
            save=False,
        )
        # current_mic_generator.speed_up_scheme.a_new_verlet_list_has_to_be_computed = True
        current_mic_generator.speed_up_scheme.new_list(current_sample.particles)
        print(
            current_mic_generator.speed_up_scheme.particle_list[trouble_pair_ind[0]],
            current_mic_generator.speed_up_scheme.particle_list[trouble_pair_ind[1]],
        )
        print(trouble_pair_ind)
        plot_particles_2d(
            trouble_pair
            + [
                current_mic_generator.speed_up_scheme.verlet_neighborhoods[ind]
                for ind in trouble_pair_ind
            ],
            [1, 1],
            "",
            show=True,
            save=False,
        )
