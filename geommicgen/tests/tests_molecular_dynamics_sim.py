"""Module for tests related to the molecular dynamics simulation."""

import unittest
from unittest.mock import sentinel, Mock, MagicMock, patch

# from microstructure.phase import Phase


from micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
import numpy as np


class TestRVENormalization(unittest.TestCase):
    """Tests for rve normalization."""

    @unittest.skip
    @patch("micgenmethod.molecular_dynamics_sim.run_molecular_dynamics_simulation")
    @patch("micgenmethod.molecular_dynamics_sim.__init__")
    def test_normalize_rve_context(self):

        particle_1 = Mock()
        particle_2 = Mock()
        particle_3 = Mock()
        mock_1 = Mock()
        mock_2 = Mock()
        mock_3 = Mock()
        particle_1.normalize_size_according_to_rve = mock_1
        particle_2.normalize_size_according_to_rve = mock_2
        particle_2.normalize_size_according_to_rve = mock_3
        mdsim = MolecularDynamicsSimulation()
        mdsim.generate_microstructure(MagicMock())
        # for
        # check calls to normalize_size_according_to_rve

    @patch.object(
        MolecularDynamicsSimulation,
        "__init__",
        lambda max_residue_per_particle, max_step, max_steps_to_relax, dt, min_distance, type_init_conf, save_history: None,
    )
    def setUp(self):
        self.mdsim = MolecularDynamicsSimulation(None, None, None, None, None, None)
        self.mdsim.microstructure_sample = MagicMock()
        self.mdsim.thermostat = MagicMock()
        self.mdsim.save_history = MagicMock()
        self.mdsim.offset = False
        self.mdsim.box = [1, 1]
        self.mdsim.min_distance = 1
        self.mdsim._original_box = None

    @patch("iofuncs.printing.print_virtual_total_volume_fraction")
    def test_characterization_virtual_particle_sizes_1(self, mock_print_to_file):

        particles = [MagicMock(), MagicMock()]
        with self.mdsim.virtual_particle_sizes(particles):
            pass

        for particle in particles:
            particle.dilate.assert_called_with(1 / 2)
            particle.contract.assert_called_with(1 / 2)

    @patch("iofuncs.printing.print_virtual_total_volume_fraction")
    def test_characterization_virtual_particle_sizes_2(self, mock_print_to_file):

        self.mdsim.microstructure_sample = MagicMock()
        self.mdsim.thermostat = thermostat_mock = MagicMock()
        self.mdsim.thermostat.__class__.__name__ = "MultiTemperatureIsokineticScheme"
        self.mdsim.thermostat.jump_list = jump_list = Mock()
        self.mdsim.save_history = MagicMock()
        self.mdsim.offset = False
        self.mdsim.min_distance = 1
        particles = []
        with self.mdsim.virtual_particle_sizes(particles):
            pass

        thermostat_mock.equilibration_steps.append.assert_called_with(jump_list)

    @patch("iofuncs.printing.print_virtual_total_volume_fraction")
    def test_characterization_virtual_particle_sizes_3(self, mock_print_to_file):

        self.mdsim.microstructure_sample = MagicMock()
        self.mdsim.thermostat = MagicMock()
        self.mdsim.save_history = False
        self.mdsim.position_center_history = [[], []]
        self.mdsim.offset = False
        self.mdsim.min_distance = 1
        particles = [MagicMock(), MagicMock()]
        particles[0].position_center.flatten = Mock(return_value=1)
        particles[1].position_center.flatten = Mock(return_value=0)
        with self.mdsim.virtual_particle_sizes(particles):
            pass

        self.assertEqual(
            self.mdsim.position_center_history,
            [
                [particles[0].position_center.flatten()],
                [particles[1].position_center.flatten()],
            ],
        )

    @patch("iofuncs.printing.print_virtual_total_volume_fraction")
    def test_characterization_virtual_particle_sizes_4(self, mock_print_to_file):

        self.mdsim.microstructure_sample = MagicMock()
        self.mdsim.thermostat = MagicMock()
        self.mdsim.save_history = True
        self.mdsim.offset = True
        self.mdsim.box = [1, 1]
        offset = np.array([1, 1])
        self.mdsim.compute_rve_offset = MagicMock(return_value=offset)
        self.mdsim.min_distance = 1
        particles = [MagicMock(), MagicMock()]
        init_positions = [np.array([1, 1]), np.array([2, 1])]
        particles[0].position_center = init_positions[0]
        particles[1].position_center = init_positions[1]
        with self.mdsim.virtual_particle_sizes(particles):
            pass

        for ind, particle in enumerate(particles):
            self.assertTrue(all(particle.position_center == init_positions[ind]))

    def test_dilate_all_particles(self):

        self.mdsim.min_distance = 1
        particles = [MagicMock(), MagicMock()]
        self.mdsim.dilate_all_particles(particles)

        for particle in particles:
            particle.dilate.assert_called_with(1 / 2)

    def test_contract_all_particles(self):

        self.mdsim.min_distance = 1
        particles = [MagicMock(), MagicMock()]
        self.mdsim.contract_all_particles(particles)

        for particle in particles:
            particle.contract.assert_called_with(1 / 2)

    def test_resize_sim_box_and_all_particles_inside(self):

        self.mdsim.box = [2, 2]
        particles = [MagicMock(), MagicMock()]
        self.mdsim.resize_sim_box_and_all_particles_inside(particles, "unitary")
        self.assertEqual(self.mdsim._original_box, [2, 2])
        self.assertEqual(self.mdsim.box, [1, 1])
        for particle in particles:
            particle.rescale.assert_called_with(1 / 2)
        self.mdsim.resize_sim_box_and_all_particles_inside(particles, "original")
        self.assertEqual(self.mdsim.box, [2, 2])
        for particle in particles:
            particle.rescale.assert_called_with(2)
