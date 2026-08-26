"""Module for tests related to the molecular dynamics simulation.

All the MolecularDynamicsSimulation-specific behavior lives here: initial configuration
setup, force computation, and virtual particle sizing. Behavior shared with other generation
methods (e.g. set_box) is tested in test_generation_method.py instead.
"""

import unittest
from unittest.mock import sentinel, Mock, MagicMock, patch

# from microstructure.phase import Phase


from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
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

    @patch("geommicgen.iofuncs.printing.print_virtual_total_volume_fraction")
    def test_characterization_virtual_particle_sizes_1(self, mock_print_to_file):

        particles = [MagicMock(), MagicMock()]
        with self.mdsim.virtual_particle_sizes(particles):
            pass

        for particle in particles:
            particle.dilate.assert_called_with(1 / 2)
            particle.contract.assert_called_with(1 / 2)

    @patch("geommicgen.iofuncs.printing.print_virtual_total_volume_fraction")
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

    @patch("geommicgen.iofuncs.printing.print_virtual_total_volume_fraction")
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

    @patch("geommicgen.iofuncs.printing.print_virtual_total_volume_fraction")
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


class TestMolecularDynamicSimulation(unittest.TestCase):
    """Test class for the MolecularDynamicsSimulation class"""

    def setUp(self):
        self.md_init_mock_kwargs = {
            key: Mock()
            for key in [
                "max_residue_per_particle",
                "max_step",
                "max_steps_to_relax",
                "dt",
                "min_distance",
                "type_init_conf",
                "save_history",
                "sample_dir",
            ]
        }

    # @patch("micgenmethod.microstructure_gen_method.GenerationMethod.generate_particles")
    # @patch(
    #     "micgenmethod.molecular_dynamics_sim.MolecularDynamicsSimulation.run_molecular_dynamics_simulation",
    # )
    # def test_generate_microstructure_particles_are_generated(
    #     self,
    #     _,
    #     mock_generate_particles,
    # ):
    #     """Test if the particles are generated for each phase"""
    #
    #     current_generation_method = MolecularDynamicsSimulation(
    #         *self.md_init_mock_kwargs
    #     )
    #     current_generation_method.type_init_conf = "random"
    #     mock_microstructure_sample = Mock(rve_dims=[1.0, 1.0])
    #     phase_1 = Mock()
    #     phase_2 = Mock()
    #     phase_3 = Mock()
    #     mock_microstructure_sample.phases = {
    #         "1": phase_1,
    #         "2": phase_2,
    #         "3": phase_3,
    #     }
    #
    #     current_generation_method.generate_microstructure(mock_microstructure_sample)
    #     mock_generate_particles.assert_has_calls(
    #         [
    #             call(
    #                 mock_microstructure_sample.rve_dims,
    #                 mock_microstructure_sample.phases["1"].type,
    #                 mock_microstructure_sample.phases["1"].phase_name,
    #                 mock_microstructure_sample.phases["1"].descriptors,
    #             ),
    #             call(
    #                 mock_microstructure_sample.rve_dims,
    #                 mock_microstructure_sample.phases["2"].type,
    #                 mock_microstructure_sample.phases["2"].phase_name,
    #                 mock_microstructure_sample.phases["2"].descriptors,
    #             ),
    #             call(
    #                 mock_microstructure_sample.rve_dims,
    #                 mock_microstructure_sample.phases["3"].type,
    #                 mock_microstructure_sample.phases["3"].phase_name,
    #                 mock_microstructure_sample.phases["3"].descriptors,
    #             ),
    #         ],
    #         any_order=True,
    #     )

    # @patch(
    #     "particleclassesmicgenmethod.microstructure_gen_method.GenerationMethod.generate_particles"
    # )
    # def test_generate_microstructure_set_box(self, mock_generate_particles):
    #     """Set the simulation box correctly."""
    #
    #     mock_generate_particles.return_value = Mock()
    #     mock_generate_particles.return_value
    #     mock_microstructure_sample = Mock()
    #     phase_1 = Mock()
    #     phase_2 = Mock()
    #     mock_microstructure_sample.phases = {
    #         "1": phase_1,
    #         "2": phase_2,
    #     }
    #     phase_2.type == Mock()
    #
    #     self.current_generation_method.generate_microstructure(
    #         mock_microstructure_sample
    #     )
    #     self.assertEqual(self.current_generation_method.box, [1.0, 1.0])

    def test_generate_initial_configuration_inside_box_random(self):
        """Check if the particles are all inside the simulation box for random initial
        configuration"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 2.0])
        current_generation_method.type_init_conf = "random"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for particle in particles:
            self.assertTrue(all(particle.position_center < np.array([1.0, 2.0])))

    def test_generate_initial_configuration_inside_box_grid_2d(self):
        """Check if the particles are all inside the simulation box for a grid configuration
        in 2D"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([0.5, 2.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for particle in particles:
            self.assertTrue(all(particle.position_center < np.array([0.5, 2.0])))

    def test_generate_initial_configuration_inside_box_grid_3d(self):
        """Check if the particles are all inside the simulation box for a grid configuration
        in 3D"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=3, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 0.3, 5.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for particle in particles:
            self.assertTrue(all(particle.position_center < np.array([1.0, 0.3, 5.0])))

    def test_generate_initial_configuration_velocities_zero_random(self):
        """Check if the particles for a random initial configuration all have zero
        velocity"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 2.0])
        current_generation_method.type_init_conf = "random"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        self.assertTrue(
            np.all(np.array(current_generation_method.particle_velocities) < 1e-4)
        )

    def test_generate_initial_configuration_velocities_grid_2d(self):
        """Check if any of the particles for a grid configuration in 2D has non-zero
        velocity"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2) for _ in range(10)]
        current_generation_method.box = np.array([0.5, 2.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        self.assertTrue(any(current_generation_method.particle_velocities != 0))

    def test_generate_initial_configuration_velocities_grid_3d(self):
        """Check if any of the particles for a grid configuration in 3D has non-zero
        velocity"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=3) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 0.3, 5.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        self.assertTrue(np.any(current_generation_method.particle_velocities != 0))

    def test_generate_initial_configuration_save_history_random(self):
        """Check if particle's position is saved for a random initial configuration"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 2.0])
        current_generation_method.type_init_conf = "random"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for part_ind, particle in enumerate(particles):
            self.assertTrue(
                all(
                    current_generation_method.position_center_history[part_ind][0]
                    == particle.position_center
                )
            )

    def test_generate_initial_configuration_save_history_grid_2d(self):
        """Check if particle's position is saved for a grid configuration in 2D"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([0.5, 2.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for part_ind, particle in enumerate(particles):
            self.assertTrue(
                all(
                    current_generation_method.position_center_history[part_ind][0]
                    == particle.position_center
                )
            )

    def test_generate_initial_configuration_save_history_grid_3d(self):
        """Check if particle's position is saved for a grid configuration in 3D"""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=3, position_center=None) for _ in range(10)]
        current_generation_method.box = np.array([1.0, 0.3, 5.0])
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        for part_ind, particle in enumerate(particles):
            self.assertTrue(
                all(
                    current_generation_method.position_center_history[part_ind][0]
                    == particle.position_center
                )
            )


class TestMolecularDynamicSimulationForce(unittest.TestCase):
    def setUp(self):
        self.md_init_mock_kwargs = {
            key: Mock()
            for key in [
                "max_residue_per_particle",
                "max_step",
                "max_steps_to_relax",
                "dt",
                "min_distance",
                "type_init_conf",
                "save_history",
                "sample_dir",
            ]
        }

    def test_compute_forces_overlap(self):
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.force_option = "intersection_area"
        current_generation_method.thermostat = Mock()
        current_generation_method.thermostat.kin_energy_div = False
        current_generation_method.set_speed_up_scheme(Mock(particle_list=[[1], [0]]))
        current_generation_method.particle_forces = [0, 0]
        particle_1 = Mock(position_center=np.array([0.6, 0.5]))
        particle_1.intersection_area.return_value = (0.1, np.array([1, 0]))
        particle_2 = Mock(position_center=np.array([0.5, 0.5]))
        particle_2.intersection_area.return_value = (0.1, np.array([-1, 0]))
        particles = [particle_1, particle_2]
        current_generation_method.compute_forces_overlap(particles)
        self.assertTrue(current_generation_method.total_overlap == 0.1)
        self.assertTrue(
            np.all(current_generation_method.particle_forces[0] == np.array([-0.1, 0]))
        )
        self.assertTrue(
            np.all(current_generation_method.particle_forces[1] == np.array([0.1, 0]))
        )

    def test_compute_forces_thermostat(self):
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.particle_forces = [0, 0]
        particle_1 = Mock(position_center=np.array([0.6, 0.5]))
        particle_2 = Mock(position_center=np.array([0.5, 0.5]))
        particles = [particle_1, particle_2]
        current_generation_method.particle_velocities = [
            np.array([0.1, 0.2]),
            np.array([0.2, -0.1]),
        ]
        current_generation_method.set_thermostat(Mock(force_coeff=0.1))
        current_generation_method.compute_forces_thermostat(particles)
        self.assertTrue(
            np.all(
                np.abs(
                    current_generation_method.particle_forces[0]
                    - np.array([-0.01, -0.02])
                )
                < 1e-4
            )
        )
        self.assertTrue(
            np.all(
                np.abs(
                    current_generation_method.particle_forces[1]
                    - np.array([-0.02, 0.01])
                )
                < 1e-4
            )
        )

    def test_compute_forces_damping(self):
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.particle_forces = [0, 0]
        particle_1 = Mock(position_center=np.array([0.6, 0.5]))
        particle_2 = Mock(position_center=np.array([0.5, 0.5]))
        particles = [particle_1, particle_2]
        current_generation_method.particle_velocities = [
            np.array([0.1, 0.2]),
            np.array([0.2, -0.1]),
        ]
        current_generation_method.damping_coeff = 0.1
        current_generation_method.compute_forces_damping(particles)
        self.assertTrue(
            np.all(
                np.abs(
                    current_generation_method.particle_forces[0]
                    - np.array([-0.01, -0.02])
                )
                < 1e-4
            )
        )
        self.assertTrue(
            np.all(
                np.abs(
                    current_generation_method.particle_forces[1]
                    - np.array([-0.02, 0.01])
                )
                < 1e-4
            )
        )

