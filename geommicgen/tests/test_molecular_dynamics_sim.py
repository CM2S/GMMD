"""Module for tests related to the molecular dynamics simulation, class MolecularDynamicsSimulation.

All methods and functionalities defined in MolecularDynamicsSimulation parent class, GenerationMethod, are not tested in this module, but in test_generation.py"""

import unittest
from unittest.mock import sentinel, Mock, MagicMock, patch

# from microstructure.phase import Phase

from geommicgen.microstructure.particleclasses import (
    CylindricalFiber, Matrix, Disk, Ellipsoid
)
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from geommicgen.microstructure.microstructure import Microstructure
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
        particle_3.normalize_size_according_to_rve = mock_3
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

    def test_resize_sim_box_and_all_particles_inside(self):

        self.mdsim.box = [2, 2]
        particles = [MagicMock(), MagicMock()]
        self.mdsim.resize_sim_box_and_all_particles_inside(particles, size="unitary")
        self.assertEqual(self.mdsim._original_box, [2, 2])
        self.assertEqual(self.mdsim.box, [1, 1])
        for particle in particles:
            particle.rescale.assert_called_with(1 / 2)
        self.mdsim.resize_sim_box_and_all_particles_inside(particles, size="original")
        self.assertEqual(self.mdsim.box, [2, 2])
        for particle in particles:
            particle.rescale.assert_called_with(2)

    def test_resize_sim_box_and_all_particles_inside_invalid_size_raises(self):
        """Check that an unknown size choice raises a ValueError."""
        particles = [MagicMock(), MagicMock()]
        with self.assertRaises(ValueError):
            self.mdsim.resize_sim_box_and_all_particles_inside(particles, size="unknown")


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
            ]
        }


    def _build_mock_phase(self, volume = 0.1, type = None):
        mock_phase = Mock()
        if type == Matrix:
            mock_phase.volume = None
        else:
            mock_phase.volume = volume
        mock_phase.type = type
        mock_phase.inner_phase = False
        mock_phase.generate_particles.return_value = None

        return mock_phase


    def test_generate_microstructure_particles_are_generated(self):
        """Test if the particles are generated for each phase"""

        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )

        mock_microstructure_sample = Mock()
        mock_phase1 = self._build_mock_phase(type=Matrix)
        mock_phase2 = self._build_mock_phase(type=Disk)
        mock_phase3 = self._build_mock_phase(type=Ellipsoid)
        mock_microstructure_sample.phases = {
            "mock_phase1": mock_phase1,
            "mock_phase2": mock_phase2,
            "mock_phase3": mock_phase3,
        }
        mock_microstructure_sample.volume_fraction = 2
        # A volume fraction over 1 makes generate_microstructure raise right after the
        # phase-generation loop runs, so the rest of the simulation never needs mocking.

        with self.assertRaises(ValueError):
            current_generation_method.generate_microstructure(mock_microstructure_sample)

        mock_phase1.generate_particles.assert_not_called()
        mock_phase2.generate_particles.assert_called_once_with(
            mock_microstructure_sample.rve_dims
        )
        mock_phase3.generate_particles.assert_called_once_with(
            mock_microstructure_sample.rve_dims
        )

    @patch("geommicgen.iofuncs.printing.print_final_message_md")
    @patch("geommicgen.iofuncs.printing.print_microstructure_info")
    def test_generate_microstructure_success_path_places_inner_phases(
        self, mock_print_info, mock_print_final
    ):
        """Check the full successful path: particles generated, the simulation run, the
        total overlap saved, and inner phases placed once the outer phases are done."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.set_box = Mock()
        current_generation_method.generate_initial_configuration = Mock()
        current_generation_method.run_molecular_dynamics_simulation = Mock()
        current_generation_method.place_inner_phase_rsa = Mock()
        current_generation_method.final_overlap_check = False
        current_generation_method.total_overlap = 0.05
        current_generation_method.total_overlap_history = [0.05]
        current_generation_method.max_residue = 0.1

        mock_microstructure_sample = Mock()
        outer_phase = self._build_mock_phase(volume=0.2, type=Disk)
        outer_phase.volume_fraction = 0.5
        inner_phase = self._build_mock_phase(volume=0.05, type=Disk)
        inner_phase.inner_phase = True
        inner_phase.outer_phase = "mock_phase1"
        inner_phase.volume_fraction = 0.1
        mock_microstructure_sample.phases = {
            "mock_phase1": outer_phase,
            "mock_phase2": inner_phase,
        }
        mock_microstructure_sample.volume_fraction = 0.5
        mock_microstructure_sample.rve_dims = [1, 1]
        mock_microstructure_sample.particles = [Mock(), Mock()]

        current_generation_method.generate_microstructure(mock_microstructure_sample)

        outer_phase.generate_particles.assert_called_once_with(
            mock_microstructure_sample.rve_dims
        )
        current_generation_method.set_box.assert_called_once_with(
            mock_microstructure_sample.particles, mock_microstructure_sample.rve_dims
        )
        current_generation_method.generate_initial_configuration.assert_called_once_with(
            mock_microstructure_sample.particles
        )
        current_generation_method.run_molecular_dynamics_simulation.assert_called_once_with(
            mock_microstructure_sample.particles
        )
        self.assertEqual(mock_microstructure_sample.total_overlap, 0.05)
        inner_phase.generate_particles.assert_called_once_with(
            mock_microstructure_sample.rve_dims
        )
        current_generation_method.place_inner_phase_rsa.assert_called_once_with(
            inner_phase, outer_phase
        )

    @patch("geommicgen.iofuncs.printing.print_final_message_md")
    @patch("geommicgen.iofuncs.printing.print_microstructure_info")
    def test_generate_microstructure_inner_phase_volume_fraction_over_outer_raises(
        self, mock_print_info, mock_print_final
    ):
        """Check that an inner phase with a volume fraction over its outer phase's raises,
        without ever placing it."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.set_box = Mock()
        current_generation_method.generate_initial_configuration = Mock()
        current_generation_method.run_molecular_dynamics_simulation = Mock()
        current_generation_method.place_inner_phase_rsa = Mock()
        current_generation_method.final_overlap_check = False
        current_generation_method.total_overlap = 0.0
        current_generation_method.total_overlap_history = [0.0]
        current_generation_method.max_residue = 0.1

        mock_microstructure_sample = Mock()
        outer_phase = self._build_mock_phase(volume=0.2, type=Disk)
        outer_phase.volume_fraction = 0.3
        inner_phase = self._build_mock_phase(volume=0.05, type=Disk)
        inner_phase.inner_phase = True
        inner_phase.outer_phase = "mock_phase1"
        inner_phase.volume_fraction = 0.5
        mock_microstructure_sample.phases = {
            "mock_phase1": outer_phase,
            "mock_phase2": inner_phase,
        }
        mock_microstructure_sample.volume_fraction = 0.3
        mock_microstructure_sample.rve_dims = [1, 1]
        mock_microstructure_sample.particles = [Mock(), Mock()]

        with self.assertRaises(ValueError):
            current_generation_method.generate_microstructure(mock_microstructure_sample)

        current_generation_method.place_inner_phase_rsa.assert_not_called()

    def test_set_thermostat(self):
        """Check that the thermostat is stored and wired back to the simulation."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        thermostat = Mock()
        current_generation_method.set_thermostat(thermostat)
        self.assertIs(current_generation_method.thermostat, thermostat)
        self.assertIs(thermostat.molecular_dynamics_sim, current_generation_method)

    def test_set_speed_up_scheme(self):
        """Check that the speed up scheme is stored and wired back to the simulation."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        speed_up_scheme = Mock()
        current_generation_method.set_speed_up_scheme(speed_up_scheme)
        self.assertIs(current_generation_method.speed_up_scheme, speed_up_scheme)
        self.assertIs(speed_up_scheme.molecular_dynamics_sim, current_generation_method)

    def test_set_box(self):
        """Check if the simulation box is set correctly for any particle except a cylindrical fiber."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        rve_dims_options = [
            [1.0, 2.0],
            [1.0, 2.0, 3.0]
        ]
        for rve_dims in rve_dims_options:
            with self.subTest(rve_dims):
                mock_particle_1 = Mock()
                mock_particle_2 = Mock()
                particles = [mock_particle_1, mock_particle_2]
                current_generation_method.set_box(particles, rve_dims)
                self.assertEqual(current_generation_method.box, rve_dims)


    def test_set_box_cylindrical_fiber_set_box(self):
        """Check if the simulation box is correctly set if there a cylindrical fibers."""

        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        rve_dims = [1.0, 2.0, 3.0]
        mock_cylindrical_fiber_1 = Mock()
        mock_cylindrical_fiber_2 = Mock()
        mock_cylindrical_fiber_1.__class__ = CylindricalFiber
        mock_cylindrical_fiber_1.direction_fibers = 0
        mock_cylindrical_fiber_2.direction_fibers = 0
        particles = [mock_cylindrical_fiber_1, mock_cylindrical_fiber_2]
        current_generation_method.set_box(particles, rve_dims)
        self.assertEqual(current_generation_method.box, [2.0, 3.0])

    def test_set_box_other_particles(self):
        """Check if the simulation box is correctly set if there no a cylindrical fibers."""

        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        rve_dims = [2.0, 3.0]
        mock_disk = Mock()
        mock_ellipse = Mock()
        particles = [mock_disk, mock_ellipse]
        current_generation_method.set_box(particles, rve_dims)
        self.assertEqual(current_generation_method.box, [2.0, 3.0])

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

    @patch("builtins.print")
    def test_generate_initial_configuration_unsupported_type_prints_message(
        self, mock_print
    ):
        """Check that an unsupported initial configuration type does not raise, and instead prints an explanatory message."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=2) for _ in range(3)]
        current_generation_method.box = np.array([1.0, 1.0])
        current_generation_method.type_init_conf = "unsupported_type"
        current_generation_method.generate_initial_configuration(particles)
        mock_print.assert_called_once_with(
            "The initial configuration type unsupported_type is not supported."
        )

    def test_generate_initial_configuration_bcc(self):
        """Check that a bcc initial configuration places all particles, with zero initial
        velocity, and saves their positions in the history."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=3, position_center=None) for _ in range(128)]
        current_generation_method.box = np.array([4.0, 4.0, 4.0])
        current_generation_method.type_init_conf = "bcc"
        current_generation_method.generate_initial_configuration(particles)
        for part_ind, particle in enumerate(particles):
            self.assertIsNotNone(particle.position_center)
            self.assertTrue(
                np.all(current_generation_method.particle_velocities[part_ind] == 0)
            )
            self.assertTrue(
                all(
                    current_generation_method.position_center_history[part_ind][0]
                    == particle.position_center
                )
            )

    def test_generate_initial_configuration_fcc(self):
        """Check that a fcc initial configuration places all particles, with zero initial
        velocity, and saves their positions in the history."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock(dim=3, position_center=None) for _ in range(256)]
        current_generation_method.box = np.array([4.0, 4.0, 4.0])
        current_generation_method.type_init_conf = "fcc"
        current_generation_method.generate_initial_configuration(particles)
        for part_ind, particle in enumerate(particles):
            self.assertIsNotNone(particle.position_center)
            self.assertTrue(
                np.all(current_generation_method.particle_velocities[part_ind] == 0)
            )
            self.assertTrue(
                all(
                    current_generation_method.position_center_history[part_ind][0]
                    == particle.position_center
                )
            )

    def test_set_initial_temp_sets_reference_temp_when_none(self):
        """Check that the reference temperature is computed from the equipartition
        theorem when it was not already set."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = False
        current_generation_method.delta_t = 0.1
        current_generation_method.initial_vel_coeff = 0.5
        current_generation_method.particle_mass_opt = "unit"
        current_generation_method.thermostat = Mock(reference_temp=None, k_b=1.0)
        particle_1 = Mock(dim=2, radius=0.2)
        particle_1.mass.return_value = 1.0
        particle_2 = Mock(dim=2, radius=0.2)
        particle_2.mass.return_value = 1.0
        current_generation_method.set_initial_temp([particle_1, particle_2])
        self.assertAlmostEqual(current_generation_method.thermostat.reference_temp, 0.5)

    def test_set_initial_temp_skips_when_already_set(self):
        """Check that an already-set reference temperature is left untouched."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.thermostat = Mock(reference_temp=1.23)
        current_generation_method.set_initial_temp([Mock(), Mock()])
        self.assertEqual(current_generation_method.thermostat.reference_temp, 1.23)

    def test_check_overlap_naive(self):
        """Check that the naive overlap check sums the overlap of every particle pair
        exactly once."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.force_option = "intersection_area"
        current_generation_method.box = [1, 1]
        particles = [Mock() for _ in range(3)]
        for particle in particles:
            particle.intersection_area.return_value = (0.1, np.array([1, 0]))
        current_generation_method.check_overlap_naive(particles)
        self.assertAlmostEqual(current_generation_method.total_overlap, 0.3)
        particles[0].intersection_area.assert_any_call(particles[1], [1, 1])
        particles[0].intersection_area.assert_any_call(particles[2], [1, 1])
        particles[1].intersection_area.assert_any_call(particles[2], [1, 1])


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

    def test_compute_forces_calls_overlap_thermostat_and_damping(self):
        """Check that compute_forces resets the forces and delegates to the three
        force-computation methods."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.force_rescale = False
        current_generation_method.compute_forces_overlap = Mock()
        current_generation_method.compute_forces_thermostat = Mock()
        current_generation_method.compute_forces_damping = Mock()
        particles = [Mock(dim=2), Mock(dim=2)]
        current_generation_method.compute_forces(particles)
        current_generation_method.compute_forces_overlap.assert_called_once_with(particles)
        current_generation_method.compute_forces_thermostat.assert_called_once_with(
            particles
        )
        current_generation_method.compute_forces_damping.assert_called_once_with(particles)
        self.assertEqual(len(current_generation_method.particle_forces), 2)
        for force in current_generation_method.particle_forces:
            self.assertTrue(np.all(force == np.zeros(2)))

    def test_compute_forces_with_force_rescale(self):
        """Check that the forces are rescaled by force_rescale_coeff when force_rescale is
        enabled and the current max force is non-zero."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.force_rescale = True
        current_generation_method.force_rescale_coeff = 2.0
        current_generation_method.current_max_force = 1.0

        def _set_forces(particles):
            current_generation_method.particle_forces = [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
            ]

        current_generation_method.compute_forces_overlap = Mock(side_effect=_set_forces)
        current_generation_method.compute_forces_thermostat = Mock()
        current_generation_method.compute_forces_damping = Mock()
        particles = [Mock(dim=2), Mock(dim=2)]
        current_generation_method.compute_forces(particles)
        self.assertTrue(
            np.allclose(current_generation_method.particle_forces[0], np.array([2.0, 0.0]))
        )
        self.assertTrue(
            np.allclose(current_generation_method.particle_forces[1], np.array([0.0, 2.0]))
        )

    def test_compute_forces_thermostat_no_force_coeff_is_a_no_op(self):
        """Check that no force is added when the thermostat has no force coefficient."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.particle_forces = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        current_generation_method.particle_velocities = [
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
        ]
        current_generation_method.set_thermostat(Mock(force_coeff=None))
        particles = [Mock(), Mock()]
        current_generation_method.compute_forces_thermostat(particles)
        self.assertTrue(
            np.array_equal(
                current_generation_method.particle_forces[0], np.array([1.0, 2.0])
            )
        )
        self.assertTrue(
            np.array_equal(
                current_generation_method.particle_forces[1], np.array([3.0, 4.0])
            )
        )

    def test_compute_forces_damping_zero_coeff_is_a_no_op(self):
        """Check that no force is added when the damping coefficient is zero."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.particle_forces = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        current_generation_method.particle_velocities = [
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
        ]
        current_generation_method.damping_coeff = 0
        particles = [Mock(), Mock()]
        current_generation_method.compute_forces_damping(particles)
        self.assertTrue(
            np.array_equal(
                current_generation_method.particle_forces[0], np.array([1.0, 2.0])
            )
        )
        self.assertTrue(
            np.array_equal(
                current_generation_method.particle_forces[1], np.array([3.0, 4.0])
            )
        )

    def test_compute_adaptive_time_step_disabled_keeps_delta_t(self):
        """Check that delta_t is left untouched, but still recorded, when dt_adapt is
        disabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = False
        current_generation_method.delta_t = 0.42
        particles = [Mock(radius=0.1), Mock(radius=0.2)]
        current_generation_method.compute_adaptive_time_step(particles)
        self.assertEqual(current_generation_method.delta_t, 0.42)
        self.assertEqual(current_generation_method.all_dt, [0.42])

    def test_compute_adaptive_time_step_intersection_length(self):
        """Check the adaptive time step formula for the intersection_length force
        option."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = True
        current_generation_method.force_option = "intersection_length"
        current_generation_method.coord_number = 4
        particles = [Mock(radius=0.1), Mock(radius=0.1)]
        current_generation_method.compute_adaptive_time_step(particles)
        harm_r = 0.1
        expected_delta_t = np.sqrt(2 / 4) * np.sqrt(harm_r)
        self.assertAlmostEqual(current_generation_method.delta_t, expected_delta_t)

    def test_compute_adaptive_time_step_force_spring_with_velocity(self):
        """Check the adaptive time step formula for the force_spring force option, when
        the particles are moving fast enough to affect the effective stiffness."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = True
        current_generation_method.force_option = "force_spring"
        current_generation_method.coord_number = 1
        current_generation_method.delta_t = 0.1
        current_generation_method.particle_velocities = [
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
        ]
        particles = [Mock(radius=1.0), Mock(radius=1.0)]
        current_generation_method.compute_adaptive_time_step(particles)
        harm_r = 1.0
        max_vel = 1.0
        k_eff = 2 * (2 * harm_r - max_vel * 0.1) / (2 * harm_r)
        expected_delta_t = np.sqrt(2 / 1) * np.sqrt(harm_r / k_eff)
        self.assertAlmostEqual(current_generation_method.delta_t, expected_delta_t)

    @patch("geommicgen.micgenmethod.molecular_dynamics_sim.verlet_sync_integration")
    def test_integrate_updates_state_and_wraps_periodic_boundary(
        self, mock_verlet_integration
    ):
        """Check that integrate updates the particle's position/velocity from the
        integration scheme's result, wrapping the position around the periodic box."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = False
        current_generation_method.delta_t = 0.1
        current_generation_method.particle_mass_opt = "unit"
        current_generation_method.save_history = False
        current_generation_method.box = np.array([1.0])
        current_generation_method.particle_forces = [np.array([0.0])]
        current_generation_method.particle_velocities = [np.array([0.0])]
        particle = Mock(dim=1, position_center=np.array([0.9]))
        particle.mass.return_value = 1.0
        mock_verlet_integration.return_value = [np.array([[1.5]]), np.array([[0.2]])]
        current_generation_method.integrate([particle])
        self.assertTrue(np.allclose(particle.position_center, np.array([0.5])))
        self.assertTrue(
            np.allclose(current_generation_method.particle_velocities[0], np.array([0.2]))
        )

    @patch("geommicgen.micgenmethod.molecular_dynamics_sim.verlet_sync_integration")
    def test_integrate_saves_position_history_when_enabled(
        self, mock_verlet_integration
    ):
        """Check that the wrapped position is appended to the history when save_history is
        enabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.dt_adapt = False
        current_generation_method.delta_t = 0.1
        current_generation_method.particle_mass_opt = "unit"
        current_generation_method.save_history = True
        current_generation_method.position_center_history = [[]]
        current_generation_method.box = np.array([1.0])
        current_generation_method.particle_forces = [np.array([0.0])]
        current_generation_method.particle_velocities = [np.array([0.0])]
        particle = Mock(dim=1, position_center=np.array([0.2]))
        particle.mass.return_value = 1.0
        mock_verlet_integration.return_value = [np.array([[0.3]]), np.array([[0.1]])]
        current_generation_method.integrate([particle])
        self.assertEqual(len(current_generation_method.position_center_history[0]), 1)
        self.assertTrue(
            np.allclose(
                current_generation_method.position_center_history[0][0], np.array([0.3])
            )
        )

    def test_compute_relative_energy(self):
        """Check that the relative energy is the squared norm of the vector of force
        norms."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = False
        current_generation_method.particle_forces = [
            np.array([3.0, 4.0]),
            np.array([0.0, 0.0]),
        ]
        current_generation_method.compute_relative_energy()
        self.assertAlmostEqual(current_generation_method.relative_energy, 25.0)

    def test_relative_energy_setter_appends_to_history_when_save_history(self):
        """Check that setting the relative energy appends to its history only when
        save_history is enabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = True
        current_generation_method.relative_energy = 1.5
        current_generation_method.relative_energy = 2.5
        self.assertEqual(current_generation_method.relative_energy_history, [1.5, 2.5])

    def test_relative_energy_setter_does_not_append_when_not_save_history(self):
        """Check that setting the relative energy does not append to its history when
        save_history is disabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = False
        current_generation_method.relative_energy = 1.5
        self.assertEqual(current_generation_method.relative_energy_history, [])

    def test_compute_kinetic_energy(self):
        """Check that the kinetic energy sums 1/2 * mass * speed**2 over all particles."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = False
        current_generation_method.particle_mass_opt = "unit"
        particle_1 = Mock()
        particle_1.mass.return_value = 2.0
        particle_2 = Mock()
        particle_2.mass.return_value = 1.0
        current_generation_method.particle_velocities = [
            np.array([1.0, 0.0]),
            np.array([0.0, 2.0]),
        ]
        current_generation_method.compute_kinetic_energy([particle_1, particle_2])
        self.assertAlmostEqual(current_generation_method.kinetic_energy, 3.0)

    def test_kinetic_energy_setter_appends_to_history_when_save_history(self):
        """Check that setting the kinetic energy appends to its history only when
        save_history is enabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = True
        current_generation_method.kinetic_energy = 1.0
        current_generation_method.kinetic_energy = 2.0
        self.assertEqual(current_generation_method.kinetic_energy_history, [1.0, 2.0])

    def test_compute_thermic_energy(self):
        """Check that the thermic energy follows 1/2 * k_b * reference_temp * dim *
        n_particles."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = False
        current_generation_method.thermostat = Mock(k_b=2.0, reference_temp=3.0)
        particles = [Mock(dim=2), Mock(dim=2)]
        current_generation_method.compute_thermic_energy(particles)
        self.assertAlmostEqual(current_generation_method.thermic_energy, 12.0)

    def test_thermic_energy_setter_appends_to_history_when_save_history(self):
        """Check that setting the thermic energy appends to its history only when
        save_history is enabled."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.save_history = True
        current_generation_method.thermic_energy = 5.0
        self.assertEqual(current_generation_method.thermic_energy_history, [5.0])


class TestMolecularDynamicSimulationRun(unittest.TestCase):
    """Test class for `.MolecularDynamicsSimulation.run_molecular_dynamics_simulation`."""

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
            ]
        }

    @patch("geommicgen.iofuncs.file_handling.save_mic")
    @patch("geommicgen.iofuncs.printing.print_to_terminal_refresh")
    def test_stops_after_configuration_stays_legal_for_max_steps_to_relax(
        self, mock_print_refresh, mock_save_mic
    ):
        """Check that the simulation loop stops as soon as the configuration has stayed
        legal for max_steps_to_relax consecutive steps, and reports success."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        current_generation_method.virtual_particle_sizes = MagicMock()
        current_generation_method.compute_forces = Mock()
        current_generation_method.compute_relative_energy = Mock()
        current_generation_method.compute_kinetic_energy = Mock()
        current_generation_method.compute_thermic_energy = Mock()
        current_generation_method.set_initial_temp = Mock()
        current_generation_method.integrate = Mock()
        current_generation_method.thermostat = Mock()
        current_generation_method.max_step = 100
        current_generation_method.max_steps_to_relax = 2
        current_generation_method.max_residue_per_particle = 0.1
        current_generation_method.total_overlap = 0.0
        current_generation_method.total_overlap_history = []
        current_generation_method.save_history = False
        current_generation_method.kinetic_energy = 0.0

        particles = [Mock(), Mock()]
        current_generation_method.run_molecular_dynamics_simulation(particles)

        self.assertEqual(current_generation_method.step, 3)
        self.assertTrue(current_generation_method.status)
        self.assertEqual(current_generation_method.max_residue, 0.2)


class TestPlaceInnerPhaseRSA(unittest.TestCase):
    """Test class for `.MolecularDynamicsSimulation.place_inner_phase_rsa`."""

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
            ]
        }

    def test_places_a_single_inner_particle_inside_the_only_outer_particle(self):
        """Check the RSA placement for the simplest case: one outer particle able to
        entirely accept one inner particle on the first attempt."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )

        outer_particle = Mock()
        outer_particle.volume = 1.0
        outer_particle.generate_point_inside.return_value = np.array([0.5, 0.5])

        inner_particle = Mock()
        inner_particle.volume = 0.1
        inner_particle.radius = 0.05

        outer_phase = Mock(particles=[outer_particle], volume_fraction=1.0)
        inner_phase = Mock(particles=[inner_particle])
        inner_phase.microstructure.rve_dims = [1, 1]
        inner_phase.microstructure.volume = 1.0

        current_generation_method.place_inner_phase_rsa(inner_phase, outer_phase)

        self.assertTrue(
            np.allclose(inner_particle.position_center, np.array([0.5, 0.5]))
        )
        inner_particle.dilate.assert_called_once_with(0.01 * 0.05)
        inner_particle.contract.assert_called_once_with(0.01 * 0.05)
        outer_particle.contract.assert_called_once_with(0.05 * 1.05)
        outer_particle.dilate.assert_called_once_with(0.05 * 1.05)
