"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np


# pylint: disable=import-error
from geommicgen.micgenmethod.microstructure_gen_method import (
    GenerationMethod,
)
from geommicgen.micgenmethod.molecular_dynamics_sim import (
    MolecularDynamicsSimulation,
    grid_side,
)
from geommicgen.micgenmethod.speed_up_schemes import Naive
from geommicgen.micgenmethod.thermostats import MultiTemperatureIsokineticThermostat
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.particleclasses import (
    CylindricalFiber,
)
from geommicgen.microstructure.phase import Phase


class MicGenTest(GenerationMethod):
    def generate_microstructure(self, microstructure_sample):
        pass


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        class MicGenTestIncomp(GenerationMethod):
            pass

        with self.assertRaises(TypeError):

            _ = MicGenTestIncomp()


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
        current_generation_method.box = [0.5, 2.0]
        current_generation_method.type_init_conf = "grid"
        current_generation_method.generate_initial_configuration(
            particles,
        )
        velocities = current_generation_method.particle_velocities
        self.assertEqual(len(velocities), len(particles))
        self.assertTrue(all(np.any(i_velocity != 0) for i_velocity in velocities))
        # The box is the list set_box makes it, which the grid used to divide by an
        # integer; and every particle has a velocity of its own, where the grid used to
        # leave one velocity in place of the list of them

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


class TestGridInitialConfiguration(unittest.TestCase):
    """Test class for the initial configuration that places the particles on a grid."""

    def place(self, n_particles, dim):
        """Place mock particles on a grid in a unit box."""
        generator = MolecularDynamicsSimulation(0.0, 10, 5, 1e-3, 0.0, "grid", True)
        particles = [Mock(dim=dim, position_center=None) for _ in range(n_particles)]
        generator.box = [1.0] * dim
        generator.generate_initial_configuration(particles)

        return particles

    def test_every_particle_is_placed(self):
        for i_dim, i_count in ((2, 10), (3, 273), (3, 8), (2, 1)):
            with self.subTest(dim=i_dim, n_particles=i_count):
                particles = self.place(i_count, i_dim)
                self.assertTrue(
                    all(i_particle.position_center is not None for i_particle in particles)
                )
        # 273 is the three dimensional example deck, which used to leave 57 of them with
        # no position because the grid was fixed at six cells a side

    def test_the_particles_are_at_the_centres_of_distinct_cells(self):
        particles = self.place(27, 3)
        centres = {tuple(np.round(i_particle.position_center, 12)) for i_particle in particles}
        self.assertEqual(len(centres), 27)
        side = grid_side(27, 3)
        self.assertEqual(side, 3)
        self.assertTrue(
            all(
                np.allclose((np.array(i_centre) * side) % 1, 0.5)
                for i_centre in centres
            )
        )
        # Twenty-seven particles fit a three-by-three-by-three grid exactly; the float
        # cube root of 27 is a hair over three and used to ask for four

    def test_the_grid_is_the_smallest_that_fits(self):
        for i_dim, i_count, i_side in ((2, 10, 4), (2, 16, 4), (2, 17, 5), (3, 8, 2), (3, 9, 3), (3, 216, 6), (3, 217, 7), (2, 1, 1)):
            with self.subTest(dim=i_dim, n_particles=i_count):
                self.assertEqual(grid_side(i_count, i_dim), i_side)


class TestFixedSeed(unittest.TestCase):
    """Test class for the seed that makes a generation the same in every run."""

    def generate(self, seed):
        """Generate a small microstructure of ellipses with the given seed."""
        microstructure = Microstructure([1.0, 1.0])
        microstructure.add_phase(Phase("0", {"phase_type": 1}))
        microstructure.add_phase(
            Phase(
                "1",
                {
                    "phase_type": 3,
                    "vf": 0.2,
                    "n": 4,
                    "ratio": 1.5,
                    "angle_distribution": "normal",
                    "angle_mean": 0.0,
                    "angle_sigma": 0.3,
                },
            )
        )
        generator = MolecularDynamicsSimulation(
            0.0, 3, 1, 1e-3, 0.0, "random", False, fixed_seed=seed
        )
        generator.set_thermostat(
            MultiTemperatureIsokineticThermostat(
                None, criterion="ratio_in_out", max_ratio_osc=2, temp_low_ratio=1 / 4
            )
        )
        # The thermostat a deck gets when it names none, with the deck's defaults
        generator.set_speed_up_scheme(Naive())
        generator.generate_microstructure(microstructure)

        return microstructure

    def particle_records(self, microstructure):
        """The centre and the orientation of every particle."""
        return [
            (tuple(i_particle.position_center), i_particle.angle)
            for i_particle in microstructure.particles
        ]

    def test_the_same_seed_gives_the_same_microstructure(self):
        first = self.particle_records(self.generate(7))
        second = self.particle_records(self.generate(7))
        self.assertEqual(first, second)
        # The orientations are drawn from the descriptors before the run starts, and
        # the seed used to be set after them, with the initial positions: the positions
        # came out the same in every run and the orientations never did

    def test_another_seed_gives_another_microstructure(self):
        first = self.particle_records(self.generate(7))
        other = self.particle_records(self.generate(8))
        self.assertNotEqual(
            [i_angle for _, i_angle in first], [i_angle for _, i_angle in other]
        )
        # The control: the equality above is not the orientations being constant


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
