"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np


# pylint: disable=import-error
from micgenmethod.microstructure_gen_method import (
    GenerationMethod,
)
from micgenmethod.molecular_dynamics_sim import (
    MolecularDynamicsSimulation,
)
from microstructure.particleclasses import (
    CylindricalFiber,
)


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
        print(self.md_init_mock_kwargs)
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

    def test_virtual_size(self):
        """Test if the virtual size context manager is working."""
        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs.values()
        )
        particles = [Mock() for _ in range(10)]
        current_generation_method.min_distance = 0.1
        with current_generation_method.virtual_particle_sizes(particles):
            pass
        for particle in particles:
            particle.dilate.assert_called_with(0.1)
            particle.contract.assert_called_with(0.1)


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
        current_generation_method.set_speed_up_scheme(Mock(particle_list=[[1], [0]]))
        current_generation_method.particle_forces = [0, 0]
        particle_1 = Mock(position_center=np.array([0.6, 0.5]))
        particle_1.intersection_area.return_value = 0.1
        particle_1.intersection_vector.return_value = np.array([1, 0])
        particle_2 = Mock(position_center=np.array([0.5, 0.5]))
        particle_2.intersection_area.return_value = 0.1
        particle_2.intersection_vector.return_value = np.array([-1, 0])
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
