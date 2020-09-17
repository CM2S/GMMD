"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

from geommicgen.micgenmethod.microstructure_gen_method import GenerationMethod
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from geommicgen.microstructure.particle_classes import CylindricalFiber


class MicGenTest(GenerationMethod):
    def generate_microstructure(self, microstructure_sample):
        pass


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        # with self.assertRaises(ValueError):

        class MicGenTest(GenerationMethod):
            pass

        with self.assertRaises(TypeError):

            _ = MicGenTest()

    @patch("geommicgen.microstructure.particle_classes.Disk")
    @patch("geommicgen.microstructure.phase.FixedValue")
    def test_generate_particles_number(self, mock_fixed_value, mock_disk):
        gen_method = MicGenTest()
        rve_dims = [1.0, 1.0]
        descriptors = {
            "n": mock_fixed_value("n", 10),
            "vf": mock_fixed_value("vf", 0.1),
        }
        phase = sentinel.phase
        particles = gen_method.generate_particles(
            rve_dims, mock_disk, phase, **descriptors
        )
        self.assertTrue(len(particles), 10)
        for particle in particles:
            self.assertTrue(particle.name, "Disk()")

    @patch("geommicgen.microstructure.particle_classes.Disk")
    @patch("geommicgen.microstructure.phase.FixedValue")
    def test_generate_particles_vf(self, mock_fixed_value, mock_disk):
        gen_method = MicGenTest()
        rve_dims = [1.0, 1.0]
        vf_mock = mock_fixed_value("vf", 0.1)
        vf_mock.value = 0.1
        descriptors = {
            "r": mock_fixed_value("r", 0.1),
            "vf": vf_mock,
        }
        mock_disk.return_value = sentinel.particle
        sentinel.particle.volume = np.pi * 0.1 ** 2
        phase = sentinel.phase
        particles = gen_method.generate_particles(
            rve_dims, mock_disk, phase, **descriptors
        )
        self.assertEqual(len(particles), 4)
        for particle in particles:
            self.assertTrue(particle.name, "Disk()")

    def test_generate_particles_matrix(self):
        self.assertTrue(False)


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

    @patch(
        "geommicgen.micgenmethod.microstructure_gen_method.GenerationMethod.generate_particles"
    )
    @patch(
        "geommicgen.micgenmethod.molecular_dynamics_sim.MolecularDynamicsSimulation.run_molecular_dynamics_simulation",
    )
    def test_generate_microstructure_particles_are_generated(
        self,
        _,
        mock_generate_particles,
    ):
        """Test if the particles are generated for each phase"""

        current_generation_method = MolecularDynamicsSimulation(
            *self.md_init_mock_kwargs
        )
        current_generation_method.type_init_conf = "random"
        mock_microstructure_sample = Mock(rve_dims=[1.0, 1.0])
        phase_1 = Mock()
        phase_2 = Mock()
        phase_3 = Mock()
        mock_microstructure_sample.phases = {
            "1": phase_1,
            "2": phase_2,
            "3": phase_3,
        }

        current_generation_method.generate_microstructure(mock_microstructure_sample)
        mock_generate_particles.assert_has_calls(
            [
                call(
                    mock_microstructure_sample.rve_dims,
                    mock_microstructure_sample.phases["1"].type,
                    mock_microstructure_sample.phases["1"].phase_name,
                    mock_microstructure_sample.phases["1"].descriptors,
                ),
                call(
                    mock_microstructure_sample.rve_dims,
                    mock_microstructure_sample.phases["2"].type,
                    mock_microstructure_sample.phases["2"].phase_name,
                    mock_microstructure_sample.phases["2"].descriptors,
                ),
                call(
                    mock_microstructure_sample.rve_dims,
                    mock_microstructure_sample.phases["3"].type,
                    mock_microstructure_sample.phases["3"].phase_name,
                    mock_microstructure_sample.phases["3"].descriptors,
                ),
            ],
            any_order=True,
        )

    # @patch(
    #     "geommicgen.micgenmethod.microstructure_gen_method.GenerationMethod.generate_particles"
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
        mock_cylindrical_fiber_1.direction_fibers = "x"
        mock_cylindrical_fiber_2.direction_fibers = "x"
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
        self.assertTrue(all(current_generation_method.particle_velocities < 1e-4))

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
        self.assertTrue(any(current_generation_method.particle_velocities != 0))

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
                    current_generation_method.position_center_history[0][part_ind]
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
                    current_generation_method.position_center_history[0][part_ind]
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
                    current_generation_method.position_center_history[0][part_ind]
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
