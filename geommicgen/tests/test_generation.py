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
    OPTIONAL_OPTIONS,
    REQUIRED_OPTIONS,
    MolecularDynamicsSimulation,
    grid_side,
)
from geommicgen.iofuncs.keywords import top_level_reader
from geommicgen.micgenmethod.speed_up_schemes import CellList, Naive
from geommicgen.micgenmethod.thermostats import (
    IsokineticThermostat,
    MultiTemperatureIsokineticThermostat,
)
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.particleclasses import (
    CylindricalFiber,
    Disk,
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


def deck_defaults():
    """Give the generation parameters of a deck that sets only what has no default."""
    options = {
        i_keyword.name.lower(): i_keyword.default_value
        for i_keyword in top_level_reader.top_level_keywords
        if getattr(i_keyword, "keyword_group", None) == "Mic_Gen_Parameters"
        and hasattr(i_keyword, "default_value")
    }
    options.update({"max_residue_per_particle": 0.0, "max_step": 10, "initial_temp": 1e5})

    return options


class TestFromOptions(unittest.TestCase):
    """Test class for the simulation built from the generation parameters of a deck."""

    def test_a_deck_of_defaults_builds_the_documented_simulation(self):
        simulation = MolecularDynamicsSimulation.from_options(deck_defaults())
        self.assertIsInstance(
            simulation.thermostat, MultiTemperatureIsokineticThermostat
        )
        self.assertIs(simulation.thermostat.molecular_dynamics_sim, simulation)
        self.assertIsInstance(simulation.speed_up_scheme, CellList)
        self.assertIs(simulation.speed_up_scheme.molecular_dynamics_sim, simulation)
        self.assertEqual(simulation.max_step, 10)
        self.assertEqual(simulation.delta_t, 0.05)
        self.assertEqual(simulation.type_init_conf, "random")
        # Read off the keywords, so the defaults the documentation states are the ones
        # a simulation is built with

    def test_every_parameter_of_the_deck_is_read(self):
        options = deck_defaults()
        options.update(
            {
                "damping_coeff": 0.3,
                "particle_mass_opt": "unit",
                "force_rescale": True,
                "dt_adapt": False,
                "offset": False,
                "fixed_seed": 7,
                "initial_vel_coeff": 0.5,
                "final_overlap_check": True,
                "save_history": True,
                "min_distance": 0.01,
            }
        )
        simulation = MolecularDynamicsSimulation.from_options(options)
        self.assertEqual(simulation.damping_coeff, 0.3)
        self.assertEqual(simulation.particle_mass_opt, "unit")
        self.assertTrue(simulation.force_rescale)
        self.assertFalse(simulation.dt_adapt)
        self.assertFalse(simulation.offset)
        self.assertEqual(simulation.fixed_seed, 7)
        self.assertEqual(simulation.initial_vel_coeff, 0.5)
        self.assertTrue(simulation.final_overlap_check)
        self.assertTrue(simulation.save_history)
        self.assertEqual(simulation.min_distance, 0.01)

    def test_the_options_are_the_ones_the_initializer_takes(self):
        simulation = MolecularDynamicsSimulation.from_options(deck_defaults())
        for i_name in REQUIRED_OPTIONS + OPTIONAL_OPTIONS:
            with self.subTest(option=i_name):
                self.assertTrue(
                    hasattr(simulation, i_name)
                    or i_name in ("dt", "type_initial_configuration")
                )
        # Two are stored under another name; the rest are attributes of the same name

    def test_a_missing_parameter_is_named(self):
        options = deck_defaults()
        del options["max_step"]
        with self.assertRaises(ValueError) as context:
            MolecularDynamicsSimulation.from_options(options)
        self.assertIn("max_step", str(context.exception))
        # It used to be a bare KeyError once the program had printed a line about it

    def test_the_thermostat_and_scheme_named_are_the_ones_built(self):
        options = deck_defaults()
        options.update({"thermostat": "isokinetic", "speed_up_scheme": "Naive"})
        simulation = MolecularDynamicsSimulation.from_options(options)
        self.assertIs(type(simulation.thermostat), IsokineticThermostat)
        self.assertIsInstance(simulation.speed_up_scheme, Naive)

    def test_a_misspelt_name_is_refused(self):
        for i_option in ("thermostat", "speed_up_scheme"):
            with self.subTest(option=i_option):
                with self.assertRaises(ValueError) as context:
                    MolecularDynamicsSimulation.from_options(
                        dict(deck_defaults(), **{i_option: "nope"})
                    )
                self.assertIn("nope", str(context.exception))


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


class TestStatus(unittest.TestCase):
    """Test class for whether a run reports the configuration it kept as legal."""

    def generate(self, vf, max_step):
        """Run a seeded generation of disks, giving the simulation."""
        microstructure = Microstructure.from_descriptors(
            [1.0, 1.0], {"0": {"phase_type": 1}, "1": {"phase_type": 2, "vf": vf, "n": 6}}
        )
        generator = MolecularDynamicsSimulation(
            0.0, max_step, 1, 1e-3, 0.0, "random", False, fixed_seed=3
        )
        generator.set_thermostat(
            MultiTemperatureIsokineticThermostat(
                None, criterion="ratio_in_out", max_ratio_osc=2, temp_low_ratio=1 / 4
            )
        )
        generator.set_speed_up_scheme(Naive())
        generator.generate_microstructure(microstructure)

        return generator

    def test_a_run_that_reaches_the_overlap_is_a_success(self):
        generator = self.generate(0.1, 200)
        self.assertLessEqual(generator.total_overlap, generator.max_residue + 1e-12)
        self.assertTrue(generator.status)

    def test_a_run_that_runs_out_of_steps_overlapping_is_a_failure(self):
        generator = self.generate(0.5, 2)
        self.assertGreater(generator.total_overlap, generator.max_residue)
        self.assertFalse(generator.status)

    def test_the_status_is_that_of_the_configuration_kept(self):
        generator = self.generate(0.5, 2)
        generator.status = True
        generator.max_residue = generator.total_overlap * 2
        generator.generate_microstructure(
            Microstructure.from_descriptors(
                [1.0, 1.0], {"0": {"phase_type": 1}, "1": {"phase_type": 2, "vf": 0.5, "n": 6}}
            )
        )
        self.assertEqual(
            generator.status, generator.total_overlap <= generator.max_residue + 1e-12
        )
        # Set once from the overlap the run ended with. It used to be set the first
        # time the overlap dipped under the tolerance and never unset, so a run that
        # was legal once and ran out of steps illegal reported success


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


class TestNonSquareRVE(unittest.TestCase):
    """Test class for a generation in an RVE longer in one direction than in another."""

    def generate(self, rve_dims):
        """Generate ten disks in an RVE of the given dimensions, with a fixed seed."""
        microstructure = Microstructure(list(rve_dims))
        microstructure.add_phase(Phase("0", {"phase_type": 1}))
        microstructure.add_phase(Phase("1", {"phase_type": 2, "r": 0.1, "n": 10}))
        generator = MolecularDynamicsSimulation(
            0.0, 400, 1, 1e-3, 0.0, "random", False, fixed_seed=3
        )
        generator.set_thermostat(
            MultiTemperatureIsokineticThermostat(
                None, criterion="ratio_in_out", max_ratio_osc=2, temp_low_ratio=1 / 4
            )
        )
        generator.set_speed_up_scheme(Naive())
        generator.generate_microstructure(microstructure)

        return microstructure, generator

    def test_the_particles_spread_over_the_whole_cell(self):
        for i_dims in ([2.0, 1.0], [1.0, 2.0]):
            with self.subTest(rve_dims=i_dims):
                microstructure, generator = self.generate(i_dims)
                self.assertTrue(generator.status)
                long_axis = int(np.argmax(i_dims))
                along = np.sort(
                    [i_particle.position_center[long_axis]
                     for i_particle in microstructure.particles]
                )
                gaps = np.diff(np.concatenate([along, [along[0] + i_dims[long_axis]]]))
                self.assertLess(gaps.max(), 1.0)
        # The cell is two long along one side, and the particles used to be folded into
        # a part of it one long: however they then fell, the gap between two neighbours
        # along that side, around the period, was at least one

    def test_the_box_keeps_its_proportions_while_it_is_normalised(self):
        generator = MolecularDynamicsSimulation(0.0, 1, 1, 1e-3, 0.0, "random", False)
        generator.box = [4.0, 2.0]
        disk = Disk("1", {"r": 0.2}, [4.0, 2.0])
        disk.position_center = np.array([3.0, 1.0])

        generator.resize_sim_box_and_all_particles_inside([disk], "unitary")
        self.assertEqual(generator.box, [2.0, 1.0])
        np.testing.assert_allclose(disk.position_center, [1.5, 0.5])
        self.assertAlmostEqual(disk.radius, 0.1)

        generator.resize_sim_box_and_all_particles_inside([disk], "original")
        self.assertEqual(generator.box, [4.0, 2.0])
        np.testing.assert_allclose(disk.position_center, [3.0, 1.0])
        self.assertAlmostEqual(disk.radius, 0.2)
        # The shortest side is brought to one and the particles with it, and the rest of
        # the box keeps its proportion to that side rather than becoming one as well


SCALE_FREE_PHASES = {
    "disks": ([2.0, 1.0], lambda s: {"phase_type": 2, "r": 0.1 * s, "n": 14}),
    "ellipses": (
        [1.0, 1.0],
        lambda s: {
            "phase_type": 3,
            "vf": 0.35,
            "n": 10,
            "ratio": 1.5,
            "angle_distribution": "normal",
            "angle_mean": 0.3,
            "angle_sigma": 0.5,
        },
    ),
    "spheres": ([1.0, 1.0, 1.0], lambda s: {"phase_type": 4, "r": 0.15 * s, "n": 8}),
    "cylinders": (
        [1.0, 1.0, 1.0],
        lambda s: {
            "phase_type": 7,
            "r_cyl": 0.06 * s,
            "length": 0.3 * s,
            "n": 6,
            "azimuth_angle": 0.3,
            "polar_angle": 0.7,
        },
    ),
    "fibres": (
        [1.0, 1.0, 1.0],
        lambda s: {"phase_type": 6, "r": 0.1 * s, "n": 10, "direction": 2},
    ),
}
# A microstructure of each particle type in an RVE multiplied by s, with its sizes
# multiplied by s too. The sizes of the ellipses come from their volume fraction, and
# so from the RVE


class TestScaleInvariance(unittest.TestCase):
    """Test class for a generation that is the same in any units of length."""

    def generate(self, rve_dims, phase, scale, max_step=300):
        """Generate a microstructure in which every length is multiplied by *scale*."""
        microstructure = Microstructure([i_dim * scale for i_dim in rve_dims])
        microstructure.add_phase(Phase("0", {"phase_type": 1}))
        microstructure.add_phase(Phase("1", phase(scale)))
        generator = MolecularDynamicsSimulation(
            1e-4 * scale,
            max_step,
            1,
            1e-3,
            0.01 * scale,
            "random",
            False,
            fixed_seed=11,
            final_overlap_check=True,
        )
        generator.set_thermostat(
            MultiTemperatureIsokineticThermostat(
                None, criterion="ratio_in_out", max_ratio_osc=2, temp_low_ratio=1 / 4
            )
        )
        generator.set_speed_up_scheme(CellList())
        generator.generate_microstructure(microstructure)

        return microstructure, generator

    def record(self, microstructure, generator, scale):
        """Give what a generation produced and reported, with its lengths divided."""
        return (
            [
                (
                    tuple(np.asarray(i_particle.position_center) / scale),
                    i_particle.radius / scale,
                    i_particle.delta / scale,
                )
                for i_particle in microstructure.particles
            ],
            [
                [tuple(np.asarray(j_position) / scale) for j_position in i_history]
                for i_history in generator.position_center_history
            ],
            generator.total_overlap / scale,
            generator.max_residue / scale,
            generator.status,
            generator.step,
        )

    def assert_scale_free(self, rve_dims, phase, max_step=300):
        """Check that a generation gives the same at scales a power of two apart."""
        reference = self.record(*self.generate(rve_dims, phase, 1.0, max_step), 1.0)
        for i_scale in (2.0**-20, 2.0**20):
            with self.subTest(scale=i_scale):
                self.assertEqual(
                    self.record(
                        *self.generate(rve_dims, phase, i_scale, max_step), i_scale
                    ),
                    reference,
                )
        # Powers of two are multiplied out exactly, so a generation that depends on
        # nothing but the ratios of its lengths gives the same to the last bit

        return reference

    def test_every_particle_type(self):
        for i_name, (i_rve_dims, i_phase) in SCALE_FREE_PHASES.items():
            with self.subTest(particles=i_name):
                reference = self.assert_scale_free(i_rve_dims, i_phase)
                self.assertTrue(reference[4])
        # With a minimum distance, which was kept in the user's units inside a box
        # normalised to one: at a millionth of the unit it vanished, and at a million
        # the particles, dilated past the box, could not be placed or crashed the cell
        # list. The fibres are weighed by their volume, whose length along the fibres
        # was left in the user's units. Every run converges, so the equality is not
        # that of runs that all failed the same way

    def test_an_overlap_left_is_given_in_the_users_units(self):
        _, phase = SCALE_FREE_PHASES["disks"]
        reference = self.assert_scale_free([1.0, 1.0], phase, max_step=3)
        self.assertGreater(reference[2], reference[3])
        self.assertFalse(reference[4])
        # Three steps leave the disks overlapping, so the overlap reported is not zero
        # at every scale. It was reported in the units of the normalised box, and the
        # naive check that measured it again measured it in the user's, against a
        # tolerance in the box's

    def test_the_minimum_distance_is_kept_in_the_users_units(self):
        rve_dims, phase = SCALE_FREE_PHASES["disks"]
        scale = 2.0**-10
        microstructure, generator = self.generate(rve_dims, phase, scale)
        self.assertTrue(generator.status)
        box = np.asarray(microstructure.rve_dims)
        particles = microstructure.particles
        gaps = [
            np.linalg.norm(
                i_particle.position_center
                - i_particle.nearest_periodic_image(
                    j_particle.position_center, i_particle.position_center, box
                )
            )
            - i_particle.radius
            - j_particle.radius
            for i_index, i_particle in enumerate(particles)
            for j_particle in particles[i_index + 1:]
        ]
        self.assertGreaterEqual(min(gaps), 0.01 * scale * (1 - 1e-9))

    def test_no_dilation_is_left_on_the_particles(self):
        microstructure = Microstructure([3.0, 3.0])
        microstructure.add_phase(Phase("0", {"phase_type": 1}))
        microstructure.add_phase(Phase("1", {"phase_type": 2, "r": 0.3, "n": 10}))
        generator = MolecularDynamicsSimulation(
            0.0, 300, 1, 1e-3, 0.03, "random", False, fixed_seed=11
        )
        generator.set_thermostat(
            MultiTemperatureIsokineticThermostat(
                None, criterion="ratio_in_out", max_ratio_osc=2, temp_low_ratio=1 / 4
            )
        )
        generator.set_speed_up_scheme(CellList())
        generator.generate_microstructure(microstructure)
        self.assertEqual(
            [i_particle.delta for i_particle in microstructure.particles], [0] * 10
        )
        for i_particle in microstructure.particles:
            self.assertAlmostEqual(i_particle.radius, 0.3, places=15)
        # The shortest side, three, is not a power of two, so the box does not scale
        # the particles exactly; the dilation is taken off in the box, as the box scaled
        # it, and none is left

    def test_a_deck_in_other_units_gives_the_same_microstructure(self):
        rve_dims, phase = SCALE_FREE_PHASES["disks"]
        reference = self.record(*self.generate(rve_dims, phase, 1.0), 1.0)
        for i_scale in (1e-3, 1e-6, 3.0):
            with self.subTest(scale=i_scale):
                record = self.record(*self.generate(rve_dims, phase, i_scale), i_scale)
                np.testing.assert_allclose(
                    [i_center for i_center, _, _ in record[0]],
                    [i_center for i_center, _, _ in reference[0]],
                    rtol=0,
                    atol=1e-12,
                )
                self.assertEqual(record[4:], reference[4:])
        # Scales that are not powers of two change the lengths in their last digit, and
        # nothing more: the box is brought to a shortest side of one in every case. It
        # was brought to the power of two nearest, which ran a deck in millimetres and
        # the same deck in micrometres in boxes of sides 1.95 and 1.05, whose dynamics
        # differ, and gave two microstructures

    def test_an_error_that_stops_the_run_is_the_one_raised(self):
        rve_dims, phase = SCALE_FREE_PHASES["disks"]
        with patch.object(
            MolecularDynamicsSimulation,
            "compute_forces",
            side_effect=RuntimeError("stopped"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                self.generate(rve_dims, phase, 1.0)
        # The run stopped before it measured an overlap, and the overlap it had not
        # measured was divided on the way out, raising a TypeError in place of this

    def test_the_path_ends_where_the_particle_is(self):
        rve_dims, phase = SCALE_FREE_PHASES["disks"]
        scale = 2.0**-10
        microstructure, generator = self.generate(rve_dims, phase, scale)
        for i_particle, i_history in zip(
            microstructure.particles, generator.position_center_history
        ):
            np.testing.assert_array_equal(i_history[-1], i_particle.position_center)
            self.assertTrue(
                all(
                    np.all((0 <= j_position) & (j_position < microstructure.rve_dims))
                    for j_position in i_history
                )
            )
        # The path is recorded in the normalised box, and was left there, before the
        # offset: the motion analysis drew it in a cell of the user's units, against
        # particles of the user's sizes


class TestFinalOverlapCheck(unittest.TestCase):
    """Test class for the naive check of the overlap a run ends with."""

    def test_it_measures_the_way_the_run_measured_last(self):
        generator = MolecularDynamicsSimulation(0.0, 1, 1, 1e-3, 0.0, "random", False)
        generator.thermostat = Mock(kin_energy_div=False)
        generator.last_distance_method = "dist_exact"
        generator.box = [1.0, 1.0]
        particles = [Mock(), Mock()]
        particles[0].intersection_length.return_value = (0.25, None)
        generator.check_overlap_naive(particles)
        self.assertEqual(generator.total_overlap, 0.25)
        particles[0].intersection_length.assert_called_once_with(
            particles[1], generator.box, dist_met="dist_exact"
        )
        # The thermostat has changed its mind since the last step, which measured the
        # overlap exactly; the check measured it the way the thermostat said next


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
