"""
Integration test that exercises the real 2D ellipse example shipped in
geommicgen/resources/examples, from parsing the .mdsim input file all the way
through building real Phase/Microstructure objects and generating real
particles, without mocking the domain objects being tested.
"""

import os
import unittest

import numpy as np

# pylint: disable=import-error
from geommicgen.iofuncs.keywords import Keyword, top_level_reader
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase, FixedValue, NormalDistribution
from geommicgen.microstructure.particleclasses import Matrix, Ellipse

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEOMMICGEN_DIR = os.path.dirname(_TESTS_DIR)
ELLIPSE_EXAMPLE_PATH = os.path.join(
    _GEOMMICGEN_DIR, "resources", "examples", "2D_example_ellipses_vf_50_n_10.mdsim"
)


@unittest.skip("Integration test using a full example file; skipped from the regular run.")
class EllipseExampleTests(unittest.TestCase):
    " Integration tests using the real ellipse .mdsim example as input "

    def setUp(self):
        # top_level_reader is a process-wide singleton, and Keyword.input_reader
        # is a class attribute shared by every Keyword instance. Other test
        # modules (see test_keywords.py) construct their own throwaway
        # TopLevelReader() instances, which re-points Keyword.input_reader away
        # from the real, keyword-populated singleton. Point it back so this
        # test doesn't depend on module run order. all_options itself is left
        # alone: some keywords (e.g. save_min) only get their default value
        # stored once, at import time, and are never re-applied afterwards, so
        # clearing all_options here would permanently lose those defaults.
        Keyword.input_reader = top_level_reader

    def test_parses_the_example_file_into_the_expected_options(self):
        " The whole example file must be parsed into the expected option groups "
        top_level_reader.read_input_file(ELLIPSE_EXAMPLE_PATH)
        all_options = top_level_reader.all_options

        self.assertEqual(all_options["problem_type"], 1)
        self.assertEqual(all_options["n_dp_samples"], 1)
        self.assertFalse(all_options["save_min"])

        np.testing.assert_array_equal(
            all_options["mic_gen_parameters"]["rve_dimensions"], [1.0, 1.0]
        )
        self.assertEqual(all_options["mic_gen_parameters"]["max_step"], 150)
        self.assertEqual(all_options["mic_gen_parameters"]["speed_up_scheme"], "Verlet")
        self.assertEqual(all_options["mic_gen_parameters"]["verlet_factor"], 1.1)
        self.assertTrue(all_options["mic_gen_parameters"]["save_history"])
        # dt is not set in the example file, so its default must still apply.
        self.assertEqual(all_options["mic_gen_parameters"]["dt"], 0.05)

        self.assertEqual(
            all_options["mic_gen_descriptors"],
            {
                "0": {"phase_type": 1},
                "1": {
                    "phase_type": 3,
                    "vf": 0.5,
                    "n": 100,
                    "angle_distribution": "normal",
                    "angle_mean": 0.0,
                    "angle_sigma": 0.2,
                    "ratio": 1.2,
                },
            },
        )

        self.assertEqual(
            all_options["mesh_options"],
            {
                "femsh": {"mesh_size": 0.1, "element_type": "tri3"},
                "rgmsh": {"n_voxels_dims": [[1000, 1000]]},
            },
        )
        self.assertTrue(all_options["post_proc"]["final_config"])
        # motion_analysis is not set in the example file, so its default must
        # still apply.
        self.assertFalse(all_options["post_proc"]["motion_analysis"])

    def test_builds_matrix_and_ellipse_phases_from_the_parsed_descriptors(self):
        " The parsed descriptors must build a Matrix phase and a fully described Ellipse phase "
        top_level_reader.read_input_file(ELLIPSE_EXAMPLE_PATH)
        descriptors = top_level_reader.all_options["mic_gen_descriptors"]

        matrix_phase = Phase("0", descriptors["0"])
        ellipse_phase = Phase("1", descriptors["1"])

        self.assertIs(matrix_phase.type, Matrix)
        self.assertEqual(matrix_phase.descriptors, {})

        self.assertIs(ellipse_phase.type, Ellipse)
        self.assertEqual(
            set(ellipse_phase.descriptors.keys()), {"vf", "n", "angle", "ratio"}
        )

        self.assertIsInstance(ellipse_phase.descriptors["angle"], NormalDistribution)
        self.assertEqual(ellipse_phase.descriptors["angle"].mean, 0.0)
        self.assertEqual(ellipse_phase.descriptors["angle"].sigma, 0.2)

        self.assertIsInstance(ellipse_phase.descriptors["ratio"], FixedValue)
        self.assertEqual(ellipse_phase.descriptors["ratio"].value, 1.2)
        self.assertIsInstance(ellipse_phase.descriptors["n"], FixedValue)
        self.assertEqual(ellipse_phase.descriptors["n"].value, 100)
        self.assertIsInstance(ellipse_phase.descriptors["vf"], FixedValue)
        self.assertEqual(ellipse_phase.descriptors["vf"].value, 0.5)

    def test_assembles_a_microstructure_with_both_phases(self):
        " Adding both real phases to a real Microstructure must identify the matrix phase "
        top_level_reader.read_input_file(ELLIPSE_EXAMPLE_PATH)
        rve_dims = list(
            top_level_reader.all_options["mic_gen_parameters"]["rve_dimensions"]
        )
        descriptors = top_level_reader.all_options["mic_gen_descriptors"]

        sample = Microstructure(rve_dims)
        for phase_name, phase_descriptors in descriptors.items():
            sample.add_phase(Phase(phase_name, phase_descriptors))

        self.assertEqual(sample.matrix_phase, "0")
        self.assertEqual(set(sample.phases.keys()), {"0", "1"})

    def test_generates_the_requested_number_of_ellipse_particles(self):
        " generate_particles must produce exactly n=100 Ellipse particles matching the ratio "
        top_level_reader.read_input_file(ELLIPSE_EXAMPLE_PATH)
        rve_dims = list(
            top_level_reader.all_options["mic_gen_parameters"]["rve_dimensions"]
        )
        descriptors = top_level_reader.all_options["mic_gen_descriptors"]
        ellipse_phase = Phase("1", descriptors["1"])

        ellipse_phase.generate_particles(rve_dims)

        self.assertEqual(len(ellipse_phase.particles), 100)
        expected_ratio = descriptors["1"]["ratio"]
        for particle in ellipse_phase.particles:
            self.assertIsInstance(particle, Ellipse)
            self.assertAlmostEqual(
                particle.major_axis / particle.minor_axis, expected_ratio
            )
            self.assertTrue(np.isfinite(particle.angle))
