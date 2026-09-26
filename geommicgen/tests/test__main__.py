"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import sentinel, Mock, patch, call
import subprocess
import sys
import textwrap

import numpy as np
import yaml

from geommicgen.app import run_program


class TestMainFromCommandLine(unittest.TestCase):
    """Class for the unit tests regarding the __main__ of the geommicgen package."""

    def run_with(self, argv):
        """Run the program with these arguments, giving what it printed to stderr."""
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as context:
                run_program(argv)
        self.assertEqual(context.exception.code, 2)

        return stderr.getvalue()

    def test_no_arguments(self):
        self.assertIn("usage: geommicgen", self.run_with([]))
        # A usage line and status 2, as the other commands give, where it used to be
        # a traceback

    def test_too_many_arguments(self):
        self.assertIn("unrecognized arguments", self.run_with(["a.mdsim", "b.yaml"]))
        # A microstructure generated earlier used to be a second argument; it is meshed
        # and analysed by the commands made for that

    def test_help_is_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as context:
                run_program(["--help"])
        self.assertEqual(context.exception.code, 0)
        self.assertIn("input_file", stdout.getvalue())
        # It used to be opened as the input data file, and to fail for not existing

    # @patch("microstructure.particleclasses.Disk")
    # @patch("microstructure.phase.FixedValue")
    # def test_generate_particles_number(self, mock_fixed_value, mock_disk):
    #     gen_method = MicGenTest()
    #     rve_dims = [1.0, 1.0]
    #     descriptors = {
    #         "n": mock_fixed_value("n", 10),
    #         "vf": mock_fixed_value("vf", 0.1),
    #     }
    #     phase = sentinel.phase
    #     particles = gen_method.generate_particles(
    #         rve_dims, mock_disk, phase, **descriptors
    #     )
    #     self.assertTrue(len(particles), 10)
    #     for particle in particles:
    #         self.assertTrue(particle.name, "Disk()")

SEEDED_DECK = """
N_DP_Samples 2
RVE_Dimensions [1, 1]
Fixed_Seed 3
Mic_Gen_Descriptors
Phase 0
Phase_Type 1
Phase 1
Phase_Type 2
vf 0.2
n 4
Max_Residue_Per_Particle 0
Max_Step 3
Speed_Up_Scheme Naive
Save_History False
"""


class DeckRunTest(unittest.TestCase):
    """Base class for tests that run a deck in a process of its own."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    RUN = (
        "from geommicgen.app import run_program; import sys; run_program(sys.argv[1:])"
    )
    # What the process runs: the program, on the deck it is given

    def run_deck(self, text):
        """Run a deck in a process of its own, giving its status and its stderr."""
        deck_path = os.path.join(self.temp_dir.name, "deck.mdsim")
        with open(deck_path, "w") as deck:
            deck.write(text)
        completed = subprocess.run(
            [sys.executable, "-c", self.RUN, deck_path],
            capture_output=True, text=True, check=False, cwd=self.temp_dir.name,
        )
        # In a process of its own because the reader keeps what an earlier deck gave
        # for a keyword this one leaves out, and there is one reader per process

        return completed.returncode, completed.stderr, completed.stdout


class TestMandatoryKeywords(DeckRunTest):
    """Test class for what a deck has to give before anything is generated."""

    def test_the_missing_keywords_are_named_together(self):
        status, stderr, _ = self.run_deck(
            SEEDED_DECK.replace("N_DP_Samples 2\n", "").replace(
                "RVE_Dimensions [1, 1]\n", ""
            )
        )
        self.assertNotEqual(status, 0)
        self.assertIn("ValueError", stderr)
        self.assertIn("N_DP_Samples, RVE_Dimensions", stderr)
        self.assertNotIn("KeyError", stderr)
        # It used to be a KeyError on the first of them, after a line saying that a
        # mandatory parameter was missing without saying which

    def test_a_deck_without_problem_type_runs(self):
        self.assertNotIn("Problem_Type", SEEDED_DECK)
        status, stderr, _ = self.run_deck(SEEDED_DECK)
        self.assertEqual(status, 0, stderr)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "deck", "mic_1", "mic.yaml"))
        )
        # The keyword was required and read by nothing; it is still accepted

    def test_a_deck_with_problem_type_still_reads(self):
        status, stderr, _ = self.run_deck("Problem_Type 1\n" + SEEDED_DECK)
        self.assertEqual(status, 0, stderr)

    def test_no_samples_is_refused(self):
        status, stderr, _ = self.run_deck(SEEDED_DECK.replace("N_DP_Samples 2", "N_DP_Samples 0"))
        self.assertNotEqual(status, 0)
        self.assertIn("positive integer", stderr)


class TestFailedSample(DeckRunTest):
    """Test class for a sample the run leaves overlapping."""

    OVERLAPPING_DECK = SEEDED_DECK.replace("vf 0.2", "vf 0.5").replace(
        "Max_Step 3", "Max_Step 2"
    ) + "Mesh_Options\nvoxel\nn_voxels_dims [8, 8]\n"
    # Six disks at half the area in two steps stay overlapping, with a mesh asked for

    def test_it_is_written_but_not_meshed_and_the_run_fails(self):
        status, stderr, stdout = self.run_deck(self.OVERLAPPING_DECK)
        self.assertEqual(status, 1, stderr)
        self.assertNotIn("Traceback", stderr)
        sample_dir = os.path.join(self.temp_dir.name, "deck", "mic_0")
        self.assertTrue(os.path.exists(os.path.join(sample_dir, "mic.yaml")))
        self.assertFalse(os.path.exists(os.path.join(sample_dir, "meshes")))
        with open(os.path.join(sample_dir, "status")) as status_file:
            self.assertIn("Status: False", status_file.read())
        self.assertIn("2 of the samples asked for could not be generated", stdout)
        self.assertIn("mic_1: overlap", stdout)
        # Reported per sample and again in the summary, with the exit status of a
        # run that failed; it used to mesh the overlapping particles and exit 0

    def test_a_sample_that_converges_is_meshed_and_the_run_succeeds(self):
        status, stderr, stdout = self.run_deck(
            self.OVERLAPPING_DECK.replace("vf 0.5", "vf 0.1").replace(
                "Max_Step 2", "Max_Step 200"
            )
        )
        self.assertEqual(status, 0, stderr)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "deck", "mic_0", "meshes"))
        )
        self.assertNotIn("could not be generated", stdout)


class TestFailedAnalysis(DeckRunTest):
    """Test class for an analysis that fails in one sample of a run."""

    RUN = textwrap.dedent(
        """
        import sys
        import geommicgen.postproc.postproc as postproc
        from geommicgen.app import run_program

        drawn = []

        def failing_first(*args):
            drawn.append(args)
            if len(drawn) == 1:
                raise RuntimeError("drawn wrong")
            return plot_particles(*args)

        plot_particles = postproc.plot_particles
        postproc.plot_particles = failing_first
        run_program(sys.argv[1:])
        """
    )
    # The program, with the final configuration of its first sample failing to be drawn

    def test_the_run_goes_on_and_fails(self):
        status, stderr, stdout = self.run_deck(
            TestFailedSample.OVERLAPPING_DECK.replace("vf 0.5", "vf 0.1").replace(
                "Max_Step 2", "Max_Step 200"
            )
            + "final_config True\nstat_nearest_neighbor True\n"
        )
        self.assertEqual(status, 1, stderr)
        self.assertNotIn("Traceback", stderr)
        results_dir = os.path.join(self.temp_dir.name, "deck")
        self.assertTrue(
            os.path.exists(
                os.path.join(results_dir, "mic_0", "stat_analysis_results")
            )
        )
        self.assertFalse(
            os.path.exists(os.path.join(results_dir, "mic_0", "final_config.pdf"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(results_dir, "mic_1", "final_config.pdf"))
        )
        self.assertIn("1 of the analyses asked for could not be carried out", stdout)
        self.assertIn("mic_0: Generating final configuration", stdout)
        # The statistics of the sample, and the whole of the next sample, are still
        # carried out, and the run fails at the end, as it does for a mesh


class TestFixedSeedAcrossSamples(unittest.TestCase):
    """Test class for a seeded run of several samples."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def run_deck(self, name):
        """Run the seeded deck under a name of its own, returning its microstructures."""
        deck_path = os.path.join(self.temp_dir.name, name + ".mdsim")
        with open(deck_path, "w") as deck:
            deck.write(SEEDED_DECK)
        run_program([deck_path])
        results_dir = os.path.join(self.temp_dir.name, name)
        documents = []
        for i_sample in range(2):
            with open(os.path.join(results_dir, "mic_{0}".format(i_sample), "mic.yaml")) as f:
                documents.append(yaml.safe_load(f))

        return documents

    def test_each_sample_has_a_seed_of_its_own(self):
        first, second = self.run_deck("a")
        self.assertEqual(first["provenance"]["fixed_seed"], 3)
        self.assertEqual(second["provenance"]["fixed_seed"], 4)
        self.assertNotEqual(first["particles"], second["particles"])
        # One seed for every sample made every sample the same microstructure

    def test_the_samples_are_the_same_in_another_run(self):
        first_run = self.run_deck("a")
        second_run = self.run_deck("b")
        for i_first, i_second in zip(first_run, second_run):
            self.assertEqual(i_first["particles"], i_second["particles"])
