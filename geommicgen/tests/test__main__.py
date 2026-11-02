"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import sentinel, Mock, patch, call
import subprocess
import sys
import textwrap

import numpy as np

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
# A deck of the keyword format, which the program still reads

SEEDED_RUN = """
samples: 2
rve_dimensions: [1, 1]
seed: 3
phases:
  0: {type: matrix}
  1: {type: disk, vf: 0.2, n: 4}
generation:
  molecular_dynamics:
    max_residue_per_particle: 0
    max_step: 3
    speed_up: naive
"""
# The seeded deck, as an input file


class DeckRunTest(unittest.TestCase):
    """Base class for tests that run a deck in a process of its own."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    RUN = (
        "from geommicgen.app import run_program; import sys; run_program(sys.argv[1:])"
    )
    # What the process runs: the program, on the deck it is given

    def run_deck(self, text, extension=".yaml"):
        """Run a file in a process of its own, giving its status and its stderr."""
        deck_path = os.path.join(self.temp_dir.name, "deck" + extension)
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
            SEEDED_RUN.replace("samples: 2\n", "").replace(
                "rve_dimensions: [1, 1]\n", ""
            )
        )
        self.assertEqual(status, 2)
        self.assertIn("  - rve_dimensions: is missing", stderr)
        self.assertIn("  - samples: is missing", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "deck")))
        # Refused before anything is written. It used to be a KeyError on the first of
        # them, after a line saying that a mandatory parameter was missing without
        # saying which

    def test_a_deck_without_problem_type_runs(self):
        self.assertNotIn("Problem_Type", SEEDED_DECK)
        status, stderr, _ = self.run_deck(SEEDED_DECK, ".mdsim")
        self.assertEqual(status, 0, stderr)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "deck", "mic_1", "mic.json"))
        )
        # The keyword was required and read by nothing; it is still accepted

    def test_a_deck_with_problem_type_still_reads(self):
        status, stderr, _ = self.run_deck("Problem_Type 1\n" + SEEDED_DECK, ".mdsim")
        self.assertEqual(status, 0, stderr)

    def test_no_samples_is_refused(self):
        status, stderr, _ = self.run_deck(
            SEEDED_RUN.replace("samples: 2", "samples: 0")
        )
        self.assertEqual(status, 2)
        self.assertIn("samples: Input should be greater than or equal to 1", stderr)


class TestInputFile(DeckRunTest):
    """Test class for the run an input file asks for, and one a deck does."""

    def test_an_input_file_runs_as_the_deck_it_was_written_from(self):
        status, stderr, _ = self.run_deck(SEEDED_RUN)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        results_dir = os.path.join(self.temp_dir.name, "deck")
        from_file = self.particles(results_dir)
        os.rename(results_dir, results_dir + "_file")
        status, stderr, _ = self.run_deck(SEEDED_DECK, ".mdsim")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(self.particles(results_dir), from_file)

    @staticmethod
    def particles(results_dir):
        """Give the particles of each sample of a run of two."""
        particles = []
        for i_sample in range(2):
            path = os.path.join(results_dir, "mic_{0}".format(i_sample), "mic.json")
            with open(path) as mic_file:
                particles.append(json.load(mic_file)["particles"])

        return particles

    def test_the_run_keeps_the_file_and_each_sample_the_code_it_ran(self):
        status, stderr, _ = self.run_deck(SEEDED_RUN)
        self.assertEqual(status, 0, stderr)
        results_dir = os.path.join(self.temp_dir.name, "deck")
        with open(os.path.join(results_dir, "deck.yaml")) as copy:
            self.assertEqual(copy.read(), SEEDED_RUN)
        with open(os.path.join(results_dir, "mic_0", "mic.json")) as mic_file:
            provenance = json.load(mic_file)["provenance"]
        self.assertEqual(provenance["input_file"], "deck.yaml")
        for i_key in ("git_commit", "git_dirty", "libraries"):
            self.assertIn(i_key, provenance)
        # The input file, the commit of GMMD and the versions of the libraries are
        # the whole of the run, its defaults among it

    def test_a_deck_is_said_to_be_of_the_old_format(self):
        status, stderr, _ = self.run_deck(SEEDED_DECK, ".mdsim")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(
            stderr.splitlines(),
            [
                "geommicgen: deck.mdsim is a deck of the keyword format, which is read "
                "for now; geommicgen-convert-deck turns it into an input file."
            ],
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "deck", "deck.mdsim"))
        )

    def test_a_file_of_another_kind_is_refused_and_nothing_written(self):
        status, stderr, _ = self.run_deck(SEEDED_RUN, ".txt")
        self.assertEqual(status, 2)
        self.assertIn(
            "deck.txt is neither an input file (.yaml) nor a deck (.mdsim).", stderr
        )
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "deck")))
        # The extension used to be read as the method of generation, after the results
        # had been written

    def test_every_problem_is_said_before_anything_is_written(self):
        status, stderr, _ = self.run_deck(
            SEEDED_RUN.replace("max_step: 3", "max_steps: 3").replace(
                "n: 4", "n: 4, r: {distribution: normal, mean: 0.1, sigma: -1}"
            )
        )
        self.assertEqual(status, 2)
        self.assertIn("deck.yaml has 3 problems:", stderr)
        self.assertIn("  - generation.molecular_dynamics.max_step: is missing", stderr)
        self.assertIn(
            "  - generation.molecular_dynamics.max_steps: is not a key this section "
            "takes",
            stderr,
        )
        self.assertIn("  - phases.1.r.sigma: Input should be greater than", stderr)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "deck")))


class TestFailedSample(DeckRunTest):
    """Test class for a sample the run leaves overlapping."""

    OVERLAPPING_RUN = SEEDED_RUN.replace("vf: 0.2", "vf: 0.5").replace(
        "max_step: 3", "max_step: 2"
    ) + "meshes:\n  voxel: {n_voxels_dims: [8, 8]}\n"
    # Six disks at half the area in two steps stay overlapping, with a mesh asked for

    def test_it_is_written_but_not_meshed_and_the_run_fails(self):
        status, stderr, stdout = self.run_deck(self.OVERLAPPING_RUN)
        self.assertEqual(status, 1, stderr)
        self.assertNotIn("Traceback", stderr)
        sample_dir = os.path.join(self.temp_dir.name, "deck", "mic_0")
        self.assertTrue(os.path.exists(os.path.join(sample_dir, "mic.json")))
        self.assertFalse(os.path.exists(os.path.join(sample_dir, "meshes")))
        with open(os.path.join(sample_dir, "status")) as status_file:
            self.assertIn("Status: False", status_file.read())
        self.assertIn("2 of the samples asked for could not be generated", stdout)
        self.assertIn("mic_1: overlap", stdout)
        # Reported per sample and again in the summary, with the exit status of a
        # run that failed; it used to mesh the overlapping particles and exit 0

    def test_a_sample_that_converges_is_meshed_and_the_run_succeeds(self):
        status, stderr, stdout = self.run_deck(
            self.OVERLAPPING_RUN.replace("vf: 0.5", "vf: 0.1").replace(
                "max_step: 2", "max_step: 200"
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
            TestFailedSample.OVERLAPPING_RUN.replace("vf: 0.5", "vf: 0.1").replace(
                "max_step: 2", "max_step: 200"
            )
            + "analyses:\n"
            + "  final_config: true\n"
            + "  statistics: {descriptors: [nearest_neighbor]}\n"
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
        deck_path = os.path.join(self.temp_dir.name, name + ".yaml")
        with open(deck_path, "w") as deck:
            deck.write(SEEDED_RUN)
        run_program([deck_path])
        results_dir = os.path.join(self.temp_dir.name, name)
        documents = []
        for i_sample in range(2):
            with open(os.path.join(results_dir, "mic_{0}".format(i_sample), "mic.json")) as f:
                documents.append(json.load(f))

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
