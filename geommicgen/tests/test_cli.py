import argparse
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.cli import (
    add_declared_arguments,
    analyze_command,
    declared_options,
    mesh_command,
    translate_command,
)
from geommicgen.postproc.options import ANALYSES, ANALYSIS_OPTIONS, with_defaults
from geommicgen.iofuncs.keywords import top_level_reader
from geommicgen.iofuncs.md_state import save_md_state
from geommicgen.iofuncs.microstructure_yaml import write_microstructure_yaml
from geommicgen.tests.helpers import a_generation_run, disk_microstructure


class TestMeshCommand(unittest.TestCase):
    """Test class for the command that discretises a microstructure."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.microstructure_path = os.path.join(self.temp_dir.name, "mic.yaml")
        write_microstructure_yaml(disk_microstructure(), self.microstructure_path)
        self.output_dir = os.path.join(self.temp_dir.name, "out")

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_command(self, arguments):
        """Run the meshing command, returning its status and what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = mesh_command(arguments)

        return status, printed.getvalue()

    def written(self):
        """Names of the files the command wrote."""
        return sorted(os.listdir(self.output_dir))

    def test_a_grid_without_gmsh(self):
        status, _ = self.run_command(
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "16", "16",
             "--to", "crate", "-o", self.output_dir]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            self.written(),
            [
                "mic_16_16.mesh.json",
                "mic_16_16.rgmsh.npy",
                "mic_16_16.vti",
                "mic_16_16_example.dat",
            ],
        )
        # The whole path from a microstructure to what a spectral solver reads, with no
        # geometry kernel installed

    def test_the_mesh_alone_when_no_format_is_asked_for(self):
        status, _ = self.run_command(
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8", "8",
             "-o", self.output_dir]
        )
        self.assertEqual(status, 0)
        self.assertEqual(self.written(), ["mic_8_8.mesh.json", "mic_8_8.vti"])

    def test_the_files_can_be_named(self):
        self.run_command(
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8", "8",
             "--name", "chosen", "-o", self.output_dir]
        )
        self.assertEqual(self.written(), ["chosen.mesh.json", "chosen.vti"])

    def test_a_grid_needs_its_number_of_voxels(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                mesh_command([self.microstructure_path, "--mesher", "voxel"])

    def test_an_unknown_format_is_refused_before_anything_runs(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                mesh_command(
                    [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8",
                     "8", "--to", "linkss", "-o", self.output_dir]
                )
        self.assertFalse(os.path.exists(self.output_dir))

    def test_a_failure_is_reported_and_gives_a_status(self):
        status, printed = self.run_command(
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8", "8", "8",
             "-o", self.output_dir]
        )
        self.assertEqual(status, 1)
        self.assertIn("ValueError", printed)
        # A grid of three directions cannot discretise a microstructure of two

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_two_elements_into_one_directory_do_not_collide(self):
        for i_element in ("tri3", "tri6"):
            status, _ = self.run_command(
                [self.microstructure_path, "--mesh-size", "0.2",
                 "--element-type", i_element, "-o", self.output_dir]
            )
            self.assertEqual(status, 0)
        self.assertEqual(
            self.written(),
            ["mic_tri3.mesh.json", "mic_tri3.vtu", "mic_tri6.mesh.json", "mic_tri6.vtu"],
        )
        # Both were called mic.vtu, so the second was written over the first without
        # a word; a deck has always named them after the label of the mesher

    def test_a_finite_element_mesh(self):
        status, _ = self.run_command(
            [self.microstructure_path, "--mesher", "gmsh", "--mesh-size", "0.15",
             "--element-type", "tri3", "--to", "links", "-o", self.output_dir]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            self.written(),
            [
                "mic_tri3.mesh",
                "mic_tri3.mesh.json",
                "mic_tri3.vtu",
                "mic_tri3_example.rve",
            ],
        )
        # Named after the microstructure and the element, so that meshing it again
        # with another element does not write over this


class TestTranslateCommand(unittest.TestCase):
    """Test class for the command that writes a mesh in the formats solvers read."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        microstructure_path = os.path.join(self.temp_dir.name, "mic.yaml")
        write_microstructure_yaml(disk_microstructure(), microstructure_path)
        self.staged = os.path.join(self.temp_dir.name, "staged")
        with contextlib.redirect_stdout(io.StringIO()):
            mesh_command(
                [microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "16", "16",
                 "-o", self.staged]
            )
        self.output_dir = os.path.join(self.temp_dir.name, "out")
        # Only the second stage is run, so the third one has nothing but the file

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_the_third_stage_runs_off_the_second_stage_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status = translate_command(
                [os.path.join(self.staged, "mic_16_16.vti"), "--to", "crate",
                 "-o", self.output_dir]
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            sorted(os.listdir(self.output_dir)),
            ["mic_16_16.rgmsh.npy", "mic_16_16_example.dat"],
        )
        grid = np.load(os.path.join(self.output_dir, "mic_16_16.rgmsh.npy"))
        self.assertEqual(grid.shape, (16, 16))
        # An image is read back as the grid it is, so the writer that needs a grid
        # accepts it

    def test_the_formats_can_be_listed(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = translate_command(["--list-formats"])
        self.assertEqual(status, 0)
        self.assertIn("links", printed.getvalue())
        self.assertIn(".rgmsh.npy", printed.getvalue())

    def test_a_mesh_and_a_format_are_both_needed(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                translate_command([os.path.join(self.staged, "mic_16_16.vti")])


class TestAnalyzeCommand(unittest.TestCase):
    """Test class for the command that analyses a microstructure from its file."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_dir = os.path.join(self.temp_dir.name, "sample")
        os.makedirs(self.sample_dir)
        self.microstructure = disk_microstructure()
        self.microstructure_path = os.path.join(self.sample_dir, "mic.yaml")
        write_microstructure_yaml(self.microstructure, self.microstructure_path)
        self.output_dir = os.path.join(self.temp_dir.name, "out")

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_command(self, *arguments):
        """Run the analysis command, returning its status and what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = analyze_command(
                [self.microstructure_path, *arguments, "-o", self.output_dir]
            )

        return status, printed.getvalue()

    def written(self, *names):
        """Say whether a file or directory was written to the output directory."""
        return os.path.exists(os.path.join(self.output_dir, *names))

    def save_state(self, run):
        """Write the state of a run beside the microstructure."""
        save_md_state(self.sample_dir, run)

    def test_a_statistical_analysis_without_gmsh(self):
        status, _ = self.run_command("--stat-nearest-neighbor")
        self.assertEqual(status, 0)
        self.assertTrue(self.written("stat_analysis_results", "stat_results.npz"))
        self.assertFalse(self.written("mic.screen"))
        self.assertFalse(os.path.exists("mic.screen"))
        # The command reports to the terminal alone, as the other commands do; a
        # screen file is a run's

    def test_the_final_configuration(self):
        status, printed = self.run_command("--final-config")
        self.assertEqual(status, 0)
        self.assertTrue(self.written("final_config.pdf"))
        self.assertIn("final_config.pdf", printed)

    def test_nothing_asked_for_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_command()
        self.assertFalse(os.path.exists(self.output_dir))

    def test_the_motion_analysis_needs_the_state(self):
        status, printed = self.run_command("--final-config", "--motion-analysis")
        self.assertEqual(status, 1)
        self.assertIn("ValueError", printed)
        self.assertIn("md_state.npz", printed)
        self.assertFalse(self.written("final_config.pdf"))
        # Refused before anything is written, the final configuration included

    def test_running_twice_writes_over_the_first_time(self):
        first, _ = self.run_command("--stat-nearest-neighbor")
        second, _ = self.run_command("--stat-nearest-neighbor")
        self.assertEqual((first, second), (0, 0))

    def test_a_run_that_recorded_no_positions(self):
        run = a_generation_run()
        run.position_center_history = None
        self.save_state(run)
        status, printed = self.run_command("--motion-analysis")
        self.assertEqual(status, 0)
        self.assertTrue(self.written("motion_results", "relative_energy.pdf"))
        self.assertFalse(self.written("motion_results", "paths"))
        self.assertIn("recorded no positions", printed)
        # The whole motion analysis, with no geometry kernel installed

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_the_motion_analysis(self):
        run = a_generation_run()
        run.position_center_history = [
            [i_center - 0.2, i_center - 0.1, i_center.copy()]
            for i_center in (i.position_center for i in self.microstructure.particles)
        ]
        self.save_state(run)
        status, _ = self.run_command("--motion-analysis")
        self.assertEqual(status, 0)
        self.assertTrue(os.listdir(os.path.join(self.output_dir, "motion_results", "paths")))


class TestCommandReports(unittest.TestCase):
    """Test class for a command opening and closing as a run of a deck does."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.microstructure_path = os.path.join(self.temp_dir.name, "mic.yaml")
        write_microstructure_yaml(disk_microstructure(), self.microstructure_path)
        self.output_dir = os.path.join(self.temp_dir.name, "out")

    def run_command(self, command, argv):
        """Run a command and give what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = command(argv)
        self.assertEqual(status, 0, printed.getvalue())

        return printed.getvalue()

    def assert_framed(self, printed, label):
        """Check the heading, the file worked on and the table of times are there."""
        self.assertIn("Geometrical microstructure generation", printed)
        self.assertIn("{0}: ".format(label), printed)
        self.assertIn("Starting program execution at", printed)
        self.assertIn("Execution times:", printed)
        self.assertIn("Program Completed", printed)

    def test_the_meshing_command_is_framed(self):
        printed = self.run_command(
            mesh_command,
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8", "8",
             "-o", self.output_dir],
        )
        self.assert_framed(printed, "Microstructure")
        self.assertIn("Regular mesh generation", printed)

    def test_the_analysis_command_is_framed_and_lists_what_it_wrote(self):
        printed = self.run_command(
            analyze_command,
            [self.microstructure_path, "--stat-nearest-neighbor", "-o", self.output_dir],
        )
        self.assert_framed(printed, "Microstructure")
        self.assertIn("Statistical analysis", printed)
        self.assertIn(os.path.join("stat_analysis_results", "stat_results.npz"), printed)
        self.assertIn(
            os.path.join("stat_analysis_results", "nearest_neighbor_dist.pdf"), printed
        )
        # It reported no file at all, the analyses not saying what they write

    def test_the_translating_command_is_framed(self):
        with contextlib.redirect_stdout(io.StringIO()):
            mesh_command(
                [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims",
                 "8", "8", "-o", self.output_dir]
            )
        printed = self.run_command(
            translate_command,
            [os.path.join(self.output_dir, "mic_8_8.vti"), "--to", "crate",
             "-o", os.path.join(self.temp_dir.name, "translated")],
        )
        self.assert_framed(printed, "Mesh")


class TestAnalysisOptions(unittest.TestCase):
    """Test class for the analyses being asked for the same way from a deck and here."""

    def test_the_deck_reads_the_analyses_declared(self):
        keywords = {
            i_keyword.name.lower(): i_keyword
            for i_keyword in top_level_reader.top_level_keywords
            if getattr(i_keyword, "keyword_group", None) == "post_proc"
        }
        self.assertEqual(set(keywords), set(ANALYSIS_OPTIONS))
        for i_name, i_description in ANALYSIS_OPTIONS.items():
            with self.subTest(option=i_name):
                self.assertEqual(keywords[i_name].default_value, i_description["default"])
                self.assertEqual(keywords[i_name].type_str, i_description["type"])
        # Built from the one declaration, so this is a test of the wiring rather than
        # of two lists kept alike by hand, which is what it used to be

    def test_a_bare_command_asks_for_what_a_bare_deck_asks_for(self):
        parser = argparse.ArgumentParser()
        add_declared_arguments(parser, ANALYSIS_OPTIONS)
        options = with_defaults(declared_options(parser.parse_args([]), ANALYSIS_OPTIONS))
        self.assertEqual(
            options,
            {i_name: i_description["default"] for i_name, i_description in ANALYSIS_OPTIONS.items()},
        )
        self.assertFalse(any(options[i_name] for i_name in ANALYSES))


if __name__ == "__main__":
    unittest.main()
