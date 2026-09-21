import argparse
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.cli import (
    ANALYSIS_OPTIONS,
    add_declared_arguments,
    analysis_options,
    analyze_command,
    mesh_command,
    translate_command,
)
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
            self.written(), ["mic.mesh.json", "mic.rgmsh.npy", "mic.vti"]
        )
        # The whole path from a microstructure to what a spectral solver reads, with no
        # geometry kernel installed

    def test_the_mesh_alone_when_no_format_is_asked_for(self):
        status, _ = self.run_command(
            [self.microstructure_path, "--mesher", "voxel", "--n-voxels-dims", "8", "8",
             "-o", self.output_dir]
        )
        self.assertEqual(status, 0)
        self.assertEqual(self.written(), ["mic.mesh.json", "mic.vti"])

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
    def test_a_finite_element_mesh(self):
        status, _ = self.run_command(
            [self.microstructure_path, "--mesher", "gmsh", "--mesh-size", "0.15",
             "--element-type", "tri3", "--to", "links", "-o", self.output_dir]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            self.written(),
            ["mic.mesh", "mic.mesh.json", "mic.vtu", "mic_example.rve"],
        )


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
                [os.path.join(self.staged, "mic.vti"), "--to", "crate",
                 "-o", self.output_dir]
            )
        self.assertEqual(status, 0)
        self.assertEqual(os.listdir(self.output_dir), ["mic.rgmsh.npy"])
        grid = np.load(os.path.join(self.output_dir, "mic.rgmsh.npy"))
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
                translate_command([os.path.join(self.staged, "mic.vti")])


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
        self.assertTrue(self.written("stat_analysis_results", "stat_results.stat"))
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


class TestAnalysisOptions(unittest.TestCase):
    """
    Test class holding the command's options to the input data file's.

    An analysis is asked for by the same name with the same default from either, so
    that the two cannot drift apart.
    """

    def setUp(self):
        self.keywords = [
            i_keyword
            for i_keyword in top_level_reader.top_level_keywords
            if getattr(i_keyword, "keyword_group", None) == "post_proc"
        ]
        self.assertTrue(self.keywords)

    def deck_defaults(self):
        """The post processing options of a deck that names none of them."""
        return {
            i_keyword.name.lower(): i_keyword.default_value
            for i_keyword in self.keywords
        }

    def test_the_options_are_the_keywords_of_the_input_file(self):
        self.assertEqual(
            set(ANALYSIS_OPTIONS), {i_keyword.name.lower() for i_keyword in self.keywords}
        )

    def test_the_defaults_are_the_input_files(self):
        self.assertEqual(
            {i_name: i_description["default"] for i_name, i_description in ANALYSIS_OPTIONS.items()},
            self.deck_defaults(),
        )

    def test_a_bare_command_asks_for_what_a_bare_deck_asks_for(self):
        parser = argparse.ArgumentParser()
        add_declared_arguments(parser, ANALYSIS_OPTIONS)
        self.assertEqual(analysis_options(parser.parse_args([])), self.deck_defaults())
        # The keyword objects carry their defaults; the reader's option dictionary is
        # left alone, since other tests change it by reading decks


if __name__ == "__main__":
    unittest.main()
