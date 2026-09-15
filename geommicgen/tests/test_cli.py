import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.cli import mesh_command, translate_command
from geommicgen.iofuncs.microstructure_yaml import write_microstructure_yaml
from geommicgen.tests.helpers import disk_microstructure


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


if __name__ == "__main__":
    unittest.main()
