"""
Tests running the example input data files shipped with the repository.

Each example is run as it is written, with the numbers that set its size brought down
so that it converges and meshes in seconds: the samples, the particles, the steps and
the resolution of the meshes. What the example asks for is what is checked for.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

import geommicgen
from geommicgen._optional import has_gmsh

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(geommicgen.__file__)), "examples")


def reduced(text, **values):
    """Give a deck with the value of each named keyword replaced."""
    for i_keyword, i_value in values.items():
        text, count = re.subn(
            r"^{0} .*$".format(i_keyword), "{0} {1}".format(i_keyword, i_value), text, flags=re.M
        )
        assert count == 1, i_keyword
    return text


@unittest.skipUnless(os.path.isdir(EXAMPLES), "not run from the repository")
@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestExamples(unittest.TestCase):
    """Test class running the examples, reduced to a size that runs in seconds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def run_example(self, name, **values):
        """Run a shipped example with some of its values replaced, giving its results."""
        with open(os.path.join(EXAMPLES, name + ".mdsim")) as example:
            text = reduced(example.read(), **values)
        deck_path = os.path.join(self.temp_dir.name, "example.mdsim")
        with open(deck_path, "w") as deck:
            deck.write(text)
        completed = subprocess.run(
            [sys.executable, "-c", "from geommicgen.app import run_program; import sys; "
             "run_program(sys.argv[1:])", deck_path],
            capture_output=True, text=True, check=False, cwd=self.temp_dir.name,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

        return os.path.join(self.temp_dir.name, "example", "mic_0")

    def test_the_two_dimensional_example(self):
        sample = self.run_example(
            "2D_example_ellipses_vf_50_n_10", n=10, vf=0.2, mesh_size=0.2,
            n_voxels_dims="[32, 32]",
        )
        self.assertTrue(os.path.exists(os.path.join(sample, "final_config.pdf")))
        meshes = os.listdir(os.path.join(sample, "meshes"))
        self.assertIn("example_tri3.vtu", meshes)
        self.assertIn("example_32_32.vti", meshes)
        # A final configuration, a finite element mesh and a grid, as the example asks

    def test_the_three_dimensional_example(self):
        sample = self.run_example(
            "3D_example_ellipsoids_vf_10_r_1_7", N_DP_Samples=1, vf=0.03, Max_Step=300,
            n_voxels_dims="[16, 16, 16]",
        )
        self.assertTrue(os.path.exists(os.path.join(sample, "final_config.vtk")))
        self.assertTrue(
            os.path.exists(os.path.join(sample, "motion_results", "relative_energy.pdf"))
        )
        self.assertIn("example_16_16_16.vti", os.listdir(os.path.join(sample, "meshes")))
        # A three dimensional final configuration, the motion analysis and a grid


if __name__ == "__main__":
    unittest.main()
