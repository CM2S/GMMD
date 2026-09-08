"""
Unit tests regarding the reading of the input data file.

These cover the mesh options, whose sub keywords are read under the discretisation they
belong to and are assembled in part from what the solver writers declare.
"""

import os
import shutil
import tempfile
import unittest

from geommicgen.iofuncs.keywords import FORMAT_KEYWORDS, top_level_reader
from geommicgen.translators import writer_options

DECK_HEAD = """Problem_Type 1
N_DP_Samples 1
RVE_Dimensions [1, 1]

Mic_Gen_Descriptors
Phase 0
Phase_Type 1
Phase 1
Phase_Type 2
vf 0.3
n 4

Max_Residue_Per_Particle 0
Max_Step 10
"""


class MeshOptionsTest(unittest.TestCase):
    """Tests for the mesh options of an input data file."""

    def setUp(self):
        """Create a directory for the input data files."""
        self.deck_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.deck_dir)

    def read(self, mesh_options):
        """Read an input data file with the mesh options given, and give them back."""
        deck_path = os.path.join(self.deck_dir, "deck.mdsim")
        with open(deck_path, "w") as deck:
            deck.write(DECK_HEAD + "\nMesh_Options\n" + mesh_options)
        top_level_reader.read_input_file(deck_path)

        return top_level_reader.all_options["mesh_options"]

    def test_each_discretisation_keeps_its_own_options(self):
        options = self.read(
            "femsh\nmesh_size 0.1\nelement_type tri3\n"
            "rgmsh\nn_voxels_dims [16, 16]\nvoxel_filename my_grid\n"
        )

        self.assertEqual(options["femsh"]["element_type"], "tri3")
        self.assertEqual(options["rgmsh"]["voxel_filename"], "my_grid")

    def test_an_option_of_the_other_discretisation_is_refused(self):
        with self.assertRaises(ValueError) as context:
            self.read("femsh\nmesh_size 0.1\nelement_type tri3\nvoxel_filename mine\n")

        self.assertIn("Voxel_Filename", str(context.exception))
        self.assertIn("femsh", str(context.exception))
        # It used to be read and then ignored, which is the same as not writing it

    def test_an_option_of_a_format_is_read_under_either(self):
        options = self.read(
            "femsh\nmesh_size 0.1\nelement_type tri3\ngauss_points 6\n"
            "rgmsh\nn_voxels_dims [16, 16]\ngauss_points 4\n"
        )

        self.assertEqual(options["femsh"]["gauss_points"], 6)
        self.assertEqual(options["rgmsh"]["gauss_points"], 4)
        # A grid is written to a solver as readily as a mesh is, so what belongs to the
        # format is read under both

    def test_the_formats_options_come_from_the_formats(self):
        declared = set(writer_options())
        read_here = {i_keyword.name for i_keyword in FORMAT_KEYWORDS}

        self.assertTrue(declared)
        self.assertTrue(declared <= read_here)
        # Whatever a writer declares is read from the input data file without this
        # module naming it


if __name__ == "__main__":
    unittest.main()
