"""
Unit tests regarding the reading of the input data file.

These cover the mesh options, whose sub keywords are read under the discretisation they
belong to and are assembled in part from what the solver writers declare.
"""

import os
import tempfile
import unittest

from geommicgen.iofuncs.keywords import (
    FORMAT_KEYWORDS,
    mesher_keywords,
    top_level_reader,
)
from geommicgen.meshing.mesher import get_mesher
from geommicgen.pipeline import DECK_MESHERS
from geommicgen.translators import writer_options


class MeshOptionsTest(unittest.TestCase):
    """Tests for the mesh options of an input data file."""

    def setUp(self):
        """Create a directory for the input data files."""
        self.deck_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.deck_dir.cleanup)

    def read(self, mesh_options):
        """Read an input data file with the mesh options given, and give them back."""
        deck_path = os.path.join(self.deck_dir.name, "deck.mdsim")
        with open(deck_path, "w") as deck:
            deck.write("Mesh_Options\n" + mesh_options)
        top_level_reader.read_input_file(deck_path)

        return top_level_reader.all_options["mesh_options"]

    def test_each_discretisation_keeps_its_own_options(self):
        options = self.read(
            "femsh\nmesh_size 0.1\nelement_type tri3\n"
            "rgmsh\nn_voxels_dims [16, 16]\nfile_name my_grid\n"
        )

        self.assertEqual(options["femsh"]["element_type"], "tri3")
        self.assertEqual(options["rgmsh"]["n_voxels_dims"], [[16, 16]])
        self.assertEqual(options["rgmsh"]["file_name"], "my_grid")

    def test_several_resolutions_are_read_as_one_list_each(self):
        options = self.read("rgmsh\nn_voxels_dims [16, 16] [32, 32]\n")

        self.assertEqual(options["rgmsh"]["n_voxels_dims"], [[16, 16], [32, 32]])

    def test_an_option_of_the_other_discretisation_is_refused(self):
        with self.assertRaises(ValueError) as context:
            self.read("femsh\nmesh_size 0.1\nelement_type tri3\nn_voxels_dims [4, 4]\n")

        self.assertIn("N_Voxels_Dims", str(context.exception))
        self.assertIn("femsh", str(context.exception))
        # It used to be read and then ignored, which is the same as not writing it

    def test_an_option_that_no_longer_exists_is_refused(self):
        with self.assertRaises(ValueError) as context:
            self.read("rgmsh\nn_voxels_dims [4, 4]\nslice_dir 2\n")

        self.assertIn("slice_dir", str(context.exception))
        # Slice_Dir decided whether the grid of a three dimensional microstructure was
        # written at all; the grid is now always written, and the keyword is gone

    def test_the_meshers_options_come_from_the_meshers(self):
        for i_header, i_mesher in DECK_MESHERS.items():
            declared = set(get_mesher(i_mesher).options)
            read_here = {i_keyword.name for i_keyword in mesher_keywords(i_mesher)}
            sub_keys = top_level_reader.all_keywords["Mesh_Options"].sub_keys[i_header]
            read_under = {i_keyword.name for i_keyword in sub_keys}

            self.assertTrue(declared)
            self.assertEqual(declared, read_here)
            self.assertTrue(declared <= read_under)
        # Whatever a mesher declares is read under the discretisation it produces,
        # without this module naming it

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
