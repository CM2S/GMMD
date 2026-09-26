"""
Unit tests regarding the reading of a deck of the keyword format.

These cover the mesh options, whose sub keywords are read under the discretisation they
belong to and are assembled in part from what the solver writers declare, and the
reader itself, which reads each deck anew.
"""

import os
import tempfile
import unittest

from geommicgen.iofuncs.legacy_deck import (
    LEGACY_MESHERS,
    build_reader,
    format_keywords,
    mesher_keywords,
    read_legacy_deck,
)
from geommicgen.meshing.mesher import available_meshers, get_mesher
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

        return read_legacy_deck(deck_path)["mesh_options"]

    def test_each_discretisation_keeps_its_own_options(self):
        options = self.read(
            "gmsh\nmesh_size 0.1\nelement_type tri3\n"
            "voxel\nn_voxels_dims [16, 16]\nfile_name my_grid\n"
        )

        self.assertEqual(options["gmsh"]["element_type"], "tri3")
        self.assertEqual(options["voxel"]["n_voxels_dims"], [[16, 16]])
        self.assertEqual(options["voxel"]["file_name"], "my_grid")

    def test_several_resolutions_are_read_as_one_list_each(self):
        options = self.read("voxel\nn_voxels_dims [16, 16] [32, 32]\n")

        self.assertEqual(options["voxel"]["n_voxels_dims"], [[16, 16], [32, 32]])

    def test_an_option_of_the_other_discretisation_is_refused(self):
        with self.assertRaises(ValueError) as context:
            self.read("gmsh\nmesh_size 0.1\nelement_type tri3\nn_voxels_dims [4, 4]\n")

        self.assertIn("N_Voxels_Dims", str(context.exception))
        self.assertIn("gmsh", str(context.exception))
        # It used to be read and then ignored, which is the same as not writing it

    def test_an_option_that_no_longer_exists_is_refused(self):
        with self.assertRaises(ValueError) as context:
            self.read("voxel\nn_voxels_dims [4, 4]\nslice_dir 2\n")

        self.assertIn("slice_dir", str(context.exception))
        # Slice_Dir decided whether the grid of a three dimensional microstructure was
        # written at all; the grid is now always written, and the keyword is gone

    def test_the_meshers_options_come_from_the_meshers(self):
        mesh_options = build_reader().all_keywords["Mesh_Options"]
        headers = {i_keyword.name for i_keyword in mesh_options.header_keys}
        self.assertEqual(headers, set(available_meshers()) | set(LEGACY_MESHERS))
        for i_name in available_meshers():
            declared = set(get_mesher(i_name).options)
            read_here = {i_keyword.name for i_keyword in mesher_keywords(i_name)}
            sub_keys = mesh_options.sub_keys[i_name]
            read_under = {i_keyword.name for i_keyword in sub_keys}

            self.assertTrue(declared)
            self.assertEqual(declared, read_here)
            self.assertTrue(declared <= read_under)
        # A deck names a mesher under the mesher's own name, and whatever the mesher
        # declares is read under it, without this module naming either

    def test_an_option_of_a_format_is_read_under_either(self):
        options = self.read(
            "gmsh\nmesh_size 0.1\nelement_type tri3\ngauss_points 6\n"
            "voxel\nn_voxels_dims [16, 16]\ngauss_points 4\n"
        )

        self.assertEqual(options["gmsh"]["gauss_points"], 6)
        self.assertEqual(options["voxel"]["gauss_points"], 4)
        # A grid is written to a solver as readily as a mesh is, so what belongs to the
        # format is read under both

    def test_the_formats_options_come_from_the_formats(self):
        declared = set(writer_options())
        read_here = {i_keyword.name for i_keyword in format_keywords()}

        self.assertTrue(declared)
        self.assertTrue(declared <= read_here)
        # Whatever a writer declares is read from the input data file without this
        # module naming it

    def test_a_mesher_is_read_under_the_name_it_had(self):
        options = self.read(
            "femsh\nmesh_size 0.1\nelement_type tri3\nrgmsh\nn_voxels_dims [8, 8]\n"
        )
        self.assertEqual(
            options,
            {
                "gmsh": {"mesh_size": 0.1, "element_type": "tri3"},
                "voxel": {"n_voxels_dims": [[8, 8]]},
            },
        )
        with self.assertRaisesRegex(ValueError, "gmsh was specified twice"):
            self.read("femsh\nmesh_size 0.1\ngmsh\nmesh_size 0.2\n")
        # The decks of the thesis ask for them so

    def test_a_mesher_that_does_not_exist_is_named_with_the_line(self):
        with self.assertRaises(ValueError) as context:
            self.read("# comment\nnosuchmesher\nmesh_size 0.1\n")

        self.assertIn("Line 3", str(context.exception))
        self.assertIn("nosuchmesher", str(context.exception))
        self.assertIn(", ".join(available_meshers()), str(context.exception))
        # It used to end the block, and be reported as a line with no keyword

    def test_an_option_that_does_not_exist_is_named_with_the_line(self):
        with self.assertRaises(ValueError) as context:
            self.read("gmsh\nmesh_sise 0.1\n")

        self.assertIn("Line 3", str(context.exception))
        self.assertIn("mesh_sise", str(context.exception))

    def test_a_deck_is_not_given_what_an_earlier_one_gave(self):
        self.read("voxel\nn_voxels_dims [8, 8]\nfile_name first\n")
        deck_path = os.path.join(self.deck_dir.name, "second.mdsim")
        with open(deck_path, "w") as deck:
            deck.write("Verlet_Factor 1.5\n")
        self.assertNotIn("mesh_options", read_legacy_deck(deck_path))
        with open(deck_path, "w") as deck:
            deck.write("Max_Step 3\n")
        parameters = read_legacy_deck(deck_path)["mic_gen_parameters"]
        self.assertNotIn("verlet_factor", parameters)
        # One reader read every deck, and gave a keyword a deck left out the value an
        # earlier deck gave it


if __name__ == "__main__":
    unittest.main()
