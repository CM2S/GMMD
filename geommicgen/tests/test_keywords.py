"""
Unit tests regarding the reading of the input data file.

These cover the mesh options, whose sub keywords are read under the discretisation they
belong to and are assembled in part from what the solver writers declare, and the
reference input data file, which is held to what the reader declares.
"""

import os
import re
import tempfile
import unittest

import geommicgen
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


REFERENCE_DECK = os.path.join(
    os.path.dirname(os.path.dirname(geommicgen.__file__)),
    "examples",
    "MIC_input_data_file.dat",
)
# The documented input data file, beside the examples at the top of the repository;
# an installed package has no repository around it, and these tests are skipped

REQUIRED_KEYWORDS = {
    "N_DP_Samples",
    "RVE_Dimensions",
    "Mic_Gen_Descriptors",
    "Max_Residue_Per_Particle",
    "Max_Step",
}
# What a deck has to give: the three the program asks for before generating, and the
# two parameters of the simulation that have no default


@unittest.skipUnless(os.path.exists(REFERENCE_DECK), "not run from the repository")
class TestReferenceInputFile(unittest.TestCase):
    """Test class holding the documented input data file to what the reader declares."""

    @classmethod
    def setUpClass(cls):
        with open(REFERENCE_DECK) as reference:
            cls.text = reference.read()
        cls.syntax = re.findall(r"^#\s*Syntax:\s*([A-Za-z_][A-Za-z0-9_]*)", cls.text, re.M)
        cls.entries = re.findall(
            r"^#.*\[([MO])\]\n# =+\n(?:#.*\n)*?#\s*Syntax:\s*([A-Za-z_][A-Za-z0-9_]*)",
            cls.text,
            re.M,
        )
        # Every entry opens with a title marked [M] or [O], then a rule, then the
        # syntax line naming the keyword; the sub options of the meshes are documented
        # the same way, indented

    def declared(self):
        """Give every keyword name the reader accepts, top level and sub keyword."""
        names = {i_keyword.name for i_keyword in top_level_reader.top_level_keywords}
        for i_name in ("Mesh_Options",):
            group = top_level_reader.all_keywords[i_name]
            names |= {i_keyword.name for i_keyword in group.all_sub_keys}
        return {i_name.lower() for i_name in names}

    def test_every_keyword_declared_is_documented(self):
        documented = {i_name.lower() for i_name in self.syntax}
        documented |= {
            i_name.lower()
            for i_name in re.findall(r"^#\s+([A-Z][A-Za-z_]+) y", self.text, re.M)
        }
        # A parameter of another keyword, like Verlet_Factor, is shown under it as an
        # example line rather than with a syntax line of its own
        self.assertEqual(self.declared() - documented, set())

    def test_every_keyword_documented_is_declared(self):
        self.assertEqual({i_name.lower() for i_name in self.syntax} - self.declared(), set())

    def test_no_keyword_is_documented_twice(self):
        top_level = re.findall(r"^# Syntax:\s*([A-Za-z_][A-Za-z0-9_]*)", self.text, re.M)
        self.assertEqual(len(top_level), len(set(top_level)), sorted(top_level))
        # An entry once carried the syntax line of another keyword, so the one it was
        # about had none

    def test_mandatory_is_what_the_program_requires(self):
        mandatory = {i_name for i_mark, i_name in self.entries if i_mark == "M"}
        self.assertEqual(mandatory, REQUIRED_KEYWORDS)

    def test_the_options_named_are_the_ones_there_are(self):
        from geommicgen.micgenmethod.speed_up_schemes import SPEED_UP_SCHEMES
        from geommicgen.micgenmethod.thermostats import LOWERING_TEMP_CRITERIA, THERMOSTATS

        for i_keyword, i_options in (
            ("Speed_Up_Scheme", SPEED_UP_SCHEMES),
            ("Thermostat", THERMOSTATS),
            ("Lowering_Temp_Criterion", LOWERING_TEMP_CRITERIA),
        ):
            with self.subTest(keyword=i_keyword):
                entry = self.text[self.text.index("Syntax:    " + i_keyword):]
                listed = re.search(r"^# x:\s*\{([^}]*)\}", entry, re.M).group(1)
                self.assertEqual(
                    set(re.findall(r"['\"]([^'\"]*)['\"]", listed)), set(i_options)
                )
        # The names a deck can give are the names the program accepts, no more and no
        # fewer; Verlet2 was documented as a scheme after it had stopped existing


if __name__ == "__main__":
    unittest.main()
