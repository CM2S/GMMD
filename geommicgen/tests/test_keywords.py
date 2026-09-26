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

        self.assertEqual(options["gmsh"]["element_type"], ["tri3"])
        self.assertEqual(options["gmsh"]["mesh_size"], [0.1])
        self.assertEqual(options["voxel"]["n_voxels_dims"], [[16, 16]])
        self.assertEqual(options["voxel"]["file_name"], "my_grid")

    def test_several_sizes_and_elements_are_read_as_lists(self):
        options = self.read(
            "gmsh\nelements_per_particle [3, 6]\nmesh_size 0.1, 0.05\n"
            "element_type [tri3, tri6]\n"
        )

        self.assertEqual(options["gmsh"]["elements_per_particle"], [3.0, 6.0])
        self.assertEqual(options["gmsh"]["mesh_size"], [0.1, 0.05])
        self.assertEqual(options["gmsh"]["element_type"], ["tri3", "tri6"])
        # A mesh for each combination, as a grid for each resolution

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
        reader = build_reader()
        names = {i_keyword.name for i_keyword in reader.top_level_keywords}
        for i_name in ("Mesh_Options",):
            group = reader.all_keywords[i_name]
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

    def test_every_distribution_is_documented(self):
        from geommicgen.microstructure import phase

        source = open(phase.__file__).read()
        offered = set(re.findall(r'descriptor_distribution == "(\w+)"', source))
        documented = set(re.findall(r"^#\s+parameter_distribution (\w+)$", self.text, re.M))
        self.assertEqual(offered - {"fixed"}, documented)
        # Read off the dispatch in the phase, where a distribution is added; the fixed
        # value is the one given without naming a distribution

    def test_every_shape_parameter_is_documented_with_its_sets(self):
        from geommicgen.microstructure.phase import Phase

        for i_code, i_class in Phase.phase_types.items():
            if i_code == 1:
                continue
            with self.subTest(shape=i_class.__name__):
                usable = set(i_class.possible_parameters) & set().union(
                    *i_class.acceptable_descriptions
                )
                table = self.text[self.text.index("({0})\n".format(i_code)):]
                table = table[: table.index("\n#        ====")]
                rows = set(re.findall(r"^#        ([a-z_0-9]+) +(?:float|integer) ", table, re.M))
                self.assertEqual(rows, usable)
                sets = {
                    frozenset(i_line.replace(",", "").replace("\n#", " ").split())
                    for i_line in re.findall(r"^#          - (.*(?:\n#            .*)*)", table, re.M)
                }
                self.assertEqual(
                    sets,
                    {
                        frozenset(i_set)
                        for i_set in i_class.acceptable_descriptions
                        if i_set <= set(i_class.possible_parameters)
                    },
                )
        # Each shape's table lists the parameters one of its acceptable descriptions
        # uses, and the descriptions themselves, as the classes declare them


if __name__ == "__main__":
    unittest.main()
