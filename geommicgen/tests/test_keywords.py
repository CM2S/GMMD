"""
Unit tests regarding the Keyword classes and the TopLevelReader class used to read the
input file.
"""
import unittest

import numpy as np

# pylint: disable=import-error
from geommicgen.iofuncs.keywords import (
    Keyword,
    KeywordTypeA,
    KeywordTypeB,
    KeywordTypeC,
    TopLevelReader,
    generate_all_possible_keywords_from_particle_attributes,
)


class TestKeywordIsIn(unittest.TestCase):
    """Class for the unit tests regarding `.Keyword.is_in`."""

    def setUp(self):
        TopLevelReader()

    def test_is_in_matching_name(self):
        """Check that a line starting with the keyword name is recognized."""
        keyword = Keyword("dt")
        self.assertTrue(keyword.is_in("dt 0.05"))

    def test_is_in_case_insensitive(self):
        """Check that the match is case insensitive."""
        keyword = Keyword("Max_Step")
        self.assertTrue(keyword.is_in("max_step 10"))

    def test_is_in_not_matching(self):
        """Check that a line starting with a different name is not recognized."""
        keyword = Keyword("dt")
        self.assertFalse(keyword.is_in("dt_adapt True"))

    def test_is_in_only_checks_first_token(self):
        """Check that only the first token of the line is compared."""
        keyword = Keyword("0.05")
        self.assertFalse(keyword.is_in("dt 0.05"))


class TestKeywordReadValue(unittest.TestCase):
    """Class for the unit tests regarding `.Keyword.read_value`."""

    def setUp(self):
        self.reader = TopLevelReader()

    def _set_input(self, line):
        self.reader.input = [line]
        self.reader.i_line = 0

    def test_read_value_float_scalar(self):
        """Check that a single float value is read correctly."""
        self._set_input("Verlet_Factor 1.5")
        keyword = Keyword("Verlet_Factor", type_str="float")
        self.assertEqual(keyword.read_value(), 1.5)
        self.assertEqual(self.reader.i_line, 1)

    def test_read_value_float_vector(self):
        """Check that a vector of float values is read correctly."""
        self._set_input("RVE_Dimensions [1.0, 2.0, 3.0]")
        keyword = Keyword("RVE_Dimensions", type_str="float")
        value = keyword.read_value()
        np.testing.assert_array_equal(value, np.array([1.0, 2.0, 3.0]))

    def test_read_value_int_scalar(self):
        """Check that a single int value is read correctly."""
        self._set_input("Max_Step 10")
        keyword = Keyword("Max_Step", type_str="int")
        self.assertEqual(keyword.read_value(), 10)

    def test_read_value_int_list_of_lists(self):
        """Check that a list of lists of int values is read correctly."""
        self._set_input("keyword_name [1,2,3] [4,5]")
        keyword = Keyword("keyword_name", type_str="int")
        self.assertEqual(keyword.read_value(), [[1, 2, 3], [4, 5]])

    def test_read_value_bool_true(self):
        """Check that a bool value of True is read correctly."""
        self._set_input("Save_History True")
        keyword = Keyword("Save_History", type_str="bool")
        self.assertIs(keyword.read_value(), True)

    def test_read_value_bool_false(self):
        """Check that a bool value of False is read correctly."""
        self._set_input("Save_History False")
        keyword = Keyword("Save_History", type_str="bool")
        self.assertIs(keyword.read_value(), False)

    def test_read_value_bool_invalid_raises(self):
        """Check that an invalid bool value raises a ValueError."""
        self._set_input("Save_History Maybe")
        keyword = Keyword("Save_History", type_str="bool")
        with self.assertRaises(ValueError):
            keyword.read_value()

    def test_read_value_str(self):
        """Check that a string value is read correctly."""
        self._set_input("Speed_Up_Scheme Cell")
        keyword = Keyword("Speed_Up_Scheme", type_str="str")
        self.assertEqual(keyword.read_value(), "Cell")

    def test_read_value_none_returns_name(self):
        """Check that a 'none' type keyword returns its own name."""
        self._set_input("femsh")
        keyword = Keyword("femsh", type_str="none")
        self.assertEqual(keyword.read_value(), "femsh")

    def test_read_value_default_type_reads_raw_string(self):
        """Check that a keyword without a type_str reads the raw second token."""
        self._set_input("Phase 1")
        keyword = Keyword("Phase")
        self.assertEqual(keyword.read_value(), "1")

    def test_read_value_advances_i_line(self):
        """Check that reading a value moves the reader to the next line."""
        self._set_input("dt 0.05")
        keyword = Keyword("dt", type_str="float")
        self.assertEqual(self.reader.i_line, 0)
        keyword.read_value()
        self.assertEqual(self.reader.i_line, 1)


class TestKeywordTypeA(unittest.TestCase):
    """Class for the unit tests regarding `.KeywordTypeA`."""

    def setUp(self):
        self.reader = TopLevelReader()

    def test_default_value_stored_on_init(self):
        """Check that supplying a default value stores it immediately."""
        KeywordTypeA(
            "Max_Steps_To_Relax", "Mic_Gen_Parameters", default_value=0, type_str="int"
        )
        self.assertEqual(
            self.reader.all_options["mic_gen_parameters"]["max_steps_to_relax"], 0
        )

    def test_no_default_value_not_stored_on_init(self):
        """Check that no value is stored if no default value is given."""
        KeywordTypeA("Verlet_Factor", "Mic_Gen_Parameters", type_str="float")
        self.assertNotIn("mic_gen_parameters", self.reader.all_options)

    def test_store_value(self):
        """Check that a value is stored in the correct group and lower-cased."""
        keyword = KeywordTypeA("dt", "Mic_Gen_Parameters", type_str="float")
        keyword.store_value(0.1)
        self.assertEqual(self.reader.all_options["mic_gen_parameters"]["dt"], 0.1)

    def test_store_value_multiple_keywords_same_group(self):
        """Check that several keywords in the same group are stored together."""
        keyword_1 = KeywordTypeA("dt", "Mic_Gen_Parameters", type_str="float")
        keyword_2 = KeywordTypeA("Max_Step", "Mic_Gen_Parameters", type_str="int")
        keyword_1.store_value(0.1)
        keyword_2.store_value(10)
        self.assertEqual(
            self.reader.all_options["mic_gen_parameters"],
            {"dt": 0.1, "max_step": 10},
        )


class TestKeywordTypeB(unittest.TestCase):
    """Class for the unit tests regarding `.KeywordTypeB`."""

    def setUp(self):
        self.reader = TopLevelReader()

    def test_default_value_stored_on_init(self):
        """Check that supplying a default value stores it immediately."""
        KeywordTypeB("save_min", type_str="bool", default_value=False)
        self.assertIs(self.reader.all_options["save_min"], False)

    def test_no_default_value_not_stored_on_init(self):
        """Check that no value is stored if no default value is given."""
        KeywordTypeB("Problem_Type", type_str="int")
        self.assertNotIn("problem_type", self.reader.all_options)

    def test_store_value(self):
        """Check that a value is stored directly and lower-cased."""
        keyword = KeywordTypeB("N_DP_Samples", type_str="int")
        keyword.store_value(5)
        self.assertEqual(self.reader.all_options["n_dp_samples"], 5)


class TestKeywordTypeC(unittest.TestCase):
    """Class for the unit tests regarding `.KeywordTypeC`."""

    def setUp(self):
        self.reader = TopLevelReader()
        self.header_keys = {Keyword("Phase")}
        self.sub_keys = {
            Keyword("Phase_Type", type_str="int"),
            Keyword("n", type_str="int"),
        }
        self.keyword = KeywordTypeC(
            "Mic_Gen_Descriptors",
            header_keys=self.header_keys,
            sub_keys=self.sub_keys,
        )

    def test_read_value_single_group(self):
        """Check that a single header with sub keys is read correctly."""
        self.reader.input = [
            "Mic_Gen_Descriptors",
            "Phase 1",
            "Phase_Type 2",
            "n 10",
        ]
        self.reader.i_line = 0
        options = self.keyword.read_value()
        self.assertEqual(options, {"1": {"phase_type": 2, "n": 10}})

    def test_read_value_multiple_groups(self):
        """Check that several header groups are read correctly."""
        self.reader.input = [
            "Mic_Gen_Descriptors",
            "Phase 1",
            "Phase_Type 2",
            "n 10",
            "Phase 2",
            "Phase_Type 3",
        ]
        self.reader.i_line = 0
        options = self.keyword.read_value()
        self.assertEqual(
            options,
            {"1": {"phase_type": 2, "n": 10}, "2": {"phase_type": 3}},
        )

    def test_read_value_stops_at_unknown_line(self):
        """Check that reading stops once a line has no known keyword."""
        self.reader.input = [
            "Mic_Gen_Descriptors",
            "Phase 1",
            "Phase_Type 2",
            "Some_Other_Keyword 1",
        ]
        self.reader.i_line = 0
        self.keyword.read_value()
        self.assertEqual(self.reader.i_line, 3)

    def test_read_value_duplicate_header_raises(self):
        """Check that specifying the same header value twice raises a ValueError."""
        self.reader.input = [
            "Mic_Gen_Descriptors",
            "Phase 1",
            "Phase 1",
        ]
        self.reader.i_line = 0
        with self.assertRaises(ValueError):
            self.keyword.read_value()

    def test_read_value_duplicate_sub_key_raises(self):
        """Check that supplying the same sub key twice under a header raises a
        ValueError."""
        self.reader.input = [
            "Mic_Gen_Descriptors",
            "Phase 1",
            "Phase_Type 2",
            "Phase_Type 3",
        ]
        self.reader.i_line = 0
        with self.assertRaises(ValueError):
            self.keyword.read_value()

    def test_store_value(self):
        """Check that the value is stored directly under the keyword's own name."""
        self.keyword.store_value({"1": {"phase_type": 2}})
        self.assertEqual(
            self.reader.all_options["mic_gen_descriptors"], {"1": {"phase_type": 2}}
        )


class TestTopLevelReaderIgnoreComments(unittest.TestCase):
    """Class for the unit tests regarding `.TopLevelReader.ignore_comments`."""

    def setUp(self):
        self.reader = TopLevelReader()

    def test_skips_blank_lines(self):
        """Check that blank lines are skipped."""
        self.reader.input = ["", "  ", "dt 0.05"]
        self.reader.i_line = 0
        self.reader.ignore_comments()
        self.assertEqual(self.reader.i_line, 2)

    def test_skips_comment_lines(self):
        """Check that lines starting with '#' are skipped."""
        self.reader.input = ["# a comment", "dt 0.05"]
        self.reader.i_line = 0
        self.reader.ignore_comments()
        self.assertEqual(self.reader.i_line, 1)

    def test_skips_insert_here_placeholder(self):
        """Check that the '[insert here]' placeholder line is skipped."""
        self.reader.input = ["[insert here]", "dt 0.05"]
        self.reader.i_line = 0
        self.reader.ignore_comments()
        self.assertEqual(self.reader.i_line, 1)

    def test_stops_on_real_line(self):
        """Check that a real line is not skipped over."""
        self.reader.input = ["dt 0.05", "# comment"]
        self.reader.i_line = 0
        self.reader.ignore_comments()
        self.assertEqual(self.reader.i_line, 0)

    def test_reaches_end_of_input(self):
        """Check that the reader stops at the end of the input if only comments and
        blank lines remain."""
        self.reader.input = ["", "# comment", "[insert here]"]
        self.reader.i_line = 0
        self.reader.ignore_comments()
        self.assertEqual(self.reader.i_line, 3)


class TestTopLevelReaderCheckTopLevelKeywords(unittest.TestCase):
    """Class for the unit tests regarding `.TopLevelReader.check_top_level_keywords`."""

    def setUp(self):
        self.reader = TopLevelReader()
        self.reader.add_top_level_keyword(KeywordTypeB("dt", type_str="float"))

    def test_known_keyword_is_stored(self):
        """Check that a recognized keyword's value is stored."""
        self.reader.input = ["dt 0.05"]
        self.reader.i_line = 0
        self.reader.check_top_level_keywords()
        self.assertEqual(self.reader.all_options["dt"], 0.05)

    def test_unknown_keyword_raises(self):
        """Check that an unrecognized keyword raises a ValueError."""
        self.reader.input = ["unknown_keyword 0.05"]
        self.reader.i_line = 0
        with self.assertRaises(ValueError):
            self.reader.check_top_level_keywords()


class TestTopLevelReaderMoveAlong(unittest.TestCase):
    """Class for the unit tests regarding `.TopLevelReader.move_along`."""

    def setUp(self):
        self.reader = TopLevelReader()
        self.reader.add_top_level_keyword(
            KeywordTypeB("dt", type_str="float"),
            KeywordTypeB("Max_Step", type_str="int"),
        )

    def test_reads_all_keywords_ignoring_comments_and_blanks(self):
        """Check that all keywords in the input are read, skipping comments/blanks."""
        self.reader.input = [
            "# header comment",
            "",
            "dt 0.05",
            "Max_Step 10",
            "",
        ]
        self.reader.i_line = 0
        self.reader.move_along()
        self.assertEqual(self.reader.all_options, {"dt": 0.05, "max_step": 10})


class TestTopLevelReaderAddTopLevelKeyword(unittest.TestCase):
    """Class for the unit tests regarding `.TopLevelReader.add_top_level_keyword`."""

    def setUp(self):
        self.reader = TopLevelReader()

    def test_add_single_keyword(self):
        """Check that a single keyword is registered."""
        keyword = KeywordTypeB("dt", type_str="float")
        self.reader.add_top_level_keyword(keyword)
        self.assertIn(keyword, self.reader.top_level_keywords)
        self.assertEqual(self.reader.all_keywords["dt"], keyword)

    def test_add_multiple_keywords(self):
        """Check that multiple keywords are registered at once."""
        keyword_1 = KeywordTypeB("dt", type_str="float")
        keyword_2 = KeywordTypeB("Max_Step", type_str="int")
        self.reader.add_top_level_keyword(keyword_1, keyword_2)
        self.assertEqual(
            self.reader.top_level_keywords, {keyword_1, keyword_2}
        )
        self.assertEqual(self.reader.all_keywords["Max_Step"], keyword_2)


class TestGenerateAllPossibleKeywords(unittest.TestCase):
    """Class for the unit tests regarding
    `.generate_all_possible_keywords_from_particle_attributes`."""

    def setUp(self):
        TopLevelReader()

    def test_base_particle_parameters_included(self):
        """Check that the base Particle parameters are always included."""
        keyword_set = generate_all_possible_keywords_from_particle_attributes()
        keyword_names = {keyword.name: keyword for keyword in keyword_set}
        self.assertIn("n", keyword_names)
        self.assertEqual(keyword_names["n"].type_str, "int")
        self.assertIn("vf", keyword_names)
        self.assertEqual(keyword_names["vf"].type_str, "float")

    def test_returns_a_set_of_keywords(self):
        """Check that the function returns a set of `.Keyword` instances."""
        keyword_set = generate_all_possible_keywords_from_particle_attributes()
        self.assertIsInstance(keyword_set, set)
        for keyword in keyword_set:
            self.assertIsInstance(keyword, Keyword)

