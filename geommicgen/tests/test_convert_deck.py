"""Test module for the conversion of a deck into an input file."""

import contextlib
import copy
import io
import os
import re
import tempfile
import textwrap
import unittest

from geommicgen.config.base import ConfigError, write_config
from geommicgen.config.generation import thermostat_defaults
from geommicgen.config.run import read_config
from geommicgen.iofuncs.convert_deck import convert_deck, convert_deck_command
from geommicgen.iofuncs.legacy_deck import build_reader, read_legacy_deck
from geommicgen.tests.test__main__ import SEEDED_DECK
from geommicgen.tests.test_config import DECK_FIXTURES, assert_same_options

KITCHEN_SINK_2D = """
    Problem_Type 1
    N_DP_Samples 2
    RVE_Dimensions [2, 1]
    Fixed_Seed 7
    save_min True
    Mic_Gen_Descriptors
    Phase 0
    Phase_Type 1
    Phase 1
    Phase_Type 2
    n 5
    r_distribution discrete
    r_value_1 0.05
    r_prob_1 0.25
    r_value_2 0.1
    r_prob_2 0.75
    Phase 2
    Phase_Type 3
    vf 0.1
    major_axis_distribution uniform
    major_axis_low 0.1
    major_axis_high 0.2
    minor_axis_distribution lognormal
    minor_axis_mean 0.05
    minor_axis_sigma 0.1
    angle_distribution vonmises
    angle_kappa 2
    angle_loc 0
    angle_scale 1
    Phase 3
    Phase_Type 2
    inner_phase True
    outer_phase 1
    r_distribution specified
    r [0.01, 0.02]
    n 2
    Max_Residue_Per_Particle 1e-4
    Max_Step 50
    Max_Steps_To_Relax 3
    dt 0.01
    dt_adapt False
    Save_History True
    Type_Initial_Configuration grid
    Min_Distance 0.001
    initial_vel_coeff 0.2
    final_overlap_check True
    Damping_Coeff 0.1
    Particle_Mass_Opt volume
    offset False
    Force_Option intersection_length
    Force_Rescale False
    Thermostat multi_temperature
    Lowering_Temp_Criterion rolling_ave
    Average_Window 10
    temp_low_ratio 0.5
    Initial_Temp 100
    Berendsen_Coeff 0.05
    Max_Ratio_Osc 3
    Min_Eq_Steps_At_Temp 30
    Speed_Up_Scheme Naive
    Verlet_Factor 1.2
    final_config True
    motion_analysis True
    voronoi_analysis True
    voronoi_type set
    n_surf_points 5
    plot_voronoi True
    plot_imts True
    stat_nearest_neighbor True
    stat_ripleys_k True
    stat_two_pt_corr True
    stat_seed 3
    Mesh_Options
    gmsh
    mesh_size 0.05
    element_type tri6
    formats [links, abaqus]
    gauss_points 4
    boundary_type Linear_Condition
    periodic_constraints False
    file_name kitchen
    voxel
    n_voxels_dims [16, 8] [32, 16]
    formats [crate]
    """
# A deck giving every keyword of the generation, the analyses and the meshes, most at
# other than their default, and a distribution of every kind. The thermostat reads the
# criterion's window and not the other criteria's parameters, or the Berendsen
# coefficient, and the naive list reads no Verlet factor

KITCHEN_SINK_3D = """
    N_DP_Samples 1
    RVE_Dimensions [1, 1, 1]
    Mic_Gen_Descriptors
    Phase 0
    Phase_Type 1
    Phase 1
    Phase_Type 4
    n 3
    r_distribution normal
    r_mean 0.1
    r_sigma 0.01
    Phase 2
    Phase_Type 5
    vf 0.02
    axis_1 0.2
    ratio_321 0.5
    p_3 0.1
    phi_z_distribution uniform
    phi_z_low 0
    phi_z_high 1
    Phase 3
    Phase_Type 7
    r_cyl 0.05
    ratio 4
    n 2
    azimuth_angle 0.1
    polar_angle 0.2
    Max_Residue_Per_Particle 0
    Max_Step 10
    Thermostat berendsen
    Initial_Temp 5
    Berendsen_Coeff 0.02
    Speed_Up_Scheme Verlet
    Verlet_Factor 1.3
    Mesh_Options
    femsh
    elements_per_particle 4
    element_type tetra4
    rgmsh
    n_voxels_dims 8 8 8
    """
# A deck in three dimensions, of the particles the other leaves out, with the meshers
# asked for by their old names and a grid written as a bare row of numbers

FIBRES = """
    N_DP_Samples 1
    RVE_Dimensions [1, 1, 1]
    Mic_Gen_Descriptors
    Phase 0
    Phase_Type 1
    Phase 1
    Phase_Type 6
    vf 0.2
    area 0.01
    direction 1
    Max_Residue_Per_Particle 0
    Max_Step 10
    Thermostat isokinetic
    Speed_Up_Scheme Cell
    """

MICRO_CANONICAL = """
    N_DP_Samples 1
    RVE_Dimensions [1, 1]
    Mic_Gen_Descriptors
    Phase 0
    Phase_Type 1
    Phase 1
    Phase_Type 2
    r 0.1
    vf 0.1
    Max_Residue_Per_Particle 0
    Max_Step 10
    Thermostat micro_canonical
    Initial_Temp 3
    Lowering_Temp_Criterion original
    """

DECKS = {
    "kitchen_sink_2d": KITCHEN_SINK_2D,
    "kitchen_sink_3d": KITCHEN_SINK_3D,
    "fibres": FIBRES,
    "micro_canonical": MICRO_CANONICAL,
    "seeded": SEEDED_DECK,
}
# The decks written here, and the one the tests of the program run

THERMOSTAT_READS = {
    "multi_temperature": {"initial_temp", "lowering_temp_criterion", "temp_low_ratio"},
    "isokinetic": {"initial_temp"},
    "micro_canonical": set(),
    "berendsen": {"initial_temp", "berendsen_coeff"},
}
CRITERION_READS = {
    "ratio_in_out": "max_ratio_osc",
    "rolling_ave": "average_window",
    "original": "min_eq_steps_at_temp",
}
# What each thermostat reads of the deck, as `thermostat_from_options` reads it; the
# multi-temperature one reads the parameter of its criterion as well


def canonical(options):
    """
    Give what a deck gives, with what nothing read of it at its default.

    Parameters
    ----------
    options: dict
        What the reader of decks gave.

    Returns
    -------
    dict
        The same, with the parameters of the thermostats the deck does not use at
        their defaults, and without a Verlet factor it does not use, the problem type,
        a fixed distribution named, or an inner phase that is not one; a grid given as
        a bare row of numbers is the one resolution it is.
    """
    options = copy.deepcopy(options)
    options.pop("problem_type", None)
    parameters = options["mic_gen_parameters"]
    read = set(THERMOSTAT_READS[parameters["thermostat"]])
    if parameters["thermostat"] == "multi_temperature":
        read.add(CRITERION_READS[parameters["lowering_temp_criterion"]])
    for i_name, i_default in thermostat_defaults().items():
        if i_name not in read:
            parameters[i_name] = i_default
    if parameters["speed_up_scheme"] != "Verlet":
        parameters.pop("verlet_factor", None)
    for i_phase in options.get("mic_gen_descriptors", {}).values():
        if not i_phase.get("inner_phase"):
            i_phase.pop("inner_phase", None)
            i_phase.pop("outer_phase", None)
        for i_key in list(i_phase):
            if i_key.endswith("_distribution") and i_phase[i_key] == "fixed":
                del i_phase[i_key]
    for i_mesher in options.get("mesh_options", {}).values():
        resolutions = i_mesher.get("n_voxels_dims")
        if resolutions and isinstance(resolutions[0], int):
            i_mesher["n_voxels_dims"] = [resolutions]

    return options


class DeckTest(unittest.TestCase):
    """Base class for tests that write decks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write(self, text, name="deck.mdsim"):
        """Write a deck, giving its path."""
        path = os.path.join(self.temp_dir.name, name)
        with open(path, "w") as deck:
            deck.write(textwrap.dedent(text))

        return path


class TestEquivalence(DeckTest):
    """Test class for a deck and the input file it is converted into."""

    def decks(self):
        """Give the path of every deck the equivalence is held on."""
        paths = {
            i_name: self.write(i_text, i_name + ".mdsim")
            for i_name, i_text in DECKS.items()
        }
        for i_name in sorted(os.listdir(DECK_FIXTURES)):
            paths[i_name] = os.path.join(DECK_FIXTURES, i_name)

        return paths

    def test_a_deck_gives_through_its_input_file_what_it_gives_itself(self):
        for i_name, i_path in self.decks().items():
            with self.subTest(deck=i_name):
                config, _, _ = convert_deck(i_path)
                path = os.path.join(self.temp_dir.name, i_name + ".yaml")
                write_config(config, path)
                assert_same_options(
                    self,
                    read_config(path).to_options(),
                    canonical(read_legacy_deck(i_path)),
                )
        # Read back from the file written, so that what is written is what is held;
        # whatever reads the options is then the same for either file

    def test_the_decks_give_every_keyword(self):
        given = set()
        for i_text in DECKS.values():
            given |= {
                i_line.split()[0].lower()
                for i_line in textwrap.dedent(i_text).splitlines()
                if i_line.strip()
            }
        reader = build_reader()
        keywords = {i_keyword.name.lower() for i_keyword in reader.top_level_keywords}
        mesh_options = reader.all_keywords["Mesh_Options"]
        keywords |= {i_keyword.name.lower() for i_keyword in mesh_options.all_sub_keys}
        keywords |= {i_keyword.name.lower() for i_keyword in mesh_options.header_keys}
        keywords |= {"phase", "phase_type", "inner_phase", "outer_phase"}
        self.assertEqual(keywords - given, set())
        # So that the equivalence is held on every keyword there is


class TestConversion(DeckTest):
    """Test class for what the conversion leaves out and refuses."""

    def test_what_nothing_read_is_left_out_and_said(self):
        _, _, notes = convert_deck(self.write(KITCHEN_SINK_2D))
        self.assertEqual(
            notes,
            [
                "Problem_Type 1, which nothing read, is left out",
                "berendsen_coeff 0.05, which the multi_temperature thermostat does "
                "not read, is left out",
                "max_ratio_osc 3, which the multi_temperature thermostat does not "
                "read, is left out",
                "min_eq_steps_at_temp 30, which the multi_temperature thermostat "
                "does not read, is left out",
                "verlet_factor 1.2, which the Naive scheme does not read, is left out",
            ],
        )

    def test_the_file_written_names_the_phases_by_number_and_lays_out_the_rest(self):
        config, _, _ = convert_deck(self.write(KITCHEN_SINK_3D))
        path = os.path.join(self.temp_dir.name, "deck.yaml")
        write_config(config, path, header="Converted from deck.mdsim")
        with open(path) as yaml_file:
            text = yaml_file.read()
        self.assertTrue(text.startswith("# Converted from deck.mdsim"))
        self.assertIn("\n  0: {type: matrix}\n", text)
        self.assertIn(
            "\n    r: {distribution: normal, mean: 0.1, sigma: 0.01}\n", text
        )
        self.assertIn(
            "\n  thermostat: {name: berendsen, initial_temp: 5.0, coeff: 0.02}\n", text
        )
        self.assertIn("\n  voxel: {n_voxels_dims: [8, 8, 8]}\n", text)

    def test_a_value_the_reader_of_decks_read_as_an_empty_list_is_refused(self):
        with self.assertRaises(ConfigError) as context:
            convert_deck(
                self.write(SEEDED_DECK.replace("Max_Step 3", "Max_Step 3 # few"))
            )
        self.assertIn(
            "generation.max_step: Input should be a valid integer (given [])",
            str(context.exception),
        )
        # A whole number followed by a comment was read as no number at all

    def test_a_key_that_gives_no_parameter_is_refused_under_its_phase(self):
        with self.assertRaises(ConfigError) as context:
            convert_deck(
                self.write(SEEDED_DECK.replace("n 4", "n 4\nmajor_axis 0.2"))
            )
        self.assertIn(
            "phases.1.major_axis: is not a key this section takes",
            str(context.exception),
        )

    def test_the_command_writes_the_file_beside_the_deck_or_where_it_is_told(self):
        deck = self.write(KITCHEN_SINK_2D)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(convert_deck_command([deck]), 0)
        written = os.path.join(self.temp_dir.name, "deck.yaml")
        self.assertTrue(os.path.exists(written))
        self.assertEqual(stdout.getvalue().splitlines()[-1], written)
        self.assertIn("Note: Problem_Type 1, which nothing read", stdout.getvalue())
        elsewhere = os.path.join(self.temp_dir.name, "elsewhere.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            convert_deck_command([deck, "-o", elsewhere])
        self.assertTrue(os.path.exists(elsewhere))

    def test_the_command_refuses_a_deck_that_does_not_validate_and_writes_nothing(self):
        deck = self.write(SEEDED_DECK.replace("N_DP_Samples 2\n", ""))
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as context:
                convert_deck_command([deck])
        self.assertEqual(context.exception.code, 2)
        self.assertIn("samples: is missing", stderr.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "deck.yaml")))


if __name__ == "__main__":
    unittest.main()
