"""Test module for the models of the input file."""

import copy
import os
import re
import tempfile
import unittest
from typing import Annotated, Literal, Union

import numpy as np
import pydantic
import yaml

from geommicgen.config.base import (
    ConfigError,
    ConfigModel,
    load_yaml,
    named,
    read_model,
)
from geommicgen.config import generation
from geommicgen.config.analyses import ANALYSIS_MODELS, AnalysesConfig
from geommicgen.config.phases import (
    PHASE_MODELS,
    Descriptor,
    PhaseModel,
    PhaseName,
    type_name,
)
from geommicgen.config.run import RunConfig, read_config
from geommicgen.iofuncs.legacy_deck import build_reader, read_legacy_deck
from geommicgen.meshing.mesher import available_meshers
from geommicgen.micgenmethod import speed_up_schemes, thermostats
from geommicgen.micgenmethod.molecular_dynamics_sim import (
    OPTIONAL_OPTIONS,
    REQUIRED_OPTIONS,
)
from geommicgen.microstructure.phase import Phase
from geommicgen.pipeline import build_mesh_jobs


class Cell(ConfigModel):
    """Model of a section chosen by its name alone."""

    name: Literal["cell"]


class Verlet(ConfigModel):
    """Model of a section chosen by its name, with a parameter."""

    name: Literal["verlet"]
    factor: float = pydantic.Field(gt=1)


class Section(ConfigModel):
    """Model of a file with a union, a map and a required key."""

    scheme: Annotated[
        Union[Cell, Verlet],
        pydantic.Field(discriminator="name"),
        pydantic.BeforeValidator(named),
    ] = Cell(name="cell")
    steps: int
    sizes: dict[str, float] = {}


class TestReadingAFile(unittest.TestCase):
    """Test class for a YAML file read before it is validated."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def load(self, text):
        """Write a file and read it."""
        path = os.path.join(self.temp_dir.name, "run.yaml")
        with open(path, "w") as yaml_file:
            yaml_file.write(text)

        return load_yaml(path)

    def test_a_key_given_twice_is_refused_with_its_line(self):
        with self.assertRaisesRegex(ConfigError, "key steps is given twice") as context:
            self.load("steps: 3\nsizes: {a: 1}\nsteps: 4\n")
        self.assertIn("line 3", str(context.exception))
        # YAML keeps the last and says nothing

    def test_a_file_that_is_not_yaml_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "not a YAML file"):
            self.load("steps: [3\n")


class TestValidation(unittest.TestCase):
    """Test class for the errors a file that does not validate is refused with."""

    def refused(self, data):
        """Give the lines of the error a file is refused with."""
        with self.assertRaises(ConfigError) as context:
            read_model(Section, data, "run.yaml")

        return str(context.exception).splitlines()

    def test_every_problem_is_named_at_once_by_its_path(self):
        lines = self.refused({"scheme": {"name": "verlet", "factor": 0.5}, "stepz": 3})
        self.assertEqual(lines[0], "run.yaml has 3 problems:")
        self.assertEqual(
            sorted(lines[1:]),
            [
                "  - scheme.factor: Input should be greater than 1 (given 0.5)",
                "  - steps: is missing",
                "  - stepz: is not a key this section takes",
            ],
        )
        # The union's tag, verlet, is in pydantic's location of the factor and not in
        # the file, so it is not in the path

    def test_a_key_of_a_map_that_is_not_right_is_said_to_be_the_key(self):
        lines = self.refused({"steps": 1, "sizes": {"a": "big"}})
        self.assertEqual(
            lines[1],
            "  - sizes.a: Input should be a valid number, unable to parse string as a "
            "number (given 'big')",
        )

    def test_a_name_alone_chooses_a_section(self):
        self.assertIsInstance(
            read_model(Section, {"scheme": "cell", "steps": 1}).scheme, Cell
        )
        self.assertEqual(
            self.refused({"scheme": "verlet", "steps": 1})[1:],
            ["  - scheme.factor: is missing"],
        )
        self.assertIn(
            "does not match any of the expected tags: 'cell', 'verlet'",
            self.refused({"scheme": "hash", "steps": 1})[1],
        )


class Phases(ConfigModel):
    """Model of a file holding phases alone."""

    phases: dict[PhaseName, PhaseModel]


def read_phases(phases):
    """Validate phases, giving their models."""
    return read_model(Phases, {"phases": phases}, "run.yaml").phases


def refused_phases(test, phases):
    """Give the lines of the error the phases are refused with."""
    with test.assertRaises(ConfigError) as context:
        read_phases(phases)

    return str(context.exception).splitlines()[1:]


class TestPhases(unittest.TestCase):
    """Test class for the phases of the input file."""

    def test_each_type_takes_the_parameters_its_class_declares(self):
        self.assertEqual(
            set(PHASE_MODELS),
            {type_name(i_class) for i_class in Phase.phase_types.values()},
        )
        for i_class in Phase.phase_types.values():
            with self.subTest(particle=i_class.__name__):
                model = PHASE_MODELS[type_name(i_class)]
                self.assertEqual(
                    set(model.model_fields) - {"type", "inside"},
                    set(i_class.possible_parameters),
                )
        # Built from the class, so the class is the one place a parameter is declared

    def test_a_description_of_the_type_is_read_and_another_is_refused(self):
        phase = read_phases({1: {"type": "disk", "r": 0.1, "n": 4}})["1"]
        self.assertEqual(phase.to_options(), {"phase_type": 2, "r": 0.1, "n": 4})
        self.assertEqual(
            refused_phases(self, {1: {"type": "disk", "r": 0.1, "n": 4, "vf": 0.1}}),
            [
                "  - phases.1: n, r, vf do not describe a disk, which is described "
                "by one of: n, r; r, vf; n, vf; area, vf; area, n"
            ],
        )

    def test_a_key_that_is_not_a_parameter_of_the_type_is_refused(self):
        self.assertEqual(
            refused_phases(
                self, {1: {"type": "ellipse", "r": 0.1, "n": 4, "angle": 0}}
            ),
            ["  - phases.1.r: is not a key this section takes"],
        )
        # The keys of a phase were matched to its parameters by how they began, and one
        # that matched none was ignored

    def test_the_number_and_the_fraction_are_fixed_and_whole_numbers_whole(self):
        normal = {"distribution": "normal", "mean": 0.1, "sigma": 0.01}
        self.assertEqual(
            refused_phases(self, {1: {"type": "disk", "r": 0.1, "vf": normal}}),
            ["  - phases.1.vf: has to be a fixed value, a number"],
        )
        self.assertEqual(
            refused_phases(
                self,
                {1: {"type": "cylindrical_fiber", "r": 0.1, "n": 3, "direction": 1.5}},
            ),
            ["  - phases.1.direction: has to be a whole number"],
        )

    def test_values_specified_are_one_for_each_particle(self):
        specified = {"distribution": "specified", "values": [0.1, 0.1, 0.1]}
        read_phases({1: {"type": "disk", "r": specified, "n": 3}})
        read_phases({1: {"type": "disk", "r": specified, "vf": 0.1}})
        self.assertEqual(
            refused_phases(self, {1: {"type": "disk", "r": specified, "n": 4}}),
            [
                "  - phases.1.r.values: gives 3 values, one for each particle, and the "
                "phase has 4 particles"
            ],
        )
        # Drawing the fourth particle went past them, and stopped the run

    def test_a_phase_is_named_by_a_whole_number(self):
        self.assertEqual(
            list(read_phases({0: {"type": "matrix"}, "1": {"type": "matrix"}})),
            ["0", "1"],
        )
        self.assertEqual(
            refused_phases(self, {"fibres": {"type": "matrix"}}),
            [
                "  - phases.fibres.(the key): a phase is named by a whole number, "
                "which the meshes label it with"
            ],
        )

    def test_a_phase_inside_another_is_given_as_the_deck_gives_it(self):
        phase = read_phases({2: {"type": "disk", "r": 0.01, "n": 4, "inside": 1}})["2"]
        self.assertEqual(
            phase.to_options(),
            {"phase_type": 2, "inner_phase": True, "outer_phase": 1, "r": 0.01, "n": 4},
        )

    def test_a_fixed_value_is_held_to_the_bounds_the_rve_sets(self):
        phase = read_phases({1: {"type": "disk", "r": 0.3, "n": 4}})["1"]
        self.assertEqual(
            phase.problems([1.0, 1.0]),
            [(("r",), "the radius cannot be 0.3 in an RVE of [1.0, 1.0]")],
        )
        self.assertEqual(phase.problems([2.0, 2.0]), [])


class TestDescriptors(unittest.TestCase):
    """Test class for the descriptors of a phase."""

    def options(self, descriptor):
        """Give a descriptor of the radius of disks as the deck keys it."""
        options = read_phases({1: {"type": "disk", "r": descriptor, "n": 3}})[
            "1"
        ].to_options()

        return {i_key: i_value for i_key, i_value in options.items() if i_key[0] == "r"}

    def test_each_is_given_as_the_deck_gives_it(self):
        for i_descriptor, i_options in (
            (0.1, {"r": 0.1}),
            ({"distribution": "fixed", "value": 1}, {"r": 1.0}),
            (
                {"distribution": "normal", "mean": 0.1, "sigma": 0.01},
                {"r_distribution": "normal", "r_mean": 0.1, "r_sigma": 0.01},
            ),
            (
                {"distribution": "lognormal", "mean": 0.1, "sigma": 0.5},
                {"r_distribution": "lognormal", "r_mean": 0.1, "r_sigma": 0.5},
            ),
            (
                {"distribution": "uniform", "low": 0.1, "high": 0.2},
                {"r_distribution": "uniform", "r_low": 0.1, "r_high": 0.2},
            ),
            (
                {"distribution": "vonmises", "kappa": 2, "loc": 0, "scale": 1},
                {
                    "r_distribution": "vonmises",
                    "r_kappa": 2.0,
                    "r_loc": 0.0,
                    "r_scale": 1.0,
                },
            ),
            (
                {
                    "distribution": "discrete",
                    "values": [0.1, 0.2],
                    "probabilities": [0.25, 0.75],
                },
                {
                    "r_distribution": "discrete",
                    "r_value_1": 0.1,
                    "r_prob_1": 0.25,
                    "r_value_2": 0.2,
                    "r_prob_2": 0.75,
                },
            ),
        ):
            with self.subTest(descriptor=i_descriptor):
                options = self.options(i_descriptor)
                self.assertEqual(options, i_options)
                self.assertEqual(
                    [type(i_value) for i_value in options.values()],
                    [type(i_value) for i_value in i_options.values()],
                )
        specified = self.options(
            {"distribution": "specified", "values": [0.1, 0.2, 0.3]}
        )
        self.assertEqual(specified["r_distribution"], "specified")
        np.testing.assert_array_equal(specified["r"], [0.1, 0.2, 0.3])
        # Floats where the deck reads floats, and an array for values given particle by
        # particle, as the deck reads a list of them

    def test_the_parameters_of_each_are_held_to_what_it_can_draw_from(self):
        for i_descriptor, i_problem in (
            ({"distribution": "normal", "mean": 0.1, "sigma": -1}, "r.sigma"),
            ({"distribution": "lognormal", "mean": 0, "sigma": 1}, "r.mean"),
            ({"distribution": "lognormal", "mean": 1, "sigma": 0}, "r.sigma"),
            ({"distribution": "uniform", "low": 0.2, "high": 0.1}, "r: the lower"),
            ({"distribution": "vonmises", "kappa": 0, "loc": 0, "scale": 1}, "r.kappa"),
            ({"distribution": "vonmises", "kappa": 1, "loc": 0, "scale": 0}, "r.scale"),
            ({"distribution": "specified", "values": []}, "r.values"),
            (
                {"distribution": "discrete", "values": [1], "probabilities": [1, 0]},
                "r: there are 1 values and 2 probabilities",
            ),
            (
                {
                    "distribution": "discrete",
                    "values": [1, 2],
                    "probabilities": [0.5, 0.495],
                },
                "r: the probabilities add up to 0.995, not to one",
            ),
            (
                {
                    "distribution": "discrete",
                    "values": [1, 2],
                    "probabilities": [-1, 2],
                },
                "r.probabilities.0",
            ),
            ({"mean": 0.1}, "r: gives no 'distribution', which says what it is"),
        ):
            with self.subTest(descriptor=i_descriptor):
                lines = refused_phases(
                    self, {1: {"type": "disk", "r": i_descriptor, "n": 3}}
                )
                self.assertEqual(len(lines), 1)
                self.assertIn("phases.1." + i_problem, lines[0])
        # Probabilities within a hundredth of one were let through by the distribution
        # and refused by the drawing
        self.options(
            {"distribution": "discrete", "values": [1, 2], "probabilities": [0.1, 0.9]}
        )
        # Added up in floating point, 0.1 and 0.9 are one


EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
)

DECK_FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "decks"
)
# The decks of the examples, as they were written before the input files were


def assert_same_options(test, options, expected, path="options"):
    """Check two sets of options are the same, arrays and the type of every value."""
    if isinstance(expected, dict):
        test.assertIsInstance(options, dict, path)
        test.assertEqual(set(options), set(expected), path)
        for i_key in expected:
            assert_same_options(
                test, options[i_key], expected[i_key], "{0}.{1}".format(path, i_key)
            )
    elif isinstance(expected, np.ndarray):
        test.assertIsInstance(options, np.ndarray, path)
        np.testing.assert_array_equal(options, expected, err_msg=path)
    else:
        test.assertEqual(options, expected, path)
        test.assertIs(type(options), type(expected), path)


MINIMAL_RUN = {
    "samples": 1,
    "rve_dimensions": [1, 1],
    "phases": {0: {"type": "matrix"}, 1: {"type": "disk", "r": 0.1, "n": 4}},
    "generation": {
        "molecular_dynamics": {"max_residue_per_particle": 0, "max_step": 10}
    },
}
# A run that gives what has no default, and nothing else


class TestRun(unittest.TestCase):
    """Test class for the input file as a whole."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def refused(self, change):
        """Give the lines of the error a change to the minimal run is refused with."""
        data = copy.deepcopy(MINIMAL_RUN)
        change(data)
        with self.assertRaises(ConfigError) as context:
            read_model(RunConfig, data, "run.yaml")

        return str(context.exception).splitlines()[1:]

    @unittest.skipUnless(os.path.isdir(EXAMPLES), "not run from the repository")
    def test_the_examples_give_what_their_decks_give(self):
        names = [
            i_name[: -len(".mdsim")]
            for i_name in sorted(os.listdir(DECK_FIXTURES))
            if i_name.endswith(".mdsim")
        ]
        self.assertEqual(len(names), 2)
        for i_name in names:
            with self.subTest(example=i_name):
                assert_same_options(
                    self,
                    read_config(os.path.join(EXAMPLES, i_name + ".yaml")).to_options(),
                    read_legacy_deck(os.path.join(DECK_FIXTURES, i_name + ".mdsim")),
                )
        # Whatever reads the options is then the same for either file

    def test_what_is_not_given_is_what_a_deck_that_does_not_give_it_gets(self):
        options = read_model(RunConfig, MINIMAL_RUN).to_options()
        for i_keyword in build_reader().top_level_keywords:
            group = getattr(i_keyword, "keyword_group", None)
            if not hasattr(i_keyword, "default_value") or group is None:
                continue
            with self.subTest(keyword=i_keyword.name):
                self.assertEqual(
                    options[group.lower()][i_keyword.name.lower()],
                    i_keyword.default_value,
                )
        self.assertEqual(options["save_min"], False)
        self.assertNotIn("mesh_options", options)

    def test_every_parameter_of_the_simulation_is_given(self):
        parameters = read_model(RunConfig, MINIMAL_RUN).to_options()[
            "mic_gen_parameters"
        ]
        for i_name in REQUIRED_OPTIONS + OPTIONAL_OPTIONS:
            self.assertIn(i_name, parameters)

    def test_the_names_are_the_ones_the_simulation_offers(self):
        for i_models, i_names in (
            (generation.SPEED_UP_SCHEMES, speed_up_schemes.SPEED_UP_SCHEMES),
            (generation.THERMOSTATS, thermostats.THERMOSTATS),
            (generation.CRITERIA, thermostats.LOWERING_TEMP_CRITERIA),
        ):
            with self.subTest(names=i_names):
                self.assertEqual(
                    {
                        j_model.model_fields["name"].annotation.__args__[0]
                        for j_model in i_models
                    },
                    {j_name.lower() for j_name in i_names},
                )
        self.assertEqual(
            set(RunConfig.model_fields["meshes"].annotation.model_fields),
            set(available_meshers()),
        )

    def test_the_sections_are_checked_against_each_other(self):
        for i_change, i_lines in (
            (
                lambda d: d["phases"].update({2: {"type": "matrix"}}),
                ["  - phases: one phase has to be the matrix, and phases 0, 2 are"],
            ),
            (
                lambda d: d["phases"].pop(0),
                ["  - phases: one phase has to be the matrix, and none is"],
            ),
            (
                lambda d: d["phases"].update({2: {"type": "sphere", "r": 0.1, "n": 3}}),
                ["  - phases.2.type: a sphere is in 3 dimensions, and the RVE in 2"],
            ),
            (
                lambda d: d["phases"][1].update(r=0.3),
                ["  - phases.1.r: the radius cannot be 0.3 in an RVE of [1.0, 1.0]"],
            ),
            (
                lambda d: d["phases"].update(
                    {2: {"type": "disk", "r": 0.01, "n": 3, "inside": 0}}
                ),
                ["  - phases.2.inside: has to name another phase of particles"],
            ),
            (
                lambda d: d.update(meshes={"voxel": {"n_voxels_dims": [8, 8, 8]}}),
                [
                    "  - meshes.voxel.n_voxels_dims: [8, 8, 8] has 3 directions, and "
                    "the RVE 2"
                ],
            ),
        ):
            with self.subTest(lines=i_lines):
                self.assertEqual(self.refused(i_change), i_lines)

    def test_fibres_go_alone_in_one_direction_and_not_on_a_grid(self):
        def fibres(data, second, meshes=None):
            data.update(rve_dimensions=[1, 1, 1], meshes=meshes or {})
            data["phases"][1] = {
                "type": "cylindrical_fiber", "r": 0.1, "n": 3, "direction": 2
            }
            data["phases"][2] = second

        self.assertEqual(
            self.refused(lambda d: fibres(d, {"type": "sphere", "r": 0.1, "n": 3})),
            [
                "  - phases.2.type: cylindrical fibres go with no other particles, and "
                "phase 1 is of them"
            ],
        )
        self.assertEqual(
            self.refused(
                lambda d: fibres(
                    d,
                    {"type": "cylindrical_fiber", "r": 0.1, "n": 3, "direction": 1},
                    {"voxel": {"n_voxels_dims": [8, 8, 8]}},
                )
            ),
            [
                "  - meshes.voxel: cylindrical fibres are not stamped on a grid of "
                "voxels; gmsh meshes them",
                "  - phases: the cylindrical fibres have to run in one direction, "
                "given as a fixed value",
            ],
        )
        # Each was found once the particles had been generated, or not at all: fibres
        # in two directions were moved in the plane across the first

    def test_the_meshes_are_refused_as_their_meshers_and_writers_refuse_them(self):
        for i_meshes, i_line in (
            (
                {"femsh": {"mesh_size": 0.1}},
                "  - meshes.femsh: There is no mesher named femsh. The available "
                "meshers are gmsh, voxel.",
            ),
            (
                {"gmsh": {"element_type": "tri3"}},
                "  - meshes.gmsh: Either mesh_size or elements_per_particle must be "
                "specified.",
            ),
            (
                {"gmsh": {"mesh_size": 0.1, "formats": ["nope"]}},
                "  - meshes.gmsh: There is no writer for the format nope.",
            ),
            (
                {"voxel": {"n_voxels_dims": [[8, 8], [16, 16]], "file_name": "x"}},
                "  - meshes.voxel: File_Name x would be given to each of the 2 "
                "discretisations",
            ),
        ):
            with self.subTest(meshes=i_meshes):
                lines = self.refused(lambda d: d.update(meshes=i_meshes))
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith(i_line), lines[0])
        # Built from the options before anything is generated, which is where a deck
        # found them too

    def test_several_sizes_and_elements_are_a_mesh_each(self):
        run = read_model(
            RunConfig,
            dict(
                MINIMAL_RUN,
                meshes={
                    "gmsh": {
                        "elements_per_particle": [4, 8],
                        "element_type": ["tri3", "tri6"],
                    }
                },
            ),
        )
        self.assertEqual(
            [
                i_job.base_name
                for i_job in build_mesh_jobs(
                    run.to_options()["mesh_options"], "deck.yaml"
                )
            ],
            ["deck_tri3_epp4", "deck_tri3_epp8", "deck_tri6_epp4", "deck_tri6_epp8"],
        )
        # As a grid at each of several resolutions, and named so that none is written
        # over another

    def test_a_name_for_several_meshes_is_refused(self):
        lines = self.refused(
            lambda d: d.update(
                meshes={"gmsh": {"elements_per_particle": [4, 8], "file_name": "x"}}
            )
        )
        self.assertEqual(len(lines), 1)
        self.assertTrue(
            lines[0].startswith(
                "  - meshes.gmsh: File_Name x would be given to each of the 2 "
                "discretisations"
            ),
            lines[0],
        )

    def test_the_simulation_is_refused_what_it_would_stop_on(self):
        for i_generation, i_line in (
            (
                {"thermostat": "berendsen"},
                "thermostat.initial_temp: is missing",
            ),
            (
                {"thermostat": {"name": "isokinetic", "criterion": "original"}},
                "thermostat.criterion: is not a key this section takes",
            ),
            (
                {"speed_up": {"name": "verlet", "factor": 1}},
                "speed_up.factor: Input should be greater than 1 (given 1)",
            ),
            (
                {"type_initial_configuration": "hexagonal"},
                "type_initial_configuration: Input should be 'random', 'grid', 'bcc' "
                "or 'fcc' (given 'hexagonal')",
            ),
            (
                {"force_option": "intersection_area"},
                "force_option: is not a key this section takes",
            ),
        ):
            with self.subTest(generation=i_generation):
                self.assertEqual(
                    self.refused(
                        lambda d: d["generation"]["molecular_dynamics"].update(
                            i_generation
                        )
                    ),
                    ["  - generation.molecular_dynamics." + i_line],
                )

    def test_a_file_of_no_sections_is_refused(self):
        path = os.path.join(self.temp_dir.name, "run.yaml")
        with open(path, "w") as yaml_file:
            yaml_file.write("- 1\n- 2\n")
        with self.assertRaisesRegex(ConfigError, "run.yaml holds no sections"):
            read_config(path)


REFERENCE_FILE = os.path.join(EXAMPLES, "reference.yaml")
# The reference input file, which documents every key


def model_fields():
    """Give the name of every key of every section of an input file, and whether each
    has no default, keyed by the section's model."""
    models = [
        RunConfig,
        generation.Generation,
        generation.MolecularDynamics,
        *generation.SPEED_UP_SCHEMES,
        *generation.THERMOSTATS,
        *generation.CRITERIA,
        RunConfig.model_fields["meshes"].annotation,
        RunConfig.model_fields["analyses"].annotation,
        *ANALYSIS_MODELS.values(),
        *PHASE_MODELS.values(),
        *[
            i_type
            for i_type in Descriptor.__origin__.__args__
        ],
    ]
    meshes = RunConfig.model_fields["meshes"].annotation
    models += [
        i_field.annotation.__args__[0] for i_field in meshes.model_fields.values()
    ]

    return {
        i_model: {
            j_name: j_field.is_required() for j_name, j_field in i_model.model_fields.items()
        }
        for i_model in models
    }


@unittest.skipUnless(os.path.exists(REFERENCE_FILE), "not run from the repository")
class TestReferenceFile(unittest.TestCase):
    """Test class holding the reference input file to the models of the file."""

    @classmethod
    def setUpClass(cls):
        with open(REFERENCE_FILE) as reference:
            cls.text = reference.read()
        cls.keys = set(re.findall(r"^\s*#?\s*([a-z_][a-z0-9_]*):", cls.text, re.M))
        for i_mapping in re.findall(r"\{([^{}]*)\}", cls.text):
            cls.keys |= set(re.findall(r"([a-z_][a-z0-9_]*):", i_mapping))
        # The keys the file gives or shows, commented out or not: each opens a line, or
        # is inside a mapping written on one

    def test_it_is_a_run(self):
        read_config(REFERENCE_FILE)

    def test_every_key_there_is_is_shown(self):
        for i_model, i_fields in model_fields().items():
            with self.subTest(section=i_model.__name__):
                self.assertEqual(set(i_fields) - self.keys, set())

    def test_every_key_shown_is_one_there_is(self):
        fields = set().union(*model_fields().values())
        self.assertEqual(self.keys - fields, set())

    def test_the_keys_without_a_default_are_the_ones_marked_required(self):
        lines = self.text.splitlines()
        for i_model in (
            RunConfig,
            generation.Generation,
            generation.MolecularDynamics,
        ):
            for i_name, i_required in model_fields()[i_model].items():
                with self.subTest(key=i_name):
                    index = next(
                        j_index
                        for j_index, j_line in enumerate(lines)
                        if re.match(r"\s*{0}:".format(i_name), j_line)
                    )
                    comment = []
                    while index > 0 and lines[index - 1].strip().startswith("#"):
                        index -= 1
                        comment.append(lines[index])
                    marked = any(
                        re.fullmatch(r"\s*# (Required\.|.* \(required\))", j_line)
                        for j_line in comment
                    ) or (
                        not comment
                        and "# {0} (required)".format(i_name.capitalize()) in self.text
                    )
                    self.assertEqual(marked, i_required)
        # A key is documented in the comment right above it, which ends in its default
        # or in Required., and where the lines indented further document the keys of
        # its section; a section is documented under its heading

    def test_the_defaults_stated_are_the_ones_the_keys_have(self):
        lines = self.text.splitlines()
        for i_model, i_section, i_indent in (
            (RunConfig, None, ""),
            (generation.Generation, "generation:", "  "),
            (generation.MolecularDynamics, "  molecular_dynamics:", "    "),
            (AnalysesConfig, "analyses:", "  "),
            (ANALYSIS_MODELS["voronoi"], "  voronoi:", "    "),
            (ANALYSIS_MODELS["statistics"], "  statistics:", "    "),
        ):
            start = lines.index(i_section) if i_section else 0
            for i_name, i_field in i_model.model_fields.items():
                if i_field.is_required() or i_name in ("phases", "meshes", "analyses"):
                    continue
                with self.subTest(key=i_name):
                    index = next(
                        j_index
                        for j_index in range(start, len(lines))
                        if lines[j_index].startswith("{0}{1}:".format(i_indent, i_name))
                    )
                    stated = None
                    while lines[index - 1].strip().startswith("#"):
                        index -= 1
                        match = re.fullmatch(r"\s*# Default: (.*)", lines[index])
                        if match:
                            stated = yaml.safe_load(match.group(1))
                    default = getattr(i_field.default, "name", i_field.default)
                    self.assertEqual(stated if stated != "none" else None, default)
        # The default each key is documented with, in the Default: line of the comment
        # above it, looked for below the key of its section; a section chosen by its
        # name by the name

    def test_each_type_is_shown_with_its_parameters_and_descriptions(self):
        types = [i_name for i_name in PHASE_MODELS if i_name != "matrix"]
        for i_index, i_name in enumerate(types):
            with self.subTest(type=i_name):
                start = self.text.index("\n#   {0} ".format(i_name))
                end = (
                    self.text.index("\n#   {0} ".format(types[i_index + 1]))
                    if i_index + 1 < len(types)
                    else self.text.index("\n# Every type takes")
                )
                block = self.text[start:end]
                particle_class = PHASE_MODELS[i_name].particle_class
                self.assertEqual(
                    set(re.findall(r"^#     ([a-z_0-9]+):", block, re.M)),
                    set(particle_class.possible_parameters) - {"n", "vf"},
                )
                listing = block[block.index("described by one of:") :].split(":", 1)[1]
                sets = [
                    re.sub(r"\n#\s*", " ", j_set)
                    for j_set in re.split(r"\n#\s+- ", listing)[1:]
                ]
                self.assertEqual(
                    {
                        frozenset(j_name.strip() for j_name in j_set.split(","))
                        for j_set in sets
                    },
                    {
                        frozenset(j_description)
                        for j_description in particle_class.acceptable_descriptions
                    },
                )


if __name__ == "__main__":
    unittest.main()
