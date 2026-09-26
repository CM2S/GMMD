"""Test module for the models of the input file."""

import os
import tempfile
import unittest
from typing import Annotated, Literal, Union

import numpy as np
import pydantic

from geommicgen.config.base import (
    ConfigError,
    ConfigModel,
    load_yaml,
    named,
    read_model,
)
from geommicgen.config.phases import PHASE_MODELS, PhaseModel, PhaseName, type_name
from geommicgen.microstructure.phase import Phase


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
        specified = self.options({"distribution": "specified", "values": [0.1, 0.2]})
        self.assertEqual(specified["r_distribution"], "specified")
        np.testing.assert_array_equal(specified["r"], [0.1, 0.2])
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


if __name__ == "__main__":
    unittest.main()
