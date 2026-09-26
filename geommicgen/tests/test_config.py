"""Test module for the models of the input file."""

import os
import tempfile
import unittest
from typing import Annotated, Literal, Union

import pydantic

from geommicgen.config.base import (
    ConfigError,
    ConfigModel,
    load_yaml,
    named,
    read_model,
)


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
        with self.assertRaisesRegex(ConfigError, "the key steps is given twice") as context:
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


if __name__ == "__main__":
    unittest.main()
