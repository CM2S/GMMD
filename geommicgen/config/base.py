"""
Module holding what every model of the input file is built on.

The input file is YAML, read into a tree of pydantic models, one per section, each
refusing a key it does not declare. A file that does not validate is refused whole,
with one error naming every problem in it by its path in the file, before anything is
generated or written.
"""

import pydantic
import yaml

MISSING_MESSAGE = "is missing"
EXTRA_MESSAGE = "is not a key this section takes"
# Messages that replace pydantic's for the two problems a hand written file has most,
# which pydantic words for the author of a model rather than of a file


class ConfigError(ValueError):
    """Raised for an input file that cannot be read, or does not validate."""


class Problems(ValueError):
    """
    Raised by a validator for problems it finds below the section it validates.

    A validator of a section is told its location, and raises a plain error at it; one
    that checks what the section holds against itself names the key each problem is
    at, so that the error says where in the file it is.

    Attributes
    ----------
    problems: list
        Tuples *(location, message)*, the location a tuple of keys below the section.
    """

    def __init__(self, problems):
        """Initizalizer for the Problems Class."""
        self.problems = list(problems)
        super().__init__("; ".join(i_message for _, i_message in self.problems))


class ConfigModel(pydantic.BaseModel):
    """
    Class for a section of the input file.

    A key the section does not declare is refused rather than ignored, so that a
    misspelt one is found when the file is read and not after a run that left it out.
    """

    model_config = pydantic.ConfigDict(extra="forbid")


class UniqueKeyLoader(yaml.SafeLoader):
    """
    Class for a YAML loader that refuses a key given twice in one mapping.

    YAML keeps the last of two equal keys and says nothing, so the first value given
    would be dropped without a word.
    """

    def construct_mapping(self, node, deep=False):
        """Build a mapping, refusing a key that it has already."""
        keys = set()
        for i_key_node, _ in node.value:
            key = self.construct_object(i_key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "the key {0} is given twice".format(key),
                    i_key_node.start_mark,
                )
            keys.add(key)

        return super().construct_mapping(node, deep=deep)


def load_yaml(path):
    """
    Read a YAML file.

    Parameters
    ----------
    path: str
        Path of the file.

    Returns
    -------
    object
        What the file holds.

    Raises
    ------
    ConfigError:
        If the file is not YAML, or gives a key twice in one mapping.
    """
    with open(path) as yaml_file:
        try:
            return yaml.load(yaml_file, Loader=UniqueKeyLoader)
        except yaml.YAMLError as error:
            raise ConfigError(
                "{0} is not a YAML file that can be read: {1}".format(path, error)
            ) from None


def named(value):
    """
    Take a bare name for a section that is chosen by its name alone.

    Parameters
    ----------
    value: object
        What the file gives: a mapping with a *name*, or the name alone.

    Returns
    -------
    object
        The mapping, which a name alone is turned into.
    """
    if isinstance(value, str):
        return {"name": value}

    return value
    # So that speed_up: cell means speed_up: {name: cell}, as a section whose every
    # other key has a default is most often written


def file_path(loc, data, missing=False):
    """
    Give the path in the file of what a validation error is about.

    Parameters
    ----------
    loc: tuple
        Location pydantic gives the error at.

    data: object
        What the file holds.

    missing: bool
        Whether the error is about a key the file does not give, which is then the
        last part of the location.

    Returns
    -------
    str
        The keys that lead to it in the file, joined by dots.
    """
    path = []
    node = data
    for i_index, i_part in enumerate(loc):
        last = i_index == len(loc) - 1
        key = None
        if isinstance(node, dict):
            for j_key in (i_part, str(i_part)):
                if j_key in node:
                    key = j_key
            if key is None and isinstance(i_part, str) and i_part.isdigit():
                key = int(i_part) if int(i_part) in node else None
        elif isinstance(node, list) and isinstance(i_part, int) and i_part < len(node):
            key = i_part
        if key is not None:
            path.append(str(key))
            node = node[key]
        elif i_part == "[key]":
            path.append("(the key)")
        elif last and missing:
            path.append(str(i_part))
    # A part that leads nowhere in the file is the tag of a union, or a member of one,
    # which pydantic puts in its location and the file does not show; a missing key is
    # the one part that is not in the file and still belongs

    return ".".join(path) if path else "(the file)"


def problem_lines(error, data):
    """
    Put one validation error in lines naming where in the file each problem is.

    Parameters
    ----------
    error: dict
        Error as pydantic describes it.

    data: object
        What the file holds.

    Returns
    -------
    list(str)
        The lines, one for each problem: a validator that raised `.Problems` found
        several.
    """
    raised = error.get("ctx", {}).get("error")
    if isinstance(raised, Problems):
        return [
            "{0}: {1}".format(file_path(tuple(error["loc"]) + i_loc, data), i_message)
            for i_loc, i_message in raised.problems
        ]
    if isinstance(raised, ValueError):
        return ["{0}: {1}".format(file_path(error["loc"], data), raised)]
    # Raised by a validator of the package, whose message is written for the file, and
    # not prefixed with pydantic's "Value error"

    messages = {
        "missing": MISSING_MESSAGE,
        "extra_forbidden": EXTRA_MESSAGE,
        "union_tag_not_found": "gives no {0}, which says what it is".format(
            error.get("ctx", {}).get("discriminator", "")
        ),
    }
    message = messages.get(error["type"], error["msg"])
    line = "{0}: {1}".format(
        file_path(error["loc"], data, missing=error["type"] == "missing"), message
    )
    if error["type"] in messages or error["type"] == "union_tag_invalid":
        return [line]
    # The value is left out of a missing key, which has none, and of a section, which
    # pydantic would give whole, when the message names what is wrong with it
    if "input" in error:
        line += " (given {0!r})".format(error["input"])

    return [line]


def read_model(model, data, source="The input file"):
    """
    Validate what a file holds against a model.

    Parameters
    ----------
    model: class
        Model of the file, a `.ConfigModel`.

    data: object
        What the file holds.

    source: str
        What the file is called in the error, if there is one.

    Returns
    -------
    `.ConfigModel`
        The model, with every default filled in.

    Raises
    ------
    ConfigError:
        If what the file holds does not validate, naming every problem.
    """
    try:
        return model.model_validate(data)
    except pydantic.ValidationError as error:
        problems = sorted(
            {
                j_line
                for i_error in error.errors()
                for j_line in problem_lines(i_error, data)
            }
        )
        raise ConfigError(
            "{0} has {1} problem{2}:\n{3}".format(
                source,
                len(problems),
                "" if len(problems) == 1 else "s",
                "\n".join("  - " + i_problem for i_problem in problems),
            )
        ) from None
    # Every problem at once, so that a file is not fixed one run at a time
