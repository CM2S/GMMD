"""
Module holding the model of the meshes section of the input file.

The section names each discretisation by the mesher that produces it, and gives under
it the options that mesher declares, the options of the formats it is written in, the
formats themselves and the name of the files. The model of each mesher's options is
built from what the mesher and the writers declare, so that a mesher or a format that
is registered is asked for here without this module naming it.
"""

from typing import Optional, Union

import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import ConfigModel, Problems
from geommicgen.meshing.mesher import MESHERS, available_meshers, get_mesher
from geommicgen.pipeline import build_mesh_jobs
from geommicgen.translators import writer_options

OPTION_TYPES = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "int_list": Union[list[int], list[list[int]]],
    "float_list": Union[float, list[float]],
    "str_list": Union[str, list[str]],
}
# The type of a value for each type a mesher or a format declares an option with. A
# list of whole numbers may be several, one per discretisation, as the grid is asked
# for at several resolutions at once. A list of sizes or of elements may also be one
# value, which gmsh meshes as a list of one

JOB_OPTIONS = {
    "Formats": {
        "type": "str_list",
        "help": "formats the mesh is written in, beside the standard output",
    },
    "File_Name": {"type": "str", "help": "name of the files, without an extension"},
}
# Options of a discretisation that belong to neither the mesher nor a format


def mesher_model(mesher_class):
    """
    Build the model of the options of a mesher, from what it and the writers declare.

    Parameters
    ----------
    mesher_class: class
        Class of the mesher.

    Returns
    -------
    class
        The model, a `.ConfigModel` with an optional field for each option, at the
        default declared for it, if any, and described by its help.
    """
    declared = {**mesher_class.options, **writer_options(), **JOB_OPTIONS}
    fields = {
        i_name.lower(): (
            Optional[OPTION_TYPES[i_description["type"]]],
            pydantic.Field(
                i_description.get("default"), description=i_description["help"]
            ),
        )
        for i_name, i_description in declared.items()
    }

    return pydantic.create_model(
        mesher_class.__name__ + "Options", __base__=ConfigModel, **fields
    )
    # An option not given is at the default its mesher or writer declares, which the
    # resolved file then states; one that declares none is left to the initializer, as
    # the input data file left it


def to_mesh_options(section):
    """
    Give the options of a discretisation as the input data file keys them.

    Parameters
    ----------
    section: `.ConfigModel`
        Options of the discretisation.

    Returns
    -------
    dict
        The options given, a list of whole numbers as a list of such lists.
    """
    options = {}
    for i_name, i_value in section.model_dump(exclude_none=True).items():
        if (
            isinstance(i_value, list)
            and i_value
            and all(isinstance(j_value, int) for j_value in i_value)
        ):
            i_value = [i_value]
        options[i_name] = i_value
    # A list written in brackets, as every list in the input data file is, was read as
    # the first of several

    return options


class MeshesBase(ConfigModel):
    """
    Model of the meshes section, from which the model of the registered meshers derives.
    """

    @pydantic.model_validator(mode="before")
    @classmethod
    def names_a_mesher(cls, data):
        """Refuse a discretisation named after no mesher, naming the ones there are."""
        if isinstance(data, dict):
            problems = []
            for i_name in data:
                try:
                    get_mesher(i_name)
                except ValueError as error:
                    problems.append(((i_name,), str(error)))
            if problems:
                raise Problems(problems)

        return data

    @pydantic.model_validator(mode="after")
    def builds_the_jobs(self):
        """Refuse options a mesher or a writer refuses, or files that would collide."""
        problems = []
        for i_name, i_options in self.to_options().items():
            try:
                build_mesh_jobs({i_name: i_options})
            except ValueError as error:
                problems.append(((i_name,), str(error)))
        if not problems:
            try:
                build_mesh_jobs(self.to_options())
            except ValueError as error:
                problems.append(((), str(error)))
        # Each discretisation is built on its own first, so that what is wrong with
        # one is said under its name; the whole set then, for the names of their
        # files, which only collide between two of them
        if problems:
            raise Problems(problems)

        return self

    def to_options(self):
        """
        Give the discretisations as the input data file keys them.

        Returns
        -------
        dict
            The options of each discretisation asked for, under its mesher.
        """
        return {
            i_name: to_mesh_options(getattr(self, i_name))
            for i_name in type(self).model_fields
            if getattr(self, i_name) is not None
        }


available_meshers()
MeshesConfig = pydantic.create_model(
    "MeshesConfig",
    __base__=MeshesBase,
    **{
        i_name: (Optional[mesher_model(i_class)], None)
        for i_name, i_class in sorted(MESHERS.items())
    }
)
# A field for each registered mesher, under its name
