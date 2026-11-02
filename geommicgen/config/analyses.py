"""
Module holding the model of the analyses section of the input file.

The model is built from `ANALYSES`, the declaration the analysis command is built from
as well, so that an analysis is asked for the same way from either. Each analysis is
given under its name, with its options, as the meshes section gives each mesher.
"""

from typing import Annotated, Literal, Optional

import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import ConfigModel
from geommicgen.postproc.options import ANALYSES, post_proc_options

OPTION_TYPES = {"bool": bool, "int": int, "str": str, "str_list": list[str]}
# The type of a value for each type an analysis option is declared with

LIST_BOUNDS = {"str_list": {"min_length": 1}}
# A list of names names one at least, so that an analysis asked for has work to do


def option_type(description):
    """
    Give the type of the value of an analysis option.

    Parameters
    ----------
    description: dict
        The option, as `ANALYSES` declares it.

    Returns
    -------
    type
        One of the names it takes, when it takes one of a few; may be None, when its
        default is None.
    """
    value_type = OPTION_TYPES[description["type"]]
    if "choices" in description:
        choice = Literal[tuple(description["choices"])]
        value_type = list[choice] if description["type"] == "str_list" else choice
    if description["default"] is None:
        value_type = Optional[value_type]

    return value_type


def asked_for(value):
    """
    Take true for an analysis at its defaults, and false for no analysis at all.

    Parameters
    ----------
    value: object
        What the file gives for an analysis.

    Returns
    -------
    object
        The options of the analysis, none given for true, or None for false.
    """
    if value is True:
        return {}
    if value is False:
        return None

    return value
    # So that final_config: true asks for the drawing, as an analysis whose every option
    # has a default is most often asked for


ANALYSIS_MODELS = {
    i_name: pydantic.create_model(
        "".join(i_word.capitalize() for i_word in i_name.split("_")),
        __base__=ConfigModel,
        __doc__="Model of the options of the {0} analysis.".format(i_name),
        **{
            j_name: (
                option_type(j_description),
                pydantic.Field(
                    j_description["default"],
                    description=j_description["help"],
                    **LIST_BOUNDS.get(j_description["type"], {}),
                ),
            )
            for j_name, j_description in i_declared["options"].items()
        }
    )
    for i_name, i_declared in ANALYSES.items()
}
# A model for the options of each analysis, each option at its default


class AnalysesBase(ConfigModel):
    """
    Model of the analyses section, from which the model of the declared analyses
    derives.

    Each analysis is carried out when it is given, under its name, with its options;
    none is by default.
    """

    def to_options(self):
        """
        Give the analyses as the post processing reads them.

        Returns
        -------
        dict
            Every analysis option, whether the file names it or not.
        """
        return post_proc_options(
            {
                i_name: getattr(self, i_name).model_dump()
                for i_name in ANALYSES
                if getattr(self, i_name) is not None
            }
        )


AnalysesConfig = pydantic.create_model(
    "AnalysesConfig",
    __base__=AnalysesBase,
    **{
        i_name: (
            Annotated[
                Optional[ANALYSIS_MODELS[i_name]], pydantic.BeforeValidator(asked_for)
            ],
            pydantic.Field(None, description=i_declared["help"]),
        )
        for i_name, i_declared in ANALYSES.items()
    }
)
# A field for each analysis, which is not carried out unless it is given
