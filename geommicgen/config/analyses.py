"""
Module holding the model of the analyses section of the input file.

The model is built from `ANALYSIS_OPTIONS`, the declaration the analysis command is
built from as well, so that an analysis is asked for the same way from either.
"""

from typing import Literal, Optional

import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import ConfigModel
from geommicgen.postproc.options import ANALYSIS_OPTIONS, VORONOI_TYPES

OPTION_TYPES = {"bool": bool, "int": int, "str": str}
# The type of a value for each type an analysis option is declared with

CHOICES = {"voronoi_type": Literal[VORONOI_TYPES]}
# The options that take one of a few names, and the names

AnalysesConfig = pydantic.create_model(
    "AnalysesConfig",
    __base__=ConfigModel,
    **{
        i_name: (
            CHOICES.get(
                i_name,
                OPTION_TYPES[i_description["type"]]
                if i_description["default"] is not None
                else Optional[OPTION_TYPES[i_description["type"]]],
            ),
            pydantic.Field(i_description["default"], description=i_description["help"]),
        )
        for i_name, i_description in ANALYSIS_OPTIONS.items()
    }
)
# A field for each option, at its default


def analyses_to_options(analyses):
    """
    Give the analyses as the input data file keys them.

    Parameters
    ----------
    analyses: `.ConfigModel`
        The analyses section.

    Returns
    -------
    dict
        Every analysis option, whether the file names it or not.
    """
    return analyses.model_dump()
