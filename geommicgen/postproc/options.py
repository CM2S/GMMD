"""
Module declaring the analyses, and the options that configure them.

This is the one place the analyses are named. The input data file reader builds its
post processing keywords from it, and the analysis command builds its arguments from
it, so an analysis is asked for the same way from either and one added here is offered
by both. It imports nothing of the analyses themselves, so that the reader of a deck
does not pay for the plotting they pull in.
"""

ANALYSIS_OPTIONS = {
    "final_config": {
        "type": "bool",
        "default": False,
        "help": "plot the final configuration of the particles",
    },
    "motion_analysis": {
        "type": "bool",
        "default": False,
        "help": "plot what the run recorded -- overlap, energies, time step and the "
        "paths of the particles -- read from md_state.npz beside the microstructure "
        "file",
    },
    "voronoi_analysis": {
        "type": "bool",
        "default": False,
        "help": "compute the Voronoi diagram of the particles and the Minkowski "
        "tensors of its cells",
    },
    "voronoi_type": {
        "type": "str",
        "default": "standard",
        "help": "kind of Voronoi diagram: standard or set (default: standard)",
    },
    "n_surf_points": {
        "type": "int",
        "default": 10,
        "help": "surface points per particle of a set Voronoi diagram (default: 10)",
    },
    "plot_voronoi": {
        "type": "bool",
        "default": False,
        "help": "plot the Voronoi diagram",
    },
    "plot_imts": {
        "type": "bool",
        "default": False,
        "help": "plot the Voronoi cells coloured by their Minkowski tensors",
    },
    "stat_nearest_neighbor": {
        "type": "bool",
        "default": False,
        "help": "distribution of the distance to the nearest neighbour",
    },
    "stat_ripleys_k": {
        "type": "bool",
        "default": False,
        "help": "Ripley's K function",
    },
    "stat_two_pt_corr": {
        "type": "bool",
        "default": False,
        "help": "two point correlation function",
    },
}
# Each option by the name the input data file and the command line give it, with the
# type it is read as, the value it has when it is not given, and the line that
# describes it

ANALYSES = (
    "final_config",
    "motion_analysis",
    "voronoi_analysis",
    "stat_nearest_neighbor",
    "stat_ripleys_k",
    "stat_two_pt_corr",
)
# The options that ask for work; the others configure the Voronoi analysis

ANALYSIS_GROUPS = (
    ("the analyses to run", ANALYSES),
    (
        "options of the Voronoi analysis",
        tuple(i_name for i_name in ANALYSIS_OPTIONS if i_name not in ANALYSES),
    ),
)
# The options under the heading each belongs to, so that a help says which of them
# ask for an analysis and which configure one


def with_defaults(options):
    """
    Fill in the default of every analysis option that was not given.

    Parameters
    ----------
    options: dict
        Options given, keyed as `ANALYSIS_OPTIONS` is; one not given is absent or None.

    Returns
    -------
    dict
        Every option, which is what a deck that names none of them produces.
    """
    filled = dict(options)
    for i_name, i_description in ANALYSIS_OPTIONS.items():
        if filled.get(i_name) is None:
            filled[i_name] = i_description["default"]

    return filled
