"""
Module declaring the analyses, and the options that configure them.

This is the one place the analyses are named. Each analysis is declared under its name
with its options, as each mesher declares its own; the model of the analyses section of
the input file and the arguments of the analysis command are built from `ANALYSES`, so
an analysis is asked for the same way from either and one added here is offered by
both. `post_proc_options` gives the analyses asked for as the post processing reads
them. It imports nothing of the analyses themselves, so that reading an input file does
not pay for the plotting they pull in.
"""

VORONOI_TYPES = ("standard", "set")
# The Voronoi diagrams the analysis knows how to compute. A name outside this tuple
# used to be found out deep inside the analysis, as an unbound variable; a weighted
# one was accepted and quietly computed the standard diagram, its implementation
# never having been written

DESCRIPTORS = ("nearest_neighbor", "ripleys_k", "two_pt_corr")
# The statistical descriptors there are, in the order they are worked out

ANALYSES = {
    "final_config": {
        "help": "draw the final configuration of the particles",
        "options": {},
    },
    "motion": {
        "help": "plot what the run recorded -- overlap, energies, time step and the "
        "paths of the particles -- read from md_state.npz beside the microstructure "
        "file",
        "options": {},
    },
    "voronoi": {
        "help": "work out the Voronoi diagram of the particles and the Minkowski "
        "tensors of its cells",
        "options": {
            "type": {
                "type": "str",
                "default": "standard",
                "choices": VORONOI_TYPES,
                "help": "kind of Voronoi diagram: standard, of the centres, or set, of "
                "points on the surfaces of the particles",
            },
            "n_surf_points": {
                "type": "int",
                "default": 10,
                "help": "surface points per particle of a set Voronoi diagram",
            },
            "plot": {
                "type": "bool",
                "default": False,
                "help": "draw the Voronoi diagram",
            },
            "plot_imts": {
                "type": "bool",
                "default": False,
                "help": "draw the cells coloured by their Minkowski tensors",
            },
        },
    },
    "statistics": {
        "help": "work out statistical descriptors of the arrangement of the particles",
        "options": {
            "descriptors": {
                "type": "str_list",
                "default": list(DESCRIPTORS),
                "choices": DESCRIPTORS,
                "help": "descriptors to work out: nearest_neighbor, the distribution "
                "of the distance to the nearest neighbour; ripleys_k, Ripley's K "
                "function; two_pt_corr, the two point correlation function",
            },
            "seed": {
                "type": "int",
                "default": None,
                "help": "seed of the draws Ripley's K function and the two point "
                "correlation function estimate with, so that an analysis of a "
                "microstructure gives the same numbers on every run",
            },
        },
    },
}
# Each analysis by its name, with the line that describes it and its options: each
# option with the type it is read as, the value it has when it is not given, the names
# it takes when it takes one of a few, and the line that describes it

POST_PROC_FLAGS = {
    "final_config": "final_config",
    "motion": "motion_analysis",
    "voronoi": "voronoi_analysis",
}
# The key the post processing asks for each analysis by; the statistics are asked for
# a descriptor at a time

VORONOI_OPTION_NAMES = {
    "type": "voronoi_type",
    "n_surf_points": "n_surf_points",
    "plot": "plot_voronoi",
    "plot_imts": "plot_imts",
}
# The key the post processing reads each option of the Voronoi analysis by

STAT_FLAGS = tuple("stat_" + i_name for i_name in DESCRIPTORS)
# The keys the post processing asks for each statistical descriptor by


def post_proc_options(asked):
    """
    Give the analyses asked for as the post processing reads them.

    Parameters
    ----------
    asked: dict
        The analyses asked for, each by its name, with the options given for it; an
        option not given is left out, and an analysis not asked for is absent or None.

    Returns
    -------
    dict
        Every key the post processing reads, at its default when it was not given.
    """
    asked = {
        i_name: i_options
        for i_name, i_options in asked.items()
        if i_options is not None
    }
    options = {i_key: i_name in asked for i_name, i_key in POST_PROC_FLAGS.items()}
    voronoi = asked.get("voronoi", {})
    for i_option, i_key in VORONOI_OPTION_NAMES.items():
        options[i_key] = voronoi.get(
            i_option, ANALYSES["voronoi"]["options"][i_option]["default"]
        )
    statistics = asked.get("statistics")
    descriptors = (
        []
        if statistics is None
        else statistics.get(
            "descriptors", ANALYSES["statistics"]["options"]["descriptors"]["default"]
        )
    )
    for i_name in DESCRIPTORS:
        options["stat_" + i_name] = i_name in descriptors
    options["stat_seed"] = None if statistics is None else statistics.get("seed")

    return options
    # The options of an analysis not asked for are at their defaults, as they were when
    # the analyses and their options were given side by side
