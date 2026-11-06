"""
Module containing the conversion of a deck into an input file, and its command.

A deck of the keyword format is read by the reader of decks, and what it gives is laid
out as the sections of an input file: the phases keyed by their names with their
descriptors as the microstructure file writes them, and the thermostat and the speed up
scheme as sections of their own. The file is validated before it is written, so a deck
that could not run is refused rather than turned into an input file that cannot.

What a deck gave that nothing read is left out and said so: a parameter of a thermostat
other than the one the deck asks for, the factor of a Verlet list it does not use, and
the problem type.
"""

import argparse
import os

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import read_model, write_config
from geommicgen.config.generation import (
    CRITERIA,
    Generation,
    MolecularDynamics,
    MultiTemperature,
    thermostat_defaults,
)
from geommicgen.config.phases import type_name
from geommicgen.config.run import RunConfig
from geommicgen.iofuncs.legacy_deck import read_legacy_deck
from geommicgen.micgenmethod.speed_up_schemes import SPEED_UP_SCHEMES
from geommicgen.micgenmethod.thermostats import LOWERING_TEMP_CRITERIA, THERMOSTATS
from geommicgen.microstructure.phase import PHASE_KEYS, Phase, gives_parameter
from geommicgen.postproc.options import (
    ANALYSES,
    DESCRIPTORS,
    POST_PROC_FLAGS,
    STAT_FLAGS,
    VORONOI_OPTION_NAMES,
)

THERMOSTAT_PARAMETERS = {
    "multi_temperature": {"initial_temp", "lowering_temp_criterion", "temp_low_ratio"},
    "isokinetic": {"initial_temp"},
    "micro_canonical": set(),
    "berendsen": {"initial_temp", "berendsen_coeff"},
}
# The parameters of the deck each thermostat reads; the multi-temperature one reads
# the parameter of its criterion as well

CRITERION_PARAMETERS = {
    i_model.model_fields["name"].annotation.__args__[0]: next(
        j_name for j_name in i_model.model_fields if j_name != "name"
    )
    for i_model in CRITERIA
}
# The parameter of the deck each criterion of the multi-temperature thermostat reads


def plain(value):
    """Give a value the reader of decks gave as YAML writes it, a list for an array."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()

    return value


def legacy_descriptor(fields):
    """
    Lay out the keys of a deck that give one parameter as a descriptor of the file.

    Parameters
    ----------
    fields: dict
        Values of the keys, keyed by what follows the parameter in them: the empty
        string for the parameter itself, *mean* for its mean, and so on.

    Returns
    -------
    object
        A number for a fixed value, or the distribution as a dictionary. A key the
        distribution does not take is kept in it, for the validation to refuse.
    """
    fields = dict(fields)
    distribution = fields.pop("distribution", "fixed")
    if distribution == "fixed" and set(fields) == {""}:
        return plain(fields[""])
    record = {"distribution": distribution}
    if distribution == "specified" and "" in fields:
        record["values"] = plain(fields.pop(""))
    elif distribution == "discrete":
        indices = sorted(
            int(i_key.split("_")[1]) for i_key in fields if i_key.startswith("value_")
        )
        record["values"] = [plain(fields.pop("value_{0}".format(i))) for i in indices]
        record["probabilities"] = [
            plain(fields.pop("prob_{0}".format(i)))
            for i in indices
            if "prob_{0}".format(i) in fields
        ]
    for i_field, i_value in fields.items():
        record[i_field or "value"] = plain(i_value)

    return record


def legacy_phase(descriptors, notes, name):
    """
    Lay out a phase of a deck as a phase of the file.

    Parameters
    ----------
    descriptors: dict
        The phase as the reader of decks gave it.

    notes: list
        List what is left out of the phase is appended to, said in a line each.

    name: str
        Name of the phase.

    Returns
    -------
    dict
        The phase.
    """
    particle_class = Phase.phase_types.get(descriptors.get("phase_type"))
    phase = {
        "type": type_name(particle_class)
        if particle_class
        else descriptors.get("phase_type")
    }
    if descriptors.get("inner_phase"):
        phase["inside"] = descriptors.get("outer_phase")
    elif "outer_phase" in descriptors:
        notes.append(
            "Outer_Phase of phase {0}, which is not an inner phase, is left out".format(
                name
            )
        )
    parameters = particle_class.possible_parameters if particle_class else {}
    grouped = {}
    for i_key, i_value in descriptors.items():
        if i_key in PHASE_KEYS:
            continue
        parameter = next(
            (j_name for j_name in parameters if gives_parameter(i_key, j_name)), None
        )
        if parameter is None:
            phase[i_key] = plain(i_value)
        else:
            grouped.setdefault(parameter, {})[i_key[len(parameter) + 1:]] = i_value
    # A key that gives no parameter of the type is kept, for the validation to refuse
    for i_parameter, i_fields in grouped.items():
        phase[i_parameter] = legacy_descriptor(i_fields)

    return phase


def legacy_thermostat(parameters, notes):
    """
    Lay out the thermostat of a deck as a section of the file.

    Parameters
    ----------
    parameters: dict
        The generation parameters of the deck.

    notes: list
        List a parameter the thermostat does not read, and the deck gave other than at
        its default, is appended to.

    Returns
    -------
    object
        The section, its name alone when it takes nothing else, or None for the
        default thermostat at its defaults.
    """
    defaults = thermostat_defaults()
    name = parameters.get("thermostat", THERMOSTATS[0])
    read = set(THERMOSTAT_PARAMETERS.get(name, defaults))
    section = {"name": name}
    if parameters.get("initial_temp") is not None and "initial_temp" in read:
        section["initial_temp"] = parameters["initial_temp"]
    if name == "multi_temperature":
        criterion = parameters.get(
            "lowering_temp_criterion", LOWERING_TEMP_CRITERIA[0]
        )
        criterion_section = {"name": criterion}
        if criterion in CRITERION_PARAMETERS:
            parameter = CRITERION_PARAMETERS[criterion]
            read.add(parameter)
            if parameters.get(parameter, defaults[parameter]) != defaults[parameter]:
                criterion_section[parameter] = parameters[parameter]
        if criterion_section != {
            "name": MultiTemperature.model_fields["criterion"].default.name
        }:
            section["criterion"] = (
                criterion if len(criterion_section) == 1 else criterion_section
            )
        ratio = parameters.get("temp_low_ratio", defaults["temp_low_ratio"])
        if ratio != MultiTemperature.model_fields["temp_low_ratio"].default:
            section["temp_low_ratio"] = ratio
    if name == "berendsen" and parameters.get("berendsen_coeff") not in (
        None,
        defaults["berendsen_coeff"],
    ):
        section["coeff"] = parameters["berendsen_coeff"]
    for i_name in sorted(set(defaults) - read):
        if parameters.get(i_name, defaults[i_name]) != defaults[i_name]:
            notes.append(
                "{0} {1}, which the {2} thermostat does not read, is left out".format(
                    i_name, parameters[i_name], name
                )
            )

    if section == {"name": MolecularDynamics.model_fields["thermostat"].default.name}:
        return None

    return name if len(section) == 1 else section


def legacy_speed_up(parameters, notes):
    """
    Lay out the speed up scheme of a deck as a section of the file.

    Parameters
    ----------
    parameters: dict
        The generation parameters of the deck.

    notes: list
        List the factor of a Verlet list the deck does not use is appended to.

    Returns
    -------
    object
        The section, its name alone when it takes nothing else, or None for the
        default scheme.
    """
    scheme = parameters.get("speed_up_scheme", SPEED_UP_SCHEMES[0])
    if scheme == "Verlet":
        section = {"name": "verlet"}
        if "verlet_factor" in parameters:
            section["factor"] = parameters["verlet_factor"]
        return section
    if "verlet_factor" in parameters:
        notes.append(
            "verlet_factor {0}, which the {1} scheme does not read, is left out".format(
                parameters["verlet_factor"], scheme
            )
        )
    name = scheme.lower() if scheme in SPEED_UP_SCHEMES else scheme

    default = MolecularDynamics.model_fields["speed_up"].default.name

    return None if name == default else name


def legacy_generation(parameters, notes):
    """
    Lay out the generation parameters of a deck as the generation section of the file.

    Parameters
    ----------
    parameters: dict
        The generation parameters of the deck.

    notes: list
        List what is left out is appended to.

    Returns
    -------
    dict
        The section: the options every method takes and those of the molecular
        dynamics simulation under its name, which a deck asked for by its parameters,
        each with no default or given other than at its default.
    """
    generation = {}
    simulation = {}
    for i_model, i_section in (
        (Generation, generation),
        (MolecularDynamics, simulation),
    ):
        for i_name, i_field in i_model.model_fields.items():
            if i_name in ("molecular_dynamics", "speed_up", "thermostat"):
                continue
            if i_name not in parameters:
                continue
            value = plain(parameters[i_name])
            if i_field.is_required() or value != i_field.default:
                i_section[i_name] = value
    for i_name, i_default in (
        ("force_option", "intersection_length"),
        ("force_rescale", False),
    ):
        if parameters.get(i_name, i_default) != i_default:
            simulation[i_name] = parameters[i_name]
    # Kept, for the validation to refuse: each has one value now
    for i_name, i_section in (
        ("thermostat", legacy_thermostat(parameters, notes)),
        ("speed_up", legacy_speed_up(parameters, notes)),
    ):
        if i_section is not None:
            simulation[i_name] = i_section
    generation["molecular_dynamics"] = simulation

    return generation


def legacy_analyses(post_proc, notes):
    """
    Lay out the analyses of a deck as the analyses section of the file.

    Parameters
    ----------
    post_proc: dict
        The analyses the deck gives, each keyword at its default when the deck does not
        give it.

    notes: list
        List an option of an analysis the deck does not ask for, given other than at
        its default, is appended to.

    Returns
    -------
    dict
        The section: each analysis the deck asks for under its name, with the options
        the deck gave other than at their default, or true when it gave none.
    """
    analyses = {
        i_name: True
        for i_name, i_flag in POST_PROC_FLAGS.items()
        if not ANALYSES[i_name]["options"] and post_proc.get(i_flag)
    }
    # The analyses that take no options, each asked for by its flag alone

    voronoi = {
        i_option: plain(post_proc[i_keyword])
        for i_option, i_keyword in VORONOI_OPTION_NAMES.items()
        if i_keyword in post_proc
        and post_proc[i_keyword]
        != ANALYSES["voronoi"]["options"][i_option]["default"]
    }
    if post_proc.get(POST_PROC_FLAGS["voronoi"]):
        analyses["voronoi"] = voronoi or True
    else:
        for i_option, i_value in voronoi.items():
            notes.append(
                "{0} {1}, which configures the Voronoi analysis the old file does not "
                "ask for, is left out".format(VORONOI_OPTION_NAMES[i_option], i_value)
            )

    descriptors = [
        i_name
        for i_name, i_flag in zip(DESCRIPTORS, STAT_FLAGS)
        if post_proc.get(i_flag)
    ]
    seed = post_proc.get("stat_seed")
    if descriptors:
        statistics = {}
        if descriptors != list(DESCRIPTORS):
            statistics["descriptors"] = descriptors
        if seed is not None:
            statistics["seed"] = seed
        analyses["statistics"] = statistics or True
    elif seed is not None:
        notes.append(
            "stat_seed {0}, which seeds the statistics the old file does not ask for, "
            "is left out".format(seed)
        )

    return analyses


def legacy_to_config(options):
    """
    Lay out what the reader of decks gave as the sections of an input file.

    Parameters
    ----------
    options: dict
        The options of a deck, as `read_legacy_deck` gives them.

    Returns
    -------
    tuple
        The input file, as a dictionary, and the lines saying what was left out.
    """
    notes = []
    parameters = dict(options.get("mic_gen_parameters", {}))
    config = {}
    if "n_dp_samples" in options:
        config["samples"] = options["n_dp_samples"]
    if "rve_dimensions" in parameters:
        config["rve_dimensions"] = plain(parameters.pop("rve_dimensions"))
    if parameters.get("fixed_seed") is not None:
        config["seed"] = parameters["fixed_seed"]
    if options.get("save_min"):
        config["save_min"] = True
    if "problem_type" in options:
        notes.append(
            "Problem_Type {0}, which nothing read, is left out".format(
                options["problem_type"]
            )
        )
    if "mic_gen_descriptors" in options:
        config["phases"] = {
            i_name: legacy_phase(i_phase, notes, i_name)
            for i_name, i_phase in options["mic_gen_descriptors"].items()
        }
    config["generation"] = legacy_generation(parameters, notes)
    meshes = {}
    for i_name, i_options in options.get("mesh_options", {}).items():
        meshes[i_name] = {}
        for j_name, j_value in i_options.items():
            j_value = plain(j_value)
            if j_name == "n_voxels_dims" and len(j_value) == 1:
                j_value = j_value[0]
            meshes[i_name][j_name] = j_value
    if meshes:
        config["meshes"] = meshes
    analyses = legacy_analyses(options.get("post_proc", {}), notes)
    if analyses:
        config["analyses"] = analyses

    return config, notes


def convert_deck(path):
    """
    Read a deck and give the input file it amounts to, validated.

    Parameters
    ----------
    path: str
        Path of the deck.

    Returns
    -------
    tuple
        The input file, as a dictionary, the run it validates to, and the lines saying
        what was left out.

    Raises
    ------
    ConfigError:
        If the deck does not amount to an input file that validates, naming every
        problem by its path in the file it would be.
    """
    config, notes = legacy_to_config(read_legacy_deck(path))
    run = read_model(
        RunConfig,
        config,
        "{0}, laid out as an input file,".format(os.path.basename(path)),
    )

    return config, run, notes


def convert_deck_command(argv=None):
    """
    Convert a deck into an input file.

    Parameters
    ----------
    argv: list
        Arguments, taken from the command line when they are not given.

    Returns
    -------
    int
        Status to exit with.
    """
    parser = argparse.ArgumentParser(
        prog="geommicgen-convert-mdsim",
        description="Convert an old input data file (.mdsim) into an input file "
        "(.yaml), which is validated before it is written.",
    )
    parser.add_argument("input_data_file", help="old input data file to be converted")
    parser.add_argument(
        "-o",
        "--output",
        help="input file to be written, beside the old file and named after it by "
        "default",
    )
    arguments = parser.parse_args(argv)

    file_path = arguments.output
    if file_path is None:
        file_path = os.path.splitext(arguments.input_data_file)[0] + ".yaml"
    try:
        config, _, notes = convert_deck(arguments.input_data_file)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # A deck that cannot be read, or does not validate, is said so as a command line
    # that cannot be carried out is, with nothing written
    write_config(
        config,
        file_path,
        header="Converted from {0} by geommicgen-convert-mdsim.".format(
            os.path.basename(arguments.input_data_file)
        ),
    )
    for i_note in notes:
        print("Note: {0}.".format(i_note))
    print(file_path)

    return 0
