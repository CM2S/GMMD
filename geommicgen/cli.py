"""
Module containing the command line entry points that pick up from a file.

The stages of the output are usable one at a time, and these are how that is done from
a shell. `geommicgen-mesh` reads a microstructure and discretises it; the mesh it
writes is a whole stage, so `geommicgen-translate` picks up from that file alone, or
from a mesh some other tool produced. `geommicgen-analyze` is a path off the
microstructure file rather than a stage after it: the analyses read the microstructure
and, for the motion of its particles, the state of the run written beside it.

The program that generates a microstructure is `geommicgen` itself; these never
generate one, and nothing here reads a microstructure until the command that needs one
asks for it.
"""

import argparse
import os
import time

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
import geommicgen.iofuncs.printing as print_funcs
from geommicgen.pipeline import (
    MeshJob,
    job_base_name,
    write_formats,
    writers_from_options,
)
from geommicgen.meshing.mesher import available_meshers, get_mesher, mesher_options
from geommicgen.meshing.writers import read_mesh
from geommicgen.postproc.options import (
    ANALYSES,
    ANALYSIS_GROUPS,
    ANALYSIS_OPTIONS,
    with_defaults,
)
from geommicgen.translators.base import available_writers, get_writer, writer_options


def format_names(value):
    """
    Split the names of the formats given to --to.

    The names are only split here, not resolved: argparse replaces whatever a type
    callable raises with a message of its own, and the message that says which formats
    there are, or why one is deliberately absent, is worth more than that.

    Parameters
    ----------
    value: str
        Names of the formats, separated by commas.

    Returns
    -------
    list
        The names.
    """
    return [i_name.strip() for i_name in value.split(",") if i_name.strip()]


ARGUMENT_KWARGS = {
    "int": {"type": int},
    "int_list": {"type": int, "nargs": "+", "metavar": "N"},
    "float": {"type": float},
    "str": {"type": str},
    "str_list": {"type": format_names, "metavar": "NAMES"},
    "bool": {"action": argparse.BooleanOptionalAction, "default": None},
}
# How each type a mesher or a format declares an option with is read from the command
# line. Every type the input data file accepts is here: one that is missing would read
# from a deck and then fail when the parser is built, which is the wrong end to find out



def resolve_writers(parser, names, options=None):
    """
    Turn the names of formats into the writers, ending the program if one is unknown.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser to report through.

    names: list
        Names of the formats.

    options: dict
        Options the writers are built from, keyed the way the input data file keys
        them, so that a format is configured the same way from either.

    Returns
    -------
    list
        The writers.
    """
    options = {} if options is None else options
    try:
        return writers_from_options(dict(options, formats=names), ())
    except ValueError as error:
        parser.error(str(error))
    # Resolved before any work is done, and reported in the words of the registry,
    # which names the formats there are and why one is deliberately not offered

    return []


def add_declared_arguments(parser, options, added=None):
    """
    Add the arguments for the options the meshers, the formats or the analyses declare.

    Parameters
    ----------
    parser: argparse.ArgumentParser or argparse._ArgumentGroup
        Parser, or a group of one, to add them to.

    options: dict
        The declared options, of the form *{option_name: description}*.

    added: set
        Names added already, for a caller adding one group after another; a name in it
        is skipped, and every name added is put in it.
    """
    added = set() if added is None else added
    for i_name in sorted(options):
        if i_name in added:
            continue
        description = options[i_name]
        parser.add_argument(
            "--{0}".format(i_name.lower().replace("_", "-")),
            help=description["help"],
            **ARGUMENT_KWARGS[description["type"]],
        )
        added.add(i_name)
    # Taken from the meshers, the formats and the analyses themselves, so one that
    # declares an option is asked for it here without this module naming it


def add_grouped_arguments(parser, titled_options):
    """
    Add the options of each mesher or of each format under a heading of its own.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser to add them to.

    titled_options: list
        Tuples *(title, options)*, one per mesher or format.
    """
    added = set()
    for i_title, i_options in titled_options:
        if not set(i_options) - added:
            continue
        add_declared_arguments(parser.add_argument_group(i_title), i_options, added)
    # The help then says which mesher, or which format, reads each of the options,
    # where one block of them said only that the command takes them all. An option two
    # of them declare alike is added once, under the first


def mesher_arguments(parser):
    """Add the options of every mesher, each under the name of the mesher."""
    add_grouped_arguments(
        parser,
        [
            ("options of the {0} mesher".format(i_name), get_mesher(i_name).options)
            for i_name in available_meshers()
        ],
    )


def writer_arguments(parser):
    """Add the options of every format, each under the name of the format."""
    add_grouped_arguments(
        parser,
        [
            ("options of the {0} format".format(i_name), get_writer(i_name).options)
            for i_name in available_writers()
        ],
    )

def declared_options(arguments, options):
    """Collect the declared arguments the way the input data file keys them."""
    return {i_name.lower(): getattr(arguments, i_name.lower()) for i_name in options}


def add_output_arguments(parser):
    """Add the arguments saying where the files go and what they are called."""
    parser.add_argument(
        "-o", "--output-dir", default=".", help="directory to write into"
    )
    parser.add_argument("--name", help="name of the files, without an extension")


def output_name(name, source_path):
    """Name the files after the file they came from, when no name was given."""
    return name or os.path.splitext(os.path.basename(source_path))[0]


def analysis_files(output_dir, options):
    """
    Give the files the analyses asked for left in the output directory.

    Parameters
    ----------
    output_dir: str
        Directory the analyses were run into.

    options: dict
        The analyses that were asked for.

    Returns
    -------
    list
        Paths of the files, sorted, over the final configuration and the directory
        each analysis names for itself.
    """
    from geommicgen.postproc.postproc import ANALYSIS_DIRECTORIES

    found = []
    for i_name in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
        path = os.path.join(output_dir, i_name)
        if i_name.startswith("final_config") and os.path.isfile(path):
            found.append(path)
        elif i_name in ANALYSIS_DIRECTORIES.values() and os.path.isdir(path):
            found += [
                os.path.join(path, j_name) for j_name in sorted(os.listdir(path))
            ]

    return found
    # The analyses do not say what they wrote, each naming a directory of its own
    # instead, so what is in those is what the command reports


def opened(source_path, label):
    """
    Open a command with what it is working on, and give the moment it started.

    Parameters
    ----------
    source_path: str
        File the command was given.

    label: str
        What that file is.

    Returns
    -------
    float
        The moment the work started, for the step that follows to be timed from.
    """
    print_funcs.log_to_terminal()
    print_funcs.print_initial_message(source_path, label=label)

    return time.time()
    # A command opens and closes as a run of an input data file does, with the same
    # heading, the same sections in between and the same table of times


def print_files(files):
    """Print the paths of the files that were written."""
    for i_file in files:
        print("  {0}".format(i_file))


def report_progress(index, total):
    """Report that one more particle has been dealt with."""
    print("  particle {0} of {1}".format(index + 1, total), end="\r")
    if index + 1 == total:
        print()


def report_outcome(error, files):
    """
    Report what was written and what stopped it, and give the status to exit with.

    Parameters
    ----------
    error: Exception
        Error that stopped the work, or None.

    files: list
        Paths of the files that were written.

    Returns
    -------
    int
        Status the program should exit with.
    """
    if error is not None:
        print("{0}: {1}".format(type(error).__name__, error))
        if files:
            print("written before it failed:")
    print_files(files)

    return 1 if error is not None else 0


def mesh_command(argv=None):
    """
    Discretise a microstructure read from a file.

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
        prog="geommicgen-mesh",
        description="Discretise a microstructure and write the mesh.",
    )
    parser.add_argument("microstructure", help="microstructure file to be meshed")
    parser.add_argument(
        "--mesher",
        default="gmsh",
        choices=available_meshers(),
        help="mesher to discretise with (default: gmsh)",
    )
    mesher_arguments(parser)
    parser.add_argument(
        "--to",
        type=format_names,
        default=[],
        metavar="FORMATS",
        help="formats to write besides the mesh itself, separated by commas",
    )
    writer_arguments(parser)
    add_output_arguments(parser)
    arguments = parser.parse_args(argv)

    writers = resolve_writers(
        parser, arguments.to, declared_options(arguments, writer_options())
    )
    try:
        meshers = get_mesher(arguments.mesher).from_options(
            declared_options(arguments, mesher_options())
        )
    except ValueError as error:
        parser.error(str(error))
    if len(meshers) != 1:
        parser.error("a command line asks for one discretisation at a time")
    mesher = meshers[0]
    # Built the way a deck builds it, from the options the mesher declares

    from geommicgen.iofuncs.microstructure_yaml import read_microstructure_yaml

    # Imported here rather than at the top: reading a microstructure pulls in the
    # particle classes and the parts of scipy they use, which is most of the cost of
    # starting up, and the other command never reads one

    started = opened(arguments.microstructure, "Microstructure")
    microstructure = read_microstructure_yaml(arguments.microstructure)
    read_seconds = time.time() - started
    job = MeshJob(
        mesher,
        writers,
        arguments.name
        or job_base_name(
            os.path.basename(arguments.microstructure), mesher.label
        ),
    )
    # Named after the microstructure file and the label of the mesher, as a deck names
    # a discretisation after the deck and the label: the label is what tells one
    # discretisation of a microstructure from another, so meshing the same
    # microstructure with two elements, or at two resolutions, into one directory no
    # longer writes the second over the first. --name still says it outright
    print_funcs.print_to_file("Generating meshes")
    print_funcs.print_to_file("-" * 80 + "\n")
    print_funcs.print_to_file("\t> {0}".format(job.description))
    job.run(microstructure, arguments.output_dir, report=report_progress)
    for i_warning in mesher.warnings:
        print_funcs.print_to_file("\t\t- {0}".format(i_warning))

    status = report_outcome(job.error, job.files)
    print_funcs.print_final_message(
        {"Reading the microstructure": read_seconds, job.description: job.time}
    )

    return status


def translate_command(argv=None):
    """
    Translate a mesh read from a file into the formats a solver reads.

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
        prog="geommicgen-translate",
        description="Write a mesh in the formats solvers read.",
    )
    parser.add_argument(
        "mesh", nargs="?", help="mesh file to be translated, in any format meshio reads"
    )
    parser.add_argument(
        "--to",
        type=format_names,
        metavar="FORMATS",
        help="formats to write, separated by commas",
    )
    parser.add_argument(
        "--rve-dims",
        type=float,
        nargs="+",
        metavar="L",
        help="dimensions of the RVE, when the mesh comes without them",
    )
    parser.add_argument(
        "--matrix-phase", help="name of the matrix phase, when the mesh does not say"
    )
    writer_arguments(parser)
    add_output_arguments(parser)
    parser.add_argument(
        "--list-formats", action="store_true", help="list the formats and stop"
    )
    arguments = parser.parse_args(argv)

    if arguments.list_formats:
        for i_name in available_writers():
            print("  {0:8s} {1}".format(i_name, get_writer(i_name).extension))

        return 0
    if arguments.mesh is None or not arguments.to:
        parser.error("a mesh and --to are needed, unless --list-formats is given")

    started = opened(arguments.mesh, "Mesh")
    writers = resolve_writers(
        parser, arguments.to, declared_options(arguments, writer_options())
    )
    mesh = read_mesh(
        arguments.mesh,
        rve_dims=arguments.rve_dims,
        matrix_phase=arguments.matrix_phase,
    )
    read_seconds = time.time() - started
    print_funcs.print_to_file("Writing the formats asked for")
    print_funcs.print_to_file("-" * 80 + "\n")
    writing = time.time()
    os.makedirs(arguments.output_dir, exist_ok=True)
    base_path = os.path.join(
        arguments.output_dir, output_name(arguments.name, arguments.mesh)
    )
    written = []
    try:
        write_formats(
            mesh, base_path, writers, protected=[arguments.mesh], written=written
        )
    except Exception as error:  # pylint: disable=broad-except
        status = report_outcome(error, written)
    else:
        status = report_outcome(None, written)
    # The mesh that was read is protected, so asking for the format it is already in
    # cannot overwrite the file this was given. A format failing part of the way
    # through still reports what reached the disk, as it does when a deck drives it

    print_funcs.print_final_message(
        {"Reading the mesh": read_seconds, "Writing the formats": time.time() - writing}
    )

    return status


def analyze_command(argv=None):
    """
    Analyse a microstructure read from a file, and the run that produced it.

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
        prog="geommicgen-analyze",
        description="Analyse a generated microstructure, and the run that produced it.",
    )
    parser.add_argument(
        "microstructure",
        help="microstructure file to be analysed; the state of the run that produced "
        "it is read from md_state.npz beside it, when an analysis needs it",
    )
    add_grouped_arguments(
        parser,
        [
            (i_title, {i_name: ANALYSIS_OPTIONS[i_name] for i_name in i_names})
            for i_title, i_names in ANALYSIS_GROUPS
        ],
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="directory to write into; an earlier analysis there is written over",
    )
    arguments = parser.parse_args(argv)
    # No --name: the analyses name their own directories and files

    options = with_defaults(declared_options(arguments, ANALYSIS_OPTIONS))
    if not any(options[i_name] for i_name in ANALYSES):
        parser.error(
            "no analysis was asked for; give at least one of {0}".format(
                ", ".join("--" + i_name.replace("_", "-") for i_name in ANALYSES)
            )
        )

    from geommicgen.iofuncs.md_state import STATE_FILE_NAME, load_md_state
    from geommicgen.iofuncs.microstructure_yaml import read_microstructure_yaml
    from geommicgen.postproc.postproc import run_analyses

    # Imported here rather than at the top, as the meshing command does: the analyses
    # pull in matplotlib and the particle classes, and the other commands never need
    # them

    started = opened(arguments.microstructure, "Microstructure")
    # The analyses report through the logger; the terminal is where a command's
    # report goes, and no screen file is written -- as the other commands do not

    microstructure = read_microstructure_yaml(arguments.microstructure)
    state = None
    if options["motion_analysis"]:
        state = load_md_state(
            os.path.join(
                os.path.dirname(os.path.abspath(arguments.microstructure)),
                STATE_FILE_NAME,
            )
        )
    read_seconds = time.time() - started
    # Read only when an analysis wants it: the positions are the bulk of the file. The
    # absolute path is taken first, since the directory of a bare file name is empty

    try:
        times = run_analyses(microstructure, state, arguments.output_dir, options)
    except Exception as error:  # pylint: disable=broad-except
        return report_outcome(error, [])
    # What was refused before anything was written, and why, is what the report
    # carries; nothing is listed, since nothing was written

    status = report_outcome(None, analysis_files(arguments.output_dir, options))
    print_funcs.print_final_message(
        dict({"Reading the microstructure": read_seconds}, **times)
    )

    return status
