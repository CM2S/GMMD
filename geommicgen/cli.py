"""
Module containing the command line entry points of the meshing and the translation.

The stages of the output are usable one at a time, and these are how that is done from
a shell. `geommicgen-mesh` reads a microstructure and discretises it; the mesh it
writes is a whole stage, so `geommicgen-translate` picks up from that file alone, or
from a mesh some other tool produced.

The program that generates a microstructure is `geommicgen` itself; these two never
generate one, and nothing here reads a microstructure until the command that needs one
asks for it.
"""

import argparse
import os

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.from_deck import (
    MeshJob,
    write_formats,
    writers_from_options,
)
from geommicgen.meshing.mesher import available_meshers, get_mesher
from geommicgen.meshing.writers import read_mesh
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
    "float": {"type": float},
    "str": {"type": str},
    "str_list": {"type": format_names, "metavar": "NAMES"},
    "bool": {"action": argparse.BooleanOptionalAction, "default": None},
}
# How each type a format declares an option with is read from the command line. Every
# type the input data file accepts is here: one that is missing would read from a deck
# and then fail when the parser is built, which is the wrong end to find out


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


def add_format_arguments(parser):
    """Add the arguments a written format is configured with."""
    for i_name in sorted(writer_options()):
        description = writer_options()[i_name]
        parser.add_argument(
            "--{0}".format(i_name.lower().replace("_", "-")),
            help=description["help"],
            **ARGUMENT_KWARGS[description["type"]]
        )
    # Taken from the formats themselves, so a writer that declares an option is asked
    # for it here without this module naming it


def format_options(arguments):
    """Collect the format arguments the way the input data file keys them."""
    return {
        i_name.lower(): getattr(arguments, i_name.lower())
        for i_name in writer_options()
    }


def add_output_arguments(parser):
    """Add the arguments saying where the files go and what they are called."""
    parser.add_argument(
        "-o", "--output-dir", default=".", help="directory to write into"
    )
    parser.add_argument("--name", help="name of the files, without an extension")


def output_name(name, source_path):
    """Name the files after the file they came from, when no name was given."""
    return name or os.path.splitext(os.path.basename(source_path))[0]


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
    parser.add_argument("--mesh-size", type=float, help="largest element size")
    parser.add_argument(
        "--elements-per-particle",
        type=float,
        help="elements across the smallest particle, instead of a size",
    )
    parser.add_argument(
        "--element-type", default="tri3", help="element to mesh with (default: tri3)"
    )
    parser.add_argument(
        "--n-voxels",
        type=int,
        nargs="+",
        metavar="N",
        help="number of voxels in each direction, for the voxel mesher",
    )
    parser.add_argument(
        "--to",
        type=format_names,
        default=[],
        metavar="FORMATS",
        help="formats to write besides the mesh itself, separated by commas",
    )
    add_format_arguments(parser)
    add_output_arguments(parser)
    arguments = parser.parse_args(argv)

    writers = resolve_writers(parser, arguments.to, format_options(arguments))
    if arguments.mesher == "voxel":
        if not arguments.n_voxels:
            parser.error("the voxel mesher needs --n-voxels")
        mesher = get_mesher("voxel")(arguments.n_voxels)
    else:
        mesher = get_mesher(arguments.mesher)(
            mesh_size=arguments.mesh_size,
            element_type=arguments.element_type,
            elements_per_particle=arguments.elements_per_particle,
        )

    from geommicgen.iofuncs.microstructure_yaml import read_microstructure_yaml
    # Imported here rather than at the top: reading a microstructure pulls in the
    # particle classes and the parts of scipy they use, which is most of the cost of
    # starting up, and the other command never reads one

    microstructure = read_microstructure_yaml(arguments.microstructure)
    job = MeshJob(
        mesher, writers, output_name(arguments.name, arguments.microstructure)
    )
    job.run(microstructure, arguments.output_dir, report=report_progress)
    for i_warning in mesher.warnings:
        print(i_warning)

    return report_outcome(job.error, job.files)


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
        "--to", type=format_names, metavar="FORMATS",
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
    add_format_arguments(parser)
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

    writers = resolve_writers(parser, arguments.to, format_options(arguments))
    mesh = read_mesh(
        arguments.mesh,
        rve_dims=arguments.rve_dims,
        matrix_phase=arguments.matrix_phase,
    )
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
        return report_outcome(error, written)
    # The mesh that was read is protected, so asking for the format it is already in
    # cannot overwrite the file this was given. A format failing part of the way
    # through still reports what reached the disk, as it does when a deck drives it

    return report_outcome(None, written)
