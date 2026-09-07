"""
Module containing the command line entry points of the meshing and the translation.

The stages of the output are usable one at a time, and these are how that is done from
a shell. `geommicgen-mesh` reads a microstructure and discretises it; the mesh it
writes is a whole stage, so `geommicgen-translate` picks up from that file alone, or
from a mesh some other tool produced.

The program that generates a microstructure is `geommicgen` itself; these two never
generate one.
"""

import argparse
import os

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.iofuncs.microstructure_yaml import read_microstructure_yaml
from geommicgen.meshing.from_deck import MeshJob
from geommicgen.meshing.mesher import available_meshers, get_mesher
from geommicgen.meshing.writers import read_mesh
from geommicgen.translators.base import available_writers, get_writer


def formats_argument(value):
    """
    Read the formats given to --to.

    Parameters
    ----------
    value: str
        Names of the formats, separated by commas.

    Returns
    -------
    list
        Classes of the writers.
    """
    return [
        get_writer(i_name.strip())
        for i_name in value.split(",")
        if i_name.strip()
    ]


def report_progress(index, total):
    """Report that one more particle has been dealt with."""
    print("  particle {0} of {1}".format(index + 1, total), end="\r")
    if index + 1 == total:
        print()


def report_outcome(job):
    """
    Report what a job produced, and give back what the program should exit with.

    Parameters
    ----------
    job: `.MeshJob`
        Job that has been run.

    Returns
    -------
    int
        Status the program should exit with.
    """
    if job.error is not None:
        print("{0}: {1}".format(type(job.error).__name__, job.error))
        if job.files:
            print("written before it failed:")
    for i_file in job.files:
        print("  {0}".format(i_file))

    return 1 if job.error is not None else 0


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
        type=formats_argument,
        default=[],
        metavar="FORMATS",
        help="formats to write besides the mesh itself, separated by commas",
    )
    parser.add_argument(
        "-o", "--output-dir", default=".", help="directory to write into"
    )
    parser.add_argument("--name", help="name of the files, without an extension")
    arguments = parser.parse_args(argv)

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

    microstructure = read_microstructure_yaml(arguments.microstructure)
    name = arguments.name or os.path.splitext(
        os.path.basename(arguments.microstructure)
    )[0]
    job = MeshJob(mesher, arguments.to, name)
    job.run(microstructure, arguments.output_dir, report=report_progress)
    for i_warning in mesher.warnings:
        print(i_warning)

    return report_outcome(job)


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
        "--to", type=formats_argument, metavar="FORMATS",
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
    parser.add_argument(
        "-o", "--output-dir", default=".", help="directory to write into"
    )
    parser.add_argument("--name", help="name of the files, without an extension")
    parser.add_argument(
        "--list-formats", action="store_true", help="list the formats and stop"
    )
    arguments = parser.parse_args(argv)

    if arguments.list_formats:
        for i_name in available_writers():
            writer = get_writer(i_name)
            print("  {0:8s} {1}".format(i_name, writer.extension))

        return 0
    if arguments.mesh is None or not arguments.to:
        parser.error("a mesh and --to are needed, unless --list-formats is given")

    mesh = read_mesh(
        arguments.mesh,
        rve_dims=arguments.rve_dims,
        matrix_phase=arguments.matrix_phase,
    )
    name = arguments.name or os.path.splitext(os.path.basename(arguments.mesh))[0]
    os.makedirs(arguments.output_dir, exist_ok=True)
    written = []
    for i_writer in arguments.to:
        written += i_writer().write(
            mesh, os.path.join(arguments.output_dir, name + i_writer.extension)
        )
    for i_file in written:
        print("  {0}".format(i_file))

    return 0
