"""
Module for file handling.

Making directories. Load and save files.
"""
import argparse
import os
import shutil

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.micgenmethod.mic_from_imagej import generate_microstructure_from_csv
from geommicgen.iofuncs.md_state import (
    STATE_FILE_NAME,
    load_md_state,
    save_md_state,
)
from geommicgen.iofuncs.microstructure_yaml import (
    read_microstructure_yaml,
    write_microstructure_yaml,
)
from .printing import print_output, screen_to, SCREEN_FILE_NAME

MIC_FILE_NAME = "mic.yaml"
MIC_EXTENSIONS = {".yaml", ".yml"}
LOADABLE_EXTENSIONS = MIC_EXTENSIONS | {".csv"}


def create_sample_results_directory(dp_dir):
    """
    Create the results directory.

    Parameters
    ----------
    dp_dir: string
        Directory where the results are going to be stored.
    """
    results_folder = os.path.join(dp_dir, "mic")
    # Creating a tentative path for the results folder
    results_folder_old = results_folder
    # Saving the original name of the results folder
    i = 0
    # Initializing the filename suffix
    while True:
        results_folder = "{0}_{1}".format(results_folder_old, i)
        # Creating a new folder name appending an integer to the name of the original
        # folder
        i += 1
        # Increasing the filenam suffix
        if not os.path.exists(results_folder):
            # Repeat while the folder names already exists
            break
    os.makedirs(results_folder)
    # Creating the directory
    file_path = os.path.join(results_folder, "mic")
    # Saving the file path in the Particle class

    return results_folder, file_path


def copy_input_file(input_file_path, results_folder):
    """Copy the input file to the results directory."""
    _, input_file_name = os.path.split(input_file_path)
    shutil.copyfile(input_file_path, os.path.join(results_folder, input_file_name))


def create_design_point_results_directory(
    input_file_dir: str, input_file_name: str
) -> str:
    """
    Create the results directory.

    Parameters
    ----------
    input_file_dir: str
        Directory where the results are going to be stored.

    input_file_name: str
        Name of the input file.

    Returns
    -------
    results_folder: str
        Directory created to stored the results with same name as the input file.
    """
    results_folder = os.path.join(input_file_dir, input_file_name)
    # Creating a tentative path for the results folder
    results_folder_old = results_folder
    # Saving the original name of the results folder
    i = 0
    # Initializing the filename suffix
    results_folder = results_folder_old
    # Creating a new folder name appending an integer to the name of the original
    while True:
        # folder
        i += 1
        # Increasing the filenam suffix
        if not os.path.exists(results_folder):
            # Repeat while the folder names already exists
            break
        results_folder = "{0}_{1}".format(results_folder_old, i)
        # Creating a new folder name appending an integer to the name of the original
    os.makedirs(results_folder)
    # Creating the directory

    return results_folder


def parse_arguments(argv=None):
    """
    Read the arguments of the program, ending it with a usage line when they are wrong.

    Parameters
    ----------
    argv: list
        Arguments, taken from the command line when they are not given.

    Returns
    -------
    argparse.Namespace
        The arguments: *input_file*, the input data file, and *previous_mic*, the
        microstructure file to post-process instead of generating, or None.
    """
    parser = argparse.ArgumentParser(
        prog="geommicgen",
        description="Generate a set of microstructures from an input data file, and "
        "mesh and analyse each one as the file asks.",
    )
    parser.add_argument("input_file", help="input data file (.mdsim)")
    parser.add_argument(
        "previous_mic",
        nargs="?",
        help="a microstructure file generated earlier: it is meshed and analysed as "
        "the input data file asks, and nothing is generated",
    )
    arguments = parser.parse_args(argv)
    if arguments.previous_mic is not None:
        try:
            check_loadable(arguments.previous_mic)
        except ValueError as error:
            parser.error(str(error))
        # Refused while the arguments are read rather than once the run is under way,
        # by the same check that will do the loading

    return arguments


def check_loadable(previous_mic_path):
    """
    Refuse a previous microstructure that is not of a kind that can be read.

    Parameters
    ----------
    previous_mic_path: str
        Path of the microstructure file.

    Returns
    -------
    str
        Extension of the file.

    Raises
    ------
    ValueError:
        If it is not one of the kinds that can be read.
    """
    _, ext = os.path.splitext(os.path.basename(previous_mic_path))
    if ext not in LOADABLE_EXTENSIONS:
        raise ValueError(
            "Wrong extension for the previous microstructure file: {0}".format(ext)
        )

    return ext


def load_previous_sample(previous_mic_path):
    """Load a microstructure sample."""
    ext = check_loadable(previous_mic_path)
    if ext in MIC_EXTENSIONS:
        current_sample = read_microstructure_yaml(previous_mic_path)
        current_mic_generator = load_md_state(
            os.path.join(os.path.dirname(previous_mic_path), STATE_FILE_NAME)
        )
        # No need to generate a new microstructure. Using a previous microstructure. The
        # state of the run that produced it sits beside it, and is simply absent for a
        # microstructure that came from somewhere else
    else:
        current_sample = generate_microstructure_from_csv(previous_mic_path)
        current_mic_generator = None
    return current_sample, current_mic_generator


def save_mic(
    sample_dir, current_sample, current_mic_generator, print_out=True, provenance=None
):
    """
    Save the microstructure, and the state of the run that produced it.

    Parameters
    ----------
    sample_dir: str
        Directory of the sample.

    current_sample: `.Microstructure`
        Microstructure to be saved.

    current_mic_generator: `.MolecularDynamicsSimulation`
        Generation method that produced it, when there is one.

    print_out: bool
        Whether to report the file that was written.

    provenance: dict
        What the microstructure came from, such as the input data file and the random
        seed. Optional.

    Returns
    -------
    str
        Path of the microstructure file that was written.
    """
    file_path = os.path.join(sample_dir, MIC_FILE_NAME)
    write_microstructure_yaml(current_sample, file_path, provenance=provenance)
    if current_mic_generator is not None:
        save_md_state(sample_dir, current_mic_generator)
    if print_out:
        print_output(file_path)
    # Saving the configuration for later use

    return file_path


def save_status(sample_dir, current_sample, current_mic_generator, mesh_jobs=()):
    """
    Save status with a minimal amount of information (time, total overlap and status).

    Parameters
    ----------
    sample_dir: str
        Directory of the sample.

    current_sample: `.Microstructure`
        Microstructure the status is of.

    current_mic_generator: `.MolecularDynamicsSimulation`
        Generation method that produced it.

    mesh_jobs: list
        The `.MeshJob` objects that were run, when they have been. One line per job
        says whether the discretisation it asked for was produced, so that a sample
        which meshed only in part says so where the sample is, rather than only in the
        summary the run prints and then loses.
    """
    status_file_name = os.path.join(sample_dir, "status")
    with open(status_file_name, "w") as status:
        time_line = "Time: {0}\n".format(
            _status_value(current_mic_generator.time, "{0:.3f}s")
        )
        overlap_line = "Overlap: {0}\n".format(
            _status_value(current_sample.total_overlap, "{0:.3f}")
        )
        status_line = "Status: {0}\n".format(current_mic_generator.status)
        status.writelines(time_line)
        status.writelines(overlap_line)
        status.writelines(status_line)
        for i_job in mesh_jobs:
            status.writelines("Mesh {0}\n".format(i_job.summary()))


def _status_value(value, layout):
    """
    Lay out a value of the status, or say that there is none.

    Parameters
    ----------
    value: float
        The value, or None when the run never produced it.

    layout: str
        Format of the value when there is one.

    Returns
    -------
    str
        The value laid out, or *none*.
    """
    return "none" if value is None else layout.format(value)
    # The status is written in the finally of a run that may have stopped before its
    # time or its overlap existed, and an error raised there would replace the one
    # that stopped it


def delete_screen(screen_dir):
    """Delete screen file."""
    screen_to(None)
    os.remove(os.path.join(screen_dir, SCREEN_FILE_NAME))
    # Detached first: the file is held open while it is written to, and Windows
    # refuses to remove an open file
