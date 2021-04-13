"""
Module for file handling.

Making directories. Load and save files.
"""
import os
import sys
import pickle
import shutil

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from micgenmethod.mic_from_imagej import generate_microstructure_from_csv
from micgenmethod.mic_from_file import generate_microstructure_from_txt
from .printing import print_output

SAMPLE_DIR = ""
RESULTS_FOLDER = ""


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


def get_arguments_from_command_line():
    """Get arguments from the command line."""
    if len(sys.argv) == 1:
        # No input file has been supplied
        raise ValueError("No input file was supplied.")
        # Exiting the script
    if len(sys.argv) > 3:
        raise ValueError("Too many input files were supplied.")

    input_file_path = sys.argv[1]
    input_file_dir = os.path.dirname(sys.argv[1])
    input_file_name, ext = os.path.splitext(os.path.basename(sys.argv[1]))
    # Obtaining the directory and the name of the input file
    previous_mic_path = None
    if len(sys.argv) == 3:
        _, ext = os.path.splitext(os.path.basename(sys.argv[2]))
        if ext in {".mic", ".csv", ".txt"}:
            previous_mic_path = sys.argv[2]
        else:
            raise ValueError(
                "Wrong extension for the previous microstructure file: {0}".format(ext)
            )
    return input_file_path, input_file_dir, input_file_name, ext, previous_mic_path


def load_previous_sample(previous_mic_path):
    """Load a microstructure sample."""
    _, ext = os.path.splitext(os.path.basename(previous_mic_path))
    if ext == ".mic":
        info_previous_sample = pickle.load(open(previous_mic_path, "rb"))
        # No need to generate a new microstructure. Using a previous microstructure.
        current_sample = info_previous_sample["microstructure"]
        current_mic_generator = info_previous_sample["generation_method"]
        # Reconstructing the relevant Particle attributes that could not be pickled
    elif ext == ".csv":
        current_sample = generate_microstructure_from_csv(previous_mic_path)
        current_mic_generator = None
    elif ext == ".txt":
        current_sample = generate_microstructure_from_txt(previous_mic_path)
        current_mic_generator = None
    return current_sample, current_mic_generator


def save_mic(sample_dir, current_sample, current_mic_generator, print_out=True):
    """Save microstructure usign pickle."""
    if os.path.exists(os.path.join(sample_dir, "mic.mic")):
        # Repeat while the folder names already exists
        os.remove(os.path.join(sample_dir, "mic.mic"))
    pickle.dump(
        {
            "microstructure": current_sample,
            "generation_method": current_mic_generator,
        },
        open(os.path.join(sample_dir, "mic.mic"), "wb"),
    )
    if print_out:
        print_output(os.path.join(sample_dir, "mic.mic"))
    # Saving the configuration for later use


def save_status(sample_dir, current_sample, current_mic_generator):
    """Save status with a minimal amount of information (time, total overlap and status)."""

    status_file_name = os.path.join(sample_dir, "status")
    with open(status_file_name, "w") as status:
        time_line = "Time: {0:.3f}s\n".format(current_mic_generator.time)
        overlap_line = "Overlap: {0:.3f}\n".format(current_sample.total_overlap)
        status_line = "Status: {0}\n".format(current_mic_generator.status)
        status.writelines(time_line)
        status.writelines(overlap_line)
        status.writelines(status_line)


def delete_screen(screen_dir):
    os.remove(os.path.join(screen_dir, "mic.screen"))
