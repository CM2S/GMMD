import os
import sys
import pickle

from .printing import print_output


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
        results_folder = results_folder_old + "_" + str(i)
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
        results_folder = results_folder_old + "_" + str(i)
        # Creating a new folder name appending an integer to the name of the original
    os.makedirs(results_folder)
    # Creating the directory

    return results_folder


def get_arguments_from_command_line():
    if len(sys.argv) == 1:
        # No input file has been supplied
        raise ValueError("No input file was supplied.")
        # Exiting the script
    elif len(sys.argv) > 3:
        raise ValueError("Too many input files were supplied.")

    input_file_path = sys.argv[1]
    input_file_dir = os.path.dirname(sys.argv[1])
    input_file_name, ext = os.path.splitext(os.path.basename(sys.argv[1]))
    # Obtaining the directory and the name of the input file
    previous_mic_path = None
    if len(sys.argv) == 3:
        _, ext = os.path.splitext(os.path.basename(sys.argv[2]))
        if ext == ".mic":
            previous_mic_path = sys.argv[2]
        else:
            raise ValueError(
                "Wrong extension for the previous microstucutre file: {0}".format(ext)
            )
    return input_file_path, input_file_dir, input_file_name, ext, previous_mic_path


def load_previous_sample(previous_mic_path):
    info_previous_sample = pickle.load(open(previous_mic_path, "rb"))
    # No need to generate a new microstructure. Using a previous microstructure.
    current_sample = info_previous_sample["microstructure"]
    current_mic_generator = info_previous_sample["generation_method"]
    # Reconstructing the relevant Particle attributes that could not be pickled
    return current_sample, current_mic_generator


def save_mic(sample_file_path, current_sample, current_mic_generator):
    pickle.dump(
        {
            "microstructure": current_sample,
            "generation_method": current_mic_generator,
        },
        open(sample_file_path + ".mic", "wb"),
    )
    print_output(sample_file_path + ".mic")
    # Saving the configuration for later use
