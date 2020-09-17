import os
import shutil

import sys


def createResultsDirectory(particles, dp_dir, remesh=False):
    """
    Create the results directory.

    Parameters
    ----------
    particles: `.particle`
        Particles in the system.

    dp_dir: string
        Directory where the results are going to be stored.

    remesh: boolean, optional
        Signals if the program is currently being used for a remesh action.
    """
    Particle.file_name = "mic"
    # Defining the file name associated with this sampling. The filenames of the particles
    # are always prefixed by mic
    results_folder = os.path.join(dp_dir, Particle.file_name)
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
    if os.path.exists("input_data\\info_micro.p") and not remesh:
        shutil.copy(
            "input_data\\info_micro.p", os.path.join(results_folder, "info_micro.p")
        )
        # copying input file
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    # Saving the file path in the Particle class
