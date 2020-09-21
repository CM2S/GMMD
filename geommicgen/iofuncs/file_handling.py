import os


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


