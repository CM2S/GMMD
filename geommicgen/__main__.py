import sys
import os
from iofuncs.keywords import top_level_reader


def str2type(value_option):
    """Convert string containing a parameter value to the correct type."""
    if value_option == "True":
        return True
    elif value_option == "False":
        return False
    else:
        try:
            value_option = int(value_option)
            return value_option
        except ValueError:
            pass
        try:
            value_option = float(value_option)
            return value_option
        except ValueError:
            pass
        return value_option


def runProgram():

    if len(sys.argv) == 0:
        # No input file has been supplied
        print("No input file was supplied.")
        quit()
        # Exiting the script
    input_file_path = sys.argv[1]
    input_file_dir = os.path.dirname(sys.argv[1])
    input_file_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
    # Obtaining the directory and the name of the input file

    top_level_reader.readInputFile(input_file_path)

    print(keywords.Keyword.input_reader.all_options)
    #
    current_sim = Simulation(input_file_dir)

    current_sim.setOptionsSimulation(Keyword.input_reader.all_options)

    main(
        current_sim.dp_dir,
        current_sim.mic_gen_descriptors,
        current_sim.phase_types,
        current_sim.mic_gen_descriptors,
        current_sim.n_dp_samples,
        current_sim.problem_type,
        current_sim.mesh_options,
        current_sim.mesh_ext,
    )

    # main(dp_dir, mic_gen_descriptors, phase_types, mic_gen_parameters, n_dp_samples,
    #      problem_type, discret_spec_array, discret_file_ext)
    # Executing the script for microstructure generation


if __name__ == "__main__":
    runProgram()
