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

    if options.get("remesh"):
        # It is a remesh action
        current_RVE = pickle.load(open(options["dir_previous_mic"], "rb"))
        # No need to generate a new microstructure. Using a previous microstructure.
        particles, rve_dims = current_RVE.useThisRVE(dp_dir)
        # Reconstructing the relevant Particle attributes that could not be pickled
        createResultsDirectory(particles, dp_dir, remesh=True)
        # Create results directory
        for disc_ext in discret_file_ext:
            # For each file extension asked
            generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
            # Generate corresponding mesh
        motion_analysis = options.get("motion_analysis", False)
        if motion_analysis:
            doMotionAnalysis(particles, rve_dims, Particle.file_path)
            # Do analysis of the motion of the particles
    else:
        # Generating samples of microstructures and meshing
        for i_sample in range(n_samples):
            # Producing the number of samples required
            print_funcs.printInitialMessage()
            # Printing initial message
            rve_dims = options.get("rve_dims")
            save_history = options.get("save_history", True)
            voronoi_analysis = options.get("voronoi_analysis", False)
            motion_analysis = options.get("motion_analysis", False)
            type_init_conf = options.get("type_initial_configuration", "random")
            max_residue_per_particle = options.get("max_residue_per_particle", 0)
            max_step = options.get("max_step", 1)
            # Collecting options
            start = time.time()
            # Keeping track of the simulation time
            particles = particleGeneration(
                descriptors,
                phase_types,
                rve_dims,
                problem_type,
                dp_dir,
                type_init_conf=type_init_conf,
                save_history=save_history,
            )
            # Generating the list of particles from the geometrical descriptors
            run(particles, max_residue_per_particle, max_step, options)
            # Running the molecular dynamics simulation
            end = time.time()
            Particle.time = end - start
            print_funcs.printFinalMessage(
                Particle.time,
                Particle.total_overlap,
                len(Particle.total_overlap_history),
                i_sample + 1,
                Particle.max_residue,
            )
            # Time spent on microstructure generation
            current_RVE = RVE(particles, rve_dims)
            # Saving the RVE properties in an RVE object
            pickle.dump(current_RVE, open(Particle.file_path + ".p", "wb"))
            # Saving the configuration for later use
            for disc_ext in discret_file_ext:
                # For each file extension asked
                generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
                # Generate corresponding mesh
            if motion_analysis:
                doMotionAnalysis(particles, rve_dims, Particle.file_path)
                # Do analysis of the motion of the particles
            if voronoi_analysis:
                voronoi_type = options.get("voronoi_type", "standard")
                doVoronoiAnalysis(
                    particles, rve_dims, Particle.file_path, voronoi_type=voronoi_type
                )
                # Do a voronoi analysis
            os.replace("temp.screen", Particle.file_path + ".screen")
            # Moving the screnn of this sample to the respective directory
            Particle.resetRVE()
            # Clearing the properties of the simulation box

    # print(keywords.Keyword.input_reader.all_options)
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
