"""Initialization module for the geommicgen module."""

import os

# pylint: disable=import-error
# import postproc.voronoimetrics.motion_analysis as motion_analysis
# import postproc.voronoimetrics.stat_analysis as stat_analysis

from geommicgen.micgenmethod.rand_seq_adsorption import RandomSequentialAdsorption
import geommicgen.iofuncs.printing as print_funcs

# from postproc.plotfuncs.plotting_functions import plot_particles

from geommicgen.postproc.mshgen.meshing_interface import (
    FEMMeshGenerator,
    RegularGridMeshGenerator,
)

from geommicgen.postproc.postproc import post_proc

import geommicgen.iofuncs.file_handling as fileio
from geommicgen.iofuncs.keywords import top_level_reader

from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from geommicgen.micgenmethod.thermostats import (
    IsokineticThermostat,
    MultiTemperatureIsokineticThermostat,
    MicroCanonicalEnsemble,
    BerendsenForceThermostat,
)

import geommicgen.micgenmethod.speed_up_schemes.MD_speed_up_schemes as MD_speed_up_schemes
import geommicgen.micgenmethod.speed_up_schemes.RSA_speed_up_schemes as RSA_speed_up_schemes

# from geommicgen.micgenmethod.speed_up_schemes import (
#     CellList,
#     VerletList,
#     VerletPartialUpdate,
#     Naive,
# )


def run_program():
    """Run program."""
    (
        input_file_path,
        input_file_dir,
        input_file_name,
        ext,
        previous_mic_path,
    ) = fileio.get_arguments_from_command_line()

    top_level_reader.read_input_file(input_file_path)
    # Create results directory
    # Generate corresponding mesh
    results_folder = fileio.create_design_point_results_directory(
        input_file_dir, input_file_name
    )
    fileio.RESULTS_FOLDER = results_folder
    fileio.copy_input_file(input_file_path, results_folder)
    
    if previous_mic_path is not None:
        print_funcs.SCREEN_DIR = results_folder
        print_funcs.print_initial_message(input_file_path)
        print_funcs.print_analysis_previous(previous_mic_path)
        # It is an action on a previously generated microstructure
        current_sample, current_mic_generator = fileio.load_previous_sample(
            previous_mic_path
        )
        rve_dims = current_sample.rve_dims
        mesh_generators = set()
        if "mesh_options" in top_level_reader.all_options:
            for disc_ext in top_level_reader.all_options["mesh_options"]:
                # For each file extension asked
                if disc_ext == "femsh":
                    femsh_options = top_level_reader.all_options["mesh_options"][
                        "femsh"
                    ]
                    mesh_generators.add(
                        FEMMeshGenerator(
                            femsh_options["mesh_size"],
                            femsh_options["element_type"],
                            rve_dims,
                        )
                    )
                elif disc_ext == "rgmsh":
                    rgmsh_options = top_level_reader.all_options["mesh_options"][
                        "rgmsh"
                    ]
                    for i_n_voxel_dims in rgmsh_options["n_voxels_dims"]:
                        mesh_generators.add(
                            RegularGridMeshGenerator(
                                i_n_voxel_dims,
                                rve_dims,
                                slice_dir=rgmsh_options.get("slice_dir", None),
                            )
                        )
                else:
                    raise ValueError(
                        "Specified mesh {0} is not supported.".format(disc_ext)
                    )
        print_funcs.print_output_header()
        post_proc(
            mesh_generators,
            current_sample,
            current_mic_generator,
            results_folder,
            top_level_reader.all_options["post_proc"],
        )

        # Initializing the mesh generators
    else:
        try:  
            n_dp_samples = top_level_reader.all_options["n_dp_samples"]
            _ = top_level_reader.all_options["problem_type"]
            mic_gen_descriptors = top_level_reader.all_options["mic_gen_descriptors"]
            mic_gen_parameters = top_level_reader.all_options["mic_gen_parameters"]
            post_proc_parameters = top_level_reader.all_options["post_proc"]
            rve_dims = mic_gen_parameters["rve_dimensions"]
            mic_gen_method = mic_gen_parameters["mic_gen_method"]
            # Mandatory top level parameters
        except KeyError:
            print("Mandatory parameter not supplied.")
            raise

        if n_dp_samples < 1 and isinstance(n_dp_samples, int):
            raise ValueError(
                "Number of samples must be a positve integer larger than 1."
            )

        for _ in range(n_dp_samples):
            sample_dir, sample_file_path = fileio.create_sample_results_directory(
                results_folder
            )
            fileio.SAMPLE_DIR = sample_dir
            # Producing the number of samples required

            print_funcs.SCREEN_DIR = sample_dir
            print_funcs.print_initial_message(input_file_path)
            # Printing initial message

            mesh_generators = set()
            if "mesh_options" in top_level_reader.all_options:
                for disc_ext in top_level_reader.all_options["mesh_options"]:
                    # For each file extension asked
                    if disc_ext == "femsh":
                        femsh_options = top_level_reader.all_options["mesh_options"][
                            "femsh"
                        ]
                        mesh_generators.add(
                            FEMMeshGenerator(
                                femsh_options["mesh_size"],
                                femsh_options["element_type"],
                                rve_dims,
                            )
                        )
                    elif disc_ext == "rgmsh":
                        rgmsh_options = top_level_reader.all_options["mesh_options"][
                            "rgmsh"
                        ]
                        for i_n_voxel_dims in rgmsh_options["n_voxels_dims"]:
                            print(rgmsh_options.get("slice_dir", None), "\n\n")
                            mesh_generators.add(
                                RegularGridMeshGenerator(
                                    i_n_voxel_dims,
                                    rve_dims,
                                    slice_dir=rgmsh_options.get("slice_dir", None),
                                )
                            )
                    else:
                        raise ValueError(
                            "Specified mesh {0} is not supported.".format(disc_ext)
                        )
            # Initializing the mesh generators

            current_sample = Microstructure(rve_dims)
            # Initializing the current sample
            
            if ext == ".mgsim":
                if mic_gen_method == "MD":
                    # Molecular dynamics simulation
                    md_kwargs_keys = {
                        "damping_coeff",
                        "particle_mass_opt",
                        "force_option",
                        "force_rescale",
                        "dt_adapt",
                        "offset",
                        "fixed_seed",
                        "initial_vel_coeff",
                        "final_overlap_check",
                        "sim_gif",
                    }
                    md_kwargs = {
                        key: value
                        for key, value in mic_gen_parameters.items()
                        if key in md_kwargs_keys
                    }
                    md_kwargs["sim_gif"] = post_proc_parameters.get("sim_gif", False)
                    #if md_kwargs["sim_gif"]:
                        #raise NotImplementedError("The simulation_gif option for MD simulations is not implemented yet.")
                    try: 
                        current_mic_generator = MolecularDynamicsSimulation(
                            mic_gen_parameters["max_residue_per_particle"],
                            mic_gen_parameters["max_step"],
                            mic_gen_parameters["max_steps_to_relax"],
                            mic_gen_parameters["dt"],
                            mic_gen_parameters["min_distance"],
                            mic_gen_parameters["type_initial_configuration"],
                            mic_gen_parameters["save_history"],
                            sample_dir,
                            **md_kwargs
                        )
                    except KeyError:
                        print("Missing mandatory parameter defining a MD simulation.")
                        raise

                    try:
                        if mic_gen_parameters.get("thermostat") == "isokinetic":
                            current_thermostat = IsokineticThermostat(
                                mic_gen_parameters["initial_temp"]
                            )
                        elif mic_gen_parameters.get("thermostat") == "multi_temperature":
                            if (
                                mic_gen_parameters.get("lowering_temp_criterion")
                                == "rolling_ave"
                            ):
                                kwargs = {
                                    "criterion": "rolling_ave",
                                    "average_window": mic_gen_parameters["average_window"],
                                }
                            elif (
                                mic_gen_parameters.get("lowering_temp_criterion")
                                == "ratio_in_out"
                            ):
                                kwargs = {
                                    "criterion": "ratio_in_out",
                                    "max_ratio_osc": mic_gen_parameters["max_ratio_osc"],
                                }
                            else:
                                kwargs = {
                                    "criterion": "original",
                                    "min_eq_steps_at_temp": mic_gen_parameters[
                                        "min_eq_steps_at_temp"
                                    ],
                                }
                            kwargs.update(
                                {"temp_low_ratio": mic_gen_parameters["temp_low_ratio"]}
                            )
                            current_thermostat = MultiTemperatureIsokineticThermostat(
                                mic_gen_parameters["initial_temp"], **kwargs
                            )
                        elif mic_gen_parameters.get("thermostat") == "micro_canonical":
                            current_thermostat = MicroCanonicalEnsemble()
                        elif mic_gen_parameters.get("thermostat") == "berendsen":
                            current_thermostat = BerendsenForceThermostat(
                                mic_gen_parameters["initial_temp"],
                                mic_gen_parameters["berendsen_coeff"],
                            )
                        else:
                            current_thermostat = MicroCanonicalEnsemble()
                    except KeyError:
                        print(
                            "Missing mandatory parameter defining the {0} thermostat.".format(
                                mic_gen_parameters["thermostat"]
                            )
                        )
                        raise

                    current_mic_generator.set_thermostat(current_thermostat)
                    # Adding a thermostat to the MD simulation


                    if mic_gen_parameters.get("speed_up_scheme") == "Cell" or mic_gen_parameters.get("speed_up_scheme") == None:
                        # CellList is the default
                        current_speed_up_scheme = MD_speed_up_schemes.CellList()
                    elif mic_gen_parameters.get("speed_up_scheme") == "Verlet":
                        current_speed_up_scheme = MD_speed_up_schemes.VerletList(
                            mic_gen_parameters["verlet_factor"]
                        )
                    elif mic_gen_parameters.get("speed_up_scheme") == "Verlet2":
                        current_speed_up_scheme = MD_speed_up_schemes.VerletPartialUpdate(
                            mic_gen_parameters["verlet_factor"]
                        )
                    elif mic_gen_parameters["speed_up_scheme"] == "Naive":
                        current_speed_up_scheme = MD_speed_up_schemes.Naive()
                    else:
                        raise ValueError(f"Unknown speed up scheme for MD simulation: '{mic_gen_parameters['speed_up_scheme']}'. ")

                    current_mic_generator.set_speed_up_scheme(current_speed_up_scheme)
                    # Adding a speed up scheme to the MD simulation
                elif mic_gen_method == "RSA":
                    # Random sequential adsorption method

                    # Uncomment following lines if kwargs from mic_gen_parameters are needed in the RSA simulation
                    #rsa_kwargs_keys = {"sim_gif"}
                    #rsa_kwargs = {
                        #key: value
                        #for key, value in mic_gen_parameters.items()
                        #if key in rsa_kwargs_keys
                    #}
                    rsa_kwargs={} # delete this line if the above lines are uncommented
                    rsa_kwargs["sim_gif"] = post_proc_parameters.get("sim_gif", False)
                    try:
                        current_mic_generator = RandomSequentialAdsorption(
                            mic_gen_parameters["max_step"],
                            mic_gen_parameters["min_distance"],
                            mic_gen_parameters["save_history"],
                            sample_dir,
                            **rsa_kwargs
                            )
                    except KeyError:
                        print("Missing mandatory parameter defining the RSA simulation.")
                        raise

                    if mic_gen_parameters.get("speed_up_scheme") == "Cell" or mic_gen_parameters.get("speed_up_scheme") == None:
                        current_speed_up_scheme = RSA_speed_up_schemes.CellList()
                    elif mic_gen_parameters["speed_up_scheme"] == "Naive":
                        current_speed_up_scheme = RSA_speed_up_schemes.Naive()
                    else:
                        raise ValueError(f"Unknown speed up scheme for RSA simulation: '{mic_gen_parameters['speed_up_scheme']}'. ")

                    current_mic_generator.set_speed_up_scheme(current_speed_up_scheme)
                    # Adding a speed up scheme to the RSA simulation

                else:
                    raise ValueError("Unknown microstructure generation method: {0}".format(mic_gen_method))
            else:
                raise ValueError("Unknown input file extension: {0}".format(ext))

            for phase_name, phase_descriptors in mic_gen_descriptors.items():
                current_phase = Phase(phase_name, phase_descriptors)
                current_sample.add_phase(current_phase)
                # Populating the microstructure sample with phases
            if current_sample.matrix_phase is None:
                raise ValueError("No matrix phase was specified.")

            try:
                current_mic_generator.generate_microstructure(current_sample)
            finally:
                # Use in data-driven framework
                if top_level_reader.all_options["save_min"]:
                    fileio.save_mic(sample_dir, current_sample, None)
                    fileio.save_status(
                        sample_dir, current_sample, current_mic_generator
                    )

                else:
                    fileio.save_mic(sample_dir, current_sample, current_mic_generator)
                # Saving the RVE properties

            try:
                times_dict = {}
                times_dict = post_proc(
                    mesh_generators,
                    current_sample,
                    current_mic_generator,
                    sample_dir,
                    top_level_reader.all_options["post_proc"],
                )

            finally:
                print_funcs.print_final_message(
                    current_mic_generator, mesh_generators, times_dict
                )
                if top_level_reader.all_options["save_min"]:
                    fileio.delete_screen(print_funcs.SCREEN_DIR)
