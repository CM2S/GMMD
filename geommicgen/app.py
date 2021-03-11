"""Initialization module for the geommicgen module."""

import os

# pylint: disable=import-error
# import postproc.voronoimetrics.motion_analysis as motion_analysis
# import postproc.voronoimetrics.stat_analysis as stat_analysis

import iofuncs.printing as print_funcs

# from postproc.plotfuncs.plotting_functions import plot_particles

from postproc.mshgen.meshing_interface import FEMMeshGenerator, RegularGridMeshGenerator

from postproc.postproc import post_proc

import iofuncs.file_handling as fileio
from iofuncs.keywords import top_level_reader

from microstructure.microstructure import Microstructure
from microstructure.phase import Phase
from micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from micgenmethod.thermostats import (
    IsokineticThermostat,
    MultiTemperatureIsokineticThermostat,
    MicroCanonicalEnsemble,
    BerendsenForceThermostat,
)
from micgenmethod.speed_up_schemes import CellList, VerletList, VerletList2, Naive


def run_program():
    """Run program."""
    (
        input_file_path,
        input_file_dir,
        input_file_name,
        ext,
        previous_mic_path,
    ) = fileio.get_arguments_from_command_line()

    top_level_reader.readInputFile(input_file_path)
    # Create results directory
    # Generate corresponding mesh
    results_folder = fileio.create_design_point_results_directory(
        input_file_dir, input_file_name
    )
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
        try:
            print_funcs.print_output_header()
            post_proc(
                mesh_generators,
                current_sample,
                current_mic_generator,
                results_folder,
                top_level_reader.all_options["post_proc"],
            )
        finally:

            # os.replace("temp.screen", os.path.join(results_folder, "mic.screen"))
            pass
            # Moving the screnn of this sample to the respective directory

        # Initializing the mesh generators
    else:
        try:
            n_dp_samples = top_level_reader.all_options["n_dp_samples"]
            _ = top_level_reader.all_options["problem_type"]
            mic_gen_descriptors = top_level_reader.all_options["mic_gen_descriptors"]
            mic_gen_parameters = top_level_reader.all_options["mic_gen_parameters"]
            rve_dims = mic_gen_parameters["rve_dimensions"]
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

            if ext == ".mdsim":
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
                }
                md_kwargs = {
                    key: value
                    for key, value in mic_gen_parameters.items()
                    if key in md_kwargs_keys
                }
                try:
                    current_mic_generator = MolecularDynamicsSimulation(
                        mic_gen_parameters["max_residue_per_particle"],
                        mic_gen_parameters["max_step"],
                        mic_gen_parameters["max_steps_to_relax"],
                        mic_gen_parameters["dt"],
                        mic_gen_parameters["min_distance"],
                        mic_gen_parameters["type_initial_configuration"],
                        mic_gen_parameters["save_history"],
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

                try:
                    if mic_gen_parameters.get("speed_up_scheme") == "Cell":
                        current_speed_up_scheme = CellList()
                    elif mic_gen_parameters.get("speed_up_scheme") == "Verlet":
                        current_speed_up_scheme = VerletList(
                            mic_gen_parameters["verlet_factor"]
                        )
                    elif mic_gen_parameters.get("speed_up_scheme") == "Verlet2":
                        current_speed_up_scheme = VerletList2(
                            mic_gen_parameters["verlet_factor"]
                        )
                    elif mic_gen_parameters["speed_up_scheme"] == "Naive":
                        current_speed_up_scheme = Naive()
                    else:
                        current_speed_up_scheme = CellList()
                except KeyError:
                    print(
                        "Missing mandatory parameter defining the"
                        + "{0} speed up scheme.".format(
                            mic_gen_parameters["speed_up_scheme"]
                        )
                    )
                    raise

                current_mic_generator.set_speed_up_scheme(current_speed_up_scheme)
                # Adding a speed up scheme to the MD simulation
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
                if top_level_reader.all_options["save_min"]:
                    fileio.save_mic(sample_file_path, current_sample, None)
                else:
                    fileio.save_mic(
                        sample_file_path, current_sample, current_mic_generator
                    )
                # Saving the RVE properties

            try:
                post_proc(
                    mesh_generators,
                    current_sample,
                    current_mic_generator,
                    sample_dir,
                    top_level_reader.all_options["post_proc"],
                )
                # for mesh_generator in mesh_generators:
                #     mesh_generator.generate_mesh(current_sample, sample_dir)
                #     # Generate corresponding mesh
                # if top_level_reader.all_options["final_config"]:
                #     # Plot and save the final configuration
                #     plot_particles(current_sample.particles, rve_dims, sample_dir)
                # if "motion_analysis" in top_level_reader.all_options:
                #     motion_analysis.doMotionAnalysis(
                #         current_sample.particles,
                #         rve_dims,
                #         sample_dir,
                #         position_center_history=current_mic_generator.position_center_history,
                #         total_overlap_history=current_mic_generator.total_overlap_history,
                #         max_residue=current_mic_generator.max_residue,
                #         kinetic_energy_history=current_mic_generator.kinetic_energy_history,
                #         temp_change_steps=current_mic_generator.thermostat.temp_change_steps,
                #         temp_change=True,
                #         overlap_ratio=current_mic_generator.thermostat.ratio,
                #         len_sim=current_mic_generator.step,
                #     )
                #     # Do analysis of the motion of the particles
                #
                # # Voronoi analysis
                # # --------------------------------------------------------------------------
                # if "voronoi_metrics" in top_level_reader.all_options:
                #     pass
                #     # FIXME: do_voronoi_analysis(particles, rve_dims, Particle.file_path)
                #
                #     # voronoi_type = options.get("voronoi_type", "standard")
                #     # do_voronoi_analysis(
                #     # particles, rve_dims, Particle.file_path, voronoi_type=voronoi_type
                #     # )
                #     # Do a voronoi analysis
                #
                # # Statistical analysis
                # # --------------------------------------------------------------------------
                # all_stat_options = {
                #     "stat_nearest_neighbor",
                #     "stat_ripleys_k",
                #     "stat_two_pt_corr",
                # }
                # stat_options_req = {
                #     i_stat_opt
                #     for i_stat_opt in all_stat_options
                #     if i_stat_opt in top_level_reader.all_options
                # }
                # if len(stat_options_req) > 0:
                #     stat_analysis.do_stat_analysis(
                #         current_sample, sample_dir, stat_options_req
                #     )
            finally:

                # os.replace("temp.screen", sample_file_path + ".screen")
                pass
                # Moving the screnn of this sample to the respective directory

    # Executing the script for microstructure generation
