"""Initialization module for the geommicgen module."""

import os

# pylint: disable=import-error
# import postproc.voronoimetrics.motion_analysis as motion_analysis
# import postproc.voronoimetrics.stat_analysis as stat_analysis

import geommicgen.iofuncs.printing as print_funcs

# from postproc.plotfuncs.plotting_functions import plot_particles

from geommicgen.meshing.from_deck import build_mesh_jobs

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
from geommicgen.micgenmethod.speed_up_schemes import (
    CellList,
    VerletList,
    VerletPartialUpdate,
    Naive,
)


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
        mesh_jobs = build_mesh_jobs(
            top_level_reader.all_options.get("mesh_options", {}), input_file_name
        )
        print_funcs.print_output_header()
        post_proc(
            mesh_jobs,
            current_sample,
            current_mic_generator,
            results_folder,
            top_level_reader.all_options["post_proc"],
        )
        if print_funcs.print_failed_jobs(mesh_jobs):
            raise SystemExit(1)
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

        fileio.PROVENANCE = {"source_deck": os.path.basename(input_file_path)}
        if mic_gen_parameters.get("fixed_seed") is not None:
            fileio.PROVENANCE["fixed_seed"] = mic_gen_parameters["fixed_seed"]
        # Carried by every microstructure file the run writes, so that a sample says
        # what produced it

        failed_jobs = []
        for _ in range(n_dp_samples):
            sample_dir, sample_file_path = fileio.create_sample_results_directory(
                results_folder
            )
            # Producing the number of samples required

            print_funcs.SCREEN_DIR = sample_dir
            print_funcs.print_initial_message(input_file_path)
            # Printing initial message

            mesh_jobs = build_mesh_jobs(
                top_level_reader.all_options.get("mesh_options", {}), input_file_name
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
                        current_speed_up_scheme = VerletPartialUpdate(
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
                # Use in data-driven framework
                if top_level_reader.all_options["save_min"]:
                    fileio.save_mic(sample_dir, current_sample, None)
                else:
                    fileio.save_mic(sample_dir, current_sample, current_mic_generator)
                fileio.save_status(sample_dir, current_sample, current_mic_generator)
                # Saving the RVE properties. The status is written here as well as
                # after the meshing, so that a generation that raised still leaves one

            try:
                times_dict = {}
                times_dict = post_proc(
                    mesh_jobs,
                    current_sample,
                    current_mic_generator,
                    sample_dir,
                    top_level_reader.all_options["post_proc"],
                )
            finally:
                print_funcs.print_final_message(
                    current_mic_generator, mesh_jobs, times_dict
                )
                fileio.save_status(
                    sample_dir, current_sample, current_mic_generator, mesh_jobs
                )
                if top_level_reader.all_options["save_min"]:
                    fileio.delete_screen(print_funcs.SCREEN_DIR)
                # Rewritten now that the discretisations have been attempted, so that
                # the sample records which of them were produced

            failed_jobs.extend(i_job for i_job in mesh_jobs if i_job.error)
            # Collected across the batch, so that one sample failing to mesh does not
            # cost the samples after it

        if print_funcs.print_failed_jobs(failed_jobs):
            raise SystemExit(1)
