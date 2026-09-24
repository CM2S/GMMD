"""Initialization module for the geommicgen module."""

import os

# pylint: disable=import-error
# import postproc.voronoimetrics.motion_analysis as motion_analysis
# import postproc.voronoimetrics.stat_analysis as stat_analysis

import geommicgen.iofuncs.printing as print_funcs

# from postproc.plotfuncs.plotting_functions import plot_particles

from geommicgen.pipeline import build_mesh_jobs

from geommicgen.postproc.postproc import post_proc

import geommicgen.iofuncs.file_handling as fileio
from geommicgen.iofuncs.keywords import top_level_reader

from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation


def run_program(argv=None):
    """
    Run the program.

    Parameters
    ----------
    argv: list
        Arguments, taken from the command line when they are not given.
    """
    print_funcs.log_to_terminal()
    arguments = fileio.parse_arguments(argv)
    input_file_path = arguments.input_file
    input_file_dir = os.path.dirname(input_file_path)
    input_file_name, ext = os.path.splitext(os.path.basename(input_file_path))
    # The results go in a directory named after the input data file, beside it, and
    # the extension says which generation method the file describes

    top_level_reader.read_input_file(input_file_path)
    results_folder = fileio.create_design_point_results_directory(
        input_file_dir, input_file_name
    )
    fileio.copy_input_file(input_file_path, results_folder)

    options = top_level_reader.all_options
    mic_gen_parameters = options.get("mic_gen_parameters", {})
    missing = [
        i_keyword
        for i_keyword, i_given in (
            ("N_DP_Samples", "n_dp_samples" in options),
            ("RVE_Dimensions", "rve_dimensions" in mic_gen_parameters),
            ("Mic_Gen_Descriptors", "mic_gen_descriptors" in options),
        )
        if not i_given
    ]
    if missing:
        raise ValueError(
            "The input data file does not give {0}, which a generation "
            "needs.".format(", ".join(missing))
        )
    n_dp_samples = options["n_dp_samples"]
    mic_gen_descriptors = options["mic_gen_descriptors"]
    rve_dims = mic_gen_parameters["rve_dimensions"]
    # Named in the spelling of the input data file, all of the missing ones at
    # once, before anything is generated; the parameters of the method are
    # checked the same way when it is built

    if n_dp_samples < 1 or not isinstance(n_dp_samples, int):
        raise ValueError("Number of samples must be a positive integer.")

    provenance = {"source_deck": os.path.basename(input_file_path)}
    fixed_seed = mic_gen_parameters.get("fixed_seed")
    # Carried by every microstructure file the run writes, so that a sample says
    # what produced it

    failed_jobs = []
    failed_samples = []
    for i_sample in range(n_dp_samples):
        if fixed_seed is not None:
            mic_gen_parameters["fixed_seed"] = fixed_seed + i_sample
            provenance["fixed_seed"] = fixed_seed + i_sample
        # Each sample of a seeded run gets a seed of its own, the deck's plus its
        # index, so the samples differ from each other and each is the same in
        # every run -- one seed for all of them made every sample the same
        sample_dir = fileio.create_sample_results_directory(results_folder)
        # Producing the number of samples required

        print_funcs.screen_to(sample_dir)
        print_funcs.print_initial_message(input_file_path)
        # Printing initial message

        mesh_jobs = build_mesh_jobs(
            top_level_reader.all_options.get("mesh_options", {}), input_file_name
        )
        # Initializing the mesh generators

        current_sample = Microstructure.from_descriptors(
            rve_dims, mic_gen_descriptors
        )
        # A microstructure of the phases the input data file describes, built anew
        # for each sample, since the generation fills it with particles

        if ext == ".mdsim":
            current_mic_generator = MolecularDynamicsSimulation.from_options(
                mic_gen_parameters
            )
            # The method builds itself from the generation parameters, thermostat
            # and speed up scheme included, and refuses a name it does not know
        else:
            raise ValueError("Unknown input file extension: {0}".format(ext))

        try:
            current_mic_generator.generate_microstructure(current_sample)
        finally:
            # Use in data-driven framework
            fileio.save_mic(
                sample_dir,
                current_sample,
                (
                    None
                    if top_level_reader.all_options["save_min"]
                    else current_mic_generator
                ),
                provenance=provenance,
            )
            fileio.save_status(sample_dir, current_sample, current_mic_generator)
            # Saving the RVE properties. The status is written here as well as
            # after the meshing, so that a generation that raised still leaves one

        try:
            times_dict = {}
            if current_mic_generator.status:
                times_dict = post_proc(
                    mesh_jobs,
                    current_sample,
                    current_mic_generator,
                    sample_dir,
                    top_level_reader.all_options["post_proc"],
                )
            else:
                failed_samples.append(
                    (i_sample, current_sample.total_overlap, current_mic_generator.max_residue)
                )
                print_funcs.print_failed_sample(
                    current_sample.total_overlap, current_mic_generator.max_residue
                )
            # A sample whose particles still overlap is not meshed or analysed: a mesh
            # of it is a wrong input to a solver, and the run says so instead
        finally:
            print_funcs.print_final_message(
                print_funcs.step_times(current_mic_generator, mesh_jobs, times_dict)
            )
            fileio.save_status(
                sample_dir, current_sample, current_mic_generator, mesh_jobs
            )
            if top_level_reader.all_options["save_min"]:
                fileio.delete_screen(sample_dir)
            print_funcs.screen_to(None)
            # Rewritten now that the discretisations have been attempted, so that
            # the sample records which of them were produced. The screen file is
            # then let go of, so that the summary of the batch that follows the
            # loop is not written into the last sample

        failed_jobs.extend(i_job for i_job in mesh_jobs if i_job.error)
        # Collected across the batch, so that one sample failing does not cost the
        # samples after it

    failed = print_funcs.print_failed_samples(failed_samples)
    failed = print_funcs.print_failed_jobs(failed_jobs) or failed
    if failed:
        raise SystemExit(1)
    # Both are reported, whichever happened, and either fails the run
