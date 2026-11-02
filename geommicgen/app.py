"""Initialization module for the geommicgen module."""

import os
import sys

# pylint: disable=import-error
# import postproc.voronoimetrics.motion_analysis as motion_analysis
# import postproc.voronoimetrics.stat_analysis as stat_analysis

import geommicgen.iofuncs.printing as print_funcs

# from postproc.plotfuncs.plotting_functions import plot_particles

from geommicgen.pipeline import build_mesh_jobs

from geommicgen.postproc.postproc import post_proc

import geommicgen.iofuncs.file_handling as fileio
from geommicgen.config.base import ConfigError
from geommicgen.config.run import read_config
from geommicgen.iofuncs.convert_deck import convert_deck

from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation

GENERATION_METHODS = {"molecular_dynamics": MolecularDynamicsSimulation}
# The class of each method a generation can be asked to use, under its name in the file

def read_input(path):
    """
    Read the run an input file, or a deck of the keyword format, asks for.

    A deck is laid out as an input file and validated as one, and said to be of a
    format that is on its way out.

    Parameters
    ----------
    path: str
        Path of the file.

    Returns
    -------
    `.RunConfig`
        The run.

    Raises
    ------
    ConfigError:
        If the file is of neither kind, or does not validate.
    """
    reader = INPUT_READERS.get(os.path.splitext(path)[1])
    if reader is None:
        raise ConfigError(
            "{0} is neither an input file (.yaml) nor a deck (.mdsim).".format(
                os.path.basename(path)
            )
        )

    return reader(path)


def read_deck(path):
    """Read a deck as the input file it lays out as, saying the format is going."""
    print(
        "geommicgen: {0} is a deck of the keyword format, which is read for now; "
        "geommicgen-convert-deck turns it into an input file.".format(
            os.path.basename(path)
        ),
        file=sys.stderr,
    )

    return convert_deck(path)[1]


INPUT_READERS = {".yaml": read_config, ".yml": read_config, ".mdsim": read_deck}
# What reads a file, by its extension


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
    input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    # The results go in a directory named after the input file, beside it

    try:
        run = read_input(input_file_path)
    except (OSError, ValueError) as error:
        print("geommicgen: {0}".format(error), file=sys.stderr)
        raise SystemExit(2) from None
    # The whole of the file is validated before anything is written, and what is
    # wrong with it said at once, as a command line that cannot be carried out is
    results_folder = fileio.create_design_point_results_directory(
        input_file_dir, input_file_name
    )
    fileio.copy_input_file(input_file_path, results_folder)

    options = run.to_options()
    mic_gen_descriptors = options["mic_gen_descriptors"]
    rve_dims = options["mic_gen_parameters"]["rve_dimensions"]
    generation_method = GENERATION_METHODS[run.generation.method]
    # The options the rest of the package is built from, as the reader of decks gave
    # them

    provenance = {"input_file": os.path.basename(input_file_path)}
    # Carried by every microstructure file the run writes, so that a sample says
    # what produced it: with the commit of GMMD and the versions of the libraries,
    # which the file adds, the input file is the whole of the run

    failed_jobs = []
    failed_samples = []
    failed_analyses = []
    for i_sample in range(run.samples):
        mic_gen_parameters = dict(options["mic_gen_parameters"])
        if run.seed is not None:
            mic_gen_parameters["fixed_seed"] = run.seed + i_sample
            provenance["fixed_seed"] = run.seed + i_sample
        # Each sample of a seeded run gets a seed of its own, the deck's plus its
        # index, so the samples differ from each other and each is the same in
        # every run -- one seed for all of them made every sample the same
        sample_dir = fileio.create_sample_results_directory(results_folder)
        # Producing the number of samples required

        print_funcs.screen_to(sample_dir)
        print_funcs.print_initial_message(input_file_path)
        # Printing initial message

        mesh_jobs = build_mesh_jobs(
            options.get("mesh_options", {}), input_file_name
        )
        # Initializing the mesh generators

        current_sample = Microstructure.from_descriptors(
            rve_dims, mic_gen_descriptors
        )
        # A microstructure of the phases the input data file describes, built anew
        # for each sample, since the generation fills it with particles

        current_mic_generator = generation_method.from_options(mic_gen_parameters)
        # The method the file names builds itself from the generation parameters,
        # thermostat and speed up scheme included

        try:
            current_mic_generator.generate_microstructure(current_sample)
        finally:
            # Use in data-driven framework
            fileio.save_mic(
                sample_dir,
                current_sample,
                (
                    None
                    if options["save_min"]
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
                times_dict, analysis_failures = post_proc(
                    mesh_jobs,
                    current_sample,
                    current_mic_generator,
                    sample_dir,
                    options["post_proc"],
                )
                failed_analyses.extend(
                    (i_sample,) + i_failure for i_failure in analysis_failures
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
            if options["save_min"]:
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
    failed = print_funcs.print_failed_analyses(failed_analyses) or failed
    if failed:
        raise SystemExit(1)
    # Every kind is reported, whichever happened, and any of them fails the run
