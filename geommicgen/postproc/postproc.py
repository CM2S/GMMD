"""
Module containing the post processing: the discretisations, then the analyses.

The two halves are separate functions because they are entered from different places.
The discretisations run from a deck, and on their own through the meshing command; the
analyses run from a deck, and on their own through the analysis command, from the
microstructure file and the state of the run written beside it. `post_proc` is the
composition the deck runs.
"""

import contextlib
import os
import time
import traceback

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import geommicgen.iofuncs.printing as print_funcs
from geommicgen._optional import require_gmsh
from geommicgen.iofuncs.md_state import STATE_FILE_NAME
from geommicgen.pipeline import MESH_DIRECTORY
from geommicgen.postproc.options import ANALYSES, VORONOI_OPTION_NAMES
import geommicgen.postproc.voronoimetrics.motion_analysis as motion_analysis
import geommicgen.postproc.voronoimetrics.stat_analysis as stat_analysis
import geommicgen.postproc.voronoimetrics.voronoi_analysis as voronoi_analysis

from geommicgen.postproc.plotfuncs.plotting_functions import plot_particles

FINAL_CONFIG_STEP = "Generating final configuration for visualization"
MOTION_STEP = "Generating simulation plots"
VORONOI_STEP = "Voronoi analysis"
STAT_STEP = "Statistical analysis"
# Names the analyses are announced under, and the time each of them took reported
# under in the summary that closes a run


@contextlib.contextmanager
def announced(step, times, failures):
    """
    Announce an analysis, report how long it took, and record it if it fails.

    Parameters
    ----------
    step: str
        Name of the analysis.

    times: dict
        Dictionary the seconds it took are put in, under that name.

    failures: list
        List a failure is appended to, as *(name, error, traceback)*.
    """
    print_funcs.print_to_file(step)
    print_funcs.print_to_file("-" * 80 + "\n")
    start = time.time()
    try:
        yield
    except Exception as error:  # pylint: disable=broad-except
        failures.append((step, error, traceback.format_exc()))
        error.with_traceback(None)
        print_funcs.print_to_file(
            "\t\t- FAILED: {0}: {1}".format(type(error).__name__, error)
        )
    # An analysis that fails is recorded rather than raised, as a discretisation is,
    # so that the analyses after it, and the samples after it, still get their chance;
    # an error in one, or gmsh ending the process a view was drawn in, ended the batch.
    # The traceback is kept as text and taken off the error, which would otherwise
    # keep every frame down to the raise alive until the batch ends
    times[step] = time.time() - start
    print_funcs.print_to_file("Time ellapsed: {0:.3f}s\n".format(times[step]))
    # Every analysis opens with its name and closes with its time, where only the
    # final configuration did and the others went unreported


VORONOI_TYPES = ("standard", "set")
# The Voronoi diagrams the analysis knows how to compute. A name outside this tuple
# used to be found out deep inside the analysis, as an unbound variable; a weighted
# one was accepted and quietly computed the standard diagram, its implementation
# never having been written

VORONOI_OPTIONS = VORONOI_OPTION_NAMES
# The options handed on to the Voronoi analysis, when they are given

STAT_OPTIONS = tuple(i_name for i_name in ANALYSES if i_name.startswith("stat_"))
# The statistical analyses, each asked for by its own option

ANALYSIS_DIRECTORIES = {
    "motion_analysis": "motion_results",
    "voronoi_analysis": "voronoi_analysis_results",
    "statistical_analysis": "stat_analysis_results",
}
# The directory each analysis names for itself inside the sample, which is where what
# it wrote is to be found


def run_mesh_jobs(mesh_jobs, microstructure, sample_dir):
    """
    Run the discretisations asked for, reporting each one as it is attempted.

    Parameters
    ----------
    mesh_jobs: list
        The `.MeshJob` objects to run. Nothing is reported when there are none.

    microstructure: `.Microstructure`
        Microstructure to discretise.

    sample_dir: str
        Directory of the sample; the meshes go in a directory of their own inside it.
    """
    if not mesh_jobs:
        return

    print_funcs.print_to_file("Generating meshes")
    print_funcs.print_to_file("-" * 80 + "\n")
    for i_job in mesh_jobs:
        print_funcs.print_to_file("\t> {0}".format(i_job.description))
        i_job.run(
            microstructure,
            os.path.join(sample_dir, MESH_DIRECTORY),
            report=print_funcs.print_particle_progress,
        )
        for j_warning in i_job.mesher.warnings:
            print_funcs.print_to_file("\t\t- {0}".format(j_warning))
        if i_job.error is None:
            for j_file in i_job.files:
                print_funcs.print_to_file("\t\t- {0}".format(j_file))
        else:
            print_funcs.print_to_file("\t\t- FAILED: {0}".format(i_job.error))
        print_funcs.print_to_file("Time ellapsed: {0:.3f}s\n".format(i_job.time))
    # Each discretisation is attempted whatever became of the ones before it


def needs_gmsh(microstructure, state, options):
    """
    Say whether the analyses asked for reach a gmsh view.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be analysed.

    state: `.GenerationState`
        What the run that produced it recorded, or None.

    options: dict
        The analyses asked for.

    Returns
    -------
    bool
        True when one of them draws through gmsh.
    """
    if options.get("final_config", False) and microstructure.dim == 3:
        return True
    if (
        options.get("motion_analysis", False)
        and microstructure.dim == 2
        and state is not None
        and state.position_center_history is not None
    ):
        return True
    if (
        options.get("voronoi_analysis", False)
        and microstructure.dim == 3
        and (options.get("plot_voronoi", False) or options.get("plot_imts", False))
    ):
        return True
    # The final configuration of a three dimensional microstructure, the paths of the
    # particles of a two dimensional one, and the three dimensional Voronoi views are
    # the gmsh views; everything else is matplotlib, or no view at all

    return False


def check_analyses(microstructure, state, options):
    """
    Refuse an analysis that cannot be carried out, before anything is written.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be analysed.

    state: `.GenerationState`
        What the run that produced it recorded, or None when there is no record.

    options: dict
        The analyses asked for, keyed as the input data file keys them.

    Raises
    ------
    ValueError:
        If the motion analysis is asked for without a record of the run, or if the
        Voronoi diagram asked for is not one the analysis computes.

    MissingOptionalDependency:
        If an analysis asked for draws through gmsh and gmsh is not installed.
    """
    if options.get("motion_analysis", False) and state is None:
        raise ValueError(
            "The motion analysis plots what the run that produced the microstructure "
            "recorded, and there is no record of it: no {0} was found beside the "
            "microstructure file.".format(STATE_FILE_NAME)
        )
    if options.get("voronoi_analysis", False):
        voronoi_type = options.get("voronoi_type", "standard")
        if voronoi_type not in VORONOI_TYPES:
            raise ValueError(
                "The Voronoi diagram {0} is not one of the kinds computed: {1}.".format(
                    voronoi_type, ", ".join(VORONOI_TYPES)
                )
            )
    if needs_gmsh(microstructure, state, options):
        require_gmsh()
    # Asked before anything is written, so that a request that cannot be honoured
    # leaves nothing behind rather than half of it


def run_analyses(microstructure, state, sample_dir, options):
    """
    Run the analyses asked for on a microstructure, and on the run that produced it.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be analysed.

    state: `.GenerationState` or `.MolecularDynamicsSimulation`
        What the run that produced it recorded: the histories the motion analysis
        plots. None when there is no record, which only the motion analysis minds.

    sample_dir: str
        Directory the results are written in.

    options: dict
        The analyses asked for, keyed as the input data file keys them.

    Returns
    -------
    dict
        Dictionary of the form *{step: seconds}* for the steps that report a time.

    list
        The analyses that failed, as *(name, error, traceback)*; the others were still
        carried out.

    Raises
    ------
    ValueError:
        If an analysis is asked for that cannot be carried out; see `check_analyses`.
    """
    check_analyses(microstructure, state, options)
    os.makedirs(sample_dir, exist_ok=True)
    times = {}
    failures = []
    # The directory is made once it is known something will be written in it

    # Plotting final configuration
    # --------------------------------------------------------------------------------------
    if options.get("final_config", False):
        # Plot and save the final configuration
        with announced(FINAL_CONFIG_STEP, times, failures):
            plot_particles(
                microstructure.particles, microstructure.rve_dims, sample_dir
            )

    # Motion analysis
    # --------------------------------------------------------------------------------------
    if options.get("motion_analysis", False):
        with announced(MOTION_STEP, times, failures):
            histories = {
                "total_overlap_history": state.total_overlap_history,
                "max_residue": state.max_residue,
                "kinetic_energy_history": state.kinetic_energy_history,
                "temp_change_steps": state.thermostat.temp_change_steps,
                "temp_change": True,
                "overlap_ratio": state.thermostat.ratio,
                "len_sim": state.step,
                "thermic_energy_history": state.thermic_energy_history,
                "dt_history": state.all_dt,
            }
            if state.position_center_history is not None:
                histories["position_center_history"] = state.position_center_history
            else:
                print_funcs.print_to_file(
                    "\t\t- The run recorded no positions, so the paths are not plotted"
                )
            # The analysis plots the paths when it is given them, so a run that kept
            # no positions is simply not given any
            motion_analysis.do_motion_analysis(
                microstructure.particles,
                microstructure.rve_dims,
                sample_dir,
                **histories
            )

    # Voronoi analysis
    # --------------------------------------------------------------------------
    if options.get("voronoi_analysis", False):
        voronoi_kwargs = {
            i_option: options[i_option]
            for i_option in VORONOI_OPTIONS
            if i_option in options
        }
        with announced(VORONOI_STEP, times, failures):
            voronoi_analysis.do_voronoi_analysis(
                microstructure.particles,
                microstructure.rve_dims,
                sample_dir,
                **voronoi_kwargs
            )

    # Statistical analysis
    # --------------------------------------------------------------------------
    stat_options = {
        i_option for i_option in STAT_OPTIONS if options.get(i_option, False)
    }
    if stat_options:
        with announced(STAT_STEP, times, failures):
            stat_analysis.do_stat_analysis(
                microstructure, sample_dir, stat_options, seed=options.get("stat_seed")
            )

    return times, failures


def post_proc(
    mesh_jobs, current_sample, current_mic_generator, sample_dir, post_proc_opts
):
    """
    Do the post processing an input data file asks for: the meshes, then the analyses.

    Parameters
    ----------
    mesh_jobs: list
        The `.MeshJob` objects to run.

    current_sample: `.Microstructure`
        Microstructure the run produced.

    current_mic_generator: `.MolecularDynamicsSimulation`
        The run that produced it.

    sample_dir: str
        Directory of the sample.

    post_proc_opts: dict
        The analyses asked for.

    Returns
    -------
    dict
        Dictionary of the form *{step: seconds}* for the steps that report a time.

    list
        The analyses that failed, as *(name, error, traceback)*.
    """
    check_analyses(current_sample, current_mic_generator, post_proc_opts)
    run_mesh_jobs(mesh_jobs, current_sample, sample_dir)
    # The meshes come first, and the analyses are checked before them: an analysis
    # that cannot be carried out is found out at once rather than after the meshing,
    # and one that fails while running no longer costs the meshes

    return run_analyses(current_sample, current_mic_generator, sample_dir, post_proc_opts)
