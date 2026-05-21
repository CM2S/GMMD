"""Module for motion analysis for MD simulation."""

import os

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
from geommicgen.postproc.plotfuncs.plotting_functions import (
    plot_overlap_history,
    plot_kinetic_energy_history,
    plot_delta_t_history,
    plot_paths,
)


def do_motion_analysis(particles, rve_dims, sample_dir, **kwargs):

    motion_results_dir = os.path.join(sample_dir, "motion_results")
    os.makedirs(motion_results_dir)
    if "position_center_history" in kwargs:
        plot_paths(
            particles, rve_dims, kwargs["position_center_history"], motion_results_dir
        )

    if "total_overlap_history" in kwargs and "max_residue" in kwargs:
        plot_overlap_history(
            kwargs.pop("total_overlap_history"),
            kwargs.pop("max_residue"),
            motion_results_dir,
            **kwargs
        )

    if "kinetic_energy_history" in kwargs:
        plot_kinetic_energy_history(
            kwargs["kinetic_energy_history"],
            kwargs["thermic_energy_history"],
            motion_results_dir,
        )

    if "dt_history" in kwargs:
        plot_delta_t_history(
            kwargs["dt_history"],
            motion_results_dir,
        )
