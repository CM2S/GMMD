import os

from postproc.plotfuncs.plotting_functions import (
    plot_particles,
    plot_paths,
    plot_overlap_history,
    plot_kinetic_energy_history,
    plot_ratio_new_old_overlap,
)


def doMotionAnalysis(particles, rve_dims, sample_dir, **kwargs):

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
            kwargs["thermic_enegy_history"],
            motion_results_dir,
        )
        )
