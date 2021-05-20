"""Module containing the main function relative to post processsing."""

import time

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import iofuncs.printing as print_funcs
import postproc.voronoimetrics.motion_analysis as motion_analysis
import postproc.voronoimetrics.stat_analysis as stat_analysis
import postproc.voronoimetrics.voronoi_analysis as voronoi_analysis

from postproc.plotfuncs.plotting_functions import plot_particles


def post_proc(
    mesh_generators, current_sample, current_mic_generator, sample_dir, post_proc_opts
):
    """Do the post processing, such as meshing and statistical analysis."""
    dict_times = {}
    # Plotting final configuration
    # --------------------------------------------------------------------------------------
    if post_proc_opts.get("final_config", False):
        # Plot and save the final configuration
        print_funcs.print_to_file("Generating final configuration for visualization")
        print_funcs.print_to_file("-" * 80 + "\n")
        start = time.time()
        plot_particles(current_sample.particles, current_sample.rve_dims, sample_dir)
        plot_particles_time = time.time() - start
        print_funcs.print_to_file(
            "Time ellapsed: {0:.3f}s\n".format(plot_particles_time)
        )
        dict_times[
            "Generating final configuration for visualization"
        ] = plot_particles_time

    # Generating meshes
    # --------------------------------------------------------------------------------------
    if mesh_generators:
        print_funcs.print_to_file("Generating meshes")
        print_funcs.print_to_file("-" * 80 + "\n")
        for mesh_generator in mesh_generators:
            mesh_generator.generate_mesh(current_sample, sample_dir)
        # Generate corresponding mesh

    # Motion analysis
    # --------------------------------------------------------------------------------------
    if post_proc_opts.get("motion_analysis", False):
        print_funcs.print_to_file("Generating simulation plots")
        print_funcs.print_to_file("-" * 80 + "\n")
        motion_analysis.do_motion_analysis(
            current_sample.particles,
            current_sample.rve_dims,
            sample_dir,
            position_center_history=current_mic_generator.position_center_history,
            total_overlap_history=current_mic_generator.total_overlap_history,
            max_residue=current_mic_generator.max_residue,
            kinetic_energy_history=current_mic_generator.kinetic_energy_history,
            temp_change_steps=current_mic_generator.thermostat.temp_change_steps,
            temp_change=True,
            overlap_ratio=current_mic_generator.thermostat.ratio,
            len_sim=current_mic_generator.step,
            thermic_energy_history=current_mic_generator.thermic_energy_history,
            dt_history=current_mic_generator.all_dt,
        )
        # Do analysis of the motion of the particles

    # Voronoi analysis
    # --------------------------------------------------------------------------
    if post_proc_opts.get("voronoi_analysis", False):
        print_funcs.print_to_file("Voronoi analysis")
        print_funcs.print_to_file("-" * 80 + "\n")

        all_voronoi_kwargs_options = {
            "n_surf_points",
            "plot_voronoi",
            "plot_imts",
            "voronoi_type",
        }
        voronoi_kwargs = {
            i_vor_opt: post_proc_opts[i_vor_opt]
            for i_vor_opt in all_voronoi_kwargs_options
            if i_vor_opt in post_proc_opts
        }
        voronoi_analysis.do_voronoi_analysis(
            current_sample.particles,
            current_sample.rve_dims,
            sample_dir,
            **voronoi_kwargs
        )
        # Do a voronoi analysis

    # Statistical analysis
    # --------------------------------------------------------------------------
    all_stat_options = {
        "stat_nearest_neighbor",
        "stat_ripleys_k",
        "stat_two_pt_corr",
    }
    stat_options_req = {
        i_stat_opt
        for i_stat_opt in all_stat_options
        if post_proc_opts.get(i_stat_opt, False)
    }
    if stat_options_req:
        print_funcs.print_to_file("Statistical analysis")
        print_funcs.print_to_file("-" * 80 + "\n")
        stat_analysis.do_stat_analysis(current_sample, sample_dir, stat_options_req)

    return dict_times
