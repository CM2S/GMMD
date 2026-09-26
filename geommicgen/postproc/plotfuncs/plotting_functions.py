# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import numpy as np

import scipy.integrate as integrate

import contextlib
import copy
import os


from PIL import Image


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# Simple math tools
# Finite element mesh conversor to LINKS


from geommicgen.microstructure.particleclasses import Ellipse, Particle

from geommicgen.meshing.gmsh_mesher import GmshMesher, gmsh_session
from geommicgen.meshing.images import periodic_images
from geommicgen.microstructure.microstructure import unit_scale
import geommicgen.iofuncs.printing as print_funcs

latex_textwidth = 5.92  # in = 496pt
latex_textheigth = 9.63  # in = 674pt


def _adjust_bounds(ax, points):
    ptp_bound = np.ptp(points, axis=0)
    ax.set_xlim(
        points[:, 0].min() - 0.1 * ptp_bound[0], points[:, 0].max() + 0.1 * ptp_bound[0]
    )
    ax.set_ylim(
        points[:, 1].min() - 0.1 * ptp_bound[1], points[:, 1].max() + 0.1 * ptp_bound[1]
    )


def create_figure(
    nrows: int = 2,
    ncols: int = 1,
    nrows_sub: int = 1,
    ncols_sub: int = 1,
    sharey: bool = False,
    sharex: bool = False,
    **kwargs
):
    """Create a matplotlib figure for a a4 report latex page.

    The figure is created with a size such that it fits in a4 report latex page, in an array

    *nrows* rows and *ncols* columns.

    Parameters
    ----------
    nrows: optional, int
        Number of rows in the a4 page.

    """
    # Set LaTeX option
    LaTeX_option = 1
    # Set LaTeX font
    if LaTeX_option == 1:
        # Default LaTeX Computer Modern Roman
        plt.rc("text", usetex=True)
        plt.rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
    else:
        # LaTeX Fourier
        plt.rc("text", usetex=True)
        plt.rc(
            "text.latex",
            preamble=r"\usepackage[widespace]{fourier} \usepackage{amsmath} \usepackage{amssymb}",
        )
    #
    if "w" not in kwargs:
        w = (latex_textwidth - 0.2 * (ncols - 1)) / ncols
    else:
        w = kwargs["w"]
    if "h" not in kwargs:
        h = (latex_textheigth - 1.63 * (nrows - 1)) / nrows
    else:
        h = kwargs["h"]
    fig, axs = plt.subplots(
        figsize=(w, h), nrows=nrows_sub, ncols=ncols_sub, sharey=sharey, sharex=sharex
    )

    #     # Set axes labels
    #     if x_label != None:
    #         axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    #     if y_label != None:
    #         axes.set_ylabel(y_label, fontsize=12, labelpad=10)
    # 15h56
    #     # Configure ticks appearance
    #     axes.tick_params(which='major', width=1.0, length=10, labelcolor='0.0', labelsize=12)
    #     axes.tick_params(which='minor', width=1.0, length=5, labelsize=12)
    #     # Configure grid
    #     axes.grid(linestyle='-', linewidth=0.5, color='0.5', zorder=0)
    # 15h56
    #     # Set line width
    #     line_width = 2
    #     # Set marker size and frequency
    #     marker_size = 5

    return [fig, axs, (w, h)]


def create_legend(artists, labels, axes, to_fig=False, fig_h=4, ncols=3):
    if to_fig:
        lw_common = axes.spines["bottom"].get_linewidth()
        fig = plt.gcf()
        legend = fig.legend(
            handles=artists,
            labels=labels,
            bbox_to_anchor=(0.0, 0.9, 1.0, 0.102),
            loc="lower center",
            ncol=ncols,
            mode="tight",
            borderaxespad=0.0,
            fontsize=12,
            bbox_transform=plt.gcf().transFigure,
        )

        frame = legend.get_frame()

        frame.set_linewidth(lw_common)
        frame.set_edgecolor("k")
        plt.tight_layout(rect=(0, 0, 1, 0.9))
    else:
        lw_common = axes.spines["bottom"].get_linewidth()
        plt.legend(
            handles=artists,
            labels=labels,
            bbox_to_anchor=(0.0, 1 + 4 * 0.02 / fig_h, 1.0, 0.102),
            loc="lower center",
            ncol=ncols,
            mode="tight",
            borderaxespad=0.0,
            fontsize=12,
        )

        legend = axes.get_legend()
        frame = legend.get_frame()

        frame.set_linewidth(lw_common)
        frame.set_edgecolor("k")
    return legend


def set_style(artists, ax, style):

    if style == "divergent":
        colors = [
            cm.RdBu(level) for level in np.linspace(0, 1, len(artists), endpoint=True)
        ]
    if style == "qualitative":
        color_scheme = np.array(
            [
                (68 / 255, 119 / 255, 170 / 255, 1),
                (102 / 255, 204 / 255, 238 / 255, 1),
                (34 / 255, 136 / 255, 51 / 255, 1),
                (204 / 255, 187 / 255, 68 / 255, 1),
                (238 / 255, 102 / 255, 119 / 255, 1),
                (170 / 255, 51 / 255, 119 / 255, 1),
                (187 / 255, 187 / 255, 187 / 255, 1),
            ]
        )
        colors = color_scheme[np.array(np.linspace(0, 6, len(artists)), dtype=int)]
    if style == "qualitative_pairs":
        color_scheme = np.array(
            [
                (119 / 255, 170 / 255, 221 / 255, 1),
                (153 / 255, 221 / 255, 255 / 255, 1),
                (170 / 255, 170 / 255, 0 / 255, 1),
                (238 / 255, 221 / 255, 136 / 255, 1),
                (238 / 255, 136 / 255, 102 / 255, 1),
                (255 / 255, 170 / 255, 187 / 255, 1),
            ]
        )
        colors = color_scheme[np.array(np.linspace(0, 5, len(artists)), dtype=int)]
    for ind, artist in enumerate(artists):
        artist.set_color(colors[ind])


def generate_colors(n_colors):
    """Generate a color for each pahse."""
    colors_def = [
        (68 / 255, 119 / 255, 170 / 255, 1),
        (102 / 255, 204 / 255, 238 / 255, 1),
        (34 / 255, 136 / 255, 51 / 255, 1),
        (204 / 255, 187 / 255, 68 / 255, 1),
        (238 / 255, 102 / 255, 119 / 255, 1),
        (170 / 255, 51 / 255, 119 / 255, 1),
        (187 / 255, 187 / 255, 187 / 255, 1),
    ]
    return colors_def[0 : n_colors + 1]


def plot_particles(particles, rve_dims, sample_dir, **kwargs):
    """Plot the particles."""
    if len(rve_dims) == 2:
        plot_particles_2d(particles, rve_dims, sample_dir)
    elif len(rve_dims) == 3:
        # plot_particles_3d_one_by_one(particles, rve_dims, sample_dir)
        plot_particles_3d(particles, rve_dims, sample_dir)


def plot_particles_2d(particles, rve_dims, sample_dir, **kwargs):
    """Plot 2D particles."""
    if "ax" in kwargs:
        ax = kwargs["ax"]
        plt.sca(ax)
    else:
        _ = plt.figure()
        ax = plt.gca()
    # Axes where the plot will be drawn

    if rve_dims[0] == rve_dims[1]:
        ax.axis("square")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(0, rve_dims[1])
    ax.set_xlim(0, rve_dims[0])
    # Setting the correct proportions

    phases = list({i_particle.phase for i_particle in particles})
    colors = generate_colors(len(phases))
    phase_colors = dict(zip(phases, colors))

    for i_particle in particles:
        for j_dim, k_dim in [
            (j_dim, k_dim) for j_dim in range(-1, 2) for k_dim in range(-1, 2)
        ]:
            ellip = mpatches.Ellipse(
                i_particle.position_center
                + np.array(rve_dims) * np.array([1 * j_dim, 1 * k_dim]),
                i_particle.major_axis,
                i_particle.minor_axis,
                angle=180 / np.pi * i_particle.angle,
                alpha=0.8,
                edgecolor=None,
                facecolor=phase_colors[i_particle.phase],
            )
            ax.add_artist(ellip)
            # plt.annotate(str(k), tuple(i_particle.position_center))

    if kwargs.get("save", True):
        plt.savefig(os.path.join(sample_dir, "final_config.pdf"), bbox_inches="tight")
        print_funcs.print_to_file(
            "\t\t- {0}".format(os.path.join(sample_dir, "final_config.pdf"))
        )

    if kwargs.get("show", False):
        plt.show()


VIEW_ELEMENTS = {2: "tri3", 3: "tetra4"}
# The element a view is built with, which is only ever a choice of order. A view is a
# picture: it is written as a surface and looked at, never solved, so a second order
# element buys nothing and costs about four times the nodes.

MESH_SIZE_VORONOI = 0.03
MESH_SIZE_VORONOI_IMTS = 0.1
# Largest element of the views of the Voronoi cells. The one that carries the Minkowski
# tensors covers the whole of a three by three block of images, so it is coarser.


def at_unit_scale(particles, rve_dims):
    """
    Give copies of particles, and the dimensions of their RVE, brought to unit scale.

    A view is built from them as the mesher builds its model, so that the tolerances of
    OpenCASCADE and gmsh, which are lengths, mean the same in any units; `gmsh_view`
    writes its files back in the user's.

    Parameters
    ----------
    particles: list(`.Particle`)
        Particles of the view. They are copied, not changed.

    rve_dims: list(float)
        Dimensions of the microstructure in each spatial direction.

    Returns
    -------
    tuple
        The copies, the dimensions, and the factor, from `unit_scale`, they were
        multiplied by.
    """
    scale = unit_scale(rve_dims)
    copies = [copy.deepcopy(i_particle) for i_particle in particles]
    for i_copy in copies:
        i_copy.rescale(scale)

    return copies, [i_dim * scale for i_dim in rve_dims], scale


@contextlib.contextmanager
def gmsh_view(name, mesh_size, dim=3, scale=1.0):
    """
    Open a gmsh session set up to build a view of the particles, and close it after.

    Parameters
    ----------
    name: str
        Name given to the model.

    mesh_size: float
        Largest element size, in the units the view is built in.

    dim: {2, 3}
        Number of spatial dimensions of the microstructure.

    scale: float
        Factor the view is built at, from `at_unit_scale`. The files the session
        writes are divided by it, back into the units of the microstructure.

    Yields
    ------
    tuple
        The gmsh module, its model and its geometry kernel.
    """
    with gmsh_session() as gmsh:
        GmshMesher(
            mesh_size=mesh_size, element_type=VIEW_ELEMENTS[dim]
        ).set_options(gmsh)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        # The size the session was asked for reaches gmsh here. The options a mesher
        # sets are the ones that produce its element; the size is set where that mesher
        # builds its model, which a view never does, so asking for one used to do
        # nothing and every view set it again itself
        gmsh.option.setNumber("Mesh.ScalingFactor", 1 / scale)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        # The factor is applied to the nodes that are written and not to the model. The
        # entities a file of version 4 lists beside the nodes were written unscaled, or
        # scaled about their own centres, in neither unit; version 2.2 lists none, and
        # a viewer reads the nodes alone. A view was built in the user's units, where
        # OpenCASCADE refuses an edge shorter than 1e-7 and its booleans lose pieces: a
        # micrometre RVE gave an empty view, without an error. The session meshes on
        # one thread, as the mesher does, so a view is the same from one run to the next
        model = gmsh.model
        model.add(name)

        yield gmsh, model, model.occ
    # The session is closed however the block ends. Leaving it open after a failure
    # would leave the next view building its model inside the failed one


def add_particles_to_view(factory, model, particles, rve_dims, add_images=True,
                          report=None):
    """
    Add particles, and the periodic images that reach the RVE, to an open model.

    Parameters
    ----------
    factory: module
        The geometry kernel of the open gmsh session.

    model: module
        The model of the open gmsh session.

    particles: list
        Particles to be added.

    rve_dims: list(float)
        Dimensions of the microstructure in each spatial direction.

    add_images: bool
        Whether the periodic images are wanted.

    report: callable
        Called with the index of the particle that was added and the total number.

    Returns
    -------
    tuple
        The tags of everything that was added, and the *(dimension, tag)* pairs of each
        phase.
    """
    particle_tags = []
    phase_dim_tag = {i_particle.phase: [] for i_particle in particles}
    for i_particle_ind, i_particle in enumerate(particles):
        for j_center in periodic_images(i_particle, rve_dims, add_images):
            for k_dim_tag in GmshMesher.add_primitive(
                factory, model, i_particle, j_center
            ):
                particle_tags.append(k_dim_tag[1])
                phase_dim_tag[i_particle.phase].append(k_dim_tag)
        if report is not None:
            report(i_particle_ind, len(particles))
    # The same geometry the mesher builds, so a view shows what would be meshed

    return particle_tags, phase_dim_tag


def keep_what_the_cut_left(factory, dim, box_tag, particle_tags, phase_dim_tag):
    """
    Cut the particles against the box, dropping the tags the cut removed.

    Parameters
    ----------
    factory: module
        The geometry kernel of the open gmsh session.

    dim: int
        Number of spatial dimensions.

    box_tag: int
        Tag of the box of the RVE.

    particle_tags: list
        Tags of everything that was added for the particles.

    phase_dim_tag: dict
        The *(dimension, tag)* pairs of each phase, as they were before the cut.

    Returns
    -------
    dict
        The pairs of each phase that survived the cut.
    """
    out_dim_tag, _ = factory.intersect(
        [(dim, box_tag)],
        [(dim, i_tag) for i_tag in particle_tags],
        removeObject=True,
        removeTool=True,
    )
    if particle_tags and not out_dim_tag:
        raise ValueError(
            "Cutting {0} particles against the RVE left nothing to view.".format(
                len(particle_tags)
            )
        )
    # Written as an empty view without a word, which is what OpenCASCADE made of a
    # micrometre RVE before the views were built at unit scale
    kept = set(out_dim_tag)
    factory.synchronize()
    # Synchronizing here is what lets the model be read below, by getBoundary

    return {
        i_phase: [i_dim_tag for i_dim_tag in i_dim_tags if i_dim_tag in kept]
        for i_phase, i_dim_tags in phase_dim_tag.items()
    }


def tag_phase_boundaries(model, phase_dim_tag, entity_dim, group_dim):
    """
    Name the boundary of every phase as a physical group, so that a viewer shows it.

    Parameters
    ----------
    model: module
        The model of the open gmsh session.

    phase_dim_tag: dict
        The *(dimension, tag)* pairs of each phase.

    entity_dim: int
        Dimension of the entities whose boundary is wanted.

    group_dim: int
        Dimension of the boundary itself.
    """
    for i_phase, i_dim_tags in phase_dim_tag.items():
        bound_dim_tags = model.getBoundary(
            [(entity_dim, i_tag) for _, i_tag in i_dim_tags]
        )
        material_tag = model.addPhysicalGroup(
            group_dim, [i_tag for i_dim, i_tag in bound_dim_tags if i_dim == group_dim]
        )
        model.setPhysicalName(group_dim, material_tag, "Phase {0}".format(i_phase))


def write_gmsh_view(gmsh, results_dir, name):
    """
    Write an open gmsh model for viewing.

    Parameters
    ----------
    gmsh: module
        The gmsh module, in an open session.

    results_dir: str
        Directory the files are written into.

    name: str
        Name of the files, without an extension.

    Returns
    -------
    list
        Paths of the files that were written.
    """
    mesh_path = os.path.join(results_dir, name + ".msh")
    vtk_path = os.path.join(results_dir, name + ".vtk")
    gmsh.write(mesh_path)
    gmsh.write(vtk_path)

    for i_path in (mesh_path, vtk_path):
        with open(i_path, "rt") as written:
            contents = written.read()
        if "," in contents:
            with open(i_path, "wt") as written:
                written.write(contents.replace(",", "."))

    return [mesh_path, vtk_path]
    # Gmsh sometimes writes a comma for a decimal point, depending on the locale. On a
    # machine where it does not, which is the usual case, the file is left alone rather
    # than read and written back identical


def plot_particles_3d(particles, rve_dims, sample_dir, **kwargs):

    particles, rve_dims, scale = at_unit_scale(particles, rve_dims)
    dim = len(rve_dims)
    mesh_size = particles[0].radius / 5
    with gmsh_view(sample_dir, mesh_size, scale=scale) as (gmsh, model, factory):

        box_tag = factory.addBox(
            0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
        )

        print_funcs.print_to_file(
            "\t> Adding particles to the model",
        )
        particle_tags, phase_dim_tag = add_particles_to_view(
            factory, model, particles, rve_dims,
            report=print_funcs.print_particle_progress,
        )
        print_funcs.print_to_file("")

        print_funcs.print_to_file("\t> Processing model\n")
        phase_dim_tag = keep_what_the_cut_left(
            factory, dim, box_tag, particle_tags, phase_dim_tag
        )

        tag_phase_boundaries(model, phase_dim_tag, 3, 2)

        # Generate a 3D mesh
        print_funcs.print_to_file("\t> Generating mesh\n")
        model.mesh.generate(2)

        write_gmsh_view(gmsh, sample_dir, "final_config")


def plot_particles_3d_one_by_one(particles, rve_dims, sample_dir, **kwargs):
    final_config_dir = os.path.join(sample_dir, "final_config")
    os.makedirs(final_config_dir)
    particles, rve_dims, scale = at_unit_scale(particles, rve_dims)
    for i_ind, i_particle in enumerate(particles):
        dim = len(rve_dims)
        mesh_size = particles[0].radius / 2
        with gmsh_view(sample_dir, mesh_size, scale=scale) as (gmsh, model, factory):

            box_tag = factory.addBox(
                0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
            )

            particle_tags, phase_dim_tag = add_particles_to_view(
                factory, model, [i_particle], rve_dims
            )

            phase_dim_tag = keep_what_the_cut_left(
                factory, dim, box_tag, particle_tags, phase_dim_tag
            )

            tag_phase_boundaries(model, phase_dim_tag, 3, 2)

            # Generate a 3D mesh
            model.mesh.generate(2)

            write_gmsh_view(
                gmsh, final_config_dir, "final_config_{0}".format(i_ind)
            )


def plot_kinetic_energy_history(
    kinetic_energy_history,
    thermic_energy_history,
    results_dir,
    save=True,
    show=False,
    temp_change=0,
    **kwargs
):
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()
    plt.semilogy(range(len(kinetic_energy_history)), kinetic_energy_history)
    plt.semilogy(range(len(thermic_energy_history)), thermic_energy_history)
    if "axes" not in kwargs:
        if save:
            plt.savefig(os.path.join(results_dir, "kinetic_energy.pdf"))

        if show:
            plt.show()
        plt.close()


def plot_delta_t_history(
    delta_t_history, results_dir, save=True, show=False, temp_change=0, **kwargs
):
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()
    plt.plot(range(len(delta_t_history)), delta_t_history)
    if "axes" not in kwargs:
        if save:
            plt.savefig(os.path.join(results_dir, "delta_t_history.pdf"))

        if show:
            plt.show()
        plt.close()


def plot_overlap_history(
    total_overlap_history,
    max_residue,
    results_dir,
    temp_change=False,
    save=True,
    show=False,
    **kwargs
):
    """Plot the overlap history as a function of the iteration step."""
    # Plot to a given axes or to a new figure
    if "axes" in kwargs:
        ax = kwargs["axes"]
        plt.sca(ax)
    else:
        plt.figure()
    # Plot temperature changes
    if temp_change and "temp_change_steps" in kwargs:
        for line in kwargs["temp_change_steps"]:
            plt.axvline(line, linewidth=0.01, linestyle="--", color="k")
    # Plotting the maximum residue if it is larger than 0
    if max_residue != 0:
        plt.semilogy([0, len(total_overlap_history)], [max_residue, max_residue])
    # Plot the overlap history in a semilogy plot
    graph_overlap_history = plt.semilogy(
        range(len(total_overlap_history)),
        total_overlap_history,
        color=kwargs.get("color", (68 / 255, 119 / 255, 170 / 255, 1)),
    )

    if "overlap_ratio" in kwargs:
        _ = plt.twinx(plt.gca())
        graph_ratio = plt.semilogy([None, None] + kwargs["overlap_ratio"])
        for artist in graph_ratio:
            artist.set_linestyle(":")
        plt.axhline(1, linewidth=0.01, linestyle="-", color="k")

    # Save and/or show if no axes was supplied
    if "axes" not in kwargs:
        if save:
            plt.savefig(
                os.path.join(
                    results_dir,
                    "{0}.pdf".format(kwargs.get("fig_name", "relative_energy")),
                )
            )

        if show:
            plt.show()
        plt.close()
    else:
        if "overlap_ratio" in kwargs:
            return graph_overlap_history, graph_ratio
        else:
            return graph_overlap_history


def plot_paths(particles, box, position_center_history, motion_results_dir):
    """Plot particle paths."""
    path_results_dir = os.path.join(motion_results_dir, "paths")
    os.makedirs(path_results_dir, exist_ok=True)
    if particles[0].dim == 2:
        particles, box, scale = at_unit_scale(particles, box)
        for step in range(len(position_center_history[0])):
            # Updating particle position to current time
            for i_particle_ind, i_particle in enumerate(particles):
                i_particle.position_center = (
                    np.asarray(position_center_history[i_particle_ind][step]) * scale
                )
            # Copies of the particles are walked along the path, at the scale the
            # view is built at; the microstructure's own were, and had to be put back
            # for the analyses that read them afterwards

            dim = len(box)
            mesh_size = particles[0].radius / 5
            with gmsh_view(path_results_dir, mesh_size, dim, scale=scale) as (
                gmsh,
                model,
                factory,
            ):

                box_tag = factory.addRectangle(
                    0,
                    0,
                    0,
                    box[0],
                    box[1],
                )

                particle_tags, phase_dim_tag = add_particles_to_view(
                    factory, model, particles, box
                )

                phase_dim_tag = keep_what_the_cut_left(
                    factory, dim, box_tag, particle_tags, phase_dim_tag
                )

                tag_phase_boundaries(model, phase_dim_tag, dim, dim - 1)

                # Generate a 3D mesh
                model.mesh.generate(2)

                write_gmsh_view(gmsh, path_results_dir, "mic_step_{0}".format(step))

    elif particles[0].dim == 3:

        for step in range(len(position_center_history[0])):
            with open(
                os.path.join(path_results_dir, "mic_step_{0}.vtk".format(step)),
                "w",
            ) as msh_vtk:
                msh_vtk.write("# vtk DataFile Version 2.0")
                msh_vtk.write("\n3D triangulation data")
                msh_vtk.write("\nASCII")
                msh_vtk.write("\n\nDATASET POLYDATA")
                msh_vtk.write("\nPOINTS {0} {1}".format(len(particles), "float"))
                for i_particle_index, i_particle in enumerate(particles):
                    position = position_center_history[i_particle_index][step]
                    msh_vtk.write(
                        "\n{0} {1} {2}".format(position[0], position[1], position[2])
                    )
                msh_vtk.write("\n\nPOINT_DATA {0}".format(len(particles)))
                msh_vtk.write("\nSCALARS {0} {1} {2}".format("radius", "float", "1"))
                msh_vtk.write("\nLOOKUP_TABLE default")
                for i_particle in particles:
                    msh_vtk.write("\n{0}".format(i_particle.radius))
                msh_vtk.write("\nSCALARS {0} {1} {2}".format("phase", "float", "1"))
                msh_vtk.write("\nLOOKUP_TABLE default")
                for i_particle in particles:
                    msh_vtk.write("\n{0}".format(i_particle.phase))


def plot_ratio_new_old_overlap(
    overlap_pairs_history, len_sim, results_dir, show=False, save=True, **kwargs
):
    """Plot the ratio of increasing and decreasing intersection overlap."""

    def moving_average(vec, ind, n=5):
        if ind > n // 2 and ind < len(vec) - n // 2:
            value = np.sum(vec[ind - n // 2 : ind + n // 2 + 1]) / (2 * (n // 2) + 1)
        else:
            value = None
        return value

    if "axes" in kwargs:
        ax = kwargs["axes"]
        plt.sca(ax)
    else:
        plt.figure()
    inc_history = [0 for _ in range(len_sim + 2)]
    dec_history = [0 for _ in range(len_sim + 2)]
    for pair_overlap_history in overlap_pairs_history.values():
        change_history_pair = (
            np.array(pair_overlap_history)[1:] - np.array(pair_overlap_history)[:-1]
        )
        for i_step, i_change in enumerate(change_history_pair):
            if i_change > 0:
                inc_history[i_step] += i_change
            elif i_change < 0:
                dec_history[i_step] += i_change
    ratio = [
        np.abs(inc_hist / dec_hist) if dec_hist != 0 else 1e12 if inc_hist != 0 else 1
        for inc_hist, dec_hist in zip(inc_history, dec_history)
    ]
    count = 0
    for ind, (rat_1, rat_2) in enumerate(zip(ratio[:-1], ratio[1:])):
        if (rat_1 - 1) * (rat_2 - 1) <= 0:
            count += 1
        if count == 2:
            plt.axvline(ind + 3, linewidth=0.1, linestyle=":", color="r", alpha=0.5)
            count = 0
    graph_overlap = plt.semilogy(list(range(2, len_sim + 3)), ratio)
    # plt.plot(
    #     list(range(len_sim)), [moving_average(ratio, ind) for ind in range(len_sim)]
    # )
    # plt.plot(
    #     list(range(len_sim)), [moving_average(ratio, ind, 20) for ind in range(len_sim)]
    # )
    plt.grid()
    # plt.ylim((0, 2))
    if "axes" not in kwargs:
        if save:
            plt.savefig(os.path.join(results_dir, "ratio_new_old_overlap.pdf"))

        if show:
            plt.show()
        plt.close()
    else:
        return graph_overlap

def finish_statistical_plot(kwargs, labels, default_name, legend=False):
    """
    Label a plot of a statistical descriptor, and save it unless it was given axes.

    Parameters
    ----------
    kwargs: dict
        What the plotting function was given: *axes* to draw into, *results_dir* to
        save in, *fig_name* to save under, and *show* to show it.

    labels: tuple
        The label of the horizontal axis, the label of the vertical one, and the title.

    default_name: str
        Name of the file when none was given, without an extension.

    legend: bool
        Whether the lines drawn are named and a legend is to be drawn.

    Returns
    -------
    bool
        Whether the figure was closed, which is to say that it was this function's to
        save rather than the caller's to go on drawing into.
    """
    horizontal, vertical, title = labels
    axes = kwargs.get("axes", plt.gca())
    axes.set_xlabel(horizontal)
    axes.set_ylabel(vertical)
    axes.set_title(title)
    if legend:
        axes.legend()
    if "axes" in kwargs:
        return False

    if "results_dir" in kwargs:
        plt.savefig(
            os.path.join(
                kwargs["results_dir"],
                "{0}.pdf".format(kwargs.get("fig_name", default_name)),
            ),
            bbox_inches="tight",
        )
    if kwargs.get("show", False):
        plt.show()
    plt.close()

    return True
    # Every plot of a descriptor ends the same way, and the labels are the difference
    # between a figure that is read and one that has to be explained: the plots used
    # to carry no axis label, no title and no legend, so the two lines of Ripley's K
    # function said nothing about which was the measurement


def plot_nearest_neighbor_dist(vals, **kwargs):
    """Plot the distribution of the distance from a particle to its nearest."""
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()

    graph = plt.hist(vals, bins="auto", histtype="step")
    closed = finish_statistical_plot(
        kwargs,
        (
            "Distance to the nearest neighbour",
            "Number of particles",
            "Nearest neighbour distances",
        ),
        "nearest_neighbor_dist",
    )

    return None if closed else graph


def plot_ripleys_k_func(vals, radii_vec, **kwargs):
    """Plot Ripley's K function against the one of a Poisson point process."""
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()

    artists = []
    artists += plt.plot(radii_vec, vals, label="Microstructure")
    artists += plt.plot(
        radii_vec, np.pi * radii_vec**2, linestyle="--", label="Poisson point process"
    )
    # The second is the K function of a process with no interaction between the
    # points, which is what the first is read against

    closed = finish_statistical_plot(
        kwargs,
        ("Radius", "K(r)", "Ripley's K function"),
        "k_ripleys_func",
        legend=True,
    )

    return None if closed else artists


SMOOTHING_WINDOW = 41
# Points of the two point correlation a smoothing runs over. A run asking for fewer
# radii than this is smoothed over all of them instead, where the filter used to raise


def plot_two_point_correlation(vals, radii_vec, **kwargs):
    """Plot the two point correlation function, and a smoothing of it."""
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()

    from scipy.signal import savgol_filter

    artists = []
    artists += plt.plot(radii_vec, vals, lw=0, marker="+", label="Estimate")
    window = min(SMOOTHING_WINDOW, len(vals) - (1 - len(vals) % 2))
    if window > 2:
        artists += plt.plot(
            radii_vec, savgol_filter(vals, window, 2), label="Smoothed"
        )
    # The estimate is a Monte Carlo one, so it is drawn as the points it is and the
    # line through them is said to be a smoothing

    closed = finish_statistical_plot(
        kwargs,
        ("Distance", "Two point correlation", "Two point correlation function"),
        "two_pt_corr",
        legend=True,
    )

    return None if closed else artists


def plot_pixels(pixel_grid, dir, show=False, save=True):
    import matplotlib.pyplot as plt

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(pixel_grid.T, cmap="Greys", interpolation="nearest")
    # plt.axis([0, np.size(pixel_grid.T, 0), 0, np.size(pixel_grid.T, 1)])
    if save:
        plt.savefig(dir + ".png", dpi=len(pixel_grid[0]))
        plt.close()
    if show:
        plt.show(block=False)


def plot_voxels(voxel_grid, matrix_phase, list_phase, dir, show=True, save=True):
    import matplotlib.pyplot as plt

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    particle_voxels = voxel_grid.T != int(matrix_phase)
    colors = np.empty(particle_voxels.shape, dtype=object)
    color_def = ["c", "r", "g", "y", "m", "b"]
    k_color = 0
    for phase in list_phase:
        if phase == matrix_phase:
            continue
        colors[voxel_grid.T == int(phase)] = color_def[k_color]
        k_color += 1
    ax.voxels(particle_voxels, facecolors=colors, edgecolor="k")
    if save:
        plt.savefig(dir + ".pdf")
    if show:
        plt.show()


def plot_voronoi_2d(
    particles, rve_dims, voronoi, dir, voronoi_type, save=True, show=False
):
    """Plot the Voronoi for circular particles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import voronoi_plot_2d

    fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=3)

    ax = plt.gca()

    N = len(particles)

    if rve_dims[0] == rve_dims[1]:
        ax.axis("square")
    else:
        ax.set_aspect("equal", adjustable="box")

    ax.set_ylim(0, rve_dims[1])
    ax.set_xlim(0, rve_dims[0])

    phases = list({i_particle.phase for i_particle in particles})
    colors = generate_colors(len(phases))
    phase_colors = dict(zip(phases, colors))

    for i_particle in particles:
        for j_dim, k_dim in [
            (j_dim, k_dim) for j_dim in range(-1, 2) for k_dim in range(-1, 2)
        ]:
            ellip = mpatches.Ellipse(
                i_particle.position_center
                + np.array(rve_dims) * np.array([1 * j_dim, 1 * k_dim]),
                i_particle.major_axis,
                i_particle.minor_axis,
                angle=180 / np.pi * i_particle.angle,
                alpha=0.8,
                edgecolor=None,
                facecolor=phase_colors[i_particle.phase],
            )
            ax.add_artist(ellip)
            # plt.annotate(str(k), tuple(i_particle.position_center))

    # print("here2")
    # voronoi_type = "standard"
    if voronoi_type == "set":
        set_voronoi_plot_2d(
            voronoi, ax=plt.gca(), show_vertices=False, point_size=0, line_width=0.1
        )
    elif voronoi_type == "standard":
        voronoi_plot_2d(
            voronoi, ax=plt.gca(), show_vertices=False, point_size=0, line_width=0.1
        )

    plt.axis([0, rve_dims[0], 0, rve_dims[1]])

    plt.xticks([])
    plt.yticks([])

    if save:
        plt.savefig(os.path.join(dir, "voronoi.pdf"))

    if show:
        plt.show()


def set_voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
    Diagram to plot

    ax : matplotlib.axes.Axes instance, optional
    Axes to plot on

    show_points: bool, optional

    Add the Voronoi points to the plot.

    show_vertices : bool, optional
    Add the Voronoi vertices to the plot.

    line_colors : string, optional
    Specifies the line color for polygon boundaries

    line_width : float, optional
    Specifies the line width for polygon boundaries

    line_alpha: float, optional
    Specifies the line alpha for polygon boundaries

    Returns
    -------
    fig : matplotlib.figure.Figure instance
    Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get("show_points", True):
        ax.plot(vor.points[:, 0], vor.points[:, 1], ".", ms=kw.get("point_size", 0.1))
    if kw.get("show_vertices", True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], "o")
        # for ind_vert, vert in enumerate(vor.vertices):
        #     plt.text(vert[0], vert[1], str(ind_vert))

    line_colors = kw.get("line_colors", "k")
    line_width = kw.get("line_width", 1.0)
    line_alpha = kw.get("line_alpha", 1.0)

    line_segments = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            line_segments.append([(x, y) for x, y in vor.vertices[simplex]])

    lc = LineCollection(
        line_segments,
        colors=line_colors,
        lw=line_width,
        linestyle="solid",
    )
    lc.set_alpha(line_alpha)
    ax.add_collection(lc)
    ptp_bound = np.ptp(vor.points, axis=0)
    #
    # line_segments = []
    # center = vor.points.mean(axis=0)
    # for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    #     simplex = np.asarray(simplex)
    #     if np.any(simplex < 0):
    #         i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
    #
    #         t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
    #         t /= np.linalg.norm(t)
    #         n = np.array([-t[1], t[0]])  # normal
    #
    #         midpoint = vor.points[pointidx].mean(axis=0)
    #         direction = np.sign(np.dot(midpoint - center, n)) * n
    #         far_point = vor.vertices[i] + direction * ptp_bound.max()
    #
    #         line_segments.append([(vor.vertices[i, 0], vor.vertices[i, 1]),
    #                               (far_point[0], far_point[1])])
    #
    # lc = LineCollection(line_segments,
    #                     colors=line_colors,
    #                     lw=line_width,
    #                     linestyle='dashed')
    # lc.set_alpha(line_alpha)
    # ax.add_collection(lc)
    _adjust_bounds(ax, vor.points)

    return ax.figure


def plot_voronoi_2d_with_imts(
    particles, rve_dims, voronoi, imts, in_box, dir, voronoi_type, save=True, show=False
):
    """
    Plot the Voronoi for circular particles.

    Parameters
    ----------
    in_box: list(int)
        Indices, among the cells the tensors were computed for, of the cells of the
        particles themselves rather than of their periodic images.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    from scipy.spatial import voronoi_plot_2d

    plt.rc("text", usetex=True)
    plt.rc(
        "text.latex",
        preamble=r"\usepackage[widespace]{fourier} \usepackage{amsmath} \usepackage{amssymb}",
    )

    N = len(particles)

    for i_order in range(7):

        fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        if rve_dims[0] == rve_dims[1]:
            ax.axis("square")
        else:
            ax.set_aspect("equal", adjustable="box")

        ax.set_ylim(0, rve_dims[1])
        ax.set_xlim(0, rve_dims[0])

        phases = list({i_particle.phase for i_particle in particles})
        colors = generate_colors(len(phases))
        phase_colors = dict(zip(phases, colors))

        for i_particle in particles:
            for j_dim, k_dim in [
                (j_dim, k_dim) for j_dim in range(-1, 2) for k_dim in range(-1, 2)
            ]:
                ellip = mpatches.Ellipse(
                    i_particle.position_center
                    + np.array(rve_dims) * np.array([1 * j_dim, 1 * k_dim]),
                    i_particle.major_axis,
                    i_particle.minor_axis,
                    angle=180 / np.pi * i_particle.angle,
                    alpha=0.5,
                    edgecolor="k",
                    linewidth=0.1,
                    linestyle="-.",
                    facecolor="None",  # phase_colors[i_particle.phase],
                )
                ax.add_artist(ellip)
                # plt.annotate(str(k), tuple(i_particle.position_center))

        if voronoi_type == "set":
            set_voronoi_plot_2d(
                voronoi, ax=plt.gca(), show_vertices=False, point_size=0, line_width=0.1
            )
        elif voronoi_type == "standard":
            voronoi_plot_2d(
                voronoi, ax=plt.gca(), show_vertices=False, point_size=0, line_width=0.1
            )

        plt.axis([0, rve_dims[0], 0, rve_dims[1]])

        # cmap = matplotlib.cm.get_cmap("jet")
        cmap = matplotlib.cm.get_cmap("Blues")
        perimeters = np.abs(np.array(imts)[in_box, 0])
        perimeter_scale = matplotlib.colors.Normalize(
            vmin=perimeters.min(), vmax=perimeters.max()
        )
        # The perimeters are lengths, so they are coloured over the range the cells of
        # the particles span; the cells of the images, far larger at the edges of the
        # diagram, would take the range up. They were handed to the colour map as they
        # were, which takes a number between zero and one: every cell of an RVE of unit
        # side came out one saturated colour
        # Initializing the list containing the list of imts for each Voronoi cell
        k_cell = 0
        for ind, i_region in enumerate(voronoi.regions):
            if len(i_region) == 0:
                continue
            if any([vertex == -1 for vertex in i_region]):
                continue
            # Running through all the cells in the Voronoi
            # plt.sca(ax)
            if i_order > 0:
                color = cmap(
                    np.ceil(
                        10 * np.abs(imts[k_cell][i_order]) / np.abs(imts[k_cell][0])
                    )
                    / 10
                )
            else:
                color = cmap(perimeter_scale(np.abs(imts[k_cell][0])))
            x = [voronoi.vertices[i_vertex][0] for i_vertex in i_region]
            y = [voronoi.vertices[i_vertex][1] for i_vertex in i_region]
            current_cell = plt.fill(x, y, edgecolor=None, linewidth=0)
            current_cell[0].set_color(color)
            k_cell += 1

        if rve_dims[0] == rve_dims[1]:
            ax.axis("square")
        else:
            ax.set_aspect("equal", adjustable="box")

        ax.set_ylim(0, rve_dims[1])
        ax.set_xlim(0, rve_dims[0])

        plt.xticks([])
        plt.yticks([])

        if i_order == 0:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=perimeter_scale, cmap=cmap),
                ax=ax,
                label=r"Perimeter",
            )
        else:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(cmap=cmap),
                ax=ax,
                label=r"$q_{0}$".format(str(i_order)),
                boundaries=np.linspace(0, 1, 11),
            )
        # The axes are named: a colour bar for a mappable drawn on none is refused by
        # the matplotlib installed, so the plot of the tensors never got written
        if save:
            plt.savefig(
                os.path.join(dir, "voronoi_{0}.pdf".format(i_order)),
                bbox_inches="tight",
            )

        if show:
            plt.show()

    for i_order in range(7):

        fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        N = len(particles)

        if i_order == 0:
            plt.hist(
                np.abs(np.array(imts)[in_box, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
            )
            ax.set_xlabel(r"Perimeter")
        else:
            plt.hist(
                np.abs(np.array(imts)[in_box, i_order])
                / np.real(np.array(imts)[in_box, 0]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
                range=(0, 1),
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            )
            plt.axvline(
                np.mean(
                    np.abs(np.array(imts)[in_box, i_order])
                    / np.real(np.array(imts)[in_box, 0])
                ),
                color="k",
                linestyle="--",
            )
            ax.set_xlabel(r"$q_{0}$".format(str(i_order)))
            plt.xlim([0, 1])
            plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])

        ax.set_ylabel(r"$N$")

        if save:
            plt.savefig(
                os.path.join(dir, "voronoi_hist_{0}.pdf".format(i_order)),
                bbox_inches="tight",
            )

        if show:
            plt.show()
        plt.close()


def plot_voronoi_3d(particles, voronoi, rve_dims, sample_dir, save=True, show=False):
    """Plot the Voronoi for circular particles."""
    particles, rve_dims, scale = at_unit_scale(particles, rve_dims)
    vertices = np.asarray(voronoi.vertices) * scale
    dim = len(rve_dims)
    with gmsh_view(sample_dir, MESH_SIZE_VORONOI, scale=scale) as (
        gmsh,
        model,
        factory,
    ):

        box_tag = factory.addBox(
            0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
        )

        particle_tags, phase_dim_tag = add_particles_to_view(
            factory, model, particles, rve_dims, add_images=False
        )

        # Set the mesh size on the geometry points
        # Synchronize the CAD engine (always needed before generating the mesh)
        # It may also be useful for some intermidate operations, like checking the tags of
        # entities
        factory.synchronize()

        verticesTags = np.array(
            [factory.addPoint(vertex[0], vertex[1], vertex[2]) for vertex in vertices]
        )
        edgeTags = {}
        edge_point = {}
        points = set()
        for ridge_ind, (ridge_pt_1, ridge_pt_2) in enumerate(voronoi.ridge_points):

            ridge = voronoi.ridge_vertices[ridge_ind]
            if -1 in ridge or (
                ridge_pt_1 not in range(13, 3**3 * len(particles), 27)
                and ridge_pt_2 not in range(13, 3**3 * len(particles), 27)
            ):
                continue
            points.add(ridge_pt_1)
            points.add(ridge_pt_2)
            ridge_out_phase = ridge[-1:] + ridge[0:-1]
            for vertex_1, vertex_2 in zip(ridge, ridge_out_phase):
                if (vertex_1, vertex_2) not in edgeTags and (
                    vertex_2,
                    vertex_1,
                ) not in edgeTags:
                    edgeTags[(vertex_1, vertex_2)] = factory.addLine(
                        verticesTags[vertex_1], verticesTags[vertex_2]
                    )
                    edge_point[(vertex_1, vertex_2)] = {ridge_pt_1, ridge_pt_2}
                elif (vertex_1, vertex_2) in edgeTags:
                    if edge_point[(vertex_1, vertex_2)] == {
                        ridge_pt_1,
                        ridge_pt_2,
                    }:
                        del edgeTags[(vertex_1, vertex_2)]
                elif (vertex_2, vertex_1) in edgeTags:
                    if edge_point[(vertex_2, vertex_1)] == {
                        ridge_pt_1,
                        ridge_pt_2,
                    }:
                        del edgeTags[(vertex_2, vertex_1)]

        factory.synchronize()
        # all_voronoi_lines = list(set([voronoi_line[1] for voronoi_line in voronoi_lines] + [edgeTag[1] for edgeTag in out_dim_tag4]))
        voronoiWires = model.addPhysicalGroup(
            1, list(edgeTags.values())
        )  # [(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
        model.setPhysicalName(1, voronoiWires, "Voronoi")
        # Named in the dimension of the group, the lines; named in two, it was not
        # voronoiWires = model.addPhysicalGroup(1, [tag[1] for tag in out_dim_tag_3]) #[(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
        # model.setPhysicalName(1, voronoiWires, "Voronoi")

        tag_phase_boundaries(model, phase_dim_tag, 3, 2)

        # Generate a 3D mesh
        model.mesh.generate(2)

        write_gmsh_view(gmsh, sample_dir, "voronoi")


def plot_voronoi_3d_with_imts(
    particles, voronoi, rve_dims, imts, dir, save=True, show=False
):
    """Plot the Voronoi for circular particles."""
    title = os.path.join(dir, "voronoi_wIMTs")
    scale = unit_scale(rve_dims)
    rve_dims = [i_dim * scale for i_dim in rve_dims]
    vertices = np.asarray(voronoi.vertices) * scale
    # Built at unit scale, as every view is. The element size, and the margin the
    # volumes of a cell are found within below, are lengths: at a millionth of the unit
    # the margin took in every cell, and every cell was painted with the first's values
    with gmsh_view(title, MESH_SIZE_VORONOI_IMTS, scale=scale) as (
        gmsh,
        model,
        factory,
    ):

        boxTag = factory.addBox(
            -rve_dims[0],
            -rve_dims[1],
            -rve_dims[2],
            3 * rve_dims[0],
            3 * rve_dims[1],
            3 * rve_dims[2],
        )
        # RVE

        verticesTags = np.array(
            [factory.addPoint(vertex[0], vertex[1], vertex[2]) for vertex in vertices]
        )
        planeSurfaceTags = []
        planeSurfaceDictTags = {}
        edgeTags = {}
        for i_particle in range(13, len(voronoi.point_region), 27):
            particle_region = voronoi.regions[voronoi.point_region[i_particle]]
            for ridge in voronoi.ridge_vertices:
                edgeFaceTags = []
                if -1 in ridge or any([vertex not in particle_region for vertex in ridge]):
                    continue

                ridge_vertices = vertices[ridge]
                center_gravity = 1 / len(ridge) * np.sum(ridge_vertices, axis=0)
                # Computing the center of the polygon
                ref_vec_x = ridge_vertices[0] - center_gravity
                ref_vec_y = (ridge_vertices[1] - center_gravity) - np.dot(
                    ridge_vertices[1] - center_gravity, ref_vec_x
                ) / np.dot(ref_vec_x, ref_vec_x) * ref_vec_x
                angles = []
                for i_vertex in ridge_vertices:
                    i_ref_vec = i_vertex - center_gravity
                    angles.append(
                        np.arctan2(i_ref_vec.dot(ref_vec_y), i_ref_vec.dot(ref_vec_x))
                    )

                sorted_ridge = [ridge[i_vert] for i_vert in np.argsort(angles)]

                ridge_out_phase = sorted_ridge[-1:] + sorted_ridge[0:-1]
                for vertex_1, vertex_2 in zip(sorted_ridge, ridge_out_phase):
                    if (vertex_1, vertex_2) not in edgeTags or (
                        vertex_2,
                        vertex_1,
                    ) not in edgeTags:
                        edgeTags[(vertex_1, vertex_2)] = factory.addLine(
                            verticesTags[vertex_1], verticesTags[vertex_2]
                        )
                    edgeFaceTags.append(
                        edgeTags.get(
                            (vertex_1, vertex_2), edgeTags.get((vertex_2, vertex_1))
                        )
                    )
                curveLoopTag = factory.addCurveLoop(edgeFaceTags)

                planeSurfaceTags.append(factory.addPlaneSurface([curveLoopTag]))
                planeSurfaceDictTags[tuple(ridge)] = planeSurfaceTags[-1]

        factory.synchronize()
        # box_surface = gmsh.model.getBoundary([(3, boxTag)])
        _, _ = factory.fragment(
            [(2, planeSurface) for planeSurface in planeSurfaceTags],
            [(3, boxTag)],
            removeObject=False,
            removeTool=True,
        )

        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        eps = 1e-2
        number_cells = 0
        cellCheckTags = []
        for i_particle in range(13, len(voronoi.point_region), 27):
            region = voronoi.regions[voronoi.point_region[i_particle]]
            voronoiSurfaceTags = []
            if -1 in region:
                continue
            for ridge in voronoi.ridge_vertices:
                if all([vertex in region for vertex in ridge]):
                    voronoiSurfaceTags.append(planeSurfaceDictTags[tuple(ridge)])
            surfaceLoop = factory.addSurfaceLoop(voronoiSurfaceTags)
            volumeCell = factory.addVolume([surfaceLoop])
            factory.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, volumeCell)
            cellCheckTags.append(volumeCell)
            i_voronoi_cell = gmsh.model.getEntitiesInBoundingBox(
                xmin - eps,
                ymin - eps,
                zmin - eps,
                xmax + eps,
                ymax + eps,
                zmax + eps,
                dim=3,
            )
            factory.synchronize()
            material_tag = model.addPhysicalGroup(
                3, [cell[1] for cell in i_voronoi_cell if cell[0] == 3]
            )
            model.setPhysicalName(3, material_tag, "Cell " + str(number_cells))
            number_cells += 1
        gmsh.model.removeEntities([(3, tag) for tag in cellCheckTags])

        # factory.synchronize()
        # # box_surface = gmsh.model.getBoundary([(3, boxTag)])
        # out_dim_tag_3, _ = factory.fragment(
        #     [(2, planeSurface) for planeSurface in planeSurfaceTags], [(3, boxTag)],
        #     removeObject=True, removeTool=True)
        #
        # factory.synchronize()
        # number_cells = 0
        # particle_centers = np.array([voronoi.points[point] for point in range(13, len(voronoi.point_region), 27)])
        # for index, i_voronoi_cell in enumerate(out_dim_tag_3):
        #     if i_voronoi_cell[0] == 3:
        #         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, i_voronoi_cell[1])
        #         number_cells += 1
        #         material_tag = model.addPhysicalGroup(3, [i_voronoi_cell[1]])
        #         model.setPhysicalName(3, material_tag, "Cell " + str(number_cells))
        #
        #
        # getElementByCoordinates
        gmsh.option.setNumber("Mesh.Algorithm3D", 2)
        # Frontal, which is what this view has always been built with. The element the
        # session is set up for settles the algorithm for a mesh that goes to a solver,
        # and this is the one view that fills a volume rather than covering a surface

        # Generate a 3D mesh
        model.mesh.generate(3)

        _, vtk_file = write_gmsh_view(gmsh, dir, "voronoi_wIMTs")
    # The session is closed by the block; what follows reads back the file it wrote, so
    # it has to happen after and not inside

    dataType = "float"
    numComp = "1"

    fin = open(vtk_file, "rt")

    element_cell = []
    in_cell_data = False
    for line in fin:
        if line.startswith("CELL_DATA"):
            in_cell_data = True
            continue
        if in_cell_data:
            element_cell.append(line.rstrip("\n"))
    # Reading the cells back used to be flagged with the name of the parameter that says
    # whether to save the histograms, so asking not to save them saved them anyway

    fin.close()

    with open(vtk_file, "a") as msh_vtk:
        for i_IMT in range(7):
            if i_IMT == 0:
                dataName = "Surface_Area"
            else:
                dataName = "q_" + str(i_IMT)
            msh_vtk.write("\n\nSCALARS {0} {1} {2}".format(dataName, dataType, numComp))
            msh_vtk.write("\nLOOKUP_TABLE default")
            for cell_id in element_cell[2:]:
                msh_vtk.write("\n{0}".format(imts[int(cell_id) - 1][i_IMT]))

    for i_order in range(7):

        _, ax, (_, _) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        if i_order == 0:
            plt.hist(
                np.abs(np.array(imts)[:, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
            )
            ax.set_xlabel(r"Surface Area")
        else:
            plt.hist(
                np.abs(np.array(imts)[:, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
                range=(0, 1),
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            )
            plt.axvline(
                np.mean(np.abs(np.array(imts)[:, i_order])), color="k", linestyle="--"
            )
            ax.set_xlabel(r"$q_{0}$".format(str(i_order)))
            plt.xlim([0, 1])
            plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])

        ax.set_ylabel(r"$N$")

        if save:
            plt.savefig(
                dir + "_" + str(i_order) + "_hist" + ".pdf", bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close()


# def rescaleAxis(ax):
#     for line in ax.lines:
