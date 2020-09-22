import numpy as np

import scipy.integrate as integrate

import os

import gmsh


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# Simple math tools
# Finite element mesh conversor to LINKS


from microstructure.particle_classes import (
    Disk,
    Ellipse,
    CylindricalFiber,
    Sphere,
    Ellipsoid,
)

from postproc.mshgen.meshing_interface import FEMMeshGenerator

latex_textwidth = 5.92  # in = 496pt
latex_textheigth = 9.63  # in = 674pt


def _adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
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
):
    """Create a matplotlib figure for a a4 report latex page.

    The figure is created with a size such that it fits in a4 report latex page, in an array

    *nrows* rows and *ncols* columns.

    Parameters
    ----------
    nrows: optional, int
        Number of rows in the a4 page.

    """
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "Adobe Caslon Pro"
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.unicode"] = True
    w = (latex_textwidth - 0.2 * (ncols - 1)) / ncols
    h = (latex_textheigth - 1.63 * (nrows - 1)) / nrows
    fig, axs = plt.subplots(
        figsize=(w, h), nrows=nrows_sub, ncols=ncols_sub, sharey=sharey
    )
    return [fig, axs, (w, h)]


def create_legend(artists, labels, axes, to_fig=False, fig_h=4, ncols=3):
    if to_fig:
        lw_common = axes.spines["bottom"].get_linewidth()
        fig = plt.gcf()
        legend = fig.legend(
            handles=artists,
            labels=labels,
            bbox_to_anchor=(0.0, 1, 1.0, 0.102),
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
    else:
        print("h_fig", fig_h)
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
        plot_particles_3d(particles, rve_dims, sample_dir)


def plot_particles_2d(particles, rve_dims, sample_dir, **kwargs):

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
        for (j_dim, k_dim) in [
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

    if kwargs.get("save", True):
        plt.savefig(os.path.join(sample_dir, "final_config.pdf"), bbox_inches="tight")

    if kwargs.get("show", False):
        plt.show()


def plot_particles_3d(particles, rve_dims, sample_dir, **kwargs):

    dim = len(rve_dims)
    mesh_generator = FEMMeshGenerator(particles[0].radius / 10, "tetra4", rve_dims)

    mesh_generator.init_gmsh_model()

    model = gmsh.model
    factory = model.occ
    # occ - OpenCASCADE CAD (more advanced)

    model.add(sample_dir)

    mesh_generator.box_tag = factory.addBox(
        0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
    )

    mesh_generator.phase_dim_tag = {
        phase_name: [] for phase_name in {i_particle.phase for i_particle in particles}
    }
    for i_particle in particles:
        mesh_generator.add_particle_pbc_to_model(i_particle, rve_dims)

    out_dim_tag, _ = factory.intersect(
        [(dim, mesh_generator.box_tag)],
        [(dim, particle_tag) for particle_tag in mesh_generator.particle_tags],
        removeObject=True,
        removeTool=True,
    )

    temp = set(out_dim_tag)
    for i_phase in mesh_generator.phase_dim_tag:
        mesh_generator.phase_dim_tag[i_phase] = [
            value for value in mesh_generator.phase_dim_tag[i_phase] if value in temp
        ]

    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of
    # entities
    factory.synchronize()

    for i_phase, i_tags in mesh_generator.phase_dim_tag.items():
        material_tag = model.addPhysicalGroup(3, [tag for _, tag in i_tags])
        model.setPhysicalName(3, material_tag, "Phase {0}".format(i_phase))

    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_generator.mesh_size)

    # Generate a 3D mesh
    model.mesh.generate(3)

    _ = mesh_generator.write_mesh_gmsh(sample_dir)


def plot_kinetic_energy_history(
    kinetic_energy_history, results_dir, save=True, show=False, **kwargs
):
    if "axes" in kwargs:
        plt.sca(kwargs["axes"])
    else:
        plt.figure()
    plt.plot(range(len(kinetic_energy_history)), kinetic_energy_history)
    if "axes" not in kwargs:
        if save:
            plt.savefig(os.path.join(results_dir, "kinetic_energy.pdf"))

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

    if "axes" in kwargs:
        ax = kwargs["axes"]
        plt.sca(ax)
    else:
        plt.figure()
    if temp_change and "temp_change_steps" in kwargs:
        for line in kwargs["temp_change_steps"]:
            plt.axvline(line, linewidth=0.01, linestyle="--", color="k")
    if max_residue != 0:
        plt.semilogy([0, len(total_overlap_history)], [max_residue, max_residue])
    graph_overlap_history = plt.semilogy(
        range(len(total_overlap_history)), total_overlap_history
    )
    plt.grid()
    if "axes" not in kwargs:
        if save:
            plt.savefig(os.path.join(results_dir, "relative_energy.pdf"))

        if show:
            plt.show()
        plt.close()
    else:
        return graph_overlap_history


def plot_paths(particles, position_center_history, motion_results_dir):
    """Plot particle paths."""
    path_results_dir = os.path.join(motion_results_dir, "paths")
    os.makedirs(path_results_dir)
    if particles[0].dim == 2:
        for step in range(len(position_center_history[0])):
            with open(
                os.path.join(path_results_dir, "mic_step_{0}.vtk".format(step)),
                "a",
            ) as msh_vtk:
                if particles[0].dim == 2:
                    msh_vtk.write("# vtk DataFile Version 2.0")
                    msh_vtk.write("\n3D triangulation data")
                    msh_vtk.write("\nASCII")
                    msh_vtk.write("\n\nDATASET POLYDATA")
                    msh_vtk.write(
                        "\nPOINTS {0} {1}".format(9 * len(particles), "float")
                    )
                    for i_particle_index, i_particle in enumerate(particles):
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                position = (
                                    position_center_history[i_particle_index][step]
                                    + [j, k] * box
                                )
                                msh_vtk.write(
                                    "\n{0} {1} 0".format(position[0], position[1])
                                )
                    msh_vtk.write("\n\nPOINT_DATA {0}".format(9 * len(particles)))
                    msh_vtk.write(
                        "\nSCALARS {0} {1} {2}".format("radius", "float", "1")
                    )
                    msh_vtk.write("\nLOOKUP_TABLE default")
                    for i_particle in particles:
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                msh_vtk.write("\n{0}".format(i_particle.radius))
                    msh_vtk.write("\nSCALARS {0} {1} {2}".format("phase", "float", "1"))
                    msh_vtk.write("\nLOOKUP_TABLE default")
                    for i_particle in particles:
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                msh_vtk.write("\n{0}".format(i_particle.phase))

    elif particles[0].dim == 3:

        for step in range(len(position_center_history[0])):
            with open(
                os.path.join(path_results_dir, "mic_step_{0}.vtk".format(step)),
                "a",
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


def plot_pixels(pixel_grid, dir, show=False, save=True):
    import matplotlib.pyplot as plt

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    fig = plt.figure()
    plt.imshow(pixel_grid.T)
    plt.axis([0, np.size(pixel_grid.T, 0), 0, np.size(pixel_grid.T, 1)])
    if save:
        plt.savefig(dir + ".pdf")
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


def plotVoronoi2D(particles, voronoi, dir, voronoi_type, save=True, show=False):
    """Plot the Voronoi for circular particles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import voronoi_plot_2d

    fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=3)

    ax = plt.gca()

    N = len(particles)

    if Particle.box[0] == Particle.box[1]:
        ax.axis("square")
    else:
        ax.set_aspect("equal", adjustable="box")

    ax.set_ylim(0, Particle.box[1])
    ax.set_xlim(0, Particle.box[0])

    colors = generate_colors(particles)

    for i in range(N):
        class_name_i_particle = particles[i].__class__.__name__
        # particles[i].dilate(global_crit_ero_thick)
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (
                    "Disk" == class_name_i_particle
                    or "CylindricalFiber" == class_name_i_particle
                ):
                    circ = mpatches.Circle(
                        particles[i].position_center + Particle.box * np.array([j, k]),
                        radius=particles[i].radius,
                        edgecolor=(0, 0, 0, 0),
                        facecolor=colors[particles[i].phase],
                    )
                    ax.add_artist(circ)
                if "Ellipse" == class_name_i_particle:
                    ellip = mpatches.Ellipse(
                        particles[i].position_center
                        + Particle.box * np.array([1 * j, 1 * k]),
                        particles[i].major_axis,
                        particles[i].minor_axis,
                        angle=180 / np.pi * particles[i].angle,
                        edgecolor=(0, 0, 0, 0),
                        facecolor=colors[particles[i].phase],
                    )
                    ax.add_artist(ellip)

    if voronoi_type == "set":
        set_voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False, point_size=1)
    elif voronoi_type == "standard":
        voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False, point_size=1)

    plt.axis([0, Particle.box[0], 0, Particle.box[1]])

    plt.xticks([])
    plt.yticks([])

    if save:
        plt.savefig(dir + "_voronoi" + ".pdf")

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
            ax.plot(vor.points[:, 0], vor.points[:, 1], ".")
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
                        ptp_bound = vor.points.ptp(axis=0)
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


def plotVoronoi2DwithIMTs(
    particles, voronoi, IMTs, dir, voronoi_type, save=True, show=False
):
    """Plot the Voronoi for circular particles."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    from scipy.spatial import voronoi_plot_2d

    N = len(particles)
    # for i in range(N):
    #     print(particles[i].position_center[0], particles[i].position_center[1])

    for i_order in range(7):

        fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect("equal", adjustable="box")

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])

        colors = generate_colors(particles)

        for i in range(N):
            # print(particles[i].position_center[0], particles[i].position_center[1])
            class_name_i_particle = particles[i].__class__.__name__
            # particles[i].dilate(global_crit_ero_thick)
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (
                        "Disk" == class_name_i_particle
                        or "CylindricalFiber" == class_name_i_particle
                    ):
                        circ = mpatches.Circle(
                            particles[i].position_center
                            + Particle.box * np.array([j, k]),
                            radius=particles[i].radius,
                            facecolor=(0, 0, 0, 0),
                            edgecolor="k",
                            linewidth=0.5,
                            linestyle="-.",
                        )  # colors[particles[i].phase])
                        ax.add_artist(circ)
                    if "Ellipse" == class_name_i_particle:
                        ellip = mpatches.Ellipse(
                            particles[i].position_center
                            + Particle.box * np.array([1 * j, 1 * k]),
                            particles[i].major_axis,
                            particles[i].minor_axis,
                            angle=180 / np.pi * particles[i].angle,
                            facecolor=(0, 0, 0, 0),
                            edgecolor="k",
                            linewidth=0.5,
                            linestyle="-.",
                        )
                        ax.add_artist(ellip)

        if voronoi_type == "set":
            set_voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False)
        elif voronoi_type == "standard":
            voronoi_plot_2d(voronoi, ax=plt.gca(), show_vertices=False)

        plt.axis([0, Particle.box[0], 0, Particle.box[1]])

        cmap = matplotlib.cm.get_cmap("Blues")
        # Initializing the list containing the list of IMTs for each Voronoi cell
        k_cell = 0
        for ind, i_region in enumerate(voronoi.regions):
            if len(i_region) == 0:
                continue
            if any([vertex == -1 for vertex in i_region]):
                continue
            # Running through all the cells in the Voronoi
            # plt.sca(ax)
            if i_order > 0:
                color = cmap(np.abs(IMTs[k_cell][i_order]) / np.abs(IMTs[k_cell][0]))
            else:
                color = cmap(np.abs(IMTs[k_cell][0]))
            x = [voronoi.vertices[i_vertex][0] for i_vertex in i_region]
            y = [voronoi.vertices[i_vertex][1] for i_vertex in i_region]
            current_cell = plt.fill(x, y)
            current_cell[0].set_color(color)
            k_cell += 1

        if Particle.box[0] == Particle.box[1]:
            ax.axis("square")
        else:
            ax.set_aspect("equal", adjustable="box")

        ax.set_ylim(0, Particle.box[1])
        ax.set_xlim(0, Particle.box[0])

        colors = generate_colors(particles)
        plt.xticks([])
        plt.yticks([])

        if i_order == 0:
            plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), label=r"Perimeter")
        else:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(cmap=cmap),
                label=r"$q_{0}$".format(str(i_order)),
            )
        if save:
            plt.savefig(dir + "_" + str(i_order) + ".pdf", bbox_inches="tight")

        if show:
            plt.show()

    region_point = np.zeros((len(voronoi.regions)), dtype=int)
    in_box = []
    for point_ind, region_ind in enumerate(voronoi.point_region):
        if point_ind == -1:
            continue
        region_point[region_ind] = int(point_ind)
    k_used_region = 0
    for ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        print(region_point[ind])
        pos_center = voronoi.points[region_point[ind]]
        if 0 < pos_center[0] < 1 and 0 < pos_center[1] < 1:
            in_box.append(k_used_region)
        k_used_region += 1

    print(len(in_box))

    for i_order in range(7):

        fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        N = len(particles)

        if i_order == 0:
            plt.hist(
                np.abs(np.array(IMTs)[in_box, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
            )
            ax.set_xlabel(r"Perimeter")
        else:
            plt.hist(
                np.abs(np.array(IMTs)[in_box, i_order])
                / np.real(np.array(IMTs)[in_box, 0]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
                range=(0, 1),
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
            plt.axvline(
                np.mean(
                    np.abs(np.array(IMTs)[in_box, i_order])
                    / np.real(np.array(IMTs)[in_box, 0])
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
                dir + "_" + str(i_order) + "_hist" + ".pdf", bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close()


def plotVoronoi3Dpbc(
    particles, voronoi, rve_dims, dir, voronoi_type, save=True, show=True
):
    """Plot the Voronoi for circular particles."""
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", 1)

    # 2D Meshing algorithm
    # --------------------
    # 1 - Mesh Adapt
    # 2 - Automatic
    # 5 - Delaunay (default)
    # 6 - Frontal-Delaunay
    # 7 - BAMG
    # 8 - Frontal-Delaunay for Quads
    # 9 - Packing of Parallelograms
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # 3D Meshing algorithm
    # --------------------
    # 1 - Delaunay (default)
    # 2 - Frontal
    # 7 - MMG3D
    # 9 - R-tree
    # 10 - HXT
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 0)

    # MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Quad/Hex recombination algorithms
    # ---------------------------------
    # 0 - simple
    # 1 - blossom (default)
    # 2 - simple full-quad
    # 3 - blosson full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Number of topological optimization passes of recombined surface meshes (5 by default)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

    # Force recombination in all volumes
    gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

    # Recombination level in 3D
    # -------------------------
    # 0 - hex (default)
    # 1 - hex + prisms
    # 2 - hex + prisms + pyramids
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

    # Recombination conformity type in 3D meshes
    # ------------------------------------------
    # 0 - nonconforming (default)
    # 1 - trihedra
    # 2 - pyramids + trihedra
    # 2 - pyramids + hexSplit + trihedra
    # 4 - hexSplit + trihedra
    gmsh.option.setNumber("Mesh.Recombine3DConformity", 1)

    # Renumber nodes and elements after mesh generation
    gmsh.option.setNumber("Mesh.Renumber", 1)

    # Save all elements even if they do not belong to physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Number of smoothing step applied to the final mesh
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Element order
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Crete second-order nodes by linear interpolation
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Second-order incomplete elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    # ==========================================================================================
    # Generate the finite element mesh
    # ==========================================================================================
    # Define model name

    title = Particle.file_path
    model.add(title)

    boxTag = factory.addBox(0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2])
    # RVE

    particle_tags = []
    k_particle_image = 0
    phase_dim_tag = {phase: [] for phase in Particle.list_phases}
    for i_particle in particles:
        # Running through all the particles
        class_name_i_particle = i_particle.__class__.__name__
        # Saving the class name of the particle as a string
        for j in range(1):
            # Periodic images in the x direction
            for p in range(1):
                # Periodic images in the y direction
                for l in range(1):
                    # Periodic images in the z direction
                    if "CylindricalFiber" == class_name_i_particle:
                        if l != 0:
                            continue
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = 0
                        rx = i_particle.radius
                        ry = i_particle.radius
                        # Speciying the position and the radius of the fibers face
                        face_tag = factory.addDisk(xc, yc, zc, rx, ry)
                        # Saving the properties of the particles
                        if i_particle.direction_fibers == 0:
                            # The fibers run in the x direction
                            factory.rotate(
                                [(2, face_tag)], 0, 0, 0, 0, 1, 0, 3 * np.pi / 2
                            )
                            # Rotating the fiber face to the yz plane as it was ploted
                            # in the xy plane
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], i_particle.length_dir_fibers, 0, 0
                            )
                            # Extruding the fiber from the fiber face in the yz plane in the
                            # x direction
                        elif i_particle.direction_fibers == 1:
                            # The fibers run in the y direction
                            factory.rotate([(2, face_tag)], 0, 0, 0, 1, 0, 0, np.pi / 2)
                            # Rotating the fiber faces to the xz plane as it was ploted
                            # in the xy plane
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], 0, i_particle.length_dir_fibers, 0
                            )
                            # Extruding the fiber from the fiber face in the xz plane in the
                            # y direction
                        elif i_particle.direction_fibers == 2:
                            # The fibers run in the z direction
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], 0, 0, i_particle.length_dir_fibers
                            )
                            # Extruding the fiber from the fiber face in the xy plane in the
                            # z direction

                        for i_dim_tag in extrusion_tags:
                            if i_dim_tag[0] == 3:
                                particle_tags.append(i_dim_tag[1])
                                break

                        phase_dim_tag[str(i_particle.phase)].append(
                            (3, particle_tags[-1])
                        )

                        factory.synchronize()
                        k_particle_image += 1
                    if "Sphere" == class_name_i_particle:
                        # Particle is a Sphere
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = i_particle.position_center[2] + Particle.box[2] * l
                        r = i_particle.radius
                        # Saving the properties of the particles
                        sphereTag = factory.addSphere(xc, yc, zc, r)

                        factory.synchronize()
                        particle_tags.append(
                            gmsh.model.getBoundary((3, sphereTag))[0][1]
                        )
                        gmsh.model.removeEntities((3, sphereTag))
                        phase_dim_tag[str(i_particle.phase)].append(
                            (2, particle_tags[k_particle_image])
                        )

                        factory.synchronize()
                        k_particle_image += 1
                    elif "Ellipsoid" == class_name_i_particle:
                        # Particle is an Ellipsoid
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = i_particle.position_center[2] + Particle.box[2] * l
                        r = 1
                        # Saving the properties of the particles
                        particle_tags.append(factory.addSphere(xc, yc, zc, r))
                        # Creating a sphere without rotation
                        # Rotate the disk
                        factory.synchronize()
                        affine_tags = [(3, particle_tags[k_particle_image])]
                        # affine_tags.extend(
                        #     model.getBoundary([(3, particle_tags[k_particle_image])]))
                        factory.dilate(
                            affine_tags,
                            xc,
                            yc,
                            zc,
                            i_particle.semi_axis_1,
                            i_particle.semi_axis_2,
                            i_particle.semi_axis_3,
                        )
                        factory.rotate(
                            affine_tags,
                            xc,
                            yc,
                            zc,
                            i_particle.rotation_axis[0],
                            i_particle.rotation_axis[1],
                            i_particle.rotation_axis[2],
                            i_particle.angle,
                        )

                        phase_dim_tag[str(i_particle.phase)].append(
                            (3, particle_tags[k_particle_image])
                        )

                        factory.synchronize()
                        k_particle_image += 1

    verticesTags = np.array(
        [
            factory.addPoint(vertex[0], vertex[1], vertex[2])
            for vertex in voronoi.vertices
        ]
    )
    planeSurfaceTags = []
    edgeTags = {}
    for ridge in voronoi.ridge_vertices:
        edgeFaceTags = []
        if -1 in ridge:
            continue
        ridge_out_phase = ridge[-1:] + ridge[0:-1]
        for vertex_1, vertex_2 in zip(ridge, ridge_out_phase):
            if (vertex_1, vertex_2) not in edgeTags or (
                vertex_2,
                vertex_1,
            ) not in edgeTags:
                edgeTags[(vertex_1, vertex_2)] = factory.addLine(
                    verticesTags[vertex_1], verticesTags[vertex_2]
                )
            edgeFaceTags.append(
                edgeTags.get((vertex_1, vertex_2), edgeTags.get((vertex_2, vertex_1)))
            )
        curveLoopTag = factory.addCurveLoop(edgeFaceTags)

        print(edgeFaceTags)
        print(curveLoopTag)
        planeSurfaceTags.append(factory.addPlaneSurface([curveLoopTag]))

    factory.synchronize
    # box_surface = gmsh.model.getBoundary([(3, boxTag)])
    out_dim_tag_3, _ = factory.intersect(
        [(2, planeSurface) for planeSurface in planeSurfaceTags],
        [(3, boxTag)],
        removeObject=True,
        removeTool=False,
    )

    # out_dim_tag4, _ = factory.intersect(
    #     [(3, boxTag)], [(1, edgeTag) for edgeTag in list(edgeTags.values())],
    #     removeObject=False, removeTool=True)

    # print(out_dim_tag_3)
    factory.synchronize()
    voronoi_lines = gmsh.model.getBoundary(out_dim_tag_3, combined=False)
    gmsh.model.removeEntities(out_dim_tag_3)

    # all_voronoi_lines = list(set([voronoi_line[1] for voronoi_line in voronoi_lines] + [edgeTag[1] for edgeTag in out_dim_tag4]))
    voronoiWires = model.addPhysicalGroup(
        1, [lineTag[1] for lineTag in voronoi_lines]
    )  # [(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
    model.setPhysicalName(1, voronoiWires, "Voronoi")
    # voronoiWires = model.addPhysicalGroup(1, [tag[1] for tag in out_dim_tag_3]) #[(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
    # model.setPhysicalName(1, voronoiWires, "Voronoi")

    out_dim_tag, out_dim_tag_map = factory.intersect(
        [(3, boxTag)],
        [(2, particleTag) for particleTag in particle_tags],
        removeObject=False,
        removeTool=True,
    )

    temp = set(out_dim_tag)
    for i_phase in Particle.list_phases:
        phase_dim_tag[i_phase] = [
            value for value in phase_dim_tag[i_phase] if value in temp
        ]

    # factory.synchronize()
    #
    # out_dim_tag_2, out_dim_tag_map2 = factory.fragment(
    #     [(3, boxTag)], out_dim_tag, removeObject=True, removeTool=True)

    # phase_dim_tag[Particle.matrix_phase] = out_dim_tag_2[len(out_dim_tag):]
    # gmsh.model.removeEntities(out_dim_tag_2[len(out_dim_tag):], True)
    materials = []
    for i_phase in Particle.list_phases:
        temp = set(phase_dim_tag[i_phase])
        materials.append([value[1] for value in out_dim_tag if value in temp])

    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of
    # entities
    factory.synchronize()

    for i_phase in range(len(Particle.list_phases)):
        material_tag = model.addPhysicalGroup(2, materials[i_phase])
        model.setPhysicalName(2, material_tag, "Phase " + Particle.list_phases[i_phase])

    points = model.getEntities(0)

    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.03)

    # Generate a 3D mesh
    model.mesh.generate(2)

    # Write the mesh to the .msh file
    meshfile_temp = title + "_temp.vtk"
    meshfile = title + ".vtk"
    gmsh.write(meshfile_temp)

    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()
    os.remove(meshfile_temp)


def plotVoronoi3D(
    particles, voronoi, rve_dims, dir, voronoi_type, save=True, show=True
):
    """Plot the Voronoi for circular particles."""
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", 1)

    # 2D Meshing algorithm
    # --------------------
    # 1 - Mesh Adapt
    # 2 - Automatic
    # 5 - Delaunay (default)
    # 6 - Frontal-Delaunay
    # 7 - BAMG
    # 8 - Frontal-Delaunay for Quads
    # 9 - Packing of Parallelograms
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # 3D Meshing algorithm
    # --------------------
    # 1 - Delaunay (default)
    # 2 - Frontal
    # 7 - MMG3D
    # 9 - R-tree
    # 10 - HXT
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 0)

    # MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Quad/Hex recombination algorithms
    # ---------------------------------
    # 0 - simple
    # 1 - blossom (default)
    # 2 - simple full-quad
    # 3 - blosson full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Number of topological optimization passes of recombined surface meshes (5 by default)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

    # Force recombination in all volumes
    gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

    # Recombination level in 3D
    # -------------------------
    # 0 - hex (default)
    # 1 - hex + prisms
    # 2 - hex + prisms + pyramids
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

    # Recombination conformity type in 3D meshes
    # ------------------------------------------
    # 0 - nonconforming (default)
    # 1 - trihedra
    # 2 - pyramids + trihedra
    # 2 - pyramids + hexSplit + trihedra
    # 4 - hexSplit + trihedra
    gmsh.option.setNumber("Mesh.Recombine3DConformity", 1)

    # Renumber nodes and elements after mesh generation
    gmsh.option.setNumber("Mesh.Renumber", 1)

    # Save all elements even if they do not belong to physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Number of smoothing step applied to the final mesh
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Element order
    gmsh.option.setNumber("Mesh.ElementOrder", 2)

    # Crete second-order nodes by linear interpolation
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Second-order incomplete elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    # ==========================================================================================
    # Generate the finite element mesh
    # ==========================================================================================
    # Define model name

    title = Particle.file_path
    model.add(title)

    particle_tags = []
    k_particle_image = 0
    phase_dim_tag = {phase: [] for phase in Particle.list_phases}
    for i_particle in particles:
        # Running through all the particles
        class_name_i_particle = i_particle.__class__.__name__
        # Saving the class name of the particle as a string
        for j in range(1):
            # Periodic images in the x direction
            for p in range(1):
                # Periodic images in the y direction
                for l in range(1):
                    # Periodic images in the z direction
                    if "CylindricalFiber" == class_name_i_particle:
                        if l != 0:
                            continue
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = 0
                        rx = i_particle.radius
                        ry = i_particle.radius
                        # Speciying the position and the radius of the fibers face
                        face_tag = factory.addDisk(xc, yc, zc, rx, ry)
                        # Saving the properties of the particles
                        if i_particle.direction_fibers == 0:
                            # The fibers run in the x direction
                            factory.rotate(
                                [(2, face_tag)], 0, 0, 0, 0, 1, 0, 3 * np.pi / 2
                            )
                            # Rotating the fiber face to the yz plane as it was ploted
                            # in the xy plane
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], i_particle.length_dir_fibers, 0, 0
                            )
                            # Extruding the fiber from the fiber face in the yz plane in the
                            # x direction
                        elif i_particle.direction_fibers == 1:
                            # The fibers run in the y direction
                            factory.rotate([(2, face_tag)], 0, 0, 0, 1, 0, 0, np.pi / 2)
                            # Rotating the fiber faces to the xz plane as it was ploted
                            # in the xy plane
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], 0, i_particle.length_dir_fibers, 0
                            )
                            # Extruding the fiber from the fiber face in the xz plane in the
                            # y direction
                        elif i_particle.direction_fibers == 2:
                            # The fibers run in the z direction
                            extrusion_tags = factory.extrude(
                                [(2, face_tag)], 0, 0, i_particle.length_dir_fibers
                            )
                            # Extruding the fiber from the fiber face in the xy plane in the
                            # z direction

                        for i_dim_tag in extrusion_tags:
                            if i_dim_tag[0] == 3:
                                particle_tags.append(i_dim_tag[1])
                                break

                        phase_dim_tag[str(i_particle.phase)].append(
                            (3, particle_tags[-1])
                        )

                        factory.synchronize()
                        k_particle_image += 1
                    if "Sphere" == class_name_i_particle:
                        # Particle is a Sphere
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = i_particle.position_center[2] + Particle.box[2] * l
                        r = i_particle.radius
                        # Saving the properties of the particles
                        sphereTag = factory.addSphere(xc, yc, zc, r)

                        factory.synchronize()
                        particle_tags.append(
                            gmsh.model.getBoundary((3, sphereTag))[0][1]
                        )
                        gmsh.model.removeEntities((3, sphereTag))
                        phase_dim_tag[str(i_particle.phase)].append(
                            (2, particle_tags[k_particle_image])
                        )

                        factory.synchronize()
                        k_particle_image += 1
                    elif "Ellipsoid" == class_name_i_particle:
                        # Particle is an Ellipsoid
                        xc = i_particle.position_center[0] + Particle.box[0] * j
                        yc = i_particle.position_center[1] + Particle.box[1] * p
                        zc = i_particle.position_center[2] + Particle.box[2] * l
                        r = 1
                        # Saving the properties of the particles
                        particle_tags.append(factory.addSphere(xc, yc, zc, r))
                        # Creating a sphere without rotation
                        # Rotate the disk
                        factory.synchronize()
                        affine_tags = [(3, particle_tags[k_particle_image])]
                        # affine_tags.extend(
                        #     model.getBoundary([(3, particle_tags[k_particle_image])]))
                        factory.dilate(
                            affine_tags,
                            xc,
                            yc,
                            zc,
                            i_particle.semi_axis_1,
                            i_particle.semi_axis_2,
                            i_particle.semi_axis_3,
                        )
                        factory.rotate(
                            affine_tags,
                            xc,
                            yc,
                            zc,
                            i_particle.rotation_axis[0],
                            i_particle.rotation_axis[1],
                            i_particle.rotation_axis[2],
                            i_particle.angle,
                        )

                        phase_dim_tag[str(i_particle.phase)].append(
                            (3, particle_tags[k_particle_image])
                        )

                        factory.synchronize()
                        k_particle_image += 1

    verticesTags = np.array(
        [
            factory.addPoint(vertex[0], vertex[1], vertex[2])
            for vertex in voronoi.vertices
        ]
    )
    edgeTags = {}
    for i_particle in range(13, len(voronoi.point_region), 27):
        particle_region = voronoi.regions[voronoi.point_region[i_particle]]
        for ridge in voronoi.ridge_vertices:
            if -1 in ridge or any([vertex not in particle_region for vertex in ridge]):
                continue
            ridge_out_phase = ridge[-1:] + ridge[0:-1]
            for vertex_1, vertex_2 in zip(ridge, ridge_out_phase):
                if (vertex_1, vertex_2) not in edgeTags or (
                    vertex_2,
                    vertex_1,
                ) not in edgeTags:
                    edgeTags[(vertex_1, vertex_2)] = factory.addLine(
                        verticesTags[vertex_1], verticesTags[vertex_2]
                    )

    # all_voronoi_lines = list(set([voronoi_line[1] for voronoi_line in voronoi_lines] + [edgeTag[1] for edgeTag in out_dim_tag4]))
    voronoiWires = model.addPhysicalGroup(
        1, list(edgeTags.values())
    )  # [(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
    model.setPhysicalName(1, voronoiWires, "Voronoi")
    # voronoiWires = model.addPhysicalGroup(1, [tag[1] for tag in out_dim_tag_3]) #[(1, all_voronoi_line) for all_voronoi_line in all_voronoi_lines])
    # model.setPhysicalName(1, voronoiWires, "Voronoi")

    materials = []
    for i_phase in Particle.list_phases:
        materials.append(phase_dim_tag[i_phase])

    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of
    # entities
    factory.synchronize()

    for i_phase in range(len(Particle.list_phases)):
        print(materials[i_phase])
        material_tag = model.addPhysicalGroup(
            2, [particle[1] for particle in materials[i_phase]]
        )
        model.setPhysicalName(2, material_tag, "Phase " + Particle.list_phases[i_phase])

    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.03)

    # Generate a 3D mesh
    model.mesh.generate(2)

    # Write the mesh to the .msh file
    meshfile = title + ".msh"
    meshfile_temp = title + "_temp.msh"
    vtkfile = title + ".vtk"
    vtkfile_temp = title + "_temp.vtk"
    gmsh.write(meshfile_temp)
    gmsh.write(vtkfile_temp)

    # Close GMSH
    gmsh.finalize()

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()
    os.remove(meshfile_temp)

    fin = open(vtkfile_temp, "rt")
    fout = open(vtkfile, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()
    os.remove(vtkfile_temp)


def plotVoronoi3DwithIMTspbc(
    particles, voronoi, rve_dims, dir, voronoi_type, save=True, show=True
):
    """Plot the Voronoi for circular particles."""
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", 1)

    # 2D Meshing algorithm
    # --------------------
    # 1 - Mesh Adapt
    # 2 - Automatic
    # 5 - Delaunay (default)
    # 6 - Frontal-Delaunay
    # 7 - BAMG
    # 8 - Frontal-Delaunay for Quads
    # 9 - Packing of Parallelograms
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # 3D Meshing algorithm
    # --------------------
    # 1 - Delaunay (default)
    # 2 - Frontal
    # 7 - MMG3D
    # 9 - R-tree
    # 10 - HXT
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 0)

    # MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Quad/Hex recombination algorithms
    # ---------------------------------
    # 0 - simple
    # 1 - blossom (default)
    # 2 - simple full-quad
    # 3 - blosson full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Number of topological optimization passes of recombined surface meshes (5 by default)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

    # Force recombination in all volumes
    gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

    # Recombination level in 3D
    # -------------------------
    # 0 - hex (default)
    # 1 - hex + prisms
    # 2 - hex + prisms + pyramids
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

    # Recombination conformity type in 3D meshes
    # ------------------------------------------
    # 0 - nonconforming (default)
    # 1 - trihedra
    # 2 - pyramids + trihedra
    # 2 - pyramids + hexSplit + trihedra
    # 4 - hexSplit + trihedra
    gmsh.option.setNumber("Mesh.Recombine3DConformity", 1)

    # Renumber nodes and elements after mesh generation
    gmsh.option.setNumber("Mesh.Renumber", 1)

    # Save all elements even if they do not belong to physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Number of smoothing step applied to the final mesh
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Element order
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Crete second-order nodes by linear interpolation
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Second-order incomplete elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    # ==========================================================================================
    # Generate the finite element mesh
    # ==========================================================================================
    # Define model name

    title = Particle.file_path
    model.add(title)

    boxTag = factory.addBox(0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2])
    # RVE

    verticesTags = np.array(
        [
            factory.addPoint(vertex[0], vertex[1], vertex[2])
            for vertex in voronoi.vertices
        ]
    )
    planeSurfaceTags = []
    planeSurfaceDictTags = {}
    edgeTags = {}
    for ridge in voronoi.ridge_vertices:
        edgeFaceTags = []
        if -1 in ridge:
            continue
        ridge_out_phase = ridge[-1:] + ridge[0:-1]
        for vertex_1, vertex_2 in zip(ridge, ridge_out_phase):
            if (vertex_1, vertex_2) not in edgeTags or (
                vertex_2,
                vertex_1,
            ) not in edgeTags:
                edgeTags[(vertex_1, vertex_2)] = factory.addLine(
                    verticesTags[vertex_1], verticesTags[vertex_2]
                )
            edgeFaceTags.append(
                edgeTags.get((vertex_1, vertex_2), edgeTags.get((vertex_2, vertex_1)))
            )
        curveLoopTag = factory.addCurveLoop(edgeFaceTags)

        planeSurfaceTags.append(factory.addPlaneSurface([curveLoopTag]))
        planeSurfaceDictTags[tuple(ridge)] = planeSurfaceTags[-1]

    factory.synchronize()
    # box_surface = gmsh.model.getBoundary([(3, boxTag)])
    out_dim_tag_3, _ = factory.fragment(
        [(2, planeSurface) for planeSurface in planeSurfaceTags],
        [(3, boxTag)],
        removeObject=True,
        removeTool=True,
    )

    factory.synchronize()
    number_cells = 0
    for index, i_voronoi_cell in enumerate(out_dim_tag_3):
        if i_voronoi_cell[0] == 3:
            number_cells += 1
            material_tag = model.addPhysicalGroup(3, [i_voronoi_cell[1]])
            model.setPhysicalName(3, material_tag, "Cell " + str(number_cells))

    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.03)

    # Generate a 3D mesh
    model.mesh.generate(3)

    # Write the mesh to the .msh file
    meshfile_temp = title + "_temp.msh"
    meshfile = title + ".msh"
    vtk_file_temp = title + "_temp.msh"
    vtk_file = title + ".vtk"
    gmsh.write(meshfile_temp)
    gmsh.write(vtk_file_temp)

    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()

    os.remove(meshfile_temp)

    fin = open(vtk_file_temp, "rt")
    fout = open(vtk_file, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()

    os.remove(vtk_file_temp)

    dataName = "test"
    dataType = "float"
    numComp = "1"

    fin = open(vtk_file, "rt")

    element_cell = []
    save = 0
    for line in fin:
        if line.startswith("CELL_DATA"):
            save = True
            continue
        if save:
            print(line.rstrip("\n"))
            element_cell.append(line.rstrip("\n"))

    fin.close()

    colors = np.arange(number_cells)
    np.random.shuffle(colors)

    with open(vtk_file, "a") as msh_vtk:
        msh_vtk.write("\n\nSCALARS {0} {1} {2}".format(dataName, dataType, numComp))
        msh_vtk.write("\nLOOKUP_TABLE default")
        for cell_id in element_cell[2:]:
            msh_vtk.write("\n{0}".format(colors[int(cell_id) - 1]))


def plotVoronoi3DwithIMTs(
    particles, voronoi, rve_dims, IMTs, dir, voronoi_type, save=True, show=True
):
    """Plot the Voronoi for circular particles."""
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", 1)

    # 2D Meshing algorithm
    # --------------------
    # 1 - Mesh Adapt
    # 2 - Automatic
    # 5 - Delaunay (default)
    # 6 - Frontal-Delaunay
    # 7 - BAMG
    # 8 - Frontal-Delaunay for Quads
    # 9 - Packing of Parallelograms
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # 3D Meshing algorithm
    # --------------------
    # 1 - Delaunay (default)
    # 2 - Frontal
    # 7 - MMG3D
    # 9 - R-tree
    # 10 - HXT
    gmsh.option.setNumber("Mesh.Algorithm3D", 2)

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 0)

    # MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Quad/Hex recombination algorithms
    # ---------------------------------
    # 0 - simple
    # 1 - blossom (default)
    # 2 - simple full-quad
    # 3 - blosson full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Number of topological optimization passes of recombined surface meshes (5 by default)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

    # Force recombination in all volumes
    gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

    # Recombination level in 3D
    # -------------------------
    # 0 - hex (default)
    # 1 - hex + prisms
    # 2 - hex + prisms + pyramids
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

    # Recombination conformity type in 3D meshes
    # ------------------------------------------
    # 0 - nonconforming (default)
    # 1 - trihedra
    # 2 - pyramids + trihedra
    # 2 - pyramids + hexSplit + trihedra
    # 4 - hexSplit + trihedra
    gmsh.option.setNumber("Mesh.Recombine3DConformity", 0)

    # Renumber nodes and elements after mesh generation
    gmsh.option.setNumber("Mesh.Renumber", 1)

    # Save all elements even if they do not belong to physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Number of smoothing step applied to the final mesh
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Element order
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Crete second-order nodes by linear interpolation
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Second-order incomplete elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    # ==========================================================================================
    # Generate the finite element mesh
    # ==========================================================================================
    # Define model name

    title = dir + "voronoi_wIMTs"
    model.add(title)

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
        [
            factory.addPoint(vertex[0], vertex[1], vertex[2])
            for vertex in voronoi.vertices
        ]
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
            ridge_out_phase = ridge[-1:] + ridge[0:-1]
            for vertex_1, vertex_2 in zip(ridge, ridge_out_phase):
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
    out_dim_tag_3, _ = factory.fragment(
        [(2, planeSurface) for planeSurface in planeSurfaceTags],
        [(3, boxTag)],
        removeObject=False,
        removeTool=True,
    )

    print(out_dim_tag_3)

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
        print(volumeCell)
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
        print(i_voronoi_cell)
        factory.synchronize()
        material_tag = model.addPhysicalGroup(
            3, [cell[1] for cell in i_voronoi_cell if cell[0] == 3]
        )
        model.setPhysicalName(3, material_tag, "Cell " + str(number_cells))
        number_cells += 1
    print(gmsh.model.getEntities(3))
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
    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)

    # Generate a 3D mesh
    model.mesh.generate(3)

    # Write the mesh to the .msh file
    meshfile_temp = title + "_temp.msh"
    meshfile = title + ".msh"
    vtk_file_temp = title + "_temp.vtk"
    vtk_file = title + ".vtk"
    gmsh.write(meshfile_temp)
    gmsh.write(vtk_file_temp)

    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()

    os.remove(meshfile_temp)

    fin = open(vtk_file_temp, "rt")
    fout = open(vtk_file, "wt")

    for line in fin:
        fout.write(line.replace(",", "."))

    fin.close()
    fout.close()

    os.remove(vtk_file_temp)

    dataType = "float"
    numComp = "1"

    fin = open(vtk_file, "rt")

    element_cell = []
    save = 0
    for line in fin:
        if line.startswith("CELL_DATA"):
            save = True
            continue
        if save:
            element_cell.append(line.rstrip("\n"))

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
                msh_vtk.write("\n{0}".format(IMTs[int(cell_id) - 1][i_IMT]))

    for i_order in range(7):

        fig, ax, (w_fig, h_fig) = create_figure(nrows=3, ncols=2)

        ax = plt.gca()

        N = len(particles)

        if i_order == 0:
            plt.hist(
                np.abs(np.array(IMTs)[:, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
            )
            ax.set_xlabel(r"Surface Area")
        else:
            plt.hist(
                np.abs(np.array(IMTs)[:, i_order]),
                color=(68 / 255, 119 / 255, 170 / 255, 1),
                range=(0, 1),
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
            plt.axvline(
                np.mean(np.abs(np.array(IMTs)[:, i_order])), color="k", linestyle="--"
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
