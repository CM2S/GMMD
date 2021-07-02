"""Module for the statistical analysis of microstructures."""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import pickle
import os
import sys
import numpy as np

from PIL import Image

from geommicgen.iofuncs.file_handling import create_design_point_results_directory

from geommicgen.postproc.plotfuncs.plotting_functions import (
    plot_nearest_neighbor_dist,
    plot_ripleys_k_func,
    plot_two_point_correlation,
)


class MicrostructureImage:
    """Class for a microstructure image.

    Attributes
    ----------
    rve_dims: np.array(float)
        Array containing the dimensions of the RVE.

    image: `.Image`
        Image object of the microstructure. Assumed to be 8-bit.

    image_resolution: tuple
        Resolution of the microstructure image.

    average_radius: float
        Average radius of the particles.
    """

    def __init__(self, rve_dims, file_path, average_radius):
        """Initialize a `.MicrostructureImage` object.

        Parameters
        ----------
        rve_dims: np.array(float)
            Array containing the dimensions of the RVE.

        file_path: str
            File path to the image of the microstructure.

        average_radius: float
            Average radius of the particles.
        """
        self.rve_dims = np.array(rve_dims)
        self.image = Image.open(file_path)
        # convert to black and white
        self.image.convert("L")
        self.image_resolution = np.array(self.image.size)
        self.average_radius = average_radius

    def inside_particle_phase(self, pts):
        """Check if *pts* is inside the particle phase.

        Parameters
        ----------
        pts: list(array)
            List of points whose position relative to the particle phase we want to know.

        Returns
        -------
        pt_in: list(int)
            List of 1s and 0s in accordance with the position of the respective point in
            *pts*.
        """
        pt_in = [0 for _ in pts]
        for i_ind_pt, i_pt in enumerate(pts):
            i_pt_in_box = i_pt - np.floor(i_pt / self.rve_dims) * self.rve_dims
            pt_coord_image = self.image_resolution / self.rve_dims * i_pt_in_box
            pixel = self.image.getpixel(tuple(pt_coord_image))
            if isinstance(pixel, tuple):
                pt_in[i_ind_pt] = pixel[0] == 0
            else:
                pt_in[i_ind_pt] = pixel == 255

        return pt_in


def remove_particles_at_boundary(particles, rve_dims):
    """Remove from the microstructure the particles intersecting the boundary of the box."""
    rem_particles = list(particles)
    # copy of all the particles that we can mutate
    for i_particle in rem_particles:
        all_lims = []
        for i_dim in range(particles[0].dim):
            direction = np.full(particles[0].dim, 0)
            direction[i_dim] = 1
            all_lims.append(
                [
                    i_particle.support_function(-1 * direction)[i_dim],
                    i_particle.support_function(direction)[i_dim],
                ]
            )
        print(all_lims)
        if any(
            [
                i_lim[0] < 0 or i_lim[1] > rve_dims[i_dim]
                for (i_dim, i_lim) in enumerate(all_lims)
            ]
        ):
            rem_particles.remove(i_particle)

    return rem_particles


def adjust_rve_dims(particles):
    """Ajust the RVE and return the new dimensions.

    Adjust the postion of the particles and compute the new dimensions of the RVE, so that
    its boundaries touch the outermost particles in each direction.

    Parameters
    ---------
    particles: list(`.Particle`)
        List of particles in RVE.

    Returns
    -------
    new_rve_dims: list(float)
        New RVE dimensions.
    """
    # Collecting all extrema of the particles in all Cartesian directions
    # --------------------------------------------------------------------------------------
    all_bound = [[[], []] for _ in range(particles[0].dim)]
    for i_particle in particles:
        for i_dim in range(particles[0].dim):
            direction = np.full(particles[0].dim, 0)
            direction[i_dim] = 1
            all_bound[i_dim][0].append(
                i_particle.support_function(-1 * direction)[i_dim]
            )
            all_bound[i_dim][1].append(i_particle.support_function(direction)[i_dim])

    # Obtaining the maximum and minimum values of the extrema
    # --------------------------------------------------------------------------------------
    max_bound = []
    min_bound = []
    for i_dim in range(particles[0].dim):
        min_bound.append(np.min(all_bound[i_dim][0]))
        max_bound.append(np.max(all_bound[i_dim][1]))

    # Offsetting the particles and computing the new rve dims
    # --------------------------------------------------------------------------------------
    # so that the origin coincides with the minimum bound on each axis
    offset = np.array(min_bound)
    for i_particle in particles:
        i_particle.position_center -= offset

    new_rve_dims = [
        max_bound[i_dim] - min_bound[i_dim] for i_dim in range(particles[0].dim)
    ]
    return new_rve_dims


def two_point_correlation(
    microstructure, max_radius=8, n_samples=5000, n_points=50, **kwargs
):
    """Compute the two-point correlation function for *microstructure*.

    This function computes the two point correlation function for a 2D microstructure, given
    as  an object possessing the attribute *rve_dims* and the method
    *inside_particle_phase*.

    It is defined as the probability of two random points at some distance r/R, where r is
    the distance betweeen them and R is the radius of the particles, both landing on the
    particle phase.

    Parameters
    ----------
    microstructure: `.Object`
        Microstructure object.

    max_radius: optional, float
        Maximum relative radius (r/R) in the computation of the two point correlation
        function.

    n_samples: optional, int
        Number of points used to compute the value of the two point correlation function for
        each value of the relative radius (r/R).

    n_points: optional, int
        Number of points of the two point correlation function computed, between 0 and
        *max_radius*.

    Keyword Parameters
    ------------------
    vec_direction: optional, array
        Preferencial direction. Used when the material is not isotropic.

    Returns
    -------
    two_point_correlation_vals: array(float)
        Values of the two point correlation function.

    Raises
    -------
    ValueError:
        If the vector direction supplied and the RVE dimensions are incompatible.
    """

    def random_unit_vec(dim):
        """Random unit vector."""
        if dim == 2:
            theta = np.random.uniform(0, 2 * np.pi)
            random_unit_vec = np.array([np.cos(theta), np.sin(theta)])
        else:
            raise ValueError("Dimensions not supported.")

        return random_unit_vec

    rve_dims = microstructure.rve_dims
    two_point_correlation_vals = [None for _ in range(n_points * max_radius)]
    if isinstance(microstructure, MicrostructureImage):
        radius = microstructure.average_radius
    else:
        radius = np.mean([i_particle.radius for i_particle in microstructure.particles])
    radii_vec = np.arange(0, max_radius, 1 / n_points)
    if "vec_direction" in kwargs:
        pref_direction = True
        unit_vec = kwargs["vec_direction"] / np.linalg.norm(kwargs["vec_direction"])
        if len(unit_vec) != len(rve_dims):
            raise ValueError(
                """The vector direction supplied and the RVE dimensions are incompatible:
                 {0}, {1}""".format(
                    unit_vec, rve_dims
                )
            )
    else:
        pref_direction = False
    for i_ind_length, i_length in enumerate(
        radius * np.arange(0, max_radius, 1 / n_points)
    ):

        all_pts = [None for _ in range(2 * n_samples)]
        for i_pt in range(n_samples):
            # Using n_samples random pts to compute each pt of the two point correlation
            # function
            pt_1 = np.random.uniform(0, 1, np.shape(rve_dims)) * np.array(rve_dims)
            all_pts[2 * i_pt] = pt_1
            if pref_direction:
                pt_2 = pt_1 + i_length * unit_vec
            else:
                pt_2 = pt_1 + i_length * random_unit_vec(len(rve_dims))
            all_pts[2 * i_pt + 1] = pt_2

        all_pts_inside = microstructure.inside_particle_phase(all_pts)
        line_seg_inside = [
            (all_pts_inside[2 * i_ind] + all_pts_inside[2 * i_ind + 1]) // 2
            for i_ind in range(n_samples)
        ]
        two_point_correlation_vals[i_ind_length] = np.sum(line_seg_inside) / n_samples

        # import matplotlib.pyplot as plt
        #
        # from postproc.plotfuncs.plotting_functions import plot_particles_2d
        #
        # plot_particles_2d(
        #     microstructure.particles, microstructure.rve_dims, "", save=False
        # )
        # all_pts_inside = np.array(all_pts_inside)
        # all_pts = np.array(all_pts)
        # plt.scatter(all_pts[:, 0], all_pts[:, 1])
        # plt.scatter(
        #     all_pts[:, 0][all_pts_inside.astype(np.bool)],
        #     all_pts[:, 1][all_pts_inside.astype(np.bool)],
        #     c="r",
        # )
        # for i_pt in range(n_samples):
        #     plt.plot(
        #         all_pts[2 * i_pt : 2 * i_pt + 2, 0], all_pts[2 * i_pt : 2 * i_pt + 2, 1]
        #     )
        #
        # plt.show()

    return two_point_correlation_vals, radii_vec


def ripleys_k_func(microstructure, max_radius=10, n_points=20):
    """Compute Ripley's K function for *microstructure*.

    Ripley's K function is defined as the expected number of extra events within distance
    t of a randomly chosen event divided by the number density.

    It is estimated for each distance t summing for all the points the number of other
    points that lie in the disk/shpere of radius t. An edge correction factor is used which
    is fraction of the area corresponding to the disk/sphere inside the domain.

    Parameters
    ----------
    microstructure: `.Object`
        Microstructure object.

    max_radius: optional, float
        Maximum relative radius (r/R) in the computation of the two point correlation
        function.

    Returns
    -------
    k_ripleys_func_vals: array(float)
        Values of Ripley's K function.
    """

    def ripleys_k_func_edge_corr(center_pt, radius, box):
        """Compute the correction factor for Ripley's K function.

        This function computes the correction factor Ripley's K function for a disk/sphere
        centered at *center_pt* with radius *radius* in a box *box*, as the fraction of its
        area/volume inside the box.
        """

        def intersection_area_search_box(center_pt, radius, box, n_samples=200):

            # Generating random points inside the disk
            # ------------------------------------------------------------------------------
            points = []
            for _ in range(n_samples):
                z = np.array([0.0, 0.0])
                z[0] = np.random.normal()
                z[1] = np.random.normal()
                r = np.random.uniform() ** (1 / 2)
                R = np.linalg.norm(z)
                x_loc = r * radius * z[0] / R
                y_loc = r * radius * z[1] / R
                [x_glob, y_glob] = np.array([x_loc, y_loc]) + center_pt
                points.append(np.array([x_glob, y_glob]))

            # Estimating the intersection area using a Monte Carlo method
            # ------------------------------------------------------------------------------
            points_in = 0
            for i_point in points:
                if np.all(
                    np.logical_and(
                        np.array([0, 0]) <= i_point, i_point <= np.array(box)
                    )
                ):
                    points_in += 1
            int_area = points_in / len(points) * (np.pi * radius ** 2)
            # import matplotlib.pyplot as plt

            # plt.figure()
            # points = np.array(points)
            # plt.scatter(points[:, 0], points[:, 1])
            # plt.show()
            return int_area

        max_length = np.max(box) / 2
        if radius >= max_length and False:
            edge_correction = 1
        else:
            area_outside = 0
            if len(center_pt) == 2:
                # for i_dim in range(2):
                #     if (
                #         center_pt[i_dim] + radius > box[i_dim]
                #         or center_pt[i_dim] - radius < 0
                #     ):
                #         # lower or upper
                #         min_dist_to_bound = np.abs(
                #             np.min([center_pt[i_dim], box[i_dim] - center_pt[i_dim]])
                #         )
                #         # Distance from the centerpoint to  the closest box boundary
                #         base_triangle = np.sqrt(radius ** 2 - min_dist_to_bound ** 2)
                #         area_triangle = base_triangle * min_dist_to_bound
                #         angle = np.arctan(base_triangle / min_dist_to_bound)
                #         area_sector = angle * radius ** 2
                #         area_outside += area_sector - area_triangle
                # edge_correction = 1 - area_outside / (np.pi * radius ** 2)
                edge_correction = intersection_area_search_box(
                    center_pt, radius, box
                ) / (np.pi * radius ** 2)
            if len(center_pt) == 3:
                for i_dim in range(3):
                    if (
                        center_pt[i_dim] + radius > box[i_dim]
                        or center_pt[i_dim] - radius < 0
                    ):
                        # lower or upper
                        base_cone = (
                            np.pi
                            * (
                                np.sqrt(
                                    radius ** 2 - (center_pt[i_dim] - box[i_dim]) ** 2
                                )
                                / 2
                            )
                            ** 2
                        )
                        volume_cone = (
                            1 / 3 * base_cone * np.abs(box[i_dim] - center_pt[i_dim])
                        )
                        volume_sector = (
                            radius ** 3
                            * 2
                            / 3
                            * np.pi(1 - np.abs(box[i_dim] - center_pt[i_dim]) / radius)
                        )
                        area_outside += volume_sector - volume_cone
                    edge_correction = 1 - area_outside / (np.pi * 4 / 3 * radius ** 3)

        return edge_correction

    rem_particles = remove_particles_at_boundary(
        microstructure.particles, microstructure.rve_dims
    )
    adj_rve_dims = adjust_rve_dims(rem_particles)

    from postproc.plotfuncs.plotting_functions import plot_particles_2d

    plot_particles_2d(rem_particles, adj_rve_dims, "", save=False, show=False)

    # plt.show()
    radius = np.mean([i_particle.radius for i_particle in rem_particles])
    n_part = len(rem_particles)
    # dist_part = [0 for _ in np.arange(np.ceil(n_part * (n_part - 1) / 2))]
    # correction = [1 for _ in np.arange(np.ceil(n_part * (n_part - 1) / 2))]
    dist_part = [0 for _ in range(n_part ** 2 - n_part)]
    correction = [1 for _ in range(n_part ** 2 - n_part)]
    k_pair = 0
    for i_ind_part, i_particle in enumerate(rem_particles):
        for j_ind_part, j_particle in enumerate(rem_particles):
            if j_ind_part == i_ind_part:
                continue
            dist_part[k_pair] = np.linalg.norm(
                i_particle.position_center - j_particle.position_center
            )
            if dist_part[k_pair] < radius:
                print(i_particle.position_center, j_particle.position_center)
            # print(k_pair)
            # print(dist_part[k_pair])
            correction[k_pair] = ripleys_k_func_edge_corr(
                i_particle.position_center,
                dist_part[k_pair],
                adj_rve_dims,
            )
            # print("correction", correction[k_pair])
            k_pair += 1
    k_ripleys_func_vals = np.array([0.0 for _ in range(n_points * max_radius)])
    # print(any(dist_part < radius))
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.hist(dist_part / radius)
    # plt.show()
    for i_ind_length, i_length in enumerate(np.arange(0, max_radius, 1 / n_points)):
        # for _ in range(1):
        #     i_ind_length = 0
        #     i_length = 0
        # print(i_ind_length, i_length)
        current_val = 0
        for j_dist, j_correction in zip(dist_part, correction):
            # print(j_dist, j_correction)
            if j_dist < i_length * radius:
                current_val += 1 / j_correction / n_part

        print(current_val)
        k_ripleys_func_vals[i_ind_length] = current_val
        print(k_ripleys_func_vals[i_ind_length])

    k_ripleys_func_vals = k_ripleys_func_vals * np.prod(adj_rve_dims) / n_part

    print(k_ripleys_func_vals)
    # k_ripleys_func_vals = np.sqrt(k_ripleys_func_vals / np.pi) - radius * np.arange(
    #     0, max_radius, 1 / n_points
    # )
    print(radius * np.arange(0, max_radius, 1 / n_points))
    print(k_ripleys_func_vals)
    print(dist_part)
    print(correction)
    return k_ripleys_func_vals, radius * np.arange(0, max_radius, 1 / n_points)


def nearest_neighbor_dist(microstructure):
    """Compute the nearest neighbor distance function for *microstructure*.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure whose nearest neighbor function we are going to compute.

    Returns
    -------
    nearest_neighbor_dist_vals: array
        Array containing the values of the nearest neighbor distance function.
    """
    # cell_list = CellList()
    # cell_list.box = np.array(microstructure.rve_dims)
    # cell_list.new_list(microstructure.particles)

    radius = np.mean([i_particle.radius for i_particle in microstructure.particles])
    nearest_neighbor_dist_vals = []
    already_computed = []
    for i_particle_ind, i_particle in enumerate(microstructure.particles):
        if i_particle_ind in already_computed:
            continue
        nearest_neighbor_dist_vals_i = []
        for j_particle_ind, j_particle in enumerate(microstructure.particles):
            if j_particle_ind == i_particle_ind:
                continue
            nearest_neighbor_dist_vals_i.append(
                np.linalg.norm(i_particle.position_center - j_particle.position_center)
            )
        ind_min = np.argmin(nearest_neighbor_dist_vals_i)
        nearest_neighbor_dist_vals.append(
            nearest_neighbor_dist_vals_i[ind_min] / radius
        )
        already_computed.append(ind_min)

    return nearest_neighbor_dist_vals


def do_stat_analysis(microstructure, sample_dir, stat_options):
    """Do the statistical analysis of *microstructure*.

    The statistical functions available are the two point correlation function, Ripleys's K
    function and the nearest neighbor function.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be analyzed.

    stat_options: set(str)
        Options for the statistical analysis.
        Options are {"stat_nearest_neighbor", "stat_ripleys_k", "stat_two_pt_corr"}.
    """
    stat_anal_results_dir = os.path.join(sample_dir, "stat_analysis_results")
    os.makedirs(stat_anal_results_dir)
    stat_results = {}
    # Creating a directory for the results

    # Statistical analysis
    # --------------------------------------------------------------------------------------
    if "stat_nearest_neighbor" in stat_options:
        nearest_neighbor_dist_vals = nearest_neighbor_dist(microstructure)
        stat_results["stat_nearest_neighbor"] = nearest_neighbor_dist_vals
        plot_nearest_neighbor_dist(
            nearest_neighbor_dist_vals, results_dir=stat_anal_results_dir
        )

    if "stat_ripleys_k" in stat_options:
        k_ripleys_func_vals, radii_vec = ripleys_k_func(microstructure)
        stat_results["stat_ripleys_k"] = [k_ripleys_func_vals, radii_vec]
        plot_ripleys_k_func(
            k_ripleys_func_vals, radii_vec, results_dir=stat_anal_results_dir
        )

    if "stat_two_pt_corr" in stat_options:
        two_point_correlation_vals, radii_vec = two_point_correlation(
            microstructure, max_radius=2, n_points=100
        )
        stat_results["stat_two_pt_corr"] = [two_point_correlation_vals, radii_vec]
        plot_two_point_correlation(
            two_point_correlation_vals, radii_vec, results_dir=stat_anal_results_dir
        )

    # Saving the results
    # --------------------------------------------------------------------------------------
    pickle.dump(
        stat_results,
        open(os.path.join(stat_anal_results_dir, "stat_results.stat"), "wb"),
    )


if __name__ == "__main__":
    print(os.path.basename(sys.argv[1]))
    filename, ext = os.path.splitext(os.path.basename(sys.argv[1]))
    if ext == ".png":
        print("here")
        # current_mic = MicrostructureImage([1, 1], sys.argv[1], 0.052582841614906825 / 2)
        current_mic = MicrostructureImage([1, 1], sys.argv[1], 0.3474627652491758 / 2)
        results_folder = create_design_point_results_directory(
            os.path.dirname(sys.argv[1]), filename
        )
        do_stat_analysis(current_mic, results_folder, {"stat_two_pt_corr"})
