"""Module for the statistical analysis of microstructures."""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import os
import sys
import numpy as np

from PIL import Image

from geommicgen.iofuncs.file_handling import create_design_point_results_directory
from geommicgen.microstructure.microstructure import Microstructure, unit_scale

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


STAT_FILE_NAME = "stat_results.npz"
# File of a sample the statistical descriptors are written into


def crosses_boundary(particle, rve_dims):
    """
    Say whether a particle reaches past a face of the box.

    Parameters
    ----------
    particle: `.Particle`
        Particle to be placed.

    rve_dims: list
        Dimensions of the box in each spatial direction.

    Returns
    -------
    bool
        True when the particle reaches past one of the faces.
    """
    for i_dim in range(particle.dim):
        direction = np.full(particle.dim, 0)
        direction[i_dim] = 1
        lower = particle.support_function(-1 * direction)[i_dim]
        upper = particle.support_function(direction)[i_dim]
        if lower < 0 or upper > rve_dims[i_dim]:
            return True

    return False


def remove_particles_at_boundary(particles, rve_dims):
    """
    Give the particles that do not intersect the boundary of the box.

    Parameters
    ----------
    particles: list(`.Particle`)
        Particles of the microstructure.

    rve_dims: list
        Dimensions of the box in each spatial direction.

    Returns
    -------
    list
        The particles that lie whole inside the box.
    """
    return [
        i_particle
        for i_particle in particles
        if not crosses_boundary(i_particle, rve_dims)
    ]
    # Built anew rather than removed from while it is walked, which skipped the
    # particle after every one that was taken out: of two that cross the boundary one
    # after the other, the second was left in, and the statistics counted it


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

    adjusted_centers: list(array)
        Position of the center of each particle in the adjusted RVE.
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
    adjusted_centers = [i_particle.position_center - offset for i_particle in particles]
    # These are the particles of the microstructure itself, so the shift is returned
    # rather than applied to them

    new_rve_dims = [
        max_bound[i_dim] - min_bound[i_dim] for i_dim in range(particles[0].dim)
    ]
    return new_rve_dims, adjusted_centers


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

    def random_unit_vecs(dim, how_many):
        """Directions drawn at random, one per row."""
        if dim != 2:
            raise ValueError("Dimensions not supported.")
        angles = np.random.uniform(0, 2 * np.pi, how_many)

        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    if isinstance(microstructure, Microstructure):
        microstructure = microstructure.scaled(unit_scale(microstructure.rve_dims))
    # The points are thrown at a copy of the microstructure brought to a shortest side
    # of one, so that they are the same points of it in any units: the test of a
    # sphere lets in a point a length of 1e-3 outside it, which in a micrometre RVE
    # made most of the box particle. What is computed is a probability against a
    # ratio of lengths, so nothing is brought back
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
    first_points = np.random.uniform(0, 1, (n_samples, len(rve_dims))) * np.array(
        rve_dims
    )
    first_inside = np.asarray(
        microstructure.inside_particle_phase(first_points), dtype=bool
    )
    # The first point of a pair does not depend on the separation asked about, so it is
    # drawn and looked up once for every radius instead of afresh for each: the same
    # estimate of half the work, and a curve whose points are read against each other

    for i_ind_length, i_length in enumerate(
        radius * np.arange(0, max_radius, 1 / n_points)
    ):
        if pref_direction:
            directions = np.broadcast_to(unit_vec, first_points.shape)
        else:
            directions = random_unit_vecs(len(rve_dims), n_samples)
        second_points = first_points + i_length * directions
        second_inside = np.asarray(
            microstructure.inside_particle_phase(second_points), dtype=bool
        )
        two_point_correlation_vals[i_ind_length] = (
            np.count_nonzero(first_inside & second_inside) / n_samples
        )
        # Both ends of the segment on the particle phase, which is what the function is

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


def disk_area_in_corner(radius, width, height):
    """
    Give the area of a disk at the origin that lies in the corner *[0, w] x [0, h]*.

    Parameters
    ----------
    radius: array
        Radius of the disk, one per corner asked about.

    width: array
        Distance to the vertical side of the corner, positive.

    height: array
        Distance to the horizontal side of the corner, positive.

    Returns
    -------
    array
        The area of the quarter disk that falls inside each corner.
    """
    radius = np.asarray(radius, dtype=float)
    width = np.minimum(np.asarray(width, dtype=float), radius)
    height = np.asarray(height, dtype=float)
    crossing = np.sqrt(np.maximum(radius**2 - height**2, 0.0))
    crossing = np.minimum(crossing, width)
    # Where the circle crosses the horizontal side, and never past the vertical one

    def under_arc(bound):
        """Area under the arc from the vertical axis out to a bound."""
        safe = np.divide(bound, radius, out=np.zeros_like(bound), where=radius > 0)

        return (
            bound * np.sqrt(np.maximum(radius**2 - bound**2, 0.0))
            + radius**2 * np.arcsin(np.clip(safe, -1.0, 1.0))
        ) / 2.0

    return height * crossing + under_arc(width) - under_arc(crossing)
    # A rectangle under the arc up to the crossing, and the integral of the arc from
    # there on: the exact area, where this was estimated by throwing two hundred
    # points at each of the pairs of particles


def ripleys_k_edge_correction(centers, radii, rve_dims):
    """
    Give the share of a disk around each centre that lies inside the box.

    Parameters
    ----------
    centers: array
        Centres of the disks, of shape *(n, 2)*, inside the box.

    radii: array
        Radius of each disk.

    rve_dims: array
        Dimensions of the box.

    Returns
    -------
    array
        The fraction of the area of each disk that falls inside the box, which is what
        the count of a pair at that distance is weighted by.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    left, bottom = centers[:, 0], centers[:, 1]
    right, top = rve_dims[0] - left, rve_dims[1] - bottom
    inside = (
        disk_area_in_corner(radii, left, bottom)
        + disk_area_in_corner(radii, left, top)
        + disk_area_in_corner(radii, right, bottom)
        + disk_area_in_corner(radii, right, top)
    )
    # The centre is inside the box, so the box around it is four corners, and the disk
    # in each of them is a quarter disk clipped by two sides

    area = np.pi * radii**2

    return np.divide(inside, area, out=np.ones_like(inside), where=area > 0)


def ripleys_k_func(microstructure, max_radius=10, n_points=20):
    """Compute Ripley's K function for *microstructure*.

    Ripley's K function is defined as the expected number of extra events within distance
    t of a randomly chosen event divided by the number density.

    It is estimated for each distance t summing for all the points the number of other
    points that lie in the disk of radius t. Each pair counts for the reciprocal of the
    share of the disk of its own separation that lies inside the box, which is the
    correction for the pairs the box cuts off.

    Parameters
    ----------
    microstructure: `.Object`
        Microstructure object.

    max_radius: optional, float
        Maximum relative radius (r/R) in the computation of the function.

    n_points: optional, int
        Radii per unit of relative radius the function is given at.

    Returns
    -------
    tuple
        Values of Ripley's K function, and the radii they are given at.

    Raises
    ------
    NotImplementedError:
        If the microstructure is three dimensional.
    """
    rem_particles = remove_particles_at_boundary(
        microstructure.particles, microstructure.rve_dims
    )
    adj_rve_dims, adj_centers = adjust_rve_dims(rem_particles)
    if len(adj_rve_dims) != 2:
        raise NotImplementedError(
            "Ripley's K function is computed for two dimensional microstructures "
            "alone: the correction for the edge of the box is the share of a disk "
            "inside a rectangle, and the share of a sphere inside a box is not "
            "written. Ask for the other descriptors, or for a two dimensional "
            "microstructure."
        )
    # It raised where a three dimensional microstructure reached the plotting of a two
    # dimensional one, and the branch meant for three dimensions called the number pi

    adj_centers = np.asarray(adj_centers, dtype=float)
    radius = np.mean([i_particle.radius for i_particle in rem_particles])
    n_part = len(rem_particles)

    separations = np.linalg.norm(
        adj_centers[:, None, :] - adj_centers[None, :, :], axis=2
    )
    pairs = ~np.eye(n_part, dtype=bool)
    dist_part = separations[pairs]
    centers_of_pairs = np.repeat(adj_centers, n_part - 1, axis=0)
    # Every ordered pair of distinct particles, and the centre of the first of each

    correction = ripleys_k_edge_correction(
        centers_of_pairs, dist_part, adj_rve_dims
    )
    weights = 1.0 / np.where(correction > 0, correction, 1.0) / n_part

    radii_vec = radius * np.arange(0, max_radius, 1 / n_points)
    order = np.argsort(dist_part)
    running = np.concatenate([[0.0], np.cumsum(weights[order])])
    counted = np.searchsorted(dist_part[order], radii_vec, side="left")
    k_ripleys_func_vals = running[counted] * np.prod(adj_rve_dims) / n_part
    # The pairs closer than each radius are a prefix of the sorted separations, so the
    # sum over them is read off one running total instead of walking every pair again
    # for every radius

    return k_ripleys_func_vals, radii_vec


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


def do_stat_analysis(microstructure, sample_dir, stat_options, seed=None):
    """Do the statistical analysis of *microstructure*.

    The statistical functions available are the two point correlation function, Ripleys's K
    function and the nearest neighbor function. The values are written into an .npz of
    the sample, one array per descriptor, and plotted beside it.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be analyzed.

    stat_options: set(str)
        Options for the statistical analysis.
        Options are {"stat_nearest_neighbor", "stat_ripleys_k", "stat_two_pt_corr"}.

    seed: int
        Seed of the draws Ripley's K function and the two point correlation function
        estimate with. None leaves the generator where it is, and the numbers then
        differ from one run to the next.

    Returns
    -------
    dict
        The values of every descriptor computed, as they are written.
    """
    stat_anal_results_dir = os.path.join(sample_dir, "stat_analysis_results")
    os.makedirs(stat_anal_results_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
    # Two of the three descriptors are Monte Carlo estimates -- Ripley's K function
    # draws points to find how much of a disk falls inside the box, and the two point
    # correlation draws the pairs of points it correlates -- so the same microstructure
    # gives numbers a fraction of a percent apart on every run unless the draws are
    # seeded, where a generation is repeatable through Fixed_Seed

    stat_results = {}
    # Creating a directory for the results
    # Each descriptor puts its values under its own name, and the radii they are given
    # at under that name followed by _radii, which is what `STAT_FILE_NAME` holds

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
        stat_results["stat_ripleys_k"] = k_ripleys_func_vals
        stat_results["stat_ripleys_k_radii"] = radii_vec
        plot_ripleys_k_func(
            k_ripleys_func_vals, radii_vec, results_dir=stat_anal_results_dir
        )

    if "stat_two_pt_corr" in stat_options:
        two_point_correlation_vals, radii_vec = two_point_correlation(
            microstructure, max_radius=2, n_points=100
        )
        stat_results["stat_two_pt_corr"] = two_point_correlation_vals
        stat_results["stat_two_pt_corr_radii"] = radii_vec
        plot_two_point_correlation(
            two_point_correlation_vals, radii_vec, results_dir=stat_anal_results_dir
        )

    # Saving the results
    # --------------------------------------------------------------------------------------
    np.savez(
        os.path.join(stat_anal_results_dir, STAT_FILE_NAME),
        **{i_name: np.asarray(i_values) for i_name, i_values in stat_results.items()}
    )
    # An .npz of one array per descriptor, as the state of a run is written, where it
    # used to be a pickle of a dictionary: numpy reads it anywhere, and so does
    # anything that reads a zip of .npy files

    return stat_results


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
