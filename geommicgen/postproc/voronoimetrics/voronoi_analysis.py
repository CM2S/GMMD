"""Module for Voronoi analysis.

Functions to draw standard and set Voronoi of 2D and 3D particles assemblies.
Functions also for the analysis of the Minkowski structure metrics and associated concepts.
"""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-name-in-module
import os
import pickle
import numpy as np

from scipy.spatial import Voronoi
from scipy.special import sph_harm

from geommicgen.postproc.plotfuncs.plotting_functions import (
    plot_voronoi_2d,
    plot_voronoi_2d_with_imts,
    plot_voronoi_3d,
    plot_voronoi_3d_with_imts,
)


class Polygon:
    """Class for a single polygon."""

    def __init__(self, vertices, region):
        """Initialize a `.Polygon` instance."""
        self.vertices = vertices
        self.regions = region


def normal_density(phi, coeffs):
    """
    Compute the value of the normal density functions.

    Compute the value of the normal density functions defined by the Fourier series with
    coefficients *coeffs* at *phi*.

    Parameters
    ----------
    phi: float
        Tangent angle

    coeffs: list(floats)
        Positive coefficients of the complex Fourier series.

    Returns
    -------
    rho: float
        Normal density at *phi*.
    """
    rho = 0
    for i_ind, coeff in enumerate(coeffs):
        if i_ind == 0:
            rho += coeff
        elif i_ind == 1:
            continue
        else:
            rho += (
                coeff * np.exp(1j * i_ind * phi) / 2
                + np.conj(coeff) * np.exp(-1j * i_ind * phi) / 2
            )

    return np.real(rho)


def pos_vec(phi, coeffs):
    """
    Compute position vector.

    Compute the position vector whose tangent makes a *phi* with the x axis with normal
    density defined by a Fourier series with coefficients *coeffs*.

    Parameters
    ----------
    phi: float
        Tangent angle

    coeffs: list(floats)
        Positive coefficients of the complex Fourier series for normal density.

    Returns
    -------
    position_vec: float
        Normal density at *phi*.
    """
    pos_complex = 0
    for i_ind, coeff in enumerate(coeffs):
        if i_ind == 0:
            pos_complex += 1j * coeff * (np.exp(-1j * phi) - 1) / (2 * np.pi)
        elif i_ind == 1:
            continue
        else:
            pos_complex += -1j * coeff / (i_ind - 1) * (
                np.exp(1j * (i_ind - 1) * phi) - 1
            ) / (4 * np.pi) - 1j * np.conj(coeff) / (-i_ind - 1) * (
                np.exp(1j * (-i_ind - 1) * phi) - 1
            ) / (
                4 * np.pi
            )

    position_vec = np.array([np.imag(pos_complex), -1 * np.real(pos_complex)])
    return position_vec


class Set2DVoronoi:
    """Class for the set Voronoi.

    Attributes
    ----------
    points: list(array)
        List of center points of the shapes defining the set Voronoi.

    vertices: ndarray of double, shape(nvertices, ndim)
        Coordinates of the Voronoi vertices.

    ridge_vertices: ndarray of double, shape (nrdiges, *)
        Indices of the Voronoi vertices forming a Voronoi ridge.

    regions: list of list of ints, shape (nregions, *)
        Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex
        outside the Vornoi diagram.

    point_region: list of ints, shape(npoints)
        Index of the Voronoi region for each input point.
    """

    def __init__(self, construction_voronoi, particles, rve_dims, n_surf_points):
        """Initialize an instance of the `.Set2DVoronoi`class.

        A set Voronoi is constructed from an appropriate auxiliar standard Voronoi. The
        auxiliar Voronoi is built using as seed points uniformly distributed points on the
        surface of the particles.

        Parameters
        ----------
        construction_voronoi: `.scipy.Qhull.Voronoi`
            Auxiliar Voronoi.

        particles: list of `.Particle`
            List of particles in the RVE.

        rve_dims: list(float)
            Dimensions of the RVE in each direction.

        n_surf_points: int
            Number of surface points per particle used to generate the construction voronoi.
        """
        # Removing all ridges between surface points on the same particle
        # ----------------------------------------------------------------------------------
        # removed_vertices = set()
        # dont_remove = set()
        self.ridge_points = []
        self.ridge_vertices = []
        self.regions = [set() for _ in range(3 ** 2 * len(particles))]
        for i_ridge_ind, (i_vert_ind_1, i_vert_ind_2) in enumerate(
            construction_voronoi.ridge_points
        ):
            if i_vert_ind_1 // n_surf_points != i_vert_ind_2 // n_surf_points:
                # for j_vert in construction_voronoi.ridge_vertices[i_ridge_ind]:
                #     dont_remove.add(j_vert)
                part_ind_1 = i_vert_ind_1 // n_surf_points
                part_ind_2 = i_vert_ind_2 // n_surf_points
                self.ridge_points.append([part_ind_1, part_ind_2])
                self.ridge_vertices.append(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )
                self.regions[part_ind_1] = self.regions[part_ind_1].union(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )
                self.regions[part_ind_2] = self.regions[part_ind_2].union(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )

        # removed_vertices = set(range(len(construction_voronoi.vertices))).difference(
        #     dont_remove
        # )

        # Adding the center points for the set Voronoi
        # ----------------------------------------------------------------------------------
        self.points = []
        for i_particle in particles:
            for (j_pbc, k_pbc) in (
                (j_pbc, k_pbc) for k_pbc in range(-1, 2) for j_pbc in range(-1, 2)
            ):
                # Adding center point of particle as a Voronoi seed
                self.points.append(
                    i_particle.position_center + rve_dims * np.array([j_pbc, k_pbc])
                )

        self.vertices = construction_voronoi.vertices
        for i_ind_region, i_region in enumerate(self.regions):
            for j_ind_pt in i_region:
                if all(
                    (0 < coord < 1 for coord in construction_voronoi.vertices[j_ind_pt])
                ):
                    self.regions[i_ind_region] = vert_sort(
                        i_region, self.ridge_vertices
                    )
                    break
                self.regions[i_ind_region] = []
        self.points = np.array(self.points)
        self.point_region = list(range(len(self.points)))


class Set3DVoronoi:
    """Class for the set Voronoi.

    Attributes
    ----------
    points: list(array)
        List of center points of the shapes defining the set Voronoi.

    vertices: ndarray of double, shape(nvertices, ndim)
        Coordinates of the Voronoi vertices.

    ridge_vertices: ndarray of double, shape (nrdiges, *)
        Indices of the Voronoi vertices forming a Voronoi ridge.

    regions: list of list of ints, shape (nregions, *)
        Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex
        outside the Vornoi diagram.

    point_region: list of ints, shape(npoints)
        Index of the Voronoi region for each input point.
    """

    def __init__(self, construction_voronoi, particles, rve_dims, n_surf_points):
        """Initialize an instance of the `.Set2DVoronoi`class.

        A set Voronoi is constructed from an appropriate auxiliar standard Voronoi. The
        auxiliar Voronoi is built using as seed points uniformly distributed points on the
        surface of the particles.

        Parameters
        ----------
        construction_voronoi: `.scipy.Qhull.Voronoi`
            Auxiliar Voronoi.

        particles: list of `.Particle`
            List of particles in the RVE.

        rve_dims: list(float)
            Dimensions of the RVE in each direction.

        n_surf_points: int
            Number of surface points per particle used to generate the construction voronoi.
        """
        # Removing all ridges between surface points on the same particle
        # ----------------------------------------------------------------------------------
        # removed_vertices = set()
        # dont_remove = set()
        self.ridge_points = []
        self.ridge_vertices = []
        self.regions = [set() for _ in range(3 ** 3 * len(particles))]
        for i_ridge_ind, (i_vert_ind_1, i_vert_ind_2) in enumerate(
            construction_voronoi.ridge_points
        ):
            if i_vert_ind_1 // n_surf_points != i_vert_ind_2 // n_surf_points:
                # for j_vert in construction_voronoi.ridge_vertices[i_ridge_ind]:
                #     dont_remove.add(j_vert)
                part_ind_1 = i_vert_ind_1 // n_surf_points
                part_ind_2 = i_vert_ind_2 // n_surf_points
                self.ridge_points.append([part_ind_1, part_ind_2])
                self.ridge_vertices.append(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )
                self.regions[part_ind_1] = self.regions[part_ind_1].union(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )
                self.regions[part_ind_2] = self.regions[part_ind_2].union(
                    construction_voronoi.ridge_vertices[i_ridge_ind]
                )

        # removed_vertices = set(range(len(construction_voronoi.vertices))).difference(
        #     dont_remove
        # )

        # Adding the center points for the set Voronoi
        # ----------------------------------------------------------------------------------
        self.points = []
        for i_particle in particles:
            for (j_pbc, k_pbc, l_pbc) in (
                (j_pbc, k_pbc, l_pbc)
                for l_pbc in range(-1, 2)
                for k_pbc in range(-1, 2)
                for j_pbc in range(-1, 2)
            ):
                # Adding center point of particle as a Voronoi seed
                self.points.append(
                    i_particle.position_center
                    + rve_dims * np.array([j_pbc, k_pbc, l_pbc])
                )

        self.vertices = construction_voronoi.vertices
        self.points = np.array(self.points)
        self.point_region = list(range(len(self.points)))


def vert_sort(region_to_sort, all_ridges):
    """Sort the points clockwilse relative to the particle's center."""
    region_ridges = []
    # Ignore regions containing vertices outside the Voronoi
    # --------------------------------------------------------------------------------------
    if -1 in region_to_sort:
        return region_to_sort

    # Collecting all the ridges of this region
    # --------------------------------------------------------------------------------------
    for i_vert in region_to_sort:
        for j_vert in region_to_sort:
            if [i_vert, j_vert] in all_ridges:
                region_ridges.append([i_vert, j_vert])

    # Setting up lists to search for either the 1st or 2nd element in a ridge
    # --------------------------------------------------------------------------------------
    region_ridge_init = [i_ridge[0] for i_ridge in region_ridges]
    region_ridge_end = [i_ridge[1] for i_ridge in region_ridges]

    # First element of the ordered region. Deleted from the lists containing information
    # about the ridge, so it is not found again
    # --------------------------------------------------------------------------------------
    ord_region = region_ridges[0]
    del region_ridge_init[0]
    del region_ridge_end[0]
    del region_ridges[0]

    # Assembling the ordered region
    # --------------------------------------------------------------------------------------
    k_counter = 0
    while len(ord_region) < len(region_to_sort):
        try:
            ind_next = region_ridge_init.index(ord_region[-1])
            ord_region.append(region_ridges[ind_next][1])
        except ValueError:
            ind_next = region_ridge_end.index(ord_region[-1])
            ord_region.append(region_ridges[ind_next][0])

        del region_ridge_init[ind_next]
        del region_ridge_end[ind_next]
        del region_ridges[ind_next]
        k_counter += 1

    if len(region_to_sort) != len(ord_region):
        raise ValueError(
            "The ordered region and the original region do not have the same length."
        )

    return ord_region


def update_indices(ind_vec, removed_ind):
    """Update the indices according to the sorted removed indices."""
    # st_1 = time.time()
    # new_ind_vec = []
    # for i_ind in ind_vec:
    #     step = [rem_ind < i_ind for rem_ind in removed_ind].count(True)
    #     new_ind_vec.append(i_ind - step)
    # time_1 = time.time() - st_1
    # st_2 = time.time()
    removed_ind = list(removed_ind)
    removed_ind.sort()
    new_ind_vec_2 = [None for _ in range(len(ind_vec))]
    k_counter = 0
    for i_ind_array, i_ind in enumerate(ind_vec):
        while k_counter < len(removed_ind) and i_ind >= removed_ind[k_counter]:
            k_counter += 1
        new_ind_vec_2[i_ind_array] = i_ind - k_counter

    # time_2 = time.time() - st_2
    # print(time_1, time_2)
    # if any((i_ind != j_ind for (i_ind, j_ind) in zip(new_ind_vec, new_ind_vec_2))):
    #     print(new_ind_vec, new_ind_vec_2)
    #     raise ValueError()
    return new_ind_vec_2


def compute_global_critical_erosion_thickness(particles):
    """Compute the global critical erosion thickness."""
    loc_erosion_thick = [
        i_particle.compute_critical_erosion_thickness() for i_particle in particles
    ]
    # List of critical erosion thickness for each particle
    glob_crit_erosion_thick = np.min(loc_erosion_thick)
    # The global critical erosion thickness is the smallest of all the critical erosion
    # thickness for each particle
    return glob_crit_erosion_thick


def compute_2d_irreducible_minkowski_tensors(voronoi, degree=6):
    """Compute the Irreducible Minkowski Tensors."""
    imt_region = []
    # Initializing the list containing the list of imts for each Voronoi cell
    region_point = np.zeros((len(voronoi.regions)), dtype=int)
    for point_ind, region_ind in enumerate(voronoi.point_region):
        if point_ind == -1:
            continue
        region_point[region_ind] = int(point_ind)
        # Obtaining the point associated with a given region using its indix in
        # voronoi.regions
    k_used_region = 0
    in_box = []
    for i_ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # print(region_point[i_ind])
        pos_center = voronoi.points[region_point[i_ind]]
        if 0 < pos_center[0] < 1 and 0 < pos_center[1] < 1:
            in_box.append(k_used_region)
        k_used_region += 1
        # Obtaining the indices of the regions associated with particles inside the box,
        # i.e. excluding periodic images
    for i_ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # Running through all the cells in the Voronoi
        imt_region.append([])
        # Initializing the list containing the imts of the cell
        n_vertices = len(i_region)
        sides = [
            voronoi.vertices[i_region[j_pair]]
            - voronoi.vertices[i_region[np.mod(j_pair + 1, n_vertices)]]
            for j_pair in range(n_vertices)
        ]
        lengths = [np.linalg.norm(i_side) for i_side in sides]
        angles = []
        for i_side in sides:
            if i_side[1] >= 0:
                angles.append(np.pi / 2 + np.arccos(i_side[0] / np.linalg.norm(i_side)))
            elif i_side[1] < 0:
                angles.append(
                    np.pi / 2
                    + 2 * np.pi
                    - np.arccos(i_side[0] / np.linalg.norm(i_side))
                )
                # if i_side[0] >= 0 and i_side[1] >= 0:
            #     angles.append(np.pi/2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            # elif i_side[0] >= 0 and i_side[1] <= 0:
            #     angles.append(np.pi/2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            # elif i_side[0] <= 0 and i_side[1] <= 0:
            #     angles.append(3*np.pi/2 + np.arctan2(np.abs(i_side[1]),
            #     np.abs(i_side[0]))) # elif i_side[0] <= 0 and i_side[1] >= 0:
            #     angles.append(3*np.pi/2 - np.arctan2(np.abs(i_side[1]),
            #     np.abs(i_side[0])))
            # else:
            #     pass
            # print(i_side)
        # angles = [np.pi/2 - np.arctan(i_side[1]/i_side[0]) for i_side in sides]
        # print('angles', angles)
        # print('lengths', lengths)
        for j_tensor in range(degree + 1):
            # Computing the 7 first imts
            imt_region[-1].append(
                np.sum(
                    [
                        lengths[k_side] * np.exp(1j * j_tensor * angles[k_side])
                        for k_side in range(len(lengths))
                    ]
                )
            )

    return [imt_region, in_box, angles]


def compute_2d_irreducible_minkowski_tensors_polygon(voronoi, degree=6):
    """Compute the Irreducible Minkowski Tensors."""
    imt_region = []
    # Initializing the list containing the list of imts for each Voronoi cell
    for _, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # Running through all the cells in the Voronoi
        imt_region.append([])
        # Initializing the list containing the imts of the cell
        n_vertices = len(i_region)
        sides = [
            voronoi.vertices[i_region[j_pair]]
            - voronoi.vertices[i_region[np.mod(j_pair + 1, n_vertices)]]
            for j_pair in range(n_vertices)
        ]
        lengths = [np.linalg.norm(i_side) for i_side in sides]
        angles = []
        for i_side in sides:
            if i_side[0] >= 0 and i_side[1] >= 0:
                angles.append(
                    np.pi / 2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0]))
                )
            elif i_side[0] >= 0 and i_side[1] <= 0:
                angles.append(
                    np.pi / 2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0]))
                )
            elif i_side[0] <= 0 and i_side[1] <= 0:
                angles.append(
                    3 * np.pi / 2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0]))
                )
            elif i_side[0] <= 0 and i_side[1] >= 0:
                angles.append(
                    3 * np.pi / 2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0]))
                )
            else:
                pass
                # print(i_side)
        # angles = [np.pi/2 - np.arctan(i_side[1]/i_side[0]) for i_side in sides]
        print("angles", np.degrees(angles))
        # print('lengths', lengths)
        for j_tensor in range(degree + 1):
            # Computing the 7 first imts
            imt_region[-1].append(
                np.sum(
                    [
                        lengths[k_side] * np.exp(1j * j_tensor * angles[k_side])
                        for k_side in range(len(lengths))
                    ]
                )
            )

    return imt_region


def compute_3d_irreducible_minkowski_tensors(voronoi):
    """Compute the Irreducible Minkowski Tensors."""
    p = 0
    particle_inds = list(range(13, len(voronoi.point_region), 27))
    n_particles = len(voronoi.point_region)
    all_imt = [[] for _ in range(len(particle_inds))]
    all_phi = [np.zeros((7, 13), dtype=complex) for _ in range(n_particles)]
    all_area = [0 for _ in range(n_particles)]
    for ridge, points in zip(voronoi.ridge_vertices, voronoi.ridge_points):
        inds_to_consider = []
        if points[0] in particle_inds:
            inds_to_consider.append(points[0])
        if points[1] in particle_inds:
            inds_to_consider.append(points[1])
            # Running through all the cells in the Voronoi
            # Initializing the list containing the imts of the cell
        if len(inds_to_consider) != 0:
            area_ridge = area_face(voronoi.vertices[ridge])
            for i_particle in inds_to_consider:
                center_point = voronoi.points[i_particle]
                normal_ridge = out_normal_face(voronoi.vertices[ridge], center_point)
                angles_normal_ridge = unit_vector_to_sph_coord(normal_ridge)
                all_area[i_particle] += area_ridge
                for order in range(-6, 7):
                    for degree in range(7):
                        if np.abs(order) <= degree:
                            all_phi[i_particle][
                                degree, order + 6
                            ] += area_ridge * sph_harm(
                                order,
                                degree,
                                angles_normal_ridge[0],
                                angles_normal_ridge[1],
                            )

        p += 1
    all_phi = [all_phi[i] for i in particle_inds]
    all_area = [all_area[i] for i in particle_inds]
    for i_ind, (i_phi, i_area) in enumerate(zip(all_phi, all_area)):
        for degree in range(7):
            # Computing the 7 first imts
            if degree == 0:
                all_imt[i_ind].append(i_area)
            else:
                all_imt[i_ind].append(
                    np.sqrt(
                        4
                        * np.pi
                        / (2 * degree + 1)
                        / (i_area ** 2)
                        * np.sum(
                            [
                                np.abs(i_phi[degree, order + 6]) ** 2
                                for order in range(-6, 7)
                            ]
                        )
                    )
                )
    return all_imt, all_phi


def compute_3d_irreducible_minkowski_tensors_polyhedron(unormals, area):
    """Compute the Irreducible Minkowski Tensors."""
    phi = np.zeros((7, 13), dtype=complex)
    imt_region = []
    angles_normal = [unit_vector_to_sph_coord(unormal) for unormal in unormals]
    A_total = np.sum(area)
    for order in range(-6, 7):
        for degree in range(7):
            if np.abs(order) <= degree:
                phi[degree, order + 6] = np.sum(
                    [
                        area[k_face]
                        * sph_harm(
                            order,
                            degree,
                            angles_normal[k_face][0],
                            angles_normal[k_face][1],
                        )
                        for k_face in range(len(area))
                    ]
                )
    for degree in range(7):
        # Computing the 7 first imts
        imt_region.append(
            np.sqrt(
                4
                * np.pi
                / (2 * degree + 1)
                * 1
                / A_total ** 2
                * np.sum(
                    [np.abs(phi[degree, order + 6]) ** 2 for order in range(-6, 7)]
                )
            )
        )
    return [imt_region, phi]


def area_face(vertices):
    """Compute the area of the polygon defined by *vertices*."""
    center_gravity = 1 / len(vertices) * np.sum(vertices, axis=0)
    # Computing the center of the polygon
    ref_vec_x = vertices[0] - center_gravity
    ref_vec_y = (vertices[1] - center_gravity) - np.dot(
        vertices[1] - center_gravity, ref_vec_x
    ) / np.dot(ref_vec_x, ref_vec_x) * ref_vec_x
    angles = []
    for i_vertex in vertices:
        i_ref_vec = i_vertex - center_gravity
        angles.append(np.arctan2(i_ref_vec.dot(ref_vec_y), i_ref_vec.dot(ref_vec_x)))

    sorted_vertices = vertices[np.argsort(angles)]
    area = (
        1
        / 2
        * np.sum(
            [
                np.linalg.norm(
                    np.cross(
                        sorted_vertices[k_vertex] - center_gravity,
                        sorted_vertices[np.mod(k_vertex + 1, len(sorted_vertices))]
                        - center_gravity,
                    )
                )
                for k_vertex, _ in enumerate(sorted_vertices)
            ]
        )
    )
    # Computing the area from the croos product
    return area


def out_normal_face(vertices, center_point):
    """
    Compute outward normal.

    Compute the outward normal unit vector to the polygon defined by *vertices* relative
    to *center_point*.
    """
    normal_vector = np.cross(vertices[0] - vertices[1], vertices[0] - vertices[2])
    outward_vector = vertices[0] - center_point
    if normal_vector.dot(outward_vector) > 0:
        out_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    else:
        out_unit_vector = -1 * normal_vector / np.linalg.norm(normal_vector)
    return out_unit_vector


def unit_vector_to_sph_coord(unit_vector):
    """Convert unit vector to spherical coordinates."""
    theta = np.arctan2(unit_vector[1], unit_vector[0])
    if theta < 0:
        theta += 2 * np.pi
    phi = np.arctan2(np.sqrt(unit_vector[0] ** 2 + unit_vector[1] ** 2), unit_vector[2])
    if phi < 0:
        phi += 2 * np.pi
    return [theta, phi]


def compute_2d_set_voronoi(particles, rve_dims, n_surf_points=10):
    """
    Compute the set Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    n_surf_points: optional, integer
        Number of particles to be generated on the surface of each particle.

    Returns
    -------
    set_voronoi: `.Set2DVoronoi`
        Set voronoi of the particles.
    """
    global_crit_ero_thick = compute_global_critical_erosion_thickness(particles)
    # Computing the global critical erosion thickness
    part_counter = 0
    # Particle number counte
    for i_particle in particles:
        for (j_pbc, k_pbc) in (
            (j_pbc, k_pbc) for k_pbc in range(-1, 2) for j_pbc in range(-1, 2)
        ):
            # Running through all the particles and their periodic images
            if part_counter == 0:
                particle_surf = i_particle.generate_points_on_surface(
                    n_surf_points, erosion_thick=global_crit_ero_thick
                ) + rve_dims * np.array([j_pbc, k_pbc])
            else:
                particle_surf = np.concatenate(
                    (
                        particle_surf,
                        (
                            i_particle.generate_points_on_surface(
                                n_surf_points, erosion_thick=global_crit_ero_thick
                            )
                            + rve_dims * np.array([j_pbc, k_pbc])
                        ),
                    ),
                    axis=0,
                )
            # Sampling points on the surface of each eroded particle and collecing then
            part_counter += 1
            # Updating the counter
    auxiliar_voronoi = Voronoi(particle_surf)
    # Obraining the auxiliar voronoi for the construction of the set voronoi
    set_voronoi = Set2DVoronoi(auxiliar_voronoi, particles, rve_dims, n_surf_points)
    # Computing the set voronoi of the particles
    return set_voronoi


def compute_2d_weigthed_voronoi():  # particles):
    """
    Compute the set Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    Returns
    -------
    weighted_voronoi: `.WeightedVoronoi`
        Weighted voronoi of the particles.
    """
    # particle_centers = []
    # # Intializing the list containing the centers of the particles and their images
    # C = np.max([i_particle.radius for i_particle in particles])
    # # Obtaining the largest radius
    # for i_part_ind, i_particle in enumerate(particles):
    #     for j in range(-1, 2):
    #         for k in range(-1, 2):
    #             # Running through all the particles and their periodic images
    #             particle_centers.append(
    #                 np.concatenate(
    #                     (
    #                         i_particle.position_center + rve_dims * np.array([j, k]),
    #                         [np.sqrt(C - i_particle.radius)],
    #                     )
    #                 )
    #             )
    #             # Sampling points on the surface of each eroded particle and collecing
    # then
    # auxiliar_voronoi = Voronoi(particle_centers)
    # # Obraining the auxiliar voronoi for the construction of the set voronoi
    # # weighted_voronoi = weighted2DVoronoi(auxiliar_voronoi, particles)
    # # Computing the set voronoi of the particles
    # # return weighted_voronoi


def compute_3d_set_voronoi(particles, rve_dims, n_surf_points=10):
    """
    Compute the set Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    n_surf_points: optional, integer
        Number of particles to be generated on the surface of each particle.

    Returns
    -------
    set_voronoi: `.Set2DVoronoi`
        Set voronoi of the particles.
    """
    global_crit_ero_thick = compute_global_critical_erosion_thickness(particles)
    # Computing the global critical erosion thickness
    part_counter = 0
    # Particle number counter
    for i_particle in particles:
        for j in range(-1, 2):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Running through all the particles and their periodic images
                    if part_counter == 0:
                        particle_surf = i_particle.generate_points_on_surface(
                            n_surf_points, erosion_thick=global_crit_ero_thick
                        ) + rve_dims * np.array([j, k, l])
                    else:
                        particle_surf = np.concatenate(
                            (
                                particle_surf,
                                (
                                    i_particle.generate_points_on_surface(
                                        n_surf_points,
                                        erosion_thick=global_crit_ero_thick,
                                    )
                                    + rve_dims * np.array([j, k, l])
                                ),
                            ),
                            axis=0,
                        )
                    # Sampling points on the surface of each eroded particle and collecing
                    # then
                    part_counter += 1
                    # Updating the counter

    auxiliar_voronoi = Voronoi(particle_surf)
    # Obraining the auxiliar voronoi for the construction of the set voronoi
    set_voronoi = Set3DVoronoi(
        auxiliar_voronoi, particles, rve_dims, n_surf_points ** 2
    )
    # Computing the set voronoi of the particles
    return set_voronoi


def compute_2d_standard_voronoi(particles, rve_dims):
    """
    Compute the standard Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    rve_dims: list(float)
        Dimensions of the RVE.

    Returns
    -------
    std_voronoi: `.scipy.Qhull.Voronoi`
        Standard voronoi of the particles.
    """
    particle_centers = []
    # Intializing the list containing the centers of the particles and their images
    for i_particle in particles:
        for (j_pbc, k_pbc) in (
            (j_pbc, k_pbc) for k_pbc in range(-1, 2) for j_pbc in range(-1, 2)
        ):
            # Running through all the particles and their periodic images
            particle_centers.append(
                i_particle.position_center + rve_dims * np.array([j_pbc, k_pbc])
            )
            # Sampling points on the surface of each eroded particle and collecing then
    std_voronoi = Voronoi(particle_centers)
    # Computing the standard voronoi of the particles
    return std_voronoi


def compute_3d_standard_voronoi(particles, rve_dims):
    """
    Compute the standard Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    rve_dims: list(float)
        Dimensions of the RVE.

    Returns
    -------
    std_voronoi: `.scipy.Qhull.Voronoi`
        Standard voronoi of the particles.
    """
    particle_centers = []
    # Intializing the list containing the centers of the particles and their images
    for i_particle in particles:
        for j in range(-1, 2):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Running through all the particles and their periodic images
                    particle_centers.append(
                        i_particle.position_center + rve_dims * np.array([j, k, l])
                    )
                # Sampling points on the surface of each eroded particle and collecing then
    std_voronoi = Voronoi(particle_centers)
    # Computing the standard voronoi of the particles
    return std_voronoi


def compute_test_stat_chi_squared(array_samples):
    """
    Compute the chi square statistic for two or more independent samples, discrete outcome.

    Parameters
    ----------
    array_samples: n-array(floats)
        Each row is a sample and the columns are the values.
    """
    n_bins = 20
    array_samples_bin = np.array(
        [
            [
                np.count_nonzero(
                    np.logical_and(
                        bin * 1 / n_bins < sample, sample <= (bin + 1) * 1 / n_bins
                    )
                )
                for bin in range(n_bins)
            ]
            for sample in array_samples
        ]
    )
    chi = 0
    used_bins = 0
    for i_row in range(len(array_samples)):
        for j_col in range(n_bins):
            expected_freq = (
                np.sum(array_samples_bin[i_row, :])
                * np.sum(array_samples_bin[:, j_col])
                / np.size(array_samples)
            )
            if expected_freq == 0:
                continue
            chi += (
                array_samples_bin[i_row, j_col] - expected_freq
            ) ** 2 / expected_freq
            used_bins += 1

    return [chi, used_bins]


def do_voronoi_analysis(
    particles,
    rve_dims,
    sample_dir,
    voronoi_type="standard",
    plot_voronoi=False,
    plot_imts=False,
    **kwargs
):
    """
    Do a Voronoi analysis on the RVE.

    Saves the Voronoi diagram and the Irreducible Minkowski Tensors computed.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles inside the RVE.

    rve_dims: list(float)
        Dimensions of the RVE.

    voronoi_type: {'set', 'standard', 'weighted'}, optional
        Type of Voronoi to be computed.

    plot_voronoi: bool, optional
        Plot the voronoi diagrama of the *particles* in the RVE.

    plot_imts: bool, optional
        Plot the voronoi with the cell colored according to the Minkowski structure metrics
        as well as the corresponding histograms.

    Keyword Parameters
    ------------------
    n_surf_points: int
        Number of surface points used to generate the set Voronoi diagram.

    """
    voronoi_results_dir = os.path.join(sample_dir, "voronoi_analysis_results")
    os.makedirs(voronoi_results_dir)
    # Creating a directory for the results

    if particles[0].dim == 2:
        # 2D problem
        if voronoi_type == "set":
            # The required Voronoi is a set Voronoi
            voronoi = compute_2d_set_voronoi(particles, rve_dims, **kwargs)
        elif voronoi_type == "standard":
            # The required Voronoi is a standard Voronoi
            voronoi = compute_2d_standard_voronoi(particles, rve_dims)
        elif voronoi_type == "weighted":
            # The required Voronoi is a weighted Voronoi
            voronoi = compute_2d_standard_voronoi(particles, rve_dims)
        if plot_voronoi:
            plot_voronoi_2d(
                particles, rve_dims, voronoi, voronoi_results_dir, voronoi_type
            )
        imts, in_box, angles = compute_2d_irreducible_minkowski_tensors(voronoi)
        # Computing the irreducible Minkowski tensors for the current microsturcture
        if plot_imts:
            plot_voronoi_2d_with_imts(
                particles, rve_dims, voronoi, imts, voronoi_results_dir, voronoi_type
            )
        # Saving the results
        # --------------------------------------------------------------------------------------
        pickle.dump(
            [voronoi, imts, angles, in_box],
            open(os.path.join(voronoi_results_dir, "voronoi_results.vor"), "wb"),
        )
    elif particles[0].dim == 3:
        # if voronoi_type == 'standard':
        if voronoi_type == "set":
            voronoi = compute_3d_set_voronoi(particles, rve_dims, **kwargs)
        if voronoi_type == "standard":
            voronoi = compute_3d_standard_voronoi(particles, rve_dims)
        if plot_voronoi:
            # Saving the irreducible Minkowski tensors of the voronoi cells associated with
            # particles inside the box
            plot_voronoi_3d(particles, voronoi, rve_dims, voronoi_results_dir)
        imts, phi = compute_3d_irreducible_minkowski_tensors(voronoi)
        if plot_imts:
            plot_voronoi_3d_with_imts(
                particles, voronoi, rve_dims, imts, voronoi_results_dir
            )
        # Saving the results
        # --------------------------------------------------------------------------------------
        pickle.dump(
            [voronoi, imts, phi],
            open(os.path.join(voronoi_results_dir, "voronoi_results.vor"), "wb"),
        )
