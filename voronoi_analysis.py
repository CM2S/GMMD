from particle_classes import (
    Particle,
    Disk,
    Ellipse,
    Sphere,
    Ellipsoid,
    CylindricalFiber,
)

from plotting_functions import (
    plotVoronoi2D,
    plotVoronoi2DwithIMTs,
    plotVoronoi3D,
    plotVoronoi3DwithIMTs,
    createFigure,
)

from scipy.spatial import Voronoi

from scipy.special import sph_harm

import numpy as np


class Polygon:
    def __init__(self, vertices, region):
        self.vertices = vertices
        self.regions = region


class Polyhedron:

    pass


def normal_density(phi, coeffs):
    """
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
    pos_vec: float
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

    pos_vec = np.array([np.imag(pos_complex), -np.real(pos_complex)])
    return pos_vec


# class weightedVoronoi():
#
#     def __init__(self. auxiliar_voronoi, particles):
#         list_2D_vetices = []
#         for vertex in auxiliar_voronoi.vertices:
#         # Running through all the vertices in the auxiliar voronoi
#             list_2D_vetices.append(vertex[0:-1])
#             # Removing the third (auxiliar) coordinate
#         self.vertices = np.array(list_2D_vetices)
#         # Saving the vertices of the weighted voronoi
#         for ridge in auxiliar_voronoi.ridges:
#         # Running through all the ridges in the auxiliar voronoi
#             for region in auxiliar_voronoi.regions:
#             # Running through all the regions in the auxiliary voronoi
#                 if any([vertex in region for vertex in ridge])
#                 # If any of the vertex of the current ridge belongs to the current region
#                     clean_ridge = [vertex for vertex in ridge if vertex != -1]
#                     # Removing the -1s
#


class setVoronoi:
    def __init__(self, construction_voronoi, particles):
        old_regions = []
        self.points = []
        removed_vertices = []
        point_region = []
        region_counter = 0
        for i_particle in particles:
            for j in range(-1, 2):
                for k in range(-1, 2):
                    point_region.append(region_counter)
                    region_counter += 1
                    all_regions = []
                    current_removed = []
                    self.points.append(
                        i_particle.position_center + Particle.box * np.array([j, k])
                    )
                    for j_vertex in range(len(construction_voronoi.vertices)):
                        if i_particle.pointInside(
                            construction_voronoi.vertices[j_vertex]
                            - Particle.box * np.array([j, k])
                        ):
                            removed_vertices.append(j_vertex)
                            current_removed.append(j_vertex)
                    for region in construction_voronoi.regions:
                        if any(
                            [
                                removed_vertex in region
                                for removed_vertex in current_removed
                            ]
                        ):
                            print("region", region)
                            region_no_int_vert = [
                                ind_vert
                                for ind_vert in region
                                if ind_vert not in removed_vertices
                            ]
                            print("clean", region_no_int_vert)
                            all_regions += region_no_int_vert
                            print("current", all_regions)
                    if len(all_regions) > 0:
                        list_vert_reg = list(set(all_regions))
                        old_regions.append(list_vert_reg)

        print("old", old_regions)
        old_ridge_vertices = list(construction_voronoi.ridge_vertices)
        for ridge in construction_voronoi.ridge_vertices:
            if any(
                [
                    (removed_vertex in ridge or -1 in ridge)
                    for removed_vertex in removed_vertices
                ]
            ):
                old_ridge_vertices.remove(ridge)

        for ind_region, region in enumerate(old_regions):
            old_regions[ind_region] = vertSort(region, old_ridge_vertices)
        # for ridge in old_ridge_vertices:
        #     print(ridge, any([(removed_vertex in ridge or -1 in ridge) for removed_vertex in removed_vertices]))
        #     if any([(removed_vertex in ridge or -1 in ridge) for removed_vertex in removed_vertices]):
        #         old_ridge_vertices.remove(ridge)
        # print('new_ridge', old_ridge_vertices)

        self.vertices = np.delete(construction_voronoi.vertices, removed_vertices, 0)
        # print('old_region', old_regions)
        self.ridge_vertices = [
            newIndices(i_ridge, removed_vertices) for i_ridge in old_ridge_vertices
        ]
        self.regions = [
            newIndices(i_region, removed_vertices) for i_region in old_regions
        ]

        # self.vertices = np.array([vertex for vertex in construction_voronoi.vertices if np.where(construction_voronoi.vertices == vertex)])
        self.points = np.array(self.points)
        self.point_region = point_region

        print(self.regions)
        print("here")
        # self.points = construction_voronoi.points
        # self.ridge_points = construction_voronoi.ridge_points
        # self.ridge_vertices = construction_voronoi.ridge_vertices
        # self.vertices = construction_voronoi.vertices
        # self.furthest_site = construction_voronoi.furthest_site


class set3DVoronoi:
    def __init__(self, construction_voronoi, particles):
        old_regions = []
        self.points = []
        removed_vertices = []
        point_region = []
        region_counter = 0
        for i_particle in particles:
            for j in range(-1, 2):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        # Running through all the paricles and their periodic images
                        point_region.append(region_counter)
                        region_counter += 1
                        all_regions = []
                        current_removed = []
                        self.points.append(
                            i_particle.position_center
                            + Particle.box * np.array([j, k, l])
                        )
                        # Saving the center points of the particles
                        for j_vertex in range(len(construction_voronoi.vertices)):
                            # Going through all the vertices of the construction voronoi
                            if i_particle.pointInside(
                                construction_voronoi.vertices[j_vertex]
                                - Particle.box * np.array([j, k, l])
                            ):
                                # If the vertex is inside a particle it is removed
                                removed_vertices.append(j_vertex)
                                # Appending the vertex to the list of all removed vertices
                                current_removed.append(j_vertex)
                                # Appending the vertex to the list of vertices removed
                                # while analyising this particle
                        for region in construction_voronoi.regions:
                            # Running through all the regions in the construction voronoi
                            if any(
                                [
                                    removed_vertex in region
                                    for removed_vertex in current_removed
                                ]
                            ):
                                # If the region contains one of the removed vertices
                                # corresponding to the current particle
                                region_no_int_vert = [
                                    ind_vert
                                    for ind_vert in region
                                    if ind_vert not in removed_vertices
                                ]
                                # Obtaining the region without the removed vertices
                                all_regions += region_no_int_vert
                                # Adding the vertices of the current region to the variable
                                # containing the vertices of the region corresponding to
                                # the current particle
                        if len(all_regions) > 0:
                            # If the region is not empty
                            list_vert_reg = list(set(all_regions))
                            # Remove repeated vertices
                            old_regions.append(list_vert_reg)
                            # Appending the region corresponding to the current particle to
                            # the list of all particle regions
                        else:
                            old_regions.append([-1])

        old_ridge_vertices = list(construction_voronoi.ridge_vertices)
        # Saving the list of the ridge vertices of the construction voronoi
        for ridge in construction_voronoi.ridge_vertices:
            # Running through all the ridges of the construction voronoi
            if any(
                [
                    (removed_vertex in ridge or -1 in ridge)
                    for removed_vertex in removed_vertices
                ]
            ):
                # If there is a removed vertex in the ridge or a vertice out of boundary (-1)
                # remove it
                old_ridge_vertices.remove(ridge)

        self.vertices = np.delete(construction_voronoi.vertices, removed_vertices, 0)
        # Delete the removed vertices and save the remaining vertices
        self.ridge_vertices = [
            newIndices(i_ridge, removed_vertices) for i_ridge in old_ridge_vertices
        ]
        # Save the ridge vertices changing the indices to account for the removed vertices
        self.regions = [
            newIndices(i_region, removed_vertices) for i_region in old_regions
        ]
        # Save the regions changing the indices to account for the removed vertices
        self.points = np.array(self.points)
        # Saving the center of the particles and their periodic images as an array
        self.point_region = point_region
        # Saving the indices relating each point to its corresponding region


def vertSort(region, all_ridges, max_tol=10000):
    """Sort the points clockwilse relative to the particle's center."""
    region_ridges = []
    if -1 in region:
        return region
    for i_vert in region:
        for j_vert in region:
            if [i_vert, j_vert] in all_ridges:
                region_ridges.append([i_vert, j_vert])
    ord_region = region_ridges[0]
    region_ridges.remove(region_ridges[0])
    k_counter = 0
    while len(region_ridges) > 1 and k_counter < max_tol:
        k_counter += 1
        for ridge in reversed(region_ridges):
            if ord_region[-1] == ridge[0]:
                ord_region.append(ridge[1])
                region_ridges.remove(ridge)
            elif ord_region[-1] == ridge[1]:
                ord_region.append(ridge[0])
                region_ridges.remove(ridge)
            elif ord_region[0] == ridge[0]:
                ord_region.insert(0, ridge[1])
                region_ridges.remove(ridge)
            elif ord_region[0] == ridge[1]:
                ord_region.insert(0, ridge[0])
                region_ridges.remove(ridge)
    # if k_counter == max_tol:
    #     ord_region = []
    return ord_region


def newIndices(ind_vec, removed_ind):
    """Update the indices according to the removed indices."""
    # print("ind_vec", ind_vec)
    # print("removed_ind", removed_ind)
    new_ind_vec = []
    for i_ind in ind_vec:
        step = [rem_ind < i_ind for rem_ind in removed_ind].count(True)
        new_ind_vec.append(i_ind - step)
    # print("new_ind_vec", new_ind_vec)
    return new_ind_vec


def computeGlobalCriticalErosionThickness(particles):
    """Compute the global critical erosion thickness."""
    loc_erosion_thick = [
        i_particle.computeCriticalErosionThickness() for i_particle in particles
    ]
    # List of critical erosion thickness for each particle
    glob_crit_erosion_thick = np.min(loc_erosion_thick)
    # The global critical erosion thickness is the smallest of all the critical erosion
    # thickness for each particle
    return glob_crit_erosion_thick


def compute2DIrreducibleMinkowskiTensors(voronoi, degree=6):
    """Compute the Irreducible Minkowski Tensors."""
    IMT_region = []
    # Initializing the list containing the list of IMTs for each Voronoi cell
    region_point = np.zeros((len(voronoi.regions)), dtype=int)
    for point_ind, region_ind in enumerate(voronoi.point_region):
        if point_ind == -1:
            continue
        region_point[region_ind] = int(point_ind)
        # Obtaining the point associated with a given region using its indix in
        # voronoi.regions
    k_used_region = 0
    in_box = []
    for ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # print(region_point[ind])
        pos_center = voronoi.points[region_point[ind]]
        if 0 < pos_center[0] < 1 and 0 < pos_center[1] < 1:
            in_box.append(k_used_region)
        k_used_region += 1
        # Obtaining the indices of the regions associated with particles inside the box,
        # i.e. excluding periodic images
    for ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # Running through all the cells in the Voronoi
        IMT_region.append([])
        # Initializing the list containing the IMTs of the cell
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
            #     angles.append(3*np.pi/2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            # elif i_side[0] <= 0 and i_side[1] >= 0:
            #     angles.append(3*np.pi/2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            # else:
            #     pass
            # print(i_side)
        # angles = [np.pi/2 - np.arctan(i_side[1]/i_side[0]) for i_side in sides]
        # print('angles', angles)
        # print('lengths', lengths)
        for j_tensor in range(degree + 1):
            # Computing the 7 first IMTs
            IMT_region[-1].append(
                np.sum(
                    [
                        lengths[k_side] * np.exp(1j * j_tensor * angles[k_side])
                        for k_side in range(len(lengths))
                    ]
                )
            )

    return [IMT_region, in_box]


def compute2DIrreducibleMinkowskiTensorsPolygon(voronoi, degree=6):
    """Compute the Irreducible Minkowski Tensors."""
    IMT_region = []
    # Initializing the list containing the list of IMTs for each Voronoi cell
    for ind, i_region in enumerate(voronoi.regions):
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
        # Running through all the cells in the Voronoi
        IMT_region.append([])
        # Initializing the list containing the IMTs of the cell
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
            # Computing the 7 first IMTs
            IMT_region[-1].append(
                np.sum(
                    [
                        lengths[k_side] * np.exp(1j * j_tensor * angles[k_side])
                        for k_side in range(len(lengths))
                    ]
                )
            )

    return IMT_region


def compute3DIrreducibleMinkowskiTensors(voronoi):
    """Compute the Irreducible Minkowski Tensors."""
    IMT_region = []
    phi_region = []
    # Initializing the list containing the list of IMTs for each Voronoi cell
    for i_particle in range(13, len(voronoi.point_region), 27):
        center_point = voronoi.points[i_particle]
        i_particle_region = voronoi.regions[voronoi.point_region[i_particle]]
        area_ridge = []
        normal_ridge = []
        angles_normal_ridge = []
        IMT_region.append([])
        phi_region.append([])
        phi = np.zeros((7, 13), dtype=complex)
        for ridge in voronoi.ridge_vertices:
            if len(i_particle_region) == 0:
                continue
            if any([vertex == -1 for vertex in i_particle_region]):
                continue
            if any([vertex not in i_particle_region for vertex in ridge]):
                continue
            # Running through all the cells in the Voronoi
            # Initializing the list containing the IMTs of the cell
            area_ridge.append(areaFace(voronoi.vertices[ridge]))
            normal_ridge.append(outNormalFace(voronoi.vertices[ridge], center_point))
            angles_normal_ridge.append(unitVectorToSphCoord(normal_ridge[-1]))
            print(i_particle)
        print(area_ridge, normal_ridge)

        A_total = np.sum(area_ridge)
        print(
            "angles, normal",
            [
                (angles_normal_ridge[i], normal_ridge[i])
                for i in range(len(normal_ridge))
            ],
        )
        for order in range(-6, 7):
            for degree in range(7):
                if np.abs(order) <= degree:
                    phi[degree, order + 6] = np.sum(
                        [
                            area_ridge[k_face]
                            * sph_harm(
                                order,
                                degree,
                                angles_normal_ridge[k_face][0],
                                angles_normal_ridge[k_face][1],
                            )
                            for k_face in range(len(area_ridge))
                        ]
                    )
        print("phi", phi)
        for degree in range(7):
            # Computing the 7 first IMTs
            if degree == 0:
                IMT_region[-1].append(A_total)
            else:
                IMT_region[-1].append(
                    np.sqrt(
                        4
                        * np.pi
                        / (2 * degree + 1)
                        / (A_total ** 2)
                        * np.sum(
                            [
                                np.abs(phi[degree, order + 6]) ** 2
                                for order in range(-6, 7)
                            ]
                        )
                    )
                )
        phi_region[-1].append(phi)
    print(IMT_region)
    return IMT_region


def compute3DIrreducibleMinkowskiTensorsPoluhedron(unormals, area):
    """Compute the Irreducible Minkowski Tensors."""

    phi = np.zeros((7, 13), dtype=complex)
    IMT_region = []
    angles_normal = [unitVectorToSphCoord(unormal) for unormal in unormals]
    A_total = np.sum(area)
    print("A_total", A_total)
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
    print("phi", phi)
    for degree in range(7):
        # Computing the 7 first IMTs
        IMT_region.append(
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
    print(IMT_region)
    return [IMT_region, phi]


def areaFace(vertices):
    """Compute the area of the polygon defined by *vertices*."""
    center_gravity = 1 / len(vertices) * np.sum(vertices, axis=0)
    # Computing the center of the polygon
    area = (
        1
        / 2
        * np.sum(
            [
                np.linalg.norm(
                    np.cross(
                        vertices[k_vertex] - center_gravity,
                        vertices[np.mod(k_vertex + 1, len(vertices))] - center_gravity,
                    )
                )
                for k_vertex, _ in enumerate(vertices)
            ]
        )
    )
    # Computing the area from the croos product
    return area


def outNormalFace(vertices, center_point):
    """
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


def unitVectorToSphCoord(unit_vector):
    """Convert unit vector to spherical coordinates"""
    theta = np.arctan2(unit_vector[1], unit_vector[0])
    if theta < 0:
        theta += 2 * np.pi
    phi = np.arctan2(np.sqrt(unit_vector[0] ** 2 + unit_vector[1] ** 2), unit_vector[2])
    if phi < 0:
        phi += 2 * np.pi
    return [theta, phi]


def compute2DSetVoronoi(particles, n_surf_points=30):
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
    set_voronoi: `.SetVoronoi`
        Set voronoi of the particles.
    """
    global_crit_ero_thick = computeGlobalCriticalErosionThickness(particles)
    # Computing the global critical erosion thickness
    part_counter = 0
    # Particle number counter
    for i_part_ind, i_particle in enumerate(particles):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Running through all the particles and their periodic images
                if part_counter == 0:
                    particle_surf = i_particle.generatePointsOnSurface(
                        n_surf_points, erosion_thick=global_crit_ero_thick
                    ) + Particle.box * np.array([j, k])
                else:
                    particle_surf = np.concatenate(
                        (
                            particle_surf,
                            (
                                i_particle.generatePointsOnSurface(
                                    n_surf_points, erosion_thick=global_crit_ero_thick
                                )
                                + Particle.box * np.array([j, k])
                            ),
                        ),
                        axis=0,
                    )
                # Sampling points on the surface of each eroded particle and collecing then
            part_counter += 1
            # Updating the counter
    auxiliar_voronoi = Voronoi(particle_surf)
    # Obraining the auxiliar voronoi for the construction of the set voronoi
    set_voronoi = setVoronoi(auxiliar_voronoi, particles)
    # Computing the set voronoi of the particles
    return set_voronoi


def compute2DWeigthedVoronoi(particles):
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
    particle_centers = []
    # Intializing the list containing the centers of the particles and their images
    C = np.max([i_particle.radius for i_particle in particles])
    # Obtaining the largest radius
    for i_part_ind, i_particle in enumerate(particles):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Running through all the particles and their periodic images
                particle_centers.append(
                    np.concatenate(
                        (
                            i_particle.position_center
                            + Particle.box * np.array([j, k]),
                            [np.sqrt(C - i_particle.radius)],
                        )
                    )
                )
                # Sampling points on the surface of each eroded particle and collecing then
    auxiliar_voronoi = Voronoi(particle_centers)
    # Obraining the auxiliar voronoi for the construction of the set voronoi
    weighted_voronoi = weighted2DVoronoi(auxiliar_voronoi, particles)
    # Computing the set voronoi of the particles
    return weighted_voronoi


def compute3DSetVoronoi(particles, n_surf_points=5):
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
    set_voronoi: `.SetVoronoi`
        Set voronoi of the particles.
    """
    global_crit_ero_thick = computeGlobalCriticalErosionThickness(particles)
    # Computing the global critical erosion thickness
    part_counter = 0
    # Particle number counter
    for i_part_ind, i_particle in enumerate(particles):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Running through all the particles and their periodic images
                    if part_counter == 0:
                        particle_surf = i_particle.generatePointsOnSurface(
                            n_surf_points, erosion_thick=global_crit_ero_thick
                        ) + Particle.box * np.array([j, k, l])
                    else:
                        particle_surf = np.concatenate(
                            (
                                particle_surf,
                                (
                                    i_particle.generatePointsOnSurface(
                                        n_surf_points,
                                        erosion_thick=global_crit_ero_thick,
                                    )
                                    + Particle.box * np.array([j, k, l])
                                ),
                            ),
                            axis=0,
                        )
                    # Sampling points on the surface of each eroded particle and collecing then
            part_counter += 1
            # Updating the counter
    print(particle_surf)
    auxiliar_voronoi = Voronoi(particle_surf)
    # Obraining the auxiliar voronoi for the construction of the set voronoi
    set_voronoi = set3DVoronoi(auxiliar_voronoi, particles)
    # Computing the set voronoi of the particles
    return set_voronoi


def compute2DStandardVoronoi(particles):
    """
    Compute the standard Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    Returns
    -------
    std_voronoi: `.scipy.Qhull.Voronoi`
        Standard voronoi of the particles.
    """
    particle_centers = []
    # Intializing the list containing the centers of the particles and their images
    for i_part_ind, i_particle in enumerate(particles):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Running through all the particles and their periodic images
                particle_centers.append(
                    i_particle.position_center + Particle.box * np.array([j, k])
                )
                # Sampling points on the surface of each eroded particle and collecing then
    std_voronoi = Voronoi(particle_centers)
    # Computing the standard voronoi of the particles
    return std_voronoi


def compute3DStandardVoronoi(particles):
    """
    Compute the standard Voronoi of the *particles*.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles in the RVE.

    Returns
    -------
    std_voronoi: `.scipy.Qhull.Voronoi`
        Standard voronoi of the particles.
    """
    particle_centers = []
    # Intializing the list containing the centers of the particles and their images
    for i_part_ind, i_particle in enumerate(particles):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Running through all the particles and their periodic images
                    particle_centers.append(
                        i_particle.position_center + Particle.box * np.array([j, k, l])
                    )
                # Sampling points on the surface of each eroded particle and collecing then
    std_voronoi = Voronoi(particle_centers)
    # Computing the standard voronoi of the particles
    return std_voronoi


def computeTestStatChiSquared(array_samples):
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


def doVoronoiAnalysis(
    particles,
    rve_dims,
    dp_dir,
    voronoi_type="standard",
    plot_voronoi=True,
    plot_IMTs=True,
):
    """
    Do a Voronoi analysis on the RVE.

    Parameters
    ----------
    particles: list(`.Particle`)
        List of particles inside the RVE.

    voronoi_type: {'set', 'standard', 'weighted'}
        Type of Voronoi to be computed.
    """
    if particles[0].dim == 2:
        # 2D problem
        if voronoi_type == "set":
            # The required Voronoi is a set Voronoi
            voronoi = compute2DSetVoronoi(particles)
        elif voronoi_type == "standard":
            # The required Voronoi is a standard Voronoi
            voronoi = compute2DStandardVoronoi(particles)
        elif voronoi_type == "weighted":
            # The required Voronoi is a weighted Voronoi
            voronoi = compute2DStandardVoronoi(particles)
        if plot_voronoi:
            plotVoronoi2D(particles, voronoi, dp_dir, voronoi_type)
        if plot_IMTs:
            IMTs, in_box = compute2DIrreducibleMinkowskiTensors(voronoi)
            # Computing the irreducible Minkowski tensors for the current microsturcture
            Particle.IMTs = np.array(IMTs)[in_box, :]
            # Saving the irreducible Minkowski tensors of the voronoi cells associated with
            # particles inside the box
            plotVoronoi2DwithIMTs(particles, voronoi, IMTs, dp_dir, voronoi_type)
    elif particles[0].dim == 3:
        # if voronoi_type == 'standard':
        if voronoi_type == "set":
            voronoi = compute3DSetVoronoi(particles)
        if voronoi_type == "standard":
            voronoi = compute3DStandardVoronoi(particles)
        if plot_voronoi:
            # print(voronoi.ridge_vertices)
            # Saving the irreducible Minkowski tensors of the voronoi cells associated with
            # particles inside the box
            plotVoronoi3D(particles, voronoi, rve_dims, dp_dir, voronoi_type)
        if plot_IMTs:
            IMTs = compute3DIrreducibleMinkowskiTensors(voronoi)
            Particle.IMTs = np.array(IMTs)
            plotVoronoi3DwithIMTs(
                particles, voronoi, rve_dims, IMTs, dp_dir, voronoi_type
            )


# def plotMetrics

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    phi = np.linspace(0, 2 * np.pi, 500, endpoint=True)
    print("here")
    if False:
        dp_dir = "/home/zeluis/Documents/Tese/programa/studies/thermostats/minkowski/fundamental_forms_2D/fundamental_forms"
        radius_all = [[], [], [], [], [], [], []]
        xmax = [[], [], [], [], [], [], []]
        xmin = [[], [], [], [], [], [], []]
        ymax = [[], [], [], [], [], [], []]
        ymin = [[], [], [], [], [], [], []]
        for degree in range(7):
            coeffs = np.zeros((7))
            coeffs[0] = 1
            coeffs[degree] = 1
            radius_all[degree] = np.array([pos_vec(i_phi, coeffs) for i_phi in phi])
            xmax[degree] = np.max(radius_all[degree][:, 0])
            ymax[degree] = np.max(radius_all[degree][:, 1])
            xmin[degree] = np.min(radius_all[degree][:, 0])
            ymin[degree] = np.min(radius_all[degree][:, 1])

        xdist = np.max(xmax) - np.min(xmin)
        ydist = np.max(ymax) - np.min(ymin)
        for degree in range(7):
            _, axis, (w_fig, h_fig) = createFigure(nrows=3, ncols=2)
            plt.axis("equal")
            # plt.plot(phi, 1 + np.cos(3*phi))
            # plt.plot(phi, 3 + 1/2*np.exp(-3*1j*phi) + 1/2*np.exp(3*1j*phi) + 1)
            radius = radius_all[degree]
            plt.fill(radius[:, 0], radius[:, 1], color="k")  # , color=color)
            xcenter = (np.max(radius[:, 0]) + np.min(radius[:, 0])) / 2
            ycenter = (np.max(radius[:, 1]) + np.min(radius[:, 1])) / 2
            plt.xlim(xcenter - xdist, xcenter + xdist)
            plt.ylim(ycenter - ydist, ycenter + ydist)
            axis.axis("off")
            plt.savefig(dp_dir + "_fund" + str(degree) + ".pdf", bbox_inches="tight")
            plt.close()
    else:
        dp_dir = "/home/zeluis/Documents/Tese/programa/studies/thermostats/minkowski/example_2D_convex_4/"
        _, axis, (w_fig, h_fig) = createFigure(nrows=4, ncols=3)

        vertices_1 = np.array(
            [[230, 99.6333], [100, 339.633], [328, 340.6333], [301, 170]]
        )
        region_1 = [[3, 2, 1, 0]]
        vertices_2 = np.array(
            [[223, 113], [115, 277], [326, 322], [368, 153], [300, 115]]
        )
        region_2 = [[4, 3, 2, 1, 0]]
        vertices_3 = np.array(
            [[223, 113], [115, 277], [156, 324], [311, 102], [310, 60]]
        )
        region_3 = [[4, 3, 2, 1, 0]]
        vertices_4 = np.array(
            [
                [250, 100],
                [386, 146],
                [386, 264],
                [249, 378],
                [142, 315],
                [120.09618943233414, 175.00000000000009],
                [177, 110],
            ]
        )
        region_4 = [[5, 4, 3, 2, 1, 0]]
        vertices_5 = np.array(
            [[230, 99.6333], [100, 339.633], [328, 340.6333], [227, 179]]
        )
        region_5 = [[3, 2, 1, 0]]
        vertices_6 = np.array(
            [[223, 113], [115, 277], [326, 322], [368, 153], [261, 148]]
        )
        region_6 = [[4, 3, 2, 1, 0]]
        vertices_7 = np.array([[301, 39], [115, 277], [139, 281], [337, 54], [304, 70]])
        region_7 = [[4, 3, 2, 1, 0]]
        vertices_8 = np.array(
            [
                [250, 100],
                [378, 155],
                [386, 264],
                [257, 373],
                [142, 315],
                [126, 178],
                [196, 158],
            ]
        )
        region_8 = [[5, 4, 3, 2, 1, 0]]
        vertices_9 = np.array([[159, 288], [179, 308], [319, 206], [299, 180]])
        region_9 = [[0, 1, 2, 3]]
        vertices = vertices_9
        region = region_9
        test_pol = Polygon(vertices, region)
        coeffs = compute2DIrreducibleMinkowskiTensorsPolygon(test_pol, degree=6)[0]
        # coeffs = [4, 0, 0, 3, 0, 0, 3]
        norm_dens = np.array([normal_density(i_phi, coeffs) for i_phi in phi])
        # plt.plot(phi, norm_dens, color='k')
        plt.axis("equal")
        plt.fill(vertices[:, 0], vertices[:, 1], color="k")
        plt.xlim(np.min(vertices[:, 0]) - 0.1, np.max(vertices[:, 0]) + 0.1)
        plt.ylim(np.min(vertices[:, 1]) - 0.1, np.max(vertices[:, 1]) + 0.1)
        # radius = (
        #     phi
        #     + 1j*0.21/2*(np.exp(-2*1j*phi)-1)
        #     - 1j*0.21/2*(np.exp(2*1j*phi)-1)
        #     )
        #     # 1/(-1j)*np.exp(-1j*phi) \
        #     # + 0.21*(1/1j*np.exp(1j*phi) + 1/(-3*1j)*np.exp(-3*1j*phi)) \
        #     # + 0.18*(1/(2*1j)*np.exp(1j*2*phi) + 1/(-4*1j)*np.exp(-4*1j*phi)) \
        #     # + 0*(1/(3*1j)*np.exp(1j*3*phi) + 1/(-5*1j)*np.exp(-5*1j*phi)) \
        #     # + 0*(1/(4*1j)*np.exp(1j*4*phi) + 1/(-6*1j)*np.exp(-6*1j*phi)) \
        #     # + 0*(1/(5*1j)*np.exp(1j*5*phi) + 1/(-7*1j)*np.exp(-7*1j*phi))
        #     # 2/(-1j)*np.exp(-1j*phi) + 1/1j*np.exp(1j*phi) + 1/(2*1j)*np.exp(1j*2*phi) + 1/(-3*1j)*np.exp(-3*1j*phi) + 1/(-4*1j)*np.exp(-4*1j*phi)
        radius = np.array([pos_vec(i_phi, coeffs) for i_phi in phi])

        axis.axis("off")
        plt.savefig(dp_dir + "poly" + ".pdf", bbox_inches="tight")
        plt.close()
        _, axis, (w_fig, h_fig) = createFigure(nrows=4, ncols=3)
        plt.axis("equal")
        # plt.plot(phi, 1 + np.cos(3*phi))
        # plt.plot(phi, 3 + 1/2*np.exp(-3*1j*phi) + 1/2*np.exp(3*1j*phi) + 1)
        plt.fill(radius[:, 0], radius[:, 1], color="k")  # , color=color)
        plt.xlim(np.min(radius[:, 0]) - 0.1, np.max(radius[:, 0]) + 0.1)
        plt.ylim(np.min(radius[:, 1]) - 0.1, np.max(radius[:, 1]) + 0.1)
        axis.axis("off")
        plt.savefig(dp_dir + "approx" + ".pdf", bbox_inches="tight")
        plt.close()
        from matplotlib import cm

        print(np.angle(coeffs))
        IMTs = np.abs(coeffs) / np.abs(coeffs[0])
        colors = cm.Blues(IMTs[2:])
        _, axis, (w_fig, h_fig) = createFigure(nrows=4, ncols=3)
        # plt.polar(phi, np.real(norm_dens), color='k')
        rects = plt.barh(
            range(5),
            IMTs[2:],
            height=1,
            tick_label=["$q_2$", "$q_3$", "$q_4$", "$q_5$", "$q_6$"],
            color=colors,
        )
        rect_labels = []
        # Lastly, write in the ranking inside each bar to aid in interpretation
        for ind, rect in enumerate(rects):
            # Rectangle widths are already integer-valued but are floating
            # type, so it helps to remove the trailing decimal point and 0 by
            # converting width to int type
            width = rect.get_width()

            rankStr = str(np.round(IMTs[2 + ind], decimals=3))
            # The bars aren't wide enough to print the ranking inside
            if width < 0.5:
                # Shift the text to the right side of the right edge
                xloc = 3
                # Black against white background
                clr = "black"
                align = "left"
            else:
                # Shift the text to the left side of the right edge
                xloc = -3
                # White on magenta
                clr = "white"
                align = "right"

            # Center the text vertically in the bar
            yloc = rect.get_y() + rect.get_height() / 2
            label = axis.annotate(
                rankStr,
                xy=(width, yloc),
                xytext=(xloc, 0),
                textcoords="offset points",
                ha=align,
                va="center",
                color=clr,
                weight="bold",
                clip_on=True,
                fontsize=10,
            )
            rect_labels.append(label)
        plt.xticks([])
        plt.xlim([0, 1])
        # axis.axis('off')
        plt.savefig(dp_dir + "polar" + ".pdf", bbox_inches="tight")
        plt.show()
        plt.close()

    # vertices = np.array(
    #     [[150, 150],
    #      [150, 350],
    #      [350, 350],
    #      [350, 150],
    #      [250, 167]])
    # region = [[4, 3, 2, 1, 0]]
    # test_pol = Polygon(vertices, region)
    #
    # IMTs = computeIrreducibleMinkowskiTensors(test_pol)[0]
    # print(np.abs(IMTs[0]))
    # print(np.abs(IMTs[2]/IMTs[0]))
    # print(np.abs(IMTs[3]/IMTs[0]))
    # print(np.abs(IMTs[4]/IMTs[0]))
    # print(np.abs(IMTs[5]/IMTs[0]))
    # print(np.abs(IMTs[6]/IMTs[0]))
