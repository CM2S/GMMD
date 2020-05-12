from particle_classes import Particle, Disk, Ellipse, Sphere, Ellipsoid, CylindricalFiber

from path_analysis import plotVoronoi2D, plotVoronoi2DwithIMTs, plotVoronoi3D, plotVoronoi3DwithIMTs

from scipy.spatial import Voronoi

from scipy.special import sph_harm

import numpy as np

class Polygon():

    def __init__(self, vertices, region):
        self.vertices = vertices
        self.regions = region


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

class setVoronoi():

    def __init__(self, construction_voronoi, particles):
        old_regions = []
        self.points = []
        removed_vertices = []
        for i_particle in particles:
            for j in range(-1, 2):
                for k in range(-1, 2):
                    all_regions = []
                    current_removed = []
                    self.points.append(i_particle.position_center + Particle.box*np.array([j, k]))
                    for j_vertex in range(len(construction_voronoi.vertices)):
                        if i_particle.pointInside(construction_voronoi.vertices[j_vertex] - Particle.box*np.array([j, k])):
                            removed_vertices.append(j_vertex)
                            current_removed.append(j_vertex)
                    for region in construction_voronoi.regions:
                        if any([removed_vertex in region for removed_vertex in current_removed]):
                            print('region', region)
                            region_no_int_vert = [ind_vert for ind_vert in region if ind_vert not in removed_vertices]
                            print('clean', region_no_int_vert)
                            all_regions += region_no_int_vert
                            print('current', all_regions)
                    if len(all_regions) > 0:
                        list_vert_reg = list(set(all_regions))
                        old_regions.append(list_vert_reg)

        print('old', old_regions)
        old_ridge_vertices = list(construction_voronoi.ridge_vertices)
        for ridge in construction_voronoi.ridge_vertices:
            if any([(removed_vertex in ridge or -1 in ridge) for removed_vertex in removed_vertices]):
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
        self.ridge_vertices = [newIndices(i_ridge, removed_vertices) for i_ridge in old_ridge_vertices]
        self.regions = [newIndices(i_region, removed_vertices) for i_region in old_regions]

        # self.vertices = np.array([vertex for vertex in construction_voronoi.vertices if np.where(construction_voronoi.vertices == vertex)])
        self.points = np.array(self.points)

        print(self.regions)
        print('here')
        # self.points = construction_voronoi.points
        # self.ridge_points = construction_voronoi.ridge_points
        # self.ridge_vertices = construction_voronoi.ridge_vertices
        # self.vertices = construction_voronoi.vertices
        # self.furthest_site = construction_voronoi.furthest_site

class set3DVoronoi():

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
                        self.points.append(i_particle.position_center + Particle.box*np.array([j, k, l]))
                        # Saving the center points of the particles
                        for j_vertex in range(len(construction_voronoi.vertices)):
                        # Going through all the vertices of the construction voronoi
                            if i_particle.pointInside(construction_voronoi.vertices[j_vertex] - Particle.box*np.array([j, k, l])):
                            # If the vertex is inside a particle it is removed
                                removed_vertices.append(j_vertex)
                                # Appending the vertex to the list of all removed vertices
                                current_removed.append(j_vertex)
                                # Appending the vertex to the list of vertices removed
                                # while analyising this particle
                        for region in construction_voronoi.regions:
                        # Running through all the regions in the construction voronoi
                            if any([removed_vertex in region for removed_vertex in current_removed]):
                            # If the region contains one of the removed vertices
                            # corresponding to the current particle
                                region_no_int_vert = [ind_vert for ind_vert in region if ind_vert not in removed_vertices]
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
            if any([(removed_vertex in ridge or -1 in ridge) for removed_vertex in removed_vertices]):
            # If there is a removed vertex in the ridge or a vertice out of boundary (-1)
            # remove it
                old_ridge_vertices.remove(ridge)

        self.vertices = np.delete(construction_voronoi.vertices, removed_vertices, 0)
        # Delete the removed vertices and save the remaining vertices
        self.ridge_vertices = [newIndices(i_ridge, removed_vertices) for i_ridge in old_ridge_vertices]
        # Save the ridge vertices changing the indices to account for the removed vertices
        self.regions = [newIndices(i_region, removed_vertices) for i_region in old_regions]
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


def  computeGlobalCriticalErosionThickness(particles):
    """Compute the global critical erosion thickness."""
    loc_erosion_thick = [i_particle.computeCriticalErosionThickness() for i_particle in particles]
    # List of critical erosion thickness for each particle
    glob_crit_erosion_thick = np.min(loc_erosion_thick)
    # The global critical erosion thickness is the smallest of all the critical erosion
    # thickness for each particle
    return glob_crit_erosion_thick


def compute2DIrreducibleMinkowskiTensors(voronoi):
    """Compute the Irreducible Minkowski Tensors."""
    IMT_region = []
    # Initializing the list containing the list of IMTs for each Voronoi cell
    for i_region in voronoi.regions:
        if len(i_region) == 0:
            continue
        if any([vertex == -1 for vertex in i_region]):
            continue
    # Running through all the cells in the Voronoi
        IMT_region.append([])
        # Initializing the list containing the IMTs of the cell
        n_vertices = len(i_region)
        sides = \
            [voronoi.vertices[i_region[j_pair]]
             - voronoi.vertices[i_region[np.mod(j_pair + 1, n_vertices)]]
             for j_pair in range(n_vertices)]
        lengths = [np.linalg.norm(i_side) for i_side in sides]
        angles = []
        for i_side in sides:
            if i_side[0] >= 0 and i_side[1] >= 0:
                angles.append(np.pi/2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            elif i_side[0] >= 0 and i_side[1] <= 0:
                angles.append(np.pi/2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            elif i_side[0] <= 0 and i_side[1] <= 0:
                angles.append(3*np.pi/2 + np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            elif i_side[0] <= 0 and i_side[1] >= 0:
                angles.append(3*np.pi/2 - np.arctan2(np.abs(i_side[1]), np.abs(i_side[0])))
            else:
                print(i_side)
        # angles = [np.pi/2 - np.arctan(i_side[1]/i_side[0]) for i_side in sides]
        print('angles', angles)
        print('lengths', lengths)
        for j_tensor in range(7):
        # Computing the 7 first IMTs
            IMT_region[-1].append(np.sum([lengths[k_side]*np.exp(1j
                                  * j_tensor*angles[k_side]) for k_side in range(len(lengths))]))

    return IMT_region

def compute3DIrreducibleMinkowskiTensors(voronoi):
    """Compute the Irreducible Minkowski Tensors."""
    IMT_region = []
    # Initializing the list containing the list of IMTs for each Voronoi cell
    for i_particle in range(13, len(voronoi.point_region), 27):
        center_point = voronoi.points[i_particle]
        i_particle_region = voronoi.regions[voronoi.point_region[i_particle]]
        area_ridge = []
        normal_ridge = []
        angles_normal_ridge = []
        IMT_region.append([])
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
        print('angles, normal', [(angles_normal_ridge[i], normal_ridge[i]) for i in range(len(normal_ridge))])
        for order in range(-6, 7):
            for degree in range(7):
                if np.abs(order) <= degree:
                    phi[degree, order + 6] = np.sum([area_ridge[k_face]*sph_harm(
                        order, degree, angles_normal_ridge[k_face][0], angles_normal_ridge[k_face][1]) for k_face in range(len(area_ridge))])
        print('phi', phi)
        for degree in range(7):
        # Computing the 7 first IMTs
            IMT_region[-1].append(np.sqrt(1/A_total**2*np.sum([np.abs(phi[degree, order + 6])**2 for order in range(-6, 7)])))

    print(IMT_region)
    return IMT_region


def areaFace(vertices):
    """Compute the area of the polygon defined by *vertices*."""
    center_gravity = 1/len(vertices)*np.sum(vertices, axis=0)
    # Computing the center of the polygon
    area = 1/2*np.sum([np.linalg.norm(np.cross(vertices[k_vertex] - center_gravity, vertices[np.mod(k_vertex + 1, len(vertices))] - center_gravity)) for k_vertex, _ in enumerate(vertices)])
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
        out_unit_vector = normal_vector/np.linalg.norm(normal_vector)
    else:
        out_unit_vector = -1*normal_vector/np.linalg.norm(normal_vector)
    return out_unit_vector


def unitVectorToSphCoord(unit_vector):
    """Convert unit vector to spherical coordinates"""
    theta = np.arctan2(unit_vector[1], unit_vector[0])
    if theta < 0:
        theta += 2*np.pi
    phi = np.arctan2(np.sqrt(unit_vector[0]**2 + unit_vector[1]**2), unit_vector[2])
    if phi < 0:
        phi += 2*np.pi
    return [theta, phi]

def compute2DSetVoronoi(particles, n_surf_points=20):
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
                    particle_surf = (i_particle.generatePointsOnSurface(
                        n_surf_points, erosion_thick=global_crit_ero_thick)
                        + Particle.box*np.array([j, k]))
                else:
                    particle_surf = np.concatenate(
                        (particle_surf,
                         (i_particle.generatePointsOnSurface(
                            n_surf_points, erosion_thick=global_crit_ero_thick)
                          + Particle.box*np.array([j, k]))), axis=0)
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
                particle_centers.append(np.concatenate((i_particle.position_center
                                        + Particle.box*np.array([j, k]), [np.sqrt(C - i_particle.radius)])))
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
                        particle_surf = (i_particle.generatePointsOnSurface(
                            n_surf_points, erosion_thick=global_crit_ero_thick)
                            + Particle.box*np.array([j, k, l]))
                    else:
                        particle_surf = np.concatenate(
                            (particle_surf,
                             (i_particle.generatePointsOnSurface(
                                n_surf_points, erosion_thick=global_crit_ero_thick)
                              + Particle.box*np.array([j, k, l]))), axis=0)
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
                particle_centers.append(i_particle.position_center
                                        + Particle.box*np.array([j, k]))
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
                    particle_centers.append(i_particle.position_center
                                            + Particle.box*np.array([j, k, l]))
                # Sampling points on the surface of each eroded particle and collecing then
    std_voronoi = Voronoi(particle_centers)
    # Computing the standard voronoi of the particles
    return std_voronoi

def doVoronoiAnalysis(particles, rve_dims, dp_dir, voronoi_type='standard', plot_voronoi=True, plot_IMTs=True):
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
        if voronoi_type == 'set':
        # The required Voronoi is a set Voronoi
            voronoi = compute2DSetVoronoi(particles)
        elif voronoi_type == 'standard':
        # The required Voronoi is a standard Voronoi
            voronoi = compute2DStandardVoronoi(particles)
        elif voronoi_type == 'weighted':
        # The required Voronoi is a weighted Voronoi
            voronoi = compute2DStandardVoronoi(particles)
        if plot_voronoi:
            plotVoronoi2D(particles, voronoi, dp_dir, voronoi_type)
        IMTs = compute2DIrreducibleMinkowskiTensors(voronoi)
        if plot_IMTs:
            plotVoronoi2DwithIMTs(particles, voronoi, IMTs, dp_dir, voronoi_type)
    elif particles[0].dim == 3:
        # if voronoi_type == 'standard':
        if voronoi_type == 'set':
            voronoi = compute3DSetVoronoi(particles)
        if voronoi_type == 'standard':
            voronoi = compute3DStandardVoronoi(particles)
        if plot_voronoi:
            # print(voronoi.ridge_vertices)
            IMTs = compute3DIrreducibleMinkowskiTensors(voronoi)
            plotVoronoi3D(particles, voronoi, rve_dims, dir, voronoi_type)
            plotVoronoi3DwithIMTs(particles, voronoi, rve_dims, IMTs, dir, voronoi_type)
