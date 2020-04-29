from particle_classes import Particle, Disk, Ellipse, Sphere, Ellipsoid, CylindricalFiber

from path_analysis import plotVoronoi2D, plotVoronoi2DwithIMTs

from scipy.spatial import Voronoi

import numpy as np

class Polygon():

    def __init__(self, vertices, region):
        self.vertices = vertices
        self.regions = region

class setVoronoi():

    def __init__(self, construction_voronoi, particles):
        import matplotlib.pyplot as plt
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
    loc_erosion_thick = [i_particle.computeCriticalErosionThickness() for i_particle in particles]
    # List of critical erosion thickness for each particle
    glob_crit_erosion_thick = np.min(loc_erosion_thick)
    # The global critical erosion thickness is the smallest of all the critical erosion
    # thickness for each particle
    return glob_crit_erosion_thick


def computeIrreducibleMinkowskiTensors(voronoi):
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

def computeSetVoronoi(particles, n_surf_points=20):
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


def computeStandardVoronoi(particles, n_surf_points=20):
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


def doVoronoiAnalysis(particles, dp_dir, voronoi_type='set', plot_voronoi=True, plot_IMTs=True):
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
            voronoi = computeSetVoronoi(particles)
        elif voronoi_type == 'standard':
        # The required Voronoi is a standard Voronoi
            voronoi = computeStandardVoronoi(particles)
        if plot_voronoi:
            plotVoronoi2D(particles, voronoi, dp_dir, voronoi_type)
        IMTs = computeIrreducibleMinkowskiTensors(voronoi)
        if plot_IMTs:
            plotVoronoi2DwithIMTs(particles, voronoi, IMTs, dp_dir, voronoi_type)
