"""
Microstructure Generation Interface (DATAGEM Program)(????)
==========================================================================================
Summary:
...
------------------------------------------------------------------------------------------
Development history:
Zé Luís P. Vila-Chã | March 2020 | Initial coding.
"""
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
import numpy as np
# Working with arrays
import pickle
# Dumping files in a binary format
import time
# To compute the time taken
import matplotlib.pyplot as plt
# Plotting capabilities
from integration_methods import Newmark
# Importing an integration method for the equation of motion
from particle_classes import Disk, Particle, Ellipse, Sphere, Ellipsoid, CylindricalFiber, RVE
# Importing the particle class
from meshing_interface import generateMesh, checkMeshSpecs
# Importing meshing interfaces
import error_classes as errors
# Importing the error clases
import os
import shutil

import sys
from path_analysis import plotPaths, plotParticles
# ==========================================================================================


# def print2(*objects):
#     """Print to the terminal and to the screen."""
#     print(*objects)
#     # Print to default sys.stdout
#     screen_file = open(screen_path, 'a')
#     print(*objects, file=screen_file)
#     # Print to '.screen file'
#     screen_file.close()


def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def newVerletList(particles):
    '''
    This function creates a new Verlet list for all the particles
    '''
    dim = particles[0].dim
    # Saving the dimension of the problem

    for i_particle in range(len(particles)):
    # Running though all the particles
        particles[i_particle].verlet_list = []
        # Resetting the Verlet list of particle i
        particles[i_particle].displacement_last_verlet = np.zeros(dim)
        # Resetting the displacement of the center of mass of the particle relative to its
        # neighboorhood
        pos_cell_list_dim = []
        # Initializing the list containing the position of the particle in the grid
        # assuming: 2D: the cells are numbered from left to right and from bottom to top
        for j_dim in range(dim):
        # Running through all the dimensions
            pos_cell_list_dim.append(np.int(np.floor(
                particles[i_particle].position_center[j_dim]
                / Particle.cell_side_length[j_dim])))
            # j_dim-position of the particle in the grid
        if dim == 2:
        # 2D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[0]
            # Saving the position in the cell list of particle i_particle
            for k_neighboor_cell in range(9):
            # Running through the neighboor cells
                pos_neighboor_cell = \
                    neighboorCell(pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim)
                # Computing the index of the neighboor cell
                for j_particle in Particle.cell_list[pos_neighboor_cell]:
                # Running through all the particles in the neighboring cell
                    if particles[i_particle].intersectionVerlet(particles[j_particle]):
                    # If the neighboorhoods of the particles intersect
                        particles[i_particle].verlet_list.append(j_particle)
                        # Add the particle j_particle to i_particle's Verlet list
        elif dim == 3:
        # 3D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[0] + \
                pos_cell_list_dim[2]*Particle.n_cell_dim[0]*Particle.n_cell_dim[1]
            # Saving the position in the cell list of particle i_particle
            for k_neighboor_cell in range(3**3):
            # Running through the neighboor cells
                pos_neighboor_cell = \
                    neighboorCell(pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim)
                # Computing the index of the neighboor cell
                for j_particle in Particle.cell_list[pos_neighboor_cell]:
                # Running through all the particles in the neighboring cell
                    if particles[i_particle].intersectionVerlet(particles[j_particle]):
                    # If the neighboorhoods of the particles intersect
                        particles[i_particle].verlet_list.append(j_particle)
                        # Add the particle j_particle to i_particle's Verlet list

def neighboorCell(pos_current_cell, local_pos_neighboor_cell, dim, n_cells):
    '''
    This function computes the global cell position of the neighboor cell.

    Parameters:
        pos_current_cell: integer
            Global position of the current cell
        local_pos_neighboor_cell: integer
            Local position of the neighboor cell
        dim: integer
            Dimension of the problem
        n_cells: list
            Number of cells in each direction (0:x; 1:y; 2:z)

    Returns:
        pos_neighboor_cell: integer
            Gloval position of the neighboor cell
    '''

    if dim == 2:
    # 2D problem
        local_row_pos_neigh = np.int(np.mod(np.floor(local_pos_neighboor_cell/3), 3) - 1)
        # Local row position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_col_pos_neigh = np.int(np.mod(local_pos_neighboor_cell, 3) - 1)
        # Local column position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        pos_neighboor_cell = \
            np.int(pos_current_cell + local_col_pos_neigh + local_row_pos_neigh*n_cells[0])
        # Global position of the neighboor cell without enforcing periodic boundary
        # conditions
        if pos_current_cell < n_cells[0] and local_row_pos_neigh == -1:
        # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        elif pos_current_cell >= n_cells[0]*(n_cells[1]-1) and local_row_pos_neigh == 1:
        # Upper row of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        if np.mod(pos_current_cell + 1, n_cells[0]) == 0 and local_col_pos_neigh == 1:
        # Right column of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[0]
            # Enforcing the periodic boundary conditions
        elif np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1:
        # Left column of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[0]
            # Enforcing the periodic boundary conditions
    elif dim == 3:
    # 3D problem
        local_row_pos_neigh = np.int(np.mod(np.floor(local_pos_neighboor_cell/3), 3) - 1)
        # Local row position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_col_pos_neigh = np.int(np.mod(local_pos_neighboor_cell, 3) - 1)
        # Local column position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_lay_pos_neigh = np.int(np.mod(np.floor(local_pos_neighboor_cell/9), 3) - 1)
        # Local layer position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        pos_neighboor_cell = (
            np.int(pos_current_cell
                   + local_col_pos_neigh
                   + local_row_pos_neigh*n_cells[0]
                   + local_lay_pos_neigh*n_cells[0]*n_cells[1]))
        # Global position of the neighboor cell without enforcing periodic boundary
        # conditions
        if pos_current_cell < n_cells[0] and local_row_pos_neigh == -1:
        # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        elif pos_current_cell >= n_cells[0]*(n_cells[1] - 1) and local_row_pos_neigh == 1:
        # Upper row of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        if np.mod(pos_current_cell + 1, n_cells[0]) == 0 and local_col_pos_neigh == 1:
        # Right column of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[0]
            # Enforcing the periodic boundary conditions
        elif np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1:
        # Left column of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[0]
            # Enforcing the periodic boundary conditions
        if pos_current_cell < n_cells[1]*n_cells[0] and local_lay_pos_neigh == -1:
        # Firsl layer of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]*n_cells[0]*n_cells[2]
            # Enforcing the periodic boundary conditions
        elif (pos_current_cell > n_cells[1]*n_cells[0]*(n_cells[2] - 1) - 1
              and local_lay_pos_neigh == 1):
        # Last layer of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1]*n_cells[0]*n_cells[2]
            # Enforcing the periodic boundary conditions

    return pos_neighboor_cell

def newCellList(particles):
    '''
    This function computes a new cell list for particles
    '''

    dim = particles[0].dim

    n_cells = np.prod(np.array(Particle.n_cell_dim))

    Particle.cell_list = [[] for i in range(n_cells) ]

    for i_particle in range(len(particles)):
    # Running through all the particles
        pos_cell_list_dim = []
        # Initializing the list containing the position of the cell in each direction
        # with the origin at the top left
        for j_dim in range(dim):
        # Running through all the dimensions
            pos_cell_list_dim.append(np.int(np.floor(
                particles[i_particle].position_center[j_dim]
                / Particle.cell_side_length[j_dim])))
            # j_dim-position of the particle in the grid
        if dim == 2:
        # 2D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[0]
            # Saving the position in the cell list of particle i_particle
        if dim == 3:
        # 3D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[0] + \
                pos_cell_list_dim[2]*Particle.n_cell_dim[0]*Particle.n_cell_dim[1]
            # Saving the position in the cell list of particle i_particle
        Particle.cell_list[pos_cell_list].append(i_particle)


def computeForces(particles, speed_up_scheme):
    """
    Compute the forces between all the particle pairs in the system.

    Parameters
    ----------
    particles : list(`.Particle`)
        Array containing the Particle objects to be placed inside the RVE

    speed_up_scheme: {'Naive', 'Cell', 'Verlet'}, optional
        Speed up scheme used in the force computation
            "Naive": the forces are computed between every pair of particles (O(N**2))
            "Cell": the forces are computed making use of a cell list, such that each particle
                only interacts with the particles in its cell or the nearest neighboring
                cells (O(N))
            "Verlet": the forces are computed using a Verlet list for each particle, that in
                turn in computed using a cell list method
    """
    dim = particles[0].dim
    # Saving the dimension of the problem
    for i_particle in range(len(particles)):
    # Running through all the particles
        particles[i_particle].cleanForces()
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
        particles[i_particle].cleanOverlapArea()
        # Setting all the overlap areas to zero at the beginning of the iteration as
        # they are added sequentially as each pair is considered
    Particle.total_overlap = 0
    # Setting the total overlap to zero as it will computed again
    if speed_up_scheme == 'Naive':
    # Naive approach: O(N^2)
        for i_particle in range(len(particles)):
        # Running though all the particles
            for j_particle in range(i_particle+1, len(particles)):
            # Running through the particle pairs that have not been considered yet
                force_i_j = computeForceij(particles[i_particle], particles[j_particle])
                # Computing the force on particle i due to particle j
                particles[i_particle].force = particles[i_particle].force + force_i_j
                # Adding the force due to the interaction between particle 1 and 2 to the
                # total force acting on particle 1
                particles[j_particle].force = particles[j_particle].force - force_i_j
                # Adding the force due to the interaction between particle 1 and 2 to the
                # total force acting on particle 2
    elif speed_up_scheme == 'Cell':
    # Cell list: O(N)
        newCellList(particles)
        # Computing a new Cell list
        for i_particle in range(len(particles)):
        # Running though all the particles
            pos_cell_list_dim = []
            # Initializing the list containing the position of the particle in the grid, assuming:
            # 2D: the cells are numbered from left to right and from top to bottom
            for j_dim in range(dim):
            # Running through all the dimensions
                pos_cell_list_dim.append(np.int(np.floor(
                    particles[i_particle].position_center[j_dim]
                    / Particle.cell_side_length[j_dim])))
                # j_dim-position of the particle in the grid
            if dim==2:
            # 2D problem
                pos_cell_list = pos_cell_list_dim[0] + \
                    pos_cell_list_dim[1]*Particle.n_cell_dim[0]
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(9):
                # Running through the neighboor cells
                    pos_neighboor_cell = \
                        neighboorCell(pos_cell_list,
                                      k_neighboor_cell, dim, Particle.n_cell_dim)
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighboring cell
                        if j_particle > i_particle:
                        # Ensuring that the forces are not computed twice
                            force_i_j = computeForceij(particles[i_particle],
                                particles[j_particle])
                            # Computing the force on particle i due to particle j
                            particles[i_particle].force = particles[i_particle].force \
                                + force_i_j
                            # Adding the force due to the interaction between particle 1 and 2 to the total
                            # force acting on particle 1
                            particles[j_particle].force = particles[j_particle].force \
                                - force_i_j
                            # Adding the force due to the interaction between particle 1 and 2 to the total
                            # force acting on particle 2
            if dim==3:
            # 2D problem
                pos_cell_list = pos_cell_list_dim[0] + \
                    pos_cell_list_dim[1]*Particle.n_cell_dim[0] + \
                    pos_cell_list_dim[2]*Particle.n_cell_dim[0]*Particle.n_cell_dim[1]
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(3**3):
                # Running through the neighboor cells
                    pos_neighboor_cell = \
                        neighboorCell(pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim)
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighboring cell
                        if j_particle > i_particle:
                        # Ensuring that the forces are not computed twice
                            force_i_j = computeForceij(particles[i_particle],
                                particles[j_particle])
                            # Computing the force on particle i due to particle j
                            particles[i_particle].force = particles[i_particle].force \
                                + force_i_j
                            # Adding the force due to the interaction between particle 1 and 2 to the total
                            # force acting on particle 1
                            particles[j_particle].force = particles[j_particle].force \
                                - force_i_j
                            # Adding the force due to the interaction between particle 1 and 2 to the total
                            # force acting on particle 2


    elif speed_up_scheme == 'Verlet':
    # Cell list + Verlet list: O(N)
        newCellList(particles)
        # Computing a new cell list
        if Particle.new_verlet_list:
        # There is a need to create a new Verlet list
            newVerletList(particles)
            # Computing a new Verlet list
            Particle.new_verlet_list = False
            # Resetting the parameter that indicates the need to compute a new Verlet list
        for i_particle in range(len(particles)):
        # Running though all the particles
            # print('main',i_particle)
            for j_particle in particles[i_particle].verlet_list:
            # Running through all the particles in the neighboring cell
                # print('other',j_particle)
                if j_particle > i_particle:
                # Ensuring that the forces are not computed twice
                    force_i_j = computeForceij(particles[i_particle], particles[j_particle])
                    # Computing the force on particle i due to particle j
                    particles[i_particle].force = particles[i_particle].force + force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to the total
                    # force acting on particle 1
                    particles[j_particle].force = particles[j_particle].force - force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to the total
                    # force acting on particle 2


# ==========================================================================================
def computeForceij(particle_i, particle_j):
    '''
    Computing the force on particle_i due to particle_j
    '''
    intersection_area = particle_i.intersectionArea(particle_j)
    # Intersection area between particle i and j
    particle_i.overlap_area += intersection_area
    particle_j.overlap_area += intersection_area
    Particle.total_overlap += intersection_area
    # Updating the overlap area
    unit_vector_i_j = particle_i.intersectionVector(particle_j)
    # Unit vector from particle i to particle j
    force_i_j = -intersection_area*unit_vector_i_j
    # Computing the force on particle_i due to particle_j proportional to their
    # intersection area/volume
    return force_i_j


def integrate(particles, dt, speed_up_scheme, integration_scheme='Newmark', **kwargs):
    """Integrate the equations of motion."""
    dim = particles[0].dim
    # Dimension of the problem
    N = len(particles)
    # Number of particles
    box = Particle.box
    # Saving the size of the RVE
    for i_particle in range(N):
    # Running through all the particles
        if integration_scheme=='Newmark':
        # The integration scheme chosen was Newmark
            c = kwargs.get('damping_constant', 0)
            [new_position, new_velocity, new_accelaration] = \
                Newmark(particles[i_particle].position_center,
                particles[i_particle].velocity_center,
                Particle.global_force_factor*np.array([particles[i_particle].force],dtype='float').T,
                particles[i_particle].volume()*np.eye(particles[i_particle].dim,dtype='float'), #10e-6*np.eye(2,dtype='float'),#
                c*np.eye(particles[i_particle].dim,dtype='float'),
                np.zeros((particles[i_particle].dim,particles[i_particle].dim),dtype='float'),
                dt,
                1,
                dim)
            # print(particles[i_particle].position_center,
            #     particles[i_particle].velocity_center,
            #     Particle.global_force_factor*np.array([particles[i_particle].force],dtype='float').T,
            #     1e1*particles[i_particle].volume()*np.eye(2,dtype='float'), #10e-6*np.eye(2,dtype='float'),#
            #     c*np.eye(2,dtype='float'),
            #     np.zeros((2,2),dtype='float'),
            #     dt,
            #     1,
            #     dim)
            # Obtaining the new position and velocity of particle i
        elif integration_scheme == 'Verlet':
        # The integration scheme chosen was Verlet
            pass
        else:
        # No integration scheme was chosen
            print('No integration scheme was chosen')
        if speed_up_scheme == 'Verlet':
            particles[i_particle].displacement_last_verlet += \
                particles[i_particle].position_center - new_position[:, 0]
            # Computing the displacement of the center of the particle
            # class_name_i_particle = particles[i_particle].__class__.__name__
            # if "Disk" == class_name_i_particle:
            #     radial_dimension = particles[i_particle].radius
            # elif "Eliipse" == class_name_i_particle:
            #     radial_dimension = particles[i_particle].semi_minor_axis
            # # FIX: MAKE GENERAL
            # if np.linalg.norm(particles[i_particle].displacement_last_verlet) >= \
            #         radial_dimension * (Particle.verlet_factor - 1):
            if not particles[i_particle].insideVerlet():
            # Checking if the displacement takes the particle out of its neighboorhood
                Particle.new_verlet_list = True
                # There is a need to compute a new verlet list
        new_position[:, 0] = new_position[:, 0] - box*np.floor(new_position[:, 0]/box)
        # New position enforcing boundary conditions
        particles[i_particle].position_center = new_position[:, 0]
        particles[i_particle].velocity_center = new_velocity[:, 0]
        # Updating the position and velocity of particle i
        if kwargs.get('save_history'):
        # The history of the particle's motion is required
            particles[i_particle].position_center_history.append(new_position.flatten())
            Particle.total_overlap_history.append(Particle.total_overlap)
    # putSystemAtRest(particles)
    # Putting the systemas a whole at rest

# ==========================================================================================


def generateDisks(phase, rve_dims, descriptors):
    """
    Generate disk of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: str
        Phase to which the disks will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    disks = []
    # Initializing the list containing the disks
    possible_parameters = {'r', 'area', 'n', 'vf'}
    # possible_parameters
    used_parameters = {parameter for parameter in possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    acceptable_descriptions = [{'r', 'n'}, {'r', 'vf'}, {'n', 'vf'}, {'area', 'vf'},
                               {'area', 'n'}]
    # List of acceptable collections of parameters
    if any([used_parameters == acceptable_description for
            acceptable_description in acceptable_descriptions]):
    # Checking acceptable sets of parameters
        acceptable_description = True
    else:
        acceptable_description = False
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of disks was specified
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            disks.append(Disk(phase, r[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, descriptors, phase, rve_dims)
            r = canonicalParametersDisk(current_sample, rve_dims)
            disks.append(Disk(phase, r))
            vf_real += disks[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            disks.append(Disk(phase, r))

    return disks


def generateSpheres(phase, rve_dims, descriptors):
    """Generate spheres of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: str
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    spheres = []
    # Initializing the list containing the spheres
    possible_parameters = {'r', 'volume', 'n', 'vf'}
    # possible_parameters
    used_parameters = {parameter for parameter in possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    print(used_parameters)
    # Collecting the parameters used
    acceptable_descriptions = [{'r', 'n'}, {'r', 'vf'}, {'n', 'vf'}, {'volume', 'vf'},
                               {'volume', 'n'}]
    # List of acceptable collections of parameters
    if any([used_parameters == acceptable_description for
            acceptable_description in acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of disks was specified
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors['n']):
            spheres.append(Sphere(phase, r[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, descriptors, phase, rve_dims)
            r = canonicalParametersSphere(current_sample, rve_dims)
            spheres.append(Sphere(phase, r))
            vf_real += spheres[-1].volume()/(rve_dims[0]*rve_dims[1]*rve_dims[2])
        print(vf_real)
    elif 'vf' in descriptors and 'n' in descriptors:
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors['n']):
            spheres.append(Sphere(phase, r))

    return spheres


def generateEllipses(phase, rve_dims, descriptors):
    """Generate ellipses belonging to *phase* characterized by *descriptors*."""
    ellipses = []
    # Initializing the list containing the disks
    possible_parameters = {'major_axis', 'minor_axis', 'angle', 'eccentricity', 'ratio',
                           'n', 'vf'}
    # possible_parameters
    used_parameters = {parameter for parameter in possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    print(used_parameters)
    # Collecting the parameters used
    acceptable_descriptions = [{'major_axis', 'minor_axis', 'angle', 'n'},
                               {'major_axis', 'minor_axis', 'angle', 'vf'},
                               {'major_axis', 'angle', 'n', 'vf'},
                               {'minor_axis', 'angle', 'n', 'vf'}]
    # List of acceptable collections of parameters
    if any([used_parameters == acceptable_description for
            acceptable_description in acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of ellipses was specified
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples,
                                                                     rve_dims)
        for i in range(descriptors['n']):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, descriptors, phase, rve_dims)
            [major_axis, minor_axis, angle] = canonicalParametersEllipse(current_sample,
                                                                         rve_dims)
            ellipses.append(Ellipse(phase, major_axis, minor_axis, angle))
            vf_real += ellipses[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples,
                                                                     rve_dims)
        for i in range(descriptors['n']):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))

    return ellipses


def generateEllipsoids(phase, rve_dims, descriptors):
    """Generate ellipsoids belonging to *phase* characterized by *descriptors*."""
    ellipsoids = []
    # Initializing the list containing the disks
    possible_parameters = {'axis_1', 'axis_2', 'axis_3', 'euler_angle_x', 'euler_angle_y',
                           'euler_angle_z', 'angle', 'n', 'vf'}
    # possible_parameters
    used_parameters = {parameter for parameter in possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    print(used_parameters)
    # Collecting the parameters used
    acceptable_descriptions = [
        {'axis_1', 'axis_2', 'axis_3', 'euler_angle_x', 'euler_angle_y', 'euler_angle_z',
         'angle', 'n'},
        {'axis_1', 'axis_2', 'axis_3', 'euler_angle_x', 'euler_angle_y', 'euler_angle_z',
         'angle', 'vf'}]
    # List of acceptable collections of parameters
    if any([used_parameters == acceptable_description for
            acceptable_description in acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of ellipsoids was specified
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        [axis_1, axis_2, axis_3, euler_angle_x, euler_angle_y, euler_angle_z, angle] = \
            canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors['n']):
            ellipsoids.append(Ellipsoid(
                phase, axis_1[i], axis_2[i], axis_3[i], euler_angle_x[i], euler_angle_y[i],
                euler_angle_z[i], angle[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, descriptors, phase, rve_dims)
            [axis_1, axis_2, axis_3, euler_angle_x, euler_angle_y, euler_angle_z, angle] = \
                canonicalParametersEllipsoid(current_sample, rve_dims)
            ellipsoids.append(Ellipsoid(phase, axis_1[0], axis_2[0], axis_3[0],
                                        euler_angle_x[0], euler_angle_y[0],
                                        euler_angle_z[0], angle[0]))
            vf_real += ellipsoids[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        [axis_1, axis_2, axis_3, euler_angle_x, euler_angle_y, euler_angle_z, angle] = \
            canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors['n']):
            ellipsoids.append(Ellipsoid(phase, axis_1[i], axis_2[i], axis_3[i],
                                        euler_angle_x[i], euler_angle_y, euler_angle_z[i],
                                        angle[i]))

    return ellipsoids


def generateCylindricalFibers(phase, rve_dims, descriptors):
    """
    Generate cylindrical fibers of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: str
        Phase to which the fibers will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    fibers = []
    # Initializing the list containing the fibers
    possible_parameters = {'r', 'area', 'n', 'vf', 'direction'}
    # possible_parameters
    used_parameters = {parameter for parameter in possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    print(used_parameters)
    # Collecting the parameters used
    acceptable_descriptions = [{'r', 'n', 'direction'}, {'r', 'vf', 'direction'},
                               {'n', 'vf', 'direction'}, {'area', 'vf', 'direction'},
                               {'area', 'n', 'direction'}]
    # List of acceptable collections of parameters
    if any([used_parameters == acceptable_description for
            acceptable_description in acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of fibers was specified
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            fibers.append(CylindricalFiber(phase, r[i], descriptors['direction'], rve_dims))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, descriptors, phase, rve_dims)
            r = canonicalParametersDisk(current_sample, rve_dims)
            fibers.append(CylindricalFiber(phase, r, descriptors['direction'], rve_dims))
            vf_real += fibers[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter, descriptors, phase, rve_dims, n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            fibers.append(CylindricalFiber(phase, r, descriptors['direction'], rve_dims))

    return fibers


def generateInitialConfiguration(particles, type_init_conf, **kwargs):
    """
    Generate the initial configuration (positions and velocities) for the particles.

    Parameters
    ----------
    particles: `.Particle`
        Particles in the RVE.

    type_inti_conf: {'random', 'grid'}
        Type of initial configuration.
        'random': Random configuration for the particle centers and the zero velocity.
        'grid': Particles randomly assigned to a place in a grid constructed to have an
        equal number of cells in each direction and a total number of cells larger than the
        number of particles.

    """
    if type_init_conf == 'random':
    # Random configuration for the particle centers and the zero velocity
        np.random.seed(42)
        k = 0
        for i_particle in particles:
            k += 1
        # Running through all the particles
            i_particle.setPositionCenter(Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.zeros((i_particle.dim))) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get('save_history'):
            # Saving particle history
                i_particle.position_center_history = [i_particle.position_center.flatten()]
    elif type_init_conf == 'grid':
    # Particles randomly assigned to a place in a grid constructed to have an equal number
    # of cells in each direction and a total number of cells larger than the number of
    # particles
        if particles[0].dim == 3:
            n_cells_side = np.int(np.ceil(np.cbrt(len(particles))))
            # Number of cells in each direction
            cell_length = Particle.box/n_cells_side
            # Length of the cells in each direction
            k_counter = 0
            # Initializing the counter
            grid_places = np.arange(n_cells_side**3)
            # Label of each grid place
            np.random.shuffle(grid_places)
            # Distributing the particles randomly to different cells of the grid
            for j in range(n_cells_side):
                for k in range(n_cells_side):
                    for l in range(n_cells_side):
                        if grid_places[k_counter] < len(particles):
                            # np.array([(i+1)*1/24-np.floor((i+1)*1/24), (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
                            particles[grid_places[k_counter]].setPositionCenter(np.array(
                                [j*cell_length[0]+cell_length[0]/2,
                                 k*cell_length[1]+cell_length[1]/2,
                                 l*cell_length[2]+cell_length[2]/2]))
                            # Gene<><rating the positions from a random uniform distribution between 0 and 1
                            particles[grid_places[k_counter]].setVelocityCenter(
                                np.random.uniform(low=0.01, high=0.6, size=3))  # np.array([0,0],dtype='float')
                            # Generating the velocities from a random uniform distribution between -1 and 1
                            if kwargs.get('save_history'):
                            # Saving particle history
                                particles[grid_places[k_counter]].position_center_history = [
                                    particles[grid_places[k_counter]].position_center.flatten()]
                        k_counter += 1
        elif particles[0].dim == 2:
            n_cells_side = np.int(np.ceil(np.sqrt(len(particles))))
            # Number of cells in each direction
            cell_length = Particle.box/n_cells_side
            # Length of the cells in each direction
            k_counter = 0
            # Initializing the counter
            grid_places = np.arange(n_cells_side**2)
            # Label of each grid place
            np.random.shuffle(grid_places)
            # Distributing the particles randomly to different cells of the grid
            for j in range(n_cells_side):
                for k in range(n_cells_side):
                    if grid_places[k_counter] < len(particles):
                        # np.array([(i+1)*1/24-np.floor((i+1)*1/24), (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
                        particles[grid_places[k_counter]].setPositionCenter(np.array(
                            [j*cell_length[0]+cell_length[0]/2,
                             k*cell_length[1]+cell_length[1]/2]))
                        # Gene<><rating the positions from a random uniform distribution between 0 and 1
                        particles[grid_places[k_counter]].setVelocityCenter(
                            np.random.uniform(low=0.01, high=0.6, size=2))  # np.array([0,0],dtype='float')
                        # Generating the velocities from a random uniform distribution between -1 and 1
                        if kwargs.get('save_history'):
                        # Saving particle history
                            particles[grid_places[k_counter]].position_center_history = [
                                particles[grid_places[k_counter]].position_center.flatten()]
                    k_counter += 1
    else:
        try:
            raise errors.UnsupportedInitialConfigurationType(type_init_conf)
        except errors.UnsupportedInitialConfigurationType as error:
            error.message()
            quit()


def generateSampleParameter(parameter, descriptors, phase, rve_dims, n_samples=1,
                            max_sample=100):
    """Generate a sample of values for *parameter* according to descriptors"""
    size_geom_param = {'r', 'major_axis', 'minor_axis', 'axis_1', 'axis_2', 'axis_3'}
    # Geometrical parameters related to the size of the particle that must larger than
    # ans smaller than half the size of the smallest dimension of the RVE
    if descriptors.get(parameter + '_distribution') == 'uniform':
    # the radius follows a uniform distribution
        try:
            if parameter + '_low' not in descriptors:
            # Checking if the  lower bound was supplied
                raise errors.ParameterMissing(parameter + '_low', phase)
            elif parameter + '_high' not in descriptors:
            # Checking if the upper bound was supplied
                raise errors.ParameterMissing(parameter + '_high', phase)
            elif descriptors[parameter + '_low'] >= descriptors[parameter + '_high']:
            # Checking if the lower bound is smaller than the upper bound
                raise errors.UnexpectedValue(
                    descriptors[parameter + '_low'], '{0}_low of phase {1}'.format(
                     parameter, phase),
                    'smaller than ' + parameter + '_high: {0}'.format(
                     descriptors[parameter + '_high']))
        except (errors.ParameterMissing, errors.UnexpectedValue) as error:
        # One of the parameters is missing
            error.message()
            quit()
            # Printing message and aborting
        try:
            if parameter in size_geom_param:
            # Checking if the parameter is a size parameter
                if descriptors[parameter + "_low"] < 0:
                # Ensuring that it will not produce values smaller than 0
                    raise errors.UnexpectedValue(
                        descriptors[parameter + '_low'], '{0}_low of phase {1}'.format(
                         parameter, phase),
                        'larger than 0')
                elif descriptors[parameter + "_high"] > np.min(rve_dims)/2:
                # Ensuring that it will not produce values larger than half the size of the
                # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter + '_high'], '{0}_high of phase {1}'.format(
                         parameter, phase),
                        'smaller than half the smallest dimension of the RVE: {0}'.format(
                         np.min(rve_dims)/2))
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        sample = np.random.uniform(low=descriptors[parameter + '_low'],
                                   high=descriptors[parameter + '_high'],
                                   size=n_samples)
    elif descriptors.get(parameter + '_distribution') == 'normal':
    # the radius follows a normal distribution: the paramaters 'r_sigma', the standard
    # deviation of the distribution and 'r_mean', the mean of the distribution, are needed
        try:
            if parameter + '_sigma' not in descriptors:
                raise errors.ParameterMissing(parameter + '_sigma', phase)
            elif parameter + '_mean' not in descriptors:
                raise errors.ParameterMissing(parameter + '_mean', phase)
        except errors.ParameterMissing as error:
        # One of the parameters is missing
            error.message()
            quit()
            # Printing message and aborting
        try:
            if parameter in size_geom_param:
            # Geometric size parameters
                low_25_prob = descriptors[parameter + '_mean'] - \
                    2*descriptors[parameter + '_sigma']
                # Upper bound of tail with 2.5% probability
                high_25_prob = descriptors[parameter + '_mean'] + \
                    2*descriptors[parameter + '_sigma']
                # Lower bound of tail with 2.5% probability
                if low_25_prob < 0:
                # Ensuring that the probability of generating a value smaller than 0 is
                # not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, 'low')
                elif high_25_prob > np.min(rve_dims)/2:
                # Ensuring that the probability of generating a value larger than half the
                # size of the smallest dimension of the RVE is not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, 'high')
        except errors.DangerousValueNormal as error:
            error.message()
            quit()
        k_sample = 0
        acceptable_values = False
        while k_sample < max_sample and not acceptable_values:
            sample = np.random.normal(loc=descriptors[parameter + '_mean'],
                                      scale=descriptors[parameter + '_sigma'],
                                      size=n_samples)
            # Generate a sample
            if parameter in size_geom_param:
            # Geometric size parameters
                if all([(i_sample > 0 and i_sample <= np.min(rve_dims)/2)
                       for i_sample in sample]):
                # All the values for the geometric size parameters are acceptable
                    acceptable_values = True
            else:
            # Other parameters
                acceptable_values = True
                # Any sample is acceptable
            k_sample += 1
        try:
            if not acceptable_values:
            # No acceptable sample was generated
                raise errors.UnableToGenerateSample(parameter, phase, max_sample)
        except errors.UnableToGenerateSample as error:
            error.message()
            quit()
    elif descriptors.get(parameter + '_distribution') == 'discrete':
    # the radius follows a discrete distribution
        values = []
        probabilities = []
        # Initializing the lists containing the values and corresponding probabilities that
        # characterize the discrete distribution required
        for i_descriptor in descriptors:
            if i_descriptor.startswith(parameter + '_value'):
                try:
                    if parameter in size_geom_param:
                    # Checking if the parameter is a size parameter
                        if descriptors[i_descriptor] < 0:
                        # Ensuring that it will not produce values smaller than 0
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor], '{0} of phase {1}'.format(
                                 i_descriptor, phase),
                                'larger than 0')
                        elif descriptors[parameter + "_high"] > np.min(rve_dims)/2:
                        # Ensuring that it will not produce values larger than half the size
                        # of the smallest dimension of the RVE
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor], '{0} of phase {1}'.format(
                                 i_descriptor, phase),
                                'smaller than half the smallest dimension'
                                + 'of the RVE: {0}'.format(np.min(rve_dims)/2))
                except errors.UnexpectedValue as error:
                    error.message()
                    quit()
                values.append(descriptors[i_descriptor])
                # Save the value
                try:
                    if parameter + '_prob_' + i_descriptor[-1] not in descriptors:
                        raise errors.ParameterMissing(parameter + '_prob_'
                                                      + i_descriptor[-1], phase)
                    probabilities.append(descriptors[parameter + '_prob_'
                                         + i_descriptor[-1]])
                except errors.ParameterMissing as error:
                    error.message()
                    quit()
        if len(values) == 0:
        # There are no values for the radius parameter
            try:
                raise errors.ParameterMissing(parameter + 'value_1', phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        elif np.abs(np.sum(probabilities) - 1) > 0.01:
        # The probabilities do not add up to 100%
            try:
                raise errors.ParameterErrorDiscreteProb('r_1', phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        sample = np.random.choice(values, n_samples, p=probabilities)
    elif parameter + '_distribution' in descriptors:
    # A distribution was specified but it is not supported
        try:
            raise errors.UnsupportedDistribution(
                descriptors[parameter + '_distribution'], parameter, phase)
        except errors.UnsupportedDistribution as error:
            error.message()
            quit()
    else:
    # A single value was specified
        try:
            if parameter not in descriptors:
                raise errors.ParameterMissing(parameter, phase)
        except errors.ParameterMissing as error:
            error.message()
            quit()
        try:
            if parameter in size_geom_param:
            # Checking if the parameter is a size parameter
                if descriptors[parameter] < 0:
                # Ensuring that it will not produce values smaller than 0
                    raise errors.UnexpectedValue(
                        descriptors[parameter], '{0} of phase {1}'.format(
                         parameter, phase),
                        'larger than 0')
                elif descriptors[parameter] > np.min(rve_dims)/2:
                # Ensuring that it will not produce values larger than half the size of the
                # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter], '{0} of phase {1}'.format(
                         parameter, phase),
                        'smaller than half the smallest dimension of the RVE: {0}'.format(
                         np.min(rve_dims)/2))
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        sample = np.full((n_samples), descriptors[parameter])

    return sample


def canonicalParametersEllipse(sample, rve_dims):
    """Convert the paramters in *sample* to *major_axis*, *minor_axis* and *angle*."""
    if 'major_axis' in sample and 'minor_axis' in sample:
    # Both major and minor axis were supplied
        major_axis = np.max([sample['major_axis'], sample['minor_axis']], axis=0)
        minor_axis = np.min([sample['major_axis'], sample['minor_axis']], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    elif 'major_axis' in sample and 'vf' in sample and 'n' in sample:
    # The major_axis, the volume faction and the number of particles were supplied
        volume_part = sample['vf'][0]*rve_dims[0]*rve_dims[1]/sample['n'][0]
        aux_minor_axis = volume_part/(np.pi*sample['major_axis']*1/4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([sample['major_axis'], aux_minor_axis], axis=0)
        minor_axis = np.min([sample['major_axis'], aux_minor_axis], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    # FIXME: Warnign that all particles will have the same volume
    elif 'minor_axis' in sample and 'vf' in sample and 'n' in sample:
    # The minor axis, the volume faction and the number of particles were supplied
        volume_part = sample['vf'][0]*rve_dims[0]*rve_dims[1]/sample['n'][0]
        aux_major_axis = volume_part/(np.pi*sample['minor_axis']*1/4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([aux_major_axis, sample['minor_axis']], axis=0)
        minor_axis = np.min([aux_major_axis, sample['minor_axis']], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    if 'angle' in sample:
        angle = sample['angle']

    return [major_axis, minor_axis, angle]


def canonicalParametersDisk(sample, rve_dims):
    """Convert the paramters in *sample* to *r*."""
    if 'r' in sample:
    # The radius was supplied
        r = sample['r']
    elif 'area' in sample:
    # The area of each particle was supplied
        r = np.sqrt(sample['area']/np.pi)
    elif 'vf' in sample and 'n' in sample:
    # Both the volume fraction and the number of particles was supplied
        area = sample['vf'][0]*rve_dims[0]*rve_dims[1]/sample['n'][0]
        # Area of each particle (all the same)
        r = np.sqrt(area/np.pi)
    return r


def canonicalParametersSphere(sample, rve_dims):
    """Convert the parameters in *sample* to *r* characterizing a sphere."""
    if 'r' in sample:
    # The radius was supplied
        r = sample['r']
    elif 'volume' in sample:
    # The area of each particle was supplied
        r = np.cbrt(sample['volume']/(4/3*np.pi))
    elif 'vf' in sample and 'n' in sample:
    # Both the volume fraction and the number of particles was supplied
        volume = sample['vf'][0]*rve_dims[0]*rve_dims[1]*rve_dims[2]/sample['n'][0]
        # Area of each particle (all the same)
        r = np.cbrt(volume/(4/3*np.pi))
    return r


def canonicalParametersEllipsoid(sample, rve_dims):
    """Convert parameters in *sample* to canonical params characterizing an Ellipsoid."""
    if 'axis_1' in sample and 'axis_2' in sample and 'axis_3':
    # All axis were supplied
        axis_1 = sample['axis_1']
        axis_2 = sample['axis_2']
        axis_3 = sample['axis_3']
    if 'angle' in sample:
        angle = sample['angle']
    if 'euler_angle_x' in sample and 'euler_angle_y' in sample \
        and 'euler_angle_z' in sample:
    # Euler angles
        euler_angle_x = sample['euler_angle_x']
        euler_angle_y = sample['euler_angle_y']
        euler_angle_z = sample['euler_angle_z']

    return [axis_1, axis_2, axis_3, euler_angle_x, euler_angle_y, euler_angle_z, angle]


def createResultsDirectory(particles, dp_dir):
    """Create the results directory."""
    Particle.file_name = (particles[0].__class__.__name__ + "_" + str(Particle.number)
                          + "_" + str(Particle.volume)[0:3])
    # Defining the file name associated with this sampling
    results_folder = os.path.join(dp_dir,  Particle.file_name)
    # Creating a tentative path for the results folder
    if os.path.exists(results_folder):
    # If the folder already exists
        results_folder_old = results_folder
        # Saving the original name of the results folder
        i = 0
        while os.path.exists(results_folder):
        # While the folder names already exists
            i += 1
            results_folder = results_folder_old + "_" + str(i)
            # Creating a new folder name appending an integer to the name of the original
            # folder
        os.makedirs(results_folder)
        # Creating the directory
        if os.path.exists("input_data\\info_micro.p"):
            shutil.copy("input_data\\info_micro.p",
                       os.path.join(results_folder, "info_micro.p"))
    else:
        os.makedirs(results_folder)
        # Creating the directory
        if os.path.exists("input_data\\info_micro.p"):
            shutil.copy("input_data\\info_micro.p",
                       os.path.join(results_folder, "info_micro.p"))
    # FIXME: Only the first sample keeps the info folder.
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    # Saving the file path in the Particle class


def particleGeneration(descriptors, phase_types, rve_dims, problem_type, dp_dir,
                       type_init_conf, save_history=False):
    """
    Generate all the particles from the geometrical descriptors.

    Parameters
    ----------
    descriptors: dictionary
        Dictionary containing the particle descriptors

    phase_types: dictionary(str:int)
        Dictionary containing the phase type of each phase.
            1: Matrix
            2: Disk (2D)
            3: Ellipse (2D)
            4: Sphere (3D)
            5: Ellipsoid (3D)

    rve_dims: list(float)
        Length of the RVE sides in each direction.

    problem_type: integer
        Type of problem.
            1: 2D problem (plain strain)
            2: 2D problem (plain stress)
            3: 2D problem (axisymmetric)
            4: 3D problem

    dp_dir: string
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored

    type_init_conf: {'random', 'grid'}
        Type of initial configuration for the particle centers.

    save_history: bool, optional
        Save the motion of the particles for later analysis.
    """
    Particle.box = rve_dims
    # Setting the size of the simulation box. It may be changed later if the phases are
    # made from cylindrical fibers, as their simulated in a plane despite being 3D
    Particle.volume_RVE = np.prod(rve_dims)
    # Volume of the RVE
    Particle.list_phases = [i_phase for i_phase in descriptors]
    # List containing the phases
    Particle.volume_phase = {phase: 0 for phase in Particle.list_phases}
    # Initializing the dictionary containing the volume of each phase
    try:
        if list(phase_types.values()).count(1) == 0:
        # No matrix phase was specified
            raise errors.NoMatrixPhase()
        elif list(phase_types.values()).count(1) > 1:
        # Too many phases were specified as the matrix phase
            raise errors.TooManyMatrixPhases()
    except (errors.NoMatrixPhase, errors.TooManyMatrixPhases) as error:
        error.message()
        quit()
    particles = []
    # Initializing the list containing the particles
    # if problem_type == 1:
    #     # 2D problem (plain strain)
    #     dim = 2
    #     # (FIX)
    #     # Setting the dimension
    for i_phase in descriptors:
    # Running through all the phases listed in the dictionary
        try:
            if phase_types[i_phase] == 1:
            # This phase is the matrix
                Particle.matrix_phase = i_phase
                # No particles are generated
            elif phase_types[i_phase] == 2:
            # This phase is made up by disks
                print(rve_dims)
                if len(rve_dims) != 2:
                # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase('Disks', 2, 3, i_phase)
                particles = particles + \
                    generateDisks(i_phase, rve_dims, descriptors[i_phase])
                # Generating the number of disks requested and appending them to the list of
                # particles
            elif phase_types[i_phase] == 3:
            # This phase is made up by ellipses
                if len(rve_dims) != 2:
                # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase('Ellipses', 2, 3, i_phase)
                particles = (particles
                             + generateEllipses(i_phase, rve_dims, descriptors[i_phase]))
                # Generating the number of ellipses requested and appending them to the list
                # of particles
            elif phase_types[i_phase] == 4:
            # This phase is made up by spheres
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase('Spheres', 3, 2, i_phase)
                particles = particles + \
                    generateSpheres(i_phase, rve_dims, descriptors[i_phase])
                # Generating the number of spheres requested and appending them to the list
                # of  particles
            elif phase_types[i_phase] == 5:
            # This phase is made up by ellipsoids
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase('Ellipsoids', 3, 2, i_phase)
                particles = (particles
                             + generateEllipsoids(i_phase, rve_dims, descriptors[i_phase]))
                # Generating the number of ellipsoids requested and appending them to the
                # list of particles
            elif phase_types[i_phase] == 6:
            # This phase is made up by cylindrical fibers
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        'Cylindrical Fibers', 3, 2, i_phase)
                if any([phase_type != 1 and phase_type != 6 for phase_type in
                        list(phase_types.values())]):
                    raise errors.OnlyCylindricalFibers()
                particles = (particles
                             + generateCylindricalFibers(i_phase, rve_dims,
                                                         descriptors[i_phase]))
                # Generating the number of cylindrical fibers requested and appending them
                # to the list of particles
            else:
                raise errors.UnsupportedPhaseType(phase_types[i_phase], i_phase)
        except (errors.IncompatibleDimensionsRVEphase,
                errors.OnlyCylindricalFibers) as error:
            error.message()
            quit()
    
    generateInitialConfiguration(particles, type_init_conf, save_history=True)
    # FIXME: save history as option

    createResultsDirectory(particles, dp_dir)

    return particles

# ==========================================================================================


def readDescriptors():
    """
    Load the descriptors and options to generate the microstructure.

    This function loads the descriptors and returns the microstructure descriptors, the
    phase types and options.

    Returns
    -------
    dp_dir: str
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored.

    descriptors: dict

    phase_types: dict

    options: dict

    n_dp_samples: int

    rve_dims: list

    problem_type: int

    discret_spec_array: dict
    """
    info_dict = pickle.load(open('input_data\\info_micro.p', 'rb'))
    # Loading the dictionary containing the information about the microstructure and its
    # generation
    # dp_dir: string
    #     Directory where the microstructure spatial discretization file(s) associated
    #     with the given design point are to be stored
    dp_dir = info_dict.get('dp_dir')
    # mic_gen_parameters: array
    #     An array which contains all the required parameters (or options)
    #     for the selected program to generate the microstructure(s) and
    #     and associated discretization file(s) of a given design point
    #     (to be discussed...)
    options = info_dict.get('mic_gen_parameters')
    # # Initializing the dictionary containing the options
    # #                                                                    Stopping criteria
    # # --------------------------------------------------------------------------------------
    # options['max_residue_per_particle'] = 0
    # options['max_step'] = 1000
    # # Maximum number of steps
    # options['max_steps_to_relax'] = 250
    # # Maximum number of steps after the legal configuration has been found after which the
    # # configuration is accepted
    # #                                                                   Integration scheme
    # # --------------------------------------------------------------------------------------
    # options['integration_scheme']='Newmark'
    # # Integration scheme to be used:
    # # 'Newmark'  - Newmark beta method
    # options['damping_constant'] = 0
    # # Damping constant (only for Newmark)
    # options['dt'] = 0.005
    # # Time step
    # #
    # #                                        Speed up scheme for the computation of forces
    # # --------------------------------------------------------------------------------------
    # options['speed_up_scheme']='Naive'
    # # Speed up scheme
    # # 'Naive' - the forces are computed between every pair of particles (O(N**2))
    # # 'Cell' - the forces are computed making use of a cell list, such that each particle
    # # only interacts with the particles in its cell or the nearest neighboring cells (O(N))
    # # 'Verlet' - the forces are computed using a Verlet list for each particle, that in
    # # turn in computed using a cell list method
    # options['verlet_factor'] = 1.5
    # # The Verlet list is computing making use of neighboorhood around the particle, whose
    # # shape is the same, but dilated by the 'verlet_factor'
    # #
    # #                                                                Computation of forces
    # # --------------------------------------------------------------------------------------
    # options['initial_global_force_factor'] = 200 #4
    # 
    # options['global_force_factor_multiplier'] = 1.8
    # #                                                                           Thermostat
    # # --------------------------------------------------------------------------------------
    # options['thermostat']='isokinetic'
    # # problem_type: integer
    # #     Problem type    | 1. 2D problem (plain strain)
    # #                     | 2. 2D problem (plain stress)
    # #                     | 3. 2D problem (axisymmetric)
    # #                     | 4. 3D problem
    problem_type = info_dict.get('problem_type')
    # n_dp_samples: integer
    #     Number of microstructures (samples) to be generated, associated to
    #     the given design point
    n_dp_samples = info_dict.get('n_dp_samples', 1)
    try:
        if not isinstance(n_dp_samples, int) or n_dp_samples < 1:
        # The number of samples must be an integer larger or equal to 1
            raise errors.NumberSamples(n_dp_samples)
    except errors.NumberSamples() as error:
        error.message()
        quit()
    # mic_gen_descriptors_array: dictionary
    descriptors = info_dict.get('mic_gen_descriptors', {})

    # phase_types: dictionary
    phase_types = info_dict.get('phase_types', {})
    try:
        if set(phase_types.keys()) != set(descriptors.keys()):
        # There are phases which not have descriptors or a phase type
            for phase in descriptors:
                if phase not in phase_types:
                # If there is a phase that has descriptors but no phase type
                    raise errors.PhaseDescriptorsMatch(phase)
    except errors.PhaseDescriptorsMatch as error:
        error.message()
        quit()
    try:
        for phase in phase_types:
            if not RepresentsInt(phase) or not isinstance(phase, str):
                raise errors.UnexpectedValue(phase, 'key of phase_types',
                                             'string containing an integer')
    except errors.UnexpectedValue as error:
        error.message()
        quit()

    # discret_file_ext: list
    #     List which contains the required spatial discretization file(s), stored as
    #                     array = [ < discret_type > < discret_type >  ... ]

    # discret_spec_array: dictionary
    #     Dictionary which contains the required parameters to generate
    #     each type of specified discretization file, stored as
    #                            dictionary['disc_ext']['parameter'] = [ ... ]

    discret_file_ext = info_dict.get('discret_file_ext', {})
    # Saving the list containing the meshes required
    discret_spec_array = info_dict.get('discret_spec_array', {})
    # Dictionary containing arrays with the specifications for each meash
    for ext in discret_spec_array:
    # Completing the list of extensions from the specification
        if ext not in discret_file_ext:
            discret_file_ext.append(ext)
    try:
        if len(discret_file_ext) == 0:
        # No mesh was specified
            raise errors.NoMesh()
    except errors.NoMesh as error:
        error.message()
        quit()
    try:
        for ext in discret_file_ext:
        # Check if all the required outputs have a description
            if ext not in discret_spec_array:
                raise errors.MissingInfoExtension(ext)
            for spec in discret_spec_array[ext]:
            # Check if the required extensions specify the bare minimum
                checkMeshSpecs(ext, discret_spec_array[ext])
    except errors.MissingInfoExtension as error:
        error.message()
        quit()

    rve_dims_spec = []
    for ext in discret_file_ext:
    # Running through the specified meshes
        rve_dims_spec.append(tuple(discret_spec_array[ext]['rve_dims']))
        # Collecting the RVE dimensions specified
    rve_dims_spec = set(rve_dims_spec)
    # Obtaining the unique RVE size specifications
    if len(rve_dims_spec) > 1:
    # There are multiple RVE size specifications
        print('Different RVE sizes in the mesh specifications.')
        rve_dims = np.array(list(rve_dims_spec)[0])
        # Keeping the first
    else:
        rve_dims = np.array(list(rve_dims_spec)[0])

    return [dp_dir, descriptors, phase_types, options, n_dp_samples, rve_dims, problem_type,
            discret_spec_array, discret_file_ext]


def computeRelativeEnergy(particles):
    N = Particle.number
    norm_force_vec = np.array([np.linalg.norm(particles[i].force)
                              for i in range(N)], dtype='float')
    # Obtaining a list with the norms of the vector forces
    relative_energy =  norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    print('new', relative_energy)
    Particle.relative_energy_history.append(relative_energy)
    Particle.total_overlap_history.append(Particle.total_overlap)

    return relative_energy


def computeKineticEnergy(particles):
    # N = Particle.number
    # norm_velocity_vec = np.array(
    #     [np.linalg.norm(particles[i].velocity_center) for i in range(N)], dtype='float')
    # # Obtaining a list with the norms of the vector forces
    # kin_energy = norm_velocity_vec.dot(norm_velocity_vec)
    kin_energy = np.sum([i_particle.volume()*np.sum(i_particle.velocity_center**2) for i_particle in particles])
    print('kinetic', kin_energy)
    Particle.kinetic_energy_history.append(kin_energy)

    return kin_energy


def run(particles, max_residue_per_particle, max_step, options):
    """
    Run the Molecular Dynamics simulation for the system of particles given.

    This is the main function of the Molecular Dynamics simulation. It consists of the
    initialization of the sytem, and the loop that contains the dynamics of the system:
    computation of the forces and integration of the equations of motion.

    Parameters
    ----------
    particles : list(`.Particle`)
        Array containing the Particle objects to be placed inside the RVE

    max_residue_per_particle: float
        Maximum allowable overlap residue between particles

    max_step: int
        Maxium number of time steps

    options: dictionary
        Other options. See notes.

    Options(dict)
    ----------------
    dt: float
        Time step

    verlet_factor: float
        Factor defining the Verlet neighboorhood

    initial_global_force_factor: float
        Factor multiplied at the begin of the simulation by the forces for dynamical
        adjustments

    max_steps_to_relax: int
        Number of steps the configuration has to be below the maximum overlap residual
        area before the configuration is accepted

    thermostat: {'isokinetic'}, optional
        Thermostat to be used

    speed_up_scheme: {'Naive', 'Cell', 'Verlet'}, optional
        Speed up scheme used in the force computation
            "Naive": the forces are computed between every pair of particles (O(N**2))
            "Cell": the forces are computed making use of a cell list, such that each
                particle only interacts with the particles in its cell or the nearest
                neighboring cells (O(N))
            "Verlet": the forces are computed using a Verlet list for each particle, that in
                turn in computed using a cell list method
    """
    N = Particle.number
    # Saving the number of particles
    box = Particle.box
    # Saving the array containing the size of the box
    dim = particles[0].dim
    # Saving the array containing the dimension of the problem
    speed_up_scheme = options.get('speed_up_scheme', 'Cell')
    if speed_up_scheme == 'Cell':
        # Only a cell list scheme will be used
        max_radius = np.max(np.array([particles[i].radius for i in range(N)]))
        # Saving the maximum radius of the circunscribing disk/sphere
        Particle.n_cell_dim = (
            [np.int(np.round(box[i_dim]/(2*max_radius))) for i_dim in range(dim)])
        # Obtaining a list containing the number of cells in each direction
        n_cells = np.prod(Particle.n_cell_dim)
        # Obtaining the total number of cells
        Particle.cell_list = [[] for i in range(n_cells)]
        # Initializing the cell list
        Particle.cell_side_length = (
            [box[i_dim]/Particle.n_cell_dim[i_dim] for i_dim in range(dim)])
        # Obtaining a list containing the dimensions of the cell in each direction
        print(Particle.n_cell_dim)
        print(Particle.cell_side_length)
    elif speed_up_scheme == 'Verlet':
        # A Verlet list combined with a cell list scheme will be used
        Particle.verlet_factor = options['verlet_factor']
        # Saving the Verlet radius to compute the Verlet list
        Particle.new_verlet_list = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        max_radius = (
                     np.max(np.array(
                        [particles[i].radius for i in range(Particle.number)]))
                     * Particle.verlet_factor
                     )
        # Saving the maximum radius of the circunscribing disk/sphere accounting for the
        # Verlet factor
        Particle.n_cell_dim = (
            [np.int(np.round(box[i_dim]/(2*max_radius))) for i_dim in range(dim)])
        # Obtaining a list containing the number of cells in each direction
        n_cells = np.prod(Particle.n_cell_dim)
        # Obtaining the total number of cells
        Particle.cell_list = [[] for i in range(n_cells)]
        # Initializing the cell list
        Particle.cell_side_length = (
            [box[i_dim]/Particle.n_cell_dim[i_dim] for i_dim in range(dim)])
        # Obtaining a list containing the dimensions of the cell in each direction
    else:
        # A naive approach will be used
        pass
    n_steps_relax = 0
    # Initializing the number of steps that a microstructure was complying with the
    # maximum overlap residue
    max_residue = max_residue_per_particle*N
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0
    initial_global_force_factor = 1 # options.get('initial_global_force_factor', 1)
    Particle.global_force_factor = initial_global_force_factor
    # Initializing the global force factor
    computeForces(particles, speed_up_scheme)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    relative_energy = computeRelativeEnergy(particles)
    total_overlap_vec = [Particle.total_overlap]
    Particle.max_residue = max_residue
    # Computing the relative energy
    kin_energy = computeKineticEnergy(particles)
    # Computing the kinetic energy
    # relative_energy_old = relative_energy
    # Saving the current relative energy
    max_steps_to_relax = options.get('max_steps_to_relax', 100)
    dt = options.get('dt', 0.005)
    thermostat = options.get('thermostat', 'isokinetic')
    if thermostat == 'isokinetic':
    # The thermostat used is the isokinetic scheme
        jump = 30 #20 #np.max([np.int(np.floor(1500/N)), 2])
        last_alt = 150
        T_ref = 1e-6*1/3*10
    # Setting the options
    while (step < max_step) and n_steps_relax < max_steps_to_relax:
        # Run the simulation while the number of steps the overlap has been smaller than the
        # allowed maximum residue is larger than options['max_steps_to_relax'], so that the
        # particles have time to get away from each other.
        if options.get('save_history'):
            integrate(particles, dt, speed_up_scheme, save_history=True)
        else:
            integrate(particles, dt, speed_up_scheme)
        # Integrating the equations of motion
        step += 1
        # # Moving to the next time step
        computeForces(particles, speed_up_scheme)
        # Computing the forces on all particles
        relative_energy = computeRelativeEnergy(particles)
        # Computing the relative energy
        total_overlap_vec.append(Particle.total_overlap)
        kin_energy = computeKineticEnergy(particles)
        # Computing the kinetic energy
        if thermostat == 'isokinetic':
            # The thermostat used is the isokinetic scheme
            if step > last_alt:
                if np.max(total_overlap_vec[-jump:-1]) != 0:
                    if np.min(total_overlap_vec[-jump:-1])/np.max(total_overlap_vec[-jump:-1]) >= 0.1 and Particle.total_overlap > max_residue:
                        T_ref *= 1/1.5
                        last_alt = step + jump
                        Particle.temp_change_steps.append(step)
            if kin_energy > 1e-10:
            # Compute the rescaling factor only if the kinetic energy is nonzero
                lambda_vel = np.sqrt(3*N*T_ref/kin_energy)
                # Rescalling factor (why? 250 -  equipartition theorem)
            else:
            # If the kinetic energy is zero
                lambda_vel = 0
            for i_particle in range(N):
                # Running through all the particles
                particles[i_particle].velocity_center *= lambda_vel
                # Rescalling the velocities
            kin_energy_ref = computeKineticEnergy(particles)
            if relative_energy/Particle.total_overlap < 1e-3:
                print("diverged")
                break
        else:
            # There is no thermostat
            pass
        if Particle.total_overlap <= max_residue: # and all([len(Particle.cell_list[i]) < 2 for i in range(27)]): # and all(len(particles[i].verlet_list)<4 for i in range(len(particles))):
            # If the configuration has an overlap area smaller than the tolerance
            n_steps_relax += 1
            print('n_steps_relax', n_steps_relax)
            # print('yes',n_steps_relax)
        else:
            n_steps_relax = 0
            # Restarting the count

        # Particle.global_force_factor *= 10e-3/relative_energy #options['global_force_factor_multiplier']

        # if relative_energy/relative_energy_old < 0.5:
        # # If the relative energy has decreased by a factor of two in this iteraton
        #     relative_energy_old = relative_energy
        #     # Saving the value of the previous relative energy
        #     Particle.global_force_factor *= \
        #         kwargs.get('global_force_factor_multiplier', 1)
        #     # Increase the global factor multiplying the forces
        # elif relative_energy/relative_energy_old > 2:
        # # If the relative energy has increased by a factor of two in this iteraton
        #     relative_energy_old = relative_energy
        #     # Saving the value of the previous relative energy
        #     Particle.global_force_factor *= \
        #         1/kwargs.get('global_force_factor_multiplier', 1)
        #     # Increase the global factor multiplying the forces

        print(step)

    # Integrating Newton's equations of motion


def dilateParticles(particles, min_distance):
    """ Dilate all the particles so that a minimum distance is ensured after contraction."""
    for i_particle in particles:
    # Running through all the particles
        i_particle.dilate(min_distance)
        # Dilate i_particle


def contractParticles(particles, min_distance):
    """Contract all the particles so that a minimum distance is ensured."""
    for i_particle in particles:
    # Running through all the particles
        i_particle.contract(min_distance)
        # contract i_particle


def main():
    """Run the microstructure generation program."""
    start = time.time()
    # Counting time
    screen_path = open("test.txt", 'w')
    # sys.stdout = f
    [dp_dir, descriptors, phase_types, options, n_samples, rve_dims, problem_type,
        discret_spec_array, discret_file_ext] = readDescriptors()
    # Reading the descriptors and options for the microstructure generation
    if options.get('remesh'):
    # It is a remesh action
        current_RVE = pickle.load(open(options['dir_previous_mic'], 'rb'))
        # No need to generate a new microstructure. Using a previous microstructure.
        particles, rve_dims = current_RVE.useThisRVE()
        # Reconstructing the relevant Particle attributes that could not be pickled
        end = time.time()
        for disc_ext in discret_file_ext:
        # For each file extension asked
            generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
            # Generate corresponding mesh
    else:
    # Generating samples of microstructures and meshing
        for i_sample in range(n_samples):
            # Producing the number of samples required
            save_history = options.get('save_history', False)
            # Saving if the history of the particles' motion needs to be saved
            type_init_conf = options.get('type_initial_configuration', 'random')
            # Saving the type of initial configuration specified, with 'random' as the
            # default value
            particles = particleGeneration(descriptors, phase_types, rve_dims, problem_type,
                                           dp_dir, type_init_conf=type_init_conf,
                                           save_history=save_history)
            # Generating the list of particles from the geometrical descriptors
            plotParticles(particles, Particle.file_path + "_initial_conf",
                          save=options.get('save_plot', True),
                          show=options.get('save_plot', True))
            # Ploting initial configuration
            for ext in discret_file_ext:
                if 'min_distance' in discret_spec_array[ext]:
                # If any of the extensions required specifies a minimum distance
                    dilateParticles(particles, discret_spec_array[ext]['min_distance'])
                    # Dilate all particles
                    break
            run(particles, options['max_residue_per_particle'], options['max_step'],
                options)
            # Running the molecular dynamics simulation
            for ext in discret_file_ext:
                if 'min_distance' in discret_spec_array[ext]:
                # If any of the extensions required specifies a minimum distance
                    contractParticles(particles, discret_spec_array[ext]['min_distance'])
                    # Contract all particles
                    break
            end = time.time()
            plotParticles(particles, Particle.file_path + "_final_config",
                          save=options.get('save_plot', True),
                          show=options.get('save_plot',  True))
            # Ploting final configuration
            # print('cell_list', Particle.cell_list)
            # print('verlet_list', [particles[i].verlet_list for i in range(len(particles))])
            current_RVE = RVE(particles, rve_dims)
            # Saving the RVE properties in an RVE object
            pickle.dump(current_RVE, open(Particle.file_path + ".p", "wb"))
            # Saving the configuration for later use
            for disc_ext in discret_file_ext:
            # For each file extension asked
                generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
                # Generate corresponding mesh
            if save_history:
                plotPaths(particles, particles[0].dim, Particle.file_path)
            Particle.resetRVE()
            # Clearing the properties of the RVE
    print(end - start)

    screen_path.close()
    sys.stdout = sys.__stdout__

    with open('test.txt') as file:
        data = file.read()
        print(data)
    os.replace("test.txt", Particle.file_path + ".txt")


if __name__ == '__main__':

    # ======================================================================================
    # dp_dir: string
    #     Directory where the microstructure spatial discretization file(s) associated
    #     with the given design point are to be stored
    dp_dir = ("C: \\Users\\José\\Notebooks\\Database"
              + "\\Universidade\\Dissertacao\\programa\\results")
    # ======================================================================================
    # mic_gen_program: integer
    #     Integer variable (read from the user input data file) which specifies an
    #     available program to generate the microstructure(s) and associated
    #     discretization file(s) of a given design point
    mic_gen_program = 1
    # ======================================================================================
    # mic_gen_parameters: array
    #     An array which contains all the required parameters (or options)
    #     for the selected program to generate the microstructure(s) and
    #     and associated discretization file(s) of a given design point
    #     (to be discussed...)
    mic_gen_parameters = {}
    # Initializing the dictionary containing the options
    #                                                                    Stopping criteria
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['max_residue_per_particle'] = 0
    mic_gen_parameters['max_step'] = 1000
    # Maximum number of steps
    mic_gen_parameters['max_steps_to_relax'] = 250
    # Maximum number of steps after the legal configuration has been found after which the
    # configuration is accepted
    #                                                                   Integration scheme
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['integration_scheme'] = 'Newmark'
    # Integration scheme to be used:
    # 'Newmark'  - Newmark beta method
    mic_gen_parameters['damping_constant'] = 0
    # Damping constant (only for Newmark)
    mic_gen_parameters['dt'] = 0.005
    # Time step
    #
    #                                        Speed up scheme for the computation of forces
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['speed_up_scheme'] = 'Naive'
    # Speed up scheme
    # 'Naive' - the forces are computed between every pair of particles (O(N**2))
    # 'Cell' - the forces are computed making use of a cell list, such that each particle
    # only interacts with the particles in its cell or the nearest neighboring cells (O(N))
    # 'Verlet' - the forces are computed using a Verlet list for each particle, that in
    # turn in computed using a cell list method
    mic_gen_parameters['verlet_factor'] = 1.5
    # The Verlet list is computing making use of neighboorhood around the particle, whose
    # shape is the same, but dilated by the 'verlet_factor'
    #
    #                                                                Computation of forces
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['initial_global_force_factor'] = 200  # 4

    mic_gen_parameters['global_force_factor_multiplier'] = 1.8
    #                                                                           Thermostat
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['thermostat'] = 'isokinetic'
    # problem_type: integer
    #     Problem type    | 1. 2D problem (plain strain)
    #                     | 2. 2D problem (plain stress)
    #                     | 3. 2D problem (axisymmetric)
    #                     | 4. 3D problem
    problem_type = 1
    # n_dp_samples: integer
    #     Number of microstructures (samples) to be generated, associated to
    #     the given design point
    n_dp_samples = 1
    # mic_gen_descriptors_array: dictionary
    #     A dictionary which contains all the microstructure
    #     descriptor-related information required to generate the
    #     given design point microstructure(s) automatically,
    #     stored as
    #                                     Microstructure Descriptors
    #                               _                                    _
    #     dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
    #                              |_  < value >     < value >      ...  _|
    #
    mic_gen_descriptors_array = {}

    mic_gen_descriptors_array['4'] = np.array([['rve_dims'], [[1.0, 1.0, 1.0]]])
    mic_gen_descriptors_array['2'] = np.array([['r', 'n'], [0.1, 5]], dtype=object)

    # descriptors['2'] = {'distribution':'uniform','r_low':0.02,'r_high':0.04, 'n':190}
    # descriptors['2'] = {'major_axis':0.20,'minor_axis':0.1,'angle':0,'n':10}
    # phase_types: dictionary
    #     Dictionary which contains each material phase type, stored as
    #                    dictionary['phase_id'] = phase_types
    ## phase_types = info_micro['phase_types']
    # Types of particles
    # 1 - Matrix
    # 2 - Circular particle (disk)
    # 3 - Elliptical particle
    phase_types = {}
    phase_types['4'] = 1  # Matrix
    phase_types['2'] = 4  # Elliptical particle
    # discret_file_ext: list
    #     List which contains the required spatial discretization file(s), stored as
    #                     array = [ < discret_type > < discret_type >  ... ]

    # discret_spec_array: dictionary
    #     Dictionary which contains the required parameters to generate
    #     each type of specified discretization file, stored as
    #                            dictionary['disc_ext']['parameter'] = [ ... ]

    discret_file_ext = []

    discret_spec_array = {}
    discret_spec_array['rgmsh'] = {}
    discret_spec_array['rgmsh']['rve_dims'] = np.array([1.0, 1.0, 1.0])
    discret_spec_array['rgmsh']['n_voxels_dims'] = np.array([50, 50, 50])
    discret_spec_array['femsh'] = {}
    discret_spec_array['femsh']['rve_dims'] = np.array([1.0, 1.0, 1.0])
    discret_spec_array['femsh']['mesh_size'] = 0.1

    mic_gen_descriptors_dict = {}
    # Initializing the dictionary containing the microstructure descriptors
    print(mic_gen_descriptors_array)
    for i_phase in mic_gen_descriptors_array:
        # Running through all the phases
        mic_gen_descriptors_dict[i_phase] = (
            {mic_gen_descriptors_array[i_phase][0, i]:
                mic_gen_descriptors_array[i_phase][1, i]
                for i in range(len(mic_gen_descriptors_array[i_phase][0]))})

    info_dict = {
        "dp_dir": dp_dir,
        "mic_gen_parameters": mic_gen_parameters,
        "problem_type": problem_type,
        "n_dp_samples": n_dp_samples,
        "mic_gen_descriptors_array": mic_gen_descriptors_array,
        "phase_types": phase_types,
        "discret_file_ext": discret_file_ext,
        "discret_spec_array": discret_spec_array
        }
    # Building a dictionary to be pickled with all the information coming from the
    # interfacing program
    pickle.dump(info_dict, open("src\\info_micro.p", "wb"))
    # Dumping the info_dict dictionary into info_micro.p to be loaded in the program
    # that generates microstructures
    main()
    # Executing the script for microstructure generation
    os.remove("src\\info_micro.p")
    # Deleting the file containing the input data
