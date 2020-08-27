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
from integration_methods import Newmark, VerletSync
# Importing an integration method for the equation of motion
from particle_classes import Disk, Particle, Ellipse, Sphere, Ellipsoid, CylindricalFiber, RVE, Phase, GeometricalParameter, PhaseDescriptor
# Importing the particle class
from meshing_interface import generateMesh, checkMeshSpecs
# Importing meshing interfaces
import error_classes as errors
# Importing the error clases
import printing as print_funcs

from voronoi_analysis import doVoronoiAnalysis

from motion_analysis import doMotionAnalysis

import os
import shutil

import sys


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
    Compute the global cell position of the neighboor cell.

    Parameters
    ----------
    pos_current_cell: integer
        Global position of the current cell

    local_pos_neighboor_cell: integer
        Local position of the neighboor cell

    dim: integer
        Dimension of the problem

    n_cells: list
        Number of cells in each direction (0:x; 1:y; 2:z)

    Returns
    -------
    pos_neighboor_cell: integer
        Global position of the neighboor cell
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
        if pos_current_cell - n_cells[1]*n_cells[0]*(pos_current_cell//(n_cells[1]*n_cells[0])) < n_cells[0] and local_row_pos_neigh == -1:
        # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        elif pos_current_cell - n_cells[1]*n_cells[0]*(pos_current_cell//(n_cells[1]*n_cells[0])) >= n_cells[0]*(n_cells[1] - 1) and local_row_pos_neigh == 1:
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
    Compute a new cell list for particles.
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
            # Initializing the list containing the position of the particle in the grid,
            # assuming:
            # 2D: the cells are numbered from left to right and from top to bottom
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
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 1
                            particles[j_particle].force = particles[j_particle].force \
                                - force_i_j
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 2
            if dim == 3:
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
                            force_i_j = computeForceij(
                                particles[i_particle], particles[j_particle])
                            # Computing the force on particle i due to particle j
                            particles[i_particle].force = particles[i_particle].force \
                                + force_i_j
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 1
                            particles[j_particle].force = particles[j_particle].force \
                                - force_i_j
                            # Adding the force due to the interaction between particle 1 and
                            # 2 to the total force acting on particle 2
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
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 1
                    particles[j_particle].force = particles[j_particle].force - force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 2

# ==========================================================================================
def computeForceij(particle_i, particle_j):
    '''
    Compute the force on particle_i due to particle_j
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


def putSystemAtRest(particles):
    """Put the system as a whole at rest."""
    total_linear_momentum = np.sum(
        [particle.volume()*particle.velocity_center for particle in particles], axis=0)
    # Computing total linear momemtum of the system
    for i_particle in particles:
    # Running through all the particles
        i_particle.setVelocityCenter(i_particle.velocity_center - total_linear_momentum)
        # Removing the linear momentum of the system as a whole putting at rest


def integrate(particles, dt, speed_up_scheme, integration_scheme='Verlet', **kwargs):
    """Integrate the equations of motion."""
    dim = particles[0].dim
    # Dimension of the problem
    N = len(particles)
    # Number of particles
    box = Particle.box
    # Saving the size of the RVE
    for i_particle in range(N):
    # Running through all the particles
        if integration_scheme == 'Newmark':
        # The integration scheme chosen was Newmark
            c = kwargs.get('damping_constant', 0)
            [new_position, new_velocity, new_accelaration] = \
                Newmark(
                    particles[i_particle].position_center,
                    particles[i_particle].velocity_center,
                    np.array([particles[i_particle].force], dtype='float').T,
                    particles[i_particle].volume()*np.eye(particles[i_particle].dim, dtype='float'), #10e-6*np.eye(2,dtype='float'),#
                    c*np.eye(particles[i_particle].dim, dtype='float'),
                    np.zeros((particles[i_particle].dim,particles[i_particle].dim), dtype='float'),
                    dt,
                    1,
                    dim)
            # Obtaining the new position and velocity of particle i
        elif integration_scheme == 'Verlet':
        # The integration scheme chosen was Verlet
            [new_position, new_velocity] = VerletSync(
                particles[i_particle].position_center,
                particles[i_particle].velocity_center,
                np.array([particles[i_particle].force], dtype='float').T,
                particles[i_particle].volume(),
                dt,
                1,
                dim)
        if speed_up_scheme == 'Verlet':
            particles[i_particle].displacement_last_verlet += \
                particles[i_particle].position_center - new_position[:, 0]
            # Computing the displacement of the center of the particle
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
    # putSystemAtRest(particles)
    # Putting the systemas a whole at rest

# ==========================================================================================


def generateDisks(phase, rve_dims, descriptors):
    """
    Generate disks of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the disks will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    disks = []
    # Initializing the list containing the disks
    used_parameters = {parameter for parameter in Disk.possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    if any([used_parameters == acceptable_description for
            acceptable_description in Disk.acceptable_descriptions]):
    # Checking acceptable sets of parameters
        acceptable_description = True
    else:
        acceptable_description = False
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase.type,
                                                Disk.acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of disks was specified
        phase.specNumber(descriptors['n'])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Disk.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            disks.append(Disk(phase, r[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors['vf']
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter, Disk.possible_parameters[i_parameter], descriptors, phase, rve_dims)
            r = canonicalParametersDisk(current_sample, rve_dims)
            disks.append(Disk(phase, r[0]))
            vf_real += disks[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        phase.spec_volume_fraction = descriptors['vf']
        phase.spec_number = descriptors['n']
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Disk.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        # Obtaining the radius corresponding to the specified volume fraction and number of
        # particles
        for i in range(descriptors['n']):
            disks.append(Disk(phase, r))

    return disks


def generateSpheres(phase, rve_dims, descriptors):
    """Generate spheres of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    spheres = []
    # Initializing the list containing the spheres
    used_parameters = {parameter for parameter in Sphere.possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    if any([used_parameters == acceptable_description for
            acceptable_description in Sphere.acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                Sphere.acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of disks was specified
        phase.specNumber(descriptors['n'])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Sphere.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors['n']):
            spheres.append(Sphere(phase, r[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors['vf']
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Sphere.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims)
            r = canonicalParametersSphere(current_sample, rve_dims)
            spheres.append(Sphere(phase, r))
            vf_real += spheres[-1].volume()/(rve_dims[0]*rve_dims[1]*rve_dims[2])
    elif 'vf' in descriptors and 'n' in descriptors:
        phase.spec_volume_fraction = descriptors['vf']
        phase.spec_number = descriptors['n']
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Sphere.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors['n']):
            spheres.append(Sphere(phase, r))

    return spheres


def generateEllipses(phase, rve_dims, descriptors):
    """Generate ellipses of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    ellipses = []
    # Initializing the list containing the disks
    used_parameters = {parameter for parameter in Ellipse.possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    if any([used_parameters == acceptable_description for
            acceptable_description in Ellipse.acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                Ellipse.acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of ellipses was specified
        phase.specNumber(descriptors['n'])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipse.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples,
                                                                     rve_dims)
        for i in range(descriptors['n']):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors['vf']
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipse.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims)
            [major_axis, minor_axis, angle] = canonicalParametersEllipse(current_sample,
                                                                         rve_dims)
            ellipses.append(Ellipse(phase, major_axis, minor_axis, angle))
            vf_real += ellipses[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        phase.spec_volume_fraction = descriptors['vf']
        phase.spec_number = descriptors['n']
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipse.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples,
                                                                     rve_dims)
        for i in range(descriptors['n']):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))

    return ellipses


def generateEllipsoids(phase, rve_dims, descriptors):
    """Generate ellipsoids belonging to *phase* characterized by *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the spheres will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    ellipsoids = []
    # Initializing the list containing the disks
    used_parameters = {parameter for parameter in Ellipsoid.possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    if any([used_parameters == acceptable_description for
            acceptable_description in Ellipsoid.acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                Ellipsoid.acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of ellipsoids was specified
        phase.specNumber(descriptors['n'])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        [axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle] = \
            canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors['n']):
            ellipsoids.append(Ellipsoid(
                phase, axis_1[i], axis_2[i], axis_3[i], rot_axis_comp_x[i], rot_axis_comp_y[i],
                rot_axis_comp_z[i], angle[i]))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors['vf']
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipsoid.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims)
            [axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle] = \
                canonicalParametersEllipsoid(current_sample, rve_dims)
            ellipsoids.append(Ellipsoid(phase, axis_1[0], axis_2[0], axis_3[0],
                                        rot_axis_comp_x[0], rot_axis_comp_y[0],
                                        rot_axis_comp_z[0], angle[0]))
            vf_real += ellipsoids[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        phase.spec_volume_fraction = descriptors['vf']
        phase.spec_number = descriptors['n']
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        [axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle] = \
            canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors['n']):
            ellipsoids.append(Ellipsoid(phase, axis_1[i], axis_2[i], axis_3[i],
                                        rot_axis_comp_x[i], rot_axis_comp_y[i], rot_axis_comp_z[i],
                                        angle[i]))

    return ellipsoids


def generateCylindricalFibers(phase, rve_dims, descriptors):
    """
    Generate cylindrical fibers of *phase* according to *descriptors*.

    Parameters
    ----------
    phase: `.Phase`
        Phase to which the fibers will belong.

    rve_dims: list(float)
        List containing the dimensions of the RVE.

    descriptors: dictionary
        Dictionary containing the necesary descriptors to generate the microstructure.
    """
    fibers = []
    # Initializing the list containing the fibers
    used_parameters = {parameter for parameter in CylindricalFiber.possible_parameters if
                       any([descriptor.startswith(parameter) for
                            descriptor in descriptors.keys()])}
    # Collecting the parameters used
    if any([used_parameters == acceptable_description for
            acceptable_description in CylindricalFiber.acceptable_descriptions]):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(used_parameters, phase,
                                                CylindricalFiber.acceptable_descriptions)
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if 'n' in descriptors and 'vf' not in descriptors:
    # The desired number of fibers was specified
        phase.specNumber(descriptors['n'])
        # Collecting the specified number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                CylindricalFiber.possible_parameter[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors['n']):
            fibers.append(CylindricalFiber(phase, r[i], descriptors['direction'], rve_dims))
    elif 'vf' in descriptors and 'n' not in descriptors:
    # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors['vf']
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors['vf']:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    CylindricalFiber.possible_parameter[i_parameter],
                    descriptors,
                    phase,
                    rve_dims)
            r = canonicalParametersDisk(current_sample, rve_dims)
            fibers.append(CylindricalFiber(phase, r, descriptors['direction'], rve_dims))
            vf_real += fibers[-1].volume()/(rve_dims[0]*rve_dims[1])
    elif 'vf' in descriptors and 'n' in descriptors:
        phase.spec_volume_fraction = descriptors['vf']
        phase.spec_number = descriptors['n']
        # Collecting the specified volume fraction and number of particles
        samples = {}
        # Initializing the dictionary containing the samples for each parameter used
        for i_parameter in used_parameters:
            samples[i_parameter] = generateSampleParameter(
                i_parameter,
                Ellipsoid.possible_parameters[i_parameter],
                descriptors,
                phase,
                rve_dims,
                n_samples=descriptors['n'])
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
        # np.random.seed(42)
        k = 0
        for i_particle in particles:
            k += 1
        # Running through all the particles
            i_particle.setPositionCenter(
                Particle.box*np.random.uniform(size=i_particle.dim))
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(
                np.zeros((i_particle.dim)))
            # Generating the velocities from a random uniform distribution between -1 and 1
            i_particle.position_center_history = [i_particle.position_center.flatten()]
            # Saving initial configuration
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
                            particles[grid_places[k_counter]].setPositionCenter(np.array(
                                [j*cell_length[0]+cell_length[0]/2,
                                 k*cell_length[1]+cell_length[1]/2,
                                 l*cell_length[2]+cell_length[2]/2]))
                            # Gene<><rating the positions from a random uniform distribution
                            # between 0 and 1
                            particles[grid_places[k_counter]].setVelocityCenter(
                                np.random.uniform(low=0.01, high=0.6, size=3))
                            # Generating the velocities from a random uniform distribution
                            # between -1 and 1
                            particles[grid_places[k_counter]].position_center_history = [
                                particles[grid_places[k_counter]].position_center.flatten()]
                            # Saving particle history
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
                        particles[grid_places[k_counter]].setPositionCenter(np.array(
                            [j*cell_length[0]+cell_length[0]/2,
                             k*cell_length[1]+cell_length[1]/2]))
                        # Gene<><rating the positions from a random uniform distribution between 0 and 1
                        particles[grid_places[k_counter]].setVelocityCenter(
                            np.random.uniform(low=0.01, high=0.6, size=2))  # np.array([0,0],dtype='float')
                        # Generating the velocities from a random uniform distribution between -1 and 1
                        particles[grid_places[k_counter]].position_center_history = [
                            particles[grid_places[k_counter]].position_center.flatten()]
                        # Saving particle history
                    k_counter += 1
    elif type_init_conf == "fcc":
        center_points = np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [1, 1, 0],
             [0.5, 0.5, 0],
             [0.5, 0, 0.5],
            [0.5, 1, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0, 0, 1],
             [1, 0, 1],
             [0, 1, 1],
             [1, 1, 1],
             [0.5, 0.5, 1],
             [0, 2, 0],
             [1, 2, 0],
             [0.5, 1.5, 0],
            [0.5, 2, 0.5],
            [0, 1.5, 0.5],
            [1, 1.5, 0.5],
             [0, 2, 1],
             [1, 2, 1],
             [0.5, 1.5, 1]
            ])
        k = 0
        for i_particle in particles:
        # Running through all the particles
            i_particle.setPositionCenter(center_points[k]/2) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.zeros((i_particle.dim))) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get('save_history'):
            # Saving particle history
                i_particle.position_center_history = [i_particle.position_center.flatten()]
            k += 1
    elif type_init_conf == 'overlap':
        k = 0
        for i_particle in particles:
        # Running through all the particles
            # i_particle.setPositionCenter(np.array([0.5 + 2*k*0.01, 0.5])) # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # # Generating the positions from a random uniform distribution between 0 and 1
            # i_particle.setVelocityCenter(np.array([1e-4 - 2*k*1e-4, 0])) #np.array([0,0],dtype='float')
            i_particle.setPositionCenter(np.array([0.5, 0.5])) # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.array([0, 0])) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get('save_history'):
            # Saving particle history
                i_particle.position_center_history = [i_particle.position_center.flatten()]
            k += 1
    elif type_init_conf == 'custom':
        path = "/home/zeluis/Documents/Tese/programa/studies/thermostats/minkowski/artificial_2D/ord.txt"
        positions = np.loadtxt(path)
        for ind, i_particle in enumerate(particles):
            # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            i_particle.setPositionCenter(positions[ind, 0:2]/500)
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.array([0, 0])) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get('save_history'):
            # Saving particle history
                i_particle.position_center_history = [i_particle.position_center.flatten()]
    elif type_init_conf == 'adjacent':
        k = 0
        for i_particle in particles:
        # Running through all the particles
            i_particle.setPositionCenter(np.array([0.1, 0.1, 0.01 + k*0.98])) # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.array([0, 0, 0])) #np.array([0,0],dtype='float')
            if kwargs.get('save_history'):
            # Saving particle history
                i_particle.position_center_history = [i_particle.position_center.flatten()]
            k += 1
    else:
        try:
            raise errors.UnsupportedInitialConfigurationType(type_init_conf)
        except errors.UnsupportedInitialConfigurationType as error:
            error.message()
            quit()


def generateSampleParameter(parameter, parameter_name, descriptors, phase, rve_dims,
                            n_samples=1, max_sample=100):
    """Generate a sample of values for *parameter* according to descriptors"""
    size_geom_param = {'r', 'major_axis', 'minor_axis', 'axis_1', 'axis_2', 'axis_3'}
    # Geometrical parameters related to the size of the particle that must larger than
    # ans smaller than half the size of the smallest dimension of the RVE
    if descriptors.get(parameter + '_distribution') == 'uniform':
    # the radius follows a uniform distribution
        try:
            if parameter + '_low' not in descriptors:
            # Checking if the  lower bound was supplied
                raise errors.ParameterMissing(parameter + '_low', phase.type)
            elif parameter + '_high' not in descriptors:
            # Checking if the upper bound was supplied
                raise errors.ParameterMissing(parameter + '_high', phase.type)
            elif descriptors[parameter + '_low'] >= descriptors[parameter + '_high']:
            # Checking if the lower bound is smaller than the upper bound
                raise errors.UnexpectedValue(
                    descriptors[parameter + '_low'], '{0}_low of phase {1}'.format(
                     parameter, phase.type),
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
        phase.addGeomParameter(parameter_name, 'Uniform',
                               [('Lower bound', descriptors[parameter + '_low']),
                               ('Upper bound', descriptors[parameter + '_high'])])
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
            phase.addGeomParameter(parameter_name, 'Normal',
                               [('Mean', descriptors[parameter + '_mean']),
                               ('Std Var', descriptors[parameter + '_sigma'])])
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
                        elif descriptors[i_descriptor] > np.min(rve_dims)/2:
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
        value_prob_pairs = [[('Value {0}'.format(ind+1), val), ('Proability {0}'.format(ind+1), prob)] for (ind, val), prob in zip(enumerate(values),probabilities)]
        value_prob_pairs_flat = [item for sublist in value_prob_pairs for item in sublist]
        phase.addGeomParameter(parameter_name, 'Discrete', value_prob_pairs_flat)
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
        if parameter != 'n' and parameter != 'vf':
            phase.addGeomParameter(parameter_name, 'Fixed',
                                   ('Value', descriptors[parameter]))
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
    elif 'ratio' in sample and 'vf' in sample and 'n' in sample:
        volume_part = sample['vf'][0]*rve_dims[0]*rve_dims[1]/sample['n'][0]
        minor_axis = np.sqrt(volume_part/(np.pi*sample['ratio']*1/4))
        major_axis = sample['ratio']*minor_axis
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
    if 'ratio_12' in sample and 'ratio_13' in sample and 'vf' in sample and 'n' in sample:
        volume = sample['vf']*rve_dims[0]*rve_dims[1]*rve_dims[2]/sample['n']
        axis_1 = np.cbrt(volume*sample['ratio_12']*sample['ratio_13']*8/(np.pi*4/3))
        axis_2 = axis_1/sample['ratio_12']
        axis_3 = axis_1/sample['ratio_13']
        print('axis', axis_1, axis_2, axis_3, 'volume', sample['n']*4/3*axis_1*axis_2*axis_3/8*np.pi, sample['vf'], sample['n'])
    if 'angle' in sample:
        angle = sample['angle']
    if 'rot_axis_comp_x' in sample and 'rot_axis_comp_y' in sample \
        and 'rot_axis_comp_z' in sample:
    # Euler angles
        rot_axis_comp_x = sample['rot_axis_comp_x']
        rot_axis_comp_y = sample['rot_axis_comp_y']
        rot_axis_comp_z = sample['rot_axis_comp_z']

    return [axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle]


def createResultsDirectory(particles, dp_dir, remesh=False):
    """
    Create the results directory.

    Parameters
    ----------
    particles: `.particle`
        Particles in the system.

    dp_dir: string
        Directory where the results are going to be stored.

    remesh: boolean, optional
        Signals if the program is currently being used for a remesh action.
    """
    Particle.file_name = "mic"
    # Defining the file name associated with this sampling. The filenames of the particles
    # are always prefixed by mic
    results_folder = os.path.join(dp_dir,  Particle.file_name)
    # Creating a tentative path for the results folder
    results_folder_old = results_folder
    # Saving the original name of the results folder
    i = 0
    # Initializing the filename suffix
    while True:
        results_folder = results_folder_old + "_" + str(i)
        # Creating a new folder name appending an integer to the name of the original
        # folder
        i += 1
        # Increasing the filenam suffix
        if not os.path.exists(results_folder):
        # Repeat while the folder names already exists
            break
    os.makedirs(results_folder)
    # Creating the directory
    if os.path.exists("input_data\\info_micro.p") and not remesh:
        shutil.copy("input_data\\info_micro.p", os.path.join(
            results_folder, "info_micro.p"))
        # copying input file
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    # Saving the file path in the Particle class


def particleGeneration(descriptors, phase_types, rve_dims, problem_type, dp_dir,
                       type_init_conf, save_history=True):
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
    Particle.phases = {i_phase: Phase(
        i_phase, phase_types[i_phase]) for i_phase in descriptors}
    Particle.list_phases = [i_phase for i_phase in descriptors]
    # Dictionary containing the phases
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
    for i_phase in Particle.phases.values():
    # Running through all the phases listed in the dictionary
        try:
            if i_phase.type == 1:
            # This phase is the matrix
                Particle.matrix_phase = i_phase.name
                # No particles are generated
            elif i_phase.type == 2:
            # This phase is made up by disks
                if len(rve_dims) != 2:
                # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase('Disks', 2, 3, i_phase.name)
                particles = particles + \
                    generateDisks(i_phase, rve_dims, descriptors[i_phase.name])
                # Generating the number of disks requested and appending them to the list of
                # particles
            elif i_phase.type == 3:
            # This phase is made up by ellipses
                if len(rve_dims) != 2:
                # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase('Ellipses', 2, 3, i_phase.name)
                particles = particles + \
                    generateEllipses(i_phase, rve_dims, descriptors[i_phase.name])
                # Generating the number of ellipses requested and appending them to the list
                # of particles
            elif i_phase.type == 4:
            # This phase is made up by spheres
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase('Spheres', 3, 2, i_phase.name)
                particles = particles + \
                    generateSpheres(i_phase, rve_dims, descriptors[i_phase.name])
                # Generating the number of spheres requested and appending them to the list
                # of  particles
            elif i_phase.type == 5:
            # This phase is made up by ellipsoids
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase('Ellipsoids', 3, 2, i_phase.name)
                particles = particles + \
                    generateEllipsoids(i_phase, rve_dims, descriptors[i_phase.name])
                # Generating the number of ellipsoids requested and appending them to the
                # list of particles
            elif i_phase.type == 6:
            # This phase is made up by cylindrical fibers
                if len(rve_dims) != 3:
                # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        'Cylindrical Fibers', 3, 2, i_phase)
                if any([i_phase.type != 1 and i_phase.type != 6 for i_phase.type in
                        list(phase_types.values())]):
                    raise errors.OnlyCylindricalFibers()
                particles = particles + \
                    generateCylindricalFibers(i_phase, rve_dims, descriptors[i_phase.name])
                # Generating the number of cylindrical fibers requested and appending them
                # to the list of particles
            else:
                raise errors.UnsupportedPhaseType(i_phase.type, i_phase.name)
        except (errors.IncompatibleDimensionsRVEphase,
                errors.OnlyCylindricalFibers) as error:
            error.message()
            quit()

    print_funcs.printToFile("**PHASE DESCRIPTORS**\n")
    for i_phase in Particle.phases.values():
    # Running through all the phases to print their info
        i_phase.printSpecDescriptors()
        i_phase.printRealDescriptors()
    print_funcs.printToFile('='*80)

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
    ----------
    dp_dir: string
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored

    mic_gen_program: integer
        Integer variable (read from the user input data file) which specifies an
        available program to generate the microstructure(s) and associated
        discretization file(s) of a given design point

    mic_gen_parameters: array
        An array which contains all the required parameters (or options)
        for the selected program to generate the microstructure(s) and
        and associated discretization file(s) of a given design point.

        ================================ ======================================
        Option                           Description
        ================================ ======================================
        "max_residue_per_particle"       Maximum overlap residue per particle.
        "max_step"                       Maximum number of iterations.
        "integration_scheme"             Optional. {'Newmark'}. Integration scheme
                                         for the equations of motion.
        "speed_up_scheme"                Optional. {'Naive', 'Cell', 'Verlet'}.
                                         Speed up scheme used for force computation
        "remesh"                         Optional. Boolean signaling a remesh action.
        "dir_previous_mic"               Optional. Directory where the input and
                                         output files of a previous microstructure
                                         are saved. They must have their original names.
        ================================ ======================================

    problem_type: integer
        Problem type    | 1. 2D problem (plain strain)
                        | 2. 2D problem (plain stress)
                        | 3. 2D problem (axisymmetric)
                        | 4. 3D problem

    n_dp_samples: integer
        Number of microstructures (samples) to be generated, associated to
        the given design point

    mic_gen_descriptors_array: array
        A dictionary which contains all the microstructure
        descriptor-related information required to generate the
        given design point microstructure(s) automatically stored as:

                                        Microstructure Descriptors
                                  _                                    _
        dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
                                 |_  < value >     < value >      ...  _|.

        See notes_.

    phase_types: dictionary
        Dictionary which contains each material phase type, stored as
                       dictionary['phase_id'] = phase_type
    discret_file_ext: list
        List which contains the required spatial discretization file(s), stored as:

                        array = [ < discret_type > < discret_type >  ... ]

    discret_spec_array: dictionary
        Dictionary which contains the required parameters to generate
        each type of specified discretization file, stored as:

                               dictionary['disc_ext']['parameter'] = [ ... ]
    Notes
    -----
    The parameters for microstructure generation depend on the shape of the particle. They
    are detailed in the following tables. Particular choices of their values may lead to
    incompatibilities.

        ================================ ======================================
        Disk: Choose 2 of the parameters
        -----------------------------------------------------------------------
        'r'                              Radius of the disk
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

        ================================ ======================================
        Ellipse: Choose 4 of the parameters, including 'angle'
        -----------------------------------------------------------------------
        'major_axis'                     Radius of the disk
        'minor_axis'                     Number of particles
        'angle'                          Volume fraction
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

    Any parameter may have a chosen distribution, specified as detailed below:
    - Fixed distribution: The parameters are fixed. Simply specify the parameter.
    - Discrete distribution: There parameters follow a discrete distribution, where the
    parameters take only the given values with the given probability.
        1. Specify the distribution of parameter *param* as::

                    np.array([['param_distribution']['fixed'], dtype=obj)

        2. Specify the value of the parameter and the probability of that value occuring::

                (np.array([['param_1', 'prob_param_1', 'param_2', 'prob_param_2'],
                        [1, 0.4, 2, 0.6]], dtype=obj))

    - Uniform distribution: "*_distribution"

            np.array([['distribution_param']['uniform'], dtype=obj)

    - Gaussian distribution:
    """
    info_dict = pickle.load(open('input_data\\info_micro.p', 'rb'))
    # Loading the dictionary containing the information about the microstructure and its
    # generation
    dp_dir = info_dict.get('dp_dir')
    # Directory where the microstructure spatial discretization file(s) associated
    # with the given design point are to be stored
    options = info_dict.get('mic_gen_parameters')
    # An array which contains all the required parameters (or options)
    # for the selected program to generate the microstructure(s) and
    # and associated discretization file(s) of a given design point
    problem_type = info_dict.get('problem_type')
    # Getting the problem type
    n_dp_samples = info_dict.get('n_dp_samples', 1)
    # Number of samples to be generated using the descriptors supplied
    try:
        if not isinstance(n_dp_samples, int) or n_dp_samples < 1:
        # The number of samples must be an integer larger or equal to 1
            raise errors.NumberSamples(n_dp_samples)
    except errors.NumberSamples() as error:
        error.message()
        quit()

    descriptors = info_dict.get('mic_gen_descriptors', {})
    # mic_gen_descriptors_array: dictionary

    phase_types = info_dict.get('phase_types', {})
    # phase_types: dictionary
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
        print_funcs.printToFile('Warning: Different RVE sizes in the mesh specifications.')
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
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    Particle.relative_energy_history.append(relative_energy)
    # Saving the relative energy

    return relative_energy


def computeKineticEnergy(particles):
    # Obtaining a list with the norms of the vector forces
    kin_energy = np.sum([i_particle.volume()*np.sum(i_particle.velocity_center**2)
                        for i_particle in particles])
    Particle.kinetic_energy_history.append(kin_energy)
    # Saving the kinetic energy

    return kin_energy


def forceOutTangentWall(particles, min_distance):
    tol = 0.5*min_distance
    if particles[0].dim == 2:
        for i_particle in particles:
            pos = i_particle.position_center
            if np.abs(i_particle.radius - min_distance - pos[0]) <  tol:
                i_particle.position_center += np.array([1e-2, 0])
            elif np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance) < tol:
                i_particle.position_center += np.array([-1e-2, 0])
            elif np.abs(i_particle.radius - min_distance - pos[1]) <  tol:
                i_particle.position_center += np.array([0, 1e-2])
            elif np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance) < tol:
                i_particle.position_center += np.array([0, -1e-2])
    elif particles[0].dim == 3:
        for i_particle in particles:
            pos = i_particle.position_center
            if np.abs(i_particle.radius - min_distance - pos[0]) <  tol:
                i_particle.position_center += np.array([1e-2, 0, 0])
            elif np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance) < tol:
                i_particle.position_center += np.array([-1e-2, 0, 0])
            elif np.abs(i_particle.radius - min_distance - pos[1]) <  tol:
                i_particle.position_center += np.array([0, 1e-2, 0])
            elif np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance) < tol:
                i_particle.position_center += np.array([0, -1e-2, 0])
            elif np.abs(i_particle.radius - min_distance - pos[2]) <  tol:
                i_particle.position_center += np.array([0, 0, 1e-2])
            elif np.abs(pos[2] - Particle.box[2] + i_particle.radius - min_distance) < tol:
                i_particle.position_center += np.array([0, 0, -1e-2])

def checkTangentToWall(particles, min_distance):
    tol = 0.5*min_distance
    not_tangent_to_wall = True
    print('tol', tol)
    if particles[0].dim == 2:
        for i_particle in particles:
            pos = i_particle.position_center
            # print('tol', tol, 'radius', i_particle.radius)
            if (np.abs(i_particle.radius - min_distance - pos[0]) <  tol or
               np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance) < tol or
               np.abs(i_particle.radius - min_distance - pos[1]) <  tol or
               np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance) < tol ):
                not_tangent_to_wall = False
    elif particles[0].dim == 3:
        for i_particle in particles:
            pos = i_particle.position_center
            # print('tol', tol, 'radius', i_particle.radius)
            if (np.abs(i_particle.radius - min_distance - pos[0]) <  tol or
               np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance) < tol or
               np.abs(i_particle.radius - min_distance - pos[1]) <  tol or
               np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance) < tol or
               np.abs(i_particle.radius - min_distance - pos[2]) <  tol or
               np.abs(pos[2] - Particle.box[2] + i_particle.radius - min_distance) < tol ):
                not_tangent_to_wall = False
    # for i_particle in particles:
    #     for j_image in range(-1, 2):
    #         for k_image in range(-1, 2):
    #             pos = i_particle.position_c enter + [j_image, k_image]*Particle.box
    #             if (i_particle.radius < pos[0] < i_particle.radius + tol or
    #                -i_particle.radius < pos[0] < -i_particle.radius + tol or
    #                Particle.box[0] - i_particle.radius - tol < pos[0] < Particle.box[0] - i_particle.radius or
    #                Particle.box[0] + i_particle.radius - tol < pos[0] < Particle.box[0] + i_particle.radius or
    #                i_particle.radius  < pos[1] <  i_particle.radius + tol or
    #                -i_particle.radius < pos[1] < -i_particle.radius + tol or
    #                Particle.box[1] - i_particle.radius < pos[1] < Particle.box[1] - i_particle.radius + tol or
    #                Particle.box[1] + i_particle.radius < pos[1] < Particle.box[1] + i_particle.radius + tol):
    #                 not_tangent_to_wall = False
    return not_tangent_to_wall


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
    min_distance = options.get('min_distance', 0)
    # Saving the minimum distance
    speed_up_scheme = options.get('speed_up_scheme', 'Cell')
    # What is the speed up scheme to be used
    max_steps_to_relax = options.get('max_steps_to_relax', 100)
    # Maximum number of iterations
    dt = options.get('dt', 0.05)
    # Time integration step
    thermostat = options.get('thermostat', 'multi_temperature')
    # Thermostat to be used
    save_history = options.get('save_history', True)
    # Save the complete motion
    # --------------------------------------------------------------------------------------
    N = Particle.number
    # Saving the number of particles
    box = Particle.box
    # Saving the array containing the size of the box
    dim = particles[0].dim
    # Saving the array containing the dimension of the problem
    if min_distance > 0:
    # There is a minimum distance
        dilateParticles(particles, min_distance)
        # Dilate all particles
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
    Particle.max_residue = max_residue
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0
    computeForces(particles, speed_up_scheme)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    relative_energy = computeRelativeEnergy(particles)
    # Computing the relative energy
    kin_energy = computeKineticEnergy(particles)
    # Computing the kinetic energy
    if thermostat == 'multi_temperature':
    # The thermostat used is the isokinetic scheme
    # Setting the options
        if particles[0].dim == 2:
            jump = options.get('equilibration_steps', 25)
            # Number of steps allowed for the system to equilibrate and explore and given
            # temperature before the criterion for temperature lowering is checked
        elif particles[0].dim == 3:
            jump = options.get('equilibration_steps', 25)
        jump_list = []
        last_alt = options.get('inital_temp_steps', 40)
        # Number of steps allowed for the system to equilibrate and explore the initial
        # temperature
        T_ref = options.get('initial_temp', 2.5e10) #*(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        if kin_energy > 1e-10:
        # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2*particles[0].dim*N*k_b*T_ref/kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
        else:
        # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities
    elif thermostat == 'isokinetic':
        T_ref = options.get('initial_temp',  2.5e10) #*(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        jump = options.get('equilibration_steps', 25) # + 5*100*0.65/(Particle.number*Particle.volume/Particle.volume_RVE))
        # Number of steps allowed for the system to equilibrate and explore and given
        # temperature before the criterion for temperature lowering is checked
        if kin_energy > 1e-10:
        # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2*particles[0].dim*N*k_b*T_ref/kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
            print('T_ref', T_ref)
        else:
        # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities
    print_funcs.printToTerminalRefresh(
        step, Particle.total_overlap, relative_energy, kin_energy, temp=T_ref, first=True)
    # Print info about the iteration
    while (step < max_step) and n_steps_relax < max_steps_to_relax:
        # Run the simulation while the number of steps the overlap has been smaller than the
        # allowed maximum residue is larger than options['max_steps_to_relax'], so that the
        # particles have time to get away from each other.
        if save_history:
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
        Particle.total_overlap_history.append(Particle.total_overlap)
        kin_energy = computeKineticEnergy(particles)
        # Computing the kinetic energy
        if thermostat == 'multi_temperature':
        # The thermostat used is the multi_temperature scheme
            if step > last_alt:
            # If the end of the equilibration time has been reached
                if Particle.total_overlap > max_residue:
                # If a legal configuration has not been achieved
                    if any(np.array(Particle.total_overlap_history[-jump//2 :]) - np.array(Particle.total_overlap_history[-jump//2 -1: -1]) > 0):
                        # If the total overlap has increase in the previous iterations
                        T_ref *= 1/4
                        # Lowering the temperature
                        jump += step - last_alt - 1
                        # Updating the equilibration time
                        last_alt = step + jump
                        # Updating the iteration of the last temperature change
                        Particle.temp_change_steps.append(step)
                        jump_list.append(jump)
                        # Saving minimum equilibration times and times at which the
                        # temperature has been lowered
            # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2*particles[0].dim*N*k_b*T_ref/kin_energy)
            # Rescalling factor
            for i_particle in range(N):
                # Running through all the particles
                particles[i_particle].velocity_center *= lambda_vel
                # Rescalling the velocities
            if relative_energy/Particle.total_overlap < 1e-8 and Particle.total_overlap > max_residue:
            # FIXME: this criterion is giving false positives, relative energy falls
            # much faster than total overlap
                pass
        if thermostat == "isokinetic":
        # The thermostate used is the isokinetic with constant temperature
            lambda_vel = np.sqrt(2*particles[0].dim*N*k_b*T_ref/kin_energy)
            for i_particle in range(N):
                # Running through all the particles
                particles[i_particle].velocity_center *= lambda_vel
                # Rescalling the velocities
        else:
            # There is no thermostat
            pass
        if Particle.total_overlap <= max_residue:
            check_tangent = checkTangentToWall(particles, min_distance)
            if check_tangent:
                # If the configuration has an overlap area smaller than the tolerance
                n_steps_relax += 1
                # print('yes',n_steps_relax)
            else:
                n_steps_relax = 0
                # Restarting the count
                forceOutTangentWall(particles, min_distance)
        print_funcs.printToTerminalRefresh(
            step, Particle.total_overlap, relative_energy, kin_energy)
        if step > 5*jump and all((np.abs(np.array(Particle.total_overlap_history[-5*jump:]) - np.array(Particle.total_overlap_history[-5*jump-1:-1])))/np.array(Particle.total_overlap_history[-5*jump-1:-1])*100 < 1e-5):
            print_funcs.printToFile('Failed sample')
            break
    if min_distance > 0:
    # There is a minimum distance
        contractParticles(particles, min_distance)
        # Contract all particles
    if thermostat == "multi_temperature":
        Particle.equilibration_steps.append(jump_list)
    if not save_history:
    # If the complete motion was not saved
        for i_particle in particles:
            i_particle.position_center_history.append(i_particle.position_center.flatten())
            # Saving the final configuration


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


def main(dp_dir, descriptors, phase_types, options, n_samples, rve_dims, problem_type,
         discret_spec_array, discret_file_ext):
    """Run the microstructure generation program."""
    if options.get('remesh'):
    # It is a remesh action
        current_RVE = pickle.load(open(options['dir_previous_mic'], 'rb'))
        # No need to generate a new microstructure. Using a previous microstructure.
        particles, rve_dims = current_RVE.useThisRVE(dp_dir)
        # Reconstructing the relevant Particle attributes that could not be pickled
        createResultsDirectory(particles, dp_dir, remesh=True)
        # Create results directory
        for disc_ext in discret_file_ext:
        # For each file extension asked
            generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
            # Generate corresponding mesh
        motion_analysis = options.get('motion_analysis', False)
        if motion_analysis:
            doMotionAnalysis(particles, rve_dims, Particle.file_path)
            # Do analysis of the motion of the particles
    else:
    # Generating samples of microstructures and meshing
        for i_sample in range(n_samples):
            # Producing the number of samples required
            print_funcs.printInitialMessage()
            # Printing initial message
            rve_dims = options.get('rve_dims')
            save_history = options.get('save_history', True)
            voronoi_analysis = options.get('voronoi_analysis', False)
            motion_analysis = options.get('motion_analysis', False)
            type_init_conf = options.get('type_initial_configuration', 'random')
            max_residue_per_particle = options.get('max_residue_per_particle', 0)
            max_step = options.get('max_step', 1)
            # Collecting options
            start = time.time()
            # Keeping track of the simulation time
            particles = particleGeneration(descriptors, phase_types, rve_dims, problem_type,
                                           dp_dir, type_init_conf=type_init_conf,
                                           save_history=save_history)
            # Generating the list of particles from the geometrical descriptors
            run(particles, max_residue_per_particle, max_step, options)
            # Running the molecular dynamics simulation
            end = time.time()
            Particle.time = end - start
            print_funcs.printFinalMessage(Particle.time, Particle.total_overlap, len(
                Particle.total_overlap_history), i_sample+1, Particle.max_residue)
            # Time spent on microstructure generation
            current_RVE = RVE(particles, rve_dims)
            # Saving the RVE properties in an RVE object
            pickle.dump(current_RVE, open(Particle.file_path + ".p", "wb"))
            # Saving the configuration for later use
            for disc_ext in discret_file_ext:
            # For each file extension asked
                generateMesh(particles, disc_ext, discret_spec_array[disc_ext])
                # Generate corresponding mesh
            if motion_analysis:
                doMotionAnalysis(particles, rve_dims, Particle.file_path)
                # Do analysis of the motion of the particles
            if voronoi_analysis:
                voronoi_type = options.get('voronoi_type', 'standard')
                doVoronoiAnalysis(
                    particles, rve_dims, Particle.file_path, voronoi_type=voronoi_type)
                # Do a voronoi analysis
            os.replace("temp.screen", Particle.file_path + ".screen")
            # Moving the screnn of this sample to the respective directory
            Particle.resetRVE()
            # Clearing the properties of the simulation box


class Keyword(object):
    """This is the class for keywords used in the input file.

    Class Attributes
    ----------------
    simulation: `.simulation`
    Simulation object con

    """

    simulation = None
    all_keywords = {}
    all_keyword_groups = set()
    i_line = 0
    input_file = None
    input = None
    all_options = {}
    mandatory_keywords_not_set = set()

    def __init__(self, name, keyword_group=None, parent_keyword=None, type=None,
                 mandatory=True, **kwargs):
        """
        Constructor for the Keyword class.

        Parameters
        ----------
        name: str
            Name of the keyword.

        keyword_group: {'PROBLEM_TYPE', 'N_DP_SAMPLES', 'MIC_GEN_PARAMETERS',
            'MIC_GEN_DESCRIPTORS', 'MESH_OPTIONS'}
            Group to wich the keyword belongs. Used for storage in the right variable.

        type: str, optinal
            Type of the variable

        mandatory: boolean, optional
            Mandatory or optional keyword. True by default.

        Keyword Arguments
        -----------------
        default_value: object
            Default value for the keyword
        """
        self.name = name
        self.type = type
        self.mandatory = mandatory
        if 'default_value' in kwargs:
            self.default_value = kwargs['default_value']
        self.specified = False
        self.top_level_keywords = set()

    def checkLowerLevelKeywords(self):
        print([o.name for o in self.top_level_keywords])
        # while True:
        print('iline', Keyword.i_line, len(Keyword.input))
        current_line_keyword = False
        line = Keyword.input[Keyword.i_line]
        print('line', line)
        for possible_keyword in self.top_level_keywords:
        # Checking what is the current keyword
            print(possible_keyword.name)
            print(line)
            if possible_keyword.isIn(line):
                print('here')
                current_line_keyword = True
                # General keyword has been found
                print('here_2', self.name)
                val = possible_keyword.readValue()
                possible_keyword.storeValue(val)
                possible_keyword.removeFromMandatory()
                # read the values corresponding to the keyword detected
                break
        if not current_line_keyword:
            if isinstance(self, TopLevelReader):
                print(line, 'does not contain a keyword')
                # FIXME: create an appropriate error
                quit()
            else:
                raise ValueError()


    def ignoreComments(self):
        while Keyword.i_line < len(Keyword.input):
            line = Keyword.input[Keyword.i_line]
            # Going through all the line in the input file
            if line.strip() == '' or line.startswith("#") or line.strip() == '[insert here]':
            # if the line is empty or a comment move on to the next
                Keyword.i_line += 1
                continue
            else:
                break

    def readValue(self):
        print('here_4', Keyword.i_line)
        line = Keyword.input[Keyword.i_line]
        print('here_3')
        try:
            if self.type == 'float':
                value_str = line.split()[1]
                final_val = float(value_str)
            elif self.type == 'int':
                value_str = line.split()[1]
                final_val = int(value_str)
            elif self.type == 'bool':
                value_str = line.split()[1]
                if value_str == 'True':
                    final_val = True
                elif value_str == 'False':
                    final_val = False
                else:
                    raise ValueError
            elif self.type == 'str':
                value_str = line.split()[1]
                final_val = value_str
            elif self.type == 'none':
                final_val = self.name
            else:
                value_str = line.split()[1]
                final_val = value_str
        except ValueError:
            errors.IncompatibleValue.messsage("Error")
            quit()
        self.specified = True
        # the current parameter has been specified by the used
        Keyword.i_line += 1
        return final_val

    def checkType(self, val):
        if self.type == 'float':
            correct_type = isinstance(val, float)
        elif self.type == 'int':
            correct_type = isinstance(val, int)
        elif self.type == 'bool':
            correct_type = isinstance(val, bool)
        elif self.type == 'str':
            correct_type = isinstance(val, str)
        else:
            correct_type = isinstance(val, str)
        if correct_type is not True:
            raise ValueError()

    def isIn(self, line):
        """Check if the first string in the *line* is the keyword *self*."""

        if line.split()[0] == self.name:
            isIn = True
        else:
            isIn = False

        return isIn

    def removeFromMandatory(self):
        try:
            Keyword.mandatory_keywords_not_set.remove(self)
        except KeyError:
            pass

class KeywordTypeA(Keyword):

    def __init__(self, name, store, **kwargs):
        super().__init__(name, **kwargs)
        self.keyword_group = store
        if hasattr(self, 'default_value'):
            self.storeValue(self.default_value)

    def storeValue(self, val):
        Keyword.all_options.setdefault(self.keyword_group, {})
        Keyword.all_options[self.keyword_group][self.name] = val

    def writeValue(self, input_file):
        val = Keyword.all_options[self.keyword_group][self.name]
        self.checkType(val)
        input_file.write("{0} {1}".format(self.name, val))


class KeywordTypeB(KeywordTypeA):

    def __init__(self, name, store, **kwargs):
        super().__init__(name, store, **kwargs)
        self.keyword_group = store
        if hasattr(self, 'default_value'):
            self.storeValue(self.default_value)

    def storeValue(self, value):
        Keyword.all_options[self.name] = value

    def writeValue(self, input_file):
        val = Keyword.all_options[self.name]
        self.checkType(val)
        input_file.write("{0} {1}".format(self.name, val))

class KeywordTypeC(Keyword):

    def __init__(self, name, header_keys, sub_keys, **kwargs):
        super().__init__(name, **kwargs)
        self.header_keys = header_keys
        self.sub_keys = sub_keys
        print('keyword', [keyword.name for keyword in sub_keys])

        if hasattr(self, 'default_value'):
            self.storeValue(self.default_value)

    def isIn(self, line):
        """Check if the first string in the *line* is the keyword *self*."""

        if line.split()[0] == self.name:
            isIn = True
        else:
            isIn = False

        return isIn

    def readValue(self):

        options = {}
        print('here_6')
        Keyword.i_line += 1
        # Moving over the line containing "Mic_Gen_Descriptors"
        while Keyword.i_line < len(Keyword.input):
            self.ignoreComments()
            line = Keyword.input[Keyword.i_line]
            # New line
            if all([not keyword.isIn(line)
                    for keyword in self.header_keys.union(self.sub_keys)]):
            # If another top level keyword has been specified exit the MIC_GEN_DESCRIPTORS
            # block
                break
            print('line', line)
            for header_keyword in self.header_keys:
                if header_keyword.isIn(line):
                    current_header = header_keyword.readValue()
                    options[current_header] = {}
                    break
            print('line', line)
            for sub_keyword in self.sub_keys:
                if sub_keyword.isIn(line):
                    value = sub_keyword.readValue()
                    options[current_header][sub_keyword.name] = value
                    print('here')
                    break

        return options

    def storeValue(self, val):
        Keyword.all_options[self.name] = val

    def writeValue(self, input_file):
        input_file.write("{0}".format(self.name))
        for header in Keyword.all_options[self.name]:
            if all([header_key.type == 'none' for header_key in self.header_keys]):
                input_file.write("{0}").format(header)
            else:
                input_file.write('{0} {1}'.format(self.header_keys.pop().name, header))
            for option, option_val in Keyword.all_options[self.name][header].items():
                for key in self.sub_keys:
                    if key.name == option:
                        key.checkType(option_val)
                input_file.write("{0} {1}".format(option, option_val))


class TopLevelReader(Keyword):

    def __init__(self):
        super().__init__('TopLevelKeyword')

    def moveAlong(self):
        while Keyword.i_line < len(Keyword.input):
            # Saving current line as line
            self.ignoreComments()
            self.checkLowerLevelKeywords()

    def readInputFile(self, input_file_path):
        # Possible keywords appearing in the input file
        with open(input_file_path, 'r') as input:
            Keyword.input = input.readlines()
            Keyword.i_line = 0
            # Initializing the line counter
            self.moveAlong()
        if len(Keyword.mandatory_keywords_not_set) > 0:
            print({keyword.name for keyword in Keyword.mandatory_keywords_not_set})
            raise ValueError()

    def readValue(self):
        pass

    def storeValue(self, *args):
        pass

    def addTopLevelKeyword(self, *args):
        """Add a possible keyword to the input reader."""

        for keyword in args:
            self.top_level_keywords.add(keyword)
            Keyword.all_keywords[keyword.name] = keyword
            if keyword.mandatory:
                Keyword.mandatory_keywords_not_set.add(keyword)


class Simulation():
    """Class for the simulations that generate the microstructures."""

    def __init__(self, working_directory):
        """Initizalizer for the Simulation Class."""

        self.dp_dir = working_directory
        #     Directory where the microstructure spatial discretization file(s) associated
        #     with the given design point are to be stored
        self.mic_gen_parameters = {}
        # Dictionayr with generation parameters
        self.mic_gen_descriptors = {}
        # Dictionaty containing the descriptors for the phases
        self.phase_types = {}
        # Dictionary containing the phase types
        self.discret_file_ext = []
        # list containing the extensions for the output mesh files
        self.discret_spec_array = {}
        # Parameters for the generation of the meshes
        self.problem_type = 0
        # Problem type
        self.n_dp_samples = 0
        # Number of samples to be generated


# class InputReader():
#     """Class used to read the input from a txt file.
# 
#         It checks only if all the mandatory parameters were supplied with the correct type.
#         It does not check if the value of the parameters makes sense given the values of
#         the other parameters."""
# 
#     def __init__(self, simulation):
#         """Initializer for the InputReader class"""
# 
#         self.simulation = simulation
#         self.top_level_keywords = {}
#         self.all_keywords = {}
#         self.mandatory_keywords_not_set = {}
# 
#     def addTopLevelKeyword(self, **args):
#         """Add a possible keyword to the input reader."""
# 
#         for keyword in args:
#             self.top_level_keywords.add(keyword)
#             self.all_keywords[keyword.name] = keyword
#             if keyword.mandatory:
#                 self.mandatory_keywords_not_set.add(keyword)
#             else:
#                 # FIXME: set default value
#                 pass
# 
#     def addLowLevelKeyword(self, top_keyword_name, **args):
#         """"Add a possible lower level keyword for a higher level keyword-"""
# 
#         top_level_keyword = self.all_keywords[top_keyword_name]
#         for keyword in args:
#             top_level_keyword.top_level_keywords.add(keyword)
#             self.all_keywords[keyword.name] = keyword
#             if keyword.mandatory:
#                 top_level_keyword.mandatory_keywords_not_set.add(keyword)
#             else:
#                 # FIXME: set default value
#                 pass
# 
#     def readKeywordMicGenParameters(self, input_reader):
#         """Read information for a keyword belonging to the MIC_GEN_PARAMETERS group."""
# 
#         current_sim = input_reader.simulation
#         line = input[input_reader.i_line]
#         value = self.readValue(line)
#         current_sim.set_option(self) mic_gen_parameters[self.name.lower()] = value
#         input_reader.i_line += 1
# 
#     def readKeywordMicGenDescriptors(self, keyword):
#         """Read information for a keyword belonging to the MIC_GEN_DESCRIPTORS group."""
# 
#         current_sim = self.simulation
#         self.i_line += 1
#         # Moving over the line containing "Mic_Gen_Descriptors"
#         while self.i_line < len(self.input):
#             line = input[self.i_line]
#             # New line
#             if line.strip() == '' or line.startswith("#"):
#             # if the line is empty or a comment move on to the next
#                 self.i_line += 1
#                 continue
#             if any([keyword.isIn(line)
#                     for keyword in self.top_level_keywords]):
#             # If another top level keyword has been specified exit the MIC_GEN_DESCRIPTORS
#             # block
#                 break
#             if keyword.line.startswith('Phase'):
#                 name_current_phase = line.split()[1]
#                 current_sim.mic_gen_descriptors[name_current_phase] = {}
#                 self.i_line += 1
#             elif line.startswith('phase_type'):
#                 current_sim.phase_types[name_current_phase] = int(line.split()[1])
#                 self.i_line += 1
#             else:
#                 parameter = line.split()[0]
#                 value = line.split()[1]
#                 current_sim.mic_gen_descriptors[name_current_phase][parameter] = \
#                     str2type(value)
#                 self.i_line += 1
# 
# 
#     def readKeywordProblemType(self, problem_type, i_line, input):
#         """Read information for a keyword belonging to the PROBLME_TYPE group."""
# 
#         problem_type = self.readValue(line)
#         i_line += 1
# 
#         return [problem_type, i_line]
# 
#     def readKeywordNDPSamples(self, n_dp_samples, i_line, input):
#         """Read information for a keyword belonging to the N_DP_SAMPLES group."""
# 
#         n_dp_samples = self.readValue(line)
#         i_line += 1
# 
#         return [n_dp_samples, i_line]
# 
#     def readKeywordMeshOptions(self, discret_file_ext, discret_spec_array, i_line, input):
#         """Read information for a keyword belonging to the N_DP_SAMPLES group."""
# 
#         i_line += 1
#         # Moving over the line containing "Mesh_Options"
#         while i_line < len(input):
#             line = input[i_line]
#             # New line
#             if line.strip() == '' or line.startswith("#"):
#             # if the line is empty or a comment move on to the next
#                 i_line += 1
#                 continue
#             if any([line.split()[0] == possible_keyword.name
#                     for possible_keyword in Keyword.all_general_keywords]):
#             # If another keyword has been specified exit the MIC_GEN_DESCRIPTORS
#             # block
#                 break
#             if line.strip().endswith('msh'):
#                 name_current_msh = line.strip().lower()
#                 discret_file_ext.append(name_current_msh)
#                 discret_spec_array[name_current_msh] = {}
#                 i_line += 1
#             else:
#                 parameter = line.split()[0].lower()
#                 value = line.split()[1:]
#                 if len(value) == 1:
#                     discret_spec_array[name_current_msh][parameter] = str2type(
#                         value[0])
#                 else:
#                     discret_spec_array[name_current_msh][parameter] = np.array(
#                         [str2type(val) for val in value])
#                 i_line += 1
# 
#         return [discret_file_ext, discret_spec_array, i_line]
# 
#     def currentLevelReader(self):
# 
#         current_line_keyword = False
#         while True:
#             for possible_keyword in current_keyword.top_level_keywords:
#             # Checking what is the current keyword
#                 if possible_keyword.isIn(line):
#                     current_line_keyword = True
#                     # General keyword has been found
#                     possible_keyword.readValue()
#                     self.current_keyword = possible_keyword
#                     # read the values corresponding to the keyword detected
#             if not current_line_keyword and False:
#                 print(line, 'does not contain a keyword')
#                 # FIXME: create an appropriate error
#                 quit()
# 
#     def readKeywordValues(self):
#         """
#         Read all the information from the input file regarding keyword *self*"""
# 
#         self.currentLevelReader()
#         if keyword.mandatory:
#             self.mandatory_top_level_keywords_not_set.remove(keyword)
#         # self.all_options
# 
# 
#     def MicGenDescriptorsReader(self):
# 
#         if any([keyword.isIn(line)
#                 for keyword in self.top_level_keywords]):
#         # If another top level keyword has been specified exit the MIC_GEN_DESCRIPTORS
#         # block
#             break
#         if keyword.line.startswith('Phase'):
#             name_current_phase = line.split()[1]
#             current_sim.mic_gen_descriptors[name_current_phase] = {}
#             self.i_line += 1
#         elif line.startswith('phase_type'):
#             current_sim.phase_types[name_current_phase] = int(line.split()[1])
#             self.i_line += 1
#         else:
#             parameter = line.split()[0]
#             value = line.split()[1]
#             current_sim.mic_gen_descriptors[name_current_phase][parameter] = \
#                 str2type(value)
#             self.i_line += 1
# 
# 
# 
#     def readInputFile(self, input_file_path):
#         # Possible keywords appearing in the input file
#         with open(input_file_path, 'r') as input:
#             self.input = input.readlines()
#             self.i_line = 0
#             # Initializing the line counter
#             while self.i_line < len(input):
#                 # Going through all the line in the input file
#                 line = input[self.i_line]
#                 # Saving current line as line
#                 if line.strip() == '' or line.startswith("#"):
#                 # if the line is empty or a comment move on to the next
#                     self.i_line += 1
#                     continue
#                 # self.readKeywordValues()
    def parametersChecks(self):
        pass

    def setOptionsSimulation(self, options):

        self.mic_gen_parameters = options.get('Mic_Gen_Parameters')
        self.problem_type = options.get('Problem_Type')
        self.n_dp_samples = options.get('N_DP_Samples')
        self.mic_gen_descriptors = options.get('Mic_Gen_Descriptors')
        self.mesh_options = options.get('Mesh_Options')

        self.parametersChecks()


def generateAllPossibleKeywords():

    # from main import Keyword

    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    all_particle_sub_classes = get_all_subclasses(Particle)
    all_phase_descriptor_sub_classes = get_all_subclasses(PhaseDescriptor)
    keyword_set = set()
    keyword_set.add(Keyword('vf', type='float'))
    keyword_set.add(Keyword('n', type='float'))
    for particle_type in all_particle_sub_classes:
        for descriptor in particle_type.possible_parameters:
            if descriptor == 'vf' or descriptor == 'n':
                continue
            keyword_set.add(Keyword(descriptor, type='float'))
            keyword_set.add(Keyword(descriptor + '_distribution', type='str'))
            for distribution in all_phase_descriptor_sub_classes:
                for parameter in distribution.parameters:
                    keyword_set.add(Keyword(
                        descriptor + "_" + parameter, type='float'))

    return keyword_set

if __name__ == '__main__':

    def str2type(value_option):
        '''Convert string containing a parameter value to the correct type.'''
        if value_option == 'True':
            return True
        elif value_option == 'False':
            return False
        else:
            try:
                value_option = int(value_option)
                return value_option
            except ValueError:
                pass
            try:
                value_option = float(value_option)
                return value_option
            except ValueError:
                pass
            return value_option

    if len(sys.argv) == 0:
        # No input file has been supplied
        print('No input file was supplied.')
        quit()
        # Exiting the script
    input_file_path = sys.argv[1]
    input_file_dir = os.path.dirname(sys.argv[1])
    input_file_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
    # Obtaining the directory and the name of the input file

    top_level_reader = TopLevelReader()
    top_level_reader.addTopLevelKeyword(
        KeywordTypeA('Max_Residue_Per_Particle', 'MIC_GEN_PARAMETERS', type='float'),
        KeywordTypeA('Max_Step', 'MIC_GEN_PARAMETERS', type='int'),
        KeywordTypeA('Max_Steps_To_Relax', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=0, type='int'),
        KeywordTypeA('Speed_Up_Scheme', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value='Cell', type='str'),
        KeywordTypeA('Verlet_Factor', 'MIC_GEN_PARAMETERS', mandatory=False,
                     type='float', parent_keyword=('Speed_Up_Scheme', 'Verlet')),
        KeywordTypeA('dt', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=0.05, type='float'),
        KeywordTypeA('Save_History', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=False, type='bool'),
        KeywordTypeA('Type_Initial_Configuration', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value='random', type='str'),
        KeywordTypeA('Motion_Analysis', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=False, type='bool'),
        KeywordTypeA('Thermostat', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_type='multi_temperature', type='str'),
        KeywordTypeA('Min_Distance', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=0, type='float'),
        KeywordTypeA('Initial_Temp', 'MIC_GEN_PARAMETERS', mandatory=False,
                     default_value=2.5e10, type='float'),
        KeywordTypeB('Problem_Type', 'PROBLEM_TYPE', type='int'),
        KeywordTypeB('N_DP_Samples', 'N_DP_SAMPLES', type='int'))

    top_level_reader.addTopLevelKeyword(
        KeywordTypeC('Mic_Gen_Descriptors',
                     header_keys={Keyword('PHASE')},
                     sub_keys={Keyword('phase_type', type='int'),
                               *generateAllPossibleKeywords()}))

    top_level_reader.addTopLevelKeyword(
        KeywordTypeC('Mesh_Options',
                     header_keys={
                        Keyword('Nomsh', type='none'),
                        Keyword('Femsh', type='none'),
                        Keyword('Rgmsh', type='none')},
                     sub_keys={Keyword('Element_Type', type='int')},
                     mandatory=False))

    top_level_reader.readInputFile(input_file_path)

    current_sim = Simulation(input_file_dir)

    # current_sim.setOptionsSimulation(Keyword.all_options)


    print(Keyword.all_options)

    # main(dp_dir, mic_gen_descriptors, phase_types, mic_gen_parameters, n_dp_samples,
    #      problem_type, discret_spec_array, discret_file_ext)
    # Executing the script for microstructure generation
