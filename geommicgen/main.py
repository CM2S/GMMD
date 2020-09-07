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
from particle_classes import (
    Disk,
    Particle,
    Ellipse,
    Sphere,
    Ellipsoid,
    CylindricalFiber,
    RVE,
    Phase,
    GeometricalParameter,
    PhaseDescriptor,
)

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
    """
    This function creates a new Verlet list for all the particles
    """
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
            pos_cell_list_dim.append(
                np.int(
                    np.floor(
                        particles[i_particle].position_center[j_dim]
                        / Particle.cell_side_length[j_dim]
                    )
                )
            )
            # j_dim-position of the particle in the grid
        if dim == 2:
            # 2D problem
            pos_cell_list = (
                pos_cell_list_dim[0] + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
            )
            # Saving the position in the cell list of particle i_particle
            for k_neighboor_cell in range(9):
                # Running through the neighboor cells
                pos_neighboor_cell = neighboorCell(
                    pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim
                )
                # Computing the index of the neighboor cell
                for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighboring cell
                    if particles[i_particle].intersectionVerlet(particles[j_particle]):
                        # If the neighboorhoods of the particles intersect
                        particles[i_particle].verlet_list.append(j_particle)
                        # Add the particle j_particle to i_particle's Verlet list
        elif dim == 3:
            # 3D problem
            pos_cell_list = (
                pos_cell_list_dim[0]
                + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
                + pos_cell_list_dim[2] * Particle.n_cell_dim[0] * Particle.n_cell_dim[1]
            )
            # Saving the position in the cell list of particle i_particle
            for k_neighboor_cell in range(3 ** 3):
                # Running through the neighboor cells
                pos_neighboor_cell = neighboorCell(
                    pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim
                )
                # Computing the index of the neighboor cell
                for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighboring cell
                    if particles[i_particle].intersectionVerlet(particles[j_particle]):
                        # If the neighboorhoods of the particles intersect
                        particles[i_particle].verlet_list.append(j_particle)
                        # Add the particle j_particle to i_particle's Verlet list


def neighboorCell(pos_current_cell, local_pos_neighboor_cell, dim, n_cells):
    """
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
    """

    if dim == 2:
        # 2D problem
        local_row_pos_neigh = np.int(
            np.mod(np.floor(local_pos_neighboor_cell / 3), 3) - 1
        )
        # Local row position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_col_pos_neigh = np.int(np.mod(local_pos_neighboor_cell, 3) - 1)
        # Local column position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        pos_neighboor_cell = np.int(
            pos_current_cell + local_col_pos_neigh + local_row_pos_neigh * n_cells[0]
        )
        # Global position of the neighboor cell without enforcing periodic boundary
        # conditions
        if pos_current_cell < n_cells[0] and local_row_pos_neigh == -1:
            # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1] * n_cells[0]
            # Enforcing the periodic boundary conditions
        elif (
            pos_current_cell >= n_cells[0] * (n_cells[1] - 1)
            and local_row_pos_neigh == 1
        ):
            # Upper row of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1] * n_cells[0]
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
        local_row_pos_neigh = np.int(
            np.mod(np.floor(local_pos_neighboor_cell / 3), 3) - 1
        )
        # Local row position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_col_pos_neigh = np.int(np.mod(local_pos_neighboor_cell, 3) - 1)
        # Local column position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_lay_pos_neigh = np.int(
            np.mod(np.floor(local_pos_neighboor_cell / 9), 3) - 1
        )
        # Local layer position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        pos_neighboor_cell = np.int(
            pos_current_cell
            + local_col_pos_neigh
            + local_row_pos_neigh * n_cells[0]
            + local_lay_pos_neigh * n_cells[0] * n_cells[1]
        )
        # Global position of the neighboor cell without enforcing periodic boundary
        # conditions
        if (
            pos_current_cell
            - n_cells[1] * n_cells[0] * (pos_current_cell // (n_cells[1] * n_cells[0]))
            < n_cells[0]
            and local_row_pos_neigh == -1
        ):
            # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1] * n_cells[0]
            # Enforcing the periodic boundary conditions
        elif (
            pos_current_cell
            - n_cells[1] * n_cells[0] * (pos_current_cell // (n_cells[1] * n_cells[0]))
            >= n_cells[0] * (n_cells[1] - 1)
            and local_row_pos_neigh == 1
        ):
            # Upper row of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1] * n_cells[0]
            # Enforcing the periodic boundary conditions
        if np.mod(pos_current_cell + 1, n_cells[0]) == 0 and local_col_pos_neigh == 1:
            # Right column of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[0]
            # Enforcing the periodic boundary conditions
        elif np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1:
            # Left column of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[0]
            # Enforcing the periodic boundary conditions
        if pos_current_cell < n_cells[1] * n_cells[0] and local_lay_pos_neigh == -1:
            # Firsl layer of the grid
            pos_neighboor_cell = (
                pos_neighboor_cell + n_cells[1] * n_cells[0] * n_cells[2]
            )
            # Enforcing the periodic boundary conditions
        elif (
            pos_current_cell > n_cells[1] * n_cells[0] * (n_cells[2] - 1) - 1
            and local_lay_pos_neigh == 1
        ):
            # Last layer of the grid
            pos_neighboor_cell = (
                pos_neighboor_cell - n_cells[1] * n_cells[0] * n_cells[2]
            )
            # Enforcing the periodic boundary conditions

    return pos_neighboor_cell


def newCellList(particles):
    """
    Compute a new cell list for particles.
    """

    dim = particles[0].dim

    n_cells = np.prod(np.array(Particle.n_cell_dim))

    Particle.cell_list = [[] for i in range(n_cells)]

    for i_particle in range(len(particles)):
        # Running through all the particles
        pos_cell_list_dim = []
        # Initializing the list containing the position of the cell in each direction
        # with the origin at the top left
        for j_dim in range(dim):
            # Running through all the dimensions
            pos_cell_list_dim.append(
                np.int(
                    np.floor(
                        particles[i_particle].position_center[j_dim]
                        / Particle.cell_side_length[j_dim]
                    )
                )
            )
            # j_dim-position of the particle in the grid
        if dim == 2:
            # 2D problem
            pos_cell_list = (
                pos_cell_list_dim[0] + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
            )
            # Saving the position in the cell list of particle i_particle
        if dim == 3:
            # 3D problem
            pos_cell_list = (
                pos_cell_list_dim[0]
                + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
                + pos_cell_list_dim[2] * Particle.n_cell_dim[0] * Particle.n_cell_dim[1]
            )
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
    if speed_up_scheme == "Naive":
        # Naive approach: O(N^2)
        for i_particle in range(len(particles)):
            # Running though all the particles
            for j_particle in range(i_particle + 1, len(particles)):
                # Running through the particle pairs that have not been considered yet
                force_i_j = computeForceij(particles[i_particle], particles[j_particle])
                # Computing the force on particle i due to particle j
                particles[i_particle].force = particles[i_particle].force + force_i_j
                # Adding the force due to the interaction between particle 1 and 2 to the
                # total force acting on particle 1
                particles[j_particle].force = particles[j_particle].force - force_i_j
                # Adding the force due to the interaction between particle 1 and 2 to the
                # total force acting on particle 2
    elif speed_up_scheme == "Cell":
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
                pos_cell_list_dim.append(
                    np.int(
                        np.floor(
                            particles[i_particle].position_center[j_dim]
                            / Particle.cell_side_length[j_dim]
                        )
                    )
                )
                # j_dim-position of the particle in the grid
            if dim == 2:
                # 2D problem
                pos_cell_list = (
                    pos_cell_list_dim[0] + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
                )
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(9):
                    # Running through the neighboor cells
                    pos_neighboor_cell = neighboorCell(
                        pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim
                    )
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                        # Running through all the particles in the neighboring cell
                        if j_particle > i_particle:
                            # Ensuring that the forces are not computed twice
                            force_i_j = computeForceij(
                                particles[i_particle], particles[j_particle]
                            )
                            # Computing the force on particle i due to particle j
                            particles[i_particle].force = (
                                particles[i_particle].force + force_i_j
                            )
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 1
                            particles[j_particle].force = (
                                particles[j_particle].force - force_i_j
                            )
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 2
            if dim == 3:
                # 2D problem
                pos_cell_list = (
                    pos_cell_list_dim[0]
                    + pos_cell_list_dim[1] * Particle.n_cell_dim[0]
                    + pos_cell_list_dim[2]
                    * Particle.n_cell_dim[0]
                    * Particle.n_cell_dim[1]
                )
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(3 ** 3):
                    # Running through the neighboor cells
                    pos_neighboor_cell = neighboorCell(
                        pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim
                    )
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                        # Running through all the particles in the neighboring cell
                        if j_particle > i_particle:
                            # Ensuring that the forces are not computed twice
                            force_i_j = computeForceij(
                                particles[i_particle], particles[j_particle]
                            )
                            # Computing the force on particle i due to particle j
                            particles[i_particle].force = (
                                particles[i_particle].force + force_i_j
                            )
                            # Adding the force due to the interaction between particle 1
                            # and 2 to the total force acting on particle 1
                            particles[j_particle].force = (
                                particles[j_particle].force - force_i_j
                            )
                            # Adding the force due to the interaction between particle 1 and
                            # 2 to the total force acting on particle 2
    elif speed_up_scheme == "Verlet":
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
                    force_i_j = computeForceij(
                        particles[i_particle], particles[j_particle]
                    )
                    # Computing the force on particle i due to particle j
                    particles[i_particle].force = (
                        particles[i_particle].force + force_i_j
                    )
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 1
                    particles[j_particle].force = (
                        particles[j_particle].force - force_i_j
                    )
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 2


# ==========================================================================================
def computeForceij(particle_i, particle_j):
    """
    Compute the force on particle_i due to particle_j
    """
    intersection_area = particle_i.intersectionArea(particle_j)
    # Intersection area between particle i and j
    particle_i.overlap_area += intersection_area
    particle_j.overlap_area += intersection_area
    Particle.total_overlap += intersection_area
    # Updating the overlap area
    unit_vector_i_j = particle_i.intersectionVector(particle_j)
    # Unit vector from particle i to particle j
    force_i_j = -intersection_area * unit_vector_i_j
    # Computing the force on particle_i due to particle_j proportional to their
    # intersection area/volume
    return force_i_j


def putSystemAtRest(particles):
    """Put the system as a whole at rest."""
    total_linear_momentum = np.sum(
        [particle.volume() * particle.velocity_center for particle in particles], axis=0
    )
    # Computing total linear momemtum of the system
    for i_particle in particles:
        # Running through all the particles
        i_particle.setVelocityCenter(i_particle.velocity_center - total_linear_momentum)
        # Removing the linear momentum of the system as a whole putting at rest


def integrate(particles, dt, speed_up_scheme, integration_scheme="Verlet", **kwargs):
    """Integrate the equations of motion."""
    dim = particles[0].dim
    # Dimension of the problem
    N = len(particles)
    # Number of particles
    box = Particle.box
    # Saving the size of the RVE
    for i_particle in range(N):
        # Running through all the particles
        if integration_scheme == "Newmark":
            # The integration scheme chosen was Newmark
            c = kwargs.get("damping_constant", 0)
            [new_position, new_velocity, new_accelaration] = Newmark(
                particles[i_particle].position_center,
                particles[i_particle].velocity_center,
                np.array([particles[i_particle].force], dtype="float").T,
                particles[i_particle].volume()
                * np.eye(
                    particles[i_particle].dim, dtype="float"
                ),  # 10e-6*np.eye(2,dtype='float'),#
                c * np.eye(particles[i_particle].dim, dtype="float"),
                np.zeros(
                    (particles[i_particle].dim, particles[i_particle].dim),
                    dtype="float",
                ),
                dt,
                1,
                dim,
            )
            # Obtaining the new position and velocity of particle i
        elif integration_scheme == "Verlet":
            # The integration scheme chosen was Verlet
            [new_position, new_velocity] = VerletSync(
                particles[i_particle].position_center,
                particles[i_particle].velocity_center,
                np.array([particles[i_particle].force], dtype="float").T,
                particles[i_particle].volume(),
                dt,
                1,
                dim,
            )
        if speed_up_scheme == "Verlet":
            particles[i_particle].displacement_last_verlet += (
                particles[i_particle].position_center - new_position[:, 0]
            )
            # Computing the displacement of the center of the particle
            if not particles[i_particle].insideVerlet():
                # Checking if the displacement takes the particle out of its neighboorhood
                Particle.new_verlet_list = True
                # There is a need to compute a new verlet list
        new_position[:, 0] = new_position[:, 0] - box * np.floor(
            new_position[:, 0] / box
        )
        # New position enforcing boundary conditions
        particles[i_particle].position_center = new_position[:, 0]
        particles[i_particle].velocity_center = new_velocity[:, 0]
        # Updating the position and velocity of particle i
        if kwargs.get("save_history"):
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
    used_parameters = {
        parameter
        for parameter in Disk.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Disk.acceptable_descriptions
        ]
    ):
        # Checking acceptable sets of parameters
        acceptable_description = True
    else:
        acceptable_description = False
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase.type, Disk.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of disks was specified
        phase.specNumber(descriptors["n"])
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            disks.append(Disk(phase, r[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Disk.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersDisk(current_sample, rve_dims)
            disks.append(Disk(phase, r[0]))
            vf_real += disks[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        # Obtaining the radius corresponding to the specified volume fraction and number of
        # particles
        for i in range(descriptors["n"]):
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
    used_parameters = {
        parameter
        for parameter in Sphere.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Sphere.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Sphere.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of disks was specified
        phase.specNumber(descriptors["n"])
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors["n"]):
            spheres.append(Sphere(phase, r[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Sphere.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersSphere(current_sample, rve_dims)
            spheres.append(Sphere(phase, r))
            vf_real += spheres[-1].volume() / (rve_dims[0] * rve_dims[1] * rve_dims[2])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersSphere(samples, rve_dims)
        for i in range(descriptors["n"]):
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
    used_parameters = {
        parameter
        for parameter in Ellipse.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Ellipse.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Ellipse.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of ellipses was specified
        phase.specNumber(descriptors["n"])
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
                n_samples=descriptors["n"],
            )
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipses.append(Ellipse(phase, major_axis[i], minor_axis[i], angle[i]))
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipse.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            [major_axis, minor_axis, angle] = canonicalParametersEllipse(
                current_sample, rve_dims
            )
            ellipses.append(Ellipse(phase, major_axis, minor_axis, angle))
            vf_real += ellipses[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
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
                n_samples=descriptors["n"],
            )
        [major_axis, minor_axis, angle] = canonicalParametersEllipse(samples, rve_dims)
        for i in range(descriptors["n"]):
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
    used_parameters = {
        parameter
        for parameter in Ellipsoid.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in Ellipsoid.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, Ellipsoid.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of ellipsoids was specified
        phase.specNumber(descriptors["n"])
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
                n_samples=descriptors["n"],
            )
        [
            axis_1,
            axis_2,
            axis_3,
            rot_axis_comp_x,
            rot_axis_comp_y,
            rot_axis_comp_z,
            angle,
        ] = canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[i],
                    axis_2[i],
                    axis_3[i],
                    rot_axis_comp_x[i],
                    rot_axis_comp_y[i],
                    rot_axis_comp_z[i],
                    angle[i],
                )
            )
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    Ellipsoid.possible_parameters[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            [
                axis_1,
                axis_2,
                axis_3,
                rot_axis_comp_x,
                rot_axis_comp_y,
                rot_axis_comp_z,
                angle,
            ] = canonicalParametersEllipsoid(current_sample, rve_dims)
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[0],
                    axis_2[0],
                    axis_3[0],
                    rot_axis_comp_x[0],
                    rot_axis_comp_y[0],
                    rot_axis_comp_z[0],
                    angle[0],
                )
            )
            vf_real += ellipsoids[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
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
                n_samples=descriptors["n"],
            )
        [
            axis_1,
            axis_2,
            axis_3,
            rot_axis_comp_x,
            rot_axis_comp_y,
            rot_axis_comp_z,
            angle,
        ] = canonicalParametersEllipsoid(samples, rve_dims)
        for i in range(descriptors["n"]):
            ellipsoids.append(
                Ellipsoid(
                    phase,
                    axis_1[i],
                    axis_2[i],
                    axis_3[i],
                    rot_axis_comp_x[i],
                    rot_axis_comp_y[i],
                    rot_axis_comp_z[i],
                    angle[i],
                )
            )

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
    used_parameters = {
        parameter
        for parameter in CylindricalFiber.possible_parameters
        if any([descriptor.startswith(parameter) for descriptor in descriptors.keys()])
    }
    # Collecting the parameters used
    if any(
        [
            used_parameters == acceptable_description
            for acceptable_description in CylindricalFiber.acceptable_descriptions
        ]
    ):
        acceptable_description = True
    else:
        acceptable_description = False
    # Checking acceptable sets of parameters
    try:
        if not acceptable_description:
            raise errors.UnacceptableParameters(
                used_parameters, phase, CylindricalFiber.acceptable_descriptions
            )
    except errors.UnacceptableParameters as error:
        error.message()
        quit()
    if "n" in descriptors and "vf" not in descriptors:
        # The desired number of fibers was specified
        phase.specNumber(descriptors["n"])
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            fibers.append(
                CylindricalFiber(phase, r[i], descriptors["direction"], rve_dims)
            )
    elif "vf" in descriptors and "n" not in descriptors:
        # The desired volume fraction was specfied
        phase.spec_volume_fraction = descriptors["vf"]
        # Collecting the specified volume fraction
        current_sample = {}
        # Initializing the dictionary containing the samples for each parameter used
        vf_real = 0
        # Initializing the real volume fraction
        while vf_real < descriptors["vf"]:
            for i_parameter in used_parameters:
                current_sample[i_parameter] = generateSampleParameter(
                    i_parameter,
                    CylindricalFiber.possible_parameter[i_parameter],
                    descriptors,
                    phase,
                    rve_dims,
                )
            r = canonicalParametersDisk(current_sample, rve_dims)
            fibers.append(
                CylindricalFiber(phase, r, descriptors["direction"], rve_dims)
            )
            vf_real += fibers[-1].volume() / (rve_dims[0] * rve_dims[1])
    elif "vf" in descriptors and "n" in descriptors:
        phase.spec_volume_fraction = descriptors["vf"]
        phase.spec_number = descriptors["n"]
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
                n_samples=descriptors["n"],
            )
        r = canonicalParametersDisk(samples, rve_dims)
        for i in range(descriptors["n"]):
            fibers.append(
                CylindricalFiber(phase, r, descriptors["direction"], rve_dims)
            )

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
    if type_init_conf == "random":
        # Random configuration for the particle centers and the zero velocity
        # np.random.seed(42)
        k = 0
        for i_particle in particles:
            k += 1
            # Running through all the particles
            i_particle.setPositionCenter(
                Particle.box * np.random.uniform(size=i_particle.dim)
            )
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(np.zeros((i_particle.dim)))
            # Generating the velocities from a random uniform distribution between -1 and 1
            i_particle.position_center_history = [i_particle.position_center.flatten()]
            # Saving initial configuration
    elif type_init_conf == "grid":
        # Particles randomly assigned to a place in a grid constructed to have an equal number
        # of cells in each direction and a total number of cells larger than the number of
        # particles
        if particles[0].dim == 3:
            n_cells_side = np.int(np.ceil(np.cbrt(len(particles))))
            # Number of cells in each direction
            cell_length = Particle.box / n_cells_side
            # Length of the cells in each direction
            k_counter = 0
            # Initializing the counter
            grid_places = np.arange(n_cells_side ** 3)
            # Label of each grid place
            np.random.shuffle(grid_places)
            # Distributing the particles randomly to different cells of the grid
            for j in range(n_cells_side):
                for k in range(n_cells_side):
                    for l in range(n_cells_side):
                        if grid_places[k_counter] < len(particles):
                            particles[grid_places[k_counter]].setPositionCenter(
                                np.array(
                                    [
                                        j * cell_length[0] + cell_length[0] / 2,
                                        k * cell_length[1] + cell_length[1] / 2,
                                        l * cell_length[2] + cell_length[2] / 2,
                                    ]
                                )
                            )
                            # Gene<><rating the positions from a random uniform distribution
                            # between 0 and 1
                            particles[grid_places[k_counter]].setVelocityCenter(
                                np.random.uniform(low=0.01, high=0.6, size=3)
                            )
                            # Generating the velocities from a random uniform distribution
                            # between -1 and 1
                            particles[
                                grid_places[k_counter]
                            ].position_center_history = [
                                particles[
                                    grid_places[k_counter]
                                ].position_center.flatten()
                            ]
                            # Saving particle history
                        k_counter += 1
        elif particles[0].dim == 2:
            n_cells_side = np.int(np.ceil(np.sqrt(len(particles))))
            # Number of cells in each direction
            cell_length = Particle.box / n_cells_side
            # Length of the cells in each direction
            k_counter = 0
            # Initializing the counter
            grid_places = np.arange(n_cells_side ** 2)
            # Label of each grid place
            np.random.shuffle(grid_places)
            # Distributing the particles randomly to different cells of the grid
            for j in range(n_cells_side):
                for k in range(n_cells_side):
                    if grid_places[k_counter] < len(particles):
                        particles[grid_places[k_counter]].setPositionCenter(
                            np.array(
                                [
                                    j * cell_length[0] + cell_length[0] / 2,
                                    k * cell_length[1] + cell_length[1] / 2,
                                ]
                            )
                        )
                        # Gene<><rating the positions from a random uniform distribution between 0 and 1
                        particles[grid_places[k_counter]].setVelocityCenter(
                            np.random.uniform(low=0.01, high=0.6, size=2)
                        )  # np.array([0,0],dtype='float')
                        # Generating the velocities from a random uniform distribution between -1 and 1
                        particles[grid_places[k_counter]].position_center_history = [
                            particles[grid_places[k_counter]].position_center.flatten()
                        ]
                        # Saving particle history
                    k_counter += 1
    elif type_init_conf == "fcc":
        center_points = np.array(
            [
                [0, 0, 0],
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
                [0.5, 1.5, 1],
            ]
        )
        k = 0
        for i_particle in particles:
            # Running through all the particles
            i_particle.setPositionCenter(
                center_points[k] / 2
            )  # np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(
                np.zeros((i_particle.dim))
            )  # np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get("save_history"):
                # Saving particle history
                i_particle.position_center_history = [
                    i_particle.position_center.flatten()
                ]
            k += 1
    elif type_init_conf == "overlap":
        k = 0
        for i_particle in particles:
            # Running through all the particles
            # i_particle.setPositionCenter(np.array([0.5 + 2*k*0.01, 0.5])) # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # # Generating the positions from a random uniform distribution between 0 and 1
            # i_particle.setVelocityCenter(np.array([1e-4 - 2*k*1e-4, 0])) #np.array([0,0],dtype='float')
            i_particle.setPositionCenter(
                np.array([0.5, 0.5])
            )  # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(
                np.array([0, 0])
            )  # np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get("save_history"):
                # Saving particle history
                i_particle.position_center_history = [
                    i_particle.position_center.flatten()
                ]
            k += 1
    elif type_init_conf == "custom":
        path = "/home/zeluis/Documents/Tese/programa/studies/thermostats/minkowski/artificial_2D/ord.txt"
        positions = np.loadtxt(path)
        for ind, i_particle in enumerate(particles):
            # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            i_particle.setPositionCenter(positions[ind, 0:2] / 500)
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(
                np.array([0, 0])
            )  # np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
            if kwargs.get("save_history"):
                # Saving particle history
                i_particle.position_center_history = [
                    i_particle.position_center.flatten()
                ]
    elif type_init_conf == "adjacent":
        k = 0
        for i_particle in particles:
            # Running through all the particles
            i_particle.setPositionCenter(
                np.array([0.1, 0.1, 0.01 + k * 0.98])
            )  # Particle.box*np.random.uniform(size=i_particle.dim)) #np.array([0.5, 0.87, 0.5])) # , (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            i_particle.setVelocityCenter(
                np.array([0, 0, 0])
            )  # np.array([0,0],dtype='float')
            if kwargs.get("save_history"):
                # Saving particle history
                i_particle.position_center_history = [
                    i_particle.position_center.flatten()
                ]
            k += 1
    else:
        try:
            raise errors.UnsupportedInitialConfigurationType(type_init_conf)
        except errors.UnsupportedInitialConfigurationType as error:
            error.message()
            quit()


def generateSampleParameter(
    parameter, parameter_name, descriptors, phase, rve_dims, n_samples=1, max_sample=100
):
    """Generate a sample of values for *parameter* according to descriptors"""
    size_geom_param = {"r", "major_axis", "minor_axis", "axis_1", "axis_2", "axis_3"}
    # Geometrical parameters related to the size of the particle that must larger than
    # ans smaller than half the size of the smallest dimension of the RVE
    if descriptors.get(parameter + "_distribution") == "uniform":
        # the radius follows a uniform distribution
        try:
            if parameter + "_low" not in descriptors:
                # Checking if the  lower bound was supplied
                raise errors.ParameterMissing(parameter + "_low", phase.type)
            elif parameter + "_high" not in descriptors:
                # Checking if the upper bound was supplied
                raise errors.ParameterMissing(parameter + "_high", phase.type)
            elif descriptors[parameter + "_low"] >= descriptors[parameter + "_high"]:
                # Checking if the lower bound is smaller than the upper bound
                raise errors.UnexpectedValue(
                    descriptors[parameter + "_low"],
                    "{0}_low of phase {1}".format(parameter, phase.type),
                    "smaller than "
                    + parameter
                    + "_high: {0}".format(descriptors[parameter + "_high"]),
                )
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
                        descriptors[parameter + "_low"],
                        "{0}_low of phase {1}".format(parameter, phase),
                        "larger than 0",
                    )
                elif descriptors[parameter + "_high"] > np.min(rve_dims) / 2:
                    # Ensuring that it will not produce values larger than half the size of the
                    # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter + "_high"],
                        "{0}_high of phase {1}".format(parameter, phase),
                        "smaller than half the smallest dimension of the RVE: {0}".format(
                            np.min(rve_dims) / 2
                        ),
                    )
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        phase.addGeomParameter(
            parameter_name,
            "Uniform",
            [
                ("Lower bound", descriptors[parameter + "_low"]),
                ("Upper bound", descriptors[parameter + "_high"]),
            ],
        )
        sample = np.random.uniform(
            low=descriptors[parameter + "_low"],
            high=descriptors[parameter + "_high"],
            size=n_samples,
        )
    elif descriptors.get(parameter + "_distribution") == "normal":
        # the radius follows a normal distribution: the paramaters 'r_sigma', the standard
        # deviation of the distribution and 'r_mean', the mean of the distribution, are needed
        try:
            if parameter + "_sigma" not in descriptors:
                raise errors.ParameterMissing(parameter + "_sigma", phase)
            elif parameter + "_mean" not in descriptors:
                raise errors.ParameterMissing(parameter + "_mean", phase)
        except errors.ParameterMissing as error:
            # One of the parameters is missing
            error.message()
            quit()
            # Printing message and aborting
        try:
            if parameter in size_geom_param:
                # Geometric size parameters
                low_25_prob = (
                    descriptors[parameter + "_mean"]
                    - 2 * descriptors[parameter + "_sigma"]
                )
                # Upper bound of tail with 2.5% probability
                high_25_prob = (
                    descriptors[parameter + "_mean"]
                    + 2 * descriptors[parameter + "_sigma"]
                )
                # Lower bound of tail with 2.5% probability
                if low_25_prob < 0:
                    # Ensuring that the probability of generating a value smaller than 0 is
                    # not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, "low")
                elif high_25_prob > np.min(rve_dims) / 2:
                    # Ensuring that the probability of generating a value larger than half the
                    # size of the smallest dimension of the RVE is not greater than 2.5%
                    raise errors.DangerousValueNormal(parameter, phase, "high")
        except errors.DangerousValueNormal as error:
            error.message()
            quit()
        k_sample = 0
        acceptable_values = False
        while k_sample < max_sample and not acceptable_values:
            phase.addGeomParameter(
                parameter_name,
                "Normal",
                [
                    ("Mean", descriptors[parameter + "_mean"]),
                    ("Std Var", descriptors[parameter + "_sigma"]),
                ],
            )
            sample = np.random.normal(
                loc=descriptors[parameter + "_mean"],
                scale=descriptors[parameter + "_sigma"],
                size=n_samples,
            )
            # Generate a sample
            if parameter in size_geom_param:
                # Geometric size parameters
                if all(
                    [
                        (i_sample > 0 and i_sample <= np.min(rve_dims) / 2)
                        for i_sample in sample
                    ]
                ):
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
    elif descriptors.get(parameter + "_distribution") == "discrete":
        # the radius follows a discrete distribution
        values = []
        probabilities = []
        # Initializing the lists containing the values and corresponding probabilities that
        # characterize the discrete distribution required
        for i_descriptor in descriptors:
            if i_descriptor.startswith(parameter + "_value"):
                try:
                    if parameter in size_geom_param:
                        # Checking if the parameter is a size parameter
                        if descriptors[i_descriptor] < 0:
                            # Ensuring that it will not produce values smaller than 0
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor],
                                "{0} of phase {1}".format(i_descriptor, phase),
                                "larger than 0",
                            )
                        elif descriptors[i_descriptor] > np.min(rve_dims) / 2:
                            # Ensuring that it will not produce values larger than half the size
                            # of the smallest dimension of the RVE
                            raise errors.UnexpectedValue(
                                descriptors[i_descriptor],
                                "{0} of phase {1}".format(i_descriptor, phase),
                                "smaller than half the smallest dimension"
                                + "of the RVE: {0}".format(np.min(rve_dims) / 2),
                            )
                except errors.UnexpectedValue as error:
                    error.message()
                    quit()
                values.append(descriptors[i_descriptor])
                # Save the value
                try:
                    if parameter + "_prob_" + i_descriptor[-1] not in descriptors:
                        raise errors.ParameterMissing(
                            parameter + "_prob_" + i_descriptor[-1], phase
                        )
                    probabilities.append(
                        descriptors[parameter + "_prob_" + i_descriptor[-1]]
                    )
                except errors.ParameterMissing as error:
                    error.message()
                    quit()
        if len(values) == 0:
            # There are no values for the radius parameter
            try:
                raise errors.ParameterMissing(parameter + "value_1", phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        elif np.abs(np.sum(probabilities) - 1) > 0.01:
            # The probabilities do not add up to 100%
            try:
                raise errors.ParameterErrorDiscreteProb("r_1", phase)
            except errors.ParameterMissing as error:
                error.message()
                quit()
        value_prob_pairs = [
            [
                ("Value {0}".format(ind + 1), val),
                ("Proability {0}".format(ind + 1), prob),
            ]
            for (ind, val), prob in zip(enumerate(values), probabilities)
        ]
        value_prob_pairs_flat = [
            item for sublist in value_prob_pairs for item in sublist
        ]
        phase.addGeomParameter(parameter_name, "Discrete", value_prob_pairs_flat)
        sample = np.random.choice(values, n_samples, p=probabilities)
    elif parameter + "_distribution" in descriptors:
        # A distribution was specified but it is not supported
        try:
            raise errors.UnsupportedDistribution(
                descriptors[parameter + "_distribution"], parameter, phase
            )
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
                        descriptors[parameter],
                        "{0} of phase {1}".format(parameter, phase),
                        "larger than 0",
                    )
                elif descriptors[parameter] > np.min(rve_dims) / 2:
                    # Ensuring that it will not produce values larger than half the size of the
                    # smallest dimension of the RVE
                    raise errors.UnexpectedValue(
                        descriptors[parameter],
                        "{0} of phase {1}".format(parameter, phase),
                        "smaller than half the smallest dimension of the RVE: {0}".format(
                            np.min(rve_dims) / 2
                        ),
                    )
        except errors.UnexpectedValue as error:
            error.message()
            quit()
        if parameter != "n" and parameter != "vf":
            phase.addGeomParameter(
                parameter_name, "Fixed", ("Value", descriptors[parameter])
            )
        sample = np.full((n_samples), descriptors[parameter])

    return sample


def canonicalParametersEllipse(sample, rve_dims):
    """Convert the paramters in *sample* to *major_axis*, *minor_axis* and *angle*."""
    if "major_axis" in sample and "minor_axis" in sample:
        # Both major and minor axis were supplied
        major_axis = np.max([sample["major_axis"], sample["minor_axis"]], axis=0)
        minor_axis = np.min([sample["major_axis"], sample["minor_axis"]], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    elif "major_axis" in sample and "vf" in sample and "n" in sample:
        # The major_axis, the volume faction and the number of particles were supplied
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        aux_minor_axis = volume_part / (np.pi * sample["major_axis"] * 1 / 4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([sample["major_axis"], aux_minor_axis], axis=0)
        minor_axis = np.min([sample["major_axis"], aux_minor_axis], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    # FIXME: Warnign that all particles will have the same volume
    elif "minor_axis" in sample and "vf" in sample and "n" in sample:
        # The minor axis, the volume faction and the number of particles were supplied
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        aux_major_axis = volume_part / (np.pi * sample["minor_axis"] * 1 / 4)
        # Minor axis computed assuming that all particles have the same area
        major_axis = np.max([aux_major_axis, sample["minor_axis"]], axis=0)
        minor_axis = np.min([aux_major_axis, sample["minor_axis"]], axis=0)
        # Ensuring that the major axis is greater than the minor axis
    elif "ratio" in sample and "vf" in sample and "n" in sample:
        volume_part = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        minor_axis = np.sqrt(volume_part / (np.pi * sample["ratio"] * 1 / 4))
        major_axis = sample["ratio"] * minor_axis
    if "angle" in sample:
        angle = sample["angle"]

    return [major_axis, minor_axis, angle]


def canonicalParametersDisk(sample, rve_dims):
    """Convert the paramters in *sample* to *r*."""
    if "r" in sample:
        # The radius was supplied
        r = sample["r"]
    elif "area" in sample:
        # The area of each particle was supplied
        r = np.sqrt(sample["area"] / np.pi)
    elif "vf" in sample and "n" in sample:
        # Both the volume fraction and the number of particles was supplied
        area = sample["vf"][0] * rve_dims[0] * rve_dims[1] / sample["n"][0]
        # Area of each particle (all the same)
        r = np.sqrt(area / np.pi)
    return r


def canonicalParametersSphere(sample, rve_dims):
    """Convert the parameters in *sample* to *r* characterizing a sphere."""
    if "r" in sample:
        # The radius was supplied
        r = sample["r"]
    elif "volume" in sample:
        # The area of each particle was supplied
        r = np.cbrt(sample["volume"] / (4 / 3 * np.pi))
    elif "vf" in sample and "n" in sample:
        # Both the volume fraction and the number of particles was supplied
        volume = (
            sample["vf"][0] * rve_dims[0] * rve_dims[1] * rve_dims[2] / sample["n"][0]
        )
        # Area of each particle (all the same)
        r = np.cbrt(volume / (4 / 3 * np.pi))
    return r


def canonicalParametersEllipsoid(sample, rve_dims):
    """Convert parameters in *sample* to canonical params characterizing an Ellipsoid."""
    if "axis_1" in sample and "axis_2" in sample and "axis_3":
        # All axis were supplied
        axis_1 = sample["axis_1"]
        axis_2 = sample["axis_2"]
        axis_3 = sample["axis_3"]
    if (
        "ratio_12" in sample
        and "ratio_13" in sample
        and "vf" in sample
        and "n" in sample
    ):
        volume = sample["vf"] * rve_dims[0] * rve_dims[1] * rve_dims[2] / sample["n"]
        axis_1 = np.cbrt(
            volume * sample["ratio_12"] * sample["ratio_13"] * 8 / (np.pi * 4 / 3)
        )
        axis_2 = axis_1 / sample["ratio_12"]
        axis_3 = axis_1 / sample["ratio_13"]
        print(
            "axis",
            axis_1,
            axis_2,
            axis_3,
            "volume",
            sample["n"] * 4 / 3 * axis_1 * axis_2 * axis_3 / 8 * np.pi,
            sample["vf"],
            sample["n"],
        )
    if "angle" in sample:
        angle = sample["angle"]
    if (
        "rot_axis_comp_x" in sample
        and "rot_axis_comp_y" in sample
        and "rot_axis_comp_z" in sample
    ):
        # Euler angles
        rot_axis_comp_x = sample["rot_axis_comp_x"]
        rot_axis_comp_y = sample["rot_axis_comp_y"]
        rot_axis_comp_z = sample["rot_axis_comp_z"]

    return [
        axis_1,
        axis_2,
        axis_3,
        rot_axis_comp_x,
        rot_axis_comp_y,
        rot_axis_comp_z,
        angle,
    ]


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
    results_folder = os.path.join(dp_dir, Particle.file_name)
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
        shutil.copy(
            "input_data\\info_micro.p", os.path.join(results_folder, "info_micro.p")
        )
        # copying input file
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    # Saving the file path in the Particle class


def particleGeneration(
    descriptors,
    phase_types,
    rve_dims,
    problem_type,
    dp_dir,
    type_init_conf,
    save_history=True,
):
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
    Particle.phases = {
        i_phase: Phase(i_phase, phase_types[i_phase]) for i_phase in descriptors
    }
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
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Disks", 2, 3, i_phase.name
                    )
                particles = particles + generateDisks(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of disks requested and appending them to the list of
                # particles
            elif i_phase.type == 3:
                # This phase is made up by ellipses
                if len(rve_dims) != 2:
                    # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Ellipses", 2, 3, i_phase.name
                    )
                particles = particles + generateEllipses(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of ellipses requested and appending them to the list
                # of particles
            elif i_phase.type == 4:
                # This phase is made up by spheres
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Spheres", 3, 2, i_phase.name
                    )
                particles = particles + generateSpheres(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of spheres requested and appending them to the list
                # of  particles
            elif i_phase.type == 5:
                # This phase is made up by ellipsoids
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Ellipsoids", 3, 2, i_phase.name
                    )
                particles = particles + generateEllipsoids(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of ellipsoids requested and appending them to the
                # list of particles
            elif i_phase.type == 6:
                # This phase is made up by cylindrical fibers
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Cylindrical Fibers", 3, 2, i_phase
                    )
                if any(
                    [
                        i_phase.type != 1 and i_phase.type != 6
                        for i_phase.type in list(phase_types.values())
                    ]
                ):
                    raise errors.OnlyCylindricalFibers()
                particles = particles + generateCylindricalFibers(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of cylindrical fibers requested and appending them
                # to the list of particles
            else:
                raise errors.UnsupportedPhaseType(i_phase.type, i_phase.name)
        except (
            errors.IncompatibleDimensionsRVEphase,
            errors.OnlyCylindricalFibers,
        ) as error:
            error.message()
            quit()

    print_funcs.printToFile("**PHASE DESCRIPTORS**\n")
    for i_phase in Particle.phases.values():
        # Running through all the phases to print their info
        i_phase.printSpecDescriptors()
        i_phase.printRealDescriptors()
    print_funcs.printToFile("=" * 80)

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
    info_dict = pickle.load(open("input_data\\info_micro.p", "rb"))
    # Loading the dictionary containing the information about the microstructure and its
    # generation
    dp_dir = info_dict.get("dp_dir")
    # Directory where the microstructure spatial discretization file(s) associated
    # with the given design point are to be stored
    options = info_dict.get("mic_gen_parameters")
    # An array which contains all the required parameters (or options)
    # for the selected program to generate the microstructure(s) and
    # and associated discretization file(s) of a given design point
    problem_type = info_dict.get("problem_type")
    # Getting the problem type
    n_dp_samples = info_dict.get("n_dp_samples", 1)
    # Number of samples to be generated using the descriptors supplied
    try:
        if not isinstance(n_dp_samples, int) or n_dp_samples < 1:
            # The number of samples must be an integer larger or equal to 1
            raise errors.NumberSamples(n_dp_samples)
    except errors.NumberSamples() as error:
        error.message()
        quit()

    descriptors = info_dict.get("mic_gen_descriptors", {})
    # mic_gen_descriptors_array: dictionary

    phase_types = info_dict.get("phase_types", {})
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
                raise errors.UnexpectedValue(
                    phase, "key of phase_types", "string containing an integer"
                )
    except errors.UnexpectedValue as error:
        error.message()
        quit()

    discret_file_ext = info_dict.get("discret_file_ext", {})
    # Saving the list containing the meshes required
    discret_spec_array = info_dict.get("discret_spec_array", {})
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
        rve_dims_spec.append(tuple(discret_spec_array[ext]["rve_dims"]))
        # Collecting the RVE dimensions specified
    rve_dims_spec = set(rve_dims_spec)
    # Obtaining the unique RVE size specifications
    if len(rve_dims_spec) > 1:
        # There are multiple RVE size specifications
        print_funcs.printToFile(
            "Warning: Different RVE sizes in the mesh specifications."
        )
        rve_dims = np.array(list(rve_dims_spec)[0])
        # Keeping the first
    else:
        rve_dims = np.array(list(rve_dims_spec)[0])

    return [
        dp_dir,
        descriptors,
        phase_types,
        options,
        n_dp_samples,
        rve_dims,
        problem_type,
        discret_spec_array,
        discret_file_ext,
    ]


def computeRelativeEnergy(particles):
    N = Particle.number
    norm_force_vec = np.array(
        [np.linalg.norm(particles[i].force) for i in range(N)], dtype="float"
    )
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    Particle.relative_energy_history.append(relative_energy)
    # Saving the relative energy

    return relative_energy


def computeKineticEnergy(particles):
    # Obtaining a list with the norms of the vector forces
    kin_energy = np.sum(
        [
            i_particle.volume() * np.sum(i_particle.velocity_center ** 2)
            for i_particle in particles
        ]
    )
    Particle.kinetic_energy_history.append(kin_energy)
    # Saving the kinetic energy

    return kin_energy


def forceOutTangentWall(particles, min_distance):
    tol = 0.5 * min_distance
    if particles[0].dim == 2:
        for i_particle in particles:
            pos = i_particle.position_center
            if np.abs(i_particle.radius - min_distance - pos[0]) < tol:
                i_particle.position_center += np.array([1e-2, 0])
            elif (
                np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance)
                < tol
            ):
                i_particle.position_center += np.array([-1e-2, 0])
            elif np.abs(i_particle.radius - min_distance - pos[1]) < tol:
                i_particle.position_center += np.array([0, 1e-2])
            elif (
                np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance)
                < tol
            ):
                i_particle.position_center += np.array([0, -1e-2])
    elif particles[0].dim == 3:
        for i_particle in particles:
            pos = i_particle.position_center
            if np.abs(i_particle.radius - min_distance - pos[0]) < tol:
                i_particle.position_center += np.array([1e-2, 0, 0])
            elif (
                np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance)
                < tol
            ):
                i_particle.position_center += np.array([-1e-2, 0, 0])
            elif np.abs(i_particle.radius - min_distance - pos[1]) < tol:
                i_particle.position_center += np.array([0, 1e-2, 0])
            elif (
                np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance)
                < tol
            ):
                i_particle.position_center += np.array([0, -1e-2, 0])
            elif np.abs(i_particle.radius - min_distance - pos[2]) < tol:
                i_particle.position_center += np.array([0, 0, 1e-2])
            elif (
                np.abs(pos[2] - Particle.box[2] + i_particle.radius - min_distance)
                < tol
            ):
                i_particle.position_center += np.array([0, 0, -1e-2])


def checkTangentToWall(particles, min_distance):
    tol = 0.5 * min_distance
    not_tangent_to_wall = True
    print("tol", tol)
    if particles[0].dim == 2:
        for i_particle in particles:
            pos = i_particle.position_center
            # print('tol', tol, 'radius', i_particle.radius)
            if (
                np.abs(i_particle.radius - min_distance - pos[0]) < tol
                or np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance)
                < tol
                or np.abs(i_particle.radius - min_distance - pos[1]) < tol
                or np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance)
                < tol
            ):
                not_tangent_to_wall = False
    elif particles[0].dim == 3:
        for i_particle in particles:
            pos = i_particle.position_center
            # print('tol', tol, 'radius', i_particle.radius)
            if (
                np.abs(i_particle.radius - min_distance - pos[0]) < tol
                or np.abs(pos[0] - Particle.box[0] + i_particle.radius - min_distance)
                < tol
                or np.abs(i_particle.radius - min_distance - pos[1]) < tol
                or np.abs(pos[1] - Particle.box[1] + i_particle.radius - min_distance)
                < tol
                or np.abs(i_particle.radius - min_distance - pos[2]) < tol
                or np.abs(pos[2] - Particle.box[2] + i_particle.radius - min_distance)
                < tol
            ):
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
    min_distance = options.get("min_distance", 0)
    # Saving the minimum distance
    speed_up_scheme = options.get("speed_up_scheme", "Cell")
    # What is the speed up scheme to be used
    max_steps_to_relax = options.get("max_steps_to_relax", 100)
    # Maximum number of iterations
    dt = options.get("dt", 0.05)
    # Time integration step
    thermostat = options.get("thermostat", "multi_temperature")
    # Thermostat to be used
    save_history = options.get("save_history", True)
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
    if speed_up_scheme == "Cell":
        # Only a cell list scheme will be used
        max_radius = np.max(np.array([particles[i].radius for i in range(N)]))
        # Saving the maximum radius of the circunscribing disk/sphere
        Particle.n_cell_dim = [
            np.int(np.round(box[i_dim] / (2 * max_radius))) for i_dim in range(dim)
        ]
        # Obtaining a list containing the number of cells in each direction
        n_cells = np.prod(Particle.n_cell_dim)
        # Obtaining the total number of cells
        Particle.cell_list = [[] for i in range(n_cells)]
        # Initializing the cell list
        Particle.cell_side_length = [
            box[i_dim] / Particle.n_cell_dim[i_dim] for i_dim in range(dim)
        ]
        # Obtaining a list containing the dimensions of the cell in each direction
    elif speed_up_scheme == "Verlet":
        # A Verlet list combined with a cell list scheme will be used
        Particle.verlet_factor = options["verlet_factor"]
        # Saving the Verlet radius to compute the Verlet list
        Particle.new_verlet_list = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        max_radius = (
            np.max(np.array([particles[i].radius for i in range(Particle.number)]))
            * Particle.verlet_factor
        )
        # Saving the maximum radius of the circunscribing disk/sphere accounting for the
        # Verlet factor
        Particle.n_cell_dim = [
            np.int(np.round(box[i_dim] / (2 * max_radius))) for i_dim in range(dim)
        ]
        # Obtaining a list containing the number of cells in each direction
        n_cells = np.prod(Particle.n_cell_dim)
        # Obtaining the total number of cells
        Particle.cell_list = [[] for i in range(n_cells)]
        # Initializing the cell list
        Particle.cell_side_length = [
            box[i_dim] / Particle.n_cell_dim[i_dim] for i_dim in range(dim)
        ]
        # Obtaining a list containing the dimensions of the cell in each direction
    else:
        # A naive approach will be used
        pass
    n_steps_relax = 0
    # Initializing the number of steps that a microstructure was complying with the
    # maximum overlap residue
    max_residue = max_residue_per_particle * N
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
    if thermostat == "multi_temperature":
        # The thermostat used is the isokinetic scheme
        # Setting the options
        if particles[0].dim == 2:
            jump = options.get("equilibration_steps", 25)
            # Number of steps allowed for the system to equilibrate and explore and given
            # temperature before the criterion for temperature lowering is checked
        elif particles[0].dim == 3:
            jump = options.get("equilibration_steps", 25)
        jump_list = []
        last_alt = options.get("inital_temp_steps", 40)
        # Number of steps allowed for the system to equilibrate and explore the initial
        # temperature
        T_ref = options.get("initial_temp", 2.5e10)  # *(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        if kin_energy > 1e-10:
            # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
        else:
            # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities
    elif thermostat == "isokinetic":
        T_ref = options.get("initial_temp", 2.5e10)  # *(particles[0].radius/0.045)**2)
        # Intial temperature
        k_b = 1e-15
        # Analog to the Boltzmann constant
        jump = options.get(
            "equilibration_steps", 25
        )  # + 5*100*0.65/(Particle.number*Particle.volume/Particle.volume_RVE))
        # Number of steps allowed for the system to equilibrate and explore and given
        # temperature before the criterion for temperature lowering is checked
        if kin_energy > 1e-10:
            # Compute the rescaling factor only if the kinetic energy is nonzero
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
            # Rescalling factor (why? 250 -  equipartition theorem)
            print("T_ref", T_ref)
        else:
            # If the kinetic energy is zero
            lambda_vel = 0
        for i_particle in range(N):
            # Running through all the particles
            particles[i_particle].velocity_center *= lambda_vel
            # Rescalling the velocities
    print_funcs.printToTerminalRefresh(
        step,
        Particle.total_overlap,
        relative_energy,
        kin_energy,
        temp=T_ref,
        first=True,
    )
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
        if thermostat == "multi_temperature":
            # The thermostat used is the multi_temperature scheme
            if step > last_alt:
                # If the end of the equilibration time has been reached
                if Particle.total_overlap > max_residue:
                    # If a legal configuration has not been achieved
                    if any(
                        np.array(Particle.total_overlap_history[-jump // 2 :])
                        - np.array(Particle.total_overlap_history[-jump // 2 - 1 : -1])
                        > 0
                    ):
                        # If the total overlap has increase in the previous iterations
                        T_ref *= 1 / 4
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
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
            # Rescalling factor
            for i_particle in range(N):
                # Running through all the particles
                particles[i_particle].velocity_center *= lambda_vel
                # Rescalling the velocities
            if (
                relative_energy / Particle.total_overlap < 1e-8
                and Particle.total_overlap > max_residue
            ):
                # FIXME: this criterion is giving false positives, relative energy falls
                # much faster than total overlap
                pass
        if thermostat == "isokinetic":
            # The thermostate used is the isokinetic with constant temperature
            lambda_vel = np.sqrt(2 * particles[0].dim * N * k_b * T_ref / kin_energy)
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
            step, Particle.total_overlap, relative_energy, kin_energy
        )
        if step > 5 * jump and all(
            (
                np.abs(
                    np.array(Particle.total_overlap_history[-5 * jump :])
                    - np.array(Particle.total_overlap_history[-5 * jump - 1 : -1])
                )
            )
            / np.array(Particle.total_overlap_history[-5 * jump - 1 : -1])
            * 100
            < 1e-5
        ):
            print_funcs.printToFile("Failed sample")
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
            i_particle.position_center_history.append(
                i_particle.position_center.flatten()
            )
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
