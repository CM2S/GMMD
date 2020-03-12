
import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint


def computeForces(particles):
    '''
    This function computes the forces between all the particle pairs in the system
    '''

    for i_particle in range(len(particles))
    # Running through all the particles
        particles[i_particle].cleanForces
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
    for i_particle in range(len(particles)):
    # Running though all the particles
        for j_particle in range(i+1, len(particles)):
        # Running through the particle pairs that have not been considered yet
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
    unit_vector_i_j = particles_i.intersectionVector(particle_j)
    # Unit vector from particle i to particle j
    force_i_j = -intersection_area*unit_vector_i_j
    # Computing the force on particle_i due to particle_j proportional to their
    # intersection area/volume
    return force_i_j

def integrate(particles, dt):
    '''
    This function integrates the equations of motion
    '''
    dim = particles[i_particle].dim
    # Dimension of the problem
    c = 0.1
    # Damping constant of the system
    for i_particle in range(N):
    # Running through all the particles
        [new_position, new_velocity] =
            Newmark(particles[i_particle].position_center,
            particles[i_particle].velocity_center,
            particles[i_particle].force,
            particles[i_particle].area,
            c,
            0,
            dt,
            1,
            dim)
        # Obtaining the new position and velocity of particle i
        particles[i_particle].position_center = new_position
        particles[i_particle].velocity_center = new_velocity
        # Updating the position and velocity of particle i
# ==========================================================================================




# ==========================================================================================

# ==========================================================================================


# ==========================================================================================
def particleGeneration(**kwargs):
    '''
    Function that generates all the particles from the geomtrical descriptors.

    '''

    particles[0] = Disk(0.5)
    # Disk with radius 0.5
    particles[1] = Disk(0.5)
    # Disk with radius 0.5

    dim = particles[0].dim
    # Saving the particles dimension

    particles[0].position_center = np.random.uniform(size=dim)
    particles[1].position_center = np.random.uniform(size=dim)
    # Generating the positions from a random uniform distribution between 0 and 1

    partices[0].velocity_center = np.random.uniform(low=-1,high=1,size=dim)
    partices[1].velocity_center = np.random.uniform(low=-1,high=1,size=dim)
    # Generating the velocities from a random uniform distribution between -1 and 1

    return paritcles
# ==========================================================================================

def readDescriptors():

    decriptors = {dim: 2}

    return decriptors

def run(particles, **kwargs):
    '''
    Main function of the Molecular Dynamics simulation.

    This is the main function of the Molecular Dynamics simulation. It consists of the 
    initialization of the sytem, and the loop that contains the dynamics of the system: 
    computation of the forces and integration of the equations of motion.

    Parameters:
        particles : array
            Array containing the Particle objects to be placed inside the RVE

    Returns:

    Other Parameters:

    '''

    step = 0
    # Initializing the the time step at 0
    dt = 0.01
    # Setting the time step size
    N = len(particles)
    # Number of particles

    computeForces(particles)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    norm_force_vec = [np.linalg.norm(particles[i].force) for i in range(N)]
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy

    while relative_energy >= max_residue:
    # Run the simulation while the overlap is larger than the allowed maximum residue
        integrate(particles, dt)
        # Integrating the equations of motion
        step += 1
        # Moving to the next time step
        computeForces(particles)
        # Computing the forces on all particles



    # Integrating Newton's equations of motion

if __name__ == '__main__':

    descriptors = readDescriptors()
    # Reading the descriptors

    particles = [];
    # Initializing the list of particles

    particles = particleGeneration(descriptors)
    # Generating the list of particles from the geometrical descriptors

    output = run(particles)
