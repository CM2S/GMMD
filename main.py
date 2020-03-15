
import numpy as np

import matplotlib.pyplot as plt

from integration_methods import Newmark

from particle_classes import Disk, Particle

def computeForces(particles):
    '''
    This function computes the forces between all the particle pairs in the system
    '''

    for i_particle in range(len(particles)):
    # Running through all the particles
        particles[i_particle].cleanForces()
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
    for i_particle in range(len(particles)):
    # Running though all the particles
        for j_particle in range(i_particle+1, len(particles)):
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
    unit_vector_i_j = particle_i.intersectionVector(particle_j)
    # Unit vector from particle i to particle j
    force_i_j = -intersection_area*unit_vector_i_j
    # Computing the force on particle_i due to particle_j proportional to their
    # intersection area/volume
    return force_i_j

def integrate(particles, dt):
    '''
    This function integrates the equations of motion
    '''
    dim = particles[1].dim
    # Dimension of the problem
    N = len(particles)
    c = N*1e-2
    # Damping constant of the system

    box = np.array([1,1])
    for i_particle in range(N):
    # Running through all the particles
        [new_position, new_velocity, new_accelaration] = \
            Newmark(particles[i_particle].position_center,
            particles[i_particle].velocity_center,
            Particle.global_force_factor*np.array([particles[i_particle].force],dtype='float').T,
            10e-4*np.eye(2,dtype='float'),
            c*np.eye(2,dtype='float'),
            np.zeros((2,2),dtype='float'),
            dt,
            1,
            dim)

        # Obtaining the new position and velocity of particle i
        
        particles[i_particle].position_center = new_position[:,0] -box*np.floor(new_position[:,0]/box)
        particles[i_particle].velocity_center = new_velocity[:,0]
        
        # Updating the position and velocity of particle i

# ==========================================================================================

def particleGeneration(*args):
    '''
    Function that generates all the particles from the geomtrical descriptors.

    '''

    particles = []

    dim = 2

    N = 40
    # Number of particles

    for i in range(N):
        particles.append(Disk(np.random.uniform(low=0.05,high=0.1)))
        # Disk with radius 0.5
        particles[i].position_center = np.random.uniform(size=dim) #np.array([0.5+i/50, 0.5-i/50]) #
        # Generating the positions from a random uniform distribution between 0 and 1
        particles[i].velocity_center = np.array([0,0],dtype='float')
        # Generating the velocities from a random uniform distribution between -1 and 1

    return particles
# ==========================================================================================

def readDescriptors():

    decriptors = {'dim': 2}

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
    max_residue = 10e-12
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0
    dt = 0.05
    # Setting the time step size
    N = len(particles)
    # Number of particles

    Particle.global_force_factor = 4
    # Initializing the global force factor
    computeForces(particles)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    relative_energy_old = relative_energy


    while (relative_energy >= max_residue) and (step<1000):
    # Run the simulation while the overlap is larger than the allowed maximum residue
        integrate(particles, dt)
        # Integrating the equations of motion
        step += 1
        # Moving to the next time step
        computeForces(particles)
        # Computing the forces on all particles
        norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
        # Obtaining a list with the norms of the vector forces
        relative_energy = norm_force_vec.dot(norm_force_vec)
        print('new',relative_energy)
        # Computing the relative energy
        if relative_energy != 0:
            if relative_energy_old/relative_energy > 2:
            # If the relative energy has decrease by a factor of two in this  iteraton
                relative_energy_old = relative_energy
                # Saving the value of the previous relative energy
                Particle.global_force_factor *= 2
                # Double the global factor multiplying the forces
            elif relative_energy/relative_energy_old > 2:
                Particle.global_force_factor *= 0.5
                relative_energy_old = relative_energy
                # Saving the value of the previous relative energy

        # Particle.global_force_factor = 1/relative_energy/100

        # fig = plt.figure()
        # 
        # ax = plt.gca()
        # 
        # N = len(particles)
        # 
        # for i in range(N):
        #     for j in range(-1,2):
        #         for k in range(-1,2):
        #             circ = mpatches.Circle(
        #                 particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius, alpha=0.5)
        #             ax.add_artist(circ)
        # plt.show(block=False)



    # Integrating Newton's equations of motion

if __name__ == '__main__':

    import matplotlib.patches as mpatches

    descriptors = readDescriptors()
    # Reading the descriptors

    particles = [];
    # Initializing the list of particles

    particles = particleGeneration(descriptors)
    # Generating the list of particles from the geometrical descriptors

    fig = plt.figure()

    ax = plt.gca()

    N = len(particles)

    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                circ = mpatches.Circle(
                    particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius, alpha=0.5)
                ax.add_artist(circ)
    plt.show(block=False)

    output = run(particles)

    fig = plt.figure()

    ax = plt.gca()

    N = len(particles)

    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                circ = mpatches.Circle(
                    particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.5)
                ax.add_artist(circ)
    plt.show()
