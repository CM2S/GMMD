
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
    dim = particles[i_particle].dimension
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

def Newmark(x_0, x_dot_0, f_vec, m_mat, c_mat, k_mat, dt, n_steps, dim):
    '''
    This function integrates the equations of motion using Newmark's method.

    Parameters:
        x_0: vector array
            Initial positions of the DOFs
        x_dot_0: vector array
            Initial velocities of the DOFs
        f_vec: vector array
            Forves acting on the DOFs at each time instant
        m_mat: matrix array
            Mass matrix
        c_mat: matrix array
            Damping matrix
        k_mat: matrix array
            Stiffness matrix
        dt: float
            Time step
        n_steps: int
            Number of time steps to be used
        dim: int
            Dimension of the problem

    Returns:
        x_vec: vector array
            Positions
        x_dot_vec: vector array
            Velocities
        x_ddot_vec: vector array
            Acceleration
    '''
    x_vec = np.zeros(dim, n_steps+1)
    x_vec[:,0] = x_0
    x_dot_vec = np.zeros(dim, n_steps+1)
    x_dot_vec[:,0] = x_dot_0
    x_ddot_vec = np.zeros(dim, n_steps+1)
    # Initializing the array vectors containing the positions, velocities and accelerations
    x_ddot_vec[:,0] = \
        numpy.linalg.solve(m_mat, f_vec[:,0] - c_mat.dot(x_dot_0) - k_mat.dot(x_0))
    # Computing the accelaration at time instant 0
    delta = 0.5
    alpha = 0.25
    a_0 = 1/(alpha*dt**2)
    a_1 = delta/(alpha*dt)
    a_2 = 1/(alpha*dt)
    a_3 = 1/(2*alpha) - 1
    a_4 = delta/alpha - 1
    a_5 = dt/2*(delta/alpha - 2)
    a_6 = dt*(1 - delta)
    a_7 = delta*dt
    # Computing the constants used in the integration algorithm
    k_mat_eff = k_mat + a_0*m_mat + a_1*c_mat
    # Computing the effective stiffness matrix
    step = 0
    # Initializing the step counter
    while step<n_steps:
        # Repeat n_steps times
        f_vec_eff = f_vec + m_mat.dot(a_0*x_vec[:,step] + a_2*x_dot_vec[:,step] + \
            a_3*x_ddot_vec[:,step]) + c_mat.dot(a_1*x_vec[:,step] + a_4*x_dot_vec[:,step] +\
            a_5*x_ddot_vec[:,step])
        # Computing the effective force at time step*dt
        x_vec[:,step+1] = numpy.linalg.solve(k_mat_eff, f_vec_eff)
        # Computing the position vector at time (step+1)*dt
        x_ddot_vec[:,step+1] = a_0*(x_vec[:,step+1] - x_vec[:,step]) - \
            a_2*x_dot_vec[:,step] - a_3*x_ddot[:,step]
        # Computing the acceleration vector at time (step+1)*dt
        x_dot_vec[:,step+1] = x_dot_vec[:,step] + a_6*x_ddot_vec[:,step] + \
            a_7*x_ddot_vec[:,step+1]
        # Computing the velocity vector at time (step+1)*dt
return [x_vec[:,1:], x_dot_vec[:,1:], x_ddot_vec[:,1:]]


# ==========================================================================================

# ==========================================================================================
class Particle():
    '''
    This is the class for particles

    Attributes:
        center: list
            The position vector of the center of mass of the particle
        dim: int
            Number of the dimensions of the space where the particle "lives"
    '''

    def __init__(self, dim):
    '''
    The constructor for the Particle class.

    Parameters:
        dim: int
            Number of the dimensions of the space where the particle "lives"
    '''

        self.dim = dim
        # Setting the the dimension where the particle "lives"

    def setPositionCenter(self, position):
    '''
    This function sets the position of the center of mass of the particle
    '''

    self.position_center = position_center
    # Setting the position of the center of mass of the particle

    def setVelocityCenter(self, velocity):
    '''
    This function sets the velocity of the center of mass of the particle.
    '''

    self.velocity_center = velocity_center
    # Setting the velocity of the center of mass of the particle


# ==========================================================================================
class Disk(Particle):
    '''
    This is the subclass of particles with the form of a circular disk.

    Attributes:
        radius: int
            Radius of the disk
    '''
    def __init__(self, radius):
        '''
        The constructor of the Disk particle.

        Parameters:
            center: list
                The position vector of the center of mass of the particle
            dim: int
                Number of the dimensions of the space where the particle "lives"
            radius: float
                Radius of the disk
        '''
        self.dim = 2
        self.radius = radius
        self.force = []

    def intersectionArea(self, other_particle):
        '''
        This function computes the intersection between the disk and the other particle.

        Parameters:
            other_particle: Particle
                Other particle
        '''
        class_name_other_particle = other_particle.__class__.__name__
        # Saving the class name of the other particle as a string
        switch (class_name_other_particle) {
        # Computing the intersection are according to the type of particle
            case 'Disk':
            # The other particle is also a Disk
                intersection_area = intersectionAreaDiskDisk(self, other_particle)
                # Computing the intersection area
                return intersection_area


    def intersectionAreaDiskDisk(self, other_disk):
        '''
        This function computes the intersection area between two disks
        '''

        d = np.sqrt(self.position_center.dot(other_disk.position_center))
        # Distance between the center of the disks
        if self.radius >= other_disk.radius:
        # The radius of the self is larger than the radius of the other disk
            r_1 = self.radius
            # Disk 1 is the disk with the larger radius
            r_2 = other_disk.radius

        else:
        # The radius of the other disk is larger than the radius of the self
            r_1 = other_disk.radius
            # Disk 1 is the disk with the larger radius
            r_2 = self.radius
            # Disk 2 is the disk with the smaller radius
        if d>=r_1 + r_2:
        # The disks intersect at most at one point
            intersection_area = 0
            # The intersection area of the disks is zero
        elif d<=r_1 - r_2:
        # Disk 2 is interely contained within Disk 1
            intersection_area = np.pi*r_2**2
            # The intersection area is equal to the area of the smaller disk, Disk 2
        else:
            d_1 = (r_1**2 - r_2**2 + d**2)/(2*d)
            # x coordinate of the intersection point of the two disks if the the origin is at
            # disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_area = r_1**2*np.arccos(d_1/r_1) - d_1*np.sqrt(r_1**2-d_1**2) + \
                r_2**2*np.arccos(d_2/r_2) - d_2*np.sqrt(r_2**2 - d_2**2)
            # Computing the intersection area
        return intersection_area
        # Returning the intersection area

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

    relative_energy = [particles[i].force for i in range(N)]

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
