
import numpy as np

import time

import matplotlib.pyplot as plt

from integration_methods import Newmark

from particle_classes import Disk, Particle


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
        # Initializing the list containing the position of the particle in the grid, assuming:
        # 2D: the cells are numbered from left to right and from bottom to top
        for j_dim in range(dim):
        # Running through all the dimensions
            pos_cell_list_dim.append(
                np.int(np.floor(particles[i_particle].position_center[j_dim]/Particle.cell_side_length)))
            # j_dim-position of the particle in the grid
        if dim==2:
        # 2D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[1]
            # Saving the position in the cell list of particle i_particle
            for k_neighboor_cell in range(9):
            # Running through the neighboor cells
                pos_neighboor_cell = \
                    neighboorCell(pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim)
                # Computing the index of the neighboor cell
                for j_particle in Particle.cell_list[pos_neighboor_cell]:
                # Running through all the particles in the neighbooring cell
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

    if dim==2:
    # 2D problem
        local_row_pos_neigh = np.int(np.mod(np.floor(local_pos_neighboor_cell/3), 3) - 1)
        # Local row position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        local_col_pos_neigh = np.int(np.mod(local_pos_neighboor_cell,3) - 1)
        # Local column position of the neighboor, going from -1 to 1 with the origin at the
        # current cell
        pos_neighboor_cell = \
            np.int(pos_current_cell + local_col_pos_neigh + local_row_pos_neigh*n_cells[1])
        # Global position of the neighboor cell without enforcing periodic boundary
        # conditions
        if pos_current_cell<n_cells[1] and local_row_pos_neigh==-1:
        # Upper row of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        elif pos_current_cell>=n_cells[1]*(n_cells[0]-1) and local_row_pos_neigh==1:
        # Lower row of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1]*n_cells[0]
            # Enforcing the periodic boundary conditions
        if np.mod(pos_current_cell + 1, n_cells[1])==0 and local_col_pos_neigh==1:
        # Right column of the grid
            pos_neighboor_cell = pos_neighboor_cell - n_cells[1]
            # Enforcing the periodic boundary conditions
        elif np.mod(pos_current_cell, n_cells[1])==0 and local_col_pos_neigh==-1:
        # Left column of the grid
            pos_neighboor_cell = pos_neighboor_cell + n_cells[1]
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
                particles[i_particle].position_center[j_dim]/Particle.cell_side_length)))
            # j_dim-position of the particle in the grid
        if dim==2:
        # 2D problem
            pos_cell_list = pos_cell_list_dim[0] + \
                pos_cell_list_dim[1]*Particle.n_cell_dim[1]
            # Saving the position in the cell list of particle i_particle
        Particle.cell_list[pos_cell_list].append(i_particle)

def computeForces(particles):
    '''
    This function computes the forces between all the particle pairs in the system
    '''

    dim = particles[1].dim
    # Saving the dimension of the problem

    for i_particle in range(len(particles)):
    # Running through all the particles
        particles[i_particle].cleanForces()
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered

    newCellList(particles)

    
    if Particle.speed_up_scheme == 'Naive':
    # Naive approach: O(N^2)
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
    elif Particle.speed_up_scheme == 'Cell':
    # Cell list: O(N)
        for i_particle in range(len(particles)):
        # Running though all the particles
            pos_cell_list_dim = []
            # Initializing the list containing the position of the particle in the grid, assuming:
            # 2D: the cells are numbered from left to right and from top to bottom
            for j_dim in range(dim):
            # Running through all the dimensions
                pos_cell_list_dim.append(
                    np.int(np.floor(particles[i_particle].position_center[j_dim]/Particle.cell_side_length)))
                # j_dim-position of the particle in the grid
            if dim==2:
            # 2D problem
                pos_cell_list = pos_cell_list_dim[0] + \
                    pos_cell_list_dim[1]*Particle.n_cell_dim[1]
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(9):
                # Running through the neighboor cells
                    pos_neighboor_cell = \
                        neighboorCell(pos_cell_list, k_neighboor_cell, dim, Particle.n_cell_dim)
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighbooring cell
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
    elif Particle.speed_up_scheme == 'Verlet':
    # Cell list + Verlet list: O(N)
        if Particle.new_verlet_list:
        # There is a need to create
            # print('here2')
            newVerletList(particles)
            # Computing a new Verlet list
        for i_particle in range(len(particles)):
        # Running though all the particles
            # print('main',i_particle)
            for j_particle in particles[i_particle].verlet_list:
            # Running through all the particles in the neighbooring cell
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
    c = Particle.c
    # c = 1e-2
    # Damping constant of the system


    box = np.array([1,1])
    for i_particle in range(N):
    # Running through all the particles
        [new_position, new_velocity, new_accelaration] = \
            Newmark(particles[i_particle].position_center,
            particles[i_particle].velocity_center,
            Particle.global_force_factor*np.array([particles[i_particle].force],dtype='float').T,
            10e-6*np.eye(2,dtype='float'),
            c*np.eye(2,dtype='float'),
            np.zeros((2,2),dtype='float'),
            dt,
            1,
            dim)
        # Obtaining the new position and velocity of particle i
        if Particle.speed_up_scheme == 'Verlet':
            particles[i_particle].displacement_last_verlet += particles[i_particle].position_center - new_position[:,0]
            # Computing the displacement of the center of the particle
            # print('norm disp', np.linalg.norm(particles[i_particle].displacement_last_verlet))
            if np.linalg.norm(particles[i_particle].displacement_last_verlet) >= particles[i_particle].radius*(Particle.verlet_radius - 1):
            # Checking if the displacement takes the particle out of its neighboorhood
                # print('here')
                Particle.new_verlet_list = True
                # There is a need to compute a new verlet list
        new_position[:,0] = new_position[:,0] -box*np.floor(new_position[:,0]/box)
        # New position enforcing boundary conditions
        particles[i_particle].position_center = new_position[:,0]
        particles[i_particle].velocity_center = new_velocity[:,0]

        # Updating the position and velocity of particle i

# ==========================================================================================

def particleGeneration(*args):
    '''
    Function that generates all the particles from the geomtrical descriptors.

    '''

    Particle.box = [1,1]
    Particle.volume = 0
    Particle.verlet_radius = 1.5
    Particle.new_verlet_list = True
    Particle.speed_up_scheme = 'Verlet'
    

    particles = []

    dim = 2

    N = 500
    # Number of particles

    for i in range(N):
        particles.append(Disk(0.02)) #np.random.uniform(low=0.01,high=0.2)))
        # Disk with radius 0.5
        particles[i].position_center = np.random.uniform(size=dim) #np.array([0+i**2/200, 0.5]) # # #
        # Generating the positions from a random uniform distribution between 0 and 1
        particles[i].velocity_center = np.array([0,0],dtype='float')
        # Generating the velocities from a random uniform distribution between -1 and 1
        Particle.volume += particles[i].volume()

    Particle.c = 0 #np.sqrt(Particle.volume)*1e-2

    max_radius = np.max(np.array([particles[i].radius for i in range(N)]))*Particle.verlet_radius

    print(Particle.volume)

    box = [1,1]
    dim = particles[1].dim
    # Saving the dimension of the problem
    n_cells = 1
    # Initializing the number of cells
    Particle.n_cell_dim = []
    # Initializing the list containing the number of cells in each direction
    for i_dim in range(dim):
    # Running through all the dimensions
        Particle.n_cell_dim.append(np.int(np.round(box[i_dim]/(2*max_radius))))
        n_cells *= Particle.n_cell_dim[i_dim]
        # Computing the number of cells, such that a particle interacts at most with
        # particles in its cell and nearst neighboor cells
    Particle.cell_list = [[] for i in range(n_cells) ]
    # Initializing the cell list
    Particle.cell_side_length = box[0]/Particle.n_cell_dim[0]
    # Setting the cell side length as the radius of the largest

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
    

    N = len(particles)
    # Number of particles
    dt = 0.005
    # Setting the time step size

    max_residue = 10e-12*N
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0

    Particle.global_force_factor = 4 #N
    # Initializing the global force factor
    computeForces(particles)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    norm_velocity_vec = np.array([np.linalg.norm(particles[i].velocity_center) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    kin_energy = norm_velocity_vec.dot(norm_velocity_vec)
    print('kinetic',kin_energy)
    # Computing the relative energy
    relative_energy_old = relative_energy
    print('new',relative_energy)
    k = 0

    while (relative_energy >= max_residue) and (step<2000):
    # Run the simulation while the overlap is larger than the allowed maximum residue
        integrate(particles, dt)
        # Integrating the equations of motion
        step += 1
        # Moving to the next time step
        computeForces(particles)
        # Computing the forces on all particles
        Particle.new_verlet_list = False
        # Resetting the parameter that indicates the need to compute a new Verlet list
        norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
        # Obtaining a list with the norms of the vector forces
        relative_energy = norm_force_vec.dot(norm_force_vec)
        print('new',relative_energy)
        norm_velocity_vec = np.array([np.linalg.norm(particles[i].velocity_center) for i in range(N)],dtype='float')
        # Obtaining a list with the norms of the vector forces
        kin_energy = norm_velocity_vec.dot(norm_velocity_vec)
        print('kinetic',kin_energy)
        if np.random.uniform() >(1-Particle.volume/2):
            lambda_vel = 250/kin_energy/N
            for i_particle in range(N):
                particles[i_particle].velocity_center *= lambda_vel

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

        print(step)

        # Particle.global_force_factor = 1/relative_energy/100
        
        # if relative_energy < 1e-9 and k<4:
        #     k += 1
        #     fig = plt.figure()
        # 
        #     ax = plt.gca()
        # 
        #     N = len(particles)
        # 
        #     for i in range(N):
        #         for j in range(-1,2):
        #             for k in range(-1,2):
        #                 circ = mpatches.Circle(
        #                     particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius, alpha=0.5)
        #                 ax.add_artist(circ)
        #     plt.show()



    # Integrating Newton's equations of motion

if __name__ == '__main__':

    start = time.time()

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
                    particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
                ax.add_artist(circ)
                circ = mpatches.Circle(
                    particles[i].position_center+np.array([1*j,1*k]), radius=Particle.verlet_radius*particles[i].radius, alpha=0.1)
                ax.add_artist(circ)
                # plt.annotate(xy = particles[i].position_center, s=str(i))
                # plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                plt.axis([0, 1, 0, 1])

    # plt.xticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.yticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.grid(b=True, which='both')

    plt.show(block=False)

    output = run(particles)

    fig = plt.figure()

    ax = plt.gca()

    N = len(particles)

    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                circ = mpatches.Circle(
                    particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
                ax.add_artist(circ)
                circ = mpatches.Circle(
                    particles[i].position_center+np.array([1*j,1*k]), radius=Particle.verlet_radius*particles[i].radius, alpha=0.1)
                ax.add_artist(circ)
                # plt.annotate(xy = particles[i].position_center, s=str(i))
                # plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                plt.axis([0, 1, 0, 1])


    print(Particle.cell_list)

    end = time.time()
    print(end - start)

    # plt.xticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.yticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.grid(b=True, which='both')
    plt.show()
