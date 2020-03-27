
import numpy as np

import time

import matplotlib.pyplot as plt

from integration_methods import Newmark

from particle_classes import Disk, Particle, Ellipse

import pickle

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

def computeForces(particles, options):
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


    if options['speed_up_scheme'] == 'Naive':
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
    elif options['speed_up_scheme'] == 'Cell':
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
                    particles[i_particle].position_center[j_dim]/Particle.cell_side_length)))
                # j_dim-position of the particle in the grid
            if dim==2:
            # 2D problem
                pos_cell_list = pos_cell_list_dim[0] + \
                    pos_cell_list_dim[1]*Particle.n_cell_dim[1]
                # Saving the position in the cell list of particle i_particle
                for k_neighboor_cell in range(9):
                # Running through the neighboor cells
                    pos_neighboor_cell = \
                        neighboorCell(pos_cell_list,
                            k_neighboor_cell, dim, Particle.n_cell_dim)
                    # Computing the index of the neighboor cell
                    for j_particle in Particle.cell_list[pos_neighboor_cell]:
                    # Running through all the particles in the neighbooring cell
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
    elif options['speed_up_scheme'] == 'Verlet':
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
    #print('force', force_i_j)
    return force_i_j

def integrate(particles, options):
    '''
    This function integrates the equations of motion
    '''
    dim = particles[0].dim
    # Dimension of the problem
    N = len(particles)
    # Number of particles
    dt = options['dt']
    # Time step
    box = Particle.box
    # Saving the size of the RVE
    for i_particle in range(N):
    # Running through all the particles
        if options['integration_scheme']=='Newmark':
        # The integration scheme chosen was Newmark
            c = options['damping_constant']
            [new_position, new_velocity, new_accelaration] = \
                Newmark(particles[i_particle].position_center,
                particles[i_particle].velocity_center,
                Particle.global_force_factor*np.array([particles[i_particle].force],dtype='float').T,
                particles[i_particle].volume()*np.eye(2,dtype='float'), #10e-6*np.eye(2,dtype='float'),#
                c*np.eye(2,dtype='float'),
                np.zeros((2,2),dtype='float'),
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
        elif options['integration_scheme']=='Verlet':
        # The integration scheme chosen was Verlet
            pass
        else:
        # No integration scheme was chosen
            print('No integration scheme was chosen')
        if options['speed_up_scheme'] == 'Verlet':
            particles[i_particle].displacement_last_verlet += \
                particles[i_particle].position_center - new_position[:,0]
            # Computing the displacement of the center of the particle
            if np.linalg.norm(particles[i_particle].displacement_last_verlet) >= \
                particles[i_particle].radius*(Particle.verlet_radius - 1):
            # Checking if the displacement takes the particle out of its neighboorhood
                Particle.new_verlet_list = True
                # There is a need to compute a new verlet list
        new_position[:,0] = new_position[:,0] -box*np.floor(new_position[:,0]/box)
        # New position enforcing boundary conditions
        particles[i_particle].position_center = new_position[:,0]
        particles[i_particle].velocity_center = new_velocity[:,0]
        # Updating the position and velocity of particle i

# ==========================================================================================

def generateDisks(descriptors):
    '''
        This function generates disks.
    '''

    disks = []
    # Initializing the list containing the disks

    if descriptors.get('distribution')=='uniform':
    # the radius follows an uniform distribution
        for i in range(descriptors['n']):
        # Generating n disks
            disks.append(Disk(np.random.uniform(
                low=descriptors['r_low'],high=descriptors['r_high'])))
            # Disk with radius 0.5
            disks[i].position_center = np.random.uniform(size=2) #np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            disks[i].velocity_center = np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
    else:
    # the radius is fixed
        for i in range(descriptors['n']):
        # Generating n disks
            disks.append(Disk(descriptors['r'])) #np.random.uniform(low=0.01,high=0.2)))
            # Disk with radius 0.5
            disks[i].position_center = np.random.uniform(size=2) #np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            disks[i].velocity_center = np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1

    return disks

def generateEllipses(descriptors):
    '''
        This function generates ellipses.
    '''

    ellipses = []
    # Initializing the list containing the disks

    if descriptors.get('distribution')=='uniform':
    # the radius follows an uniform distribution
        pass
    else:
    # the radius is fixed
        for i in range(descriptors['n']):
        # Generating n disks
            ellipses.append(Ellipse(descriptors['major_axis'],
                descriptors['minor_axis'], descriptors['angle']+i/7*np.pi/2)) #np.random.uniform(low=0.01,high=0.2)))
            # Disk with radius 0.5
            ellipses[i].position_center = np.random.uniform(size=2) # np.array([0.5, 0.5-i/20]) # n # #
            # Generating the positions from a random uniform distribution between 0 and 1
            ellipses[i].velocity_center = np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1

    return ellipses

def particleGeneration(descriptors, phase_type, options):
    '''
    Function that generates all the particles from the geomtrical descriptors.

    '''

    Particle.box = options['rve_dims']
    # Setting the size of the box
    Particle.volume = 0
    # Initializing the total volume fraction
    Particle.number = 0
    # Initializing the total number of particles
    particles = []
    # Initializing the list containing the particles
    if options['problem_type'] == 1:
    # 2D problem (plain strain)
        dim = 2
        # Setting the dimension

    for i_phase in descriptors:
    # Running through all the phases listed in the dictionary
        if phase_type[i_phase] == 1:
        # This phase is the matrix
            pass
            # No particles are generated
        elif phase_type[i_phase] == 2:
        # This phase is made up by disks
            particles = particles + generateDisks(descriptors[i_phase])
            # Generating the number of disks requested and appending them to the list of
            # particles
        elif phase_type[i_phase] == 3:
        # This phase is made up by ellipses
            particles = particles + generateEllipses(descriptors[i_phase])
            # Generating the number of disks requested and appending them to the list of
            # particles

    return particles

# ==========================================================================================

def readDescriptors():

    # problem_type,mic_gen_descriptors,phase_types,
    #                         discret_file_ext, discret_spec_array

    options = {}
    # Initializing the dictionary containing the options
    options['damping_constant'] = 0
    options['dt'] = 0.005
    options['max_residue_per_particle'] = 0 #10e-12
    options['initial_global_force_factor'] = 100 #4
    options['thermostat']='isokinetic'
    options['global_force_factor_multiplier'] = 1.8

    options['speed_up_scheme']='Verlet'
    options['verlet_radius'] = 1.2

    options['max_step'] = 10000

    options['integration_scheme']='Newmark'

    options['rve_dims'] = [1.0, 1.0]

    options['problem_type'] = 1

    options['minimum_distance'] = 0

    descriptors = {}

    descriptors['4'] = {}
    # descriptors['2'] = {'r':0.1, 'n':10}
    #descriptors['2'] = {'distribution':'uniform','r_low':0.02,'r_high':0.04, 'n':190}
    descriptors['2'] = {'major_axis':0.20,'minor_axis':0.10,'angle':0,'n':35}

    phase_type = {}

    phase_type['4'] = 1 # Matrix
    phase_type['2'] = 3 # Elliptical particle 

    return [descriptors, phase_type, options]

def computeRelativeEnergy(particles):
    N = Particle.number
    norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy

    return relative_energy

def computeKineticEnergy(particles):
    N = Particle.number
    norm_velocity_vec = np.array([np.linalg.norm(particles[i].velocity_center) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    kin_energy = norm_velocity_vec.dot(norm_velocity_vec)

    return kin_energy

def run(particles, options):
    '''
    Main function of the Molecular Dynamics simulation.

    This is the main function of the Molecular Dynamics simulation. It consists of the
    initialization of the sytem, and the loop that contains the dynamics of the system:
    computation of the forces and integration of the equations of motion.

    Parameters:
        particles : array
            Array containing the Particle objects to be placed inside the RVE
        options : dictionary
            Dictionary containing the options for the MD simulation
    Returns:

    Other Parameters:

    '''
    N = Particle.number
    # Saving the number of particles
    box = Particle.box
    # Saving the array containing the size of the box
    dim = particles[1].dim
    # Saving the array containing the dimension of the problem
    if options['speed_up_scheme']=='Cell':
    # Only a cell list scheme will be used
        max_radius = np.max(np.array([particles[i].radius for i in range(N)]))
        # Saving the maximum radius of the circunscribing disk/sphere
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
        Particle.cell_list = [[] for i in range(n_cells)]
        # Initializing the cell list
        Particle.cell_side_length = box[0]/Particle.n_cell_dim[0]
        # Setting the cell side length as the radius of the largest
        # Limited to squares and cubes (FIX)
    elif options['speed_up_scheme']=='Verlet':
    # A Verlet list combined with a cell list scheme will be used
        Particle.verlet_radius = options['verlet_radius']
        # Saving the Verlet radius to compute the Verlet list
        Particle.new_verlet_list = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        max_radius = np.max(np.array([particles[i].radius\
            for i in range(Particle.number)]))*Particle.verlet_radius
        # Saving the maximum radius of the circunscribing disk/sphere accounting for the
        # Verlet factor
        print(Particle.volume)
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
    else:
    # A naive approach will be used
        pass

    N = len(particles)
    # Number of particles
    dt = options['dt']
    # Setting the time step size
    max_residue = options['max_residue_per_particle']*N
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0
    Particle.global_force_factor = options['initial_global_force_factor']
    # Initializing the global force factor
    computeForces(particles, options)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    relative_energy = computeRelativeEnergy(particles)
    # Computing the relative energy
    kin_energy = computeKineticEnergy(particles)
    # Computing the kinetic energy
    print('kinetic',kin_energy)
    relative_energy_old = relative_energy
    # Saving the current relative energy
    print('new',relative_energy)
    while (relative_energy > max_residue) and (step<options['max_step']):
    # Run the simulation while the overlap is larger than the allowed maximum residue
        integrate(particles, options)
        # Integrating the equations of motion
        step += 1
        # # Moving to the next time step
        # print([particles[i].velocity_center for i in range(len(particles))])
        # print([particles[i].position_center for i in range(len(particles))])
        computeForces(particles, options)
        # Computing the forces on all particles
        relative_energy = computeRelativeEnergy(particles)
        # Computing the relative energy
        kin_energy = computeKineticEnergy(particles)
        # Computing the kinetic energy
        print('new',relative_energy)
        print('kinetic',kin_energy)
        # print([particles[i].velocity_center for i in range(len(particles))])
        # print([particles[i].force for i in range(len(particles))])
        if options['thermostat']=='isokinetic':
        # The thermostat used is the isokinetic scheme
            if np.random.uniform() > (1-Particle.volume/2): #0: #
            # Probability of rescaling the velocities modelled as Poisson
                lambda_vel = np.sqrt(np.max([1e3*relative_energy,1e-4])/kin_energy/N) #250/kin_energy/N# 
                # Rescalling factor (why? 250 -  equipartition theorem)
                for i_particle in range(N):
                # Running through all the particles
                    particles[i_particle].velocity_center *= lambda_vel
                    # Rescalling the velocities
                    # for j_component in range(2):
                    #     particles[i_particle].velocity_center[j_component] =
                    #         np.max()
        else:
        # There is no thermostat
            pass
        # Particle.global_force_factor *= 10e-3/relative_energy #options['global_force_factor_multiplier']

        # if relative_energy/relative_energy_old < 0.5:
        # # If the relative energy has decreased by a factor of two in this iteraton
        #     relative_energy_old = relative_energy
        #     # Saving the value of the previous relative energy
        #     Particle.global_force_factor *= \
        #         options['global_force_factor_multiplier']
        #     # Increase the global factor multiplying the forces
        #     print('force factor',Particle.global_force_factor)
        # elif relative_energy/relative_energy_old > 2:
        # # If the relative energy has increased by a factor of two in this iteraton
        #     relative_energy_old = relative_energy
        #     # Saving the value of the previous relative energy
        #     Particle.global_force_factor *= 1/options['global_force_factor_multiplier']
        #     # Increase the global factor multiplying the forces
        #     print('force factor',Particle.global_force_factor)

        print(step)

        # Particle.global_force_factor = 1/relative_energy/100
        
        # if relative_energy < 1e-9 and k<4:
        #     k += 1


        # fig = plt.figure()
        # 
        # ax = plt.gca()
        # 
        # N = len(particles)
        # 
        # for i in range(N):
        #     for j in range(-1,2):
        #         for k in range(-1,2):
        #             # circ = mpatches.Circle(
        #             #     particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
        #             # ax.add_artist(circ)
        #             # circ = mpatches.Circle(
        #             #     particles[i].position_center+np.array([1*j,1*k]), radius=Particle.verlet_radius*particles[i].radius, alpha=0.1)
        #             # ax.add_artist(circ)
        #             ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.8)
        #             ax.add_artist(ellip)
        #             plt.annotate(xy = particles[i].position_center, s=str(i))
        #             plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
        #             plt.axis([0, 1, 0, 1])
        # # 
        # # 
        # # # print(Particle.cell_list)
        # # 
        # # end = time.time()
        # # print(end - start)
        # # 
        # # # plt.xticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
        # # # plt.yticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
        # # # plt.grid(b=True, which='both')
        # plt.show()



    # Integrating Newton's equations of motion

    

if __name__ == '__main__':

    start = time.time()

    import matplotlib.patches as mpatches

    [descriptors, phase_type, options] = readDescriptors()
    # Reading the descriptors

    particles = particleGeneration(descriptors, phase_type, options)
    # Generating the list of particles from the geometrical descriptors

    fig = plt.figure()
    
    ax = plt.gca()
    
    N = len(particles)
    
    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.8)
                ax.add_artist(ellip)
                plt.annotate(xy = particles[i].position_center, s=str(i))
                # plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                # plt.axis([0, 1, 0, 1])
                # circ = mpatches.Circle(
                #     particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
                # ax.add_artist(circ)
                # circ = mpatches.Circle(
                #     particles[i].position_center+np.array([1*j,1*k]), radius=options['verlet_radius']*particles[i].radius, alpha=0.1)
                # ax.add_artist(circ)
                # plt.annotate(xy = particles[i].position_center, s=str(i))
                # plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                plt.axis([0, 1, 0, 1])
    
    # plt.xticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.yticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    # plt.grid(b=True, which='both')
    
    plt.show(block=False)

    run(particles, options)

    fig = plt.figure()
    
    ax = plt.gca()
    
    N = len(particles)
    
    for i in range(N):
        for j in range(-1,2):
            for k in range(-1,2):
                # circ = mpatches.Circle(
                #     particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
                # ax.add_artist(circ)
                # circ = mpatches.Circle(
                #     particles[i].position_center+np.array([1*j,1*k]), radius=Particle.verlet_radius*particles[i].radius, alpha=0.1)
                # ax.add_artist(circ)
                ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.8)
                ax.add_artist(ellip)
                plt.annotate(xy = particles[i].position_center, s=str(i))
                plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                plt.axis([0, 1, 0, 1])

    print([particles[i].position_center for i in range(len(particles))])
    print([particles[i].velocity_center for i in range(len(particles))])

    print(Particle.cell_list)
    print([particles[i].verlet_list for i in range(N)])

    end = time.time()
    print(end - start)

    plt.xticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    plt.yticks(np.linspace(0,1,Particle.n_cell_dim[0]+1,endpoint=True))
    plt.grid(b=True, which='both')
    plt.show()
