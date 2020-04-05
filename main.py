#
# Microstructure Generation Interface (DATAGEM Program)(????)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Zé Luís P. Vila-Chã | March 2020 | Initial coding.
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
from particle_classes import Disk, Particle, Ellipse, Sphere
# Importing the particle class
from meshing_interface import generateMeshFEM, generateMeshFFT
import os
import sys
# ==========================================================================================
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

def computeForces(particles, speed_up_scheme):
    '''
    This function computes the forces between all the particle pairs in the system

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
    '''

    dim = particles[1].dim
    # Saving the dimension of the problem
    for i_particle in range(len(particles)):
    # Running through all the particles
        particles[i_particle].cleanForces()
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
    if speed_up_scheme == 'Naive':
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
        elif integration_scheme=='Verlet':
        # The integration scheme chosen was Verlet
            pass
        else:
        # No integration scheme was chosen
            print('No integration scheme was chosen')
        if speed_up_scheme == 'Verlet':
            particles[i_particle].displacement_last_verlet += \
                particles[i_particle].position_center - new_position[:,0]
            # Computing the displacement of the center of the particle
            class_name_i_particle = particles[i_particle].__class__.__name__
            if "Disk"==class_name_i_particle:
                radial_dimension = particles[i_particle].radius
            elif "Eliipse"==class_name_i_particle:
                radial_dimension = particles[i_particle].semi_minor_axis
            # FIX: 
            if np.linalg.norm(particles[i_particle].displacement_last_verlet) >= \
                radial_dimension*(Particle.verlet_factor - 1):
            # if not particles[i].insideVerlet(
            #     particles[i_particle].displacement_last_verlet +\
            #     particles[i_particle].position_center):
            # Checking if the displacement takes the particle out of its neighboorhood
                Particle.new_verlet_list = True
                # There is a need to compute a new verlet list
        new_position[:,0] = new_position[:,0] -box*np.floor(new_position[:,0]/box)
        # New position enforcing boundary conditions
        particles[i_particle].position_center = new_position[:,0]
        particles[i_particle].velocity_center = new_velocity[:,0]
        # Updating the position and velocity of particle i

# ==========================================================================================

def generateDisks(phase, descriptors):
    '''
        This function generates disks.
    '''

    disks = []
    # Initializing the list containing the disks
    if descriptors.get('distribution')=='uniform':
    # the radius follows an uniform distribution
        for i in range(descriptors['n']):
        # Generating n disks
            disks.append(Disk(phase, np.random.uniform(
                low=descriptors['r_low'],high=descriptors['r_high'])))
            # Disk with radius 0.5
            disks[i].position_center = np.array([i*1/25, np.floor(i/25)*1/25 ]) # # # np.random.uniform(size=2) #
            # Generating the positions from a random uniform distribution between 0 and 1
            disks[i].velocity_center = np.random.uniform(size=2) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
    else:
    # the radius is fixed
        for i in range(descriptors['n']):
        # Generating n disks
            disks.append(Disk(phase, descriptors['r'])) #np.random.uniform(low=0.01,high=0.2)))
            # Disk with radius 0.5
            disks[i].position_center = np.random.uniform(size=2) # np.array([(i+1)*1/24-np.floor((i+1)*1/24), (1+np.floor(i/24))*1/24 ]) # np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            disks[i].velocity_center = np.random.uniform(size=2) #np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1

    return disks

def generateSpheres(phase, descriptors):
    '''
        This function generates spheres.
    '''

    spheres = []
    # Initializing the list containing the spheres

    if descriptors.get('distribution')=='uniform':
    # the radius follows an uniform distribution
        for i in range(descriptors['n']):
        # Generating n spheres
            spheres.append(Sphere(phase, np.random.uniform(
                low=descriptors['r_low'],high=descriptors['r_high'])))
            # Sphere with radius 0.5
            spheres[i].position_center = np.random.uniform(size=2) #np.array([0+i**2/200, 0.5]) # # #
            # Generating the positions from a random uniform distribution between 0 and 1
            spheres[i].velocity_center = np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1
    else:
        print(descriptors)
    # the radius is fixed
        for i in range(int(descriptors['n'])):
        # Generating n spheres
            spheres.append(Sphere(phase, descriptors['r'])) #np.random.uniform(low=0.01,high=0.2)))
            # Sphere with radius 0.5
            spheres[i].position_center = np.random.uniform(size=3) # np.array([0.5+i**2/200, 0.5, 0.5]) #    # #
            # Generating the positions from a random uniform distribution between 0 and 1
            spheres[i].velocity_center = np.array([0, 0, 0], dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1

    return spheres

def generateEllipses(phase, descriptors):
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
            ellipses.append(Ellipse(phase, descriptors['major_axis'],
                descriptors['minor_axis'], descriptors['angle']+i/7*np.pi/2)) #np.random.uniform(low=0.01,high=0.2)))
            # Generating ellipses, all with the same dimensions
            ellipses[i].position_center = np.random.uniform(size=2) # np.array([0.5, 0.5-i/20]) # n # #
            # Generating the positions from a random uniform distribution between 0 and 1
            ellipses[i].velocity_center = np.array([0,0],dtype='float')
            # Generating the velocities from a random uniform distribution between -1 and 1

    return ellipses

def particleGeneration(descriptors, phase_types, rve_dims, problem_type):
    '''
    Function that generates all the particles from the geomtrical descriptors.

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
    '''

    Particle.box = rve_dims
    # Setting the size of the box
    Particle.volume = 0
    # Initializing the total volume fraction
    Particle.number = 0
    # Initializing the total number of particles
    Particle.list_phases = [i_phase for i_phase in descriptors]
    # List containing the phases
    particles = []
    # Initializing the list containing the particles
    if problem_type == 1:
    # 2D problem (plain strain)
        dim = 2
        # Setting the dimension
    for i_phase in descriptors:
    # Running through all the phases listed in the dictionary
        if phase_types[i_phase] == 1:
        # This phase is the matrix
            Particle.matrix_phase = i_phase
            # No particles are generated
        elif phase_types[i_phase] == 2:
        # This phase is made up by disks
            particles = particles + generateDisks(i_phase, descriptors[i_phase])
            # Generating the number of disks requested and appending them to the list of
            # particles
        elif phase_types[i_phase] == 3:
        # This phase is made up by ellipses
            particles = particles + generateEllipses(i_phase, descriptors[i_phase])
            # Generating the number of ellipses requested and appending them to the list of
            # particles
        elif phase_types[i_phase] == 4:
        # This phase is made up by spheres
            particles = particles + generateSpheres(i_phase, descriptors[i_phase])
            # Generating the number of spheres requested and appending them to the list of
            # particles

    Particle.file_name = particles[0].__class__.__name__ + "_" + str(Particle.number) + "_" + str(Particle.volume)
    main_folder = os.path.dirname(os.getcwd())
    print(main_folder)
    # Path to the main folder of the program
    print(main_folder)
    results_folder = os.path.join(main_folder, "results",  Particle.file_name)
    if os.path.exists(results_folder):
        results_folder_old = results_folder
        i = 0
        while os.path.exists(results_folder):
            i += 1
            results_folder = results_folder_old + "_" + str(i)
        os.makedirs(results_folder)
    else:
        os.makedirs(results_folder)
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    print(Particle.file_path)
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

    descriptors: dict

    phase_types: dict

    options: dict

    n_dp_samples: int

    rve_dims: list

    problem_type: int

    discret_spec_array: dict
    """

    info_dict = pickle.load(open('input_data\\info_micro.p','rb'))
    # Loading the dictionary containing the information about the microstructure and its
    # generation
    # dp_dir: string
    #     Directory where the microstructure spatial discretization file(s) associated
    #     with the given design point are to be stored
    dp_dir = info_dict['dp_dir']
    # mic_gen_parameters: array
    #     An array which contains all the required parameters (or options)
    #     for the selected program to generate the microstructure(s) and
    #     and associated discretization file(s) of a given design point
    #     (to be discussed...)
    options = info_dict['mic_gen_parameters']
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
    problem_type = info_dict['problem_type']
    # n_dp_samples: integer
    #     Number of microstructures (samples) to be generated, associated to
    #     the given design point
    n_dp_samples = info_dict['n_dp_samples']
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
    descriptors = info_dict['mic_gen_descriptors']

    # descriptors['4'] = {'rve_dims':[1.0, 1.0, 1.0]}
    # rve_dims = descriptors['4']['rve_dims']
    # descriptors['2'] = {'r':0.1, 'n':3}
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
    phase_types = info_dict['phase_types']
    # phase_types['4'] = 1 # Matrix
    # phase_types['2'] = 4 # Elliptical particle
    # discret_file_ext: list
    #     List which contains the required spatial discretization file(s), stored as
    #                     array = [ < discret_type > < discret_type >  ... ]

    # discret_spec_array: dictionary
    #     Dictionary which contains the required parameters to generate
    #     each type of specified discretization file, stored as
    #                            dictionary['disc_ext']['parameter'] = [ ... ]

    discret_spec_array = info_dict['discret_spec_array']
    # discret_spec_array['rgmsh'] = {}
    # discret_spec_array['rgmsh']['rve_dims'] = np.array([1.0, 1.0, 1.0])
    # discret_spec_array['rgmsh']['n_voxels_dims'] = np.array([ 50, 50, 50])
    # discret_spec_array['femsh'] = {}
    # discret_spec_array['femsh']['rve_dims'] = np.array([ 1.0, 1.0, 1.0 ])
    # discret_spec_array['femsh']['mesh_size'] = 0.1
    if 'rgmsh' in discret_spec_array:
        rve_dims = discret_spec_array['rgmsh']['rve_dims']
    elif 'femsh' in discret_spec_array:
        rve_dims = discret_spec_array['femsh']['rve_dims']

    return [dp_dir, descriptors, phase_types, options, n_dp_samples, rve_dims, problem_type,
        discret_spec_array]


def computeRelativeEnergy(particles):
    N = Particle.number
    norm_force_vec = np.array([np.linalg.norm(particles[i].force) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    relative_energy = norm_force_vec.dot(norm_force_vec)
    # Computing the relative energy
    print('new',relative_energy)

    return relative_energy

def computeKineticEnergy(particles):
    N = Particle.number
    norm_velocity_vec = np.array([np.linalg.norm(particles[i].velocity_center) for i in range(N)],dtype='float')
    # Obtaining a list with the norms of the vector forces
    kin_energy = norm_velocity_vec.dot(norm_velocity_vec)
    print('kinetic',kin_energy)

    return kin_energy

def run(particles, dt, max_residue_per_particle, max_step, speed_up_scheme='Naive', thermostat='isokinetic', **kwargs):
    """
    Run the Molecular Dynamics simulation for the system of particles given.

    This is the main function of the Molecular Dynamics simulation. It consists of the
    initialization of the sytem, and the loop that contains the dynamics of the system:
    computation of the forces and integration of the equations of motion.

    Parameters
    ----------
    particles : list(`.Particle`)
        Array containing the Particle objects to be placed inside the RVE

    dt: float
        Time step

    max_residue_per_particle: float
        Maximum allowable overlap residue between particles

    max_step: int
        Maxium number of time steps

    thermostat: {'isokinetic'}, optional
        Thermostat to be used

    speed_up_scheme: {'Naive', 'Cell', 'Verlet'}, optional
        Speed up scheme used in the force computation
            "Naive": the forces are computed between every pair of particles (O(N**2))
            "Cell": the forces are computed making use of a cell list, such that each particle
                only interacts with the particles in its cell or the nearest neighboring
                cells (O(N))
            "Verlet": the forces are computed using a Verlet list for each particle, that in
                turn in computed using a cell list method

    Other Parameters
    ----------------
    **kwargs:
        Other keyword parameters used such as:
        verlet_factor: float
            Factor defining the Verlet neighboorhood
        initial_global_force_factor: float
            Factor multiplied at the begin of the simulation by the forces for dynamical
            adjustments
        max_steps_to_relax: int
            Number of steps the configuration has to be below the maximum overlap residual
            area before the configuration is accepted
    """
    N = Particle.number
    # Saving the number of particles
    box = Particle.box
    # Saving the array containing the size of the box
    dim = particles[1].dim
    # Saving the array containing the dimension of the problem
    if speed_up_scheme=='Cell':
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
    elif speed_up_scheme=='Verlet':
    # A Verlet list combined with a cell list scheme will be used
        Particle.verlet_factor = kwargs['verlet_factor']
        # Saving the Verlet radius to compute the Verlet list
        Particle.new_verlet_list = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        max_radius = np.max(np.array([particles[i].radius\
            for i in range(Particle.number)]))*Particle.verlet_factor
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
    n_steps_relax = 0
    # Initializing the number of steps that a microstructure was complying with the
    # maximum overlap residue
    max_residue = max_residue_per_particle*N
    # Maximum residual overlap
    step = 0
    # Initializing the the time step at 0
    initial_global_force_factor = kwargs.get('initial_global_force_factor', 1)
    Particle.global_force_factor = initial_global_force_factor
    # Initializing the global force factor
    computeForces(particles, speed_up_scheme)
    # Computing the forces in the initial configuration to obtain the initial relative
    # potential energy (related to the overlap)
    relative_energy = computeRelativeEnergy(particles)
    # Computing the relative energy
    kin_energy = computeKineticEnergy(particles)
    # Computing the kinetic energy
    # relative_energy_old = relative_energy
    # Saving the current relative energy
    max_steps_to_relax = kwargs.get('max_steps_to_relax',1)
    while (step < max_step) and n_steps_relax < max_steps_to_relax:
    # Run the simulation while the number of steps the overlap has been smaller than the
    # allowed maximum residue is larger than options['max_steps_to_relax'], so that the
    # particles have time to get away from each other.
        integrate(particles, dt, speed_up_scheme)
        # Integrating the equations of motion
        step += 1
        # # Moving to the next time step
        computeForces(particles, speed_up_scheme)
        # Computing the forces on all particles
        relative_energy = computeRelativeEnergy(particles)
        # Computing the relative energy
        kin_energy = computeKineticEnergy(particles)
        # Computing the kinetic energy
        if thermostat=='isokinetic':
        # The thermostat used is the isokinetic scheme
            if np.random.uniform() > (1-Particle.volume/2):
            # Probability of rescaling the velocities modelled as Poisson
                lambda_vel = np.sqrt(np.max([1e6*relative_energy,1e-2])/kin_energy/N)
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
        if relative_energy <= max_residue:
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

def plotParticles(particles, dir, grid='off', verlet_ngh=False, center_part=False, block=False, save=True, **kwargs):
    """Plot the particles."""
    import matplotlib.patches as mpatches

    N = len(particles)
    if particles[0].dim == 2:
    # Two dimensional problem
        fig = plt.figure()

        ax = plt.gca()        

        for i in range(N):
            class_name_i_particle = particles[i].__class__.__name__
            for j in range(-1,2):
                for k in range(-1,2):
                    if 'Disk'==class_name_i_particle:
                        circ = mpatches.Circle(
                            particles[i].position_center+np.array([1*j,1*k]), radius=particles[i].radius,alpha=0.8)
                        ax.add_artist(circ)
                        if verlet_ngh:
                            circ = mpatches.Circle(
                                particles[i].position_center+np.array([1*j,1*k]+particles[i].displacement_last_verlet), radius=Particle.verlet_factor*particles[i].radius, alpha=0.1)
                            ax.add_artist(circ)
                        if center_part:
                            plt.annotate(xy = particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],particles[i].position_center[1])
                    if 'Ellipse'==class_name_i_particle:
                        ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]), particles[i].major_axis, particles[i].minor_axis,angle=180/np.pi*particles[i].angle,alpha=0.8)
                        ax.add_artist(ellip)
                        if verlet_ngh:
                            ellip = mpatches.Ellipse(particles[i].position_center+np.array([1*j,1*k]+particles[i].displacement_last_verlet), particles[i].major_axis*Particle.verlet_factor, particles[i].minor_axis*Particle.verlet_factor,angle=180/np.pi*particles[i].angle,alpha=0.2)
                            ax.add_artist(ellip)
                        if center_part:
                            plt.annotate(xy = particles[i].position_center, s=str(i))
                            plt.scatter(particles[i].position_center[0],particles[i].position_center[1])

        if grid=='cell_list':
            plt.xticks(np.linspace(0,1,Particle.n_cell_dim+1 ,endpoint=True))
            plt.yticks(np.linspace(0,1,Particle.n_cell_dim+1,endpoint=True))
            plt.grid(b=True, which='both')
        elif grid=='fft':
            discret_spec_array = kwargs('discret_spec_array')
            plt.xticks(np.linspace(0,1,discret_spec_array['rgmsh']['n_voxels_dims'][0]+1 ,endpoint=True))
            plt.yticks(np.linspace(0,1,discret_spec_array['rgmsh']['n_voxels_dims'][1]+1,endpoint=True))
            plt.grid(b=True, which='both')

        ax.axis("square")

        plt.axis([0, 1, 0, 1])
        
        plt.savefig(dir + ".png")
        plt.show()

    elif particles[0].dim == 3:
        pass
    else:
        box = Particle.box
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        def drawSphere(pos, r):
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:5j, 0:np.pi:5j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            # shift and scale sphere
            x = r*x + pos[0]
            y = r*y + pos[1]
            z = r*z + pos[2]
            return (x, y, z)

        def plot_cube(cube_definition):
            cube_definition_array = [
                np.array(list(item))
                for item in cube_definition
            ]

            points = []
            points += cube_definition_array
            vectors = [
                cube_definition_array[1] - cube_definition_array[0],
                cube_definition_array[2] - cube_definition_array[0],
                cube_definition_array[3] - cube_definition_array[0]
            ]

            points += [cube_definition_array[0] + vectors[0] + vectors[1]]
            points += [cube_definition_array[0] + vectors[0] + vectors[2]]
            points += [cube_definition_array[0] + vectors[1] + vectors[2]]
            points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

            points = np.array(points)

            edges = [
                [points[0], points[3], points[5], points[1]],
                [points[1], points[5], points[7], points[4]],
                [points[4], points[2], points[6], points[7]],
                [points[2], points[6], points[3], points[0]],
                [points[0], points[2], points[4], points[1]],
                [points[3], points[6], points[7], points[5]]
            ]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
            faces.set_facecolor((0, 0, 1, 0.05))

            ax.add_collection3d(faces)

            # Plot the points themselves to force the scaling of the axes
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

            ax.set_aspect('equal')

        cube_definition = [
            (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)
        ]
        plot_cube(cube_definition)

        fig = plt.gcf()
        ax = fig.gca()

        for i in range(N):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        (xs, ys, zs) = drawSphere(
                            particles[i].position_center+np.array([1*j, 1*k, 1*l]),
                            particles[i].radius)
                        x_clip = np.logical_or(np.abs(np.array(xs)) > 1, xs < 0)
                        y_clip = np.logical_or(np.abs(np.array(ys)) > 1, ys < 0)
                        z_clip = np.logical_or(np.abs(np.array(zs)) > 1, zs < 0)
                        in_points = np.logical_or(np.logical_or(x_clip, y_clip), z_clip)
                        # xs[in_points] = np.nan
                        # ys[in_points] = np.nan
                        zs[in_points] = np.nan
                        ax.plot_wireframe(xs, ys, zs, color="b")
                        ax.text(particles[i].position_center[0],
                                particles[i].position_center[1],
                                particles[i].position_center[2],
                                str(i))
                        plt.scatter(
                            particles[i].position_center[0],
                            particles[i].position_center[1],
                            particles[i].position_center[2])

        plt.grid(b=False)
        ax.set_aspect('equal')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        ax.set_clip_on(True)
        # plt.axis([0, 1, 0, 1, 0, 1])


def main():

    start = time.time()
    # Counting time
    f = open("test.txt", 'w')
    # sys.stdout = f
    [dp_dir, descriptors, phase_types, options, n_samples, rve_dims, problem_type,
        discret_spec_array] = readDescriptors()
    # Reading the descriptors and options for the microstructure generation
    for i_sample in range(n_samples):
        # Producing the number of samples required
        particles = particleGeneration(descriptors, phase_types, rve_dims, problem_type)
        # Generating the list of particles from the geometrical descriptors
        plotParticles(particles, Particle.file_path + "_random", save=False)
        # FIX (options ploting, saving)
        run(particles, options['dt'], options['max_residue_per_particle'],
            options['max_step'], options['speed_up_scheme'])
        # Running the molecular dynamics simulation
        end = time.time()
        if 'rgmsh' in discret_spec_array:
            # A mesh for FFT was requested
            generateMeshFFT(particles, discret_spec_array['rgmsh'])
            # Generating the FFT mesh as a regular grid and saving it in a .dat file
        if 'femsh' in discret_spec_array:
            # A mesh for FEM was requested
            print('here')
            generateMeshFEM(particles, discret_spec_array['femsh']['mesh_size'],
                            output_term=True)
            # Generating the FEM mesh using gmsh and saving an input data file for LINKS

        print('pos_end', [ (i,particles[i].position_center) for i in range(len(particles))])

        plotParticles(particles, Particle.file_path, save=False)

    
    print(end - start)

    

    f.close()
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
