"""
Module containing the MolecularDynamicsSimulation class.

It provides a class whose methods allow the performance of a molecular dynamics simulation
where the forces between the particles are repulsive and propertional to the overlap
area/volume. A given configuration of particles is considered legal when the total overlap
area/volume is smaller than specified, It is accepted when the system remains in a legal
configuration for a specified number of steps as low as one. It requires subclasses of the
Thermostat and SpeedUpScheme abstract classes to work.
"""
import numpy as np

from .microstructure_gen_method import GenerationMethod


class MolecularDynamicsSimulation(GenerationMethod):
    """Class for the molecular dynamics simulation class.

    It stores all the options specifying how the simlulation will be run, it contains the
    methods needed to run the simulation and it also stores the relevant details of the
    simulation.

    Attributes
    ----------
    box: list
        List of the dimensions of the simulation box. Almost always equal to the the
        dimensions of the micrrostructure, except for CylindricalFibers.

    particle_velocities: list(array)
        List containing the velocities of the particles in the simulation box.

    position_center_history: list(list(array))
        It is a list containing the list of the positions of all particles in the simulation
        for each time step.
    """

    def __init__(
        self,
        max_residue_per_particle,
        max_step,
        max_steps_to_relax,
        dt,
        min_distance,
        type_init_conf,
        save_history,
    ):
        """
        Initialize the MolecularDynamicsSimulation class.

        Setting the options governing how the molecular dynamics simulation will be run.

        Parameters
        ----------
        max_residue_per_particle: float
            Maximum allowable overlap residue between particles

        max_step: int
            Maxium number of time steps

        max_steps_to_relax: int
            Number of time steps a configuration has to remain legal to be accepted.

        dt: float
            Time step for the intergration of the equations of motion.

        min_distance: float
            Minimum distance between particles.

        type_init_conf: {'random', 'grid'}
            Type of initial configuration used.

        save_history: bool
            Save all the trajectories of the particles, the history of the relative and
            kinetic energy.
        """
        self.box = None
        self.particle_velocities = None
        self.particle_forces = None
        self.particle_overlap_areas = None
        self.total_overlap = None
        self.total_overlap_history = []
        self.position_center_history = None
        self.speed_up_scheme = None
        self.thermostat = None
        self.min_distance = min_distance
        self.type_init_conf = type_init_conf
        self.max_residue_per_particle = max_residue_per_particle
        self.max_residue = None
        self.max_step = max_step
        self.max_steps_to_relax = max_steps_to_relax
        self.dt = dt
        self.save_history = save_history

    def generate_microstructure(self, microstructure_sample):
        """
        Generate the microstructure for the sample supplied.

        Generate the microstructure for microstructure_sample using the microstucutre
        generation method *self*.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure sample to be generated
        """
        particles = []
        for phase in microstructure_sample.phases.values():
            particles.append(
                self.generate_particles(
                    microstructure_sample.rve_dims,
                    phase.type,
                    phase.phase_name,
                    phase.descriptors,
                )
            )
        # Generating the particles
        self.set_box(particles, microstructure_sample)
        self.generateInitialConfiguration(particles)
        self.run_molecular_dynamics_simulation(particles)

    def set_box(self, particles, rve_dims):
        """
        Set the dimensions of the simulation box.

        Set the dimensions of the box according to the particles present and the
        dimensions of the microstructure.

        It is assumed that there are no incompatible particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            List of particles in the simulation.

        rve_dims: list(floats)
            Dimensions of the microstucutre in each spatial direction.
        """
        if any(
            [
                particle.__class__.__name__ == "CylindricalFiber"
                for particle in particles
            ]
        ):
            ax_ind = {"x": 0, "y": 1, "z": 2}
            self.box = rve_dims
            del self.box[ax_ind[particles[0].direction_fibers]]
        else:
            self.box = rve_dims

    def generate_initial_configuration(self, particles):
        """
        Generate the initial configuration for particles.

        It sets the position of the center of all the particles in the simulation box, as
        well the velocity of the particles. The initial configuration is stored.

        Parameters
        ----------
        particles: list(particles)
            Particles in the simulation box.
        """
        self.position_center_history = [np.zeros(len(particles), dtype=object)]
        if self.type_init_conf == "random":
            # Random configuration for the particle centers and the zero velocity
            # np.random.seed(42)
            for i_ind, i_particle in enumerate(particles):
                # Running through all the particles
                i_particle.position_center = self.box * np.random.uniform(
                    size=i_particle.dim
                )
                # Generating the positions from a random uniform distribution
                self.particle_velocities = np.zeros((i_particle.dim))
                self.position_center_history[0][i_ind] = i_particle.position_center
                # Saving initial configuration
        elif self.type_init_conf == "grid":
            # Particles randomly assigned to a place in a grid constructed to have an equal
            # number of cells in each direction and a total number of cells larger than the
            # number of particles
            if particles[0].dim == 3:
                n_cells_side = np.int(np.ceil(np.cbrt(len(particles))))
                # Number of cells in each direction
                cell_length = self.box / n_cells_side
                # Length of the cells in each direction
                k_counter = 0
                # Initializing the counter
                grid_places = np.arange(n_cells_side ** 3)
                # Label of each grid place
                np.random.shuffle(grid_places)
                # Distributing the particles randomly to different cells of the grid
                for x_cell in range(n_cells_side):
                    for y_cell in range(n_cells_side):
                        for z_cell in range(n_cells_side):
                            if grid_places[k_counter] < len(particles):
                                particles[
                                    grid_places[k_counter]
                                ].position_center = np.array(
                                    [
                                        x_cell * cell_length[0] + cell_length[0] / 2,
                                        y_cell * cell_length[1] + cell_length[1] / 2,
                                        z_cell * cell_length[2] + cell_length[2] / 2,
                                    ]
                                )

                                # Generating the positions from a random uniform
                                # distribution between
                                self.particle_velocities = np.random.uniform(
                                    low=-0.1, high=0.1, size=3
                                )
                                self.position_center_history[0][
                                    grid_places[k_counter]
                                ] = particles[grid_places[k_counter]].position_center
                                # Saving particle history
                            k_counter += 1
            elif particles[0].dim == 2:
                n_cells_side = np.int(np.ceil(np.sqrt(len(particles))))
                # Number of cells in each direction
                cell_length = self.box / n_cells_side
                # Length of the cells in each direction
                k_counter = 0
                # Initializing the counter
                grid_places = np.arange(n_cells_side ** 2)
                # Label of each grid place
                np.random.shuffle(grid_places)
                # Distributing the particles randomly to different cells of the grid
                for x_cell in range(n_cells_side):
                    for y_cell in range(n_cells_side):
                        if grid_places[k_counter] < len(particles):
                            particles[
                                grid_places[k_counter]
                            ].position_center = np.array(
                                [
                                    x_cell * cell_length[0] + cell_length[0] / 2,
                                    y_cell * cell_length[1] + cell_length[1] / 2,
                                ]
                            )
                            # Generating the positions from a random uniform distribution
                            self.particle_velocities = np.random.uniform(
                                low=-0.1, high=0.1, size=2
                            )
                            self.position_center_history[0][
                                grid_places[k_counter]
                            ] = particles[grid_places[k_counter]].position_center
                            # # Saving particle history
                        k_counter += 1
        else:
            try:
                raise errors.UnsupportedInitialConfigurationType(type_init_conf)
            except errors.UnsupportedInitialConfigurationType as error:
                error.message()
                quit()

    def run_molecular_dynamics_simulation(
        particles, max_residue_per_particle, max_step, options
    ):
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
