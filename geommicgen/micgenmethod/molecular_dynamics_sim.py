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
