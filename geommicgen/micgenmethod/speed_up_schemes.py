"""
Module containing the classes used in the speed up schemes for force computation.

It includes a naive scheme, a cell list class, and a Verlet list class, where the Verlet
list is computed from the cell list.
"""

import abc
from functools import cached_property
from copy import deepcopy

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import Particle

import numpy as np


class SpeedUpScheme(abc.ABC):
    """Abstract class for the speed up schemes."""

    @abc.abstractmethod
    def new_list(self, particles):
        """Compute a new list for force computation."""


class CellList(SpeedUpScheme):
    """Class for the cell list speed up scheme for force computation.

    Attributes
    ----------
    max_radius: float
        Maximum radius of the all the circumscribed disks/spheres to the particles in the
        simulation box.

    molecular_dynamics_sim: `.MolecularDynamicsSimulation`
        Molecular dynamics simulation usign the cell list for force computation.

    particle_list: list(set)
        List containing the set of particles in the neighborhood of each particle.

    cell_list: list(set)
        List containing the set of particles in each cell.

    pos_cell_list: list(int)
        List containing the cell location for each particle.
    """

    def __init__(self):
        """Initialize a cell list for the *molecular_dynamics_sim* acting on *particles."""
        self.molecular_dynamics_sim = None
        self.max_radius = None
        # Saving the maximum radius of the circunscribing disk/sphere
        self.particle_list = None
        self.cell_particle_list = None
        self.cell_list = None
        self.pos_cell_list = []

    @cached_property
    def n_cell_dim(self):
        """List containing the number of cells in each direction."""
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        n_cell_dim = [
            np.int(np.floor(self.box[i_dim] / (2 * self.max_radius)))
            for i_dim in range(len(self.box))
        ]
        return n_cell_dim

    @cached_property
    def cell_side_length(self):
        """List containing the length of the cells in each direction."""
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        cell_side_length = [
            self.box[i_dim] / self.n_cell_dim[i_dim] for i_dim in range(len(self.box))
        ]
        return cell_side_length

    @cached_property
    def box(self):
        """List containing the dimensions of the simulation box."""
        if self.molecular_dynamics_sim is not None:
            box = self.molecular_dynamics_sim.box
        else:
            box = None

        return box

    def new_list(self, particles, particle_rescale_factor=1):
        """
        Compute a new cell list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        dim = particles[0].dim

        if self.max_radius is None:
            self.max_radius = np.max(
                np.array([particle.radius for particle in particles])
            )
            self.max_radius *= particle_rescale_factor
        n_cells = np.prod(np.array(self.n_cell_dim))
        self.cell_list = [set() for i in range(n_cells)]
        self.particle_list = [set() for _ in particles]
        self.pos_cell_list = [None for _ in particles]
        for i_index, i_particle in enumerate(particles):
            # Running through all the particles
            pos_cell_list_dim = []
            # Initializing the list containing the position of the cell in each direction
            # with the origin at the top left
            for j_dim in range(dim):
                # Running through all the dimensions
                pos_cell_list_dim.append(
                    int(
                        i_particle.position_center[j_dim]
                        // self.cell_side_length[j_dim]
                    )
                )
                # j_dim-position of the particle in the grid
            if dim == 2:
                # 2D problem
                pos_cell_list = (
                    pos_cell_list_dim[0] + pos_cell_list_dim[1] * self.n_cell_dim[0]
                )
                # Saving the position in the cell list of particle i_particle
            if dim == 3:
                # 3D problem
                pos_cell_list = (
                    pos_cell_list_dim[0]
                    + pos_cell_list_dim[1] * self.n_cell_dim[0]
                    + pos_cell_list_dim[2] * self.n_cell_dim[0] * self.n_cell_dim[1]
                )
                # Saving the position in the cell list of particle i_particle
            self.cell_list[pos_cell_list].add(i_index)
            self.pos_cell_list[i_index] = pos_cell_list
        for i_particle_index, _ in enumerate(particles):
            for k_neighbor_cell in range(3 ** dim):
                # Running through the neighbor cells
                pos_neighbor_cell = self.neighbor_cell(
                    self.pos_cell_list[i_particle_index],
                    k_neighbor_cell,
                    dim,
                    self.n_cell_dim,
                )
                # Computing the index of the neighbor cell
                for j_particle_index in self.cell_list[pos_neighbor_cell]:
                    # if j_particle_index > i_particle_index:
                    # Running through all the particles in the neighboring cell
                    # If the neighborhoods of the particles intersect
                    self.particle_list[i_particle_index].add(j_particle_index)
                    # Add the particle j_particle to i_particle's Verlet list

    def new_list_partial(self, particles, lists_to_recalc):
        """
        Compute a new cell list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        # Initialization
        # ----------------------------------------------------------------------------------
        dim = particles[0].dim

        if self.max_radius is None:
            self.max_radius = np.max(
                np.array([particle.radius for particle in particles])
            )
        n_cells = np.prod(np.array(self.n_cell_dim))
        if self.cell_list is None:
            self.cell_list = [set() for i in range(n_cells)]
            self.cell_particle_list = [set() for _ in particles]
            self.pos_cell_list = [None for _ in particles]
        pos_cell_list_old = list(self.pos_cell_list)

        # Obtaining the cell position of the particle lists to recalculate
        # ----------------------------------------------------------------------------------
        for i_index, i_particle in enumerate(particles):
            if i_particle not in lists_to_recalc:
                continue
            # Running through all the particles
            pos_cell_list_dim = []
            # Initializing the list containing the position of the cell in each direction
            # with the origin at the top left
            for j_dim in range(dim):
                # Running through all the dimensions
                pos_cell_list_dim.append(
                    int(
                        i_particle.position_center[j_dim]
                        // self.cell_side_length[j_dim]
                    )
                )
                # j_dim-position of the particle in the grid
            if dim == 2:
                # 2D problem
                pos_cell_list = (
                    pos_cell_list_dim[0] + pos_cell_list_dim[1] * self.n_cell_dim[0]
                )
                # Saving the position in the cell list of particle i_particle
            if dim == 3:
                # 3D problem
                pos_cell_list = (
                    pos_cell_list_dim[0]
                    + pos_cell_list_dim[1] * self.n_cell_dim[0]
                    + pos_cell_list_dim[2] * self.n_cell_dim[0] * self.n_cell_dim[1]
                )
                # Saving the position in the cell list of particle i_particle
            if self.pos_cell_list[i_index] is not None:
                self.cell_list[self.pos_cell_list[i_index]].remove(i_index)

            self.pos_cell_list[i_index] = pos_cell_list
            self.cell_list[pos_cell_list].add(i_index)

        # Updating the particle list
        # ----------------------------------------------------------------------------------
        already_rem = set()
        for i_particle_index, i_particle in enumerate(particles):
            if (
                self.pos_cell_list[i_particle_index]
                == pos_cell_list_old[i_particle_index]
                or i_particle not in lists_to_recalc
            ):
                # No update needed
                continue
            already_rem.add(i_particle_index)
            if pos_cell_list_old[i_particle_index] is not None:
                # Removing old
                for k_neighbor_cell in range(3 ** dim):
                    # Running through the neighbor cells
                    pos_neighbor_cell = self.neighbor_cell(
                        pos_cell_list_old[i_particle_index],
                        k_neighbor_cell,
                        dim,
                        self.n_cell_dim,
                    )
                    # Computing the index of the neighbor cell
                    for j_particle_index in self.cell_list[pos_neighbor_cell]:
                        if (
                            i_particle_index > j_particle_index
                            or i_particle_index
                            in self.cell_particle_list[j_particle_index]
                        ) and j_particle_index not in already_rem:
                            self.cell_particle_list[j_particle_index].remove(
                                i_particle_index
                            )
            # Adding new
            for k_neighbor_cell in range(3 ** dim):
                # Running through the neighbor cells
                pos_neighbor_cell = self.neighbor_cell(
                    self.pos_cell_list[i_particle_index],
                    k_neighbor_cell,
                    dim,
                    self.n_cell_dim,
                )
                # Computing the index of the neighbor cell
                for j_particle_index in self.cell_list[pos_neighbor_cell]:
                    if j_particle_index > i_particle_index or True:
                        # Running through all the particles in the neighboring cell
                        # If the neighborhoods of the particles intersect
                        self.cell_particle_list[i_particle_index].add(j_particle_index)
                        # Add the particle j_particle to i_particle's Verlet list
                    if i_particle_index > j_particle_index or True:
                        self.cell_particle_list[j_particle_index].add(i_particle_index)

    def neighbor_cell(self, pos_current_cell, local_pos_neighbor_cell, dim, n_cells):
        """
        Compute the global cell position of the neighbor cell.

        Parameters
        ----------
        pos_current_cell: integer
            Global position of the current cell

        local_pos_neighbor_cell: integer
            Local position of the neighbor cell

        dim: integer
            Dimension of the problem

        n_cells: list
            Number of cells in each direction (0:x; 1:y; 2:z)

        Returns
        -------
        pos_neighbor_cell: integer
            Global position of the neighbor cell
        """

        def at_bottom():
            """Check if the current position is at the bottom of the simulation box."""
            return (
                pos_current_cell
                - n_cells[1]
                * n_cells[0]
                * (pos_current_cell // (n_cells[1] * n_cells[0]))
                < n_cells[0]
            )

        def at_top():
            """Check if the current position is at the top of the simulation box."""
            return pos_current_cell - n_cells[1] * n_cells[0] * (
                pos_current_cell // (n_cells[1] * n_cells[0])
            ) >= n_cells[0] * (n_cells[1] - 1)

        def at_right():
            """Check if the current position is at the right of the simulation box."""
            return np.mod(pos_current_cell + 1, n_cells[0]) == 0

        def at_left():
            """Check if the current position is at the left of the simulation box."""
            return np.mod(pos_current_cell, n_cells[0]) == 0

        def at_front():
            """Check if the current position is at the front of the simulation box."""
            return pos_current_cell < n_cells[1] * n_cells[0]

        def at_back():
            """Check if the current position is at the back of the simulation box."""
            return pos_current_cell > n_cells[1] * n_cells[0] * (n_cells[2] - 1) - 1

        if dim == 2:
            # 2D problem
            local_row_pos_neigh = np.int(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = np.int(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = np.int(
                pos_current_cell
                + local_col_pos_neigh
                + local_row_pos_neigh * n_cells[0]
            )
            # Global position of the neighbor cell without enforcing periodic boundary
            # conditions
            if pos_current_cell < n_cells[0] and local_row_pos_neigh == -1:
                # Lower row of the grid
                pos_neighbor_cell = pos_neighbor_cell + n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            elif (
                pos_current_cell >= n_cells[0] * (n_cells[1] - 1)
                and local_row_pos_neigh == 1
            ):
                # Upper row of the grid
                pos_neighbor_cell = pos_neighbor_cell - n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            if (
                np.mod(pos_current_cell + 1, n_cells[0]) == 0
                and local_col_pos_neigh == 1
            ):
                # Right column of the grid
                pos_neighbor_cell = pos_neighbor_cell - n_cells[0]
                # Enforcing the periodic boundary conditions
            elif (
                np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1
            ):
                # Left column of the grid
                pos_neighbor_cell = pos_neighbor_cell + n_cells[0]
                # Enforcing the periodic boundary conditions
        elif dim == 3:
            # 3D problem
            local_row_pos_neigh = np.int(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = np.int(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            local_lay_pos_neigh = np.int(
                np.mod(np.floor(local_pos_neighbor_cell / 9), 3) - 1
            )
            # Local layer position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = np.int(
                pos_current_cell
                + local_col_pos_neigh
                + local_row_pos_neigh * n_cells[0]
                + local_lay_pos_neigh * n_cells[0] * n_cells[1]
            )
            # Global position of the neighbor cell without enforcing periodic boundary
            # conditions
            if at_bottom() and local_row_pos_neigh == -1:
                # Lower row of the grid
                pos_neighbor_cell = pos_neighbor_cell + n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            elif at_top() and local_row_pos_neigh == 1:
                # Upper row of the grid
                pos_neighbor_cell = pos_neighbor_cell - n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            if at_right() and local_col_pos_neigh == 1:
                # Right column of the grid
                pos_neighbor_cell = pos_neighbor_cell - n_cells[0]
                # Enforcing the periodic boundary conditions
            elif at_left() and local_col_pos_neigh == -1:
                # Left column of the grid
                pos_neighbor_cell = pos_neighbor_cell + n_cells[0]
                # Enforcing the periodic boundary conditions
            if at_front() and local_lay_pos_neigh == -1:
                # Firsl layer of the grid
                pos_neighbor_cell = (
                    pos_neighbor_cell + n_cells[1] * n_cells[0] * n_cells[2]
                )
                # Enforcing the periodic boundary conditions
            elif at_back() and local_lay_pos_neigh == 1:
                # Last layer of the grid
                pos_neighbor_cell = (
                    pos_neighbor_cell - n_cells[1] * n_cells[0] * n_cells[2]
                )
                # Enforcing the periodic boundary conditions

        return pos_neighbor_cell


class VerletList:
    """
    Class for the verlet list used to speed up force computation.

    This Verlet list is computed from a cell list to achieve for computation of order
    O(n), where n is the number of particles in the simulation box.

    Attributes
    ----------
    verlet_factor: float
        Multiplicative factor used to compute the neighborhood of the particle.

    a_new_verlet_list_has_to_be_computed: bool
        Flag to signal the computation of a new Verlet list.

    verlet_neighborhoods: list(`.Particle`)
        List of Verlet neighborhoods, having the same shape as the corresponding particles,
        but larger.
    """

    def __init__(self, verlet_factor):
        """
        Initialize a Verlet list.

        Parameters
        ----------
        verlet_factor: float
            Multiplicative factor used to compute the neighborhood of the particle.
        """
        self.verlet_factor = verlet_factor
        # Saving the Verlet radius to compute the Verlet list
        self.a_new_verlet_list_has_to_be_computed = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        self.verlet_neighborhoods = None
        self.particle_list = None
        self.cell_list = CellList()
        self.molecular_dynamics_sim = None

    @property
    def box(self):
        """List containing the dimensions of the simulation box."""
        if self.molecular_dynamics_sim is not None:
            box = self.molecular_dynamics_sim.box
        else:
            box = None
        return box

    def new_list(self, particles):
        """
        Compute a new verlet list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        if self.verlet_neighborhoods is None:
            self.verlet_neighborhoods = deepcopy(particles)
            for i_particle_index, i_particle in enumerate(particles):
                self.verlet_neighborhoods[i_particle_index].dilate(
                    (self.verlet_factor - 1) * particles[i_particle_index].radius
                )
            self.cell_list.molecular_dynamics_sim = self.molecular_dynamics_sim
        # if self.cell_list.molecular_dynamics_sim is None:
        for i_particle_index, i_particle in enumerate(particles):
            if self.particle_intersects_its_own_neighborhood(
                i_particle, self.verlet_neighborhoods[i_particle_index]
            ):
                self.a_new_verlet_list_has_to_be_computed = True
                break
        if self.a_new_verlet_list_has_to_be_computed:
            self.a_new_verlet_list_has_to_be_computed = False
            self.cell_list.new_list(self.verlet_neighborhoods)
            self.particle_list = [[] for _ in particles]
            for i_particle_index, i_particle in enumerate(particles):
                self.verlet_neighborhoods[
                    i_particle_index
                ].position_center = i_particle.position_center
            for i_particle_index, i_particle in enumerate(particles):
                for j_particle_index in self.cell_list.particle_list[i_particle_index]:

                    if self.the_verlet_neighborhoods_of_the_particles_intersect(
                        i_particle_index, j_particle_index
                    ):
                        self.particle_list[i_particle_index].append(j_particle_index)

    def particle_intersects_its_own_neighborhood(self, particle, neighborhood):
        """Check if a particle intersects its own neighborhood.

        It assumes the neighborhood is found by dilation of the particle.
        """
        distance_between_center_of_particle_and_verlet_neighborhood = np.linalg.norm(
            particle.position_center
            - Particle.nearest_periodic_image(
                neighborhood.position_center,
                particle.position_center,
                self.box,
            ),
        )
        return (
            distance_between_center_of_particle_and_verlet_neighborhood
            > (self.verlet_factor - 1) * particle.radius
        )

    def the_verlet_neighborhoods_of_the_particles_intersect(
        self, i_particle_index, j_particle_index
    ):
        """Check if two Verlet neighborhoods intersect."""
        if i_particle_index < j_particle_index:
            return self.verlet_neighborhoods[i_particle_index].intersection(
                self.verlet_neighborhoods[j_particle_index], self.box
            )

        return (
            j_particle_index == i_particle_index
            or i_particle_index in self.particle_list[j_particle_index]
        )


class VerletPartialUpdate(VerletList):
    """Class for the Verlet list with partial update (currently not working)."""

    def new_list(self, particles):
        """
        Compute a new verlet list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        if self.verlet_neighborhoods is None:
            self.a_new_verlet_list_has_to_be_computed = True
            self.verlet_neighborhoods = deepcopy(particles)
            self.particle_list = [[] for _ in particles]
            # Resetting the Verlet list of
            for i_particle_index, i_particle in enumerate(particles):
                self.verlet_neighborhoods[i_particle_index].dilate(
                    (self.verlet_factor - 1) * particles[i_particle_index].radius
                )
            self.verlet_neighborhoods_move = deepcopy(particles)
            for i_particle_index, i_particle in enumerate(particles):
                self.verlet_neighborhoods_move[i_particle_index].contract(
                    (2 - self.verlet_factor) * particles[i_particle_index].radius
                )
            lists_to_recalc = set(particles)
            lists_to_recalc_ind = set(range(len(particles)))
            # Initializing the displacement_last_verlet
        else:
            lists_to_recalc = set()
            lists_to_recalc_ind = set()
            # Possible Verlet lists to recalculate
            for i_particle_index, i_particle in enumerate(particles):
                # Computing the displacement of the center of the particle
                if not self.verlet_neighborhoods_move[i_particle_index].point_inside(
                    i_particle.position_center, self.box
                ):
                    # if (
                    #     i_particle.intersection(
                    #         self.verlet_neighborhoods[i_particle_index],
                    #         self.box,
                    #         inside=False,
                    #     )
                    #     or not self.verlet_neighborhoods[i_particle_index].point_inside(
                    #         i_particle.position_center, self.box
                    #     )
                    # ):
                    # Checking if the displacement takes the particle out of its
                    # neighborhood
                    self.a_new_verlet_list_has_to_be_computed = True
                    # There is a need to compute a new verlet list
                    lists_to_recalc.add(i_particle)
                    lists_to_recalc_ind.add(i_particle_index)
        if self.a_new_verlet_list_has_to_be_computed:
            self.a_new_verlet_list_has_to_be_computed = False
            # old_verlet_fac = self.verlet_factor
            # self.verlet_factor = np.max([1.05, old_verlet_fac * 0.95])
            # print(self.verlet_factor, "verlet\n\n")
            # if old_verlet_fac != self.verlet_factor:
            #     for i_particle_index, i_particle in enumerate(particles):
            #         self.verlet_neighborhoods[i_particle_index].contract(
            #             (old_verlet_fac - self.verlet_factor)
            #             * particles[i_particle_index].radius
            #         )

            for i_particle_index, i_particle in enumerate(particles):
                if i_particle not in lists_to_recalc:
                    continue
                # Running though all the particles
                self.verlet_neighborhoods[
                    i_particle_index
                ].position_center = i_particle.position_center
                self.verlet_neighborhoods_move[
                    i_particle_index
                ].position_center = i_particle.position_center
                # Updating the position of all the Verlet neighborhoods to coincide with
                # the particles current position
            super().new_list_partial(particles, lists_to_recalc)
            # print(self.cell_particle_list)
            # Creating the cell list used to compute the Verlet list
            for i_particle_index, i_particle in enumerate(particles):
                # Running though all the particles
                if i_particle not in lists_to_recalc:
                    continue
                self.particle_list[i_particle_index] = []
                for j_particle_index in self.cell_particle_list[i_particle_index]:
                    # Running through all the particles in the neighboring cell
                    if self.verlet_neighborhoods[i_particle_index].intersection(
                        self.verlet_neighborhoods[j_particle_index],
                        self.box,
                    ):
                        # If the neighborhoods of the particles intersect
                        self.particle_list[i_particle_index].append(j_particle_index)
                        self.particle_list[j_particle_index].append(i_particle_index)
                        # Add the particle j_particle to i_particle's Verlet list
                    elif i_particle_index in self.particle_list[j_particle_index]:
                        self.particle_list[j_particle_index].remove(i_particle_index)


class Naive(SpeedUpScheme):
    """
    Class for the verlet list used to speed up force computation.

    This Verlet list is computed from a cell list to achieve for computation of order
    O(n), where n is the number of particles in the simulation box.

    Attributes
    ----------
    verlet_factor: float
        Multiplicative factor used to compute the neighborhood of the particle.

    a_new_verlet_list_has_to_be_computed: bool
        Flag to signal the computation of a new Verlet list.
    """

    def __init__(self):
        """Initialize a Naive class object. It does nothing."""
        self.particle_list = []

    def new_list(self, particles):
        """
        Use all the particles.

        Parameters
        ----------
        Particles in the simulatin box, whose cell list is to be computed.
        """
        self.particle_list = [list(range(len(particles))) for _ in particles]
