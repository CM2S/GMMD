"""
Module containing the classes used in the speed up schemes for RSA simulation.

It includes a naive scheme and a cell list class.
"""

import abc
from functools import cached_property
from copy import deepcopy

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import Particle

import numpy as np


class RSA_SpeedUpScheme(abc.ABC):
    """Abstract class for the speed up schemes."""

    @abc.abstractmethod
    def new_list(self, particles):
        """Compute a new list for RSA simulation."""


class CellList(RSA_SpeedUpScheme):
    """Class for the cell list speed up scheme for checking particle intersection.

    Attributes
    ----------
    max_radius: float
        Maximum radius of the all the circumscribed disks/spheres to the particles in the
        simulation box.

    rsa_sim: `.RSA_Simulation`
        RSA simulation using the cell list for intersection computation.

    particle_list: list(set)
        List containing the set of particles in the neighborhood of each particle.

    cell_list: list(set)
        List containing the set of particles in each cell.

    pos_cell_list: list(int)
        List containing the cell location for each particle.
    """

    def __init__(self):
        """Initialize a cell list for the *rsa_sim* acting on *particles."""
        self.rsa_sim = None
        self.max_radius = None
        # Saving the maximum radius of the circunscribing disk/sphere
        self.particle_list = None
        self.cell_particle_list = None
        self.cell_list = None
        self.pos_cell_list = []
        


    @cached_property
    def max_radius(self):
        """Get the maximum circunscribed radius of the particles in the simulation box.    
        Only used in the beggining of the simulation where there is a single particle in the simulation box."""
        self.max_radius = self.rsa_sim.microstructure_sample.particles[0].radius
        return self.max_radius
    
    @cached_property
    def box(self):
        """Get the dimensions of the simulation box."""
        if self.rsa_sim is not None:
            box = self.rsa_sim.box
        else:
            box = None
        return box

    @cached_property
    def n_cell_dim(self):
        """List containing the number of cells in each direction."""
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        n_cell_dim = [
            np.int_(np.floor(self.box[i_dim] / (2 * self.max_radius)))
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

    
    def get_particle_cell(self, particle):
        """Get the index of the cell in which the particle is located."""
        pos_cell_list_dim = []
        for j_dim in range(particle.dim):
            pos_cell_list_dim.append(
                int(particle.position_center[j_dim] // self.cell_side_length[j_dim])
            )
        if particle.dim == 2:
            cell_index = (
                pos_cell_list_dim[0] + pos_cell_list_dim[1] * self.n_cell_dim[0]
            )
        elif particle.dim == 3:
            cell_index = (
                pos_cell_list_dim[0]
                + pos_cell_list_dim[1] * self.n_cell_dim[0]
                + pos_cell_list_dim[2] * self.n_cell_dim[0] * self.n_cell_dim[1]
            )
        return cell_index


    def new_list(self, particles,new_particle):
        """List of particles in the neighborhood of the new particle."""
        dim = particles[0].dim

        if self.max_radius is None:
            self.max_radius = np.max(
                np.array([particle.radius for particle in particles])
            )
        
        if self.cell_list is None:
            self.cell_list = [set() for _ in range(np.prod(self.n_cell_dim))]

        # Number of particles in the simulation box
        number_particles = len(self.rsa_sim.microstructure_sample.particles)
        # Number of particles in the cell list
        number_particles_cell_list = sum( len(set) for set in self.cell_list )
        # If a particle was added to the microstructure in the previous iteration, update the cell list.
        # Otherwise, the cell list remains the same.
        if number_particles_cell_list != number_particles:
            particle_cell_index = self.get_particle_cell(particles[-1])
            self.cell_list[particle_cell_index].add(len(particles)-1)

        # Get the cell index of the new particle
        new_particle_cell_index = self.get_particle_cell(new_particle)
        # Get the cell indexes of the neighbor cells
        neighbor_cell_index = []
        for i in range(3 ** dim):
            neighbor_cell_index.append(
                self.neighbor_cell(
                    new_particle_cell_index, i, dim, self.n_cell_dim
                )
            )


        # Get the list of particles in the neighbor cells
        self.particle_list = []
        for i_neighbor_cell_index in neighbor_cell_index:
            self.particle_list.append(self.cell_list[i_neighbor_cell_index])
        # Flattens the list of sets into one long list of particle indices
        self.particle_list = [p for particle_set in self.particle_list for p in particle_set]


    
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
            local_row_pos_neigh = np.int_(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = np.int_(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = np.int_(
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
            local_row_pos_neigh = np.int_(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = np.int_(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            local_lay_pos_neigh = np.int_(
                np.mod(np.floor(local_pos_neighbor_cell / 9), 3) - 1
            )
            # Local layer position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = np.int_(
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


class Naive(RSA_SpeedUpScheme):
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

    def new_list(self, particles,new_particle):
        """
        Use all the particles.

        Parameters
        ----------
        Particles in the simulatin box, whose cell list is to be computed.
        """

        # Note: new particle is here because other RSA speed up schemes use it to compute the new list, but for the naive scheme we just ignore it and use all the particles in the simulation box. This way, we can use the same function for all the RSA speed up schemes, which is convenient for the code structure.
        self.particle_list = list(range(len(particles)))



