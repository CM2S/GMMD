"""
Module containing the classes used in the speed up schemes for force computation.

It includes a naive scheme, a cell list class, and a Verlet list class, where the Verlet
list is computed from the cell list.
"""

import abc
from functools import cached_property
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
        List containing the set of particles in the neighboorhood of each particle.

    cell_list: list(set)
        List containing the set of particles in each cell.
    """

        particle_list: list(int)
            List containing the indices of the particles in
        """
        self.molecular_dynamics_sim = molecular_dynamics_sim
        self.max_radius = np.max(np.array([particle.radius for particle in particles]))
        # Saving the maximum radius of the circunscribing disk/sphere
        self.particle_list = None

    @cached_property
    def n_cell_dim(self):
        """List containing the number of cells in each direction."""
        box = self.molecular_dynamics_sim.box
        n_cell_dim = [
            np.int(np.floor(box[i_dim] / (2 * self.max_radius)))
            for i_dim in range(len(box))
        ]
        return n_cell_dim

    @cached_property
    def cell_side_length(self):
        """List containing the length of the cells in each direction."""
        box = self.molecular_dynamics_sim.box
        cell_side_length = [
            box[i_dim] / self.n_cell_dim[i_dim] for i_dim in range(len(box))
        ]
        return cell_side_length

    def new_list(self, particles):
        """
        Compute a new cell list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        dim = particles[0].dim

        n_cells = np.prod(np.array(self.n_cell_dim))

        self.particle_list = [set() for i in range(n_cells)]

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
            self.particle_list[pos_cell_list].add(i_index)
            i_particle.cell_list_position = pos_cell_list

    def neighboor_cell(self, pos_current_cell, local_pos_neighboor_cell, dim, n_cells):
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
            # Local column position of the neighboor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighboor_cell = np.int(
                pos_current_cell
                + local_col_pos_neigh
                + local_row_pos_neigh * n_cells[0]
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
            if (
                np.mod(pos_current_cell + 1, n_cells[0]) == 0
                and local_col_pos_neigh == 1
            ):
                # Right column of the grid
                pos_neighboor_cell = pos_neighboor_cell - n_cells[0]
                # Enforcing the periodic boundary conditions
            elif (
                np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1
            ):
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
            # Local column position of the neighboor, going from -1 to 1 with the origin at
            # the current cell
            local_lay_pos_neigh = np.int(
                np.mod(np.floor(local_pos_neighboor_cell / 9), 3) - 1
            )
            # Local layer position of the neighboor, going from -1 to 1 with the origin at
            # the current cell
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
                - n_cells[1]
                * n_cells[0]
                * (pos_current_cell // (n_cells[1] * n_cells[0]))
                < n_cells[0]
                and local_row_pos_neigh == -1
            ):
                # Lower row of the grid
                pos_neighboor_cell = pos_neighboor_cell + n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            elif (
                pos_current_cell
                - n_cells[1]
                * n_cells[0]
                * (pos_current_cell // (n_cells[1] * n_cells[0]))
                >= n_cells[0] * (n_cells[1] - 1)
                and local_row_pos_neigh == 1
            ):
                # Upper row of the grid
                pos_neighboor_cell = pos_neighboor_cell - n_cells[1] * n_cells[0]
                # Enforcing the periodic boundary conditions
            if (
                np.mod(pos_current_cell + 1, n_cells[0]) == 0
                and local_col_pos_neigh == 1
            ):
                # Right column of the grid
                pos_neighboor_cell = pos_neighboor_cell - n_cells[0]
                # Enforcing the periodic boundary conditions
            elif (
                np.mod(pos_current_cell, n_cells[0]) == 0 and local_col_pos_neigh == -1
            ):
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


class VerletList(CellList):
    """
    Class for the verlet list used to speed up force computation.

    This Verlet list is computed from a cell list to achieve for computation of order
    O(n), where n is the number of particles in the simulation box.

    Attributes
    ----------
    verlet_factor: float
        Multiplicative factor used to compute the neighboorhood of the particle.

    new_verlet_list: bool
        Flag to signal the computation of a new Verlet list.

    cell_list: list(int)
        Cell list used to compute the Verlet list.
    """

    def __init__(self, verlet_factor, **kwargs):
        """
        Initialize a Verlet list.

        Parameters
        ----------
        verlet_factor: float
            Multiplicative factor used to compute the neighboorhood of the particle.

        Keyword Parameters
        ------------------
        Parameters for the initializer of the CellList
        """
        self.verlet_factor = verlet_factor
        # Saving the Verlet radius to compute the Verlet list
        self.new_verlet_list = True
        # Signaling that for the first computation of the forces there is a need to compute
        # a new Verlet list
        self.cell_list = None
        self.molecular_dynamics_sim = None
        self.verlet_neighboorhoods = None
        super().__init__(**kwargs)

    def new_list(self, particles):
        """
        Compute a new verlet list for particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulatin box, whose cell list is to be computed.
        """
        if self.verlet_neighboorhoods is None:
            self.verlet_neighboorhoods = particles
            for i_particle in particles:
                i_particle.dilate((self.verlet_factor - 1) * i_particle.radius)
            # Initializing the displacement_last_verlet
        for i_particle_index, i_particle in enumerate(particles):
            # Computing the displacement of the center of the particle
            if not i_particle.intersection(
                self.verlet_neighboorhoods[i_particle_index]
            ):
                # Checking if the displacement takes the particle out of its neighboorhood
                self.new_verlet_list = True
                break
                # There is a need to compute a new verlet list
        if self.new_verlet_list:
            self.new_verlet_list = False
            super().new_list(particles)
            # Creating the cell list used to compute the Verlet list
            self.cell_list = self.particle_list
            # Saving the cell list
            for i_particle_index, i_particle in enumerate(particles):
                # Running though all the particles
                self.verlet_neighboorhoods[
                    i_particle_index
                ] = i_particle.position_center
                # Updating the position of all the Verlet neighboorhoods to coincide with
                # the particles current position
                self.particle_list = []
                # Resetting the Verlet list of particle i
                for j_particle_index in self.cell_list:
                    # Running through all the particles in the neighboring cell
                    if self.verlet_neighboorhoods[i_particle_index].intersection(
                        self.verlet_neighboorhoods[j_particle_index]
                    ):
                        # If the neighboorhoods of the particles intersect
                        particles[i_particle].particle_list.append(j_particle_index)
                        # Add the particle j_particle to i_particle's Verlet list


class Naive(SpeedUpScheme):
    """
    Class for the verlet list used to speed up force computation.

    This Verlet list is computed from a cell list to achieve for computation of order
    O(n), where n is the number of particles in the simulation box.

    Attributes
    ----------
    verlet_factor: float
        Multiplicative factor used to compute the neighboorhood of the particle.

    new_verlet_list: bool
        Flag to signal the computation of a new Verlet list.
    """

    def new_list(self, particles):
        """
        Use all the particles.

        Parameters
        ----------
        Particles in the simulatin box, whose cell list is to be computed.
        """
        self.particle_list = list(range(len(particles)))
