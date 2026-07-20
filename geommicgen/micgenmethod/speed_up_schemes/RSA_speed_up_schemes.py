"""
Module containing the classes used in the speed up schemes for RSA simulation.

It includes a naive scheme and a cell list class.
"""

import abc
from functools import cached_property
from copy import deepcopy
import time

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import Particle
from geommicgen.microstructure.particleclasses import Matrix

import numpy as np


class RSA_SpeedUpScheme(abc.ABC):
    """Abstract class for the speed up schemes.
    Given a trial particle introduced in the simulation box, this class computes the list of particles against which intersection will be checked.
    """

    @abc.abstractmethod
    def new_list(self, particles):
        """Compute a new list for RSA simulation."""


class CellList(RSA_SpeedUpScheme):
    """Class for the cell list speed up scheme for checking particle intersection.

    Attributes
    ----------
    cell_list: list(set)
        List containing the set of particles in each cell.

    pos_cell_list: list(int)
        List containing the cell location for each particle.

    number_particles_previous_step: int
        Number of particles in the previous step

    number_particles_current_step: int
        Number of particles in the currrent step
    
    max_radius: float
        Maximum radius of the all the circumscribed disks/spheres to the particles in the
        simulation box.

    n_cell_dim: list
        List containing the number of cells in each direction.

    cell_side_length: list
        List containing the length of the cells in each direction.

    rsa_sim: `.RSA_Simulation`
        RSA simulation using the cell list for intersection computation.

    particle_list: list(set)
        List containing the set of particles in the neighborhood of each particle.


    """

    def __init__(self):
        """Initialize a cell list for the *rsa_sim* acting on *particles."""
        self.rsa_sim = None
        self.particle_list = None
        #self.cell_particle_list = None
        self.cell_list = None
        self.pos_cell_list = []
        self.number_particles_previous_step = 0
        self.number_particles_current_step = 0
        self.max_radius = None
        self.n_cell_dim = []
        self.cell_side_length = []

    @cached_property
    def box(self):
        """Get the dimensions of the simulation box."""
        if self.rsa_sim is not None:
            box = self.rsa_sim.box
        else:
            box = None
        return box

    def update_max_radius(self,particles,trial_particle):
        """
        If the trial particle has a higher radius than max_radius, max_radius is updated.
        It returns True if there has been an update.
        """
        if self.max_radius is None:
            self.max_radius = max(p.radius for p in particles)
            return True       
        if trial_particle.radius > self.max_radius:
            self.max_radius = trial_particle.radius
            return True
        return False

    
    def update_n_cell_dim(self):
        """
        It is called when the trial particle is bigger than the cells.
        It updates the variable n_cell_dim when called.
        It returns nothing.
        """
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        self.n_cell_dim = [
            np.int_(np.floor(self.box[i_dim] / (2 * self.max_radius)))
            for i_dim in range(len(self.box))
        ]


    def update_cell_side_length(self):
        """
        It is called when the trial particle is bigger than the cells.
        It updates the variable self.sell_side_legth when called.
        self.cell_side_lenght is a list containing the length of the cells in each direction.
        It returns nothing.
        """
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        cell_side_length = [
            self.box[i_dim] / self.n_cell_dim[i_dim] for i_dim in range(len(self.box))
        ]
        self.cell_side_length = cell_side_length

    
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

    def update_cell_list(self,particles):
        """
        When the trial particle is bigger than the cells, the cell size is updated.
        This method assigns the new cells for the particles already in the simulation box, it updates cell_list.
        It returns nothing.
        """
        self.cell_list = [set() for _ in range(np.prod(self.n_cell_dim))]

        for idx,particle in enumerate(particles):
            particle_cell_index = self.get_particle_cell(particle)
            self.cell_list[particle_cell_index].add(idx)


    def new_list(self, particles,trial_particle):
        """List of particles in the neighborhood of the new particle."""
        # Uncomment the lines bellow if I am running Delete_later/plot.py
        # start = time.perf_counter()

        dim = particles[0].dim
        max_radius_was_updated = self.update_max_radius(particles,trial_particle)

        if max_radius_was_updated:
            self.update_n_cell_dim()
            self.update_cell_side_length()
            self.update_cell_list(particles)

        number_particles_current_step = len(particles)

        if not max_radius_was_updated:
            # Update the cell list only if a particle was added.
            if number_particles_current_step!= self.number_particles_previous_step:
                particle_cell_index = self.get_particle_cell(particles[-1])
                self.cell_list[particle_cell_index].add(len(particles)-1)

        # Get the cell index of the new particle
        trial_particle_cell_index = self.get_particle_cell(trial_particle)
        # Get the cell indexes of the neighbor cells
        neighbor_cells_index = [self.neighbor_cell(trial_particle_cell_index, i, dim, self.n_cell_dim) 
                          for i in range(3**dim)]

        # Get the list of particles in the neighbor cells
        self.particle_list = []
        extend = self.particle_list.extend
        for idx in neighbor_cells_index:
            extend(self.cell_list[idx])

        self.number_particles_previous_step = number_particles_current_step
        
        # Uncomment the lines bellow if I am running Delete_later/plot.py
        # duration = time.perf_counter() - start
        # with open("Cell_new_list_times.txt", "a") as f:
        #         f.write(f", {duration}")


    
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
    Class for the Naive speed up scheme.

    The trial particle is checked for intersection against all the particles in the simulation box. The entire Random Sequential Adsorption method has a computational complexity of O(n), where n is the number of particles in the simulation box.

    Attributes
    ----------.
    particle_list: list(set)
        List containing the set of particles in the neighborhood of each particle.
    """

    def __init__(self):
        """Initialize a Naive class object. It does nothing."""
        self.particle_list = []

    def new_list(self, particles,trial_particle):
        """
        Creates a list containing all the particles.

        Parameters
        ----------
        Particles in the simulatin box, whose cell list is to be computed.
        """
        # Uncomment the lines bellow if I am running Delete_later/plot.py
        # start = time.perf_counter()
        # Note: new particle is here because other RSA speed up schemes use it to compute the new list, but for the naive scheme we just ignore it and use all the particles in the simulation box. This way, we can use the same function for all the RSA speed up schemes, which is convenient for the code structure.
        self.particle_list = list(range(len(particles)))

        # Uncomment the lines bellow if I am running Delete_later/plot.py
        # duration = time.perf_counter() - start
        # with open("Naive_new_list_times.txt", "a") as f:
        #     f.write(f", {duration}")