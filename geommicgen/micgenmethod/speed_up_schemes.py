import abc
from functools import cached_property
import numpy as np


class SpeedUpScheme(abc.ABC):
    """
    Abstract class for the speed up schemes.
    """

    @abc.abstractmethod
    def new_list(self, particles):
        pass


class CellList(SpeedUpScheme):
    """Class for the cell list speed up scheme for force computation.

    Attributes
    ----------
    max_radius: float
        Maximum radius of the all the circumscribed disks/spheres to the particles in the
        simulation box.

    molecular_dynamics_sim: `.MolecularDynamicsSimulation`
        Molecular dynamics simulation usign the cell list for force computation.
    """

    def __init__(self, molecular_dynamics_sim, particles):
        """
        Initialize a cell list for the *molecular_dynamics_sim* acting on *particles*.

        Parameters
        ----------
        molecular_dynamics_sim: `.MolecularDynamicsSimulation`
            Molecular dynamics simulation usign the cell list for force computation.

        particles: list(`.Particle`)
            List containing the particles in the simulation box.
        """
        self.molecular_dynamics_sim = molecular_dynamics_sim
        self.max_radius = np.max(np.array([particle.radius for particle in particles]))
        # Saving the maximum radius of the circunscribing disk/sphere
        self.cell_list = None

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
        Particles in the simulatin box, whose cell list is to be computed.
        """
        dim = particles[0].dim

        n_cells = np.prod(np.array(self.n_cell_dim))

        self.cell_list = [set() for i in range(n_cells)]

        for i_particle in particles:
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
            self.cell_list[pos_cell_list].add(i_particle)


class VerletList(SpeedUpScheme):
    def new_list(self, particles):
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
                        if particles[i_particle].intersectionVerlet(
                            particles[j_particle]
                        ):
                            # If the neighboorhoods of the particles intersect
                            particles[i_particle].verlet_list.append(j_particle)
                            # Add the particle j_particle to i_particle's Verlet list
            elif dim == 3:
                # 3D problem
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
                        if particles[i_particle].intersectionVerlet(
                            particles[j_particle]
                        ):
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


class Naive(SpeedUpScheme):
    def new_list(self, particles):
        pass

    # if speed_up_scheme == "Cell":
    #     # Only a cell list scheme will be used
    #     max_radius = np.max(np.array([particles[i].radius for i in range(N)]))
    #     # Saving the maximum radius of the circunscribing disk/sphere
    #     Particle.n_cell_dim = [
    #         np.int(np.round(box[i_dim] / (2 * max_radius))) for i_dim in range(dim)
    #     ]
    #     # Obtaining a list containing the number of cells in each direction
    #     n_cells = np.prod(Particle.n_cell_dim)
    #     # Obtaining the total number of cells
    #     Particle.cell_list = [[] for i in range(n_cells)]
    #     # Initializing the cell list
    #     Particle.cell_side_length = [
    #         box[i_dim] / Particle.n_cell_dim[i_dim] for i_dim in range(dim)
    #     ]
    #     # Obtaining a list containing the dimensions of the cell in each direction
    # elif speed_up_scheme == "Verlet":
    #     # A Verlet list combined with a cell list scheme will be used
    #     Particle.verlet_factor = options["verlet_factor"]
    #     # Saving the Verlet radius to compute the Verlet list
    #     Particle.new_verlet_list = True
    #     # Signaling that for the first computation of the forces there is a need to compute
    #     # a new Verlet list
    #     max_radius = (
    #         np.max(np.array([particles[i].radius for i in range(Particle.number)]))
    #         * Particle.verlet_factor
    #     )
    #     # Saving the maximum radius of the circunscribing disk/sphere accounting for the
    #     # Verlet factor
    #     Particle.n_cell_dim = [
    #         np.int(np.round(box[i_dim] / (2 * max_radius))) for i_dim in range(dim)
    #     ]
    #     # Obtaining a list containing the number of cells in each direction
    #     n_cells = np.prod(Particle.n_cell_dim)
    #     # Obtaining the total number of cells
    #     Particle.cell_list = [[] for i in range(n_cells)]
    #     # Initializing the cell list
    #     Particle.cell_side_length = [
    #         box[i_dim] / Particle.n_cell_dim[i_dim] for i_dim in range(dim)
    #     ]
    #     # Obtaining a list containing the dimensions of the cell in each direction
    # else:
    #     # A naive approach will be used
    #     pass
