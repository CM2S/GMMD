"""
Module containing the classes used in the speed up schemes for RSA simulation.

It includes a naive scheme and a cell list class.
"""

import abc

import numpy as np


class SpeedUpScheme(abc.ABC):
    """Abstract class for the speed up schemes.
    
    It only contains a common function used in RSA and MD simulations.

    """

    def neighbor_cell(self, pos_current_cell, local_pos_neighbor_cell, dim, n_cells):
        """
        Compute the global cell position of the neighbor cell. This method is used for the speed up scheme CellList for both MD and RSA simulations.

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
