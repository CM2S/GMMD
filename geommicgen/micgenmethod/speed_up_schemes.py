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
        self.cell_list = None
        self.pos_cell_list = []

    @cached_property
    def n_cell_dim(self):
        """List containing the number of cells in each direction."""
        if self.box is None:
            raise ValueError("The simulation box has not been defined")
        n_cell_dim = [
            int(np.floor(self.box[i_dim] / (2 * self.max_radius)))
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
            pos_cell_list = self.cell_of(i_particle.position_center)
            self.cell_list[pos_cell_list].add(i_index)
            self.pos_cell_list[i_index] = pos_cell_list
            # Saving the position in the cell list of particle i_particle
        for i_particle_index, _ in enumerate(particles):
            self.particle_list[i_particle_index] = self.candidates(i_particle_index)

    def cell_of(self, position):
        """
        Give the index in the cell list of the cell a position falls in.

        Parameters
        ----------
        position: array
            Position inside the box.

        Returns
        -------
        int
            Index of the cell, counting along the first direction first.
        """
        dim = len(self.n_cell_dim)
        cell = [
            min(int(position[i_dim] // self.cell_side_length[i_dim]), self.n_cell_dim[i_dim] - 1)
            for i_dim in range(dim)
        ]
        index = cell[0] + cell[1] * self.n_cell_dim[0]
        if dim == 3:
            index += cell[2] * self.n_cell_dim[0] * self.n_cell_dim[1]
        # A position on the far face of the box, which the wrapping of a coordinate
        # can round it onto, is counted in the last cell rather than one past it

        return index

    def move(self, index, position):
        """
        Put one particle in the cell its position falls in, taking it out of its old one.

        Parameters
        ----------
        index: int
            Index of the particle.

        position: array
            Its position inside the box.
        """
        cell = self.cell_of(position)
        old = self.pos_cell_list[index]
        if cell != old:
            self.cell_list[old].discard(index)
            self.cell_list[cell].add(index)
            self.pos_cell_list[index] = cell
        # The lists of neighbours are not brought up to date: `candidates` reads the
        # cells directly, so a caller that moves particles one at a time asks that

    @cached_property
    def cells_around(self):
        """For each cell, the indices of the cells around it, itself included."""
        dim = len(self.n_cell_dim)
        return [
            sorted(
                {
                    self.neighbor_cell(i_cell, k_neighbor_cell, dim, self.n_cell_dim)
                    for k_neighbor_cell in range(3**dim)
                }
            )
            for i_cell in range(int(np.prod(self.n_cell_dim)))
        ]
        # Worked out once per cell rather than once per particle per rebuild, which
        # was where a rebuild spent most of its time; a box only a cell or two wide
        # has the same cell around itself more than once, hence the set

    def candidates(self, index):
        """
        Give the particles in the cell of one particle and in the cells around it.

        Parameters
        ----------
        index: int
            Index of the particle.

        Returns
        -------
        set
            Indices of the particles that can be near it, itself included.
        """
        found = set()
        for i_cell in self.cells_around[self.pos_cell_list[index]]:
            found |= self.cell_list[i_cell]
        # Read off the cells as they are now, where the lists built by `new_list` say
        # what they were when it ran

        return found

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
            local_row_pos_neigh = int(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = int(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = int(
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
            local_row_pos_neigh = int(
                np.mod(np.floor(local_pos_neighbor_cell / 3), 3) - 1
            )
            # Local row position of the neighbor, going from -1 to 1 with the origin at the
            # current cell
            local_col_pos_neigh = int(np.mod(local_pos_neighbor_cell, 3) - 1)
            # Local column position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            local_lay_pos_neigh = int(
                np.mod(np.floor(local_pos_neighbor_cell / 9), 3) - 1
            )
            # Local layer position of the neighbor, going from -1 to 1 with the origin at
            # the current cell
            pos_neighbor_cell = int(
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

    verlet_neighborhoods: list(`.Particle`)
        List of Verlet neighborhoods, having the same shape as the corresponding particles,
        but larger.

    particle_list: list(set)
        For each particle, the indices of the particles whose neighbourhoods intersect
        its own, itself excluded. Every pair is in both lists.
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
        Bring the Verlet lists up to date with where the particles are.

        The list of a particle is the particles whose neighbourhoods intersect its own,
        and it holds until the particle leaves its neighbourhood. Only the lists of the
        particles that have left are recomputed, unless so many have that rebuilding
        every list is cheaper.

        Parameters
        ----------
        particles: list(`.Particle`)
            Particles in the simulation box.
        """
        if self.verlet_neighborhoods is None:
            self.verlet_neighborhoods = deepcopy(particles)
            for i_particle_index, i_particle in enumerate(particles):
                self.verlet_neighborhoods[i_particle_index].dilate(
                    (self.verlet_factor - 1) * particles[i_particle_index].radius
                )
            self.cell_list.molecular_dynamics_sim = self.molecular_dynamics_sim
            leavers = list(range(len(particles)))
        else:
            leavers = [
                i_particle_index
                for i_particle_index, i_particle in enumerate(particles)
                if self.particle_intersects_its_own_neighborhood(
                    i_particle, self.verlet_neighborhoods[i_particle_index]
                )
            ]
        if not leavers:
            return
        for i_particle_index in leavers:
            self.verlet_neighborhoods[i_particle_index].position_center = particles[
                i_particle_index
            ].position_center.copy()
        # The neighbourhoods of the leavers are moved onto them before anything is
        # looked up, so that they are binned and compared where they are now

        if (
            self.particle_list is None
            or len(leavers) > self.FULL_REBUILD_FRACTION * len(particles)
        ):
            self.cell_list.new_list(self.verlet_neighborhoods)
            self.particle_list = [set() for _ in particles]
            leavers = range(len(particles))
        else:
            for i_particle_index in leavers:
                self.cell_list.move(
                    i_particle_index,
                    self.verlet_neighborhoods[i_particle_index].position_center,
                )
                for j_particle_index in self.particle_list[i_particle_index]:
                    self.particle_list[j_particle_index].discard(i_particle_index)
                self.particle_list[i_particle_index] = set()
        # Rebuilt from scratch when most of the particles have left, early in a run,
        # and one leaver at a time once the run has settled and a step moves a handful
        # of them out: the pairs of a particle that stayed are still the pairs it had

        for i_particle_index in leavers:
            neighborhood = self.verlet_neighborhoods[i_particle_index]
            for j_particle_index in self.cell_list.candidates(i_particle_index):
                if j_particle_index == i_particle_index:
                    continue
                if j_particle_index in self.particle_list[i_particle_index]:
                    continue
                if neighborhood.intersection(
                    self.verlet_neighborhoods[j_particle_index], self.box
                ):
                    self.particle_list[i_particle_index].add(j_particle_index)
                    self.particle_list[j_particle_index].add(i_particle_index)
        # Every pair is put in both lists, which is what the force computation reads:
        # it takes each pair from the list of the lower index

    FULL_REBUILD_FRACTION = 0.5
    # Share of the particles that have to have left their neighbourhoods for a full
    # rebuild to be done instead of recomputing their lists one by one

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


class Naive(SpeedUpScheme):
    """
    Class for the scheme that checks every pair of particles, O(n**2) in their number.

    Attributes
    ----------
    particle_list: list(list)
        For each particle, every other particle: every pair is checked.
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


SPEED_UP_SCHEMES = ("Cell", "Verlet", "Naive")
# The speed up schemes an input data file can ask for, under the names it asks with. A
# Verlet list with partial update, Verlet2, was offered as well and had never worked:
# it called a method that did not exist on the first force computation


def speed_up_scheme_from_options(options):
    """
    Build the speed up scheme an input data file asks for.

    Parameters
    ----------
    options: dict
        Generation parameters, keyed as the input data file keys them.

    Returns
    -------
    `.SpeedUpScheme`
        The scheme, a cell list when none is named.

    Raises
    ------
    ValueError:
        If the scheme is not one there is, or if a Verlet list is asked for without
        its factor.
    """
    name = options.get("speed_up_scheme", SPEED_UP_SCHEMES[0])
    if name == "Cell":
        return CellList()
    if name == "Naive":
        return Naive()
    if name == "Verlet":
        if "verlet_factor" not in options:
            raise ValueError(
                "The input data file does not give verlet_factor, which the Verlet "
                "speed up scheme needs."
            )
        return VerletList(options["verlet_factor"])
    raise ValueError(
        "{0} is not a speed up scheme. The available options are {1}.".format(
            name, ", ".join(SPEED_UP_SCHEMES)
        )
    )
    # A name that is not known is refused here, where the message can list the names,
    # rather than quietly given the cell list as it used to be
