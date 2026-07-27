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
from geommicgen.micgenmethod.speed_up_schemes.Speed_up_scheme import SpeedUpScheme

import numpy as np


class RSA_SpeedUpScheme(SpeedUpScheme,abc.ABC):
    """Abstract class for the Random Sequential Adsorption speed up schemes."""

    @abc.abstractmethod
    def new_list(self, particles,trial_particle):
        """
        Given a trial particle introduced in the RSA simulation box, this method computes the list of particles against which intersection will be checked.
        """


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
        self.cell_list = None
        self.pos_cell_list = []
        self.number_particles_previous_step = 0
        self.number_particles_current_step = 0
        self.max_radius = None

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
        This method updates the attribute max_radius if the trial particle has a higher radius than max_radius.

        Args:
            particles (list of instances from the class Particle):  All particles in the simulatin box.
            trial_particle (instance from the class Particle): trial particle

        Returns:
            bool: True if there has been an update
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
        It updates the attribute n_cell_dim when called.

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
        It updates the attribute cell_side_legth when called.

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
        """List of particles in the neighborhood of the trial particle."""
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