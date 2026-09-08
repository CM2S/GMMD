"""Module for the GenerationMethod abstract class."""

import abc
import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import (
    Particle,
    Point,
    Line,
)


class GenerationMethod(abc.ABC):
    """This is the abstract class for generation methods."""

    @abc.abstractmethod
    def generate_microstructure(self, microstructure_sample):
        """Generate a microstructure."""

    def compute_rve_offset(self, particles, rve_dims):
        """Compute the origin of the RVE to minimize tangent particles to the boundaries.

        This function computes the furthest point of each particle in all cartesian
        coordinates directions using the corresponding support functions.
        Then, for each direction, chooses the point where the distance between boundaries is
        largest, so that the FEM mesh is the least distorted possible.
        It also tries to maximize the distance to from the new origin point to the surface
        of any of the particles.

        Paramaters
        ----------
        particles: list(`.Particle`)
            List of particles in the microstructure.

        rve_dims: list(float)
            List of floats.

        Returns
        -------
        offset: list(float)
            Location of the new origin.

        """

        offset = [0, 0, 0]
        all_lim_sort = [[], [], []]
        dist = [0, 0, 0]
        sort_max = [0, 0, 0]
        # Initializing variables
        for i_dim in range(particles[0].dim):
            dir_axis = np.array([0, 0, 0])
            dir_axis[i_dim] = 1
            all_lim = []
            for i_particle in particles:
                # Collecting all the limits of the particles in each direction, each
                # wrapped onto the periodic circle [0, rve_dims[i_dim]). The cell is
                # periodic, so a cutting plane only ever sees the wrapped positions;
                # mixing wrapped and unwrapped limits makes the gaps below meaningless.
                all_lim += [
                    i_particle.support_function(dir_axis)[i_dim] % rve_dims[i_dim],
                    i_particle.support_function(-dir_axis)[i_dim] % rve_dims[i_dim],
                ]
            all_lim_sort[i_dim] = np.unique(all_lim)
            # Sorted unique limits for the i_dim dimension
            dist[i_dim] = np.diff(
                np.append(all_lim_sort[i_dim], all_lim_sort[i_dim][0] + rve_dims[i_dim])
            )
            # Distance between consecutive limits ON THE CIRCLE, so that the gap that
            # wraps from the last limit round to the first one is included too
            sort_max[i_dim] = np.argsort(dist[i_dim])
            # Getting the indices sorting the distances from smallest to largest
        k_ind_sort = [1, 1, 1]
        # Initializing the list for the current indices of the sort vector
        for i_dim in range(particles[0].dim):
            while True:
                if k_ind_sort[i_dim] > len(dist[i_dim]):
                    # every gap has been tried; keep the widest one
                    k_ind_sort[i_dim] = 1
                i_gap = sort_max[i_dim][-k_ind_sort[i_dim]]
                offset[i_dim] = (
                    all_lim_sort[i_dim][i_gap] + dist[i_dim][i_gap] / 2
                ) % rve_dims[i_dim]
                # offset is the midpoint of the gap, taken on the circle, which puts
                # the RVE face as far as it can be from the nearest particle extreme
                # along this direction.
                if 0 < offset[i_dim] < rve_dims[i_dim]:
                    # accept the current offset if it is inside the simulation box
                    # else move to the next
                    break
                k_ind_sort[i_dim] += 1

        return offset
