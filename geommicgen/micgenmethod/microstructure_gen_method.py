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

        def min_dist_to_part(offset, particles, rve_dims):
            """Compute the minum distance from offset to the surface of the particles."""
            offset_array = np.array(offset)
            # list to array
            dim = particles[0].dim
            all_dist = []
            # List containing all the distances from offset to the surface of the particles
            if dim == 2 and False:
                point_offset = Point(dim, "1")
                point_offset.position_center = offset_array[0:dim]
                # Setting up the Point object corresponding to the offset
                for i_particle in particles:
                    pos_offset = Particle.nearest_periodic_image(
                        offset_array[0:dim], i_particle.position_center, rve_dims
                    )
                    # Nearest periodic image for the offset
                    if (
                        np.linalg.norm(pos_offset[:dim] - i_particle.position_center)
                        < i_particle.radius
                    ):
                        # Not considering any particles not intersecting the circumsbribed
                        # circumpherence
                        if i_particle.point_inside(offset_array[:dim], rve_dims):
                            # If the offset is inside the particle this will be the smallest
                            # distance
                            dist, _ = i_particle.intersection_length_mink_diff(
                                point_offset, rve_dims
                            )
                            all_dist = [dist]
                            break

                        dist, _ = i_particle.intersection_length_mink_diff(
                            point_offset, rve_dims, 
                        )
                        all_dist.append(dist)
            elif dim == 3:
                pass
                # In 3D the computation of a the from the line to the surface of the
                # particles is not working properly when the line intersects the particle

                # edge_1 = Line("1", 0)
                # edge_2 = Line("1", 1)
                # edge_3 = Line("1", 2)
                # edge_1.position_center = offset_array
                # edge_2.position_center = offset_array
                # edge_3.position_center = offset_array
                # edges = [edge_1, edge_2, edge_3]
                # for j_ind_edge, j_edge in enumerate(edges):
                #     indices = [0, 1, 2]
                #     indices.remove(j_ind_edge)
                #     for i_particle in particles:
                #         pos_offset = Particle.nearest_periodic_image(
                #             offset_array, i_particle.position_center, rve_dims
                #         )
                #         if (
                #             np.linalg.norm(
                #                 pos_offset[indices]
                #                 - i_particle.position_center[indices]
                #             )
                #             <= i_particle.radius
                #         ):
                #             intersection, overlap, _ = i_particle.intersection_gjk(
                #                 j_edge, rve_dims, out_dist=True
                #             )
                #             if overlap > i_particle.radius / 2 and intersection:
                #                 overlap = 0  # i_particle.radius / 100
                #                 # print(
                #                 #     intersection,
                #                 #     j_ind_edge,
                #                 #     vars(j_edge),
                #                 #     vars(i_particle),
                #                 # )
                #             all_dist.append(overlap)
            min_dist = np.min(all_dist) if len(all_dist) > 0 else np.max(rve_dims)

            return min_dist

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
                # Collecting all the limits of the particles in each direction
                all_lim += [
                    i_particle.support_function(dir_axis)[i_dim],
                    i_particle.support_function(-dir_axis)[i_dim],
                ]
                if np.floor(all_lim[-1] / rve_dims[i_dim]) != 0:
                    all_lim.append(
                        all_lim[-1]
                        - rve_dims[i_dim] * np.floor(all_lim[-1] / rve_dims[i_dim])
                    )
                if np.floor(all_lim[-2] / rve_dims[i_dim]) != 0:
                    all_lim.append(
                        all_lim[-2]
                        - rve_dims[i_dim] * np.floor(all_lim[-2] / rve_dims[i_dim])
                    )
                # Including also the periodic images
            all_lim_sort[i_dim] = np.sort(all_lim)
            # Sorting all the limits for the i_dim dimension
            dist[i_dim] = all_lim_sort[i_dim][1:] - all_lim_sort[i_dim][:-1]
            # Computing the distance between limits
            sort_max[i_dim] = np.argsort(dist[i_dim])
            # Getting the indices sorting the distances from smallest to largest
        k_ind_sort = [1, 1, 1]
        # Initializing the list for the current indices of the sort vector
        while True:
            for i_dim in range(particles[0].dim):
                while True:
                    offset[i_dim] = (
                        all_lim_sort[i_dim][sort_max[i_dim][-k_ind_sort[i_dim]]]
                        + all_lim_sort[i_dim][sort_max[i_dim][-k_ind_sort[i_dim]] + 1]
                    ) / 2
                    # offset is the midpoint between two limits
                    if 0 < offset[i_dim] < rve_dims[i_dim]:
                        # accept the current offset if it is inside the simulation box
                        # else move to the next
                        break
                    k_ind_sort[i_dim] += 1

            min_dist = min_dist_to_part(offset, particles, rve_dims)
            # Compute the minimum distance for the current point to the surface of all the
            # particles.
            if (
                min_dist
                > np.min(
                    [
                        dist[i_dim][sort_max[i_dim][-k_ind_sort[i_dim]]]
                        for i_dim in range(particles[0].dim)
                    ]
                )
                / 2
            ) or any([k_ind_sort[i_dim] > 10 for i_dim in range(particles[0].dim)]):
                # Accept the current offset if the minimum distance is smaller than half the
                # distance to any of the limits

                # self.mesh_size_min = (
                #     np.min(
                #         [self.mesh_size_min, min_dist]
                #         + [
                #             dist[i_dim][sort_max[i_dim][-k_ind_sort[i_dim]]]
                #             for i_dim in range(particles[0].dim)
                #         ]
                #     )
                #     / 2
                # )
                break
            ind_inc = np.argmax(
                [
                    dist[i_dim][sort_max[i_dim][-k_ind_sort[i_dim] - 1]]
                    for i_dim in range(particles[0].dim)
                ]
            )
            k_ind_sort[ind_inc] += 1
            # Increase the index whose next distance is the largest
        return offset
