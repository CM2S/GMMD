"""Module for the GenerationMethod abstract class."""

import abc
from contextlib import contextmanager
import time
import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import (
    Particle,
    Point,
    Line,
)
import geommicgen.iofuncs.printing as print_funcs
import geommicgen.postproc.plotfuncs.plotting_functions as plot_funcs


class GenerationMethod(abc.ABC):
    """This is the abstract class for generation methods."""

    box_scale = 1
    # Factor the method scaled its box by; one for a method that does not scale it

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
        for i_dim in range(particles[0].dim):
            dir_axis = np.array([0, 0, 0])
            dir_axis[i_dim] = 1
            extents = [
                (
                    i_particle.support_function(-dir_axis)[i_dim],
                    i_particle.support_function(dir_axis)[i_dim],
                )
                for i_particle in particles
            ]
            midpoints = all_lim_sort[i_dim] + dist[i_dim] / 2
            free = [
                not any(
                    j_upper - j_lower >= rve_dims[i_dim]
                    or (i_midpoint - j_lower) % rve_dims[i_dim] < j_upper - j_lower
                    for j_lower, j_upper in extents
                )
                for i_midpoint in midpoints
            ]
            # A gap between two consecutive limits can lie inside a third particle,
            # so a gap is free only when no particle spans its midpoint on the circle
            candidates = [i_gap for i_gap in sort_max[i_dim] if free[i_gap]]
            i_gap = candidates[-1] if candidates else sort_max[i_dim][-1]
            offset[i_dim] = midpoints[i_gap] % rve_dims[i_dim]
            # offset is the midpoint of the widest free gap, taken on the circle, which
            # puts the RVE face as far as it can be from the nearest particle extreme
            # along this direction. When the particles leave no free gap, the widest
            # gap is taken all the same. Taken on the circle the offset is in
            # [0, rve_dims[i_dim]) already, and 0 is an offset like any other, the box
            # left where it is: refusing it for the next gap put the face through the
            # middle of a lone particle

        return offset
    

    def contract_all_particles(self, particles):
        """Contract all the particles in the simulation box by their dilation."""
        for i_particle in particles:
            # Running through all the particles
            i_particle.contract(self.min_distance / 2 * self.box_scale)
            # The half distance as the box scaled it with the dilation; a method that
            # does not scale its box keeps box_scale at one

    def dilate_all_particles(self, particles):
        """Dilate all the particles in the simulation box."""
        for i_particle in particles:
            # Running through all the particles
            i_particle.dilate(self.min_distance / 2)
            # Dilate i_particle

    def set_box(self, particles, rve_dims):
        """
        Set the dimensions of the simulation box.

        Set the dimensions of the box according to the particles present and the
        dimensions of the microstructure.

        It is assumed that there are no incompatible particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            List of particles in the simulation.

        rve_dims: list(floats)
            Dimensions of the microstructure in each spatial direction.
        """
        if any(
            [
                particle.__class__.__name__ == "CylindricalFiber"
                for particle in particles
            ]
        ):
            self.box = list(rve_dims)
            del self.box[particles[0].direction_fibers]
        else:
            self.box = list(rve_dims)


    def place_inner_phase_rsa(self, inner_phase, outer_phase):
        """Place inner phase."""
        start = time.time()
        rve_dims = np.array(inner_phase.microstructure.rve_dims)
        # Pairing inner phase particles w/ outer phase particles
        # ----------------------------------------------------------------------------------
        out_part_in_part = [[] for _ in outer_phase.particles]
        available_volume = [particle.volume for particle in outer_phase.particles]
        for i_particle_ind, i_inner_particle in enumerate(inner_phase.particles):
            while True:
                outer_particle_ind = np.random.choice(
                    list(range(len(outer_phase.particles))),
                    p=[
                        particle.volume
                        / (
                            outer_phase.volume_fraction
                            * inner_phase.microstructure.volume
                        )
                        for particle in outer_phase.particles
                    ],
                )
                if i_inner_particle.volume < 0.5 * available_volume[outer_particle_ind]:
                    available_volume[outer_particle_ind] -= i_inner_particle.volume
                    out_part_in_part[outer_particle_ind].append(i_inner_particle)
                    i_inner_particle.parent = outer_phase.particles[outer_particle_ind]
                    # The outer particle that holds it, which the mesher and the
                    # microstructure file need
                    break
        # Placing inner phase particles in each outer phase particle 1-by-1
        # ----------------------------------------------------------------------------------
        make_gif = getattr(self, "make_gif", False)
        if make_gif:
            microstructure = inner_phase.microstructure
            gif_particles = [
                particle
                for particle in microstructure.particles
                if particle.phase != inner_phase.name
            ]
            # Particles drawn in the gif frames: all but the inner phase being placed
            self.gif_frame = getattr(self, "gif_frame", self.step)
            # Gif frames of the inner phase placement continue after the last frame
        n_placed = 0
        n_steps = 0
        print_funcs.print_to_terminal_refresh_inner_phase(
            inner_phase.name,
            outer_phase.name,
            n_placed,
            len(inner_phase.particles),
            n_steps,
            first=True,
        )
        not_placed_particles = []
        for i_outer_particle, i_list_inner_particles in zip(
            outer_phase.particles, out_part_in_part
        ):
            placed_particles = []
            for j_inner_particle in i_list_inner_particles:
                clearance = 0.01 * j_inner_particle.radius
                j_inner_particle.dilate(clearance)
                i_outer_particle.contract(j_inner_particle.radius * 1.05)
                placed = False
                while n_steps < self.inner_phase_max_step:
                    n_steps += 1
                    # Total number of positions tried so far
                    j_inner_particle.position_center = (
                        i_outer_particle.generate_point_inside()
                    )
                    j_inner_particle.position_center = np.squeeze(
                        j_inner_particle.position_center
                        - rve_dims
                        * np.floor(j_inner_particle.position_center / rve_dims)
                    )
                    intersect = False
                    for k_particle in placed_particles:
                        if k_particle.intersection(j_inner_particle, rve_dims):
                            intersect = True
                            break
                    if not intersect:
                        placed_particles.append(j_inner_particle)
                        placed = True
                        break

                i_outer_particle.dilate(j_inner_particle.radius * 1.05)
                j_inner_particle.contract(clearance)
                # Contracted by what it was dilated by: a hundredth of the radius the
                # dilation had grown left every inner particle a little smaller than
                # generated
                if not placed:
                    not_placed_particles.append(j_inner_particle)
                    continue
                    # The maximum number of steps was reached before placing the particle
                n_placed += 1
                print_funcs.print_to_terminal_refresh_inner_phase(
                    inner_phase.name,
                    outer_phase.name,
                    n_placed,
                    len(inner_phase.particles),
                    n_steps,
                )
                if make_gif:
                    gif_particles.append(j_inner_particle)
                    self.gif_frame += 1
                    plot_funcs.plot_particles(
                        gif_particles,
                        microstructure.rve_dims,
                        self.sample_dir,
                        sim_gif=True,
                        simulation_gif_dir=self.gif_dir,
                        iteration=self.gif_frame,
                        save=False,
                    )
                    # Saving a gif frame after each inner phase particle is placed
        if n_steps >= self.inner_phase_max_step:
            print_funcs.print_to_file(
                "Inner phase placement simulation reached the maximum number of steps.\n"
            )
        inner_phase.particles = [
            particle
            for particle in inner_phase.particles
            if not any(particle is not_placed for not_placed in not_placed_particles)
        ]
        # Removing the particles that were not placed before reaching the maximum number
        # of steps
        self.inner_phase_placements.append(
            {"phase": inner_phase.name, "steps": n_steps, "time": time.time() - start}
        )
        # Saving the number of steps and time of the inner phase placement