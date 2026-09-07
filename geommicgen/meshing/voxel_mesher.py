"""
Module containing the structured mesher of microstructures.

The structured mesher lays a regular grid over the RVE and gives every voxel the phase
of the particle its centre falls inside, which is the discretisation the spectral
solvers read. A voxel of the grid is a quadrilateral in two dimensions and a hexahedron
in three, so the same mesh also serves the finite element writers, as long as the grid
is coarse enough for its cells to be built.

The grid is periodic by construction: a particle is stamped over the whole of its
bounding box, including the part of it that reaches outside the RVE, and the voxels
that fall outside are wrapped back in.
"""

import itertools

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.mesh import DEFAULT_MAX_CELLS, Mesh, StructuredInfo
from geommicgen.meshing.mesher import Mesher, register_mesher


@register_mesher
class VoxelMesher(Mesher):
    """
    Class for the mesher that lays a regular grid over the microstructure.

    Attributes
    ----------
    n_voxels_dims: array(int)
        Number of voxels in each spatial direction.

    max_cells: int
        Largest number of cells that the meshes it produces build from their grid.
    """

    name = "voxel"
    produces_structured = True

    def __init__(self, n_voxels_dims, max_cells=DEFAULT_MAX_CELLS):
        """
        Initizalizer for the VoxelMesher Class.

        Parameters
        ----------
        n_voxels_dims: array(int)
            Number of voxels in each spatial direction.

        max_cells: int
            Largest number of cells that the meshes it produces build from their grid.

        Raises
        ------
        ValueError:
            If any direction is given fewer than one voxel.
        """
        if any([i_n_voxels < 1 for i_n_voxels in n_voxels_dims]):
            raise ValueError(
                "The number of voxels in each direction has to be at least one."
            )
        self.n_voxels_dims = np.array([int(i_n) for i_n in n_voxels_dims])
        self.max_cells = max_cells

    def mesh(self, microstructure, report=None):
        """
        Lay a regular grid over a microstructure.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        report: callable
            Called with the index of the particle that was stamped and the total number
            of particles, so that a caller can report the progress of a long run.

        Returns
        -------
        `.Mesh`
            Structured mesh carrying the grid of phases.

        Raises
        ------
        ValueError:
            If the grid has a different number of directions than the microstructure,
            or if the microstructure has no matrix phase to fill the grid with.
        """
        rve_dims = np.asarray(microstructure.rve_dims, dtype=float)
        if len(self.n_voxels_dims) != len(rve_dims):
            raise ValueError(
                "The grid has {0} directions and the microstructure has {1}.".format(
                    len(self.n_voxels_dims), len(rve_dims)
                )
            )
        if microstructure.matrix_phase is None:
            raise ValueError(
                "The microstructure has no matrix phase to fill the grid with."
            )
        spacing = rve_dims / self.n_voxels_dims
        phase_grid = np.full(
            tuple(self.n_voxels_dims), int(microstructure.matrix_phase), dtype=int
        )
        # The grid starts out as matrix everywhere and the particles are stamped on it

        particles = microstructure.particles
        for i_particle_ind, i_particle in enumerate(particles):
            self.stamp_particle(i_particle, phase_grid, spacing, rve_dims)
            if report is not None:
                report(i_particle_ind, len(particles))

        return Mesh(
            rve_dims,
            phase_names={
                int(i_name): i_name for i_name in microstructure.phases.keys()
            },
            matrix_phase=microstructure.matrix_phase,
            periodic=True,
            structured=StructuredInfo(phase_grid, spacing),
            source={"mesher": self.name, "n_voxels": self.n_voxels_dims.tolist()},
            max_cells=self.max_cells,
        )

    @staticmethod
    def stamp_particle(particle, phase_grid, spacing, rve_dims):
        """
        Give every voxel whose centre lies inside a particle the phase of the particle.

        Parameters
        ----------
        particle: `.Particle`
            Particle to be stamped on the grid.

        phase_grid: array
            Array of integers with the phase of every voxel, modified in place.

        spacing: array
            Size of a voxel in each spatial direction.

        rve_dims: array
            Dimensions of the microstructure in each spatial direction.
        """
        axis_voxels = []
        for i_dir in range(len(spacing)):
            direction = np.zeros(3)
            direction[i_dir] = -1.0
            lower = particle.support_function(direction)[i_dir]
            direction[i_dir] = 1.0
            upper = particle.support_function(direction)[i_dir]
            indices = np.arange(
                int(lower // spacing[i_dir]), int(upper // spacing[i_dir]) + 2
            )
            axis_voxels.append(
                list(
                    zip(
                        np.mod(indices, phase_grid.shape[i_dir]).tolist(),
                        ((indices + 0.5) * spacing[i_dir]).tolist(),
                    )
                )
            )
        # Furthest point of the particle in each direction, and the voxels of the
        # bounding box it spans, each paired with the coordinate of its centre. The box
        # may reach outside the RVE, which is what makes the stamp periodic, so the
        # index is wrapped back in while the coordinate is not

        phase = int(particle.phase)
        for i_voxel in itertools.product(*axis_voxels):
            center = np.array([i_axis[1] for i_axis in i_voxel])
            if particle.point_inside(center, rve_dims):
                phase_grid[tuple(i_axis[0] for i_axis in i_voxel)] = phase
                # The centre of the voxel is inside the particle, so the voxel belongs
                # to its phase
