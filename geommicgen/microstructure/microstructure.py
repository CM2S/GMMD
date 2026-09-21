"""
This module contains the Microstructure class.

Each instance of the Microstructure class is a microstructure sample, composed of instances
of the Phase class, in turn described by the adequate phase descriptors.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.micgenmethod.speed_up_schemes import CellList
from geommicgen.microstructure.phase import Phase


class Microstructure:
    """
    Class for the Microstructure.

    Attributes
    ----------
    rve_dims: array
        Array containing the dimensions of the microstructure in each spatial direction.

    matrix_phase: str
        Name of the matrix phase.

    volume: float
        Volume/area of the microstructure.

    phases: dict
        Dictionary whose keys are the name of the phases and whose values are the
        corresponding instance of `.Phase`.

    total_overlap: float
        Measure of the overlap. Zero when there is no overlap.
    """

    def __init__(self, rve_dims):
        """Initizalizer for the Microstructure Class.

        Parameters
        ----------
        rve_dims: array
            Array containing the dimnesions of the microstructure in each spatial direction.
        """
        self.matrix_phase = None
        self.rve_dims = rve_dims
        self.dim = len(rve_dims)
        if self.dim != 2 and self.dim != 3:
            # Only 2D and 3D microstructures allowed
            raise ValueError("Only 2D and 3D microstructures are supproted.")
        if any([rve_dim <= 0 for rve_dim in self.rve_dims]):
            # The dimnesion of the microstrucutre must be positive
            raise ValueError(
                "The dimensions of the microstructure must be positive values."
            )
        self.volume = np.prod(rve_dims)
        self.phases = {}
        self.total_overlap = None

    @classmethod
    def from_descriptors(cls, rve_dims, descriptors):
        """
        Build the microstructure an input data file describes, with its phases.

        Parameters
        ----------
        rve_dims: array
            Dimensions of the microstructure in each spatial direction.

        descriptors: dict
            Descriptors of each phase, keyed by the name of the phase, as the input data
            file gives them.

        Returns
        -------
        `.Microstructure`
            The microstructure, with no particles yet.

        Raises
        ------
        ValueError:
            If no phase is a matrix, or if two are.
        """
        microstructure = cls(rve_dims)
        for i_name, i_descriptors in descriptors.items():
            microstructure.add_phase(Phase(i_name, i_descriptors))
        if microstructure.matrix_phase is None:
            raise ValueError("No matrix phase was specified.")
        # Refused here, where the phases are declared, rather than by whichever
        # mesher first fills the RVE and finds nothing to fill it with

        return microstructure

    def add_phase(self, phase):
        """
        Add a phase to the microstructure.

        Parameters
        ----------
        phase: `.Phase`
        """
        self.phases[phase.name] = phase
        phase.microstructure = self
        if self.phases[phase.name].type.__name__ == "Matrix":
            if self.matrix_phase is not None:
                raise ValueError(
                    "The matrix for Phase {0} was specified twice.".format(phase.name)
                )
            self.matrix_phase = phase.name
        else:
            if self.dim != self.phases[phase.name].type.dim:
                raise ValueError(
                    "The particle type chosen is not compatible "
                    + "with the dimensions of the microstructure."
                )
            if self.phases[phase.name].type.__name__ == "CylindricalFiber":
                for i_phase in self.phases.values():
                    if i_phase.type.__name__ not in ("CylindricalFiber", "Matrix"):
                        raise ValueError(
                            "The CylindricalFiber particles are only compatible with"
                            + " each other."
                        )

    def inside_particle_phase(self, pts):
        """
        Say, for each of a set of points, whether it lies inside a particle.

        Parameters
        ----------
        pts: list(array)
            Positions of the points, which may lie outside the RVE and are wrapped
            into it.

        Returns
        -------
        list(int)
            1 for a point inside a particle, 0 otherwise, in the order given.
        """
        dim = len(pts[0])
        box = np.asarray(self.rve_dims, dtype=float)[:dim]
        particles = self.particles
        cell_list = CellList()
        cell_list.box = box
        cell_list.new_list(particles)
        # The cells are sized by the largest particle, so a point inside one lies in the
        # cell of its centre or in a neighbour of it: the guarantee the simulation relies
        # on for two particles overlapping, with a point in place of the second. The
        # points themselves are only looked up in the list; putting them in it, as used
        # to be done, tested every point against every other point in its cell

        candidates = {}
        for i_cell in range(len(cell_list.cell_list)):
            candidates[i_cell] = sorted(
                set().union(
                    *(
                        cell_list.cell_list[
                            cell_list.neighbor_cell(
                                i_cell, j_neighbor, dim, cell_list.n_cell_dim
                            )
                        ]
                        for j_neighbor in range(3**dim)
                    )
                )
            )
        # The particles a point in each cell has to be tested against, gathered once
        # per cell rather than once per point: there are a few cells and many points

        positions = np.asarray(pts, dtype=float)
        positions = positions - box * np.floor(positions / box)
        inside = [0 for _ in pts]
        for i_ind, i_position in enumerate(positions):
            if any(
                particles[j_particle].point_inside(i_position, self.rve_dims)
                for j_particle in candidates[cell_list.cell_of(i_position)]
            ):
                inside[i_ind] = 1

        return inside

    @property
    def particles(self):
        """Particles in the microstructure."""
        particles = []
        for i_phase in self.phases.values():
            particles += i_phase.particles

        return particles

    @property
    def volume_fraction(self):
        """Volume fraction of particles in the microstructure."""
        vf = 0
        for i_phase in self.phases.values():
            vf += i_phase.volume_fraction

        return vf

    @property
    def volume_fraction_circ(self):
        """Volume fraction of circumscribed sphesres/disks to the particles."""
        vf = 0
        for i_phase in self.phases.values():
            vf += i_phase.volume_fraction_circ

        return vf
