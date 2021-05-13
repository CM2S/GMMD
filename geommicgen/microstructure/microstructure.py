"""
This module contains the Microstructure class.

Each instance of the Microstructure class is a microstructure sample, composed of instances
of the Phase class, in turn described by the adequate phase descriptors.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from micgenmethod.speed_up_schemes import CellList
from microstructure.particleclasses import Point


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
        """Check if a point is inside the particle phase."""
        pts_particle = [Point(len(i_pt), "1") for i_pt in pts]
        for i_pt_ind, (i_pt, i_pt_center) in enumerate(zip(pts_particle, pts)):
            pts_particle[i_pt_ind].position_center = np.array(
                i_pt_center
            ) - self.rve_dims * np.floor(np.array(i_pt_center) / self.rve_dims)
        # Creating a point particle corresponding to the point at the same positions

        cell_list = CellList()
        cell_list.box = np.array(self.rve_dims)[: len(pts[0])]
        points_and_particles = pts_particle + self.particles
        cell_list.new_list(points_and_particles)
        # Computing the cell list

        inside = [0 for _ in pts]
        for i_pt_ind, i_pt in enumerate(pts_particle):
            for j_particle_ind in cell_list.particle_list[i_pt_ind]:
                if points_and_particles[j_particle_ind].point_inside(
                    i_pt.position_center, self.rve_dims
                ):
                    inside[i_pt_ind] = 1
                    break
        # Obtaining an array of 0s and 1s according to the positions of the points relative
        # to the particle phase

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
