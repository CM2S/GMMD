"""
This module contains the Microstructure class.
Each instance of the Microstructure class is a microstructure sample, composed of instances
 of the Phase class, in turn described by the adequate phase descriptors.
"""

import numpy as np

from .phase import Phase
from .particle_classes import Matrix

import iofuncs.printing as print_funcs


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

    matrix_phase: str
        Name of the matrix phase
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
            raise ValueError("Only 2D and 3D microstucutres are supproted.")
        if any([rve_dim <= 0 for rve_dim in self.rve_dims]):
            # The dimnesion of the microstrucutre must be positive
            raise ValueError(
                "The dimensions of the microstucutre must be positive values."
            )
        self.volume = np.prod(rve_dims)
        self.phases = {}

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
                for phase in self.phases:
                    if phase.type.__name__ not in ("CylindricalFiber", "Matrix"):
                        raise ValueError(
                            "The CylindricalFiber particles are only compatible with each other."
                        )

    @property
    def particles(self):
        """Particles in the microstucutre."""
        particles = []
        for i_phase in self.phases.values():
            particles += i_phase.particles

        return particles

    @property
    def volume_fraction(self):
        """Volume fraction of particles in the microstucutre."""
        vf = 0
        for i_phase in self.phases.values():
            vf += i_phase.volume_fraction

        return vf
