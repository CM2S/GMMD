"""
This module contains the Microstructure class.
Each instance of the Microstructure class is a microstructure sample, composed of instances
 of the Phase class, in turn described by the adequate phase descriptors.
"""

import numpy as np

from .phase import Phase
from .particle_classes import Matrix


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
            raise ValueError
        if any([rve_dim <= 0 for rve_dim in self.rve_dims]):
            # The dimnesion of the microstrucutre must be positive
            raise ValueError
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
                raise ValueError
            self.matrix_phase = phase.name
        else:
            if self.dim != self.phases[phase.name].type.dim:
                raise ValueError
