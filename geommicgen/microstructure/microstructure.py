"""
This module contains the Microstructure class.
Each instance of the Microstructure class is a microstructure sample, composed of instances
 of the Phase class, in turn described by the adequate phase descriptors.
"""

import numpy as np

from .phase import Phase


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
    """

    def __init__(self, mic_gen_descriptors, rve_dims):
        """Initizalizer for the Microstructure Class.

        Parameters
        ----------
        mic_gen_descriptors: dict
            Dictinary whose keys are the phase names and whose values are dictionaries
            specifiyng the phase descriptors.

        rve_dims: array
            Array containing the dimnesions of the microstructure in each spatial direction.

        """

        self.matrix_phase = None
        self.rve_dims = rve_dims
        self.volume = np.prod(rve_dims)
        self.phases = {}
        for phase_name, phase_descriptors in mic_gen_descriptors.items():
            self.phases[phase_name] = Phase(phase_name, phase_descriptors)
            if self.phases[phase_name].type == "Matrix":
                if self.matrix_phase is not None:
                    raise ValueError
                self.matrix_phase = phase_name
