"""Module containing the helpers shared by the test modules."""

import numpy as np

from geommicgen.meshing.mesh import Mesh, StructuredInfo
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase


def structured_mesh(shape, rve_dims, phase_grid=None):
    """
    Build a structured mesh with the supplied number of voxels.

    Parameters
    ----------
    shape: tuple
        Number of voxels in each spatial direction.

    rve_dims: list
        Dimensions of the microstructure in each spatial direction.

    phase_grid: array
        Array of integers with the phase of every voxel. Defaults to a single phase.

    Returns
    -------
    `.Mesh`
        The structured mesh.
    """
    if phase_grid is None:
        phase_grid = np.ones(shape, dtype=int)
    spacing = np.asarray(rve_dims, dtype=float) / np.asarray(shape, dtype=float)

    return Mesh(
        rve_dims,
        structured=StructuredInfo(phase_grid, spacing),
        phase_names={1: "1", 2: "2"},
        matrix_phase="1",
        periodic=True,
    )


def build_microstructure(rve_dims, phase_type, particles):
    """
    Build a microstructure with a matrix phase and the supplied particles.

    Parameters
    ----------
    rve_dims: list
        Dimensions of the microstructure in each spatial direction.

    phase_type: class
        Class of the particles, used to declare the inclusion phase.

    particles: list
        Particles of the inclusion phase, already positioned.

    Returns
    -------
    `.Microstructure`
        Microstructure with the matrix phase named *1* and the inclusion phase named
        *2*.
    """
    microstructure = Microstructure(rve_dims)
    microstructure.add_phase(Phase("1", {"phase_type": 1}))
    microstructure.add_phase(Phase.from_type("2", phase_type))
    for i_particle in particles:
        microstructure.phases["2"].particles.append(i_particle)

    return microstructure
