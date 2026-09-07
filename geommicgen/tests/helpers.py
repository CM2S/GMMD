"""Module containing the helpers shared by the test modules."""

import numpy as np

from geommicgen.meshing.mesh import Mesh, StructuredInfo


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
