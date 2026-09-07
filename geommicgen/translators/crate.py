"""
Module containing the writer of the voxel grids the spectral solvers read.

The grid of phases is written as a plain numpy array, which is what the CRATE solver
reads through its Discretization_File keyword. The array is the whole contract: the
dimensions of the RVE are declared in the input file of the solver, not here.

This writer never builds the cells of the mesh, so it is usable with grids far too fine
to be expressed as elements.
"""

import os

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import SolverWriter, register_writer


@register_writer
class CrateWriter(SolverWriter):
    """Class for the writer of the voxel grids of the spectral solvers."""

    name = "crate"
    extension = ".rgmsh.npy"
    needs_cells = False

    def write(self, mesh, file_path):
        """
        Write the grid of phases of a structured mesh.

        Parameters
        ----------
        mesh: `.Mesh`
            Structured mesh to be written.

        file_path: str
            Path of the file to be written. The *.npy* extension is appended by numpy
            when it is not already there.

        Returns
        -------
        list
            Paths of the files that were written.

        Raises
        ------
        ValueError:
            If the mesh is not structured.
        """
        if mesh.structured is None:
            raise ValueError(
                "Only a structured mesh has a grid of phases to write; mesh the "
                "microstructure with the voxel mesher to obtain one."
            )

        if file_path.endswith(".npy"):
            stem = file_path[: -len(".npy")]
        else:
            stem = file_path
        np.save(stem, mesh.structured.phase_grid)
        # numpy appends the extension itself, so the name is built without it

        return [stem + ".npy"]


def grid_file_name(deck_name, shape):
    """
    Build the name of the file holding a grid of phases.

    Parameters
    ----------
    deck_name: str
        Name of the input data file the microstructure was generated from, without its
        extension.

    shape: tuple
        Number of voxels in each spatial direction.

    Returns
    -------
    str
        Name of the file, without the extension numpy appends.
    """
    dimensions = "_".join(str(int(i_size)) for i_size in shape)
    if deck_name:
        return "{0}_{1}.rgmsh".format(os.path.splitext(deck_name)[0], dimensions)

    return "{0}.rgmsh".format(dimensions)
    # Naming the grid after the deck keeps the grids of different microstructures at the
    # same resolution apart, which the dimensions alone do not
