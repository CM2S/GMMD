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
    needs_grid = True

    def _write(self, mesh, file_path):
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
        """
        np.save(file_path, mesh.structured.phase_grid)
        # numpy appends the extension itself, unless the name already carries it

        return [file_path if file_path.endswith(".npy") else file_path + ".npy"]
