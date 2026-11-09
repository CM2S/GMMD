"""
Module containing the writer of the voxel grids the spectral solvers read.

The grid of phases is written as a plain numpy array, which is what the CRATE solver
reads through its Discretization_File keyword. The array holds the phases alone, and the
dimensions of the RVE are declared in the input data file of the solver, so an example
of that file is written beside the grid, the way the LINKS writer writes one beside its
mesh: it names the grid, declares the dimensions of the RVE and a material for every
phase the grid holds, and runs as written. The materials, the loading and the number of
clusters are placeholders, marked as such.

This writer never builds the cells of the mesh, so it is usable with grids far too fine
to be expressed as elements.
"""

import os

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import (
    PLACEHOLDER_ELASTIC,
    PLACEHOLDER_STRAIN,
    SolverWriter,
    register_writer,
)

CRATE_PROBLEM_TYPES = {2: 1, 3: 4}
# Problem type CRATE is asked for, by the number of dimensions of the grid: plane strain
# in two dimensions, which is what the other writers assume as well, and three
# dimensional in three


@register_writer
class CrateWriter(SolverWriter):
    """
    Class for the writer of the voxel grids of the spectral solvers.

    Attributes
    ----------
    write_example: bool
        Whether to write an example input data file next to the grid.
    """

    name = "crate"
    extension = ".rgmsh.npy"
    needs_cells = False
    needs_grid = True

    def __init__(self, write_example=True):
        """Initizalizer for the CrateWriter Class."""
        self.write_example = write_example

    def _write(self, mesh, file_path):
        """
        Write the grid of phases of a structured mesh, and an input file to run it.

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
        grid_path = file_path if file_path.endswith(".npy") else file_path + ".npy"
        written = [grid_path]
        # numpy appends the extension itself, unless the name already carries it

        if self.write_example:
            base = grid_path[: -len(".npy")]
            if base.endswith(".rgmsh"):
                base = base[: -len(".rgmsh")]
            example_path = base + "_example.dat"
            self._write_example(example_path, grid_path, mesh)
            written.append(example_path)

        return written

    def _write_example(self, example_path, grid_path, mesh):
        """Write an example input data file pointing at the grid file."""
        phase_grid = mesh.structured.phase_grid
        dim = phase_grid.ndim
        phases = [int(i_phase) for i_phase in np.unique(phase_grid)]
        rve_dims = [float(i_dim) for i_dim in mesh.rve_dims[:dim]]

        materials = "".join(
            "{0} elastic 1\nelastic_symmetry isotropic 2\n  E {1}\n  v {2}\n".format(
                i_phase, *PLACEHOLDER_ELASTIC
            )
            for i_phase in phases
        )
        gradient = np.zeros((dim, dim))
        gradient[0, 0] = PLACEHOLDER_STRAIN
        strain = "".join(
            "eps_{0}{1} {2}\n".format(i_row + 1, j_col + 1, gradient[i_row, j_col])
            for j_col in range(dim)
            for i_row in range(dim)
        )
        clustering = "".join(
            "{0} static\n  base_clustering\n  1 1\n".format(i_phase)
            for i_phase in phases
        )
        clusters = "".join("{0} 1\n".format(i_phase) for i_phase in phases)
        # The components of the macroscopic strain are listed a column at a time,
        # which is the order CRATE reads them in whatever they are called

        with open(example_path, "w") as example_file:
            example_file.write(
                "# Example CRATE input data file written by geommicgen.\n"
                "# The grid is read from the file named under Discretization_File;\n"
                "# the materials, the loading and the number of clusters are\n"
                "# placeholders, and are to be replaced before the file is used for\n"
                "# anything real.\n\n"
                "Strain_Formulation 1\n\n"
                "Problem_Type {0}\n\n"
                "RVE_Dimensions\n{1}\n\n"
                "# TODO the materials: every phase has the same placeholder one\n"
                "Material_Phases {2}\n{3}\n"
                "# TODO the loading: a placeholder, a stretch of {4} along x\n"
                "Macroscale_Loading 1\nMacroscale_Strain\n{5}\n"
                "Number_of_Load_Increments 1\n\n"
                "Cluster_Analysis_Scheme\n{6}\n"
                "# TODO the clusters: one a phase, which is all a cell of one\n"
                "# material has, the k-means finding no more; raise them with the\n"
                "# materials\n"
                "Number_of_Clusters\n{7}\n"
                "Discretization_File\n{8}\n".format(
                    CRATE_PROBLEM_TYPES[dim],
                    " ".join(str(i_dim) for i_dim in rve_dims),
                    len(phases),
                    materials,
                    PLACEHOLDER_STRAIN,
                    strain,
                    clustering,
                    clusters,
                    os.path.basename(grid_path),
                )
            )
        # CRATE finds a keyword on any line that does not open with a hash, and reads
        # the lines of a block by their position after it, so the comments sit between
        # the blocks and never inside one
