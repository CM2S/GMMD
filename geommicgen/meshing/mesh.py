"""
Module containing the mesh classes.

A `.Mesh` is the single representation every mesher produces and every solver writer
consumes. It holds the nodes, the cells and the phase of each cell, and it knows the
dimensions of the RVE it discretises, so that the periodicity of its boundary can be
checked without any further information.

A mesh produced by a structured mesher stores the grid of phases rather than the cells,
because that grid is what the spectral solvers read and because materialising the cells
of a fine three dimensional grid is expensive. The cells are built on demand.
"""

import math

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import MeshTooLargeError, PeriodicityError
from geommicgen.meshing.periodic import classify_periodic_boundary

DEFAULT_MAX_CELLS = 20000000
# Largest number of cells that is materialized from a structured grid

CELL_ORDER = "C"
# Order the cells built from a grid run in. It is not a choice that is recorded
# anywhere: whoever reads cells that were built from a grid folds them back with this,
# and a VTK image is read with Fortran order because that is what the format is.

VTK_CELL_CORNERS = {
    2: ((0, 0), (1, 0), (1, 1), (0, 1)),
    3: (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ),
}
# Corners of a cell of the grid, in the order VTK lists the nodes of a quadrilateral
# and of a hexahedron


class StructuredInfo:
    """
    Class for the description of a structured grid of phases.

    Attributes
    ----------
    shape: tuple
        Number of voxels in each spatial direction.

    spacing: array
        Size of a voxel in each spatial direction.

    phase_grid: array
        Array of integers with the phase of every voxel.
    """

    def __init__(self, phase_grid, spacing):
        """
        Initizalizer for the StructuredInfo Class.

        Parameters
        ----------
        phase_grid: array
            Array of integers with the phase of every voxel.

        spacing: array
            Size of a voxel in each spatial direction.
        """
        self.phase_grid = np.asarray(phase_grid)
        self.spacing = np.asarray(spacing, dtype=float)
        self.shape = tuple(int(i_size) for i_size in self.phase_grid.shape)

    @property
    def n_cells(self):
        """Number of voxels of the grid."""
        return math.prod(self.shape)


class Mesh:
    """
    Class for a mesh of a microstructure.

    Attributes
    ----------
    rve_dims: array
        Array containing the dimensions of the microstructure in each spatial direction.

    dim: int
        Number of spatial dimensions of the mesh.

    periodic: bool
        Whether the mesher that produced the mesh intended it to be periodic. Whether it
        actually is has to be established with `classify_boundary`.

    phase_names: dict
        Dictionary whose keys are the phase identifiers used in the cell data and whose
        values are the names of the phases in the microstructure.

    matrix_phase: str
        Name of the matrix phase, when there is one.

    structured: `.StructuredInfo`
        Description of the grid, when the mesh comes from a structured mesher. None
        otherwise, in which case the nodes and the cells are stored directly.

    boundary: `.PeriodicBoundary`
        Classification of the boundary nodes, once `classify_boundary` has been called.

    source: dict
        Information about the mesher that produced the mesh.
    """

    def __init__(
        self,
        rve_dims,
        points=None,
        cells=None,
        phase=None,
        phase_names=None,
        matrix_phase=None,
        periodic=False,
        structured=None,
        source=None,
        max_cells=DEFAULT_MAX_CELLS,
    ):
        """
        Initizalizer for the Mesh Class.

        Parameters
        ----------
        rve_dims: array
            Dimensions of the microstructure in each spatial direction.

        points: array
            Array of shape *(n_nodes, 3)* with the coordinates of the nodes. Not needed
            for a structured mesh, which builds them on demand.

        cells: list
            List of tuples *(cell_type, connectivity)*, with the cell types named as in
            meshio and the connectivity in the ordering used by VTK.

        phase: list
            List with one array of integers per block of cells, giving the phase of
            every cell.

        phase_names: dict
            Correspondence between the phase identifiers and the names of the phases.

        matrix_phase: str
            Name of the matrix phase.

        periodic: bool
            Whether the mesh is intended to be periodic.

        structured: `.StructuredInfo`
            Description of the grid, for a structured mesh.

        source: dict
            Information about the mesher that produced the mesh.

        max_cells: int
            Largest number of cells that is built from a structured grid.
        """
        self.rve_dims = np.asarray(rve_dims, dtype=float)
        self.max_cells = max_cells
        self.dim = len(self.rve_dims)
        self.periodic = bool(periodic)
        self.phase_names = dict(phase_names) if phase_names else {}
        self.matrix_phase = matrix_phase
        self.structured = structured
        self._boundary = None
        self.source = dict(source) if source else {}
        self._points = None if points is None else np.asarray(points, dtype=float)
        self._cells = cells
        self._phase = phase
        if structured is None and points is None:
            raise ValueError(
                "A mesh must be given either its nodes or a structured grid."
            )

    @property
    def points(self):
        """Coordinates of the nodes, as an array of shape *(n_nodes, 3)*."""
        if self._points is None:
            self._materialize()

        return self._points

    @property
    def cells(self):
        """List of tuples *(cell_type, connectivity)* describing the cells."""
        if self._cells is None:
            self._materialize()

        return self._cells

    @property
    def phase(self):
        """List with one array of phase identifiers per block of cells."""
        if self._phase is None:
            self._materialize()

        return self._phase

    @property
    def boundary(self):
        """Classification of the boundary nodes, computed on first use."""
        if self._boundary is None:
            self.classify_boundary()

        return self._boundary

    @property
    def n_cells(self):
        """Number of cells of the mesh."""
        if self.structured is not None and self._cells is None:
            return self.structured.n_cells

        return sum(len(i_connectivity) for _, i_connectivity in self.cells)

    def phase_blocks(self):
        """
        Split the cells into the groups that share a block and a phase.

        Every solver that names the elements of a phase asks the mesh the same question,
        and a cell belongs to exactly one of the groups, so the answer is the mesh's to
        give rather than each writer's to work out again.

        Returns
        -------
        list
            Tuples *(block index, cell type, phase, rows)*, where the rows are the
            indices into the connectivity of that block.
        """
        blocks = []
        for i_block, (i_type, _) in enumerate(self.cells):
            order = np.argsort(self.phase[i_block], kind="stable")
            values, starts = np.unique(self.phase[i_block][order], return_index=True)
            bounds = list(starts) + [len(order)]
            for j_ind, j_phase in enumerate(values):
                blocks.append(
                    (
                        i_block,
                        i_type,
                        int(j_phase),
                        order[bounds[j_ind] : bounds[j_ind + 1]],
                    )
                )
        # Sorting once gives both the phases present and the rows of each of them, so
        # the phase array is not scanned again for every group. The rows are indices
        # rather than the cells themselves, so a mesh too large to copy is not copied

        return blocks

    def classify_boundary(self, tol=None):
        """
        Classify the boundary nodes of the mesh and pair them across opposite faces.

        Parameters
        ----------
        tol: float
            Absolute tolerance used to place the nodes on the faces of the RVE.

        Returns
        -------
        `.PeriodicBoundary`
            The classification of the boundary nodes.
        """
        self._boundary = classify_periodic_boundary(self.points, self.rve_dims, tol)

        return self._boundary

    def check_periodic_conformity(self, tol=None):
        """
        Check that the discretisations of opposite faces of the RVE match.

        Parameters
        ----------
        tol: float
            Absolute tolerance used to place the nodes on the faces of the RVE.

        Raises
        ------
        PeriodicityError:
            If a node on a face has no partner on the opposite face.
        """
        boundary = self.classify_boundary(tol) if tol is not None else self.boundary
        if not boundary.is_conforming:
            raise PeriodicityError(boundary.describe_mismatch())

    def to_meshio(self):
        """
        Convert the mesh into a meshio mesh.

        Returns
        -------
        meshio.Mesh
            The mesh, with the phase of every cell as cell data.
        """
        import meshio

        return meshio.Mesh(
            self.points,
            self.cells,
            cell_data={
                "phase": [np.asarray(i_phase, dtype=np.int32) for i_phase in self.phase]
            },
        )
        # The boundary classification is deliberately not written, since it is derived
        # from the coordinates and is recomputed wherever it is needed. The phase is
        # written as a 32 bit integer, which is what the readers of cell tags expect;
        # FEniCS in particular reads its mesh tags as such

    def _materialize(self, max_cells=None):
        """
        Build the nodes and the cells of a structured mesh.

        Parameters
        ----------
        max_cells: int
            Largest number of cells that is built. Defaults to the limit of the mesh.

        Raises
        ------
        MeshTooLargeError:
            If the grid has more cells than the limit.
        """
        if self.structured is None:
            raise ValueError("An unstructured mesh has no grid to build the cells from.")
        max_cells = self.max_cells if max_cells is None else max_cells
        if self.structured.n_cells > max_cells:
            raise MeshTooLargeError(self.structured.n_cells, max_cells)

        shape = self.structured.shape
        spacing = self.structured.spacing
        n_per_direction = [i_size + 1 for i_size in shape]
        n_nodes = math.prod(n_per_direction)

        points = np.zeros((n_nodes, 3))
        lattice = points.reshape(*(n_per_direction + [3]))
        for i_dir, (i_size, i_spacing) in enumerate(zip(n_per_direction, spacing)):
            broadcast = [1] * len(shape)
            broadcast[i_dir] = i_size
            lattice[..., i_dir] = (np.arange(i_size) * i_spacing).reshape(broadcast)
        # The coordinates are broadcast into a view of the node array, which avoids
        # building one full grid per direction only to ravel it away

        dtype = np.int32 if n_nodes < 2 ** 31 else np.int64
        strides = [
            math.prod(n_per_direction[i_dir + 1:]) for i_dir in range(len(shape))
        ]
        first = np.zeros(shape, dtype=dtype)
        for i_dir, i_stride in enumerate(strides):
            broadcast = [1] * len(shape)
            broadcast[i_dir] = shape[i_dir]
            first += (np.arange(shape[i_dir], dtype=dtype) * i_stride).reshape(broadcast)
        offsets = np.asarray(
            [
                sum(i_corner[i_dir] * strides[i_dir] for i_dir in range(len(shape)))
                for i_corner in VTK_CELL_CORNERS[len(shape)]
            ],
            dtype=dtype,
        )
        connectivity = first.ravel()[:, None] + offsets[None, :]
        # Every cell is its lowest numbered node plus the same offsets, so the whole
        # connectivity is one addition rather than a stack of sliced index grids

        self._points = points
        self._cells = [("quad" if len(shape) == 2 else "hexahedron", connectivity)]
        self._phase = [
            self.structured.phase_grid.ravel(order=CELL_ORDER).astype(int)
        ]
