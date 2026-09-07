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

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import MeshTooLargeError, PeriodicityError
from geommicgen.meshing.periodic import classify_periodic_boundary

DEFAULT_MAX_CELLS = 20000000
# Largest number of cells that is materialized from a structured grid


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
        return int(np.prod(self.shape))


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
        """
        self.rve_dims = np.asarray(rve_dims, dtype=float)
        self.dim = len(self.rve_dims)
        self.periodic = bool(periodic)
        self.phase_names = dict(phase_names) if phase_names else {}
        self.matrix_phase = matrix_phase
        self.structured = structured
        self.boundary = None
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
    def n_cells(self):
        """Number of cells of the mesh."""
        if self.structured is not None and self._cells is None:
            return self.structured.n_cells

        return int(sum(len(i_connectivity) for _, i_connectivity in self.cells))

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
        self.boundary = classify_periodic_boundary(self.points, self.rve_dims, tol)

        return self.boundary

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
        boundary = self.boundary if self.boundary is not None else self.classify_boundary(tol)
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

        cell_data = {"phase": [np.asarray(i_phase) for i_phase in self.phase]}
        point_data = {}
        if self.boundary is not None:
            point_data["boundary_face"] = self.boundary.face_mask

        return meshio.Mesh(
            self.points,
            self.cells,
            point_data=point_data if point_data else None,
            cell_data=cell_data,
        )

    def _materialize(self, max_cells=DEFAULT_MAX_CELLS):
        """
        Build the nodes and the cells of a structured mesh.

        Parameters
        ----------
        max_cells: int
            Largest number of cells that is built.

        Raises
        ------
        MeshTooLargeError:
            If the grid has more cells than `max_cells`.
        """
        if self.structured is None:
            raise ValueError("An unstructured mesh has no grid to build the cells from.")
        if self.structured.n_cells > max_cells:
            raise MeshTooLargeError(
                self.structured.n_cells,
                max_cells,
                "the crate or vtk writers",
            )

        shape = self.structured.shape
        spacing = self.structured.spacing
        n_nodes_per_dir = [i_size + 1 for i_size in shape]

        grids = np.meshgrid(
            *[np.arange(i_size) * i_spacing for i_size, i_spacing in
              zip(n_nodes_per_dir, spacing)],
            indexing="ij",
        )
        points = np.zeros((int(np.prod(n_nodes_per_dir)), 3))
        for i_dir, i_grid in enumerate(grids):
            points[:, i_dir] = i_grid.ravel()
        self._points = points
        # The nodes are numbered in the same order the phases are stored, so that the
        # cell and the voxel with the same index describe the same region

        indices = np.arange(int(np.prod(n_nodes_per_dir))).reshape(n_nodes_per_dir)
        if len(shape) == 2:
            connectivity = np.stack(
                (
                    indices[:-1, :-1].ravel(),
                    indices[1:, :-1].ravel(),
                    indices[1:, 1:].ravel(),
                    indices[:-1, 1:].ravel(),
                ),
                axis=1,
            )
            cell_type = "quad"
        else:
            connectivity = np.stack(
                (
                    indices[:-1, :-1, :-1].ravel(),
                    indices[1:, :-1, :-1].ravel(),
                    indices[1:, 1:, :-1].ravel(),
                    indices[:-1, 1:, :-1].ravel(),
                    indices[:-1, :-1, 1:].ravel(),
                    indices[1:, :-1, 1:].ravel(),
                    indices[1:, 1:, 1:].ravel(),
                    indices[:-1, 1:, 1:].ravel(),
                ),
                axis=1,
            )
            cell_type = "hexahedron"
        # The nodes of every cell are listed in the ordering used by VTK, which is the
        # one meshio expects

        self._cells = [(cell_type, connectivity)]
        self._phase = [self.structured.phase_grid.ravel(order="C").astype(int)]
