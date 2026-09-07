"""
Module containing the reading and writing of meshes in the standard formats.

The mesh is written as a VTK unstructured grid, with the phase of every cell stored as
cell data so that the file is self describing in a viewer. The information that the
format cannot hold, namely the dimensions of the RVE and the names of the phases, is
written next to it in a small sidecar file. The boundary classification is deliberately
not stored, because it is recomputed from the coordinates whenever it is needed, which
is what lets a mesh produced elsewhere be read back as a first class mesh.
"""

import json
import os

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.mesh import Mesh, StructuredInfo

SIDECAR_SUFFIX = ".mesh.json"
# Suffix of the file holding the information the mesh format cannot carry

SIDECAR_FORMAT = "geommicgen-mesh"


def sidecar_path(file_path):
    """
    Get the path of the sidecar file of a mesh.

    Parameters
    ----------
    file_path: str
        Path of the mesh file.

    Returns
    -------
    str
        Path of the sidecar file.
    """
    return os.path.splitext(file_path)[0] + SIDECAR_SUFFIX


def write_vtu(mesh, file_path, write_sidecar=True):
    """
    Write a mesh as a VTK unstructured grid.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh to be written.

    file_path: str
        Path of the file to be written.

    write_sidecar: bool
        Whether to write the sidecar file with the dimensions of the RVE and the names
        of the phases.
    """
    import meshio

    meshio.write(file_path, mesh.to_meshio())
    if write_sidecar:
        with open(sidecar_path(file_path), "w") as sidecar:
            json.dump(_sidecar_record(mesh), sidecar, indent=2)
            sidecar.write("\n")


def _sidecar_record(mesh):
    """
    Build the record written to the sidecar file of a mesh.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    Returns
    -------
    dict
        Dictionary with the information the mesh format cannot carry.
    """
    record = {
        "format": SIDECAR_FORMAT,
        "version": 1,
        "rve_dims": [float(i_dim) for i_dim in mesh.rve_dims],
        "dim": int(mesh.dim),
        "periodic": bool(mesh.periodic),
        "matrix_phase": mesh.matrix_phase,
        "phase_names": {str(i_key): i_name for i_key, i_name in mesh.phase_names.items()},
        "source": mesh.source,
    }
    if mesh.structured is not None:
        record["structured"] = {
            "shape": list(mesh.structured.shape),
            "spacing": [float(i_spacing) for i_spacing in mesh.structured.spacing],
        }

    return record


def read_mesh(file_path, rve_dims=None, matrix_phase=None, phase_key=None):
    """
    Read a mesh from any file that meshio can read.

    Parameters
    ----------
    file_path: str
        Path of the file to be read.

    rve_dims: array
        Dimensions of the RVE. When they are not supplied and there is no sidecar file,
        they are taken from the bounding box of the nodes.

    matrix_phase: str
        Name of the matrix phase, when it is not in the sidecar file.

    phase_key: str
        Name of the cell data array holding the phase of every cell. When it is not
        supplied, the usual names are tried in turn.

    Returns
    -------
    `.Mesh`
        The mesh described by the file, with its boundary already classified.
    """
    import meshio

    read = meshio.read(file_path)
    record = _read_sidecar(file_path)

    points = np.asarray(read.points, dtype=float)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    # Every mesh is carried in three dimensions, with the third coordinate left at zero

    if rve_dims is None and record is not None:
        rve_dims = record["rve_dims"]
    if rve_dims is None:
        dim = 3 if np.ptp(points[:, 2]) > 0 else 2
        rve_dims = np.ptp(points[:, :dim], axis=0)
    rve_dims = np.asarray(rve_dims, dtype=float)
    dim = len(rve_dims)
    # Falling back to the bounding box is the same assumption a solver makes when it
    # classifies the boundary nodes itself

    cells = []
    for i_block in read.cells:
        if _cell_dimension(i_block.type) == dim:
            cells.append((i_block.type, np.asarray(i_block.data)))
    if not cells:
        raise ValueError(
            "The file {0} holds no cells of dimension {1}.".format(file_path, dim)
        )
    # Cells of a lower dimension describe the boundary and are not part of the domain

    phase = _read_phase(read, cells, dim, phase_key)

    phase_names = {}
    if record is not None:
        phase_names = {int(i_key): i_name for i_key, i_name in record["phase_names"].items()}
        matrix_phase = matrix_phase if matrix_phase is not None else record["matrix_phase"]
    elif read.field_data:
        for i_name, i_value in read.field_data.items():
            if i_name.startswith("Phase "):
                phase_names[int(i_value[0])] = i_name[len("Phase "):]
    # The physical group names written by gmsh follow the "Phase <name>" convention

    mesh = Mesh(
        rve_dims,
        points=points,
        cells=cells,
        phase=phase,
        phase_names=phase_names,
        matrix_phase=matrix_phase,
        periodic=record["periodic"] if record is not None else False,
        source=record["source"] if record is not None else {"read_from": file_path},
    )
    mesh.classify_boundary()

    return mesh


def _read_sidecar(file_path):
    """Read the sidecar file of a mesh, returning None when there is none."""
    path = sidecar_path(file_path)
    if not os.path.exists(path):
        return None
    with open(path, "r") as sidecar:
        record = json.load(sidecar)
    if record.get("format") != SIDECAR_FORMAT:
        return None

    return record


def _read_phase(read, cells, dim, phase_key):
    """Get the phase of every cell of a mesh read from a file."""
    candidates = [phase_key] if phase_key else ["phase", "gmsh:physical", "medit:ref"]
    for i_key in candidates:
        if i_key is not None and i_key in read.cell_data:
            values = [
                np.asarray(i_values, dtype=int)
                for i_block, i_values in zip(read.cells, read.cell_data[i_key])
                if _cell_dimension(i_block.type) == dim
            ]
            if len(values) == len(cells):
                return values
    # Without any phase information every cell belongs to a single phase

    return [np.ones(len(i_connectivity), dtype=int) for _, i_connectivity in cells]


def _cell_dimension(cell_type):
    """Get the spatial dimension of a cell type named as in meshio."""
    if cell_type.startswith("vertex"):
        return 0
    if cell_type.startswith("line"):
        return 1
    if cell_type.startswith(("triangle", "quad")):
        return 2

    return 3


def write_vtk_image(mesh, file_path):
    """
    Write a structured mesh as a legacy VTK image, with the phase as cell data.

    meshio has no writer for image data, and the format is simple enough to write
    directly. The file is meant for visualisation; the spectral solvers read the phase
    grid itself, through their own writers.

    Parameters
    ----------
    mesh: `.Mesh`
        Structured mesh to be written.

    file_path: str
        Path of the file to be written.

    Raises
    ------
    ValueError:
        If the mesh is not structured.
    """
    if mesh.structured is None:
        raise ValueError("Only a structured mesh can be written as a VTK image.")

    shape = list(mesh.structured.shape)
    spacing = list(mesh.structured.spacing)
    while len(shape) < 3:
        shape.append(0)
        spacing.append(1.0)
    dimensions = [i_size + 1 for i_size in shape[: mesh.dim]] + [1] * (3 - mesh.dim)

    values = mesh.structured.phase_grid.ravel(order="F")
    with open(file_path, "w") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("geommicgen structured microstructure\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET STRUCTURED_POINTS\n")
        vtk_file.write("DIMENSIONS {0} {1} {2}\n".format(*dimensions))
        vtk_file.write("ORIGIN 0 0 0\n")
        vtk_file.write("SPACING {0} {1} {2}\n".format(*(list(spacing) + [1.0])[:3]))
        vtk_file.write("CELL_DATA {0}\n".format(len(values)))
        vtk_file.write("SCALARS phase int 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for i_value in values:
            vtk_file.write("{0}\n".format(int(i_value)))
    # The values run with the first direction changing fastest, as VTK expects
