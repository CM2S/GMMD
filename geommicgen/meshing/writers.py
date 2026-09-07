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
import xml.etree.ElementTree as ElementTree

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.mesh import Mesh, StructuredInfo

SIDECAR_SUFFIX = ".mesh.json"
# Suffix of the file holding the information the mesh format cannot carry

SIDECAR_FORMAT = "geommicgen-mesh"

IMAGE_SUFFIX = ".vti"
# Extension of the VTK image a structured mesh is written as

WRITE_CHUNK = 500000
# Number of values formatted at a time when writing a grid


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


def _sidecar_record(mesh, order="C"):
    """
    Build the record written to the sidecar file of a mesh.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    order: {"C", "F"}
        Order the cells of a structured mesh are written in. The cells built from a
        grid are laid out in C order, and a VTK image is laid out in Fortran order, so
        the grid can only be rebuilt if the file says which.

    Returns
    -------
    dict
        Dictionary with the information the mesh format cannot carry.
    """
    record = {
        "format": SIDECAR_FORMAT,
        "version": 1,
        "rve_dims": mesh.rve_dims.tolist(),
        "dim": int(mesh.dim),
        "periodic": bool(mesh.periodic),
        "matrix_phase": mesh.matrix_phase,
        "phase_names": {str(i_key): i_name for i_key, i_name in mesh.phase_names.items()},
        "source": mesh.source,
    }
    if mesh.structured is not None:
        record["structured"] = {
            "shape": list(mesh.structured.shape),
            "spacing": mesh.structured.spacing.tolist(),
            "order": order,
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
    record = _read_sidecar(file_path)
    if os.path.splitext(file_path)[1].lower() == IMAGE_SUFFIX:
        return _read_image_mesh(file_path, record, rve_dims, matrix_phase)
    # An image is read without ever building its cells, which is the whole reason a
    # grid is written as one

    import meshio

    read = meshio.read(file_path)

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
        if i_block.dim == dim:
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

    structured = None
    if record is not None and "structured" in record:
        shape = tuple(record["structured"]["shape"])
        structured = StructuredInfo(
            np.ravel(phase[0]).reshape(
                shape, order=record["structured"].get("order", "C")
            ),
            record["structured"]["spacing"],
        )
    # A mesh written from a grid is read back as a grid, so that the writers that need
    # the grid rather than the cells still accept it

    return Mesh(
        rve_dims,
        points=points,
        cells=cells,
        phase=phase,
        phase_names=phase_names,
        matrix_phase=matrix_phase,
        periodic=record["periodic"] if record is not None else False,
        structured=structured,
        source=record["source"] if record is not None else {"read_from": file_path},
    )
    # The boundary is classified on first use, so reading a mesh only to convert it
    # does not pay for a classification nobody asked for


def _read_image_mesh(file_path, record, rve_dims, matrix_phase):
    """
    Build the mesh a VTK image describes.

    Parameters
    ----------
    file_path: str
        Path of the image.

    record: dict
        Contents of the sidecar file, or None when there is none.

    rve_dims: array
        Dimensions of the RVE, when they are not to be taken from the file.

    matrix_phase: str
        Name of the matrix phase, when it is not in the sidecar file.

    Returns
    -------
    `.Mesh`
        The structured mesh the image describes.
    """
    phase_grid, spacing = read_vtk_image(file_path)
    if rve_dims is None and record is not None:
        rve_dims = record["rve_dims"]
    if rve_dims is None:
        rve_dims = np.asarray(phase_grid.shape, dtype=float) * spacing
    # An image says how large it is, so unlike a mesh of loose cells there is nothing
    # to guess at when it comes without a sidecar

    phase_names = {}
    if record is not None:
        phase_names = {
            int(i_key): i_name for i_key, i_name in record["phase_names"].items()
        }
        matrix_phase = (
            matrix_phase if matrix_phase is not None else record["matrix_phase"]
        )
    else:
        phase_names = {
            int(i_phase): str(i_phase) for i_phase in np.unique(phase_grid)
        }

    return Mesh(
        rve_dims,
        phase_names=phase_names,
        matrix_phase=matrix_phase,
        periodic=record["periodic"] if record is not None else True,
        structured=StructuredInfo(phase_grid, spacing),
        source=record["source"] if record is not None else {"read_from": file_path},
    )
    # A grid discretises opposite faces alike whatever produced it, so it is periodic
    # even when no sidecar says so


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
                if i_block.dim == dim
            ]
            if len(values) == len(cells):
                return values
    # Without any phase information every cell belongs to a single phase

    return [np.ones(len(i_connectivity), dtype=int) for _, i_connectivity in cells]


def write_vtk_image(mesh, file_path, write_sidecar=True):
    """
    Write a structured mesh as a VTK image, with the phase of every voxel as cell data.

    An image states the geometry as a rule, an origin and a spacing, rather than
    listing it, so the file holds one value per voxel however fine the grid is. meshio
    writes no image format at all, and the format is simple enough to write directly.

    Parameters
    ----------
    mesh: `.Mesh`
        Structured mesh to be written.

    file_path: str
        Path of the file to be written.

    write_sidecar: bool
        Whether to write the sidecar file with the dimensions of the RVE and the names
        of the phases. Without it the file still holds the grid, but nothing says how
        large the RVE is or what the phases are called.

    Raises
    ------
    ValueError:
        If the mesh is not structured.
    """
    if mesh.structured is None:
        raise ValueError("Only a structured mesh can be written as a VTK image.")

    shape = list(mesh.structured.shape) + [0] * (3 - mesh.dim)
    spacing = list(mesh.structured.spacing) + [1.0] * (3 - mesh.dim)
    extent = " ".join("0 {0}".format(i_size) for i_size in shape)
    values = mesh.structured.phase_grid.ravel(order="F")

    with open(file_path, "w") as image_file:
        image_file.write('<?xml version="1.0"?>\n')
        image_file.write(
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        )
        image_file.write(
            '  <ImageData WholeExtent="{0}" Origin="0 0 0" Spacing="{1}">\n'.format(
                extent, " ".join(repr(float(i_size)) for i_size in spacing)
            )
        )
        image_file.write('    <Piece Extent="{0}">\n'.format(extent))
        image_file.write('      <CellData Scalars="phase">\n')
        image_file.write(
            '        <DataArray type="Int32" Name="phase" format="ascii">\n'
        )
        for i_start in range(0, len(values), WRITE_CHUNK):
            chunk = values[i_start:i_start + WRITE_CHUNK]
            image_file.write(("%d\n" * len(chunk)) % tuple(chunk.tolist()))
        image_file.write("        </DataArray>\n")
        image_file.write("      </CellData>\n")
        image_file.write("    </Piece>\n")
        image_file.write("  </ImageData>\n")
        image_file.write("</VTKFile>\n")
    # The values run with the first direction changing fastest, which is the order VTK
    # reads an image in. They are formatted a block at a time, which is an order of
    # magnitude faster than one call per value and keeps the transient string bounded

    if write_sidecar:
        with open(sidecar_path(file_path), "w") as sidecar:
            json.dump(_sidecar_record(mesh, order="F"), sidecar, indent=2)
            sidecar.write("\n")


def read_vtk_image(file_path):
    """
    Read a VTK image, giving back the grid of phases and the size of a voxel.

    meshio reads no image format, so this reads the little of the format that is used
    here. It never builds the cells, which is the point of an image: a grid too fine to
    be expressed as cells is still read in the size of its phases.

    Parameters
    ----------
    file_path: str
        Path of the file to be read.

    Returns
    -------
    tuple
        The grid of phases and the spacing, both without the directions the image does
        not use.

    Raises
    ------
    ValueError:
        If the file is not an image, or holds no phases.
    """
    root = ElementTree.parse(file_path).getroot()
    image = root.find("ImageData")
    if image is None:
        raise ValueError("The file {0} is not a VTK image.".format(file_path))

    extent = [int(i_value) for i_value in image.get("WholeExtent").split()]
    shape = [extent[2 * i_dir + 1] - extent[2 * i_dir] for i_dir in range(3)]
    spacing = [float(i_value) for i_value in image.get("Spacing").split()]
    dim = len([i_size for i_size in shape if i_size > 0])

    array = image.find("./Piece/CellData/DataArray")
    if array is None:
        raise ValueError("The image {0} holds no cell data.".format(file_path))
    values = np.fromstring(array.text, dtype=int, sep=" ")

    return (
        values.reshape(shape[:dim], order="F"),
        np.asarray(spacing[:dim], dtype=float),
    )
    # An image runs with the first direction changing fastest, so the values fold back
    # into the grid in Fortran order
