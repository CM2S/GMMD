"""
Module containing the writer of the mesh files of the LINKS solver.

LINKS reads the four mesh keywords from a separate file when the input file names one
through MESH_FILE or MESH_FILE_RELATIVE. That is what this writer produces: a file with
the nodes, the elements and the groups, and nothing else. The constitutive behaviour,
the boundary conditions and the prescribed deformation belong to the analysis, not to
the geometry, and stay in the input file the user writes.

An example input file is written alongside it so that the mesh can be run without
having to look the format up, and so that the writer is exercised end to end.

The periodicity of the mesh is checked before anything is written. LINKS establishes the
periodic node pairs from the coordinates and stops when the discretisations of opposite
faces differ, so a mesh that would be refused there is refused here first, where the
message can name the face.
"""

import os

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import (
    PLACEHOLDER_ELASTIC,
    WRITE_CHUNK,
    SolverWriter,
    register_writer,
)
from geommicgen.translators.reorder import (
    LINKS_DEFAULT_GAUSS_POINTS,
    links_element_name,
    reorder_connectivity,
)

BOUNDARY_TYPES = {
    "Taylor_Condition": False,
    "Linear_Condition": False,
    "Periodic_Condition": True,
    "Uniform_Traction_Condition": False,
    "Uniform_Traction_Condition_II": False,
    "Mortar_Periodic_Condition": False,
    "Mortar_Periodic_Condition_II": False,
    "Kouznetsova_Periodic_Condition": True,
    "Kouznetsova_Mortar_Periodic_Condition": False,
    "Luscher_Direct_Condition": False,
    "Luscher_Direct_Condition_LM": False,
    "Luscher_Periodic_Condition": True,
    "Luscher_Periodic_Condition_LM": True,
    "Luscher_Mortar_Periodic_Condition": False,
    "Luscher_Minimal_Condition": False,
    "Blanco_Minimal_Condition": False,
    "Blanco_Trial_Condition": False,
    "MMVP_NoSym_Condition": False,
    "2nd_Taylor_Condition": False,
    "2nd_Direct_Condition": False,
    "2nd_Minimal_Condition": False,
    "2nd_MinimalSym_Condition": False,
}
# RVE constraints LINKS accepts, from the keyword it reads them with in
# ioctrl/indata_mod.f90, and whether each one pairs the nodes of opposite faces and so
# needs the two faces discretised alike. The four that do are exactly the ones LINKS
# runs its own periodicity verification for, in ioctrl/rve/getbcnnodes2d.f90 and the
# three files beside it. The mortar conditions exist in order to tie faces that do not
# match, and the remaining ones constrain the boundary without pairing anything. Naming
# a constraint LINKS does not know is refused here, where the message can list them,
# rather than by the solver once the analysis is launched

DEFAULT_BOUNDARY_TYPE = "Periodic_Condition"


def uniform_gauss_points(n_points):
    """
    Ask for the same number of Gauss points on every element that takes them.

    Parameters
    ----------
    n_points: int
        Number of Gauss points.

    Returns
    -------
    dict
        Correspondence between the cell types and the number of Gauss points.
    """
    return {
        i_type: n_points
        for i_type, i_default in LINKS_DEFAULT_GAUSS_POINTS.items()
        if i_default is not None
    }
    # The three node triangle is left out: it takes no Gauss point line at all, and
    # writing one would leave a file LINKS reads the following line wrongly from


@register_writer
class LinksWriter(SolverWriter):
    """
    Class for the writer of the LINKS mesh files.

    Attributes
    ----------
    gauss_points: dict
        Correspondence between the cell types and the number of Gauss points to be
        written. Only the entries that differ from the defaults are needed.

    boundary_type: str
        RVE constraint the example input file asks for. It also settles whether the
        mesh has to be periodic: only the constraints that pair the nodes of opposite
        faces need the two faces discretised alike.

    write_example: bool
        Whether to write an example input file next to the mesh file.
    """

    name = "links"
    extension = ".mesh"

    def __init__(
        self, gauss_points=None, boundary_type=DEFAULT_BOUNDARY_TYPE, write_example=True
    ):
        """Initizalizer for the LinksWriter Class."""
        if boundary_type not in BOUNDARY_TYPES:
            raise ValueError(
                "{0} is not an RVE constraint LINKS knows. The available options are "
                "{1}.".format(boundary_type, ", ".join(BOUNDARY_TYPES))
            )
        self.gauss_points = dict(gauss_points) if gauss_points else {}
        self.boundary_type = boundary_type
        self.requires_periodic = BOUNDARY_TYPES[boundary_type]
        self.write_example = write_example
        # Asking for a mortar constraint is how a mesh whose faces do not match is
        # written out on purpose, which is not the same as the writer giving up on
        # periodicity by itself when it finds one

    @classmethod
    def from_options(cls, options):
        """
        Build the writer from the options a deck or a command line gave.

        Parameters
        ----------
        options: dict
            Options given for the discretisation, keyed by the name of the keyword.

        Returns
        -------
        `.LinksWriter`
            The writer.
        """
        gauss_points = options.get("gauss_points")

        return cls(
            gauss_points=(
                uniform_gauss_points(gauss_points) if gauss_points else None
            ),
            boundary_type=options.get("boundary_type") or DEFAULT_BOUNDARY_TYPE,
        )

    def _write(self, mesh, file_path):
        """
        Write the mesh of a microstructure in the format LINKS reads.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        file_path: str
            Path of the mesh file to be written.

        Returns
        -------
        list
            Paths of the files that were written.
        """
        groups, element_types, materials = self._build_groups(mesh)
        written = [file_path]

        with open(file_path, "w") as mesh_file:
            self._write_groups(mesh_file, groups, element_types)
            self._write_nodes(mesh_file, mesh)
            self._write_elements(mesh_file, mesh, groups)

        if self.write_example:
            example_path = os.path.splitext(file_path)[0] + "_example.rve"
            self._write_example(example_path, file_path, mesh, materials)
            written.append(example_path)

        return written

    def _build_groups(self, mesh):
        """
        Build the element groups of the mesh.

        A group holds the elements of one phase that share one element type, which is
        what LINKS expects, since a group names a single element type and a single
        material.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh being written.

        Returns
        -------
        tuple
            List of groups, correspondence between cell types and their identifiers, and
            correspondence between the phases and their material identifiers.
        """
        blocks = mesh.phase_blocks()
        phases = sorted({i_phase for _, _, i_phase, _ in blocks})
        matrix_id = next(
            (
                int(i_id)
                for i_id, i_name in mesh.phase_names.items()
                if i_name == mesh.matrix_phase
            ),
            None,
        )
        if matrix_id in phases:
            phases.remove(matrix_id)
            phases.insert(0, matrix_id)
        materials = {i_phase: i_ind + 1 for i_ind, i_phase in enumerate(phases)}
        # The matrix takes the first material, as it did before

        element_types = {}
        groups = []
        for i_index, i_type, i_phase, i_rows in blocks:
            if i_type not in element_types:
                element_types[i_type] = len(element_types) + 1
            groups.append(
                {
                    "id": len(groups) + 1,
                    "block": i_index,
                    "rows": i_rows,
                    "element_type_id": element_types[i_type],
                    "material_id": materials[i_phase],
                }
            )
        # The mesh splits the cells by phase; a group is that split with the identifiers
        # LINKS needs put on it

        return groups, element_types, materials

    def _write_groups(self, mesh_file, groups, element_types):
        """Write the element group and element type blocks."""
        mesh_file.write("ELEMENT_GROUPS {0}\n".format(len(groups)))
        for i_group in groups:
            mesh_file.write(
                "{0} {1} {2}\n".format(
                    i_group["id"], i_group["element_type_id"], i_group["material_id"]
                )
            )
        mesh_file.write("\n")

        mesh_file.write("ELEMENT_TYPES {0}\n".format(len(element_types)))
        for i_type, i_id in sorted(element_types.items(), key=lambda item: item[1]):
            mesh_file.write("{0} {1}\n".format(i_id, links_element_name(i_type)))
            gauss_points = self.gauss_points.get(
                i_type, LINKS_DEFAULT_GAUSS_POINTS[i_type]
            )
            if gauss_points is not None:
                mesh_file.write("{0} GP\n".format(gauss_points))
        mesh_file.write("\n")
        # The three node triangle takes no Gauss point line, because the routine that
        # reads it does not look for one

    def _write_nodes(self, mesh_file, mesh):
        """Write the node coordinates, formatting a block of lines at a time."""
        points = mesh.points
        mesh_file.write("NODE_COORDINATES {0} CARTESIAN\n".format(len(points)))
        row_format = "%d %.12e %.12e %.12e\n"
        for i_start in range(0, len(points), WRITE_CHUNK):
            block = points[i_start:i_start + WRITE_CHUNK]
            rows = np.empty((len(block), 4), dtype=object)
            rows[:, 0] = np.arange(i_start + 1, i_start + len(block) + 1)
            rows[:, 1:] = block
            mesh_file.write((row_format * len(block)) % tuple(rows.ravel()))
        mesh_file.write("\n")
        # Three coordinates are always written; LINKS reads only as many as the analysis
        # has dimensions

    def _write_elements(self, mesh_file, mesh, groups):
        """Write the element connectivities, formatting a block of lines at a time."""
        mesh_file.write("ELEMENTS {0}\n".format(mesh.n_cells))

        element_id = 0
        for i_group in groups:
            cell_type, connectivity = mesh.cells[i_group["block"]]
            block = reorder_connectivity(cell_type, connectivity[i_group["rows"]])
            row_format = "%d %d" + " %d" * block.shape[1] + "\n"
            for i_start in range(0, len(block), WRITE_CHUNK):
                chunk = block[i_start:i_start + WRITE_CHUNK]
                rows = np.empty((len(chunk), block.shape[1] + 2), dtype=np.int64)
                rows[:, 0] = np.arange(element_id + 1, element_id + len(chunk) + 1)
                rows[:, 1] = i_group["id"]
                rows[:, 2:] = chunk + 1
                mesh_file.write(
                    (row_format * len(chunk)) % tuple(rows.ravel().tolist())
                )
                element_id += len(chunk)
        # The identifiers of the nodes and of the elements are dense and start at one,
        # which is what the reader of LINKS requires

    def _write_example(self, example_path, mesh_path, mesh, materials):
        """Write an example input file pointing at the mesh file."""
        deformation = np.eye(mesh.dim)
        deformation[0, 0] = 1.1
        rows = "\n".join(
            " ".join("{0:.3f}".format(i_value) for i_value in i_row)
            for i_row in deformation
        )
        names = {int(i_key): i_name for i_key, i_name in mesh.phase_names.items()}
        material_lines = []
        for i_phase, i_material in sorted(materials.items(), key=lambda item: item[1]):
            material_lines.append(
                "{0} ELASTIC\n 0.0\n {2} {3}   ! TODO phase {1}: real properties"
                "".format(
                    i_material, names.get(i_phase, i_phase), *PLACEHOLDER_ELASTIC
                )
            )

        with open(example_path, "w") as example_file:
            example_file.write(
                "! Example LINKS input file written by geommicgen.\n"
                "! The mesh is read from the file named below; edit the materials and\n"
                "! the prescribed deformation before using it for anything real.\n\n"
                "TITLE\n geommicgen microstructure\n\n"
                "ANALYSIS_TYPE {0}\n\n"
                "LARGE_STRAIN_FORMULATION ON\n\n"
                "MESH_FILE_RELATIVE {1}\n\n"
                "Boundary_Type {5}\n\n"
                "Prescribed_Deformation_Gradient\n{2}\n\n"
                "Number_of_Increments 1\n\n"
                "CONVERGENCE_TOLERANCE 1E-8\n\n"
                "SOLVER PARDISO\n\n"
                "VTK_OUTPUT ASCII\n\n"
                "MATERIALS {3}\n{4}\n".format(
                    2 if mesh.dim == 2 else 6,
                    os.path.basename(mesh_path),
                    rows,
                    len(materials),
                    "\n".join(material_lines),
                    self.boundary_type,
                )
            )
        # The analysis type is 2 for a two dimensional microscale problem and 6 for a
        # three dimensional one
