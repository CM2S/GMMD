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
from geommicgen.translators.base import SolverWriter, register_writer
from geommicgen.translators.reorder import (
    LINKS_DEFAULT_GAUSS_POINTS,
    links_element_name,
    reorder_connectivity,
)

NODE_CHUNK = 4096
# Number of lines assembled before they are written, so that a large mesh does not
# build the whole file in memory


@register_writer
class LinksWriter(SolverWriter):
    """
    Class for the writer of the LINKS mesh files.

    Attributes
    ----------
    gauss_points: dict
        Correspondence between the cell types and the number of Gauss points to be
        written. Only the entries that differ from the defaults are needed.

    require_periodic: bool
        Whether to refuse a mesh whose opposite faces are not discretised alike.

    write_example: bool
        Whether to write an example input file next to the mesh file.
    """

    name = "links"
    extension = ".mesh"

    def __init__(self, gauss_points=None, require_periodic=True, write_example=True):
        """Initizalizer for the LinksWriter Class."""
        self.gauss_points = dict(gauss_points) if gauss_points else {}
        self.require_periodic = require_periodic
        self.write_example = write_example

    def write(self, mesh, file_path):
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
        if self.require_periodic:
            mesh.check_periodic_conformity()
        # A mesh LINKS would refuse is refused here, where the face can be named

        groups, element_types, materials = self._build_groups(mesh)
        written = [file_path]

        with open(file_path, "w") as mesh_file:
            self._write_groups(mesh_file, groups, element_types, materials)
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
        phases = []
        for i_block_phase in mesh.phase:
            for i_phase in np.unique(i_block_phase):
                if int(i_phase) not in phases:
                    phases.append(int(i_phase))
        phases.sort()
        matrix_id = None
        for i_id, i_name in mesh.phase_names.items():
            if mesh.matrix_phase is not None and i_name == mesh.matrix_phase:
                matrix_id = int(i_id)
        if matrix_id in phases:
            phases.remove(matrix_id)
            phases.insert(0, matrix_id)
        materials = {i_phase: i_ind + 1 for i_ind, i_phase in enumerate(phases)}
        # The matrix takes the first material, as it did before

        element_types = {}
        groups = []
        for i_block, (i_type, i_connectivity) in enumerate(mesh.cells):
            if i_type not in element_types:
                element_types[i_type] = len(element_types) + 1
            for i_phase in sorted(np.unique(mesh.phase[i_block])):
                groups.append(
                    {
                        "id": len(groups) + 1,
                        "block": i_block,
                        "cell_type": i_type,
                        "phase": int(i_phase),
                        "element_type_id": element_types[i_type],
                        "material_id": materials[int(i_phase)],
                    }
                )

        return groups, element_types, materials

    def _write_groups(self, mesh_file, groups, element_types, materials):
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
        """Write the node coordinates, in chunks."""
        points = mesh.points
        mesh_file.write("NODE_COORDINATES {0} CARTESIAN\n".format(len(points)))
        lines = []
        for i_ind, i_point in enumerate(points):
            lines.append(
                "{0} {1:.12e} {2:.12e} {3:.12e}\n".format(
                    i_ind + 1, i_point[0], i_point[1], i_point[2]
                )
            )
            if len(lines) >= NODE_CHUNK:
                mesh_file.writelines(lines)
                lines = []
        mesh_file.writelines(lines)
        mesh_file.write("\n")
        # Three coordinates are always written; LINKS reads only as many as the analysis
        # has dimensions

    def _write_elements(self, mesh_file, mesh, groups):
        """Write the element connectivities, in chunks."""
        n_elements = sum(len(i_connectivity) for _, i_connectivity in mesh.cells)
        mesh_file.write("ELEMENTS {0}\n".format(n_elements))

        element_id = 0
        lines = []
        for i_group in groups:
            cell_type, connectivity = mesh.cells[i_group["block"]]
            selection = mesh.phase[i_group["block"]] == i_group["phase"]
            block = reorder_connectivity(cell_type, connectivity[selection])
            for i_cell in block:
                element_id += 1
                lines.append(
                    "{0} {1} {2}\n".format(
                        element_id,
                        i_group["id"],
                        " ".join(str(int(i_node) + 1) for i_node in i_cell),
                    )
                )
                if len(lines) >= NODE_CHUNK:
                    mesh_file.writelines(lines)
                    lines = []
        mesh_file.writelines(lines)
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
                "{0} ELASTIC\n 0.0\n 1.0E3 0.3   ! TODO phase {1}: real properties"
                "".format(i_material, names.get(i_phase, i_phase))
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
                "Boundary_Type Periodic_Condition\n\n"
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
                )
            )
        # The analysis type is 2 for a two dimensional microscale problem and 6 for a
        # three dimensional one
