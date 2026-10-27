"""
Module containing the writer of the input files of the Abaqus solver.

Abaqus reads a mesh from the same file as everything else, so this writes the nodes, the
elements grouped by phase, the node sets of the boundary, and the multi point
constraints that make the cell periodic. The constitutive behaviour and the loading are
placeholders, marked as such: a microstructure says nothing about either. They are
written all the same, so that the deck runs as it is: one linear static step, a
macroscopic stretch along the first axis prescribed through the reference nodes, and the
reaction forces on them printed, which divided by the volume of the cell are the
homogenised stress.

meshio writes an Abaqus file of its own, and it is not this one. It names two
dimensional triangles after a rigid surface element and second order tetrahedra after a
modified hybrid one, it cannot put an element set on the elements it writes, and it has
no way to express a constraint -- which is most of what an RVE deck is. That is why the
format is written here rather than delegated with the rest.

The periodicity is imposed as one equation per degree of freedom per pair of nodes, in
the form used for a deformation driven RVE:

    u(slave) - u(master) - sum_k c_k u(RP_k) = 0

where *c* counts how many cell dimensions separate the two nodes along each axis, and
the displacement of the reference node *RP_k* stands for the k-th column of the
displacement gradient times the length of the cell. Prescribing the three reference
nodes therefore prescribes the macroscopic deformation gradient. The same expression
covers the faces, the edges and the corners: what tells them apart is only how many
entries of *c* are not zero, and that is read off the coordinates rather than from which
set the pair came from.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.periodic import AXIS_NAMES
from geommicgen.translators.base import (
    PLACEHOLDER_ELASTIC,
    PLACEHOLDER_STRAIN,
    WRITE_CHUNK,
    SolverWriter,
    register_writer,
)

SET_LINE_ITEMS = 16
# Node identifiers written per line of a set. Abaqus reads at most sixteen entries from
# an input line

EQUATION_LINE_TERMS = 4
# Terms written per line of an equation. Each one is three entries, so four of them stay
# within the sixteen an input line holds

ABAQUS_ELEMENT_NAMES = {
    "triangle": "CPE3",
    "triangle6": "CPE6",
    "quad": "CPE4",
    "quad8": "CPE8",
    "tetra": "C3D4",
    "tetra10": "C3D10",
    "hexahedron": "C3D8",
    "hexahedron20": "C3D20",
}
# Abaqus names of the elements. The two dimensional ones are the plane strain family,
# which is what a two dimensional microstructure is analysed as, the same assumption the
# LINKS example file makes when it asks for a two dimensional microscale analysis

REFERENCE_NODE_NAMES = ("RP_X", "RP_Y", "RP_Z")
# Sets of the reference nodes whose displacements carry the macroscopic deformation


def abaqus_element_name(cell_type):
    """
    Get the Abaqus name of a cell type.

    Parameters
    ----------
    cell_type: str
        Type of the cells, as meshio names it.

    Returns
    -------
    str
        Name Abaqus knows the element by.

    Raises
    ------
    ValueError:
        If there is no Abaqus element for the cell type.
    """
    if cell_type not in ABAQUS_ELEMENT_NAMES:
        raise ValueError(
            "There is no Abaqus element for cells of type {0}. The ones written are "
            "{1}.".format(cell_type, ", ".join(sorted(ABAQUS_ELEMENT_NAMES)))
        )

    return ABAQUS_ELEMENT_NAMES[cell_type]


@register_writer
class AbaqusWriter(SolverWriter):
    """
    Class for the writer of the Abaqus input files.

    Attributes
    ----------
    periodic_constraints: bool
        Whether to write the reference nodes and the equations that make the cell
        periodic. A mesh whose opposite faces are not discretised alike has no pairs to
        write them from, so asking for them requires a conforming mesh.
    """

    name = "abaqus"
    extension = ".inp"
    options = {
        "Periodic_Constraints": {
            "type": "bool",
            "help": "whether to write the constraints that make the cell periodic; "
            "without them the deck carries the mesh and its sets alone",
        }
    }

    def __init__(self, periodic_constraints=True):
        """Initizalizer for the AbaqusWriter Class."""
        self.periodic_constraints = bool(periodic_constraints)
        self.requires_periodic = self.periodic_constraints

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
        `.AbaqusWriter`
            The writer.
        """
        constraints = options.get("periodic_constraints")

        return cls(periodic_constraints=constraints is not False)
        # Absent means wanted: a mesh of an RVE is periodic unless someone says the
        # deck is not to constrain it

    def _write(self, mesh, file_path):
        """
        Write the mesh of a microstructure in the format Abaqus reads.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        file_path: str
            Path of the file to be written.

        Returns
        -------
        list
            Paths of the files that were written.
        """
        blocks = mesh.phase_blocks()
        boundary = mesh.boundary

        if self.periodic_constraints:
            placeholders = (
                "** The mesh is complete; the materials and the step are placeholders,\n"
                "** and are to be replaced before the file is used for anything real.\n"
            )
        else:
            placeholders = (
                "** The mesh is complete; the materials are placeholders, and\n"
                "** there is no step, the constraints that carry the macroscopic\n"
                "** deformation not having been asked for.\n"
            )

        with open(file_path, "w") as deck:
            deck.write(
                "** Abaqus input file written by geommicgen.\n{0}"
                "*Heading\n geommicgen microstructure\n".format(placeholders)
            )
            self._write_nodes(deck, mesh)
            self._write_elements(deck, mesh, blocks)
            self._write_node_sets(deck, boundary)
            self._write_sections(deck, mesh, blocks)
            if self.periodic_constraints:
                self._write_reference_sets(deck, mesh)
                self._write_equations(deck, mesh, boundary)
                self._write_step(deck, mesh)
            # The sets come before the constraints that refer to them, and the model
            # before the step that loads it, which is the order Abaqus needs them in.
            # Without the constraints there is nothing to prescribe the macroscopic
            # deformation through, and so no step to write

        return [file_path]

    def _write_nodes(self, deck, mesh):
        """Write the node coordinates, formatting a block of lines at a time."""
        points = mesh.points[:, : mesh.dim]
        deck.write("*Node\n")
        row_format = "%d" + ", %.12e" * mesh.dim + "\n"
        for i_start in range(0, len(points), WRITE_CHUNK):
            block = points[i_start : i_start + WRITE_CHUNK]
            rows = np.empty((len(block), mesh.dim + 1), dtype=object)
            rows[:, 0] = np.arange(i_start + 1, i_start + len(block) + 1)
            rows[:, 1:] = block
            deck.write((row_format * len(block)) % tuple(rows.ravel()))
        if self.periodic_constraints:
            for i_dir in range(mesh.dim):
                deck.write(
                    "{0}{1}\n".format(
                        _reference_node_id(mesh, i_dir), ", 0.0" * mesh.dim
                    )
                )
        # As many coordinates as the analysis has dimensions: a plane strain model with
        # a third coordinate on every node is not one Abaqus reads. The reference nodes
        # follow the mesh in the same block rather than in blocks of their own, which
        # Abaqus would accumulate but readers of the format do not all expect

    def _write_elements(self, deck, mesh, blocks):
        """Write the connectivities, one element set per phase and element type."""
        element_id = 0
        for i_index, i_type, i_phase, i_rows in blocks:
            rows_of_block = mesh.cells[i_index][1][i_rows]
            deck.write(
                "*Element, type={0}, elset={1}\n".format(
                    abaqus_element_name(i_type), _phase_set_name(mesh, i_phase)
                )
            )
            row_format = "%d" + ", %d" * rows_of_block.shape[1] + "\n"
            for i_start in range(0, len(rows_of_block), WRITE_CHUNK):
                chunk = rows_of_block[i_start : i_start + WRITE_CHUNK]
                rows = np.empty((len(chunk), chunk.shape[1] + 1), dtype=np.int64)
                rows[:, 0] = np.arange(element_id + 1, element_id + len(chunk) + 1)
                rows[:, 1:] = chunk + 1
                deck.write((row_format * len(chunk)) % tuple(rows.ravel().tolist()))
                element_id += len(chunk)
        # The nodes and the elements are numbered from one, densely and in the order the
        # mesh holds them, so that a node identifier is its index in the mesh plus one

    def _write_node_sets(self, deck, boundary):
        """Write a node set for every face, edge and corner of the cell."""
        for i_kind, i_group in (
            ("FACE", boundary.face_nodes),
            ("EDGE", boundary.edge_nodes),
            ("CORNER", boundary.corner_nodes),
        ):
            for i_name in sorted(i_group):
                _write_set(deck, _set_name(i_kind, i_name), i_group[i_name])
        # What kind of set a name belongs to is which group it came from, not how many
        # faces it names: a node on two faces is an edge in three dimensions and a
        # corner in two
        # Named after the faces they lie on, with the sign spelled out, so that
        # prescribing a face by hand does not require reading the mesh first

    def _write_sections(self, deck, mesh, blocks):
        """Write a section and a placeholder material for every phase."""
        for i_phase in sorted({i_phase for _, _, i_phase, _ in blocks}):
            label = _phase_label(mesh, i_phase)
            deck.write(
                "*Solid Section, elset=PHASE_{0}, material=MATERIAL_{0}\n,\n"
                "*Material, name=MATERIAL_{0}\n*Elastic\n {1}, {2}\n"
                "** TODO PHASE_{0}: the real properties\n".format(
                    label, *PLACEHOLDER_ELASTIC
                )
            )

    def _write_reference_sets(self, deck, mesh):
        """Name the nodes whose displacements carry the macroscopic deformation."""
        for i_dir in range(mesh.dim):
            deck.write(
                "*Nset, nset={0}\n{1}\n".format(
                    REFERENCE_NODE_NAMES[i_dir], _reference_node_id(mesh, i_dir)
                )
            )
        # They sit at the origin and are attached to no element, so they carry
        # displacement and nothing else. Prescribing the three of them prescribes the
        # macroscopic deformation gradient

    def _write_equations(self, deck, mesh, boundary):
        """Write one equation per degree of freedom per pair of periodic nodes."""
        pairs = _periodic_pairs(boundary)
        if len(pairs) == 0:
            return

        coefficients = _cell_offsets(mesh, pairs)
        patterns = (coefficients != 0) @ (1 << np.arange(mesh.dim))
        starts = np.flatnonzero(patterns[1:] != patterns[:-1]) + 1
        bounds = [0] + starts.tolist() + [len(patterns)]
        for i_start, i_end in zip(bounds[:-1], bounds[1:]):
            axes = [
                i_dir for i_dir in range(mesh.dim) if (patterns[i_start] >> i_dir) & 1
            ]
            self._write_equation_group(
                deck,
                mesh,
                pairs[i_start:i_end],
                coefficients[i_start:i_end][:, axes],
                axes,
            )
        # Which reference nodes take part is settled by the pattern of the separation,
        # so equations sharing one differ only in two node identifiers and in the
        # coefficients. The pairs arrive grouped -- a face at a time, then the slave
        # edges of an axis, then the corners -- so consecutive runs of one pattern are
        # what the writing is cut into, and the order is the order of the pairs

    def _write_step(self, deck, mesh):
        """Write a placeholder step, loading the cell through the reference nodes."""
        corner = _set_name(
            "CORNER", "".join(i_axis + "-" for i_axis in AXIS_NAMES[: mesh.dim])
        )
        gradient = np.zeros((mesh.dim, mesh.dim))
        gradient[0, 0] = PLACEHOLDER_STRAIN
        rve_dims = np.asarray(mesh.rve_dims, dtype=float)[: mesh.dim]
        # The displacement of the reference node of an axis is the column of the
        # displacement gradient of that axis times the length of the cell along it

        deck.write(
            "** TODO the step: a placeholder, a stretch of {0} along x\n"
            "*Step, name=PLACEHOLDER, nlgeom=NO\n*Static\n*Boundary\n"
            "{1}, 1, {2}, 0.0\n".format(PLACEHOLDER_STRAIN, corner, mesh.dim)
        )
        for i_axis in range(mesh.dim):
            for j_dof in range(mesh.dim):
                deck.write(
                    "{0}, {1}, {1}, {2:.12e}\n".format(
                        REFERENCE_NODE_NAMES[i_axis],
                        j_dof + 1,
                        gradient[j_dof, i_axis] * rve_dims[i_axis],
                    )
                )
        # Every degree of freedom of every reference node is prescribed, which fixes the
        # whole macroscopic deformation and with it the rotation of the cell. What the
        # constraints leave free is a translation of every node at once, and holding the
        # corner every other corner is paired to removes it: that corner is never the
        # node an equation eliminates, so prescribing it does not constrain it twice

        deck.write("*Output, field, variable=PRESELECT\n")
        deck.write("*Output, history, variable=PRESELECT\n")
        for i_name in REFERENCE_NODE_NAMES[: mesh.dim]:
            deck.write("*Node Print, nset={0}\nRF\n".format(i_name))
        deck.write("*End Step\n")
        # The reaction forces on the reference nodes are printed to the text output,
        # where a check that the deck ran can read the homogenised stress without the
        # output database

    def _write_equation_group(self, deck, mesh, pairs, coefficients, axes):
        """
        Write the equations of the pairs whose separation has the same pattern.

        Parameters
        ----------
        deck: file
            File being written.

        mesh: `.Mesh`
            Mesh being written.

        pairs: array
            Array of shape *(n, 2)* holding the slave and the master of each pair.

        coefficients: array
            Array of shape *(n, len(axes))* with how many cell dimensions separate the
            two nodes along each of the axes that take part.

        axes: list
            Spatial directions whose reference node enters the equations.
        """
        n_terms = 2 + len(axes)
        row_format = "*Equation\n{0}\n".format(n_terms)
        for i_start in range(0, n_terms, EQUATION_LINE_TERMS):
            n_line = min(EQUATION_LINE_TERMS, n_terms - i_start)
            row_format += ", ".join(["%d, %d, %.1f"] * n_line) + "\n"
        # Every equation of the group has the same number of terms, so it has the same
        # layout, and the layout is built once for all of them

        values = np.empty((len(pairs) * mesh.dim, 3 * n_terms))
        values[:, 0] = np.repeat(pairs[:, 0] + 1, mesh.dim)
        values[:, 2] = 1.0
        values[:, 3] = np.repeat(pairs[:, 1] + 1, mesh.dim)
        values[:, 5] = -1.0
        for i_ind, i_axis in enumerate(axes):
            values[:, 6 + 3 * i_ind] = _reference_node_id(mesh, i_axis)
            values[:, 8 + 3 * i_ind] = np.repeat(-coefficients[:, i_ind], mesh.dim)
        values[:, 1::3] = np.tile(np.arange(1, mesh.dim + 1), len(pairs))[:, None]
        # The degrees of freedom of one pair are consecutive, so a pair contributes as
        # many rows as the mesh has dimensions and they stay together

        for i_start in range(0, len(values), WRITE_CHUNK):
            chunk = values[i_start : i_start + WRITE_CHUNK]
            deck.write((row_format * len(chunk)) % tuple(chunk.ravel().tolist()))
        # The slave comes first because Abaqus eliminates the degree of freedom of the
        # first term, and the slave is the one that is not to be prescribed elsewhere


def _phase_label(mesh, phase):
    """
    Get the name a phase is known by.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    phase: int
        Identifier of the phase.

    Returns
    -------
    str
        Name of the phase, falling back to its identifier when it has none.
    """
    return mesh.phase_names.get(phase, phase)


def _phase_set_name(mesh, phase):
    """
    Get the name of the element set holding the cells of a phase.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    phase: int
        Identifier of the phase.

    Returns
    -------
    str
        Name of the element set.
    """
    return "PHASE_{0}".format(_phase_label(mesh, phase))


def _set_name(kind, name):
    """
    Turn the name of a face, an edge or a corner into the name of a set.

    Parameters
    ----------
    kind: str
        What the set holds, one of *"FACE"*, *"EDGE"* and *"CORNER"*.

    name: str
        Name of the face, edge or corner, such as *"x-"* or *"x-y+"*.

    Returns
    -------
    str
        Name of the set, such as *"FACE_XNEG"* or *"CORNER_XNEG_YPOS"*.
    """
    parts = [
        "{0}{1}".format(
            name[i_char].upper(), "POS" if name[i_char + 1] == "+" else "NEG"
        )
        for i_char in range(0, len(name), 2)
    ]

    return "_".join([kind] + parts)
    # Abaqus set names carry no punctuation, so the sign becomes a word


def _write_set(deck, name, nodes):
    """
    Write one node set.

    Parameters
    ----------
    deck: file
        File being written.

    name: str
        Name of the set.

    nodes: array
        Indices of the nodes in the mesh.
    """
    if len(nodes) == 0:
        return

    deck.write("*Nset, nset={0}\n".format(name))
    identifiers = np.asarray(nodes, dtype=np.int64) + 1
    for i_start in range(0, len(identifiers), SET_LINE_ITEMS):
        line = identifiers[i_start : i_start + SET_LINE_ITEMS]
        deck.write(", ".join(str(int(i_id)) for i_id in line) + "\n")


def _reference_node_id(mesh, direction):
    """
    Get the identifier of the reference node of a direction.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    direction: int
        Index of the spatial direction.

    Returns
    -------
    int
        Identifier of the node, which follows the nodes of the mesh.
    """
    return len(mesh.points) + direction + 1


def _periodic_pairs(boundary):
    """
    Collect every pair of nodes the periodicity relates.

    Parameters
    ----------
    boundary: `.PeriodicBoundary`
        Classification of the boundary nodes.

    Returns
    -------
    array
        Array of shape *(n, 2)* holding the slave and the master of each pair.
    """
    pairs = [boundary.face_pairs[i_axis] for i_axis in sorted(boundary.face_pairs)]
    pairs += [boundary.edge_pairs[i_axis] for i_axis in sorted(boundary.edge_pairs)]
    pairs.append(boundary.corner_pairs)

    return np.concatenate(pairs)
    # The faces are paired from the nodes that lie on one face alone, so no node is
    # constrained twice by collecting the three kinds together. They are one array
    # because nothing downstream asks which of the three a pair came from: what a pair
    # needs is read off its coordinates


def _cell_offsets(mesh, pairs):
    """
    Count the cell dimensions separating the two nodes of every pair.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    pairs: array
        Array of shape *(n, 2)* holding the slave and the master of each pair.

    Returns
    -------
    array
        Array of shape *(n, dim)* of integers, giving how many cell dimensions separate
        the slave from the master along each axis.
    """
    rve_dims = np.asarray(mesh.rve_dims, dtype=float)[: mesh.dim]
    points = mesh.points[:, : mesh.dim]
    offsets = points[pairs[:, 0]] - points[pairs[:, 1]]

    return np.rint(offsets / rve_dims).astype(int)
    # Reading the count off the coordinates is what lets one expression cover the faces,
    # the edges and the corners: a face pair is separated along one axis, an edge pair
    # along one or two, and a corner pair along as many as it has
