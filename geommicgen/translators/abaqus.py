"""
Module containing the writer of the input files of the Abaqus solver.

Abaqus reads a mesh from the same file as everything else, so this writes the nodes, the
elements grouped by phase, the node sets of the boundary, and the multi point
constraints that make the cell periodic. The constitutive behaviour and the loading are
placeholders, marked as such: a microstructure says nothing about either.

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
from geommicgen.translators.base import SolverWriter, register_writer

WRITE_CHUNK = 500000
# Number of lines formatted at a time. The file object buffers the writing itself, so
# the chunk is here only to bound the size of the string each format call builds

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

    young_modulus: float
        Young modulus written for every phase of the placeholder materials.

    poisson_ratio: float
        Poisson ratio written for every phase of the placeholder materials.
    """

    name = "abaqus"
    extension = ".inp"

    def __init__(
        self, periodic_constraints=True, young_modulus=1.0e3, poisson_ratio=0.3
    ):
        """Initizalizer for the AbaqusWriter Class."""
        self.periodic_constraints = bool(periodic_constraints)
        self.requires_periodic = self.periodic_constraints
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

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
        blocks = _element_blocks(mesh)
        boundary = mesh.boundary

        with open(file_path, "w") as deck:
            deck.write(
                "** Abaqus input file written by geommicgen.\n"
                "** The mesh is complete; the materials and the step are placeholders,\n"
                "** and are to be replaced before the file is used for anything real.\n"
                "*Heading\n geommicgen microstructure\n"
            )
            self._write_nodes(deck, mesh)
            self._write_elements(deck, mesh, blocks)
            self._write_node_sets(deck, boundary)
            self._write_sections(deck, blocks)
            if self.periodic_constraints:
                self._write_reference_sets(deck, mesh)
                self._write_equations(deck, mesh, boundary)
            # The sets come before the constraints that refer to them, which is the
            # order Abaqus needs them in

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
        for i_block in blocks:
            cell_type, connectivity = mesh.cells[i_block["block"]]
            rows_of_block = connectivity[i_block["rows"]]
            deck.write(
                "*Element, type={0}, elset={1}\n".format(
                    abaqus_element_name(cell_type), i_block["elset"]
                )
            )
            row_format = "%d" + ", %d" * rows_of_block.shape[1] + "\n"
            for i_start in range(0, len(rows_of_block), WRITE_CHUNK):
                chunk = rows_of_block[i_start : i_start + WRITE_CHUNK]
                rows = np.empty((len(chunk), chunk.shape[1] + 1), dtype=np.int64)
                rows[:, 0] = np.arange(element_id + 1, element_id + len(chunk) + 1)
                rows[:, 1:] = chunk + 1
                deck.write((row_format * len(chunk)) % tuple(rows.ravel()))
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

    def _write_sections(self, deck, blocks):
        """Write a section and a placeholder material for every phase."""
        for i_name in sorted({i_block["elset"] for i_block in blocks}):
            material = "MATERIAL_{0}".format(i_name[len("PHASE_") :])
            deck.write(
                "*Solid Section, elset={0}, material={1}\n,\n"
                "*Material, name={1}\n*Elastic\n {2}, {3}\n"
                "** TODO {0}: the real properties\n".format(
                    i_name, material, self.young_modulus, self.poisson_ratio
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
        for i_pairs in _periodic_pairs(boundary):
            coefficients = _cell_offsets(mesh, i_pairs)
            for j_pair, j_coefficients in zip(i_pairs, coefficients):
                for k_dof in range(1, mesh.dim + 1):
                    terms = [
                        (int(j_pair[0]) + 1, k_dof, 1.0),
                        (int(j_pair[1]) + 1, k_dof, -1.0),
                    ]
                    terms += [
                        (_reference_node_id(mesh, i_dir), k_dof, -float(i_count))
                        for i_dir, i_count in enumerate(j_coefficients)
                        if i_count != 0
                    ]
                    _write_equation(deck, terms)
        # The slave comes first because Abaqus eliminates the degree of freedom of the
        # first term, and the slave is the one that is not to be prescribed elsewhere


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


def _write_equation(deck, terms):
    """
    Write one equation.

    Parameters
    ----------
    deck: file
        File being written.

    terms: list
        Tuples *(node, degree of freedom, coefficient)* whose sum is zero.
    """
    deck.write("*Equation\n{0}\n".format(len(terms)))
    for i_start in range(0, len(terms), EQUATION_LINE_TERMS):
        line = terms[i_start : i_start + EQUATION_LINE_TERMS]
        deck.write(
            ", ".join(
                "{0}, {1}, {2:.1f}".format(i_node, i_dof, i_coefficient)
                for i_node, i_dof, i_coefficient in line
            )
            + "\n"
        )


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
    list
        Arrays of shape *(n, 2)* holding the slave and the master of each pair.
    """
    pairs = [boundary.face_pairs[i_axis] for i_axis in sorted(boundary.face_pairs)]
    pairs += [boundary.edge_pairs[i_axis] for i_axis in sorted(boundary.edge_pairs)]
    pairs.append(boundary.corner_pairs)

    return [i_pairs for i_pairs in pairs if len(i_pairs) > 0]
    # The faces are paired from the nodes that lie on one face alone, so no node is
    # constrained twice by collecting the three kinds together


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


def _element_blocks(mesh):
    """
    Split the cells of a mesh into the blocks that share an element type and a phase.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh being written.

    Returns
    -------
    list
        Dictionaries with the index of the block of cells, the rows of that block, and
        the name of the element set they belong to.
    """
    blocks = []
    for i_block, _ in enumerate(mesh.cells):
        order = np.argsort(mesh.phase[i_block], kind="stable")
        values, starts = np.unique(mesh.phase[i_block][order], return_index=True)
        bounds = list(starts) + [len(order)]
        for i_ind, i_phase in enumerate(values):
            blocks.append(
                {
                    "block": i_block,
                    "rows": order[bounds[i_ind] : bounds[i_ind + 1]],
                    "elset": "PHASE_{0}".format(
                        mesh.phase_names.get(int(i_phase), int(i_phase))
                    ),
                }
            )
    # Sorting once gives both the phases present and the rows of each of them, so the
    # phase array is not scanned again for every block

    return blocks
