"""
Module containing the element conventions of the LINKS solver.

The correspondence between the element types and the order in which LINKS expects the
nodes of an element to be listed is derived from the gmsh2links package, by
A. M. Couto Carneiro (CM2S, FEUP), and converted here from the ordering gmsh uses on
input to the one used by VTK, which is the ordering a `.Mesh` carries. The tables were
checked against the node numbering in the shape function routines of LINKS itself.

gmsh2links is licensed under the GPL-3 and this package under the BSD-3; what is taken
from it is the fact of how LINKS numbers the nodes of each element, verified against
LINKS, and no code, so the two licences do not meet.
"""

# TODO: decide how to credit A. M. Couto Carneiro beyond this docstring, before the
# JOSS submission: an Acknowledgements section in the readme naming gmsh2links; the
# Acknowledgements of paper.md; a CITATION.cff for GMMD; or co-authorship of the paper,
# which is his and the authors' to decide on the size of the LINKS-side contribution.

LINKS_ELEMENT_NAMES = {
    "triangle": "TRI3",
    "triangle6": "TRI6",
    "quad": "QUAD4",
    "quad8": "QUAD8",
    "quad9": "QUAD9",
    "tetra": "TETRA4",
    "tetra10": "TETRA10",
    "hexahedron": "HEXA8",
    "hexahedron20": "HEXA20",
}
# Correspondence between the cell types of meshio and the element names of LINKS

LINKS_DEFAULT_GAUSS_POINTS = {
    "triangle": None,
    "triangle6": 3,
    "quad": 4,
    "quad8": 4,
    "quad9": 9,
    "tetra": 4,
    "tetra10": 4,
    "hexahedron": 8,
    "hexahedron20": 8,
}
# Number of Gauss points written after the element type. TRI3 takes no such line, since
# the routine that reads it does not look for one

VTK_TO_LINKS = {
    "quad8": [0, 4, 1, 5, 2, 6, 3, 7],
    "quad9": [0, 4, 1, 5, 2, 6, 3, 7, 8],
    "hexahedron20": [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11,
        16, 17, 18, 19,
        12, 13, 14, 15,
    ],
}
# Permutations taking a cell from the VTK ordering to the LINKS one. The first order
# elements, the six node triangle and the ten node tetrahedron need none, because the
# two conventions agree for them. LINKS interleaves the mid side nodes of the second
# order quadrilaterals with the corners, and lists the vertical edges of a twenty node
# hexahedron before the ones of its top face


def links_element_name(cell_type):
    """
    Get the LINKS name of a cell type.

    Parameters
    ----------
    cell_type: str
        Name of the cell type, as used by meshio.

    Returns
    -------
    str
        Name of the element in LINKS.

    Raises
    ------
    ValueError:
        If LINKS has no element of that type.
    """
    if cell_type not in LINKS_ELEMENT_NAMES:
        raise ValueError(
            "LINKS has no element corresponding to the cell type {0}. The supported "
            "types are {1}.".format(
                cell_type, ", ".join(sorted(LINKS_ELEMENT_NAMES))
            )
        )

    return LINKS_ELEMENT_NAMES[cell_type]


def reorder_connectivity(cell_type, connectivity):
    """
    Reorder the nodes of the cells of a block into the LINKS ordering.

    Parameters
    ----------
    cell_type: str
        Name of the cell type, as used by meshio.

    connectivity: array
        Array of shape *(n_cells, n_nodes)* with the nodes of every cell.

    Returns
    -------
    array
        The connectivity with the nodes of every cell in the LINKS ordering.
    """
    if cell_type not in VTK_TO_LINKS:
        return connectivity

    return connectivity[:, VTK_TO_LINKS[cell_type]]
