"""
Module containing the classification of the boundary nodes of a periodic mesh.

A mesh of a periodic microstructure is only usable by a homogenisation solver if the
discretisation of opposite faces matches, so that every node on one face has a partner
at the same position on the opposite one. The classification here is done from the
coordinates alone, which means it applies equally to a mesh produced by this package
and to one produced elsewhere.
"""

import numpy as np
from scipy.spatial import cKDTree

AXIS_NAMES = ("x", "y", "z")
# Names used to build the keys of the faces, edges and corners

DEFAULT_RELATIVE_TOL = 1.0e-8
# Relative tolerance used to decide whether a node lies on a face of the RVE


class PeriodicBoundary:
    """
    Class for the boundary nodes of a periodic mesh.

    Attributes
    ----------
    tol: float
        Absolute tolerance used to place the nodes on the faces of the RVE.

    face_nodes: dict
        Dictionary whose keys are the names of the faces, such as *"x-"* and *"x+"*, and
        whose values are the indices of every node on that face, including the ones that
        also belong to an edge or to a corner.

    face_interior: dict
        Dictionary with the same keys as `face_nodes`, holding only the nodes that lie
        on that face and on no other.

    edge_nodes: dict
        Dictionary whose keys are the names of the edges, such as *"x-y-"*, and whose
        values are the indices of the nodes on that edge. Empty for a two dimensional
        mesh, whose edges are the faces.

    corner_nodes: dict
        Dictionary whose keys are the names of the corners, such as *"x-y-z-"*, and
        whose values are the indices of the nodes at that corner.

    face_pairs: dict
        Dictionary whose keys are the names of the directions, such as *"x"*, and whose
        values are arrays of shape *(n, 2)* with the index of a node on the positive
        face and the index of its partner on the negative one.

    edge_pairs: dict
        Dictionary whose keys are the directions the edges run along, and whose values
        are arrays of shape *(n, 2)* pairing the nodes of the three slave edges with
        those of the master edge. Empty for a two dimensional mesh.

    corner_pairs: array
        Array of shape *(n, 2)* pairing every corner node with the one at the origin.

    unmatched: dict
        Dictionary whose keys are the names of the directions and whose values are the
        indices of the nodes for which no partner was found.

    face_mask: array
        Array of shape *(n_nodes,)* where bit *2k* is set when the node lies on the
        negative face of direction *k* and bit *2k + 1* when it lies on the positive one.
    """

    def __init__(self, tol):
        """
        Initizalizer for the PeriodicBoundary Class.

        Parameters
        ----------
        tol: float
            Absolute tolerance used to place the nodes on the faces of the RVE.
        """
        self.tol = tol
        self.face_nodes = {}
        self.face_interior = {}
        self.edge_nodes = {}
        self.corner_nodes = {}
        self.face_pairs = {}
        self.edge_pairs = {}
        self.corner_pairs = np.zeros((0, 2), dtype=int)
        self.unmatched = {}
        self.face_mask = np.zeros(0, dtype=np.uint8)

    @property
    def is_conforming(self):
        """Whether every boundary node has been paired with one on the opposite face."""
        return not any(len(i_nodes) > 0 for i_nodes in self.unmatched.values())

    def describe_mismatch(self):
        """
        Describe the faces whose discretisations do not match.

        Returns
        -------
        str
            Human readable description of the mismatch, empty when there is none.
        """
        messages = []
        for i_direction in sorted(self.unmatched):
            unmatched = self.unmatched[i_direction]
            if len(unmatched) == 0:
                continue
            negative = len(self.face_nodes["{0}-".format(i_direction)])
            positive = len(self.face_nodes["{0}+".format(i_direction)])
            messages.append(
                "direction {0}: {1} nodes on {0}- and {2} on {0}+, "
                "{3} without a partner".format(
                    i_direction, negative, positive, len(unmatched)
                )
            )

        return "; ".join(messages)


def classify_periodic_boundary(points, rve_dims, tol=None):
    """
    Classify the boundary nodes of a mesh and pair them across opposite faces.

    Parameters
    ----------
    points: array
        Array of shape *(n_nodes, 3)* with the coordinates of the nodes.

    rve_dims: array
        Array containing the dimensions of the microstructure in each spatial direction.

    tol: float
        Absolute tolerance used to place the nodes on the faces of the RVE. When it is
        not supplied, a relative tolerance of `DEFAULT_RELATIVE_TOL` times the largest
        dimension of the RVE is used, which is the rule the LINKS solver applies.

    Returns
    -------
    `.PeriodicBoundary`
        The classification of the boundary nodes.
    """
    rve_dims = np.asarray(rve_dims, dtype=float)
    dim = len(rve_dims)
    coords = np.asarray(points, dtype=float)[:, :dim]
    if tol is None:
        tol = DEFAULT_RELATIVE_TOL * float(np.max(rve_dims))
    # The same rule the solver uses, so that a mesh accepted here is accepted there

    boundary = PeriodicBoundary(tol)

    on_negative = (coords > -tol) & (coords < tol)
    on_positive = (coords > rve_dims - tol) & (coords < rve_dims + tol)
    n_faces = on_negative.sum(axis=1) + on_positive.sum(axis=1)

    mask = np.zeros(len(coords), dtype=np.uint8)
    for i_dir in range(dim):
        mask |= on_negative[:, i_dir].astype(np.uint8) << (2 * i_dir)
        mask |= on_positive[:, i_dir].astype(np.uint8) << (2 * i_dir + 1)
    boundary.face_mask = mask

    for i_dir in range(dim):
        axis = AXIS_NAMES[i_dir]
        boundary.face_nodes["{0}-".format(axis)] = np.where(on_negative[:, i_dir])[0]
        boundary.face_nodes["{0}+".format(axis)] = np.where(on_positive[:, i_dir])[0]
        boundary.face_interior["{0}-".format(axis)] = np.where(
            on_negative[:, i_dir] & (n_faces == 1)
        )[0]
        boundary.face_interior["{0}+".format(axis)] = np.where(
            on_positive[:, i_dir] & (n_faces == 1)
        )[0]
    # A node on more than one face belongs to an edge or to a corner and is paired by
    # the corresponding rule instead

    for i_dir in range(dim):
        axis = AXIS_NAMES[i_dir]
        master = boundary.face_interior["{0}-".format(axis)]
        slave = boundary.face_interior["{0}+".format(axis)]
        others = [i_other for i_other in range(dim) if i_other != i_dir]
        pairs, unmatched = _match_by_coordinates(
            coords[slave][:, others], coords[master][:, others], slave, master, tol
        )
        boundary.face_pairs[axis] = pairs
        boundary.unmatched[axis] = unmatched

    if dim == 3:
        _classify_edges(boundary, coords, on_negative, on_positive, n_faces, tol)
    _classify_corners(boundary, on_negative, on_positive, n_faces, dim)

    return boundary


def _match_by_coordinates(slave_coords, master_coords, slave_index, master_index, tol):
    """
    Pair two sets of nodes by their coordinates.

    Parameters
    ----------
    slave_coords: array
        Coordinates of the nodes to be paired, with the periodic direction removed.

    master_coords: array
        Coordinates of the candidate partners, with the periodic direction removed.

    slave_index: array
        Indices of the slave nodes in the mesh.

    master_index: array
        Indices of the master nodes in the mesh.

    tol: float
        Largest distance at which two nodes are considered partners.

    Returns
    -------
    tuple
        Array of shape *(n, 2)* with the pairs, and array with the indices of the slave
        nodes for which no partner was found.
    """
    if len(slave_index) == 0 or len(master_index) == 0:
        return np.zeros((0, 2), dtype=int), np.asarray(slave_index, dtype=int)

    slave_coords = np.atleast_2d(slave_coords)
    master_coords = np.atleast_2d(master_coords)
    if slave_coords.shape[1] == 0:
        slave_coords = np.zeros((len(slave_index), 1))
        master_coords = np.zeros((len(master_index), 1))
    # A one dimensional periodic direction leaves nothing to match on, so every node is
    # a candidate partner of every other

    tree = cKDTree(master_coords)
    distances, positions = tree.query(slave_coords, distance_upper_bound=tol)
    found = np.isfinite(distances)

    pairs = np.column_stack(
        (np.asarray(slave_index)[found], np.asarray(master_index)[positions[found]])
    ).astype(int)
    unmatched = np.asarray(slave_index)[~found].astype(int)

    return pairs, unmatched


def _classify_edges(boundary, coords, on_negative, on_positive, n_faces, tol):
    """Classify and pair the nodes on the twelve edges of a three dimensional RVE."""
    candidates = np.where(n_faces == 2)[0]
    positive = on_positive[candidates]
    negative = on_negative[candidates]
    # Only the nodes on exactly two faces can lie on an edge, and there are few of
    # them, so the buckets are built from that set instead of from the whole mesh

    for i_dir in range(3):
        axis = AXIS_NAMES[i_dir]
        others = [i_other for i_other in range(3) if i_other != i_dir]
        on_edge = (
            (negative[:, others[0]] | positive[:, others[0]])
            & (negative[:, others[1]] | positive[:, others[1]])
        )
        selections = {}
        for i_signs in ((0, 0), (1, 0), (0, 1), (1, 1)):
            selection = on_edge.copy()
            name = ""
            for i_other, i_sign in zip(others, i_signs):
                selection &= (positive if i_sign else negative)[:, i_other]
                name += "{0}{1}".format(AXIS_NAMES[i_other], "+" if i_sign else "-")
            selections[i_signs] = candidates[selection]
            boundary.edge_nodes[name] = selections[i_signs]
        # The edges running along a direction are named by the two faces bounding them

        master = selections[(0, 0)]
        pairs = []
        unmatched = []
        for i_signs in ((1, 0), (0, 1), (1, 1)):
            slave = selections[i_signs]
            edge_pairs, edge_unmatched = _match_by_coordinates(
                coords[slave][:, [i_dir]], coords[master][:, [i_dir]], slave, master, tol
            )
            pairs.append(edge_pairs)
            unmatched.append(edge_unmatched)
        boundary.edge_pairs[axis] = np.concatenate(pairs)
        boundary.unmatched[axis] = np.concatenate(
            [boundary.unmatched[axis]] + unmatched
        )


def _classify_corners(boundary, on_negative, on_positive, n_faces, dim):
    """Classify and pair the corners of the RVE."""
    candidates = np.where(n_faces == dim)[0]
    positive = on_positive[candidates]
    negative = on_negative[candidates]
    # A node on as many faces as there are dimensions is a corner, and a conforming
    # mesh has exactly one at each of them

    corners = []
    for i_corner in range(2 ** dim):
        selection = np.ones(len(candidates), dtype=bool)
        name = ""
        for i_dir in range(dim):
            is_positive = bool((i_corner >> i_dir) & 1)
            selection &= (positive if is_positive else negative)[:, i_dir]
            name += "{0}{1}".format(AXIS_NAMES[i_dir], "+" if is_positive else "-")
        corners.append(candidates[selection])
        boundary.corner_nodes[name] = corners[-1]

    master = corners[0]
    pairs = []
    for i_corner in range(1, 2 ** dim):
        if len(corners[i_corner]) > 0 and len(master) > 0:
            pairs.append([int(corners[i_corner][0]), int(master[0])])
    boundary.corner_pairs = (
        np.asarray(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)
    )
