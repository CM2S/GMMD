"""
Unit tests regarding voronoi analysis.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
import time
from geommicgen.microstructure.particleclasses import (
    Ellipsoid,
    Ellipse,
    Cylinder,
    Particle,
    Disk,
    Sphere,
    Point,
    Line,
)
from geommicgen.postproc.plotfuncs.plotting_functions import (
    plot_particles_3d,
    plot_particles_2d,
)
from scipy.spatial import Voronoi
from geommicgen.postproc.voronoimetrics.voronoi_analysis import (
    compute_3d_irreducible_minkowski_tensors,
    compute_3d_irreducible_minkowski_tensors_polyhedron,
    update_indices,
)


class TestUpdateIndices(unittest.TestCase):
    def test_indices(self):
        old_indices = [1, 3, 4, 5, 7, 9, 10, 15]
        removed_ind = [2, 4, 6, 8, 12]
        new_ind = update_indices(old_indices, removed_ind)
        # print(new_ind)


class TestMinkowskiTensors3D(unittest.TestCase):
    """The 3D irreducible Minkowski tensors of a cube."""

    CUBE_Q = [1.0, 0.0, 0.0, 0.0, 0.7638, 0.0, 0.3536]
    # Simple cubic values of Table 1 of Mickel et al., J. Chem. Phys. 138, 044501
    # (2013). The cell of the lattice is a cube, whose facets have equal areas, so
    # the facet-area weighting of their Eq. (2) gives the same values

    def setUp(self):
        self.rotation = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]

    def test_polyhedron_of_a_cube(self):
        normals = np.vstack([np.eye(3), -np.eye(3)])
        for i_name, i_normals in (
            ("aligned", normals),
            ("rotated", normals @ self.rotation.T),
        ):
            with self.subTest(i_name):
                imts, _ = compute_3d_irreducible_minkowski_tensors_polyhedron(
                    i_normals, np.ones(6)
                )
                np.testing.assert_allclose(imts, self.CUBE_Q, atol=1e-4)

    def test_voronoi_cell_of_a_cubic_lattice(self):
        offsets = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        )
        # Point 13 is the particle, the others its periodic images, so its cell is
        # a unit cube
        for i_name, i_points in (
            ("aligned", offsets),
            ("rotated", offsets @ self.rotation.T),
        ):
            with self.subTest(i_name):
                imts, _ = compute_3d_irreducible_minkowski_tensors(Voronoi(i_points))
                self.assertAlmostEqual(imts[0][0], 6.0)
                np.testing.assert_allclose(imts[0][1:], self.CUBE_Q[1:], atol=1e-4)
        # The first entry of a cell is its surface area, the others its q_l. The q_l
        # do not change with a rotation. With the polar and azimuthal angles swapped,
        # they changed and were not those of a cube
