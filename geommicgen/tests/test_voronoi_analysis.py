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


class TestVoronoi3DImtsHistogramPath(unittest.TestCase):
    def test_histograms_are_written_inside_results_dir(self):
        import os
        import tempfile
        from types import SimpleNamespace
        from geommicgen.postproc.plotfuncs import plotting_functions as pf

        with tempfile.TemporaryDirectory() as parent_dir:
            results_dir = os.path.join(parent_dir, "results")
            os.makedirs(results_dir)
            vtk_path = os.path.join(results_dir, "voronoi_wIMTs.vtk")
            with open(vtk_path, "w", encoding="utf-8") as f:
                f.write("CELL_DATA 1\nSCALARS id int\n1\n")

            fake_factory = unittest.mock.MagicMock()
            fake_factory.fragment.return_value = ([], [])
            fake_model = unittest.mock.MagicMock()
            fake_model.getBoundingBox.return_value = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            fake_model.getEntitiesInBoundingBox.return_value = [(3, 1)]
            fake_gmsh = unittest.mock.MagicMock(model=fake_model)
            fake_gmsh_ctx = unittest.mock.MagicMock()
            fake_gmsh_ctx.__enter__.return_value = (fake_gmsh, fake_model, fake_factory)
            fake_gmsh_ctx.__exit__.return_value = False

            offsets = np.array(
                [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                dtype=float,
            )
            vor = Voronoi(offsets)
            imts = np.array([[6.0, 1.0, 0.0, 0.0, 0.76, 0.0, 0.35]])

            fn = getattr(pf.plot_voronoi_3d_with_imts, "__wrapped__", pf.plot_voronoi_3d_with_imts)
            with (
                patch.object(pf, "gmsh_view", return_value=fake_gmsh_ctx),
                patch.object(pf, "write_gmsh_view", return_value=(None, vtk_path)),
                patch.object(pf.plt, "savefig") as mock_savefig,
            ):
                fn([], vor, [3.0, 3.0, 3.0], imts, results_dir, save=True, show=False)

            saved_paths = [c.args[0] for c in mock_savefig.call_args_list]
            self.assertEqual(len(saved_paths), 7)
            for i_order, path in enumerate(saved_paths):
                self.assertEqual(
                    path,
                    os.path.join(results_dir, f"voronoi_hist_{i_order}.pdf"),
                )
                self.assertEqual(os.path.dirname(path), results_dir)
        # The first entry of a cell is its surface area, the others its q_l. The q_l
        # do not change with a rotation. With the polar and azimuthal angles swapped,
        # they changed and were not those of a cube
