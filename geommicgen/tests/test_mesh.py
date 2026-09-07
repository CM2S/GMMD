import os
import tempfile
import unittest

import numpy as np

from geommicgen.errors.error_classes import MeshTooLargeError, PeriodicityError
from geommicgen.meshing.mesh import DEFAULT_MAX_CELLS, Mesh
from geommicgen.meshing.periodic import classify_periodic_boundary
from geommicgen.meshing.writers import read_mesh, write_vtk_image, write_vtu
from geommicgen.tests.helpers import structured_mesh


class TestStructuredMaterialization(unittest.TestCase):
    """Test class for the building of the cells of a structured mesh."""

    def test_two_dimensional_counts(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        self.assertEqual(mesh.n_cells, 16)
        self.assertEqual(mesh.points.shape, (25, 3))
        self.assertEqual(mesh.cells[0][0], "quad")
        self.assertEqual(len(mesh.cells[0][1]), 16)

    def test_three_dimensional_counts(self):
        mesh = structured_mesh((3, 3, 3), [1.0, 1.0, 1.0])
        self.assertEqual(mesh.n_cells, 27)
        self.assertEqual(mesh.points.shape, (64, 3))
        self.assertEqual(mesh.cells[0][0], "hexahedron")

    def test_phase_fraction_matches_the_grid(self):
        phase_grid = np.ones((10, 10), dtype=int)
        phase_grid[2:6, 2:6] = 2
        mesh = structured_mesh((10, 10), [1.0, 1.0], phase_grid)
        phase = mesh.phase[0]
        self.assertEqual(len(phase), 100)
        self.assertAlmostEqual(np.mean(phase == 2), np.mean(phase_grid == 2))
        # The cells are built in the same order as the voxels, so the fractions agree

    def test_cells_are_not_built_for_the_cell_count(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        self.assertEqual(mesh.n_cells, 16)
        self.assertIsNone(mesh._cells)
        # Asking how many cells there are must not materialize them

    def test_too_large_grid_is_refused(self):
        mesh = structured_mesh((4, 4, 4), [1.0, 1.0, 1.0])
        mesh.structured.shape = (500, 500, 500)
        with self.assertRaises(MeshTooLargeError) as context:
            mesh._materialize(max_cells=DEFAULT_MAX_CELLS)
        message = str(context.exception)
        self.assertIn("125000000", message)
        self.assertIn("20000000", message)


class TestPeriodicClassification(unittest.TestCase):
    """Test class for the classification of the boundary nodes."""

    def test_two_dimensional_structure(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        boundary = mesh.classify_boundary()
        for i_face in ("x-", "x+", "y-", "y+"):
            self.assertEqual(len(boundary.face_nodes[i_face]), 5)
            self.assertEqual(len(boundary.face_interior[i_face]), 3)
        self.assertEqual(len(boundary.face_pairs["x"]), 3)
        self.assertEqual(len(boundary.face_pairs["y"]), 3)
        self.assertEqual(len(boundary.corner_pairs), 3)
        self.assertTrue(boundary.is_conforming)
        # Four corners give three pairs, since one of them is the master

    def test_three_dimensional_structure(self):
        mesh = structured_mesh((3, 3, 3), [1.0, 1.0, 1.0])
        boundary = mesh.classify_boundary()
        for i_direction in ("x", "y", "z"):
            self.assertEqual(len(boundary.face_pairs[i_direction]), 4)
            self.assertEqual(len(boundary.edge_pairs[i_direction]), 6)
        self.assertEqual(len(boundary.corner_pairs), 7)
        self.assertEqual(len(boundary.edge_nodes), 12)
        self.assertEqual(len(boundary.corner_nodes), 8)
        self.assertTrue(boundary.is_conforming)

    def test_pairs_are_at_the_same_position(self):
        mesh = structured_mesh((3, 3, 3), [1.0, 2.0, 3.0])
        boundary = mesh.classify_boundary()
        points = mesh.points
        for i_ind, i_direction in enumerate("xyz"):
            for i_slave, i_master in boundary.face_pairs[i_direction]:
                offset = points[i_slave] - points[i_master]
                expected = np.zeros(3)
                expected[i_ind] = mesh.rve_dims[i_ind]
                np.testing.assert_allclose(offset, expected, atol=1e-12)
        # A pair differs by exactly one period along the direction it is paired in

    def test_conformity_is_checked(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        mesh.check_periodic_conformity()

    def test_perturbed_node_is_detected(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        boundary = mesh.classify_boundary()
        points = mesh.points.copy()
        points[boundary.face_interior["x+"][0], 1] += 1.0e-3
        broken = Mesh(
            mesh.rve_dims, points=points, cells=mesh.cells, phase=mesh.phase
        )
        with self.assertRaises(PeriodicityError) as context:
            broken.check_periodic_conformity()
        self.assertIn("direction x", str(context.exception))
        # The message has to name the face, since that is what the user must fix

    def test_tolerance_is_relative_to_the_rve(self):
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        boundary = classify_periodic_boundary(points, [10.0, 1.0])
        self.assertAlmostEqual(boundary.tol, 1.0e-8 * 10.0)


class TestMeshRoundTrip(unittest.TestCase):
    """Test class for the writing and reading of meshes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "femsh.vtu")
        phase_grid = np.ones((4, 4), dtype=int)
        phase_grid[1:3, 1:3] = 2
        self.mesh = structured_mesh((4, 4), [1.0, 1.0], phase_grid)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vtu_round_trip(self):
        write_vtu(self.mesh, self.file_path)
        restored = read_mesh(self.file_path)
        np.testing.assert_allclose(restored.rve_dims, self.mesh.rve_dims)
        self.assertEqual(restored.dim, 2)
        self.assertEqual(restored.matrix_phase, "1")
        self.assertEqual(restored.n_cells, self.mesh.n_cells)
        np.testing.assert_array_equal(restored.phase[0], self.mesh.phase[0])
        np.testing.assert_allclose(restored.points, self.mesh.points)
        self.assertTrue(restored.periodic)

    def test_sidecar_is_written(self):
        write_vtu(self.mesh, self.file_path)
        sidecar = os.path.join(self.temp_dir.name, "femsh.mesh.json")
        self.assertTrue(os.path.exists(sidecar))

    def test_reads_a_mesh_without_a_sidecar(self):
        write_vtu(self.mesh, self.file_path, write_sidecar=False)
        restored = read_mesh(self.file_path)
        np.testing.assert_allclose(restored.rve_dims, [1.0, 1.0])
        self.assertEqual(restored.n_cells, self.mesh.n_cells)
        self.assertTrue(restored.boundary.is_conforming)
        # Without a sidecar the dimensions come from the bounding box, which is what
        # makes a mesh produced by another tool usable

    def test_vtk_image(self):
        image_path = os.path.join(self.temp_dir.name, "grid.vtk")
        write_vtk_image(self.mesh, image_path)
        with open(image_path, "r") as vtk_file:
            contents = vtk_file.read()
        self.assertIn("DATASET STRUCTURED_POINTS", contents)
        self.assertIn("DIMENSIONS 5 5 1", contents)
        self.assertIn("SCALARS phase int 1", contents)
        self.assertIn("CELL_DATA 16", contents)

    def test_vtk_image_refuses_unstructured(self):
        unstructured = Mesh(
            self.mesh.rve_dims,
            points=self.mesh.points,
            cells=self.mesh.cells,
            phase=self.mesh.phase,
        )
        with self.assertRaises(ValueError):
            write_vtk_image(unstructured, os.path.join(self.temp_dir.name, "x.vtk"))


if __name__ == "__main__":
    unittest.main()
