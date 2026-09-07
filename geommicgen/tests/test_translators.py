import os
import tempfile
import unittest

import numpy as np

from geommicgen.errors.error_classes import PeriodicityError
from geommicgen.meshing.mesh import Mesh, StructuredInfo
from geommicgen.translators import available_writers, get_writer
from geommicgen.translators.crate import grid_file_name
from geommicgen.translators.reorder import (
    LINKS_DEFAULT_GAUSS_POINTS,
    VTK_TO_LINKS,
    links_element_name,
    reorder_connectivity,
)


def structured_mesh(shape, rve_dims, phase_grid=None):
    """Build a structured mesh with the supplied number of voxels."""
    if phase_grid is None:
        phase_grid = np.ones(shape, dtype=int)
    spacing = np.asarray(rve_dims, dtype=float) / np.asarray(shape, dtype=float)

    return Mesh(
        rve_dims,
        structured=StructuredInfo(phase_grid, spacing),
        phase_names={1: "1", 2: "2"},
        matrix_phase="1",
        periodic=True,
    )


def parse_links_mesh(file_path):
    """Read back the blocks of a LINKS mesh file."""
    with open(file_path, "r") as mesh_file:
        lines = [line.strip() for line in mesh_file if line.strip()]

    blocks = {}
    current = None
    for i_line in lines:
        head = i_line.split()[0]
        if head in ("ELEMENT_GROUPS", "ELEMENT_TYPES", "NODE_COORDINATES", "ELEMENTS"):
            current = head
            blocks[current] = {"count": int(i_line.split()[1]), "lines": []}
        elif current is not None:
            blocks[current]["lines"].append(i_line)

    return blocks


class TestReorderTables(unittest.TestCase):
    """Test class for the element conventions of LINKS."""

    def test_permutations_are_permutations(self):
        for i_type, i_permutation in VTK_TO_LINKS.items():
            self.assertEqual(
                sorted(i_permutation),
                list(range(len(i_permutation))),
                "the table for {0} is not a permutation".format(i_type),
            )

    def test_element_names(self):
        self.assertEqual(links_element_name("triangle"), "TRI3")
        self.assertEqual(links_element_name("tetra"), "TETRA4")
        self.assertEqual(links_element_name("hexahedron20"), "HEXA20")
        with self.assertRaises(ValueError):
            links_element_name("wedge")

    def test_three_node_triangle_takes_no_gauss_points(self):
        self.assertIsNone(LINKS_DEFAULT_GAUSS_POINTS["triangle"])
        # The routine that reads a TRI3 does not look for a Gauss point line

    def test_first_order_elements_are_not_reordered(self):
        connectivity = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(
            reorder_connectivity("quad", connectivity), connectivity
        )

    def test_quad8_midside_nodes_sit_between_the_corners(self):
        corners = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        midsides = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
        points = np.vstack((corners, midsides))
        connectivity = np.arange(8).reshape(1, 8)
        reordered = reorder_connectivity("quad8", connectivity)[0]
        ordered_points = points[reordered]
        for i_node in range(0, 8, 2):
            before = ordered_points[i_node]
            after = ordered_points[(i_node + 2) % 8]
            middle = ordered_points[i_node + 1]
            np.testing.assert_allclose(middle, 0.5 * (before + after))
        # LINKS interleaves the mid side nodes with the corners, so every odd node has
        # to be the midpoint of the two corners around it


class TestLinksWriter(unittest.TestCase):
    """Test class for the writer of the LINKS mesh files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "femsh.mesh")
        phase_grid = np.ones((3, 3), dtype=int)
        phase_grid[1, 1] = 2
        self.mesh = structured_mesh((3, 3), [1.0, 1.0], phase_grid)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_the_mesh_and_an_example(self):
        written = get_writer("links")().write(self.mesh, self.file_path)
        self.assertEqual(len(written), 2)
        for i_path in written:
            self.assertTrue(os.path.exists(i_path))
        with open(written[1], "r") as example_file:
            example = example_file.read()
        self.assertIn("MESH_FILE_RELATIVE femsh.mesh", example)
        self.assertIn("ANALYSIS_TYPE 2", example)
        self.assertIn("MATERIALS 2", example)
        # The mesh file holds no materials and no boundary conditions; the example does

    def test_identifiers_are_dense(self):
        get_writer("links")().write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)

        node_ids = [int(line.split()[0]) for line in blocks["NODE_COORDINATES"]["lines"]]
        self.assertEqual(node_ids, list(range(1, blocks["NODE_COORDINATES"]["count"] + 1)))

        element_ids = [int(line.split()[0]) for line in blocks["ELEMENTS"]["lines"]]
        self.assertEqual(element_ids, list(range(1, blocks["ELEMENTS"]["count"] + 1)))
        # LINKS requires both to be dense and to start at one

    def test_groups_are_consistent(self):
        get_writer("links")().write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)
        n_groups = blocks["ELEMENT_GROUPS"]["count"]
        self.assertEqual(len(blocks["ELEMENT_GROUPS"]["lines"]), n_groups)
        self.assertEqual(n_groups, 2)

        for i_line in blocks["ELEMENTS"]["lines"]:
            group = int(i_line.split()[1])
            self.assertGreaterEqual(group, 1)
            self.assertLessEqual(group, n_groups)
        # An element referring to a group that was not declared makes LINKS stop

    def test_nodes_per_element_match_the_type(self):
        get_writer("links")().write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)
        for i_line in blocks["ELEMENTS"]["lines"]:
            self.assertEqual(len(i_line.split()) - 2, 4)
        self.assertIn("QUAD4", " ".join(blocks["ELEMENT_TYPES"]["lines"]))

    def test_node_indices_are_one_based_and_in_range(self):
        get_writer("links")().write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)
        n_nodes = blocks["NODE_COORDINATES"]["count"]
        for i_line in blocks["ELEMENTS"]["lines"]:
            for i_node in i_line.split()[2:]:
                self.assertGreaterEqual(int(i_node), 1)
                self.assertLessEqual(int(i_node), n_nodes)

    def test_every_element_is_written_once(self):
        get_writer("links")().write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)
        self.assertEqual(blocks["ELEMENTS"]["count"], self.mesh.n_cells)
        self.assertEqual(len(blocks["ELEMENTS"]["lines"]), self.mesh.n_cells)

    def test_refuses_a_non_periodic_mesh(self):
        boundary = self.mesh.classify_boundary()
        points = self.mesh.points.copy()
        points[boundary.face_interior["x+"][0], 1] += 1.0e-3
        broken = Mesh(
            self.mesh.rve_dims,
            points=points,
            cells=self.mesh.cells,
            phase=self.mesh.phase,
            phase_names=self.mesh.phase_names,
            matrix_phase="1",
        )
        with self.assertRaises(PeriodicityError):
            get_writer("links")().write(broken, self.file_path)

    def test_can_be_asked_not_to_check_periodicity(self):
        boundary = self.mesh.classify_boundary()
        points = self.mesh.points.copy()
        points[boundary.face_interior["x+"][0], 1] += 1.0e-3
        broken = Mesh(
            self.mesh.rve_dims,
            points=points,
            cells=self.mesh.cells,
            phase=self.mesh.phase,
            phase_names=self.mesh.phase_names,
            matrix_phase="1",
        )
        written = get_writer("links")(require_periodic=False).write(
            broken, self.file_path
        )
        self.assertTrue(os.path.exists(written[0]))
        # The escape exists because some boundary conditions tolerate a non matching mesh


class TestCrateWriter(unittest.TestCase):
    """Test class for the writer of the voxel grids."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_grid_is_written_unchanged(self):
        phase_grid = np.ones((5, 5), dtype=int)
        phase_grid[1:4, 1:4] = 2
        mesh = structured_mesh((5, 5), [1.0, 1.0], phase_grid)
        path = os.path.join(self.temp_dir.name, "grid.rgmsh")
        written = get_writer("crate")().write(mesh, path)
        restored = np.load(written[0])
        np.testing.assert_array_equal(restored, phase_grid)
        self.assertEqual(restored.dtype, phase_grid.dtype)
        # The array is the whole contract with the solver and must not change

    def test_does_not_need_the_cells(self):
        self.assertFalse(get_writer("crate").needs_cells)
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        get_writer("crate")().write(mesh, os.path.join(self.temp_dir.name, "g.rgmsh"))
        self.assertIsNone(mesh._cells)
        # Writing the grid must not build the cells, which is what makes it usable for
        # resolutions that could never be expressed as elements

    def test_refuses_an_unstructured_mesh(self):
        mesh = structured_mesh((3, 3), [1.0, 1.0])
        unstructured = Mesh(
            mesh.rve_dims, points=mesh.points, cells=mesh.cells, phase=mesh.phase
        )
        with self.assertRaises(ValueError):
            get_writer("crate")().write(
                unstructured, os.path.join(self.temp_dir.name, "g.rgmsh")
            )

    def test_file_name_carries_the_deck_and_the_resolution(self):
        self.assertEqual(
            grid_file_name("2D_example_ellipses.mdsim", (100, 100)),
            "2D_example_ellipses_100_100.rgmsh",
        )
        self.assertEqual(grid_file_name(None, (70, 70, 70)), "70_70_70.rgmsh")


class TestMeshioWriters(unittest.TestCase):
    """Test class for the writers of the formats meshio supports."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mesh = structured_mesh((3, 3), [1.0, 1.0])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_recommended_formats_are_registered(self):
        writers = available_writers()
        for i_format in ("vtu", "vtk", "xdmf", "gmsh", "links", "crate"):
            self.assertIn(i_format, writers)

    def test_unsafe_formats_are_not_registered(self):
        writers = available_writers()
        for i_format in ("abaqus", "ansys", "permas", "dolfin-xml"):
            self.assertNotIn(i_format, writers)
        # meshio would write these without the phase, or with unusable element types

    def test_vtu_is_readable_again(self):
        import meshio

        path = os.path.join(self.temp_dir.name, "mesh.vtu")
        get_writer("vtu")().write(self.mesh, path)
        read = meshio.read(path)
        self.assertEqual(len(read.points), len(self.mesh.points))
        self.assertIn("phase", read.cell_data)


if __name__ == "__main__":
    unittest.main()
