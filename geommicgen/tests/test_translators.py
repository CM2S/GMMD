import os
import tempfile
import unittest

import numpy as np

from geommicgen.errors.error_classes import (
    MissingOptionalDependency,
    PeriodicityError,
)
from geommicgen.meshing.mesh import Mesh
from geommicgen.translators import MeshioWriter, available_writers, get_writer
from geommicgen.translators.abaqus import (
    REFERENCE_NODE_NAMES,
    abaqus_element_name,
)
from geommicgen.translators.links import uniform_gauss_points
from geommicgen.translators.reorder import (
    LINKS_DEFAULT_GAUSS_POINTS,
    VTK_TO_LINKS,
    links_element_name,
    reorder_connectivity,
)
from geommicgen.tests.helpers import non_conforming_mesh, structured_mesh


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
        broken = non_conforming_mesh(self.mesh)
        with self.assertRaises(PeriodicityError):
            get_writer("links")().write(broken, self.file_path)

    def test_can_be_asked_not_to_check_periodicity(self):
        broken = non_conforming_mesh(self.mesh)
        written = get_writer("links")(boundary_type="Mortar_Periodic_Condition").write(
            broken, self.file_path
        )
        self.assertTrue(os.path.exists(written[0]))
        with open(written[1], "r") as example_file:
            self.assertIn(
                "Boundary_Type Mortar_Periodic_Condition", example_file.read()
            )
        # The escape is asking for a constraint that ties faces which do not match,
        # which is a thing the deck says rather than something the writer decides

    def test_gauss_points_asked_for_in_the_options(self):
        writer = get_writer("links").from_options({"gauss_points": 6})
        writer.write(self.mesh, self.file_path)
        blocks = parse_links_mesh(self.file_path)

        self.assertIn("6 GP", blocks["ELEMENT_TYPES"]["lines"])
        # The mesh is of quadrilaterals, whose default is four

    def test_the_three_node_triangle_is_never_given_gauss_points(self):
        self.assertNotIn("triangle", uniform_gauss_points(6))
        # It takes no Gauss point line at all, so asking for one on every element must
        # not put one there: LINKS would read the following line as the Gauss points

    def test_an_unknown_boundary_type_is_refused(self):
        with self.assertRaises(ValueError) as context:
            get_writer("links")(boundary_type="Mortar")
        self.assertIn("Mortar_Periodic_Condition", str(context.exception))
        # The message lists what LINKS does accept, which is the thing a near miss needs

    def test_the_constraints_that_pair_nodes_need_a_conforming_mesh(self):
        for i_type in (
            "Periodic_Condition",
            "Kouznetsova_Periodic_Condition",
            "Luscher_Periodic_Condition",
            "Luscher_Periodic_Condition_LM",
        ):
            self.assertTrue(
                get_writer("links")(boundary_type=i_type).requires_periodic, i_type
            )
        for i_type in (
            "Mortar_Periodic_Condition",
            "Mortar_Periodic_Condition_II",
            "Luscher_Mortar_Periodic_Condition",
            "Taylor_Condition",
            "Linear_Condition",
            "Uniform_Traction_Condition",
            "Luscher_Minimal_Condition",
        ):
            self.assertFalse(
                get_writer("links")(boundary_type=i_type).requires_periodic, i_type
            )
        # These four and no others are what LINKS runs its own periodicity verification
        # for, in ioctrl/rve/getbcnnodes2d.f90 and the three files beside it


def parse_abaqus_equations(file_path):
    """Read back the equations of an Abaqus input file as lists of terms."""
    with open(file_path, "r") as deck:
        lines = deck.read().splitlines()

    equations = []
    i_line = 0
    while i_line < len(lines):
        if lines[i_line].strip() != "*Equation":
            i_line += 1
            continue
        n_terms = int(lines[i_line + 1])
        entries = []
        i_line += 2
        while len(entries) < 3 * n_terms:
            entries += [i_entry.strip() for i_entry in lines[i_line].split(",")]
            i_line += 1
        equations.append(
            [
                (int(entries[i]), int(entries[i + 1]), float(entries[i + 2]))
                for i in range(0, 3 * n_terms, 3)
            ]
        )

    return equations


def abaqus_keyword_arguments(file_path, keyword):
    """Read back the arguments of every occurrence of one Abaqus keyword."""
    with open(file_path, "r") as deck:
        return [
            dict(
                i_argument.strip().split("=", 1)
                for i_argument in i_line.strip().split(",")[1:]
                if "=" in i_argument
            )
            for i_line in deck
            if i_line.strip().split(",")[0].strip().lower() == keyword.lower()
        ]


class TestAbaqusWriter(unittest.TestCase):
    """Test class for the writer of the Abaqus input files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "femsh.inp")

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, mesh, **kwargs):
        """Write a mesh and give back the path of the file."""
        get_writer("abaqus")(**kwargs).write(mesh, self.file_path)

        return self.file_path

    def keyword_values(self, path, keyword, argument):
        """Collect one argument of every occurrence of a keyword."""
        return [
            i_arguments[argument]
            for i_arguments in abaqus_keyword_arguments(path, keyword)
        ]

    def n_periodic_pairs(self, mesh):
        """Count the pairs of nodes the periodicity relates."""
        boundary = mesh.boundary

        return (
            sum(len(i_pairs) for i_pairs in boundary.face_pairs.values())
            + sum(len(i_pairs) for i_pairs in boundary.edge_pairs.values())
            + len(boundary.corner_pairs)
        )

    def test_meshio_reads_the_mesh_back(self):
        import meshio

        mesh = structured_mesh((3, 3, 3), [1.0, 1.0, 1.0])
        back = meshio.read(self.write(mesh))

        self.assertTrue(np.allclose(back.points[: len(mesh.points)], mesh.points))
        self.assertTrue(
            np.array_equal(
                np.vstack([i_block.data for i_block in back.cells]), mesh.cells[0][1]
            )
        )
        # The reference nodes follow the nodes of the mesh, which is why the comparison
        # is of the first ones alone

    def test_one_equation_per_degree_of_freedom_per_pair(self):
        for i_shape, i_dims in (((3, 3), [1.0, 1.0]), ((3, 3, 3), [1.0, 1.0, 1.0])):
            with self.subTest(shape=i_shape):
                mesh = structured_mesh(i_shape, i_dims)
                equations = parse_abaqus_equations(self.write(mesh))
                self.assertEqual(len(equations), mesh.dim * self.n_periodic_pairs(mesh))

    def test_every_slave_degree_of_freedom_is_eliminated_once(self):
        mesh = structured_mesh((3, 3, 3), [1.0, 1.0, 1.0])
        equations = parse_abaqus_equations(self.write(mesh))

        eliminated = [i_terms[0][:2] for i_terms in equations]
        self.assertEqual(len(eliminated), len(set(eliminated)))
        masters = {i_terms[1][0] for i_terms in equations}
        self.assertFalse(masters & {i_node for i_node, _ in eliminated})
        # Abaqus eliminates the degree of freedom of the first term, so a node named
        # first twice, or named first and also used as a master, is over constrained

    def test_the_reference_nodes_carry_the_separation(self):
        mesh = structured_mesh((3, 3), [1.0, 1.0])
        equations = parse_abaqus_equations(self.write(mesh))
        reference = {len(mesh.points) + i_dir + 1: i_dir for i_dir in range(mesh.dim)}

        for i_terms in equations:
            slave, master = i_terms[0][0] - 1, i_terms[1][0] - 1
            separation = (
                mesh.points[slave][: mesh.dim] - mesh.points[master][: mesh.dim]
            ) / np.asarray(mesh.rve_dims)
            named = {reference[i_node]: -i_value for i_node, _, i_value in i_terms[2:]}
            for i_dir in range(mesh.dim):
                self.assertAlmostEqual(named.get(i_dir, 0.0), round(separation[i_dir]))
        # A pair separated by one cell along an axis carries that axis' reference node,
        # which is what makes one expression serve the faces, the edges and the corners

    def test_an_element_set_per_phase(self):
        mesh = structured_mesh(
            (2, 2), [1.0, 1.0], phase_grid=np.array([[1, 2], [2, 1]])
        )
        path = self.write(mesh)

        self.assertEqual(
            sorted(self.keyword_values(path, "*Element", "elset")),
            ["PHASE_1", "PHASE_2"],
        )
        self.assertEqual(
            sorted(self.keyword_values(path, "*Solid Section", "elset")),
            ["PHASE_1", "PHASE_2"],
        )
        self.assertEqual(set(self.keyword_values(path, "*Element", "type")), {"CPE4"})

    def test_the_boundary_sets_are_named_after_what_holds_them(self):
        mesh = structured_mesh((3, 3, 3), [1.0, 1.0, 1.0])
        names = set(self.keyword_values(self.write(mesh), "*Nset", "nset"))

        self.assertIn("FACE_XNEG", names)
        self.assertIn("EDGE_YNEG_ZNEG", names)
        self.assertIn("CORNER_XPOS_YPOS_ZPOS", names)
        self.assertIn("RP_Z", names)

    def test_a_two_dimensional_cell_has_corners_and_no_edges(self):
        mesh = structured_mesh((3, 3), [1.0, 1.0])
        names = set(self.keyword_values(self.write(mesh), "*Nset", "nset"))

        self.assertIn("CORNER_XPOS_YPOS", names)
        self.assertFalse({i_name for i_name in names if i_name.startswith("EDGE_")})
        # A node on two faces is an edge in three dimensions and a corner in two

    def test_refuses_a_non_periodic_mesh(self):
        broken = non_conforming_mesh(structured_mesh((3, 3), [1.0, 1.0]))

        with self.assertRaises(PeriodicityError):
            get_writer("abaqus")().write(broken, self.file_path)

    def test_without_the_constraints_there_are_no_reference_nodes(self):
        mesh = structured_mesh((3, 3), [1.0, 1.0])
        path = self.write(mesh, periodic_constraints=False)

        self.assertEqual(parse_abaqus_equations(path), [])
        names = set(self.keyword_values(path, "*Nset", "nset"))
        self.assertFalse(names & set(REFERENCE_NODE_NAMES))
        with open(path) as deck:
            self.assertEqual(len(deck.read().split("*Node")), 2)
        # The mesh and its sets are still written; what a deck without the constraints
        # loses is the periodicity, and with it any reason to carry reference nodes

    def test_an_element_with_no_abaqus_name(self):
        with self.assertRaises(ValueError) as context:
            abaqus_element_name("wedge")
        self.assertIn("hexahedron20", str(context.exception))


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
        for i_format in ("ansys", "permas", "dolfin-xml"):
            self.assertNotIn(i_format, writers)
        # meshio would write these without the phase, or with unusable element types

    def test_abaqus_is_not_the_meshio_writer(self):
        self.assertNotIsInstance(get_writer("abaqus")(), MeshioWriter)
        # The format is offered, but by the writer of this package: meshio names two
        # dimensional triangles after a rigid element and cannot write a constraint

    def test_formats_needing_an_extra_package_say_so(self):
        for i_format in ("xdmf", "exodus"):
            writer = get_writer(i_format)
            if writer.requires_package is None:
                continue
            try:
                __import__(writer.requires_package)
            except ImportError:
                path = os.path.join(self.temp_dir.name, "m" + writer.extension)
                with self.assertRaises(MissingOptionalDependency) as context:
                    writer().write(self.mesh, path)
                self.assertIn(writer.requires_package, str(context.exception))
        # meshio raises a bare import error from inside itself for these formats, which
        # says nothing about how to fix it

    def test_vtu_is_readable_again(self):
        import meshio

        path = os.path.join(self.temp_dir.name, "mesh.vtu")
        get_writer("vtu")().write(self.mesh, path)
        read = meshio.read(path)
        self.assertEqual(len(read.points), len(self.mesh.points))
        self.assertIn("phase", read.cell_data)


if __name__ == "__main__":
    unittest.main()
