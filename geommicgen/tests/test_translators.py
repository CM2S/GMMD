import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from geommicgen._optional import has_gmsh, has_package
from geommicgen.errors.error_classes import (
    MissingOptionalDependency,
    PeriodicityError,
)
from geommicgen.meshing.mesh import Mesh
from geommicgen.translators import (
    MeshioWriter,
    SolverWriter,
    available_writers,
    get_writer,
    writer_options,
)
from geommicgen.translators.base import (
    PLACEHOLDER_ELASTIC,
    PLACEHOLDER_STRAIN,
    WRITERS,
    register_writer,
)
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


def crate_block(file_path, keyword):
    """Read back the lines of a block of a CRATE input data file, up to a blank one."""
    with open(file_path, "r") as deck:
        lines = [i_line.rstrip("\n") for i_line in deck]

    start = [
        i_ind for i_ind, i_line in enumerate(lines)
        if not i_line.startswith("#") and i_line.split()[:1] == [keyword]
    ][0]
    block = []
    for i_line in lines[start + 1:]:
        if not i_line.strip():
            break
        block.append(i_line.split())

    return lines[start].split(), block


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


def abaqus_block(file_path, keyword):
    """Read back the data lines of the first block a keyword opens, split at commas."""
    with open(file_path, "r") as deck:
        lines = [i_line.strip() for i_line in deck]

    heads = [i_line.split(",")[0].strip().lower() for i_line in lines]
    start = heads.index(keyword.lower()) + 1
    rows = []
    for i_line in lines[start:]:
        if i_line.startswith("*"):
            break
        rows.append([i_entry.strip() for i_entry in i_line.split(",")])

    return rows


def abaqus_node_set(file_path, name):
    """Read back the node identifiers of one node set."""
    with open(file_path, "r") as deck:
        lines = [i_line.strip() for i_line in deck]

    start = lines.index("*Nset, nset={0}".format(name)) + 1
    identifiers = []
    for i_line in lines[start:]:
        if i_line.startswith("*"):
            break
        identifiers += [int(i_entry) for i_entry in i_line.split(",")]

    return identifiers


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

    def test_the_step_prescribes_the_placeholder_stretch(self):
        for i_shape, i_dims in (((2, 2), [2.0, 1.0]), ((2, 2, 2), [2.0, 1.0, 3.0])):
            with self.subTest(shape=i_shape):
                path = self.write(structured_mesh(i_shape, i_dims))
                prescribed = {
                    (i_row[0], int(i_row[1])): float(i_row[3])
                    for i_row in abaqus_block(path, "*Boundary")
                    if i_row[0].startswith("RP_")
                }
                dim = len(i_shape)
                expected = {
                    (REFERENCE_NODE_NAMES[i_axis], j_dof + 1): (
                        PLACEHOLDER_STRAIN * i_dims[0] if (i_axis, j_dof) == (0, 0)
                        else 0.0
                    )
                    for i_axis in range(dim)
                    for j_dof in range(dim)
                }
                self.assertEqual(prescribed, expected)
        # Every degree of freedom of every reference node is prescribed, which is the
        # whole macroscopic displacement gradient: a stretch along x, times the length
        # of the cell along the axis of the node

    def test_the_held_corner_is_never_eliminated(self):
        for i_shape, i_corner in (
            ((3, 3), "CORNER_XNEG_YNEG"),
            ((3, 3, 3), "CORNER_XNEG_YNEG_ZNEG"),
        ):
            with self.subTest(shape=i_shape):
                path = self.write(structured_mesh(i_shape, [1.0] * len(i_shape)))
                held = [
                    i_row[0] for i_row in abaqus_block(path, "*Boundary")
                    if not i_row[0].startswith("RP_")
                ]
                self.assertEqual(held, [i_corner])
                eliminated = {
                    i_terms[0][0] for i_terms in parse_abaqus_equations(path)
                }
                self.assertFalse(set(abaqus_node_set(path, i_corner)) & eliminated)
        # The translation the constraints leave free is removed by holding one node,
        # and Abaqus refuses a degree of freedom that is both prescribed and the one an
        # equation eliminates

    def test_the_reaction_forces_of_the_reference_nodes_are_printed(self):
        path = self.write(structured_mesh((2, 2, 2), [1.0, 1.0, 1.0]))
        self.assertEqual(
            self.keyword_values(path, "*Node Print", "nset"),
            list(REFERENCE_NODE_NAMES),
        )

    def test_without_the_constraints_there_is_no_step(self):
        path = self.write(
            structured_mesh((3, 3), [1.0, 1.0]), periodic_constraints=False
        )
        with open(path) as deck:
            text = deck.read()
        self.assertNotIn("*Step", text)
        self.assertIn("there is no step", text)
        # There is nothing to load the cell through, and the header says so rather
        # than promising a placeholder step the deck does not have


class TestWriterOptions(unittest.TestCase):
    """Test class for the options the writers declare."""

    def test_every_declared_option_is_described(self):
        for i_name, i_description in writer_options().items():
            self.assertIn("type", i_description, i_name)
            self.assertIn("help", i_description, i_name)
            self.assertIn(i_description["type"], ("int", "float", "str", "bool"))
        # The input file needs the type and the command line needs both, so a writer
        # that declares an option without them breaks whichever asks first

    def test_the_options_of_the_writers_that_have_them(self):
        self.assertEqual(
            sorted(writer_options()),
            ["Boundary_Type", "Gauss_Points", "Periodic_Constraints"],
        )

    def test_every_declared_option_is_read_when_it_is_given(self):
        probes = {
            "int": 7,
            "float": 0.5,
            "str": "geommicgen",
            "bool": False,
            "str_list": ["geommicgen"],
        }
        for i_name, i_writer in sorted(WRITERS.items()):
            for j_option, j_description in sorted(i_writer.options.items()):
                given = {j_option.lower(): probes[j_description["type"]]}
                try:
                    built = vars(i_writer.from_options(given))
                except ValueError:
                    continue
                self.assertNotEqual(built, vars(i_writer()), (i_name, j_option))
        # A writer declares its options under the name a deck spells them with and
        # reads them under the name Python spells them with, and nothing else holds
        # the two together: an option that is declared and never read would reach the
        # command line and the input file and then quietly do nothing. Refusing the
        # value counts as reading it

    def test_two_formats_may_not_describe_one_option_differently(self):
        class Conflicting(SolverWriter):
            """A writer that reads an option of another format as something else."""

            name = "conflicting"
            extension = ".conflicting"
            options = {"Gauss_Points": {"type": "str", "help": "not an integer"}}

            def _write(self, mesh, file_path):
                return [file_path]

        register_writer(Conflicting)
        self.addCleanup(WRITERS.pop, "conflicting", None)

        with self.assertRaises(ValueError) as context:
            writer_options()
        self.assertIn("Gauss_Points", str(context.exception))


class TestCrateWriter(unittest.TestCase):
    """Test class for the writer of the voxel grids."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_grid_is_written_unchanged(self):
        phase_grid = np.ones((5, 4), dtype=int)
        phase_grid[1:4, :2] = 2
        mesh = structured_mesh((5, 4), [1.0, 1.0], phase_grid)
        path = os.path.join(self.temp_dir.name, "grid.rgmsh")
        written = get_writer("crate")().write(mesh, path)
        restored = np.load(written[0])
        np.testing.assert_array_equal(restored, phase_grid)
        self.assertEqual(restored.dtype, phase_grid.dtype)
        # The array is the whole of what the solver reads of the microstructure and must
        # not change; it is not square, so that one written transposed would show

    def test_does_not_need_the_cells(self):
        self.assertFalse(get_writer("crate").needs_cells)
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        get_writer("crate")().write(mesh, os.path.join(self.temp_dir.name, "g.rgmsh"))
        self.assertIsNone(mesh._cells)
        # Writing the grid must not build the cells, which is what makes it usable for
        # resolutions that could never be expressed as elements

    def write_example(self, shape, rve_dims, phase_grid=None):
        """Write a grid and its example input file, and give back both paths."""
        mesh = structured_mesh(shape, rve_dims, phase_grid)
        written = get_writer("crate")().write(
            mesh, os.path.join(self.temp_dir.name, "m.rgmsh.npy")
        )
        self.assertEqual(len(written), 2)

        return written

    def test_the_example_runs_off_the_grid(self):
        grid_path, example_path = self.write_example((4, 4), [2.0, 1.0])
        self.assertEqual(os.path.basename(example_path), "m_example.dat")
        self.assertEqual(crate_block(example_path, "Discretization_File")[1],
                         [["m.rgmsh.npy"]])
        self.assertEqual(crate_block(example_path, "RVE_Dimensions")[1],
                         [["2.0", "1.0"]])
        self.assertEqual(
            crate_block(example_path, "Problem_Type")[0], ["Problem_Type", "1"]
        )
        # The grid is named as it is written, beside the file, and the dimensions it
        # carries none of are declared here, where they would be copied by hand

    def test_a_three_dimensional_grid_is_a_three_dimensional_problem(self):
        _, example_path = self.write_example((2, 2, 2), [1.0, 2.0, 3.0])
        self.assertEqual(
            crate_block(example_path, "Problem_Type")[0], ["Problem_Type", "4"]
        )
        self.assertEqual(crate_block(example_path, "RVE_Dimensions")[1],
                         [["1.0", "2.0", "3.0"]])

    def test_every_phase_of_the_grid_has_a_material_and_a_cluster(self):
        phase_grid = np.ones((4, 4), dtype=int)
        phase_grid[:2] = 3
        _, example_path = self.write_example((4, 4), [1.0, 1.0], phase_grid)

        head, materials = crate_block(example_path, "Material_Phases")
        self.assertEqual(head, ["Material_Phases", "2"])
        self.assertEqual([i_row[0] for i_row in materials if i_row[1:2] == ["elastic"]],
                         ["1", "3"])
        self.assertEqual(
            [i_row[1] for i_row in materials if i_row[0] == "E"],
            [str(PLACEHOLDER_ELASTIC[0])] * 2,
        )
        self.assertEqual(crate_block(example_path, "Number_of_Clusters")[1],
                         [["1", "1"], ["3", "1"]])
        # The phases are the ones the grid holds, by the identifiers it holds them by,
        # which is what CRATE checks the grid against; one cluster each, because a cell
        # whose phases are all the placeholder material offers k-means no more

    def test_the_strain_is_listed_a_column_at_a_time(self):
        _, example_path = self.write_example((2, 2, 2), [1.0, 1.0, 1.0])
        _, strain = crate_block(example_path, "Macroscale_Strain")
        self.assertEqual(
            [i_row[0] for i_row in strain],
            ["eps_11", "eps_21", "eps_31", "eps_12", "eps_22", "eps_32",
             "eps_13", "eps_23", "eps_33"],
        )
        self.assertEqual(
            [float(i_row[1]) for i_row in strain],
            [PLACEHOLDER_STRAIN] + [0.0] * 8,
        )

    def test_no_comment_sits_inside_a_block(self):
        _, example_path = self.write_example((4, 4), [1.0, 1.0])
        with open(example_path) as deck:
            lines = deck.read().splitlines()
        keywords = (
            "Material_Phases", "Macroscale_Loading", "Number_of_Clusters",
            "Strain_Formulation",
        )
        for i_ind, i_line in enumerate(lines[:-1]):
            if i_line.startswith("#"):
                following = lines[i_ind + 1]
                self.assertTrue(
                    following.startswith("#") or not following.strip()
                    or following.split()[0] in keywords,
                    following,
                )
        # CRATE reads the lines of a block by their position after the keyword, so a
        # comment inside one would be read as data

    def test_the_example_can_be_left_out(self):
        mesh = structured_mesh((4, 4), [1.0, 1.0])
        written = get_writer("crate")(write_example=False).write(
            mesh, os.path.join(self.temp_dir.name, "m.rgmsh.npy")
        )
        self.assertEqual([os.path.basename(i_path) for i_path in written],
                         ["m.rgmsh.npy"])

    def test_refuses_an_unstructured_mesh(self):
        mesh = structured_mesh((3, 3), [1.0, 1.0])
        unstructured = Mesh(
            mesh.rve_dims, points=mesh.points, cells=mesh.cells, phase=mesh.phase
        )
        with self.assertRaises(ValueError):
            get_writer("crate")().write(
                unstructured, os.path.join(self.temp_dir.name, "g.rgmsh")
            )


class TestWriterContract(unittest.TestCase):
    """Test class holding every registered writer to what a job needs of it."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        phase_grid = np.ones((3, 3), dtype=int)
        phase_grid[1, 1] = 2
        self.mesh = structured_mesh((3, 3), [1.0, 1.0], phase_grid)
        # A grid, so that the writers of a grid and the writers of cells both have
        # what they need; the cells are built on demand

    def test_every_writer_declares_itself(self):
        for i_name in available_writers():
            with self.subTest(writer=i_name):
                writer_class = get_writer(i_name)
                self.assertEqual(writer_class.name, i_name)
                self.assertTrue(writer_class.extension.startswith("."))
                self.assertIsInstance(writer_class.options, dict)
                for j_name, j_description in writer_class.options.items():
                    self.assertIn("type", j_description, j_name)
                    self.assertIn("help", j_description, j_name)

    def test_every_writer_writes_what_it_says_it_wrote(self):
        for i_name in available_writers():
            with self.subTest(writer=i_name):
                writer = get_writer(i_name).from_options({})
                package = getattr(writer, "requires_package", None)
                if package is not None and not has_package(package):
                    continue
                base_path = os.path.join(self.temp_dir.name, "mesh_" + i_name)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    written = writer.write(self.mesh, base_path + writer.extension)
                self.assertEqual(printed.getvalue(), "")
                self.assertIsInstance(written, list)
                self.assertTrue(written)
                for j_path in written:
                    self.assertTrue(os.path.isfile(j_path), j_path)
                    self.assertGreater(os.path.getsize(j_path), 0, j_path)
                self.assertTrue(
                    any(j_path.startswith(base_path) for j_path in written), written
                )
        # The requirements listed under "Adding a writer" in the base module: a file
        # under the name given, every file written reported, nothing printed. The
        # formats needing a library that is not installed are left out


class TestReadBackByMeshio(unittest.TestCase):
    """Test class holding the files written to what a reader of the format finds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def read_back(self, mesh, format_name):
        """Write a mesh in a format and read it with meshio."""
        import meshio

        path = os.path.join(self.temp_dir.name, "m" + get_writer(format_name).extension)
        get_writer(format_name)().write(mesh, path)

        return meshio.read(path)

    def assert_same_mesh(self, mesh, back):
        """Hold the nodes and the connectivity read back to the mesh written."""
        n_points = len(mesh.points)
        np.testing.assert_allclose(
            back.points[:n_points, : mesh.dim], mesh.points[:, : mesh.dim]
        )
        ours = np.concatenate([i_block[1] for i_block in mesh.cells])
        theirs = np.concatenate([i_block.data for i_block in back.cells])
        np.testing.assert_array_equal(
            ours[np.lexsort(ours.T)], theirs[np.lexsort(theirs.T)]
        )
        # The elements come back grouped by phase, so they are compared as a set of
        # rows: what has to survive is which nodes each element joins, in the order the
        # element type numbers them

    def test_abaqus_in_three_dimensions(self):
        mesh = structured_mesh(
            (2, 2, 2), [1.0, 1.0, 1.0], phase_grid=np.array([[[1, 2], [2, 1]]] * 2)
        )
        back = self.read_back(mesh, "abaqus")

        self.assert_same_mesh(mesh, back)
        self.assertEqual(len(back.points), len(mesh.points) + 3)
        self.assertEqual(sorted(back.cell_sets), ["PHASE_1", "PHASE_2"])
        # The three nodes beyond the mesh are the reference nodes, and the phases are
        # the element sets. A two dimensional deck is not read back here: meshio knows
        # no plane strain element but the six node triangle

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_abaqus_second_order_tetrahedra(self):
        from geommicgen.meshing.gmsh_mesher import GmshMesher
        from geommicgen.tests.helpers import sphere_microstructure

        mesh = GmshMesher(mesh_size=0.3, element_type="tetra10").mesh(
            sphere_microstructure()
        )
        self.assert_same_mesh(mesh, self.read_back(mesh, "abaqus"))
        # The order of the nodes of a second order element is where two formats most
        # often disagree, and meshio reads it with the Abaqus convention

    @unittest.skipUnless(has_package("h5py"), "h5py is not installed")
    def test_xdmf(self):
        mesh = structured_mesh(
            (2, 2), [1.0, 1.0], phase_grid=np.array([[1, 2], [2, 1]])
        )
        back = self.read_back(mesh, "xdmf")

        self.assert_same_mesh(mesh, back)
        np.testing.assert_array_equal(
            np.concatenate(back.cell_data["phase"]), np.concatenate(mesh.phase)
        )
        # The phase travels as data on the cells, which is what FEniCS reads its cell
        # tags from


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

    def test_solver_formats_nobody_has_run_are_not_offered(self):
        writers = available_writers()
        for i_format, i_solver in (("med", "Code_Aster"), ("exodus", "MOOSE")):
            self.assertNotIn(i_format, writers)
            with self.assertRaises(ValueError) as context:
                get_writer(i_format)
            self.assertIn(i_solver, str(context.exception))
        # meshio writes both with the phases, but a format a solver reads is offered
        # once the solver has run what is written of it, and asking for one of these
        # says which solver that would be rather than that the format is unknown

    def test_abaqus_is_not_the_meshio_writer(self):
        self.assertNotIsInstance(get_writer("abaqus")(), MeshioWriter)
        # The format is offered, but by the writer of this package: meshio names two
        # dimensional triangles after a rigid element and cannot write a constraint

    def test_formats_needing_an_extra_package_say_so(self):
        for i_format in ("xdmf",):
            writer = get_writer(i_format)
            self.assertIsNotNone(writer.requires_package)
            path = os.path.join(self.temp_dir.name, "m" + writer.extension)
            with patch.dict(sys.modules, {writer.requires_package: None}):
                with self.assertRaises(MissingOptionalDependency) as context:
                    writer().write(self.mesh, path)
            message = str(context.exception)
            self.assertIn(writer.requires_package, message)
            self.assertIn("geommicgen[{0}]".format(i_format), message)
        # meshio raises a bare import error from inside itself for these formats, which
        # says nothing about how to fix it. Setting the entry to None makes the import
        # raise whether or not the package is installed, so this runs everywhere, and
        # the extra is named after the format so the command can be read off the message

    def test_vtu_is_readable_again(self):
        import meshio

        path = os.path.join(self.temp_dir.name, "mesh.vtu")
        get_writer("vtu")().write(self.mesh, path)
        read = meshio.read(path)
        self.assertEqual(len(read.points), len(self.mesh.points))
        self.assertEqual(read.points.shape[1], 3)
        self.assertIn("phase", read.cell_data)

    @unittest.skipUnless(has_package("h5py"), "h5py not installed")
    def test_xdmf_is_written_the_way_fenics_reads_it(self):
        import meshio

        path = os.path.join(self.temp_dir.name, "mesh.xdmf")
        get_writer("xdmf")().write(self.mesh, path)
        read = meshio.read(path)
        with open(path) as xdmf:
            self.assertIn('GeometryType="XY"', xdmf.read())
        self.assertEqual(read.points.shape, (len(self.mesh.points), 2))
        self.assertEqual(read.cell_data["phase"][0].dtype, np.int32)
        np.testing.assert_array_equal(read.cell_data["phase"][0], self.mesh.phase[0])
        # FEniCS takes the number of coordinates as the dimension of the problem, so a
        # two dimensional mesh written with three would become a surface in space, and
        # it reads its cell tags as 32 bit integers

    def test_the_vtk_family_keeps_three_coordinates(self):
        for i_format in ("vtu", "vtk", "gmsh"):
            self.assertFalse(get_writer(i_format).flat_points)
        self.assertTrue(get_writer("xdmf").flat_points)


if __name__ == "__main__":
    unittest.main()
