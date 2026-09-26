import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.errors.error_classes import UnsupportedParticleShape
from geommicgen.meshing.images import periodic_images
from geommicgen.meshing.mesher import get_mesher
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses import Disk, Point, Sphere
from geommicgen.tests.helpers import (
    build_microstructure,
    disk_microstructure,
    ellipse_microstructure,
    sphere_microstructure,
)

from geommicgen.meshing.gmsh_mesher import (
    GmshMesher,
    check_fills_rve,
    failing_surfaces,
    gmsh_session,
)
from geommicgen.microstructure.particleclasses import CylindricalFiber, Ellipse


def triangle_areas(first, second, third):
    """Give the areas of the triangles with these corners."""
    return 0.5 * np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
    )


def cell_measures(mesh):
    """Give the area or the volume of the cells of a mesh, by phase."""
    points = mesh.points
    measures = {}
    for (i_type, i_connectivity), i_phase in zip(mesh.cells, mesh.phase):
        corners = [
            points[i_connectivity[:, i_node]] for i_node in range(mesh.dim + 1)
        ]
        if mesh.dim == 2:
            first, second, third = corners
            measure = triangle_areas(first, second, third)
            if i_type.startswith("quad"):
                measure = measure + triangle_areas(
                    first, third, points[i_connectivity[:, 3]]
                )
            # A quad is its two triangles; the corners come first in the connectivity
            # of a second order element too
        else:
            first, second, third, fourth = corners
            measure = (
                np.abs(
                    np.einsum(
                        "ij,ij->i",
                        np.cross(second - first, third - first),
                        fourth - first,
                    )
                )
                / 6.0
            )
        for i_name in np.unique(i_phase):
            measures[int(i_name)] = measures.get(int(i_name), 0.0) + measure[
                i_phase == i_name
            ].sum()

    return measures


def cell_signature(points, cells):
    """Give every cell as the tuple of the coordinates of its nodes, in order."""
    rows = []
    for i_type, i_connectivity in cells:
        for i_corners in points[i_connectivity]:
            rows.append((i_type,) + tuple(np.round(i_corners.ravel(), 9)))

    return sorted(rows)


class TestPeriodicImages(unittest.TestCase):
    """Test class for the periodic images of a particle."""

    def test_particle_well_inside_has_no_images(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.5, 0.5])
        self.assertEqual(len(list(periodic_images(particle, rve_dims))), 1)

    def test_particle_crossing_one_face_has_one_image(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.02, 0.5])
        centers = list(periodic_images(particle, rve_dims))
        self.assertEqual(len(centers), 2)
        self.assertEqual(
            sorted(round(i_center[0], 6) for i_center in centers), [0.02, 1.02]
        )
        # What pokes out of the left face comes back in at the right one

    def test_particle_on_a_corner_has_three_images(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.02, 0.02])
        self.assertEqual(len(list(periodic_images(particle, rve_dims))), 4)

    def test_sphere_on_a_corner_has_seven_images(self):
        rve_dims = [1.0, 1.0, 1.0]
        particle = Sphere("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.02, 0.02, 0.02])
        self.assertEqual(len(list(periodic_images(particle, rve_dims))), 8)

    def test_an_ellipse_turned_across_a_face_keeps_its_image(self):
        rve_dims = [1.0, 1.0]
        particle = Ellipse(
            "2", {"major_axis": 0.4, "minor_axis": 0.2, "angle": np.pi / 2}, rve_dims
        )
        particle.position_center = np.array([0.5, 0.85])
        centers = list(periodic_images(particle, rve_dims))
        self.assertIn((0.5, 0.85 - 1.0, 0.0), centers)
        # Its long axis is along y and reaches past the top face by 0.05. The image
        # below the bottom face was discarded, its reach along y taken as the semi
        # minor axis, 0.1, so the part that pokes out at the top did not come back in
        # at the bottom

    def test_images_can_be_turned_off(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.02, 0.02])
        self.assertEqual(len(list(periodic_images(particle, rve_dims, False))), 1)
        # A view of the microstructure wants the particle alone


class TestGmshMesherConfiguration(unittest.TestCase):
    """Test class for the settings of the mesher, which need no gmsh."""

    def test_unknown_element(self):
        with self.assertRaises(ValueError):
            GmshMesher(mesh_size=0.1, element_type="pentagon7")

    def test_no_size_at_all(self):
        with self.assertRaises(ValueError):
            GmshMesher()

    def test_negative_size(self):
        with self.assertRaises(ValueError):
            GmshMesher(mesh_size=-1.0)

    def test_element_of_the_wrong_dimension(self):
        with self.assertRaises(ValueError):
            GmshMesher(mesh_size=0.1, element_type="tetra4").mesh(disk_microstructure())

    def test_elements_per_particle_sets_the_size(self):
        mesher = GmshMesher(elements_per_particle=6, element_type="tri3")
        mesher.resolve_mesh_size(disk_microstructure())
        self.assertAlmostEqual(mesher.mesh_size, 2 * 0.12 / 6)
        self.assertTrue(any("elements across" in i for i in mesher.warnings))
        # The size follows the smallest particle, 0.12 in radius here

    def test_inert_size_is_flagged(self):
        mesher = GmshMesher(mesh_size=0.5, element_type="tri3")
        mesher.resolve_mesh_size(disk_microstructure())
        self.assertTrue(any("no longer controls" in i for i in mesher.warnings))

    def test_unsupported_shape_is_refused(self):
        particle = Point(2, "2")
        particle.position_center = np.array([0.5, 0.5])
        with self.assertRaises(UnsupportedParticleShape):
            GmshMesher.add_primitive(Mock(), Mock(), particle, (0.5, 0.5, 0.0))
        # A shape with no geometry would otherwise be left out of the mesh in silence

    def test_failing_surfaces_are_read_from_the_message(self):
        message = "Invalid boundary mesh (overlapping facets) on surface 75 surface 76"
        self.assertEqual(failing_surfaces(Exception(message)), [75, 76])

    def test_registered_under_its_name(self):
        self.assertIs(get_mesher("gmsh"), GmshMesher)

    def test_built_from_the_options_it_declares(self):
        (mesher,) = GmshMesher.from_options(
            {"mesh_size": 0.2, "element_type": "tetra4", "elements_per_particle": None}
        )

        self.assertEqual(mesher.mesh_size, 0.2)
        self.assertEqual(mesher.element_type, "tetra4")
        self.assertIsNone(mesher.elements_per_particle)
        self.assertEqual(mesher.label, "tetra4")
        # The base class passes each declared option to the initializer by name, and
        # leaves the ones that were not given to its defaults

    def test_an_option_left_out_takes_the_default(self):
        (mesher,) = GmshMesher.from_options({"mesh_size": 0.2})

        self.assertEqual(mesher.element_type, "tri3")

    def test_the_initializer_still_checks_what_it_is_given(self):
        with self.assertRaises(ValueError):
            GmshMesher.from_options({"element_type": "tri3"})
        # Neither size given: refused by the initializer, as it is when called directly


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestGmshMesherMeshes(unittest.TestCase):
    """Test class for the meshes gmsh produces."""

    def test_the_whole_rve_is_meshed_in_two_dimensions(self):
        mesh = GmshMesher(mesh_size=0.08, element_type="tri3").mesh(
            disk_microstructure()
        )
        measures = cell_measures(mesh)
        self.assertAlmostEqual(sum(measures.values()), 1.0, places=9)
        self.assertGreater(measures[1], 0.8)
        self.assertGreater(measures[2], 0.0)
        # The matrix is by far the larger phase, and a mesh that leaves it out still
        # looks like a valid mesh of the particles

    def test_the_whole_rve_is_meshed_in_three_dimensions(self):
        mesh = GmshMesher(mesh_size=0.25, element_type="tetra4").mesh(
            sphere_microstructure()
        )
        measures = cell_measures(mesh)
        self.assertAlmostEqual(sum(measures.values()), 1.0, places=9)
        self.assertGreater(measures[1], 0.9)
        self.assertGreater(measures[2], 0.0)

    def test_no_node_is_left_unused(self):
        mesh = GmshMesher(mesh_size=0.08, element_type="tri3").mesh(
            disk_microstructure()
        )
        used = np.unique(
            np.concatenate([i_connectivity.ravel() for _, i_connectivity in mesh.cells])
        )
        self.assertEqual(len(used), len(mesh.points))
        # A node no element uses is rejected by the solvers the mesh is written for

    def test_opposite_faces_are_discretised_alike(self):
        mesh = GmshMesher(mesh_size=0.08, element_type="tri3").mesh(
            disk_microstructure()
        )
        self.assertTrue(mesh.periodic)
        mesh.check_periodic_conformity()

    def test_a_recombination_that_leaves_triangles_is_reported(self):
        mesher = GmshMesher(mesh_size=0.1, element_type="quad4")
        mesh = mesher.mesh(disk_microstructure())
        self.assertEqual({i_type for i_type, _ in mesh.cells}, {"quad", "triangle"})
        self.assertTrue(any("quad4 was asked for" in i for i in mesher.warnings))
        self.assertTrue(any("2 triangle" in i for i in mesher.warnings))
        # Blossom recombines what it can pair and keeps the rest as triangles, and at
        # this size two are left; a size of 0.05 gives a mesh of quads alone

    def test_a_mesh_of_one_element_type_is_not_reported(self):
        mesher = GmshMesher(mesh_size=0.05, element_type="quad4")
        mesh = mesher.mesh(disk_microstructure())
        self.assertEqual([i_type for i_type, _ in mesh.cells], ["quad"])
        self.assertFalse(any("was asked for" in i for i in mesher.warnings))

    def test_first_order_quads_mesh_ellipses(self):
        mesher = GmshMesher(mesh_size=0.1, element_type="quad4")
        mesh = mesher.mesh(ellipse_microstructure())
        measures = cell_measures(mesh)
        self.assertAlmostEqual(sum(measures.values()), 1.0, places=9)
        self.assertFalse(any("optimizing" in i for i in mesher.warnings))
        # The high order optimizer was run on every mesh, and on first order quads
        # around an ellipse it raised over the element quality; it is now run on the
        # second order meshes it exists for. The same mesh with tri3 or quad8 was fine

    def test_the_mesh_records_how_it_was_made(self):
        mesh = GmshMesher(mesh_size=0.08, element_type="tri3").mesh(
            disk_microstructure()
        )
        self.assertEqual(mesh.source["mesher"], "gmsh")
        self.assertEqual(mesh.source["element_type"], "tri3")
        self.assertEqual(mesh.phase_names, {1: "1", 2: "2"})
        self.assertEqual(mesh.matrix_phase, "1")


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestGmshExtraction(unittest.TestCase):
    """
    Test class for the mesh read out of gmsh in memory.

    Gmsh is not deterministic from one run to the next, so the comparison is made
    against meshio reading the file written by the very session the mesh was read from.
    Meshio knows the order VTK expects, so agreeing with it is what establishes that
    the nodes of an element come out in the right order.
    """

    def assert_matches_meshio(self, microstructure, element_type, mesh_size):
        """Extract a mesh in memory and through a file, and compare the two."""
        import meshio

        mesher = GmshMesher(mesh_size=mesh_size, element_type=element_type)
        with gmsh_session() as gmsh:
            phase_groups = mesher.build_model(gmsh, microstructure)
            in_memory = mesher.extract_mesh(gmsh, microstructure, phase_groups)
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, "mesh.msh")
                gmsh.write(file_path)
                through_file = meshio.read(file_path)
        cell_type = in_memory.cells[0][0]
        from_file = [
            (i_block.type, i_block.data)
            for i_block in through_file.cells
            if i_block.type == cell_type
        ]
        self.assertEqual(
            cell_signature(in_memory.points, in_memory.cells),
            cell_signature(through_file.points, from_file),
        )

    def test_a_mesh_that_does_not_fill_the_rve_is_not_taken(self):
        microstructure = disk_microstructure()
        mesher = GmshMesher(mesh_size=0.1, element_type="tri3")
        with gmsh_session() as gmsh:
            phase_groups = mesher.build_model(gmsh, microstructure)
            with self.assertRaisesRegex(ValueError, "part of the geometry was lost"):
                mesher.extract_mesh(gmsh, microstructure.scaled(2.0), phase_groups)
        # Read against an RVE twice as long in each direction, which it covers a
        # quarter of, as a mesh gmsh lost a particle of covers less than its RVE

    def test_first_order_triangles(self):
        self.assert_matches_meshio(disk_microstructure(), "tri3", 0.1)

    def test_second_order_triangles(self):
        self.assert_matches_meshio(disk_microstructure(), "tri6", 0.12)

    def test_second_order_tetrahedra(self):
        self.assert_matches_meshio(sphere_microstructure(), "tetra10", 0.3)
        # The only element among these whose nodes gmsh lists in an order of its own


class TestFillsTheRVE(unittest.TestCase):
    """Test class for the refusal of a mesh that does not fill its RVE."""

    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]

    def test_a_mesh_that_fills_the_rve_is_taken(self):
        check_fills_rve(self.points, self.cells, [1.0, 1.0])
        check_fills_rve(self.points, [("quad", np.array([[0, 1, 2, 3]]))], [1.0, 1.0])

    def test_a_mesh_missing_a_cell_is_refused(self):
        with self.assertRaisesRegex(
            ValueError, "cover 0.5 of an RVE of 1, off by a part in 2"
        ):
            check_fills_rve(
                self.points, [("triangle", np.array([[0, 1, 2]]))], [1.0, 1.0]
            )

    def test_what_opencascade_merges_is_not_a_loss(self):
        points = self.points.copy()
        points[1, 0] += 1e-8
        points[3, 0] -= 1e-8
        check_fills_rve(points, self.cells, [1.0, 1.0])
        # Two corners moved out of the RVE by 1e-8, which changes the area covered by
        # as much: what OpenCASCADE, whose tolerance is 1e-7 of the model, does to a
        # particle within that of a face. A tolerance of 1e-9 refused such meshes

    def test_a_mesh_reaching_outside_the_rve_is_refused(self):
        points = self.points.copy()
        points[:, 0] -= 0.5
        with self.assertRaisesRegex(ValueError, "reach outside it"):
            check_fills_rve(points, self.cells, [1.0, 1.0])
        # It covers as much as the RVE does, shifted half a cell out of it

    def test_the_volume_of_tetrahedra_is_taken(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
        )
        with self.assertRaisesRegex(ValueError, "cover 1 of an RVE of 6"):
            check_fills_rve(points, [("tetra10", np.array([[0, 1, 2, 3]]))], [2, 1, 3])


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestPhaseVolumes(unittest.TestCase):
    """Test class for the report of a phase that takes up more or less than its own."""

    def test_a_piece_lost_to_the_matrix_is_reported(self):
        import gmsh

        fragment = gmsh.model.occ.fragment

        def losing_a_piece(*args, **kwargs):
            out_dim_tags, fragment_map = fragment(*args, **kwargs)
            return out_dim_tags, [fragment_map[0], []] + fragment_map[2:]

        # The first piece of a particle is left out of the map, so what it fragmented
        # into falls to the matrix, as a piece the booleans lose does

        mesher = GmshMesher(mesh_size=0.1, element_type="tri3")
        mesher.mesh(disk_microstructure())
        self.assertFalse(any("takes up" in i for i in mesher.warnings))
        with patch.object(gmsh.model.occ, "fragment", losing_a_piece):
            mesh = mesher.mesh(disk_microstructure())
        self.assertTrue(any("phase 2 takes up" in i for i in mesher.warnings))
        self.assertAlmostEqual(sum(cell_measures(mesh).values()), 1.0, places=9)
        # The cells still fill the RVE, so the check of the mesh passes, and the loss
        # was written without a word

    def test_a_phase_inside_another_is_not_reported(self):
        rve_dims = [1.0, 1.0]
        microstructure = Microstructure(rve_dims)
        microstructure.add_phase(Phase("1", {"phase_type": 1}))
        microstructure.add_phase(Phase.from_type("2", Disk))
        microstructure.add_phase(
            Phase.from_type("3", Disk, inner_phase=True, outer_phase="2")
        )
        outer = Disk("2", {"r": 0.2}, rve_dims)
        outer.position_center = np.array([0.5, 0.5])
        microstructure.phases["2"].particles.append(outer)
        for i_center in ([0.42, 0.5], [0.58, 0.52]):
            inner = Disk("3", {"r": 0.05}, rve_dims)
            inner.position_center = np.array(i_center)
            inner.parent = outer
            microstructure.phases["3"].particles.append(inner)

        mesher = GmshMesher(mesh_size=0.05, element_type="tri3")
        measures = cell_measures(mesher.mesh(microstructure))
        self.assertFalse(any("takes up" in i for i in mesher.warnings))
        self.assertGreater(measures[3], 0.0)
        # The inner disks take their area from the outer one, whose phase is measured
        # against it less theirs


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestGmshScaleInvariance(unittest.TestCase):
    """Test class for a mesh that is the same in any units of length."""

    def record(self, mesh, scale):
        """Give the nodes, the cells and the phases of a mesh, its lengths divided."""
        return (
            (mesh.points / scale).tolist(),
            [
                (i_type, i_connectivity.tolist())
                for i_type, i_connectivity in mesh.cells
            ],
            [i_phase.tolist() for i_phase in mesh.phase],
            [i_dim / scale for i_dim in mesh.rve_dims],
            mesh.source["mesh_size"] / scale,
        )

    def assert_scale_free(self, microstructure, element_type, mesh_size):
        """Check that a mesh is the same at scales a power of two apart."""
        reference = self.record(
            GmshMesher(mesh_size=mesh_size, element_type=element_type).mesh(
                microstructure
            ),
            1.0,
        )
        for i_scale in (2.0**-20, 2.0**20):
            with self.subTest(element_type, scale=i_scale):
                mesh = GmshMesher(
                    mesh_size=mesh_size * i_scale, element_type=element_type
                ).mesh(microstructure.scaled(i_scale))
                self.assertEqual(self.record(mesh, i_scale), reference)
        # Powers of two are multiplied out exactly, so a mesher that builds its model
        # at unit scale gives the same mesh to the last bit. Built in the user's units,
        # at a millionth of the unit the faces were paired with themselves and the
        # booleans lost the particles, and at a million the matrix was left unmeshed

    def test_first_order_meshes(self):
        self.assert_scale_free(disk_microstructure(), "tri3", 0.05)
        self.assert_scale_free(ellipse_microstructure(), "quad4", 0.05)
        self.assert_scale_free(sphere_microstructure(), "tetra4", 0.1)

    def test_second_order_meshes_in_the_plane(self):
        self.assert_scale_free(disk_microstructure(), "tri6", 0.05)
        self.assert_scale_free(disk_microstructure(), "quad8", 0.05)

    def test_fibres(self):
        rve_dims = [1.0, 1.0, 1.0]
        particles = []
        for i_center in ([0.4, 0.4], [0.95, 0.1]):
            particle = CylindricalFiber("2", {"r": 0.12, "direction": 2}, rve_dims)
            particle.position_center = np.array(i_center)
            particles.append(particle)
        self.assert_scale_free(
            build_microstructure(rve_dims, CylindricalFiber, particles), "tetra4", 0.1
        )
        # The fibre is extruded by its length along the fibres, which a rescale left in
        # the user's units

    def test_second_order_tetrahedra(self):
        reference = GmshMesher(mesh_size=0.1, element_type="tetra10").mesh(
            sphere_microstructure()
        )
        scale = 2.0**-20
        mesh = GmshMesher(mesh_size=0.1 * scale, element_type="tetra10").mesh(
            sphere_microstructure().scaled(scale)
        )
        np.testing.assert_array_equal(mesh.cells[0][1], reference.cells[0][1])
        np.testing.assert_allclose(mesh.points / scale, reference.points, atol=1e-7)
        # The optimizer that places their mid-side nodes moves some of them by up to
        # 1e-9 from one run to the next, at any scale and on one thread, so the nodes
        # are compared to a tolerance; the cells are the same

    def test_the_same_microstructure_gives_the_same_mesh(self):
        first, second = (
            GmshMesher(mesh_size=0.1, element_type="tetra4").mesh(
                sphere_microstructure()
            )
            for _ in range(2)
        )
        self.assertEqual(self.record(first, 1.0), self.record(second, 1.0))
        # Gmsh meshed on four threads, and gave meshes different in their last digits
        # and in the order of their cells from one run to the next

    def test_the_element_size_is_reported_in_the_users_units(self):
        scale = 2.0**-20
        mesher = GmshMesher(elements_per_particle=6, element_type="tri3")
        mesh = mesher.mesh(disk_microstructure().scaled(scale))
        self.assertEqual(mesh.source["mesh_size"], 2 * 0.12 / 6 * scale)
        self.assertIn("{0:.4g}".format(2 * 0.12 * scale), mesher.warnings[0])
        # The size is derived from the smallest particle and reported before the model
        # is built at unit scale


if __name__ == "__main__":
    unittest.main()
