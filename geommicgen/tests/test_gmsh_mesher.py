import os
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.errors.error_classes import UnsupportedParticleShape
from geommicgen.meshing.images import periodic_images
from geommicgen.meshing.mesher import get_mesher
from geommicgen.microstructure.particleclasses import Disk, Point, Sphere
from geommicgen.tests.helpers import (
    build_microstructure,
    disk_microstructure,
    sphere_microstructure,
)

from geommicgen.meshing.gmsh_mesher import (
    GmshMesher,
    failing_surfaces,
    gmsh_session,
)


def cell_measures(mesh):
    """Give the area or the volume of the cells of a mesh, by phase."""
    points = mesh.points
    measures = {}
    for (_, i_connectivity), i_phase in zip(mesh.cells, mesh.phase):
        corners = [
            points[i_connectivity[:, i_node]] for i_node in range(mesh.dim + 1)
        ]
        if mesh.dim == 2:
            first, second, third = corners
            measure = 0.5 * np.abs(
                (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
                - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
            )
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

    def test_first_order_triangles(self):
        self.assert_matches_meshio(disk_microstructure(), "tri3", 0.1)

    def test_second_order_triangles(self):
        self.assert_matches_meshio(disk_microstructure(), "tri6", 0.12)

    def test_second_order_tetrahedra(self):
        self.assert_matches_meshio(sphere_microstructure(), "tetra10", 0.3)
        # The only element among these whose nodes gmsh lists in an order of its own


if __name__ == "__main__":
    unittest.main()
