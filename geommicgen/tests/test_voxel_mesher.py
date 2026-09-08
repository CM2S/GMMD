import os
import tempfile
import unittest

import numpy as np

from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import available_meshers, get_mesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses import Disk, Ellipse, Ellipsoid, Sphere
from geommicgen.tests.helpers import build_microstructure
from geommicgen.translators.crate import CrateWriter

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
# Grids produced by RegularGridMeshGenerator, the mesher the voxel mesher replaced,
# while that class was still in the tree. They are what pins the replacement to what it
# replaced now that the original is gone, so they are never regenerated from the code
# under test: a fixture rewritten by the thing it checks agrees with anything.


def disks_case():
    """Three disks in two dimensions, the last straddling two faces of the RVE."""
    rve_dims = [1.0, 1.0]
    particles = []
    for i_center, i_radius in (
        ([0.25, 0.75], 0.15),
        ([0.6, 0.1], 0.08),
        ([0.02, 0.98], 0.12),
    ):
        particle = Disk("2", {"r": i_radius}, rve_dims)
        particle.position_center = np.array(i_center)
        particles.append(particle)
    # The last disk straddles two faces, so it is only stamped correctly if the
    # bounding box is wrapped back in

    return build_microstructure(rve_dims, Disk, particles), [40, 40]


def ellipses_case():
    """Two ellipses on an RVE and a grid that differ between the directions."""
    rve_dims = [2.0, 1.0]
    particles = []
    for i_center, i_angle in (([0.5, 0.5], 0.4), ([1.7, 0.2], 1.9)):
        particle = Ellipse(
            "2", {"major_axis": 0.4, "minor_axis": 0.15, "angle": i_angle}, rve_dims
        )
        particle.position_center = np.array(i_center)
        particles.append(particle)
    # Nothing square anywhere, which catches a swap of the axes

    return build_microstructure(rve_dims, Ellipse, particles), [48, 32]


def spheres_case():
    """Two spheres, the second sitting on a corner of the RVE."""
    rve_dims = [1.0, 1.0, 1.0]
    particles = []
    for i_center, i_radius in (([0.5, 0.5, 0.5], 0.2), ([0.05, 0.05, 0.95], 0.15)):
        particle = Sphere("2", {"r": i_radius}, rve_dims)
        particle.position_center = np.array(i_center)
        particles.append(particle)
    # The second sphere is stamped into all eight corners

    return build_microstructure(rve_dims, Sphere, particles), [16, 16, 16]


def ellipsoid_case():
    """One turned ellipsoid, on a grid of three different resolutions."""
    rve_dims = [1.0, 1.0, 1.0]
    particle = Ellipsoid(
        "2",
        {
            "axis_1": 0.4,
            "axis_2": 0.25,
            "axis_3": 0.15,
            "angle": 0.7,
            "rot_axis_comp_x": 0.0,
            "rot_axis_comp_y": 1.0,
            "rot_axis_comp_z": 1.0,
        },
        rve_dims,
    )
    particle.position_center = np.array([0.5, 0.5, 0.5])

    return build_microstructure(rve_dims, Ellipsoid, [particle]), [14, 18, 12]


def one_disk_case():
    """A single disk well inside the RVE."""
    rve_dims = [1.0, 1.0]
    particle = Disk("2", {"r": 0.2}, rve_dims)
    particle.position_center = np.array([0.4, 0.6])

    return build_microstructure(rve_dims, Disk, [particle]), [32, 32]


EQUIVALENCE_CASES = {
    "disks": disks_case,
    "ellipses": ellipses_case,
    "spheres": spheres_case,
    "ellipsoid": ellipsoid_case,
    "one_disk": one_disk_case,
}
# Every case a fixture was frozen for


def frozen_grid(name):
    """
    Load the grid the mesher that was replaced produced for one of the cases.

    Parameters
    ----------
    name: str
        Name of the case.

    Returns
    -------
    array
        The grid of phases.
    """
    return np.load(os.path.join(FIXTURE_DIR, "voxel_{0}.npy".format(name)))


class TestVoxelMesherEquivalence(unittest.TestCase):
    """
    Test class for the agreement with the mesher the voxel mesher replaced.

    RegularGridMeshGenerator has been removed, so the comparison is against the grids
    it produced while it was still there, frozen into fixtures.
    """

    def assert_matches_the_frozen_grid(self, name):
        """Check that the mesher gives every voxel the phase the old one gave it."""
        microstructure, n_voxels_dims = EQUIVALENCE_CASES[name]()
        mesh = VoxelMesher(n_voxels_dims).mesh(microstructure)
        np.testing.assert_array_equal(mesh.structured.phase_grid, frozen_grid(name))

        return mesh

    def test_disks_in_two_dimensions(self):
        self.assert_matches_the_frozen_grid("disks")

    def test_ellipses_on_an_anisotropic_grid(self):
        self.assert_matches_the_frozen_grid("ellipses")

    def test_spheres_in_three_dimensions(self):
        self.assert_matches_the_frozen_grid("spheres")

    def test_ellipsoid_in_three_dimensions(self):
        self.assert_matches_the_frozen_grid("ellipsoid")

    def test_every_case_has_a_fixture(self):
        for i_name in EQUIVALENCE_CASES:
            self.assertTrue(
                os.path.exists(
                    os.path.join(FIXTURE_DIR, "voxel_{0}.npy".format(i_name))
                ),
                i_name,
            )
        # A case whose fixture went missing would otherwise be a test that never runs

    def test_grid_reaches_the_crate_file_unchanged(self):
        microstructure, n_voxels_dims = one_disk_case()
        mesh = VoxelMesher(n_voxels_dims).mesh(microstructure)
        with tempfile.TemporaryDirectory() as temp_dir:
            written = CrateWriter().write(mesh, os.path.join(temp_dir, "grid.rgmsh"))
            self.assertEqual(len(written), 1)
            np.testing.assert_array_equal(np.load(written[0]), frozen_grid("one_disk"))
        # The whole path a spectral solver takes, from the microstructure to the array
        # it reads, without gmsh being involved anywhere


class TestVoxelMesherMesh(unittest.TestCase):
    """Test class for the mesh the voxel mesher produces."""

    def setUp(self):
        self.rve_dims = [1.0, 2.0]
        particle = Disk("2", {"r": 0.2}, self.rve_dims)
        particle.position_center = np.array([0.5, 1.0])
        self.microstructure = build_microstructure(self.rve_dims, Disk, [particle])
        self.mesh = VoxelMesher([10, 20]).mesh(self.microstructure)

    def test_is_a_structured_mesh(self):
        self.assertIsInstance(self.mesh, Mesh)
        self.assertIsNotNone(self.mesh.structured)
        self.assertEqual(self.mesh.structured.shape, (10, 20))
        np.testing.assert_allclose(self.mesh.structured.spacing, [0.1, 0.1])
        self.assertEqual(self.mesh.n_cells, 200)

    def test_carries_the_phases_of_the_microstructure(self):
        self.assertEqual(self.mesh.matrix_phase, "1")
        self.assertEqual(self.mesh.phase_names, {1: "1", 2: "2"})
        self.assertEqual(set(np.unique(self.mesh.structured.phase_grid)), {1, 2})

    def test_records_the_mesher_that_produced_it(self):
        self.assertEqual(self.mesh.source["mesher"], "voxel")
        self.assertEqual(self.mesh.source["n_voxels"], [10, 20])

    def test_cells_are_built_from_the_grid(self):
        self.assertEqual(len(self.mesh.cells), 1)
        cell_type, connectivity = self.mesh.cells[0]
        self.assertEqual(cell_type, "quad")
        self.assertEqual(connectivity.shape, (200, 4))
        self.assertEqual(len(self.mesh.points), 11 * 21)
        self.assertEqual(len(self.mesh.phase[0]), 200)

    def test_grid_is_periodic(self):
        self.assertTrue(self.mesh.periodic)
        self.mesh.check_periodic_conformity()
        # A grid always discretises opposite faces alike, so the check has to pass

    def test_phase_fraction_follows_the_volume_fraction(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.2}, rve_dims)
        particle.position_center = np.array([0.5, 0.5])
        microstructure = build_microstructure(rve_dims, Disk, [particle])
        mesh = VoxelMesher([200, 200]).mesh(microstructure)
        fraction = np.count_nonzero(mesh.structured.phase_grid == 2) / 200 ** 2
        self.assertAlmostEqual(fraction, np.pi * 0.2 ** 2, delta=1.0e-3)
        # The voxels only approximate the disk, so the agreement is limited by the
        # resolution of the grid rather than by the mesher

    def test_report_is_called_once_per_particle(self):
        reported = []
        VoxelMesher([8, 8]).mesh(
            self.microstructure, report=lambda index, total: reported.append(
                (index, total)
            )
        )
        self.assertEqual(reported, [(0, 1)])


class TestVoxelMesherErrors(unittest.TestCase):
    """Test class for the microstructures and the grids the voxel mesher refuses."""

    def test_grid_of_the_wrong_number_of_directions(self):
        rve_dims = [1.0, 1.0]
        microstructure = build_microstructure(rve_dims, Disk, [])
        with self.assertRaises(ValueError):
            VoxelMesher([8, 8, 8]).mesh(microstructure)

    def test_direction_without_voxels(self):
        with self.assertRaises(ValueError):
            VoxelMesher([8, 0])

    def test_microstructure_without_a_matrix_phase(self):
        microstructure = Microstructure([1.0, 1.0])
        microstructure.add_phase(Phase.from_type("2", Disk))
        with self.assertRaises(ValueError):
            VoxelMesher([8, 8]).mesh(microstructure)


class TestMesherRegistry(unittest.TestCase):
    """Test class for the registry of the meshers."""

    def test_voxel_mesher_is_registered(self):
        self.assertIs(get_mesher("voxel"), VoxelMesher)
        self.assertIn("voxel", available_meshers())

    def test_unknown_mesher(self):
        with self.assertRaises(ValueError):
            get_mesher("no_such_mesher")


if __name__ == "__main__":
    unittest.main()
