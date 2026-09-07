import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import available_meshers, get_mesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses import Disk, Ellipse, Ellipsoid, Sphere
from geommicgen.postproc.mshgen.meshing_interface import RegularGridMeshGenerator
from geommicgen.tests.helpers import build_microstructure
from geommicgen.translators.crate import CrateWriter


def legacy_grid(microstructure, n_voxels_dims):
    """
    Build the grid of a microstructure with the mesher that is being replaced.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be meshed.

    n_voxels_dims: list
        Number of voxels in each spatial direction.

    Returns
    -------
    array
        The grid of phases, read back from the file the old mesher writes.
    """
    generator = RegularGridMeshGenerator(n_voxels_dims, microstructure.rve_dims)
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("geommicgen.iofuncs.printing.print_to_file"):
            with contextlib.redirect_stdout(io.StringIO()):
                generator.generate_mesh(microstructure, temp_dir)
        name = "_".join(str(int(i_size)) for i_size in n_voxels_dims)

        return np.load(os.path.join(temp_dir, "meshes", name + ".rgmsh.npy"))
    # The old mesher only ever exposed its grid through the file it wrote, which is why
    # this goes through a temporary directory


class TestVoxelMesherEquivalence(unittest.TestCase):
    """
    Test class for the agreement with the mesher the voxel mesher replaces.

    The old mesher is still in the tree, so the comparison is made against the code
    itself rather than against a stored array. It has to be frozen into a fixture when
    `.RegularGridMeshGenerator` is removed.
    """

    def assert_same_grid(self, microstructure, n_voxels_dims):
        """Check that both meshers give every voxel the same phase."""
        mesh = VoxelMesher(n_voxels_dims).mesh(microstructure)
        np.testing.assert_array_equal(
            mesh.structured.phase_grid, legacy_grid(microstructure, n_voxels_dims)
        )

        return mesh

    def test_disks_in_two_dimensions(self):
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
        # The last disk straddles two faces of the RVE, so it is only stamped correctly
        # if the bounding box is wrapped back in

        self.assert_same_grid(build_microstructure(rve_dims, Disk, particles), [40, 40])

    def test_ellipses_on_an_anisotropic_grid(self):
        rve_dims = [2.0, 1.0]
        particles = []
        for i_center, i_angle in (([0.5, 0.5], 0.4), ([1.7, 0.2], 1.9)):
            particle = Ellipse(
                "2", {"major_axis": 0.4, "minor_axis": 0.15, "angle": i_angle}, rve_dims
            )
            particle.position_center = np.array(i_center)
            particles.append(particle)
        # An RVE and a grid that differ between the two directions catch a swap of the
        # axes that a square grid would hide

        self.assert_same_grid(
            build_microstructure(rve_dims, Ellipse, particles), [48, 32]
        )

    def test_spheres_in_three_dimensions(self):
        rve_dims = [1.0, 1.0, 1.0]
        particles = []
        for i_center, i_radius in (
            ([0.5, 0.5, 0.5], 0.2),
            ([0.05, 0.05, 0.95], 0.15),
        ):
            particle = Sphere("2", {"r": i_radius}, rve_dims)
            particle.position_center = np.array(i_center)
            particles.append(particle)
        # The second sphere sits on a corner, so it is stamped into all eight of them

        self.assert_same_grid(
            build_microstructure(rve_dims, Sphere, particles), [16, 16, 16]
        )

    def test_ellipsoid_in_three_dimensions(self):
        rve_dims = [1.0, 1.0, 1.0]
        descriptors = {
            "axis_1": 0.4,
            "axis_2": 0.25,
            "axis_3": 0.15,
            "angle": 0.7,
            "rot_axis_comp_x": 0.0,
            "rot_axis_comp_y": 1.0,
            "rot_axis_comp_z": 1.0,
        }
        particle = Ellipsoid("2", dict(descriptors), rve_dims)
        particle.position_center = np.array([0.5, 0.5, 0.5])
        self.assert_same_grid(
            build_microstructure(rve_dims, Ellipsoid, [particle]), [14, 18, 12]
        )

    def test_grid_reaches_the_crate_file_unchanged(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.2}, rve_dims)
        particle.position_center = np.array([0.4, 0.6])
        microstructure = build_microstructure(rve_dims, Disk, [particle])
        mesh = VoxelMesher([32, 32]).mesh(microstructure)
        with tempfile.TemporaryDirectory() as temp_dir:
            written = CrateWriter().write(mesh, os.path.join(temp_dir, "grid.rgmsh"))
            self.assertEqual(len(written), 1)
            np.testing.assert_array_equal(
                np.load(written[0]), legacy_grid(microstructure, [32, 32])
            )
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
