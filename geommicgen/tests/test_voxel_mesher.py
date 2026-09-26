import os
import tempfile
import unittest
from functools import partial

import numpy as np

from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import available_meshers, get_mesher, mesher_options
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses import (
    Cylinder,
    Disk,
    Ellipse,
    Ellipsoid,
    Sphere,
)
from geommicgen.tests.helpers import build_microstructure
from geommicgen.translators.crate import CrateWriter

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
# Grids produced by RegularGridMeshGenerator, the mesher the voxel mesher replaced,
# while that class was still in the tree. They are what pins the replacement to what it
# replaced now that the original is gone, so they are never regenerated from the code
# under test: a fixture rewritten by the thing it checks agrees with anything.


def case(particle_class, rve_dims, specs, n_voxels_dims):
    """
    Build one of the microstructures a fixture was frozen for.

    Parameters
    ----------
    particle_class: class
        Class of the particles.

    rve_dims: list
        Dimensions of the microstructure in each spatial direction.

    specs: tuple
        One *(descriptors, centre)* pair per particle.

    n_voxels_dims: list
        Number of voxels in each spatial direction.

    Returns
    -------
    tuple
        The microstructure and the grid it is meshed on.
    """
    particles = []
    for i_descriptors, i_center in specs:
        particle = particle_class("2", dict(i_descriptors), rve_dims)
        particle.position_center = np.array(i_center)
        particles.append(particle)

    return build_microstructure(rve_dims, particle_class, particles), n_voxels_dims


EQUIVALENCE_CASES = {
    # The last disk straddles two faces, so it is only stamped correctly if the
    # bounding box is wrapped back into the RVE
    "disks": partial(case, Disk, [1.0, 1.0], (
        ({"r": 0.15}, [0.25, 0.75]),
        ({"r": 0.08}, [0.6, 0.1]),
        ({"r": 0.12}, [0.02, 0.98]),
    ), [40, 40]),
    # Nothing square anywhere, which catches a swap of the axes
    "ellipses": partial(case, Ellipse, [2.0, 1.0], (
        ({"major_axis": 0.4, "minor_axis": 0.15, "angle": 0.4}, [0.5, 0.5]),
        ({"major_axis": 0.4, "minor_axis": 0.15, "angle": 1.9}, [1.7, 0.2]),
    ), [48, 32]),
    # The second sphere is stamped into all eight corners
    "spheres": partial(case, Sphere, [1.0, 1.0, 1.0], (
        ({"r": 0.2}, [0.5, 0.5, 0.5]),
        ({"r": 0.15}, [0.05, 0.05, 0.95]),
    ), [16, 16, 16]),
    "ellipsoid": partial(case, Ellipsoid, [1.0, 1.0, 1.0], (
        ({"axis_1": 0.4, "axis_2": 0.25, "axis_3": 0.15, "angle": 0.7,
          "rot_axis_comp_x": 0.0, "rot_axis_comp_y": 1.0, "rot_axis_comp_z": 1.0},
         [0.5, 0.5, 0.5]),
    ), [14, 18, 12]),
    "one_disk": partial(case, Disk, [1.0, 1.0], (
        ({"r": 0.2}, [0.4, 0.6]),
    ), [32, 32]),
}
# Every case a fixture was frozen for. Their geometry is pinned by those fixtures and
# must not be edited without freezing new ones -- which cannot be done from this code,
# since the mesher that produced them is gone.


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

    def test_matches_the_frozen_grids(self):
        for i_name, i_case in EQUIVALENCE_CASES.items():
            with self.subTest(i_name):
                microstructure, n_voxels_dims = i_case()
                mesh = VoxelMesher(n_voxels_dims).mesh(microstructure)
                np.testing.assert_array_equal(
                    mesh.structured.phase_grid, frozen_grid(i_name)
                )
        # Driven from the table, so a case cannot be added and left untested, and a
        # fixture that went missing fails here rather than quietly not being loaded

    def test_the_grids_are_the_same_in_any_units(self):
        for i_name, i_case in EQUIVALENCE_CASES.items():
            for i_scale in (2.0**-20, 2.0**20):
                with self.subTest(i_name, scale=i_scale):
                    microstructure, n_voxels_dims = i_case()
                    mesh = VoxelMesher(n_voxels_dims).mesh(
                        microstructure.scaled(i_scale)
                    )
                    np.testing.assert_array_equal(
                        mesh.structured.phase_grid, frozen_grid(i_name)
                    )
                    np.testing.assert_array_equal(
                        mesh.structured.spacing,
                        np.asarray(microstructure.rve_dims)
                        * i_scale
                        / n_voxels_dims,
                    )
        # The test of a sphere let in a point a length of 1e-3 outside it, which at a
        # millionth of the unit made every sphere its bounding box; the grid is stamped
        # at unit scale and given the user's spacing, so that no length the particles
        # are tested with is taken in the user's units

    def test_grid_reaches_the_crate_file_unchanged(self):
        microstructure, n_voxels_dims = EQUIVALENCE_CASES["one_disk"]()
        mesh = VoxelMesher(n_voxels_dims).mesh(microstructure)
        with tempfile.TemporaryDirectory() as temp_dir:
            written = CrateWriter().write(mesh, os.path.join(temp_dir, "grid.rgmsh"))
            self.assertTrue(written[0].endswith(".rgmsh.npy"))
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

    def test_phase_fraction_follows_the_volume_fraction_of_a_sphere(self):
        rve_dims = [1.0, 1.0, 1.0]
        particle = Sphere("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.37, 0.52, 0.61])
        microstructure = build_microstructure(rve_dims, Sphere, [particle])
        mesh = VoxelMesher([160, 160, 160]).mesh(microstructure)
        fraction = np.count_nonzero(mesh.structured.phase_grid == 2) / 160**3
        self.assertAlmostEqual(fraction / (4 / 3 * np.pi * 0.1**3), 1.0, delta=0.01)
        # The test of a sphere let in a length of 1e-3 beyond its surface, which grew
        # this one by a hundredth and its phase by three per cent

    def test_a_cylinder_is_stamped_on_its_axis(self):
        rve_dims = [1.0, 1.0, 1.0]
        particle = Cylinder(
            "2",
            {
                "r_cyl": 0.1,
                "length": 0.4,
                "azimuth_angle": np.pi / 4,
                "polar_angle": np.pi / 2,
            },
            rve_dims,
        )
        particle.position_center = np.array([15.5, 15.5, 15.5]) / 32
        microstructure = build_microstructure(rve_dims, Cylinder, [particle])
        mesh = VoxelMesher([32, 32, 32]).mesh(microstructure)
        centers = (np.indices((32, 32, 32)).reshape(3, -1).T + 0.5) / 32
        np.testing.assert_array_equal(
            mesh.structured.phase_grid == 2,
            particle.points_inside(centers, rve_dims).reshape(32, 32, 32),
        )
        # The axis runs along a diagonal through the centres of voxels, two of which
        # the test of one point at a time used to leave out

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


class TestVoxelMesherFromOptions(unittest.TestCase):
    """Test class for building the voxel mesher from the options it declares."""

    def test_one_mesher_per_resolution(self):
        meshers = VoxelMesher.from_options({"n_voxels_dims": [[8, 8], [16, 16]]})

        self.assertEqual([i_mesher.label for i_mesher in meshers], ["8_8", "16_16"])
        np.testing.assert_array_equal(meshers[1].n_voxels_dims, [16, 16])

    def test_a_bare_resolution_is_one_mesher(self):
        meshers = VoxelMesher.from_options({"n_voxels_dims": [8, 8, 8]})

        self.assertEqual([i_mesher.label for i_mesher in meshers], ["8_8_8"])
        # A command line gives one resolution as a bare list

    def test_no_resolution_is_refused(self):
        with self.assertRaises(ValueError):
            VoxelMesher.from_options({})
        with self.assertRaises(ValueError):
            VoxelMesher.from_options({"n_voxels_dims": None})

    def test_declares_its_options_and_default_formats(self):
        self.assertEqual(set(VoxelMesher.options), {"N_Voxels_Dims"})
        self.assertEqual(VoxelMesher.default_formats, ("crate",))


class TestMesherRegistry(unittest.TestCase):
    """Test class for the registry of the meshers."""

    def test_voxel_mesher_is_registered(self):
        self.assertIs(get_mesher("voxel"), VoxelMesher)
        self.assertIn("voxel", available_meshers())

    def test_unknown_mesher(self):
        with self.assertRaises(ValueError):
            get_mesher("no_such_mesher")

    def test_every_mesher_option_is_declared_once(self):
        options = mesher_options()

        self.assertIn("N_Voxels_Dims", options)
        self.assertIn("Mesh_Size", options)
        for i_description in options.values():
            self.assertIn("type", i_description)
            self.assertIn("help", i_description)
        # What the command line offers is the union over the meshers, each option
        # described the same way wherever it is declared


if __name__ == "__main__":
    unittest.main()
