import os
import tempfile
import unittest

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.meshing.from_deck import MeshJob, build_mesh_jobs
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.microstructure.particleclasses import Disk
from geommicgen.tests.helpers import build_microstructure, disk_microstructure


class TestBuildMeshJobs(unittest.TestCase):
    """Test class for the meshing jobs read out of the options of an input file."""

    def test_finite_element_mesh(self):
        jobs = build_mesh_jobs({"femsh": {"element_type": "tri6", "mesh_size": 0.05}})
        self.assertEqual(len(jobs), 1)
        self.assertIsInstance(jobs[0].mesher, GmshMesher)
        self.assertEqual(jobs[0].mesher.element_type, "tri6")
        self.assertEqual(jobs[0].mesher.mesh_size, 0.05)
        self.assertEqual(jobs[0].base_name, "femsh")
        self.assertEqual(jobs[0].formats, ["links"])

    def test_elements_per_particle_is_carried_through(self):
        jobs = build_mesh_jobs(
            {"femsh": {"element_type": "tri3", "elements_per_particle": 8}}
        )
        self.assertEqual(jobs[0].mesher.elements_per_particle, 8)
        self.assertIsNone(jobs[0].mesher.mesh_size)

    def test_one_job_per_grid_named_after_the_deck(self):
        jobs = build_mesh_jobs(
            {"rgmsh": {"n_voxels_dims": [[10, 10], [20, 20]]}}, "example.mdsim"
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            [i_job.base_name for i_job in jobs],
            ["example_10_10", "example_20_20"],
        )
        self.assertIsInstance(jobs[0].mesher, VoxelMesher)
        self.assertEqual(jobs[0].formats, ["crate"])

    def test_formats_can_be_asked_for(self):
        jobs = build_mesh_jobs(
            {
                "femsh": {
                    "element_type": "tri3",
                    "mesh_size": 0.1,
                    "solver_formats": "links,vtk",
                }
            }
        )
        self.assertEqual(jobs[0].formats, ["links", "vtk"])

    def test_write_msh_asks_for_the_gmsh_format(self):
        jobs = build_mesh_jobs(
            {"femsh": {"element_type": "tri3", "mesh_size": 0.1, "write_msh": True}}
        )
        self.assertEqual(jobs[0].formats, ["links", "gmsh"])
        # The gmsh file is no longer written on the way to the solver deck, so it is
        # asked for like any other format

    def test_slice_dir_is_refused(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs({"rgmsh": {"n_voxels_dims": [[10, 10]], "slice_dir": 0}})
        self.assertIn("Slice_Dir", str(context.exception))
        # It used to decide whether the grid was written at all, and silently produced
        # nothing for a three dimensional microstructure

    def test_unknown_discretisation(self):
        with self.assertRaises(ValueError):
            build_mesh_jobs({"nosuchmesh": {}})


class TestMeshJobRun(unittest.TestCase):
    """Test class for running a meshing job."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.2}, rve_dims)
        particle.position_center = np.array([0.5, 0.5])
        self.microstructure = build_microstructure(rve_dims, Disk, [particle])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_the_standard_output_and_every_format(self):
        job = build_mesh_jobs({"rgmsh": {"n_voxels_dims": [[16, 16]]}}, "deck.mdsim")[0]
        self.assertTrue(job.run(self.microstructure, self.temp_dir.name))
        written = sorted(os.path.basename(i_file) for i_file in job.files)
        self.assertEqual(written, ["deck_16_16.rgmsh.npy", "deck_16_16.vtk"])
        for i_file in job.files:
            self.assertTrue(os.path.exists(i_file))
        self.assertIsNone(job.error)
        self.assertIsNotNone(job.time)
        # A structured mesh gets the image a viewer reads, not an unstructured grid

    def test_a_failure_is_recorded_rather_than_raised(self):
        job = MeshJob(VoxelMesher([8, 8, 8]), ["crate"], "grid", "Regular mesh")
        self.assertFalse(job.run(self.microstructure, self.temp_dir.name))
        self.assertIsInstance(job.error, ValueError)
        self.assertEqual(job.files, [])
        self.assertIsNotNone(job.time)
        # A grid of three directions cannot mesh a microstructure of two, and the other
        # discretisations asked for still have to get their chance


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestMeshJobRunWithGmsh(unittest.TestCase):
    """Test class for running the finite element job end to end."""

    def test_the_links_deck_and_the_standard_output_are_written(self):
        job = build_mesh_jobs({"femsh": {"element_type": "tri3", "mesh_size": 0.1}})[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(job.run(disk_microstructure(), temp_dir))
            written = sorted(os.path.basename(i_file) for i_file in job.files)
            self.assertEqual(
                written,
                ["femsh.mesh", "femsh.mesh.json", "femsh.vtu", "femsh_example.rve"],
            )
            for i_file in job.files:
                self.assertTrue(os.path.exists(i_file))
        # The LINKS deck no longer goes through a .msh file and a separate package


if __name__ == "__main__":
    unittest.main()
