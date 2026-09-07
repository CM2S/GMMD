import os
import tempfile
import types
import unittest
from unittest.mock import patch

from geommicgen._optional import has_gmsh
from geommicgen.iofuncs.keywords import Keyword
from geommicgen.meshing.from_deck import (
    MeshJob,
    build_mesh_jobs,
    writers_from_options,
)
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.tests.helpers import disk_microstructure
from geommicgen.translators.crate import CrateWriter
from geommicgen.errors.error_classes import MeshTooLargeError
from geommicgen.translators.links import LinksWriter


class TestBuildMeshJobs(unittest.TestCase):
    """Test class for the meshing jobs read out of the options of an input file."""

    def test_finite_element_mesh(self):
        jobs = build_mesh_jobs({"femsh": {"element_type": "tri6", "mesh_size": 0.05}})
        self.assertEqual(len(jobs), 1)
        self.assertIsInstance(jobs[0].mesher, GmshMesher)
        self.assertEqual(jobs[0].mesher.element_type, "tri6")
        self.assertEqual(jobs[0].mesher.mesh_size, 0.05)
        self.assertEqual(jobs[0].base_name, "tri6")
        self.assertEqual(jobs[0].writers, [LinksWriter])
        self.assertEqual(jobs[0].description, "Finite element mesh generation")

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
        self.assertEqual(jobs[0].writers, [CrateWriter])
        self.assertEqual(jobs[0].description, "Regular mesh generation")

    def test_formats_can_be_asked_for(self):
        jobs = build_mesh_jobs(
            {
                "femsh": {
                    "element_type": "tri3",
                    "mesh_size": 0.1,
                    "formats": ["links", "vtk"],
                }
            }
        )
        self.assertEqual([i_writer.name for i_writer in jobs[0].writers],
                         ["links", "vtk"])

    def test_write_msh_asks_for_the_gmsh_format(self):
        jobs = build_mesh_jobs(
            {"femsh": {"element_type": "tri3", "mesh_size": 0.1, "write_msh": True}}
        )
        self.assertEqual([i_writer.name for i_writer in jobs[0].writers],
                         ["links", "gmsh"])
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

    def test_unknown_format_is_refused_before_anything_is_meshed(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs(
                {
                    "femsh": {
                        "element_type": "tri3",
                        "mesh_size": 0.1,
                        "formats": ["linkss"],
                    }
                }
            )
        self.assertIn("linkss", str(context.exception))
        # Resolving the writers while the deck is read is what keeps a typo from
        # costing a whole meshing run and then being reported as a meshing failure


class TestMeshJobRun(unittest.TestCase):
    """Test class for running a meshing job."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.microstructure = disk_microstructure()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_the_standard_output_and_every_format(self):
        job = build_mesh_jobs({"rgmsh": {"n_voxels_dims": [[16, 16]]}}, "deck.mdsim")[0]
        job.run(self.microstructure, self.temp_dir.name)
        written = sorted(os.path.basename(i_file) for i_file in job.files)
        self.assertEqual(
            written,
            ["deck_16_16.mesh.json", "deck_16_16.rgmsh.npy", "deck_16_16.vti"],
        )
        for i_file in job.files:
            self.assertTrue(os.path.exists(i_file))
        self.assertIsNone(job.error)
        self.assertIsNotNone(job.time)
        # A structured mesh gets the image a viewer reads, not an unstructured grid

    def test_a_failure_is_recorded_rather_than_raised(self):
        job = MeshJob(VoxelMesher([8, 8, 8]), [CrateWriter], "grid")
        job.run(self.microstructure, self.temp_dir.name)
        self.assertIsInstance(job.error, ValueError)
        self.assertEqual(job.files, [])
        self.assertIsNotNone(job.time)
        self.assertIn("Traceback", job.trace)
        self.assertIsNone(job.error.__traceback__)
        # The traceback is kept as text and taken off the exception, which would
        # otherwise hold the whole mesh alive for as long as the job is

    def test_the_standard_output_alone_is_a_whole_job(self):
        job = build_mesh_jobs(
            {"rgmsh": {"n_voxels_dims": [[16, 16]], "formats": []}}, "deck.mdsim"
        )[0]
        self.assertEqual(job.writers, [])
        job.run(self.microstructure, self.temp_dir.name)
        self.assertIsNone(job.error)
        self.assertEqual(
            sorted(os.path.basename(i_file) for i_file in job.files),
            ["deck_16_16.mesh.json", "deck_16_16.vti"],
        )
        # An empty list is not the same as no list at all: it asks for the second stage
        # and nothing after it, which is a thing a deck should be able to say

    def test_what_reached_the_disk_is_reported_when_a_format_fails(self):
        job = MeshJob(VoxelMesher([16, 16], max_cells=10), [LinksWriter], "grid")
        job.run(self.microstructure, self.temp_dir.name)
        self.assertIsInstance(job.error, MeshTooLargeError)
        self.assertEqual(
            sorted(os.path.basename(i_file) for i_file in job.files),
            ["grid.mesh.json", "grid.vti"],
        )
        self.assertEqual(
            sorted(os.listdir(self.temp_dir.name)),
            ["grid.mesh.json", "grid.vti"],
        )
        # The standard output is written before the formats that need the cells, so a
        # failure there leaves a real file behind. Reporting nothing would say the job
        # produced nothing, which is not what happened


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestMeshJobRunWithGmsh(unittest.TestCase):
    """Test class for running the finite element job end to end."""

    def test_a_writer_may_not_write_over_the_standard_output(self):
        job = build_mesh_jobs(
            {"femsh": {"element_type": "tri3", "mesh_size": 0.15, "formats": ["vtu"]}}
        )[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            job.run(disk_microstructure(), temp_dir)
            self.assertIsInstance(job.error, ValueError)
            self.assertIn("standard output", str(job.error))
            self.assertEqual(os.listdir(temp_dir), [])
        # Both would write tri3.vtu, so one would land on top of the other and only the
        # second would survive. Nothing is written at all instead. A grid can no longer
        # collide, since no writer claims the .vti an image is written as

    def test_the_links_deck_and_the_standard_output_are_written(self):
        job = build_mesh_jobs({"femsh": {"element_type": "tri3", "mesh_size": 0.1}})[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            job.run(disk_microstructure(), temp_dir)
            self.assertIsNone(job.error)
            written = sorted(os.path.basename(i_file) for i_file in job.files)
            self.assertEqual(
                written,
                ["tri3.mesh", "tri3.mesh.json", "tri3.vtu", "tri3_example.rve"],
            )
            for i_file in job.files:
                self.assertTrue(os.path.exists(i_file))
        # The LINKS deck no longer goes through a .msh file and a separate package


class TestFormatsFromTheDeck(unittest.TestCase):
    """
    Test class for the spelling of a list of formats in an input data file.

    The reader handed on only the first word of the line, so every spelling with a
    space in it lost every format after the first, in silence.
    """

    def read_formats(self, line):
        """Read one keyword line the way the reader of the input file does."""
        reader = types.SimpleNamespace(input=[line], i_line=0)
        with patch.object(Keyword, "input_reader", reader):
            return Keyword("Formats", type_str="str_list").read_value()

    def test_bracketed_list(self):
        self.assertEqual(
            self.read_formats("Formats [links, vtk]"), ["links", "vtk"]
        )
        # The spelling every other list in the input file uses

    def test_list_without_brackets(self):
        self.assertEqual(
            self.read_formats("Formats links, vtk"), ["links", "vtk"]
        )

    def test_list_without_spaces(self):
        self.assertEqual(
            self.read_formats("Formats links,vtk"), ["links", "vtk"]
        )

    def test_a_single_format(self):
        self.assertEqual(self.read_formats("Formats links"), ["links"])

    def test_every_spelling_reaches_the_writers(self):
        for i_line in (
            "Formats [links, vtk]",
            "Formats links, vtk",
            "Formats links,vtk",
        ):
            writers = writers_from_options(
                {"formats": self.read_formats(i_line)}, ()
            )
            self.assertEqual(
                [i_writer.name for i_writer in writers], ["links", "vtk"], i_line
            )


if __name__ == "__main__":
    unittest.main()