import inspect
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from geommicgen._optional import has_gmsh
from geommicgen.iofuncs.keywords import Keyword
from geommicgen.pipeline import (
    MeshJob,
    build_mesh_jobs,
    refuse_repeated_files,
    writers_from_options,
)
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.mesher import available_meshers, get_mesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.meshing.mesh import Mesh
from geommicgen.tests.helpers import ProcessEndingMesher, disk_microstructure
from geommicgen.translators.crate import CrateWriter
from geommicgen.errors.error_classes import MeshTooLargeError, ProcessDied
from geommicgen.translators.links import LinksWriter


class TestBuildMeshJobs(unittest.TestCase):
    """Test class for the meshing jobs read out of the options of an input file."""

    def test_finite_element_mesh(self):
        jobs = build_mesh_jobs({"gmsh": {"element_type": "tri6", "mesh_size": 0.05}})
        self.assertEqual(len(jobs), 1)
        self.assertIsInstance(jobs[0].mesher, GmshMesher)
        self.assertEqual(jobs[0].mesher.element_type, "tri6")
        self.assertEqual(jobs[0].mesher.mesh_size, 0.05)
        self.assertEqual(jobs[0].base_name, "tri6_h0.05")
        self.assertEqual([type(i_writer) for i_writer in jobs[0].writers], [LinksWriter])
        self.assertEqual(jobs[0].description, "Finite element mesh generation")

    def test_elements_per_particle_is_carried_through(self):
        jobs = build_mesh_jobs(
            {"gmsh": {"element_type": "tri3", "elements_per_particle": 8}}
        )
        self.assertEqual(jobs[0].mesher.elements_per_particle, 8)
        self.assertIsNone(jobs[0].mesher.mesh_size)

    def test_one_job_per_grid_named_after_the_deck(self):
        jobs = build_mesh_jobs(
            {"voxel": {"n_voxels_dims": [[10, 10], [20, 20]]}}, "example.mdsim"
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            [i_job.base_name for i_job in jobs],
            ["example_10_10", "example_20_20"],
        )
        self.assertIsInstance(jobs[0].mesher, VoxelMesher)
        self.assertEqual([type(i_writer) for i_writer in jobs[0].writers], [CrateWriter])
        self.assertEqual(jobs[0].description, "Regular mesh generation")

    def test_formats_can_be_asked_for(self):
        jobs = build_mesh_jobs(
            {
                "gmsh": {
                    "element_type": "tri3",
                    "mesh_size": 0.1,
                    "formats": ["links", "vtk"],
                }
            }
        )
        self.assertEqual([i_writer.name for i_writer in jobs[0].writers],
                         ["links", "vtk"])

    def test_file_name_names_the_files(self):
        jobs = build_mesh_jobs(
            {
                "voxel": {"n_voxels_dims": [[10, 10]], "file_name": "my_grid"},
                "gmsh": {"mesh_size": 0.1, "file_name": "my_mesh"},
            },
            "example.mdsim",
        )
        self.assertEqual(
            [i_job.base_name for i_job in jobs], ["my_grid", "my_mesh"]
        )
        # The name belongs to the files rather than to the mesher, so either takes it

    def test_one_name_under_two_meshers_writes_the_files_of_each(self):
        jobs = build_mesh_jobs(
            {
                "gmsh": {"mesh_size": 0.1},
                "voxel": {"n_voxels_dims": [[10, 10]]},
            }
        )
        for i_job in jobs:
            i_job.base_name = "my_mesh"
        refuse_repeated_files(jobs)
        self.assertEqual(
            [i_writer.extension for i_job in jobs for i_writer in i_job.writers],
            [".mesh", ".rgmsh.npy"],
        )
        # A mesh and a grid of one name are written in formats of their own, and the
        # standard output of the one is a .vtu where the other is a .vti, so the two
        # discretisations come to no harm under one name

    def test_one_name_and_one_format_under_two_meshers_is_refused(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs(
                {
                    "gmsh": {"mesh_size": 0.1, "file_name": "my_mesh",
                             "formats": ["vtk"]},
                    "voxel": {"n_voxels_dims": [[10, 10]], "file_name": "my_mesh",
                              "formats": ["vtk"]},
                }
            )
        self.assertIn("my_mesh.vtk", str(context.exception))
        # The name was refused within one mesher already; two of them asking for a
        # format in common is the same file written twice

    def test_file_name_for_more_than_one_grid_is_refused(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs(
                {
                    "voxel": {
                        "n_voxels_dims": [[10, 10], [20, 20]],
                        "file_name": "my_grid",
                    }
                }
            )
        self.assertIn("File_Name", str(context.exception))
        self.assertIn("2", str(context.exception))
        # Every resolution would otherwise be written over the one before it

    def test_the_options_reach_the_writer(self):
        jobs = build_mesh_jobs(
            {
                "gmsh": {
                    "element_type": "tri3",
                    "mesh_size": 0.1,
                    "gauss_points": 6,
                    "boundary_type": "Mortar_Periodic_Condition",
                }
            }
        )
        writer = jobs[0].writers[0]
        self.assertEqual(writer.boundary_type, "Mortar_Periodic_Condition")
        self.assertFalse(writer.requires_periodic)
        self.assertEqual(writer.gauss_points["triangle6"], 6)
        # A writer is built from the options of the discretisation it belongs to, so a
        # deck configures a format the same way the command line does

    def test_an_option_of_another_writer(self):
        jobs = build_mesh_jobs(
            {
                "gmsh": {
                    "element_type": "tri3",
                    "mesh_size": 0.1,
                    "formats": ["abaqus"],
                    "periodic_constraints": False,
                }
            }
        )
        self.assertFalse(jobs[0].writers[0].requires_periodic)
        # Every writer is handed the options of the discretisation and takes what it
        # declared, so one format's keyword costs the others nothing

    def test_the_jobs_are_built_from_what_the_meshers_declare(self):
        for i_name in available_meshers():
            mesher_class = get_mesher(i_name)
            self.assertTrue(mesher_class.options)
            self.assertTrue(mesher_class.default_formats)
            for j_option in mesher_class.options:
                self.assertIn(
                    j_option.lower(), inspect.signature(mesher_class).parameters
                )
        # Nothing in build_mesh_jobs names a mesher, so what one takes and what it is
        # written in by default have to come from the mesher, and the options it
        # declares have to be the parameters it is built with

    def test_a_grid_without_a_resolution_is_refused(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs({"voxel": {}})
        self.assertIn("voxels", str(context.exception))

    def test_unknown_discretisation(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs({"nosuchmesh": {}})
        self.assertIn("gmsh, voxel", str(context.exception))

    def test_the_old_names_are_no_longer_read(self):
        for i_name in ("femsh", "rgmsh"):
            with self.subTest(name=i_name):
                with self.assertRaises(ValueError) as context:
                    build_mesh_jobs({i_name: {}})
                self.assertIn(i_name, str(context.exception))
        # A deck names the mesher; the names a discretisation had of its own, which
        # this module mapped onto the meshers, are refused with the names there are

    def test_unknown_format_is_refused_before_anything_is_meshed(self):
        with self.assertRaises(ValueError) as context:
            build_mesh_jobs(
                {
                    "gmsh": {
                        "element_type": "tri3",
                        "mesh_size": 0.1,
                        "formats": ["linkss"],
                    }
                }
            )
        self.assertIn("linkss", str(context.exception))
        # Resolving the writers while the deck is read is what keeps a typo from
        # costing a whole meshing run and then being reported as a meshing failure


class TestMesherContract(unittest.TestCase):
    """Test class holding every registered mesher to what the writers need of a mesh."""

    SMALLEST = {"gmsh": {"mesh_size": 0.15, "element_type": "tri6"}, "voxel": {"n_voxels_dims": [[6, 6]]}}
    # Options that mesh the disk fixture quickly with each mesher; a mesher added
    # without an entry here fails the test, which is the point

    def test_every_mesher_declares_itself(self):
        for i_name in available_meshers():
            with self.subTest(mesher=i_name):
                mesher_class = get_mesher(i_name)
                self.assertEqual(mesher_class.name, i_name)
                self.assertTrue(mesher_class.description)
                self.assertTrue(mesher_class.default_formats)
                self.assertTrue(mesher_class.options)
                self.assertIn(i_name, self.SMALLEST)

    def test_every_mesher_returns_a_mesh_the_writers_can_take(self):
        from meshio._common import num_nodes_per_cell

        for i_name in available_meshers():
            if i_name == "gmsh" and not has_gmsh():
                continue
            with self.subTest(mesher=i_name):
                (mesher,) = get_mesher(i_name).from_options(self.SMALLEST[i_name])
                self.assertTrue(mesher.label)
                self.assertIsInstance(mesher.warnings, (list, tuple))
                mesh = mesher.mesh(disk_microstructure())
                self.assertIsInstance(mesh, Mesh)
                self.assertEqual(mesh.points.shape[1], 3)
                self.assertEqual(len(mesh.cells), len(mesh.phase))
                for (j_type, j_connectivity), j_phase in zip(mesh.cells, mesh.phase):
                    self.assertIn(j_type, num_nodes_per_cell)
                    self.assertEqual(j_connectivity.shape[1], num_nodes_per_cell[j_type])
                    self.assertEqual(len(j_connectivity), len(j_phase))
                    self.assertTrue(set(j_phase.tolist()) <= set(mesh.phase_names))
                    self.assertLess(j_connectivity.max(), len(mesh.points))
                self.assertEqual(mesh.matrix_phase, "1")
                self.assertTrue(mesh.periodic)
                mesh.check_periodic_conformity()
                self.assertEqual(mesh.source["mesher"], i_name)
        # The requirements listed under "Adding a mesher" in the mesher module, held to
        # here so that they cannot drift from what the meshers do

    def test_a_mesher_that_returns_something_else_is_refused(self):
        class NotAMesher(VoxelMesher):
            def mesh(self, microstructure, report=None):
                return {"points": [], "cells": []}

        job = MeshJob(NotAMesher([4, 4]), [], "grid")
        with tempfile.TemporaryDirectory() as directory:
            job.run(disk_microstructure(), directory)
        self.assertIsInstance(job.error, TypeError)
        self.assertIn("rather than a Mesh", str(job.error))
        # Found out by the job, with the mesher named, rather than by whichever writer
        # first asks the dict for its cells


class TestMeshJobRun(unittest.TestCase):
    """Test class for running a meshing job."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.microstructure = disk_microstructure()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_the_standard_output_and_every_format(self):
        job = build_mesh_jobs({"voxel": {"n_voxels_dims": [[16, 16]]}}, "deck.mdsim")[0]
        job.run(self.microstructure, self.temp_dir.name)
        written = sorted(os.path.basename(i_file) for i_file in job.files)
        self.assertEqual(
            written,
            [
                "deck_16_16.mesh.json",
                "deck_16_16.rgmsh.npy",
                "deck_16_16.vti",
                "deck_16_16_example.dat",
            ],
        )
        for i_file in job.files:
            self.assertTrue(os.path.exists(i_file))
        self.assertIsNone(job.error)
        self.assertIsNotNone(job.time)
        # A structured mesh gets the image a viewer reads, not an unstructured grid

    def test_a_failure_is_recorded_rather_than_raised(self):
        job = MeshJob(VoxelMesher([8, 8, 8]), [CrateWriter()], "grid")
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
            {"voxel": {"n_voxels_dims": [[16, 16]], "formats": []}}, "deck.mdsim"
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
        job = MeshJob(VoxelMesher([16, 16], max_cells=10), [LinksWriter()], "grid")
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
            {"gmsh": {"element_type": "tri3", "mesh_size": 0.15, "formats": ["vtu"]}}
        )[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            job.run(disk_microstructure(), temp_dir)
            self.assertIsInstance(job.error, ValueError)
            self.assertIn("tri3_h0.15.vtu", str(job.error))
            self.assertEqual(os.listdir(temp_dir), [])
        # Both would write tri3.vtu, so one would land on top of the other and only the
        # second would survive. Nothing is written at all instead. A grid cannot collide
        # this way, since no writer claims the .vti an image is written as

    def test_a_mesher_that_ends_its_process_leaves_the_jobs_after_it(self):
        jobs = [
            MeshJob(ProcessEndingMesher(mesh_size=0.1, element_type="tri3"), [], "a"),
            build_mesh_jobs({"gmsh": {"element_type": "tri3", "mesh_size": 0.1}})[0],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for i_job in jobs:
                i_job.run(disk_microstructure(), temp_dir)
        self.assertIsInstance(jobs[0].error, ProcessDied)
        self.assertIsNone(jobs[1].error)
        # Gmsh can end its process rather than raise, and no except caught that; the
        # job that asked is told instead, and the next one runs

    def test_the_links_deck_and_the_standard_output_are_written(self):
        job = build_mesh_jobs({"gmsh": {"element_type": "tri3", "mesh_size": 0.1}})[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            job.run(disk_microstructure(), temp_dir)
            self.assertIsNone(job.error)
            written = sorted(os.path.basename(i_file) for i_file in job.files)
            self.assertEqual(
                written,
                [
                    "tri3_h0.1.mesh",
                    "tri3_h0.1.mesh.json",
                    "tri3_h0.1.vtu",
                    "tri3_h0.1_example.rve",
                ],
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