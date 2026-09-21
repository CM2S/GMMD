"""Test module for the post processing: the discretisations, then the analyses."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from geommicgen._optional import has_gmsh
from geommicgen.errors.error_classes import MissingOptionalDependency
from geommicgen.iofuncs.md_state import load_md_state, save_md_state
from geommicgen.iofuncs.printing import log_to_terminal, screen_to
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.pipeline import MESH_DIRECTORY, MeshJob
from geommicgen.postproc.postproc import (
    FINAL_CONFIG_STEP,
    check_analyses,
    post_proc,
    run_analyses,
)
from geommicgen.tests.helpers import (
    a_generation_run,
    disk_microstructure,
    sphere_microstructure,
)


def a_state(directory, positions=True):
    """
    Write the state of a run beside a microstructure and read it back.

    Parameters
    ----------
    directory: str
        Directory the state is written in.

    positions: bool
        Whether the run recorded the positions of the particles.

    Returns
    -------
    `.GenerationState`
        The state, as an analysis reads it.
    """
    run = a_generation_run()
    if not positions:
        run.position_center_history = None

    return load_md_state(save_md_state(directory, run))


class PostProcTest(unittest.TestCase):
    """Base class for tests that post process into a directory of their own."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sample_dir = self.directory.name
        log_to_terminal()
        screen_to(self.sample_dir)
        self.addCleanup(screen_to, None)

    def written(self, *names):
        """Say whether a file or directory of the sample was written."""
        return os.path.exists(os.path.join(self.sample_dir, *names))


class TestPostProc(PostProcTest):
    """Test class for the composition the input data file runs."""

    def test_the_final_configuration_time_is_returned(self):
        times = post_proc([], disk_microstructure(), None, self.sample_dir, {"final_config": True})
        self.assertEqual(list(times), [FINAL_CONFIG_STEP])
        self.assertGreater(times[FINAL_CONFIG_STEP], 0.0)
        self.assertTrue(self.written("final_config.pdf"))

    def test_nothing_asked_for_returns_no_times(self):
        self.assertEqual(post_proc([], disk_microstructure(), None, self.sample_dir, {}), {})

    def test_the_meshes_are_run(self):
        job = MeshJob(VoxelMesher([4, 4]), [], "grid")
        post_proc([job], disk_microstructure(), None, self.sample_dir, {})
        self.assertIsNone(job.error)
        self.assertTrue(self.written(MESH_DIRECTORY, "grid.vti"))

    def test_the_analyses_are_checked_before_the_meshes(self):
        job = MeshJob(VoxelMesher([4, 4]), [], "grid")
        with self.assertRaises(ValueError):
            post_proc(
                [job], disk_microstructure(), None, self.sample_dir, {"motion_analysis": True}
            )
        self.assertIsNone(job.time)
        self.assertFalse(self.written(MESH_DIRECTORY))
        # A request that cannot be honoured is found out at once, not after the meshing


class TestCheckAnalyses(PostProcTest):
    """Test class for the refusals, and for their coming before anything is written."""

    def test_the_motion_analysis_needs_the_state(self):
        with self.assertRaises(ValueError) as context:
            run_analyses(
                disk_microstructure(),
                None,
                self.sample_dir,
                {"final_config": True, "motion_analysis": True},
            )
        self.assertIn("md_state.npz", str(context.exception))
        self.assertFalse(self.written("final_config.pdf"))
        # Refused before the final configuration it was asked for alongside is drawn

    def test_an_unknown_voronoi_diagram_is_refused(self):
        with self.assertRaises(ValueError) as context:
            check_analyses(
                disk_microstructure(),
                None,
                {"voronoi_analysis": True, "voronoi_type": "other"},
            )
        self.assertIn("other", str(context.exception))
        self.assertIn("standard", str(context.exception))

    def test_the_weighted_diagram_is_no_longer_offered(self):
        with self.assertRaises(ValueError):
            check_analyses(
                disk_microstructure(),
                None,
                {"voronoi_analysis": True, "voronoi_type": "weighted"},
            )
        # It was accepted and computed the standard diagram in its place

    def test_a_diagram_is_not_checked_when_no_analysis_asks_for_it(self):
        check_analyses(disk_microstructure(), None, {"voronoi_type": "other"})

    def test_gmsh_is_required_up_front(self):
        needing_gmsh = (
            (sphere_microstructure(), None, {"final_config": True}),
            (disk_microstructure(), a_state(self.sample_dir), {"motion_analysis": True}),
            (
                sphere_microstructure(),
                None,
                {"voronoi_analysis": True, "plot_voronoi": True},
            ),
        )
        for i_microstructure, i_state, i_options in needing_gmsh:
            with self.subTest(options=i_options, dim=i_microstructure.dim):
                with patch.dict(sys.modules, {"gmsh": None}):
                    with self.assertRaises(MissingOptionalDependency):
                        run_analyses(i_microstructure, i_state, self.sample_dir, i_options)
                self.assertFalse(self.written("final_config.msh"))
                self.assertFalse(self.written("motion_results"))
                self.assertFalse(self.written("voronoi_analysis_results"))
        # Setting the module entry to None makes the import raise; nothing is written
        # before the refusal, where the paths used to leave an empty directory behind

    def test_what_does_not_draw_through_gmsh_does_not_need_it(self):
        state = a_state(self.sample_dir, positions=False)
        with patch.dict(sys.modules, {"gmsh": None}):
            run_analyses(
                disk_microstructure(),
                state,
                self.sample_dir,
                {"final_config": True, "motion_analysis": True, "stat_nearest_neighbor": True},
            )
        self.assertTrue(self.written("final_config.pdf"))
        self.assertTrue(self.written("motion_results", "relative_energy.pdf"))
        self.assertFalse(self.written("motion_results", "paths"))
        self.assertTrue(self.written("stat_analysis_results", "stat_results.stat"))
        # A two dimensional final configuration, a motion analysis of a run that kept
        # no positions, and the statistics are matplotlib and numpy alone

    def test_running_twice_writes_over_the_first_time(self):
        options = {"final_config": True, "stat_nearest_neighbor": True}
        run_analyses(disk_microstructure(), None, self.sample_dir, options)
        results = os.path.join(self.sample_dir, "stat_analysis_results", "stat_results.stat")
        first = os.stat(results).st_mtime_ns
        os.utime(results, ns=(first - 10**9, first - 10**9))
        run_analyses(disk_microstructure(), None, self.sample_dir, options)
        self.assertGreater(os.stat(results).st_mtime_ns, first - 10**9)
        # The meshers write over an earlier run into the same directory, and so do the
        # analyses; the directories they make are allowed to exist already


@unittest.skipUnless(has_gmsh(), "gmsh is not installed")
class TestMotionAnalysisWithPaths(PostProcTest):
    """Test class for the motion analysis when the run recorded where the particles went."""

    def test_the_paths_are_plotted(self):
        microstructure = disk_microstructure()
        run = a_generation_run()
        run.position_center_history = [
            [i_center - 0.2, i_center - 0.1, i_center.copy()]
            for i_center in (i.position_center for i in microstructure.particles)
        ]
        state = load_md_state(save_md_state(self.sample_dir, run))
        run_analyses(microstructure, state, self.sample_dir, {"motion_analysis": True})
        paths = os.path.join(self.sample_dir, "motion_results", "paths")
        self.assertTrue(os.listdir(paths))
        # The history ends where the particles are, as a run's does; the fixture's own
        # positions would put two overlapping disks at the origin


if __name__ == "__main__":
    unittest.main()
