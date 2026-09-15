import unittest
from unittest.mock import Mock, patch

from geommicgen.iofuncs.printing import print_failed_jobs, print_final_message
from geommicgen.pipeline import MeshJob
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher


def finished_job(seconds):
    """Build a regular grid job that ran for the given time."""
    job = MeshJob(VoxelMesher([4, 4]), [], "grid")
    job.time = seconds

    return job


def unfinished_job():
    """Build a finite element job that never ran."""
    return MeshJob(GmshMesher(mesh_size=0.1), [], "femsh")


class TestPrintFinalMessage(unittest.TestCase):
    """
    Test class for the summary of execution times printed at the end of a run.

    The summary is printed from a finally block, so anything it raises replaces the
    error that actually stopped the run, and the run then reports the wrong thing.
    """

    def setUp(self):
        self.printed = []
        patcher = patch(
            "geommicgen.iofuncs.printing.print_to_file",
            side_effect=lambda message, **kwargs: self.printed.append(str(message)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def summary(self):
        """Give everything that was printed as one string."""
        return "\n".join(self.printed)

    def test_job_that_did_not_finish_is_left_out(self):
        self.assertIsNone(unfinished_job().time)
        print_final_message(Mock(time=1.0), [finished_job(2.0), unfinished_job()], {})
        self.assertIn("Regular mesh generation", self.summary())
        self.assertNotIn("Finite element mesh generation", self.summary())
        self.assertIn("3.00e+00", self.summary())
        # The total is the two steps that finished, and the one that did not is left
        # out rather than making the report fail

    def test_step_that_does_not_name_itself_is_reported_by_its_class(self):
        nameless = Mock(spec=["time"])
        nameless.time = 3.0
        self.assertIsNone(getattr(nameless, "description", None))
        print_final_message(Mock(time=1.0), [nameless], {})
        self.assertIn("Mock", self.summary())
        # The name used to be left over from the previous iteration of the loop, and
        # unbound altogether for the first step of an unexpected class. The mock is
        # given a spec so that it genuinely has no description, which a bare Mock,
        # answering every attribute, would not

    def test_nothing_finished_at_all(self):
        print_final_message(Mock(time=None), [], {})
        self.assertIn("Program Completed", self.summary())
        # No division by a total of zero

    def test_post_processing_times_are_reported(self):
        print_final_message(Mock(time=1.0), [], {"Voronoi analysis": 3.0})
        self.assertIn("Voronoi analysis", self.summary())
        self.assertIn("75", self.summary())
        # Three quarters of a total of four seconds


class TestPrintFailedJobs(unittest.TestCase):
    """Test class for the report of the discretisations that could not be produced."""

    def setUp(self):
        self.printed = []
        self.terminal = []
        patcher = patch(
            "geommicgen.iofuncs.printing.print_to_file",
            side_effect=self.record,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, message, **kwargs):
        """Record a printed message, and whether it reached the terminal."""
        self.printed.append(str(message))
        if kwargs.get("to_terminal", True):
            self.terminal.append(str(message))

    def failed_job(self):
        """Build a job carrying a failure."""
        job = finished_job(1.0)
        job.error = ValueError("the mesher gave up")
        job.trace = "Traceback (most recent call last):\n  the frames\n"

        return job

    def test_nothing_is_printed_when_every_job_succeeded(self):
        self.assertEqual(print_failed_jobs([finished_job(1.0)]), [])
        self.assertEqual(self.printed, [])

    def test_the_failures_are_returned_and_named(self):
        failed = self.failed_job()
        self.assertEqual(print_failed_jobs([finished_job(1.0), failed]), [failed])
        summary = "\n".join(self.printed)
        self.assertIn("Regular mesh generation", summary)
        self.assertIn("ValueError", summary)
        self.assertIn("the mesher gave up", summary)
        # Returning them is what lets the caller decide the exit status without
        # working out for itself what counts as a failure

    def test_the_traceback_goes_to_the_file_and_not_the_screen(self):
        print_failed_jobs([self.failed_job()])
        self.assertIn("the frames", "\n".join(self.printed))
        self.assertNotIn("the frames", "\n".join(self.terminal))
        # A message is what a user needs; the frames are for whoever has to fix it


if __name__ == "__main__":
    unittest.main()
