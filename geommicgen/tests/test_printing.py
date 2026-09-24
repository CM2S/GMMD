import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import geommicgen.iofuncs.printing as print_funcs
from geommicgen.iofuncs.file_handling import delete_screen
from geommicgen.iofuncs.printing import (
    print_failed_jobs,
    print_final_message,
    step_times,
    print_to_file,
    screen_to,
)
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
        print_final_message(
            step_times(Mock(time=1.0), [finished_job(2.0), unfinished_job()], {})
        )
        self.assertIn("Regular mesh generation", self.summary())
        self.assertNotIn("Finite element mesh generation", self.summary())
        self.assertIn("3.00e+00", self.summary())
        # The total is the two steps that finished, and the one that did not is left
        # out rather than making the report fail

    def test_step_that_does_not_name_itself_is_reported_by_its_class(self):
        nameless = Mock(spec=["time"])
        nameless.time = 3.0
        self.assertIsNone(getattr(nameless, "description", None))
        print_final_message(step_times(Mock(time=1.0), [nameless], {}))
        self.assertIn("Mock", self.summary())
        # The name used to be left over from the previous iteration of the loop, and
        # unbound altogether for the first step of an unexpected class. The mock is
        # given a spec so that it genuinely has no description, which a bare Mock,
        # answering every attribute, would not

    def test_nothing_finished_at_all(self):
        print_final_message(step_times(Mock(time=None), [], {}))
        self.assertIn("Program Completed", self.summary())
        # No division by a total of zero

    def test_post_processing_times_are_reported(self):
        print_final_message(step_times(Mock(time=1.0), [], {"Voronoi analysis": 3.0}))
        self.assertIn("Voronoi analysis", self.summary())
        self.assertIn("75", self.summary())
        # Three quarters of a total of four seconds

    def test_a_command_reports_the_steps_it_names(self):
        print_final_message({"Meshing": 3.0, "Writing": 1.0})
        self.assertIn("Meshing", self.summary())
        self.assertIn("Writing", self.summary())
        self.assertIn("75", self.summary())
        # The summary is a dictionary of times, so a command closes with the table a
        # run closes with, naming the steps it ran


class ScreenLogTest(unittest.TestCase):
    """
    Base class for tests that read what the program reports.

    The screen file is attached to a directory of the test's own, and the terminal is
    read through a redirected standard output.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        print_funcs.log_to_terminal()
        self.screen_path = screen_to(self.directory.name)
        self.addCleanup(screen_to, None)

    def screen(self):
        """Give what reached the screen file."""
        with open(self.screen_path) as screen:
            return screen.read()

    def terminal(self, function, *arguments):
        """Give what a call sent to the terminal."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            function(*arguments)

        return printed.getvalue()


class TestScreenLog(ScreenLogTest):
    """Test class for where what the program reports ends up."""

    def test_a_line_reaches_both(self):
        terminal = self.terminal(print_to_file, "a line")
        self.assertEqual(terminal, "a line\n")
        self.assertEqual(self.screen(), "a line\n")

    def test_the_screen_file_is_appended_to(self):
        with open(self.screen_path, "a") as screen:
            screen.write("an earlier run\n")
        screen_to(self.directory.name)
        print_to_file("a later one")
        self.assertEqual(self.screen(), "an earlier run\na later one\n")

    def test_the_debug_level_is_the_file_alone(self):
        terminal = self.terminal(print_funcs.LOGGER.debug, "the frames")
        self.assertEqual(terminal, "")
        self.assertEqual(self.screen(), "the frames\n")

    def test_detaching_leaves_the_file_alone(self):
        print_to_file("before")
        screen_to(None)
        terminal = self.terminal(print_to_file, "after")
        self.assertEqual(terminal, "after\n")
        self.assertEqual(self.screen(), "before\n")

    def test_attaching_elsewhere_moves_the_log(self):
        print_to_file("first")
        with tempfile.TemporaryDirectory() as elsewhere:
            other_path = screen_to(elsewhere)
            print_to_file("second")
            with open(other_path) as other:
                self.assertEqual(other.read(), "second\n")
        self.assertEqual(self.screen(), "first\n")

    def test_the_terminal_is_attached_once(self):
        print_funcs.log_to_terminal()
        print_funcs.log_to_terminal()
        self.assertEqual(self.terminal(print_to_file, "once"), "once\n")

    def test_the_screen_file_can_be_deleted_while_attached(self):
        print_to_file("something")
        delete_screen(self.directory.name)
        self.assertFalse(os.path.exists(self.screen_path))
        # The handler holds the file open; delete_screen lets go of it first, which is
        # what a removal on Windows needs


class TestPrintFailedJobs(ScreenLogTest):
    """Test class for the report of the discretisations that failed."""

    def failed_job(self):
        """Build a job carrying a failure."""
        job = finished_job(1.0)
        job.error = ValueError("the mesher gave up")
        job.trace = "Traceback (most recent call last):\n  the frames\n"

        return job

    def test_nothing_is_printed_when_every_job_succeeded(self):
        terminal = self.terminal(print_failed_jobs, [finished_job(1.0)])
        self.assertEqual(terminal, "")
        self.assertEqual(self.screen(), "")

    def test_the_failures_are_returned_and_named(self):
        failed = self.failed_job()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            returned = print_failed_jobs([finished_job(1.0), failed])
        self.assertEqual(returned, [failed])
        self.assertIn("Regular mesh generation", printed.getvalue())
        self.assertIn("ValueError", printed.getvalue())
        self.assertIn("the mesher gave up", printed.getvalue())
        # Returning them is what lets the caller decide the exit status without
        # working out for itself what counts as a failure

    def test_the_traceback_goes_to_the_file_and_not_the_terminal(self):
        terminal = self.terminal(print_failed_jobs, [self.failed_job()])
        self.assertIn("the mesher gave up", terminal)
        self.assertNotIn("the frames", terminal)
        self.assertIn("the frames", self.screen())
        # A message is what a user needs; the frames are for whoever has to fix it


if __name__ == "__main__":
    unittest.main()
