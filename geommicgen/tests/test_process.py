"""Test module for running a function in a process of its own."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from geommicgen._process import in_own_process
from geommicgen.errors.error_classes import ProcessDied, RaisedInAnotherProcess
from geommicgen.postproc.plotfuncs import plotting_functions


@in_own_process("adding")
def adding(first, second, report=None):
    """Add two numbers in another process, reporting each."""
    if report is not None:
        report(first, "first")
        report(second, "second")
    return first + second, os.getpid()


@in_own_process("raising")
def raising():
    """Raise in another process."""
    raise ValueError("raised in the other process")


@in_own_process("ending")
def ending():
    """End the other process without raising, as gmsh can."""
    os._exit(3)


class TestInOwnProcess(unittest.TestCase):
    """Test class for a function whose every call runs in a process of its own."""

    def test_what_it_returns_comes_back(self):
        total, pid = adding(1, second=2)
        self.assertEqual(total, 3)
        self.assertNotEqual(pid, os.getpid())

    def test_what_it_reports_is_reported_here(self):
        reported = []
        adding(1, 2, report=lambda *i_args: reported.append(i_args))
        self.assertEqual(reported, [(1, "first"), (2, "second")])
        # The callable cannot be sent to the other process, so what it is called with
        # there is sent back, and it is called with that here

    def test_an_error_comes_back_with_the_traceback_of_the_other_process(self):
        with self.assertRaisesRegex(
            ValueError, "raised in the other process"
        ) as context:
            raising()
        self.assertIsInstance(context.exception.__cause__, RaisedInAnotherProcess)
        self.assertIn("raise ValueError", str(context.exception.__cause__))

    def test_a_process_that_ends_is_told_with_an_error(self):
        with self.assertRaisesRegex(
            ProcessDied, "ending ran in ended with exit code 3"
        ):
            ending()
        self.assertEqual(adding(1, 2)[0], 3)
        # Gmsh can end its process rather than raise, which took the one that called
        # it down with it

    def test_the_main_module_is_left_as_it_was(self):
        main_module = sys.modules["__main__"]
        before = (getattr(main_module, "__file__", None), main_module.__spec__)
        adding(1, 2)
        self.assertEqual(
            (getattr(main_module, "__file__", None), main_module.__spec__), before
        )

    def test_a_script_that_calls_it_at_its_top_level_runs_once(self):
        script = textwrap.dedent(
            """
            from geommicgen.tests.test_process import adding
            print("top level")
            print("total", adding(1, 2)[0])
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "script.py")
            with open(path, "w") as script_file:
                script_file.write(script)
            result = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=300
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("top level"), 1)
        self.assertIn("total 3", result.stdout)
        # A process that is not forked from the calling one ran the file that one was
        # started from, and a script with nothing to keep its top level from running
        # when imported was refused, or ran again in it

    def test_the_views_drawn_with_gmsh_run_in_their_own(self):
        for i_name in (
            "plot_particles_3d",
            "plot_particles_3d_one_by_one",
            "plot_paths",
            "plot_voronoi_3d",
            "plot_voronoi_3d_with_imts",
        ):
            with self.subTest(i_name):
                self.assertTrue(
                    hasattr(getattr(plotting_functions, i_name), "in_this_process")
                )
        # Each opens a session of gmsh, which can end the process it runs in


if __name__ == "__main__":
    unittest.main()
