import unittest
from unittest.mock import Mock, patch

from geommicgen.iofuncs.printing import print_final_message
from geommicgen.postproc.mshgen.meshing_interface import (
    FEMMeshGenerator,
    RegularGridMeshGenerator,
)


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

    def test_generator_that_did_not_finish_is_left_out(self):
        finished = RegularGridMeshGenerator([4, 4], [1.0, 1.0])
        finished.time = 2.0
        unfinished = FEMMeshGenerator(0.1, "tri3", [1.0, 1.0])
        self.assertIsNone(unfinished.time)
        print_final_message(Mock(time=1.0), [finished, unfinished], {})
        self.assertIn("Regular mesh generation", self.summary())
        self.assertNotIn("Finite element mesh generation", self.summary())
        self.assertIn("3.00e+00", self.summary())
        # The total is the two steps that finished, and the one that did not is not
        # reported rather than making the report fail

    def test_generator_of_an_unknown_class_is_reported_under_its_name(self):
        print_final_message(Mock(time=1.0), [Mock(time=3.0)], {})
        self.assertIn("Mock", self.summary())
        # The name used to be left over from the previous iteration of the loop, and
        # was unbound altogether for the first generator of an unknown class

    def test_nothing_finished_at_all(self):
        print_final_message(Mock(time=None), [], {})
        self.assertIn("Program Completed", self.summary())
        # No division by a total of zero

    def test_post_processing_times_are_reported(self):
        print_final_message(Mock(time=1.0), [], {"Voronoi analysis": 3.0})
        self.assertIn("Voronoi analysis", self.summary())
        self.assertIn("75", self.summary())
        # Three quarters of a total of four seconds


if __name__ == "__main__":
    unittest.main()
