"""Test module for the code a run is said to be made with."""

import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from geommicgen.provenance import code_provenance


class TestCodeProvenance(unittest.TestCase):
    """Test class for the commit of GMMD and the versions of the libraries."""

    def test_the_code_is_said_by_its_commit_and_the_libraries_by_version(self):
        code = code_provenance.__wrapped__()
        if code["git_commit"] is None:
            self.assertIsNone(code["git_dirty"])
        else:
            self.assertRegex(code["git_commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(code["git_dirty"], bool)
        self.assertEqual(code["libraries"]["numpy"], np.__version__)
        # A copy of the code exported from git, as a test run elsewhere may be, has
        # no commit to say

    def test_no_commit_is_said_outside_a_checkout(self):
        with patch(
            "geommicgen.provenance.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            code = code_provenance.__wrapped__()
        self.assertIsNone(code["git_commit"])
        self.assertIsNone(code["git_dirty"])
        # An install, outside the checkout, is said by its version alone


if __name__ == "__main__":
    unittest.main()
