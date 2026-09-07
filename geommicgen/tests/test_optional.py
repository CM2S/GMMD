import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from geommicgen._optional import (
    MINIMUM_GMSH_VERSION,
    has_gmsh,
    require_gmsh,
    require_gmsh2links,
)
from geommicgen.errors.error_classes import MissingOptionalDependency


BLOCK_GMSH_SCRIPT = """
import sys


class BlockGmsh:
    def find_spec(self, name, path=None, target=None):
        if name == "gmsh" or name.startswith("gmsh."):
            raise ImportError("gmsh is blocked for this test")
        return None


sys.meta_path.insert(0, BlockGmsh())

import geommicgen
import geommicgen.postproc.postproc
import geommicgen.postproc.plotfuncs.plotting_functions
import geommicgen.postproc.mshgen.meshing_interface
import geommicgen.iofuncs.microstructure_yaml

from geommicgen._optional import require_gmsh
from geommicgen.errors.error_classes import MissingOptionalDependency

try:
    require_gmsh()
except MissingOptionalDependency as error:
    assert "geommicgen[gmsh]" in str(error), str(error)
else:
    raise AssertionError("require_gmsh did not raise while gmsh was blocked")

print("IMPORTED WITHOUT GMSH")
"""


class TestRequireGmsh(unittest.TestCase):
    """Test class for the retrieval of the optional gmsh dependency."""

    def test_returns_module_when_recent_enough(self):
        gmsh = require_gmsh()
        self.assertTrue(hasattr(gmsh, "model"))
        self.assertTrue(has_gmsh())

    def test_raises_when_missing(self):
        with patch.dict(sys.modules, {"gmsh": None}):
            with self.assertRaises(MissingOptionalDependency) as context:
                require_gmsh()
        self.assertIn("geommicgen[gmsh]", str(context.exception))
        # Setting the entry to None makes the import statement raise an ImportError

    def test_raises_when_too_old(self):
        old_gmsh = types.ModuleType("gmsh")
        old_gmsh.GMSH_API_VERSION_MAJOR = 4
        old_gmsh.GMSH_API_VERSION_MINOR = 9
        with patch.dict(sys.modules, {"gmsh": old_gmsh}):
            with self.assertRaises(MissingOptionalDependency) as context:
                require_gmsh()
            self.assertFalse(has_gmsh())
        message = str(context.exception)
        self.assertIn("4.9", message)
        self.assertIn(".".join(str(i_part) for i_part in MINIMUM_GMSH_VERSION), message)

    def test_raises_when_the_api_version_is_unreadable(self):
        nameless_gmsh = types.ModuleType("gmsh")
        with patch.dict(sys.modules, {"gmsh": nameless_gmsh}):
            with self.assertRaises(MissingOptionalDependency):
                require_gmsh()
        # An installation that cannot report its API version is refused, since the
        # versions that cannot are the old ones this check exists to catch

    def test_gmsh2links_message_names_the_repository(self):
        with patch.dict(sys.modules, {"gmsh2links.main": None, "gmsh2links": None}):
            with self.assertRaises(MissingOptionalDependency) as context:
                require_gmsh2links()
        self.assertIn("CM2S/Utilities", str(context.exception))


class TestImportWithoutGmsh(unittest.TestCase):
    """Test class checking that gmsh is genuinely optional at import time."""

    def test_package_imports_without_gmsh(self):
        result = subprocess.run(
            [sys.executable, "-c", BLOCK_GMSH_SCRIPT],
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            "importing without gmsh failed:\n{0}".format(result.stderr),
        )
        self.assertIn("IMPORTED WITHOUT GMSH", result.stdout)
        # A module level import of gmsh anywhere in the post processing would make the
        # subprocess fail, which is the regression this test exists to catch


if __name__ == "__main__":
    unittest.main()
