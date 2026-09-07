import os
import subprocess
import symtable
import sys
import types
import unittest
from unittest.mock import patch

import geommicgen

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


def functions_using_unbound_gmsh(path):
    """
    Give the functions of a module that read gmsh without binding it first.

    Parameters
    ----------
    path: str
        Path of the module to examine.

    Returns
    -------
    list
        Names of the offending functions.
    """
    with open(path, "r") as source_file:
        table = symtable.symtable(source_file.read(), path, "exec")
    if "gmsh" in table.get_identifiers():
        return []
    # A module that binds gmsh at the top level makes every bare use of it legitimate.
    # There is none, and TestImportWithoutGmsh is what keeps it that way

    offenders = []
    scopes = list(table.get_children())
    while scopes:
        scope = scopes.pop()
        scopes += scope.get_children()
        if scope.get_type() != "function":
            continue
        for i_symbol in scope.get_symbols():
            if i_symbol.get_name() == "gmsh" and i_symbol.is_global():
                offenders.append(scope.get_name())
    # A parameter, an assignment, an import and a with statement all bind the name, and
    # a closure over an enclosing binding makes it free rather than global, so this asks
    # exactly the right question without spelling out the ways a name can be bound

    return offenders


def source_files():
    """Paths of every module of the package."""
    package_dir = os.path.dirname(os.path.abspath(geommicgen.__file__))
    paths = []
    for i_dir, _, i_files in os.walk(package_dir):
        paths += [
            os.path.join(i_dir, i_file)
            for i_file in i_files
            if i_file.endswith(".py")
        ]

    return sorted(paths)


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


class TestGmshIsAlwaysBound(unittest.TestCase):
    """Test class checking that gmsh is bound wherever an attribute of it is read."""

    def test_every_function_that_uses_gmsh_binds_it(self):
        offenders = []
        for i_path in source_files():
            offenders += [
                "{0}: {1}".format(os.path.basename(i_path), i_function)
                for i_function in functions_using_unbound_gmsh(i_path)
            ]
        self.assertEqual(offenders, [])
        # gmsh is optional, so it is never a module global. A function that reads an
        # attribute of it without binding it first raises a NameError, which is caught
        # by any handler broad enough to be reaching for gmsh in the first place


if __name__ == "__main__":
    unittest.main()
