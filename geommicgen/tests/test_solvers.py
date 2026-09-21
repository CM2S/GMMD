"""
Integration tests that run the files GMMD writes through the solvers that read them.

These are the tests that say whether a file works, rather than whether it is well
formed: each one hands a mesh GMMD wrote to the solver and checks that the solver ran
it to completion. A solver that is not installed skips its tests, so the module says
nothing about a format it could not run.

LINKS is found through the environment variable GEOMMICGEN_LINKS, holding the path of
the executable, or as LINKS on the PATH.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from geommicgen._optional import has_gmsh
from geommicgen.pipeline import MeshJob
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.tests.helpers import disk_microstructure, sphere_microstructure
from geommicgen.translators.links import LinksWriter


def links_executable():
    """Path of the LINKS executable, or None when there is none to be found."""
    return os.environ.get("GEOMMICGEN_LINKS") or shutil.which("LINKS")


@unittest.skipUnless(links_executable(), "LINKS is not available")
class TestLinksRunsTheMesh(unittest.TestCase):
    """Test class for running the LINKS files through LINKS itself."""

    def setUp(self):
        """Create a directory for the run, which LINKS fills with its outputs."""
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)

    def run_links(self, mesher, microstructure):
        """Mesh a microstructure, write the LINKS files and run LINKS on the example."""
        job = MeshJob(mesher, [LinksWriter()], "m")
        job.run(microstructure, self.run_dir.name)
        self.assertIsNone(job.error, job.trace)
        completed = subprocess.run(
            [links_executable(), "m_example.rve"],
            cwd=self.run_dir.name,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        log = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, log[-3000:])
        self.assertIn("successfully completed", log)
        self.assertIn("Homogenised first Piola-Kirchhoff", log)
        # The example is a one increment elastic analysis, so LINKS reaching the
        # homogenised stress is LINKS having read the mesh, paired the periodic
        # faces and converged

        return log

    def test_two_dimensional_grid(self):
        self.run_links(VoxelMesher([8, 8]), disk_microstructure())

    def test_three_dimensional_grid(self):
        self.run_links(VoxelMesher([4, 4, 4]), sphere_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_two_dimensional_triangles(self):
        self.run_links(GmshMesher(mesh_size=0.1, element_type="tri3"), disk_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_three_dimensional_tetrahedra(self):
        self.run_links(
            GmshMesher(mesh_size=0.2, element_type="tetra4"), sphere_microstructure()
        )


if __name__ == "__main__":
    unittest.main()
