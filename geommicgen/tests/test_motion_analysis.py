"""Test module for the analysis of the motion the particles went through."""

import shutil
import tempfile
import unittest

import numpy as np

from geommicgen._optional import has_gmsh
from geommicgen.postproc.voronoimetrics.motion_analysis import do_motion_analysis
from geommicgen.tests.helpers import disk_microstructure


@unittest.skipUnless(has_gmsh(), "gmsh not installed")
class TestPlotPaths(unittest.TestCase):
    """Test class for the views of the path each particle took."""

    def setUp(self):
        """Build a microstructure and a history that leads away from it."""
        self.microstructure = disk_microstructure()
        self.particles = self.microstructure.particles
        self.centers = [
            i_particle.position_center.copy() for i_particle in self.particles
        ]
        self.history = [
            [i_center - 0.2, i_center - 0.1] for i_center in self.centers
        ]
        self.sample_dir = tempfile.mkdtemp()
        # The history holds where the particles stood before the run was contracted,
        # resized and offset, so it does not end where the microstructure is

    def tearDown(self):
        """Remove the directory the views were written to."""
        shutil.rmtree(self.sample_dir)

    def test_the_microstructure_is_left_where_it_was(self):
        do_motion_analysis(
            self.particles,
            self.microstructure.rve_dims,
            self.sample_dir,
            position_center_history=self.history,
        )
        for i_particle, i_center in zip(self.particles, self.centers):
            np.testing.assert_allclose(i_particle.position_center, i_center)

    def test_the_microstructure_is_left_where_it_was_when_a_view_fails(self):
        with self.assertRaises(TypeError):
            do_motion_analysis(
                self.particles,
                None,
                self.sample_dir,
                position_center_history=self.history,
            )
        for i_particle, i_center in zip(self.particles, self.centers):
            np.testing.assert_allclose(i_particle.position_center, i_center)
        # Giving no box makes the view raise part way through the walk


if __name__ == "__main__":
    unittest.main()
