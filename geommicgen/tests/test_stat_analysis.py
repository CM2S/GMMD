"""Test module for the statistical descriptors of a microstructure, and their plots."""

import os
import tempfile
import unittest

import matplotlib
import matplotlib.pyplot
import numpy as np

matplotlib.use("Agg")

from geommicgen.microstructure.particleclasses import Disk
from geommicgen.postproc.plotfuncs.plotting_functions import (
    SMOOTHING_WINDOW,
    plot_nearest_neighbor_dist,
    plot_ripleys_k_func,
    plot_two_point_correlation,
)
from geommicgen.postproc.voronoimetrics.stat_analysis import (
    crosses_boundary,
    remove_particles_at_boundary,
)

RVE_DIMS = [1.0, 1.0]


def disk(center, radius=0.05):
    """Build a disk of the inclusion phase at a position."""
    particle = Disk("1", {"r": radius}, RVE_DIMS)
    particle.position_center = np.array(center)

    return particle


class TestRemoveParticlesAtBoundary(unittest.TestCase):
    """Test class for the particles left out of the statistics for touching a face."""

    def test_a_particle_is_judged_by_its_extremes(self):
        self.assertTrue(crosses_boundary(disk([0.02, 0.5]), RVE_DIMS))
        self.assertTrue(crosses_boundary(disk([0.98, 0.5]), RVE_DIMS))
        self.assertTrue(crosses_boundary(disk([0.5, 0.01]), RVE_DIMS))
        self.assertFalse(crosses_boundary(disk([0.5, 0.5]), RVE_DIMS))

    def test_the_particles_inside_are_the_ones_kept(self):
        inside = [disk([0.5, 0.5]), disk([0.3, 0.7])]
        outside = [disk([0.02, 0.5]), disk([0.98, 0.5]), disk([0.5, 0.99])]
        kept = remove_particles_at_boundary(outside + inside, RVE_DIMS)
        self.assertEqual(kept, inside)

    def test_particles_crossing_one_after_the_other_are_both_removed(self):
        kept = remove_particles_at_boundary(
            [disk([0.02, 0.5]), disk([0.98, 0.5]), disk([0.5, 0.5])], RVE_DIMS
        )
        self.assertEqual(len(kept), 1)
        np.testing.assert_allclose(kept[0].position_center, [0.5, 0.5])
        # The list was walked while it was removed from, so the particle after every
        # one taken out was never looked at: the second of these was left in

    def test_a_microstructure_of_particles_inside_is_left_alone(self):
        particles = [disk([0.3, 0.3]), disk([0.5, 0.5]), disk([0.7, 0.7])]
        self.assertEqual(remove_particles_at_boundary(particles, RVE_DIMS), particles)


class TestStatisticalPlots(unittest.TestCase):
    """Test class for the figures of the statistical descriptors."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.radii = np.linspace(0.0, 1.0, 200)

    def axes_of(self, plot):
        """Draw a plot into axes of its own and give them back."""
        figure, axes = matplotlib.pyplot.subplots()
        self.addCleanup(matplotlib.pyplot.close, figure)
        plot(axes=axes)

        return axes

    def test_every_plot_is_labelled(self):
        plots = (
            lambda **k: plot_nearest_neighbor_dist(np.random.rand(50), **k),
            lambda **k: plot_ripleys_k_func(np.random.rand(200), self.radii, **k),
            lambda **k: plot_two_point_correlation(np.random.rand(200), self.radii, **k),
        )
        for i_ind, i_plot in enumerate(plots):
            with self.subTest(plot=i_ind):
                axes = self.axes_of(i_plot)
                self.assertTrue(axes.get_xlabel())
                self.assertTrue(axes.get_ylabel())
                self.assertTrue(axes.get_title())
        # None of the three carried a label, a title or a legend

    def test_the_lines_read_against_each_other_are_named(self):
        for i_plot in (
            lambda **k: plot_ripleys_k_func(np.random.rand(200), self.radii, **k),
            lambda **k: plot_two_point_correlation(np.random.rand(200), self.radii, **k),
        ):
            with self.subTest(plot=i_plot):
                axes = self.axes_of(i_plot)
                self.assertGreaterEqual(len(axes.get_legend_handles_labels()[1]), 2)

    def test_a_run_of_fewer_points_than_the_smoothing_window_is_plotted(self):
        short = SMOOTHING_WINDOW // 2
        plot_two_point_correlation(
            np.random.rand(short),
            np.linspace(0.0, 1.0, short),
            results_dir=self.temp_dir.name,
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "two_pt_corr.pdf"))
        )
        # The filter was given a window of 41 whatever the number of radii asked for,
        # and raised on a run of fewer

    def test_the_figure_is_saved_under_the_name_given(self):
        plot_ripleys_k_func(
            np.random.rand(200),
            self.radii,
            results_dir=self.temp_dir.name,
            fig_name="ripley",
        )
        self.assertEqual(os.listdir(self.temp_dir.name), ["ripley.pdf"])


if __name__ == "__main__":
    unittest.main()
