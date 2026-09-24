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
    STAT_FILE_NAME,
    crosses_boundary,
    do_stat_analysis,
    remove_particles_at_boundary,
)
from geommicgen.postproc.voronoimetrics.voronoi_analysis import (
    VORONOI_FILE_NAME,
    do_voronoi_analysis,
    flatten_ragged,
)
from geommicgen.tests.helpers import disk_microstructure

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


class TestStatResultsFile(unittest.TestCase):
    """Test class for the file the statistical descriptors are written into."""

    def test_every_descriptor_is_an_array_of_its_own(self):
        with tempfile.TemporaryDirectory() as directory:
            written = do_stat_analysis(
                disk_microstructure(), directory, {"stat_nearest_neighbor"}
            )
            path = os.path.join(directory, "stat_analysis_results", STAT_FILE_NAME)
            self.assertTrue(os.path.exists(path))
            read = np.load(path)
            self.assertEqual(set(read), set(written))
            for i_name, i_values in written.items():
                np.testing.assert_allclose(read[i_name], i_values)
        # Read back with numpy alone, where the descriptors used to be a pickle of a
        # dictionary that only Python could open


class TestVoronoiResultsFile(unittest.TestCase):
    """Test class for the file the Voronoi diagram and its metrics are written into."""

    def test_rows_of_unequal_length_are_laid_end_to_end(self):
        flat, offsets = flatten_ragged([[1, 2, 3], [], [4], [5, 6]])
        np.testing.assert_array_equal(flat, [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(offsets, [0, 3, 3, 4, 6])
        rows = [flat[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
        self.assertEqual(rows, [[1, 2, 3], [], [4], [5, 6]])

    def test_the_diagram_and_its_metrics_are_named_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            do_voronoi_analysis(
                disk_microstructure().particles, [1.0, 1.0], directory
            )
            path = os.path.join(directory, "voronoi_analysis_results", VORONOI_FILE_NAME)
            read = np.load(path)
            for i_name in (
                "vertices",
                "point_region",
                "regions_flat",
                "regions_offsets",
                "ridge_points",
                "ridge_vertices_flat",
                "ridge_vertices_offsets",
                "imts",
                "angles",
                "in_box",
            ):
                self.assertIn(i_name, read)
            self.assertEqual(read["vertices"].shape[1], 2)
            self.assertEqual(len(read["regions_offsets"]), read["point_region"].max() + 2)
        # Read back with numpy alone, where this was a pickle of a list whose length
        # depended on the dimension and which held a live scipy object


if __name__ == "__main__":
    unittest.main()
