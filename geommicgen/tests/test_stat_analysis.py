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
    ripleys_k_edge_correction,
    ripleys_k_func,
    two_point_correlation,
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


class TestSeededDescriptors(unittest.TestCase):
    """Test class for an analysis giving the same numbers on every run."""

    def estimate(self, seed):
        """Estimate the two point correlation of a fixture, from a seed."""
        np.random.seed(seed)

        return two_point_correlation(
            disk_microstructure(), n_samples=200, n_points=4
        )[0]
        # Few samples and few radii: this is about which numbers are drawn, not about
        # the estimate being any good

    def test_the_same_seed_gives_the_same_estimate(self):
        np.testing.assert_array_equal(self.estimate(7), self.estimate(7))

    def test_another_seed_gives_another_estimate(self):
        self.assertFalse(np.array_equal(self.estimate(7), self.estimate(8)))

    def test_left_to_itself_it_draws_afresh(self):
        np.random.seed(3)
        first, _ = two_point_correlation(disk_microstructure(), n_samples=200, n_points=4)
        second, _ = two_point_correlation(disk_microstructure(), n_samples=200, n_points=4)
        self.assertFalse(np.array_equal(first, second))
        # Which is why the seed is worth having: the same microstructure used to give
        # numbers a fraction of a percent apart on every run

    def test_the_analysis_seeds_the_draws_before_running_the_descriptors(self):
        def drawn_after(seed):
            """Run an analysis that draws nothing, and give the next number drawn."""
            np.random.seed(101)
            with tempfile.TemporaryDirectory() as directory:
                do_stat_analysis(
                    disk_microstructure(),
                    directory,
                    {"stat_nearest_neighbor"},
                    seed=seed,
                )

            return np.random.uniform()

        self.assertEqual(drawn_after(7), drawn_after(7))
        self.assertNotEqual(drawn_after(7), drawn_after(8))
        # The nearest neighbour distances draw nothing, so what is drawn afterwards
        # comes from the seed the analysis was given, not from the one set before it


class TestRipleysKEdgeCorrection(unittest.TestCase):
    """Test class for the share of a disk around a particle that is inside the box."""

    BOX = [1.0, 1.7]

    def correction(self, center, radius):
        """The share of one disk inside the box."""
        return ripleys_k_edge_correction(
            np.asarray([center], dtype=float), np.asarray([radius]), self.BOX
        )[0]

    def test_a_disk_inside_the_box_is_whole(self):
        self.assertAlmostEqual(self.correction([0.5, 0.85], 0.2), 1.0)

    def test_a_disk_touching_one_side_is_the_circle_less_a_segment(self):
        radius, distance = 0.3, 0.2
        angle = 2 * np.arccos(distance / radius)
        segment = radius**2 * (angle - np.sin(angle)) / 2
        expected = 1 - segment / (np.pi * radius**2)
        self.assertAlmostEqual(self.correction([distance, 0.85], radius), expected)

    def test_a_disk_around_a_corner_is_a_quarter_of_it(self):
        self.assertAlmostEqual(self.correction([0.0, 0.0], 0.2), 0.25)
        # The centre on the corner leaves one of the four quarters inside

    def test_it_agrees_with_throwing_points_at_the_box(self):
        generator = np.random.default_rng(0)
        box = np.asarray(self.BOX)
        for _ in range(8):
            center = generator.uniform(0.05, 0.95, 2) * box
            radius = generator.uniform(0.05, 1.2)
            with self.subTest(center=center, radius=radius):
                angles = generator.uniform(0, 2 * np.pi, 200000)
                radii = radius * np.sqrt(generator.uniform(0, 1, 200000))
                points = center + np.stack(
                    [radii * np.cos(angles), radii * np.sin(angles)], axis=1
                )
                estimated = np.mean(np.all((points >= 0) & (points <= box), axis=1))
                self.assertAlmostEqual(
                    self.correction(center, radius), estimated, places=2
                )
        # The correction used to be this estimate, of two hundred points, computed for
        # every pair of particles; it is the area itself now


class TestRipleysKFunction(unittest.TestCase):
    """Test class for Ripley's K function itself."""

    def test_it_is_not_computed_in_three_dimensions(self):
        from geommicgen.tests.helpers import sphere_microstructure

        with self.assertRaises(NotImplementedError) as context:
            ripleys_k_func(sphere_microstructure())
        self.assertIn("two dimensional", str(context.exception))
        # It raised on a broadcast, having reached the plotting of a two dimensional
        # microstructure, and its three dimensional branch called the number pi

    def test_it_grows_with_the_radius_and_starts_at_nothing(self):
        values, radii = ripleys_k_func(disk_microstructure(), max_radius=4, n_points=5)
        self.assertEqual(len(values), len(radii))
        self.assertEqual(values[0], 0.0)
        np.testing.assert_array_equal(np.sort(values), values)
        # A pair counted at one radius is counted at every larger one


if __name__ == "__main__":
    unittest.main()
