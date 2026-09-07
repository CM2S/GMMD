"""
Unit tests regarding the Point particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Disk, Ellipse, Point


class TestParticlePoint(unittest.TestCase):
    """Testing the gjk intersection alg for the Point particle."""

    @unittest.expectedFailure
    def test_point_disk(self):
        """Testing intersection between Point and Disk."""
        rve_dims = [1, 1]
        disk = Disk(
            "1",
            {"r": 0.1},
            rve_dims,
        )
        disk.position_center = np.array([0.5, 0.5])
        point = Point(2, "1")
        point.position_center = np.array([0.5, 0.55])
        intersection = disk.intersection_gjk(point, rve_dims)
        overlap_length, _ = disk.intersection_length_mink_diff(point, rve_dims)

        self.assertTrue(np.abs(overlap_length - 0.05) < 1e-4)

        point.position_center = np.array([0.5, 0.7])
        disk.position_center = np.array([0.5, 0.5])
        intersection = disk.intersection_gjk(point, rve_dims)
        overlap_length, _ = disk.intersection_length_mink_diff(point, rve_dims)

        self.assertTrue(np.abs(np.abs(overlap_length) - 0.1) < 1e-4)

        point.position_center = np.array([0.7, 0.7])
        disk.position_center = np.array([0.5, 0.5])
        intersection = disk.intersection_gjk(
            point,
            rve_dims,
        )
        overlap_length, _ = disk.intersection_length_mink_diff(
            point,
            rve_dims,
        )

        self.assertTrue(
            np.abs(np.abs(overlap_length) - (0.2 * np.sqrt(2) - 0.1)) < 1e-4
        )

    def test_point_ellipse(self):
        """Testing intersection between Point and Ellipse."""
        rve_dims = [1, 1]
        ellipse = Ellipse(
            "1",
            {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0},
            rve_dims,
        )
        ellipse.position_center = np.array([0.5, 0.5])
        point = Point(2, "1")
        point.position_center = np.array([0.55, 0.5])
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )

        ellipse.position_center -= overlap_length * unit_vector
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )
        self.assertTrue(np.abs(overlap_length) < 1e-4)

        point.position_center = np.array([0.5, 0.72])
        ellipse.position_center = np.array([0.5, 0.5])
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )

        ellipse.position_center -= overlap_length * unit_vector
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )
        self.assertTrue(np.abs(overlap_length) < 1e-4)

        point.position_center = np.array([0.7, 0.7])
        ellipse.position_center = np.array([0.5, 0.5])
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )

        ellipse.position_center -= overlap_length * unit_vector
        intersection = ellipse.intersection_gjk(point, rve_dims)
        overlap_length, unit_vector = ellipse.intersection_length_mink_diff(
            point, rve_dims
        )
        self.assertTrue(np.abs(overlap_length) < 1e-4)
