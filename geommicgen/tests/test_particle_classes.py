"""
Unit tests regarding particle classes.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
import time
from microstructure.particle_classes import Ellipsoid, Ellipse, Cylinder


class EllipsoidTestPartiallyIntersecting(unittest.TestCase):
    """Tests for the Ellipsoid class."""

    def setUp(self):
        self.rve_dims = [1.0, 1.0, 1.0]

        self.ellipsoid_1 = Ellipsoid(
            "1",
            {
                "axis_1": 0.3,
                "axis_2": 0.3,
                "axis_3": 0.2,
                "rot_axis_comp_x": np.sqrt(3) / 3,
                "rot_axis_comp_y": np.sqrt(3) / 3,
                "rot_axis_comp_z": np.sqrt(3) / 3,
                "angle": 0,
            },
            self.rve_dims,
        )
        self.ellipsoid_1.position_center = np.array([0.95, 0.5, 0.5])

        self.ellipsoid_2 = Ellipsoid(
            "1",
            {
                "axis_1": 0.3,
                "axis_2": 0.3,
                "axis_3": 0.3,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1.0,
                "angle": 0,
            },
            self.rve_dims,
        )
        self.ellipsoid_2.position_center = np.array([0.05, 0.5, 0.6])

        box = self.rve_dims
        # Saving the array defining the RVE box
        diff_in_box = (
            self.ellipsoid_1.position_center - self.ellipsoid_2.position_center
        )
        self.diff_nearest_other = box * np.round(diff_in_box / box)
        # Computing the difference vector between the centers of the current sphere and

    def test_ellipsoid_intersection(self):
        """Test if the ellipsoids intersect."""

        self.assertTrue(
            self.ellipsoid_1.intersection_ellipsoid_ellipsoid(
                self.ellipsoid_2, self.diff_nearest_other
            )
        )

    def test_ellipsoid_intersection_volume(self):
        """Checking the computed intersection volume.

        Computed using a random distribution of points and a grid."""
        overlap_volume_1 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="random"
        )
        overlap_volume_2 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="regular"
        )
        # v_ellipsoid_2 = ellipsoid_2.volume
        # print(overlap_volume_1, end_1 - start_1, overlap_volume_2, end_2 - start_2)
        self.assertTrue(np.abs(overlap_volume_1 - overlap_volume_2) < 1e-2)


class EllipseTestPartiallyIntersecting(unittest.TestCase):
    def setUp(self):

        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.6, 0.5])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.2, "angle": np.pi / 3},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.5, 0.6])

    def test_ellipse_intersection(self):

        self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        # ellipse

        # _ = plt.figure()
        #
        # ax = plt.gca()
        #
        # particles = [self.ellipse_1, self.ellipse_2]
        #
        #
        # intersect_pts = np.array(
        #     intersection_points_ellipses(
        #         self.ellipse_1.semi_major_axis,
        #         self.ellipse_1.semi_minor_axis,
        #         self.ellipse_1.position_center,
        #         self.ellipse_1.angle,
        #         self.ellipse_2.semi_major_axis,
        #         self.ellipse_2.semi_minor_axis,
        #         self.ellipse_2.position_center + diff_nearest_other,
        #         self.ellipse_2.angle,
        #     )
        # )
        #
        # intersect_pts_ord = ellipse_1.sort_points_on_ellipse(intersect_pts)
        #
        # for i in range(N):
        #     for j in range(-1, 2):
        #         for k in range(-1, 2):
        #             ellip = mpatches.Ellipse(
        #                 particles[i].position_center + np.array([1 * j, 1 * k]),
        #                 particles[i].major_axis,
        #                 particles[i].minor_axis,
        #                 angle=180 / np.pi * particles[i].angle,
        #                 alpha=0.1,
        #             )
        #             ax.add_artist(ellip)
        #             plt.annotate(xy=particles[i].position_center, s=str(i))
        #             plt.scatter(
        #                 particles[i].position_center[0],
        #                 particles[i].position_center[1],
        #             )
        #             plt.axis([0, 1, 0, 1])

    def test_ellipse_intersection_area(self):

        n_samples = 20

        box = self.rve_dims
        # Saving the RVE dimensions
        diff_in_box = self.ellipse_1.position_center - self.ellipse_2.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current

        points = self.ellipse_1.uniform_sample_ellipse(n_samples=n_samples)
        k_uniform = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other)
            if point_in:
                # plt.scatter(x, y, c="r", s=1)
                k_uniform += 1
            else:
                pass
                # plt.scatter(x, y, c="k", s=1)

        points = self.ellipse_1.regular_sample_ellipse(n_samples=n_samples)
        k_reg = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other)
            if point_in:
                # plt.scatter(x[i_point], y[i_point], c="b", s=1)
                k_reg += 1
            else:
                pass
                # plt.scatter(x[i_point], y[i_point], c="g", s=1)

        A = self.ellipse_1.semi_major_axis
        B = self.ellipse_1.semi_minor_axis

        def pointsInside(x, y):
            pointIn = self.ellipse_2.point_inside(
                self.ellipse_1.rot_mat.dot([x, y]) + self.ellipse_1.position_center
            )
            if pointIn:
                value = 1
            else:
                value = 0
            return value

        A1 = self.ellipse_1.intersection_area(self.ellipse_2, self.rve_dims)
        # print("exact", A1)
        A2 = self.ellipse_1.volume * k_uniform / n_samples
        # print("approx", A2)
        A3 = self.ellipse_1.volume * k_reg / n_samples
        # print("approx_reg", A3)
        A4, _ = integrate.dblquad(
            pointsInside,
            -B,
            B,
            lambda y: -A * np.sqrt(1 - y ** 2 / B ** 2),
            lambda y: A * np.sqrt(1 - y ** 2 / B ** 2),
            epsrel=1,
        )
        self.assertTrue((np.abs(np.array([A1 - A2, A1 - A3, A1 - A4])) < 1e-2).all())

    # print('quad', A4[0])
    # def test_ellipse_area_calc(self):
    #     for i_intr_pt in range(len(intersect_pts_ord)):
    #     midpoint = ellipse_1.midpoint_on_ellipse(
    #     intersect_pts_ord[i_intr_pt],
    #     intersect_pts_ord[np.mod(i_intr_pt + 1, len(intersect_pts_ord))],
    #     )
    #
    #
    #         plt.scatter(midpoint[0], midpoint[1], color='r')
    #
    #     intersect_pts_ord = np.array(intersect_pts_ord)
    #     plt.scatter(intersect_pts_ord[:,0], intersect_pts_ord[:,1])
    #     for i_intr_pt in range(len(intersect_pts_ord)):
    #         plt.annotate(xy = intersect_pts_ord[i_intr_pt,:], s=str(i_intr_pt))
    #
    #
    #     # plt.axis([-1, 2, -1, 2])
    #     plt.show()


class TestCylinder(unittest.TestCase):
    def test_init(self):
        """Check if the attributes were set correctly in __init__."""
        phase = "1"
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.1,
            "n": 2,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder(phase, descriptors, rve_dims)
        self.assertEqual(cyl.length, 0.1)
        self.assertEqual(cyl.r_cyl, 0.1)
        self.assertEqual(cyl.azimuth_angle, 0)
        self.assertEqual(cyl.polar_angle, 0)

    def test_descriptors(self):
        """Check if the correct geometrical descriptors are obtained."""
        phase = "1"
        descriptors = {
            "vf": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder(phase, descriptors, rve_dims)
        self.assertEqual(cyl.length, 0.2)
        self.assertTrue(np.abs(cyl.r_cyl - np.sqrt(0.1 / (0.2 * 10 * np.pi))) < 1e-4)
        self.assertEqual(cyl.azimuth_angle, 0)
        self.assertEqual(cyl.polar_angle, 0)

    def test_volume(self):
        """Check if the volume property is correctly specified."""
        phase = "1"
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder(phase, descriptors, rve_dims)
        self.assertTrue(np.abs(cyl.volume - 0.1 ** 2 * np.pi * 0.2) < 1e-4)

    def test_invalid_inputs_radius(self):
        """Check if the proper exception is raise for negative radius."""
        phase = "1"
        descriptors = {
            "r_cyl": -0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        with self.assertRaises(ValueError):
            _ = Cylinder(phase, descriptors, rve_dims)

    def test_invalid_inputs_length(self):
        """Check if the proper exception is raise for negative length."""
        phase = "1"
        descriptors = {
            "r_cyl": 0.1,
            "length": -0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        with self.assertRaises(ValueError):
            _ = Cylinder(phase, descriptors, rve_dims)

    # def test_intersection_parallel
