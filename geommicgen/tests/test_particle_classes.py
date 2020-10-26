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
from microstructure.particle_classes import (
    Ellipsoid,
    Ellipse,
    Cylinder,
    Particle,
    Disk,
    Sphere,
)
from postproc.plotfuncs.plotting_functions import plot_particles_3d, plot_particles_2d
import pickle


class TestEllipsoid(unittest.TestCase):
    def test_support_function(self):
        """Check supprt function."""
        phase = "1"
        descriptors = {
            "axis_1": 0.1,
            "axis_2": 0.15,
            "axis_3": 0.05,
            "rot_axis_comp_x": 0,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 1,
            "angle": np.pi / 4,
            "n": 1,
        }
        rve_dims = [1, 1, 1]
        ellip = Ellipsoid(phase, descriptors, rve_dims)
        ellip.position_center = np.array([0.5, 0.5, 0.5])
        direction = np.array([0, 0, 1])
        furthest_point = ellip.support_function(direction)
        self.assertTrue(
            all(np.abs(furthest_point - np.array([0.5, 0.5, 0.525])) < 1e-4)
        )

        direction_2 = np.array([1, 1, 0])
        furthest_point_2 = ellip.support_function(direction_2)
        self.assertTrue(
            all(
                np.abs(
                    furthest_point_2
                    - np.array([0.5 + 0.05 / np.sqrt(2), 0.5 + 0.05 / np.sqrt(2), 0.5])
                )
                < 1e-4
            )
        )

        direction_3 = np.array([0.3, 0.15, 0])
        furthest_point_3 = ellip.support_function(direction_3)


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

    def test_ellipsoid_intersection_volume_general_monte_carlo(self):
        """Checking the computed intersection volume.

        Computed using a random distribution of points and a grid."""
        begin_1 = time.time()
        overlap_volume_1 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="random"
        )
        time_1 = time.time() - begin_1
        begin_2 = time.time()
        (
            overlap_volume_2,
            error_estimate,
        ) = self.ellipsoid_1.intersection_area_monte_carlo(
            self.ellipsoid_2,
            self.rve_dims,
        )
        time_2 = time.time() - begin_2
        print("time", time_1, time_2)
        print("error_estimate", error_estimate)
        print("overlap", overlap_volume_1, overlap_volume_2)
        # v_ellipsoid_2 = ellipsoid_2.volume
        # print(overlap_volume_1, end_1 - start_1, overlap_volume_2, end_2 - start_2)
        self.assertTrue(np.abs(overlap_volume_1 - overlap_volume_2) < 1e-2)

    def test_intersection_gjk(self):
        intersection, _ = self.ellipsoid_1.intersection_gjk(
            self.ellipsoid_2, self.rve_dims
        )
        self.assertTrue(intersection)

    def test_intersection_gjk_2(self):
        previous_mic_path = (
            "/home/jose/Documents/code/test_runs/3D/cylindrs_94/mic_0/mic.mic"
        )
        info_previous_sample = pickle.load(open(previous_mic_path, "rb"))
        # No need to generate a new microstructure. Using a previous microstructure.
        current_sample = info_previous_sample["microstructure"]
        current_mic_generator = info_previous_sample["generation_method"]
        trouble_pair = []
        for i_particle in current_sample.particles:
            if (
                i_particle.position_center[0] < 0.25
                and 0.25 < i_particle.position_center[1] < 0.75
                and i_particle.position_center[2] > 0.75
            ):
                trouble_pair.append(i_particle)
                print(vars(i_particle))
        intersection, overlap_length = trouble_pair[0].intersection_gjk(
            trouble_pair[1], [1, 1, 1]
        )
        self.assertTrue(intersection)


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


class EllipseTestIntersectionLength(unittest.TestCase):
    def setUp(self):

        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0}, self.rve_dims
        )
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )

    def test_ellipse_intersection_not(self):
        self.ellipse_1.position_center = np.array([0.8, 0.9])
        self.ellipse_2.position_center = np.array([0.1, 0.2])
        intersection_length = self.ellipse_1.intersection_length(
            self.ellipse_2, self.rve_dims
        )
        self.assertTrue(intersection_length == 0)

    def test_ellipse_intersection_2_pts(self):
        self.ellipse_1.position_center = np.array([0.4, 0.5])
        self.ellipse_2.position_center = np.array([0.5, 0.5])
        intersection_length = self.ellipse_1.intersection_length(
            self.ellipse_2, self.rve_dims
        )
        self.assertTrue(np.abs(intersection_length - 0.3) < 1e-4)

    def test_ellipse_intersection_4_pts(self):
        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0.3}, self.rve_dims
        )
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": np.pi / 2 + 0.3},
            self.rve_dims,
        )
        self.ellipse_1.position_center = np.array([0.5, 0.5])
        self.ellipse_2.position_center = np.array([0.5, 0.5])
        intersection_length = self.ellipse_1.intersection_length(
            self.ellipse_2, self.rve_dims
        )
        self.assertTrue(np.abs(intersection_length - 0.1) < 1e-4)

    def test_ellipse_intersection_inside(self):
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.05, "minor_axis": 0.01, "angle": np.pi / 2},
            self.rve_dims,
        )
        self.ellipse_1.position_center = np.array([0.5, 0.5])
        self.ellipse_2.position_center = np.array([0.5, 0.5])
        intersection_length = self.ellipse_1.intersection_length(
            self.ellipse_2, self.rve_dims
        )
        self.assertTrue(np.abs(intersection_length - 0.05) < 1e-4)


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

    def test_descriptors_ratio(self):
        """Check if the correct geometrical descriptors are obtained speficiyng the ratio."""
        phase = "1"
        descriptors = {
            "ratio": 2,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder(phase, descriptors, rve_dims)
        self.assertEqual(cyl.length, 0.2)
        self.assertTrue(np.abs(cyl.r_cyl - 0.1) < 1e-4)
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

    def test_support_function(self):
        """Check supprt function."""
        phase = "1"
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder(phase, descriptors, rve_dims)
        cyl.position_center = np.array([0.5, 0.5, 0.5])
        direction = np.array([1, 0, 0])
        furthest_point = cyl.support_function(direction)
        self.assertTrue(all(np.abs(furthest_point - np.array([0.6, 0.5, 0.6])) < 1e-4))
        cyl_2 = Cylinder(phase, descriptors, rve_dims)
        cyl_2.position_center = np.array([0.6, 0.7, 0.5])
        direction_2 = np.array([0, 1, -1])
        furthest_point_2 = cyl_2.support_function(direction_2)
        self.assertTrue(
            all(np.abs(furthest_point_2 - np.array([0.6, 0.8, 0.4])) < 1e-4)
        )

    def test_intersection_cylinder_cylinder_non_intersecting(self):
        """Test for intersection_cylinder_cylinder with non-intersecting cylinder."""
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.9, 0.5])
        intersection, _ = cyl_1.intersection_cylinder_cylinder(cyl_2, rve_dims)
        self.assertTrue(not intersection)

    def test_intersection_cylinder_cylinder_intersecting_cc1_1(self):
        """Test for intersection_cylinder_cylinder with intersecting cylinder, type cc1."""
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.05,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": np.pi / 2,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.6, 0.5])
        intersection, _ = cyl_1.intersection_cylinder_cylinder(cyl_2, rve_dims)
        self.assertTrue(intersection)

    def test_intersection_cylinder_cylinder_intersecting_cc1_2(self):
        """Test for intersection_cylinder_cylinder with intersecting cylinder, type cc1."""
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.65, 0.5, 0.5])
        intersection, _ = cyl_1.intersection_cylinder_cylinder(cyl_2, rve_dims)
        self.assertTrue(intersection)

    def test_intersection_cylinder_cylinder_intersecting_cd_1(self):
        """Test for intersection_cylinder_cylinder with intersecting cylinder, type cd."""
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.05,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.65, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": np.pi / 4,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.65, 0.5, 0.65])
        intersection, _ = cyl_1.intersection_cylinder_cylinder(cyl_2, rve_dims)
        self.assertTrue(intersection)

    def test_intersection_top_disks(self):
        """Test for intersection_cylinder_cylinder with intersecting cylinder, type d1."""
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.05,
            "length": 0.1,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": np.pi / 4,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.5, 0.65])
        intersection, _ = cyl_1.intersection_cylinder_cylinder(cyl_2, rve_dims)
        self.assertTrue(intersection)


class TestGJKIntersection(unittest.TestCase):
    # @unittest.skip
    def test_two_intersecting_cylinders(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": np.pi / 2,
            "polar_angle": np.pi / 3,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.55, 0.5])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(intersection)

    # @unittest.skip
    def test_two_non_intersecting_cylinders(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": np.pi / 2,
            "polar_angle": np.pi / 2,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.9, 0.5])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(not intersection)

    # @unittest.skip
    def test_two_barely_parallel_intersecting_cylinders(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.15,
            "length": 0.3,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.5, 0.7, 0.5])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(intersection)

    # @unittest.skip
    def test_two_barely_non_parallel_intersecting_cylinders(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.060,
            "length": 0.11,
            "n": 1,
            "azimuth_angle": -0.20930148133804824,
            "polar_angle": -0.11341128400795103,
        }
        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.98059724, 0.13955079, 0.08649996])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.060000000000000005,
            "length": 0.11,
            "n": 1,
            "azimuth_angle": -0.09742604257341621,
            "polar_angle": 0.34888461357284584,
        }

        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.88500169, 0.14251379, 0.119088])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(intersection)

    # @unittest.skip
    def test_two_barely_non_parallel_intersecting_cylinders_2(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.060000000000000005,
            "length": 0.11,
            "azimuth_angle": 0.5169283751534829,
            "polar_angle": 0.0508044080346002,
            "n": 1,
        }

        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.18788528, 0.80803687, 0.06790679])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.060000000000000005,
            "length": 0.11,
            "azimuth_angle": 0.5337399441494118,
            "polar_angle": 0.027687637482383134,
            "n": 1,
        }

        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.32783983, 0.79778433, 0.01264376])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(not intersection)

    # @unittest.skip
    def test_two_intersecting_disks(self):
        rve_dims = [1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.1,
            "n": 1,
        }
        disk_1 = Disk(phase_1, descriptors_1, rve_dims)
        disk_1.position_center = np.array([0.5, 0.55])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.15,
            "n": 1,
        }
        disk_2 = Disk(phase_2, descriptors_2, rve_dims)
        disk_2.position_center = np.array([0.5, 0.55])
        intersection, _ = disk_1.intersection_gjk(disk_2, rve_dims)
        self.assertTrue(intersection)

    # @unittest.skip
    def test_two_non_intersecting_disks(self):
        rve_dims = [1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.1,
            "n": 1,
        }
        disk_1 = Disk(phase_1, descriptors_1, rve_dims)
        disk_1.position_center = np.array([0.5, 0.85])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.15,
            "n": 1,
        }
        disk_2 = Disk(phase_2, descriptors_2, rve_dims)
        disk_2.position_center = np.array([0.5, 0.55])
        intersection, _ = disk_1.intersection_gjk(disk_2, rve_dims)
        self.assertTrue(not intersection)

    # @unittest.skip
    def test_two_parallel_non_intersecting_cylinders_3(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r_cyl": 0.060000000000000005,
            "length": 0.11,
            "azimuth_angle": 0.39385173716150756,
            "polar_angle": 0.35855785155877495,
            "n": 1,
        }

        cyl_1 = Cylinder(phase_1, descriptors_1, rve_dims)
        cyl_1.position_center = np.array([0.42469612, 0.24732879, 0.87700376])

        phase_2 = "1"
        descriptors_2 = {
            "r_cyl": 0.060000000000000005,
            "length": 0.11,
            "azimuth_angle": 0.39385173716150756,
            "polar_angle": 0.35855785155877495,
            "n": 1,
        }

        cyl_2 = Cylinder(phase_2, descriptors_2, rve_dims)
        cyl_2.position_center = np.array([0.3770918, 0.12216294, 0.92497031])
        intersection, _ = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(not intersection)


class testNearest(unittest.TestCase):
    def test_triangle_xy_plane(self):

        pt_1 = np.array([0.5, 0.2, 0])
        pt_2 = np.array([0.4, 0.3, 0])
        pt_3 = np.array([-0.1, 0.2, 0])
        simplex = [pt_1, pt_2, pt_3]
        simplex_new, search_direction = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_1))
        self.assertTrue(all(simplex_new[1] == pt_3))

        pt_1 = np.array([0.5, 0.2, 0])
        pt_2 = np.array([0.4, 0.3, 0])
        pt_3 = np.array([-0.1, -0.2, 0])
        simplex = [pt_1, pt_2, pt_3]
        simplex_new, search_direction = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_2))
        self.assertTrue(all(simplex_new[1] == pt_3))

        pt_1 = np.array([0.5, 0.2, 0])
        pt_2 = np.array([0.4, 0.3, 0])
        pt_3 = np.array([0.1, 0.1, 0])
        simplex = [pt_1, pt_2, pt_3]
        simplex_new, search_direction = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_3))

        pt_1 = np.array([0.5, 0, 0])
        pt_2 = np.array([0, 0.3, 0])
        pt_3 = np.array([-0.01, -0.01, -0.01])
        simplex = [pt_1, pt_2, pt_3]
        simplex_new, search_direction = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_3))
        self.assertTrue(all(simplex_new[1] == pt_2))
        self.assertTrue(all(simplex_new[2] == pt_1))

    def test_tetrahedron(self):

        pt_1 = np.array([0.5, 0.2, -0.1])
        pt_2 = np.array([-0.2, 0.3, 0.5])
        pt_3 = np.array([0.2, 0.1, 0.3])
        pt_4 = np.array([-0.01, -0.01, -0.01])
        simplex = [pt_1, pt_2, pt_3, pt_4]
        simplex_new, _ = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_1))
        self.assertTrue(all(simplex_new[1] == pt_2))
        self.assertTrue(all(simplex_new[2] == pt_3))
        self.assertTrue(all(simplex_new[3] == pt_4))

        pt_1 = np.array([0.5, 0.2, -0.1])
        pt_2 = np.array([-0.2, 0.3, -0.5])
        pt_3 = np.array([0.2, 0.1, 0.3])
        pt_4 = np.array([-0.1, 0.1, 0.1])
        simplex = [pt_1, pt_2, pt_3, pt_4]
        simplex_new, search_direction = Particle.nearest_simplex(simplex)
        self.assertTrue(all(simplex_new[0] == pt_1))
        self.assertTrue(all(simplex_new[1] == pt_2))
        self.assertTrue(all(simplex_new[2] == pt_4))


class TestNearestPeriodImage(unittest.TestCase):
    """Test the staticmethod nearest_periodic_image"""

    def test_2d(self):
        """Test nearest_periodic_image in 2D."""
        box = [2, 1]
        point_1 = np.array([0.9, 0.9])
        point_2 = np.array([0.9, 0.1])
        nearest_image_pt_1 = Particle.nearest_periodic_image(point_1, point_2, box)
        self.assertTrue(all(np.abs(nearest_image_pt_1 - np.array([0.9, -0.1])) < 1e-4))

    def test_3d(self):
        """Test nearest_periodic_image in 3D."""
        box = [1, 1, 1]
        point_1 = np.array([0.5, 0.1, 0.9])
        point_2 = np.array([0.9, 0.9, 0.9])
        nearest_image_pt_1 = Particle.nearest_periodic_image(point_1, point_2, box)
        self.assertTrue(
            all(np.abs(nearest_image_pt_1 - np.array([0.5, 1.1, 0.9])) < 1e-4)
        )


class TestSupportFuntionEllipse(unittest.TestCase):
    """Test the support function of the Ellipse."""

    def test_support_function_ellipse_1(self):
        rve_dims = [1.0, 1.0]

        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0}, rve_dims
        )
        ellipse_1.position_center = np.array([0.5, 0.5])
        pt_1 = ellipse_1.support_function(np.array([1, 0]))
        pt_3 = ellipse_1.support_function(np.array([0, 1]))
        self.assertTrue(np.all(np.abs(pt_1 - np.array([0.7, 0.5])) < 1e-4))
        self.assertTrue(np.all(np.abs(pt_3 - np.array([0.5, 0.55])) < 1e-4))


class TestSalnikovSphereCylinder(unittest.TestCase):
    """Test the intersection function from Salnikov for spheres and cylinders."""

    def setUp(self):
        self.rve_dims = [1, 1, 1]
        self.sphere = Sphere("1", {"r": 0.1, "n": 1}, self.rve_dims)
        self.cylinder = Cylinder(
            "1",
            {
                "r_cyl": 0.2,
                "length": 0.4,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
                "n": 1,
            },
            self.rve_dims,
        )

    def test_intersect_top(self):
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.7001, 0.5, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(intersection)
        print(intersection_length)
        self.assertTrue(np.abs(intersection_length - 0.1) < 1e-4)

    @unittest.skip("Incomplete")
    def test_intersect_lateral(self):
        pass

    @unittest.skip("Incomplete")
    def test_intersect_inside(self):
        pass


class TestPointInsideCylinder(unittest.TestCase):
    """Test the point_inside function for the cylinder."""

    def setUp(self):
        self.rve_dims = [1, 1, 1]
        self.cylinder = Cylinder(
            "1",
            {
                "r_cyl": 0.2,
                "length": 0.4,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
                "n": 1,
            },
            self.rve_dims,
        )
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])

    def test_point_inside_in(self):
        point_inside = self.cylinder.point_inside(np.array([0.5, 0.5, 0.5]))
        self.assertTrue(point_inside)

    def test_point_inside_out(self):
        point_inside = self.cylinder.point_inside(np.array([0.75, 0.5, 0.5]))
        self.assertTrue(not point_inside)


class TestIntegrationCylinder(unittest.TestCase):
    """Test the Monte Carlo integration for cylinders."""

    def test_cylinder_inside(self):
        """The cylinder is completly inside an ellipsoid."""
        rve_dims = [1, 1, 1]
        cylinder = Cylinder(
            "1",
            {
                "r_cyl": 0.3,
                "length": 0.8,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
                "n": 1,
            },
            rve_dims,
        )
        cylinder.position_center = np.array([0.5, 0.5, 0.5])
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 1,
                "axis_2": 1,
                "axis_3": 1,
                "rot_axis_comp_x": np.sqrt(3) / 3,
                "rot_axis_comp_y": np.sqrt(3) / 3,
                "rot_axis_comp_z": np.sqrt(3) / 3,
                "angle": 0,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        intersection_volume, error_estimate = ellipsoid.intersection_area_monte_carlo(
            cylinder, rve_dims, tol=1
        )
        print(
            "error_estimate_2",
            error_estimate,
            intersection_volume,
            cylinder.volume,
            ((intersection_volume - cylinder.volume) / cylinder.volume) * 100,
        )
        self.assertTrue(
            np.abs((intersection_volume - cylinder.volume) / cylinder.volume) * 100 < 1
        )

    def test_cylinder_outside(self):
        """An Ellipsoid is completly inside the Cylinder."""
        rve_dims = [1, 1, 1]
        cylinder = Cylinder(
            "1",
            {
                "r_cyl": 0.5,
                "length": 0.8,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
                "n": 1,
            },
            rve_dims,
        )
        cylinder.position_center = np.array([0.5, 0.5, 0.5])
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.75,
                "axis_2": 0.7,
                "axis_3": 0.6,
                "rot_axis_comp_x": np.sqrt(3) / 3,
                "rot_axis_comp_y": np.sqrt(3) / 3,
                "rot_axis_comp_z": np.sqrt(3) / 3,
                "angle": 0,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        intersection_volume, error_estimate = cylinder.intersection_area_monte_carlo(
            ellipsoid, rve_dims, tol=1
        )
        (
            intersection_volume_2,
            error_estimate_2,
        ) = cylinder.intersection_area_monte_carlo(ellipsoid, rve_dims, tol=1)
        print("error_estimate_1", error_estimate, intersection_volume, ellipsoid.volume)
        print(
            "error_estimate_2",
            error_estimate_2,
            intersection_volume_2,
            ellipsoid.volume,
        )
        self.assertTrue(
            np.abs((intersection_volume - ellipsoid.volume) / ellipsoid.volume) * 100
            < 1
        )
