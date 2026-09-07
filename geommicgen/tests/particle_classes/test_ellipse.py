"""
Unit tests regarding the Ellipse particle class.
"""
import unittest

import numpy as np
from scipy import integrate
from geommicgen.microstructure.particleclasses import Ellipse


class EllipseTestPartiallyIntersecting_1(unittest.TestCase):
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

    def test_ellipse_intersection_area(self):

        n_samples = 1000

        box = self.rve_dims
        # Saving the RVE dimensions
        diff_in_box = self.ellipse_1.position_center - self.ellipse_2.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current

        points = self.ellipse_1.uniform_sample_ellipse(n_samples=n_samples)
        k_uniform = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
            if point_in:
                # plt.scatter(x, y, c="r", s=1)
                k_uniform += 1
            else:
                pass
                # plt.scatter(x, y, c="k", s=1)

        points = self.ellipse_1.regular_sample_ellipse(n_samples=n_samples)
        k_reg = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
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
                self.ellipse_1.rot_mat.dot([x, y]) + self.ellipse_1.position_center,
                self.rve_dims,
            )
            if pointIn:
                value = 1
            else:
                value = 0
            return value

        A1 = self.ellipse_1.intersection_area(self.ellipse_2, self.rve_dims)
        # # print("exact", A1)
        A2 = self.ellipse_1.volume * k_uniform / n_samples
        # # print("approx", A2)
        A3 = self.ellipse_1.volume * k_reg / n_samples
        # # print("approx_reg", A3)
        A4, _ = integrate.dblquad(
            pointsInside,
            -B,
            B,
            lambda y: -A * np.sqrt(1 - y ** 2 / B ** 2),
            lambda y: A * np.sqrt(1 - y ** 2 / B ** 2),
            epsrel=1,
        )
        # # print(A1, A2, A3, A4)
        # plot_particles_2d(
        #     [self.ellipse_1, self.ellipse_2], self.rve_dims, "", show=True, save=False
        # )
        # # print(np.abs(np.array([(A1 - A2) / A1, (A1 - A3) / A1, (A1 - A4) / A1])))
        self.assertTrue((np.abs(np.array([(A1 - A4) / A1])) < 1e-2).all())


class EllipseTestPartiallyIntersecting_2(unittest.TestCase):
    def setUp(self):

        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.1, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.45, 0.5])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.2, "angle": np.pi / 3},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.5, 0.6])

    def test_ellipse_intersection(self):

        self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

    def test_ellipse_intersection_area(self):

        n_samples = 1000

        box = self.rve_dims
        # Saving the RVE dimensions
        diff_in_box = self.ellipse_1.position_center - self.ellipse_2.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current

        points = self.ellipse_1.uniform_sample_ellipse(n_samples=n_samples)
        k_uniform = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
            if point_in:
                # plt.scatter(x, y, c="r", s=1)
                k_uniform += 1
            else:
                pass
                # plt.scatter(x, y, c="k", s=1)

        points = self.ellipse_1.regular_sample_ellipse(n_samples=n_samples)
        k_reg = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
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
                self.ellipse_1.rot_mat.dot([x, y]) + self.ellipse_1.position_center, box
            )
            if pointIn:
                value = 1
            else:
                value = 0
            return value

        A1 = self.ellipse_1.intersection_area(self.ellipse_2, self.rve_dims)
        # # print("exact", A1)
        A2 = self.ellipse_1.volume * k_uniform / n_samples
        # # print("approx", A2)
        A3 = self.ellipse_1.volume * k_reg / n_samples
        # # print("approx_reg", A3)
        A4, _ = integrate.dblquad(
            pointsInside,
            -B,
            B,
            lambda y: -A * np.sqrt(1 - y ** 2 / B ** 2),
            lambda y: A * np.sqrt(1 - y ** 2 / B ** 2),
            epsrel=1,
        )
        # # print(A1, A2, A3, A4, self.ellipse_1.volume)
        # plot_particles_2d(
        #     [self.ellipse_1, self.ellipse_2], self.rve_dims, "", show=True, save=False
        # )
        # # print(np.abs(np.array([(A1 - A2) / A1, (A1 - A3) / A1, (A1 - A4) / A1])))
        self.assertTrue((np.abs(np.array([(A1 - A4) / A1])) < 1e-2).all())


class EllipseTestPartiallyIntersecting_3(unittest.TestCase):
    def setUp(self):

        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.45, 0.5])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.55, 0.55])

    def test_ellipse_intersection(self):

        self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

    def test_ellipse_intersection_area(self):

        n_samples = 1000

        box = self.rve_dims
        # Saving the RVE dimensions
        diff_in_box = self.ellipse_1.position_center - self.ellipse_2.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current

        points = self.ellipse_1.uniform_sample_ellipse(n_samples=n_samples)
        k_uniform = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
            if point_in:
                # plt.scatter(x, y, c="r", s=1)
                k_uniform += 1
            else:
                pass
                # plt.scatter(x, y, c="k", s=1)

        points = self.ellipse_1.regular_sample_ellipse(n_samples=n_samples)
        k_reg = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
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
                self.ellipse_1.rot_mat.dot([x, y]) + self.ellipse_1.position_center, box
            )
            if pointIn:
                value = 1
            else:
                value = 0
            return value

        A1 = self.ellipse_1.intersection_area(self.ellipse_2, self.rve_dims)
        # # print("exact", A1)
        A2 = self.ellipse_1.volume * k_uniform / n_samples
        # # print("approx", A2)
        A3 = self.ellipse_1.volume * k_reg / n_samples
        # # print("approx_reg", A3)
        A4, _ = integrate.dblquad(
            pointsInside,
            -B,
            B,
            lambda y: -A * np.sqrt(1 - y ** 2 / B ** 2),
            lambda y: A * np.sqrt(1 - y ** 2 / B ** 2),
            epsrel=1,
        )
        # # print(A1, A2, A3, A4, self.ellipse_1.volume)
        # plot_particles_2d(
        #     [self.ellipse_1, self.ellipse_2], self.rve_dims, "", show=True, save=False
        # )
        # # print(np.abs(np.array([(A1 - A2) / A1, (A1 - A3) / A1, (A1 - A4) / A1])))
        self.assertTrue((np.abs(np.array([(A1 - A4) / A1])) < 1e-2).all())


class EllipseTestPartiallyIntersecting_4(unittest.TestCase):
    def setUp(self):

        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"vf": 0.3, "n": 100, "ratio": 2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.99574763, 0.43099855])
        self.ellipse_2 = Ellipse(
            "1",
            {"vf": 0.3, "n": 100, "ratio": 2, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.99636712, 0.47419431])

    def test_ellipse_intersection(self):

        self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

    def test_ellipse_intersection_area(self):

        n_samples = 1000

        box = self.rve_dims
        # Saving the RVE dimensions
        diff_in_box = self.ellipse_1.position_center - self.ellipse_2.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current

        points = self.ellipse_1.uniform_sample_ellipse(n_samples=n_samples)
        k_uniform = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
            if point_in:
                # plt.scatter(x, y, c="r", s=1)
                k_uniform += 1
            else:
                pass
                # plt.scatter(x, y, c="k", s=1)

        points = self.ellipse_1.regular_sample_ellipse(n_samples=n_samples)
        k_reg = 0
        for i_point in points:
            point_in = self.ellipse_2.point_inside(i_point - diff_nearest_other, box)
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
                self.ellipse_1.rot_mat.dot([x, y]) + self.ellipse_1.position_center, box
            )
            if pointIn:
                value = 1
            else:
                value = 0
            return value

        A1 = self.ellipse_1.intersection_area(self.ellipse_2, self.rve_dims)
        # # print("exact", A1)
        A2 = self.ellipse_1.volume * k_uniform / n_samples
        # # print("approx", A2)
        A3 = self.ellipse_1.volume * k_reg / n_samples
        # # print("approx_reg", A3)
        A4, _ = integrate.dblquad(
            pointsInside,
            -B,
            B,
            lambda y: -A * np.sqrt(1 - y ** 2 / B ** 2),
            lambda y: A * np.sqrt(1 - y ** 2 / B ** 2),
            epsrel=1,
        )
        # # print(A1, A2, A3, A4, self.ellipse_1.volume)
        # plot_particles_2d(
        #     [self.ellipse_1, self.ellipse_2], self.rve_dims, "", show=True, save=False
        # )
        # # print(np.abs(np.array([(A1 - A2) / A1, (A1 - A3) / A1, (A1 - A4) / A1])))
        self.assertTrue(A1 < 1e-4)


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
        intersection = self.ellipse_1.intersection(self.ellipse_2, self.rve_dims)
        self.assertTrue(not intersection)

    def test_ellipse_intersection_2_pts(self):
        self.ellipse_1.position_center = np.array([0.4, 0.5])
        self.ellipse_2.position_center = np.array([0.5, 0.5])
        (
            intersection_length,
            intersection_dir,
        ) = self.ellipse_1.intersection_length_mink_diff(self.ellipse_2, self.rve_dims)
        self.ellipse_2.position_center += intersection_length * intersection_dir
        intersection = self.ellipse_1.intersection(self.ellipse_2, self.rve_dims)
        self.assertTrue(not intersection)


class EllipseTestIntersectionLength_2(unittest.TestCase):
    def test_ellipse_intersection_2_pts_pbc(self):
        rve_dims = [1.0, 1.0]

        ellipse_1 = Ellipse(
            "1",
            {
                "major_axis": 0.1888139487765259,
                "minor_axis": 0.09440697438826295,
                "angle": 0,
            },
            rve_dims,
        )
        ellipse_2 = Ellipse(
            "1",
            {
                "major_axis": 0.1888139487765259,
                "minor_axis": 0.09440697438826295,
                "angle": 0,
            },
            rve_dims,
        )

        ellipse_1.position_center = np.array([0.97394601, 0.74189084])
        ellipse_2.position_center = np.array([0.99764352, 0.64842395])
        (
            intersection_length,
            unit_vector,
        ) = ellipse_1.intersection_length_mink_diff(ellipse_2, rve_dims)
        ellipse_2.position_center += intersection_length * unit_vector
        intersection = ellipse_1.intersection_gjk(ellipse_2, rve_dims)
        self.assertTrue(not intersection)


class EllipseTestIntersectionLength_3(unittest.TestCase):
    def test_ellipse_intersection_4_pts(self):
        rve_dims = [1, 1]
        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0.3}, rve_dims
        )
        ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": np.pi / 2 + 0.3},
            rve_dims,
        )
        ellipse_1.position_center = np.array([0.5, 0.5])
        ellipse_2.position_center = np.array([0.5, 0.5])
        (
            intersection_length,
            unit_vector,
        ) = ellipse_1.intersection_length_mink_diff(ellipse_2, rve_dims)
        ellipse_2.position_center += intersection_length * unit_vector
        intersection = ellipse_1.intersection_gjk(ellipse_2, rve_dims)
        self.assertTrue(not intersection)


class EllipseTestIntersectionLength_4(unittest.TestCase):
    def test_ellipse_intersection_4_pts_2(self):
        rve_dims = [1, 1]
        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0.5}, rve_dims
        )
        ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": np.pi / 2},
            rve_dims,
        )
        ellipse_1.position_center = np.array([0.5, 0.5])
        ellipse_2.position_center = np.array([0.5, 0.5])
        (
            intersection_length,
            unit_vector,
        ) = ellipse_1.intersection_length_mink_diff(ellipse_2, rve_dims)
        ellipse_1.position_center += intersection_length * unit_vector
        intersection = ellipse_1.intersection_gjk(ellipse_2, rve_dims)
        self.assertTrue(not intersection)


class EllipseTest(unittest.TestCase):
    def test_support_ellipse(self):
        rve_dims = [1, 1]
        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0.5}, rve_dims
        )
        ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": np.pi / 2},
            rve_dims,
        )
        ellipse_1.position_center = np.array([0.5, 0.5])
        ellipse_2.position_center = np.array([0.5, 0.501])
        for i_theta in np.linspace(0, 2 * np.pi, 10):
            search_dir = np.array([np.cos(i_theta), np.sin(i_theta)])
            pt_1 = ellipse_1.support_function(search_dir)
            self.assertTrue(ellipse_1.point_inside(pt_1[0:2], rve_dims))
            pt_2 = ellipse_2.support_function(search_dir)
            self.assertTrue(ellipse_2.point_inside(pt_2[0:2], rve_dims))


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
        self.assertTrue(np.all(np.abs(pt_1[0:2] - np.array([0.7, 0.5])) < 1e-4))
        self.assertTrue(np.all(np.abs(pt_3[0:2] - np.array([0.5, 0.55])) < 1e-4))
