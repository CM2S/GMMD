"""
Unit tests regarding the Ellipse particle class.
"""
import unittest

import numpy as np
from scipy import integrate
from geommicgen.microstructure.particleclasses import Ellipse


class TestEllipse(unittest.TestCase):
    "Class to test the Ellipse object __init__, ellipses properties and other more general methods."

    def test_init(self):
        "Tests if all ways of describing an ellipse are supported by the constructor."
        rve_dims = [1, 1]
        with self.subTest("major axix and minor axis supplied"):
            rve_dims = [1,1]
            ellipse = Ellipse(
                "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, rve_dims
            )
            self.assertAlmostEqual(ellipse.major_axis, 0.4)
            self.assertAlmostEqual(ellipse.minor_axis, 0.2)
            self.assertAlmostEqual(ellipse.angle, 0)

        with self.subTest("major axis, n and vf supplied"):
            rve_dims = [1, 1]
            ellipse = Ellipse(
                "1", {"major_axis": 0.4, "angle": 0, "n": 10, "vf": 0.1}, rve_dims
            )
            volume_part = 0.1 * rve_dims[0] * rve_dims[1] / 10
            expected_minor_axis = volume_part / (np.pi * 0.4 * 1 / 4)
            self.assertAlmostEqual(ellipse.major_axis, 0.4)
            self.assertAlmostEqual(ellipse.minor_axis, expected_minor_axis)

        with self.subTest("minor axis, n and vf supplied"):
            rve_dims = [1, 1]
            ellipse = Ellipse(
                "1", {"minor_axis": 0.1, "angle": 0, "n": 10, "vf": 0.1}, rve_dims
            )
            volume_part = 0.1 * rve_dims[0] * rve_dims[1] / 10
            expected_major_axis = volume_part / (np.pi * 0.1 * 1 / 4)
            self.assertAlmostEqual(ellipse.major_axis, expected_major_axis)
            self.assertAlmostEqual(ellipse.minor_axis, 0.1)

        with self.subTest("ratio, n and vf supplied"):
            rve_dims = [1, 1]
            ratio = 2
            ellipse = Ellipse(
                "1", {"ratio": ratio, "angle": 0, "n": 10, "vf": 0.1}, rve_dims
            )
            volume_part = 0.1 * rve_dims[0] * rve_dims[1] / 10
            expected_minor_axis = np.sqrt(volume_part / (np.pi * ratio * 1 / 4))
            expected_major_axis = ratio * expected_minor_axis
            self.assertAlmostEqual(ellipse.major_axis, expected_major_axis)
            self.assertAlmostEqual(ellipse.minor_axis, expected_minor_axis)

        with self.subTest("major axis and ratio"):
            rve_dims = [1, 1]
            ellipse = Ellipse(
                "1", {"major_axis": 0.4, "ratio": 2, "angle": 0, "vf": 0.1}, rve_dims
            )
            self.assertAlmostEqual(ellipse.major_axis, 0.4)
            self.assertAlmostEqual(ellipse.minor_axis, 0.2)

    def test_volume(self):
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, [1,1]
            )
        self.assertEqual(ellipse.volume,np.pi*0.2*0.1)

    def test_properties(self):
        rve_dims = [1, 1]
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, rve_dims
        )
        with self.subTest("eccentricity"):
            self.assertAlmostEqual(ellipse.eccentricity, np.sqrt(1 - 0.2 ** 2 / 0.4 ** 2))
        with self.subTest("radius"):
            self.assertAlmostEqual(ellipse.radius, 0.2)
        with self.subTest("radius_insc"):
            self.assertAlmostEqual(ellipse.radius_insc, 0.1)

    def test_contract_and_dilate(self):
        rve_dims = [1, 1]
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, rve_dims
        )
        with self.subTest("dilate"):
            ellipse.dilate(0.05)
            self.assertAlmostEqual(ellipse.radius, 0.25)
            self.assertAlmostEqual(ellipse.volume, np.pi * 0.25 * 0.15)
        with self.subTest("contract back to the original size"):
            ellipse.contract(0.05)
            self.assertAlmostEqual(ellipse.radius, 0.2)
            self.assertAlmostEqual(ellipse.volume, np.pi * 0.2 * 0.1)

    def test_point_inside(self):
        rve_dims = [1, 1]
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, rve_dims
        )
        ellipse.position_center = np.array([0.5, 0.5])
        with self.subTest("Point inside the ellipse"):
            self.assertTrue(ellipse.point_inside(np.array([0.55, 0.5]), rve_dims))
        with self.subTest("Point outside the ellipse"):
            self.assertTrue(not ellipse.point_inside(np.array([0.9, 0.9]), rve_dims))

    def test_generate_points_on_surface(self):
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, [1, 1]
        )
        ellipse.position_center = np.array([2.0, 3.0])
        with self.subTest("Without erosion"):
            points = ellipse.generate_points_on_surface(4)
            expected = np.array(
                [[0.2, 0.0], [0.0, 0.1], [-0.2, 0.0], [0.0, -0.1]]
            ) + ellipse.position_center
            np.testing.assert_allclose(points, expected, atol=1e-10)

        with self.subTest("With erosion"):
            erosion_thick = 0.03
            points = ellipse.generate_points_on_surface(4, erosion_thick=erosion_thick)
            expected = np.array(
                [
                    [0.2 - erosion_thick, 0.0],
                    [0.0, 0.1 - erosion_thick],
                    [-(0.2 - erosion_thick), 0.0],
                    [0.0, -(0.1 - erosion_thick)],
                ]
            ) + ellipse.position_center
            np.testing.assert_allclose(points, expected, atol=1e-10)

    def test_compute_critical_erosion_thickness(self):
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, [1, 1]
        )
        self.assertAlmostEqual(ellipse.compute_critical_erosion_thickness(), 0.1 ** 2 / 0.2)

    def test_rescale(self):
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, [1, 1]
        )
        ellipse.position_center = np.array([0.3, 0.4])
        ellipse.rescale(2)
        self.assertAlmostEqual(ellipse.major_axis, 0.8)
        self.assertAlmostEqual(ellipse.minor_axis, 0.4)
        np.testing.assert_allclose(ellipse.position_center, np.array([0.6, 0.8]))

    def test_generate_point_inside(self):
        rve_dims = [1, 1]
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, rve_dims
        )
        ellipse.position_center = np.array([0.5, 0.5])
        for _ in range(20):
            self.assertTrue(ellipse.point_inside(ellipse.generate_point_inside(), rve_dims))

    def test_intersection_area_invalid_other_particle(self):
        ellipse = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, [1, 1]
        )
        with self.assertRaises(ValueError):
            ellipse.intersection_area(object(), [1, 1])


class TestEllipseIntersectionArea(unittest.TestCase):

    def test_not_intersecting(self):
        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.3, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.6, 0.8])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.3, "minor_axis": 0.2, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.1, 0.2])

        with self.subTest("Test intersection check"):
            self.assertTrue(not self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( self.ellipse_1.intersection_area(self.ellipse_2,[1,1]), 0 )

    def test_not_intersecting_1_point_in_common(self):
        self.rve_dims = [1.0, 1.0]

        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.3, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        ellipse_1.position_center = np.array([0.6, 0.2])
        ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.3, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )
        ellipse_2.position_center = np.array([0.3, 0.2])

        with self.subTest("Test intersection check"):
            self.assertTrue(not ellipse_1.intersection(ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( ellipse_1.intersection_area(ellipse_2,[1,1]), 0 )
    @unittest.skip("Exact tangency is numerically unstable in intersection_points_ellipses, and thus the case of there being only one intersection point does not happen. In this example, intersection_points_ellipses returns 0 points.")
    def test_not_intersecting_1_point_in_common(self):
        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.2, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.3, 0.5])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.2, "minor_axis": 0.2, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.5, 0.5])

        with self.subTest("Test intersection check"):
            self.assertTrue(not self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( self.ellipse_1.intersection_area(self.ellipse_2,[1,1]), 0 )

    def test_ellipse_inside_other_ellipse_1(self):
        " Ellipse 1 is bigger than ellipse 2"
        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.4, 0.7])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.4, 0.7])

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( self.ellipse_1.intersection_area(self.ellipse_2,[1,1]), np.pi*0.1*0.05 )

    def test_ellipse_inside_other_ellipse_2(self):
        " Ellipse 1 is smaller than ellipse 2"
        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0}, self.rve_dims
        )
        self.ellipse_1.position_center = np.array([0.4, 0.7])
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0},
            self.rve_dims,
        )
        self.ellipse_2.position_center = np.array([0.4, 0.7])

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( self.ellipse_1.intersection_area(self.ellipse_2,[1,1]), np.pi*0.1*0.05 )

    @unittest.skip("Exact tangency is numerically unstable in intersection_points_ellipses, and thus the case of there being only one intersection point does not happen. In this example, intersection_points_ellipses returns 2 points. Moreover, the area obtained is not as expected.")
    def test_ellipse_inside_other_ellipse_3(self):
        " Ellipse 1 and 2 are tangent"
        self.rve_dims = [1.0, 1.0]

        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.2, "angle": 0}, self.rve_dims
        )
        ellipse_1.position_center = np.array([0.4, 0.7])
        ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )
        ellipse_2.position_center = np.array([0.4, 0.65])

        with self.subTest("Test intersection check"):
            self.assertTrue(ellipse_1.intersection(ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( ellipse_1.intersection_area(ellipse_2,[1,1]), np.pi*0.1*0.05 )



    def test_partially_intersecting_1(self):
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

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
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

    def test_partially_intersecting_2(self):
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

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
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

    def test_partially_intersecting_3(self):
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

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
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

    def test_partially_intersecting_4(self):
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

        with self.subTest("Test intersection check"):
            self.assertTrue(self.ellipse_1.intersection(self.ellipse_2, self.rve_dims))

        with self.subTest("Test intersection area"):
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


class TestEllipseIntersectionLength(unittest.TestCase):

    def test_intersection_length(self):
        self.rve_dims = [1.0, 1.0]

        self.ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0}, self.rve_dims
        )
        self.ellipse_2 = Ellipse(
            "1",
            {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0},
            self.rve_dims,
        )

        with self.subTest("Test ellipse intersection not"):
            self.ellipse_1.position_center = np.array([0.8, 0.9])
            self.ellipse_2.position_center = np.array([0.1, 0.2])
            intersection = self.ellipse_1.intersection(self.ellipse_2, self.rve_dims)
            self.assertTrue(not intersection)

        with self.subTest("Test ellipse intersection 2 pts"):
            self.ellipse_1.position_center = np.array([0.4, 0.5])
            self.ellipse_2.position_center = np.array([0.5, 0.5])
            (
                intersection_length,
                intersection_dir,
            ) = self.ellipse_1.intersection_length_mink_diff(self.ellipse_2, self.rve_dims)
            self.ellipse_2.position_center += intersection_length * intersection_dir
            intersection = self.ellipse_1.intersection(self.ellipse_2, self.rve_dims)
            self.assertTrue(not intersection)

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


class TestSupportFuntionEllipse(unittest.TestCase):
    """Test the support function of the Ellipse."""

    def test_support_ellipse_1(self):
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

    def test_support_function_ellipse_2(self):
        rve_dims = [1.0, 1.0]

        ellipse_1 = Ellipse(
            "1", {"major_axis": 0.4, "minor_axis": 0.1, "angle": 0}, rve_dims
        )
        ellipse_1.position_center = np.array([0.5, 0.5])
        pt_1 = ellipse_1.support_function(np.array([1, 0]))
        pt_3 = ellipse_1.support_function(np.array([0, 1]))
        self.assertTrue(np.all(np.abs(pt_1[0:2] - np.array([0.7, 0.5])) < 1e-4))
        self.assertTrue(np.all(np.abs(pt_3[0:2] - np.array([0.5, 0.55])) < 1e-4))
