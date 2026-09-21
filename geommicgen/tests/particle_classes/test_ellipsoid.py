"""
Unit tests regarding the Ellipsoid particle class.
"""
import unittest
from unittest.mock import Mock

import numpy as np
import time
from geommicgen.microstructure.particleclasses import Ellipsoid, Cylinder


class TestEllipsoid(unittest.TestCase):
    """Tests concerning Ellipsoids"""

    def setUp(self):
        "Creates a ellipsoid with position_center assigned. It is used in some of the tests."
        self.rve_dims = [1, 1, 1]
        self.rotation_descriptors = {
            "rot_axis_comp_x": 0,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 1,
            "angle": 0,
        }
        self.ellipsoid = Ellipsoid(
                "1",
                {"axis_1": 0.4, "axis_2": 0.3, "axis_3": 0.2, **self.rotation_descriptors},
                self.rve_dims,
            )
        
        self.ellipsoid.position_center = [0.5,0.5,0.5]


    def test_init(self):
        "Tests if all ways of describing an ellipsoid are supported by the constructor."

        with self.subTest("axis_1, axis_2 and axis_3 supplied"):
            ellipsoid = Ellipsoid(
                "1",
                {"axis_1": 0.4, "axis_2": 0.3, "axis_3": 0.2, **self.rotation_descriptors},
                self.rve_dims,
            )
            self.assertAlmostEqual(ellipsoid.axis_1, 0.4)
            self.assertAlmostEqual(ellipsoid.axis_2, 0.3)
            self.assertAlmostEqual(ellipsoid.axis_3, 0.2)
            self.assertAlmostEqual(ellipsoid.angle, 0)

        with self.subTest("ratio_12, ratio_13, n and vf supplied"):
            ratio_12 = 2
            ratio_13 = 4
            ellipsoid = Ellipsoid(
                "1",
                {
                    "ratio_12": ratio_12,
                    "ratio_13": ratio_13,
                    "n": 10,
                    "vf": 0.1,
                    **self.rotation_descriptors,
                },
                self.rve_dims,
            )
            volume_part = 0.1 * self.rve_dims[0] * self.rve_dims[1] * self.rve_dims[2] / 10
            expected_axis_1 = np.cbrt(
                volume_part * ratio_12 * ratio_13 * 8 / (np.pi * 4 / 3)
            )
            expected_axis_2 = expected_axis_1 / ratio_12
            expected_axis_3 = expected_axis_1 / ratio_13
            self.assertAlmostEqual(ellipsoid.axis_1, expected_axis_1)
            self.assertAlmostEqual(ellipsoid.axis_2, expected_axis_2)
            self.assertAlmostEqual(ellipsoid.axis_3, expected_axis_3)

        with self.subTest("ratio_12, ratio_13 and axis_1 supplied"):
            ratio_12 = 2
            ratio_13 = 4
            ellipsoid = Ellipsoid(
                "1",
                {
                    "axis_1": 0.4,
                    "ratio_12": ratio_12,
                    "ratio_13": ratio_13,
                    **self.rotation_descriptors,
                },
                self.rve_dims,
            )
            self.assertAlmostEqual(ellipsoid.axis_1, 0.4)
            self.assertAlmostEqual(ellipsoid.axis_2, 0.4 / ratio_12)
            self.assertAlmostEqual(ellipsoid.axis_3, 0.4 / ratio_13)

        with self.subTest("ratio_321 and semi_axis_1 supplied"):
            ratio_321 = 0.5
            semi_axis_1 = 0.2
            ellipsoid = Ellipsoid(
                "1",
                {
                    "semi_axis_1": semi_axis_1,
                    "ratio_321": ratio_321,
                    **self.rotation_descriptors,
                },
                self.rve_dims,
            )
            expected_axis_1 = semi_axis_1*2
            self.assertAlmostEqual(ellipsoid.axis_1, expected_axis_1)
            self.assertAlmostEqual(ellipsoid.axis_2, ratio_321 * expected_axis_1)
            self.assertAlmostEqual(ellipsoid.axis_3, ratio_321 * expected_axis_1)

        with self.subTest("ratio_321 and axis_1 supplied"):
            ratio_321 = 0.5
            ellipsoid = Ellipsoid(
                "1",
                {"axis_1": 0.4, "ratio_321": ratio_321, **self.rotation_descriptors},
                self.rve_dims,
            )
            self.assertAlmostEqual(ellipsoid.axis_1, 0.4)
            self.assertAlmostEqual(ellipsoid.axis_2, ratio_321 * 0.4)
            self.assertAlmostEqual(ellipsoid.axis_3, ratio_321 * 0.4)

        with self.subTest("ratio_32, ratio_21 and semi_axis_1 supplied"):
            ratio_32 = 0.5
            ratio_21 = 0.5
            ellipsoid = Ellipsoid(
                "1",
                {
                    "semi_axis_1": 0.2,
                    "ratio_32": ratio_32,
                    "ratio_21": ratio_21,
                    **self.rotation_descriptors,
                },
                self.rve_dims,
            )
            expected_axis_1 = 2 * 0.2
            expected_axis_2 = ratio_21 * expected_axis_1
            expected_axis_3 = ratio_32 * expected_axis_2
            self.assertAlmostEqual(ellipsoid.axis_1, expected_axis_1)
            self.assertAlmostEqual(ellipsoid.axis_2, expected_axis_2)
            self.assertAlmostEqual(ellipsoid.axis_3, expected_axis_3)

        with self.subTest("ratio_32, ratio_21 and axis_2 supplied"):
            ratio_32 = 0.5
            ratio_21 = 0.5
            ellipsoid = Ellipsoid(
                "1",
                {
                    "axis_2": 0.2,
                    "ratio_32": ratio_32,
                    "ratio_21": ratio_21,
                    **self.rotation_descriptors,
                },
                self.rve_dims,
            )
            expected_axis_1 = 2 * 0.2
            expected_axis_2 = ratio_21 * expected_axis_1
            expected_axis_3 = ratio_32 * expected_axis_2
            self.assertAlmostEqual(ellipsoid.axis_1, expected_axis_1)
            self.assertAlmostEqual(ellipsoid.axis_2, expected_axis_2)
            self.assertAlmostEqual(ellipsoid.axis_3, expected_axis_3)

        with self.subTest("p_3 and phi_3 supplied"):
            p_3 = 0.8
            phi_z = 1.6
            ellipsoid = Ellipsoid(
                "1",
                {"vf" : 0.1,
                 "axis_1": 0.2,
                 "axis_2":0.1,
                 "axis_3": 0.1,
                 "p_3": p_3,
                 "phi_z" : phi_z},
                self.rve_dims,
            )
            self.assertAlmostEqual(ellipsoid.angle, 1.748142665 )
            np.testing.assert_allclose( ellipsoid.rotation_axis, np.array([0.36423807, -0.35375333, 0.86150404]) )
            



    def test_properties(self):
        with self.subTest("Volume"):
            self.assertAlmostEqual(self.ellipsoid.volume, (4/3)*np.pi * 0.2*0.15*0.1)
        with self.subTest("radius"):
            self.assertAlmostEqual(self.ellipsoid.radius, 0.2)
        with self.subTest("radius_insc"):
            self.assertAlmostEqual(self.ellipsoid.radius_insc, 0.1)
        with self.subTest("all semi_axis"):
            self.assertAlmostEqual(self.ellipsoid.semi_axis_1, 0.2)
            self.assertAlmostEqual(self.ellipsoid.semi_axis_2, 0.15)


    def test_contract_and_dilate(self):
        with self.subTest("dilate"):
            self.ellipsoid.dilate(0.05)
            self.assertAlmostEqual(self.ellipsoid.volume, (4/3)*np.pi * 0.25*0.2*0.15)
        with self.subTest("contract back to the original size"):
            self.ellipsoid.contract(0.05)
            self.assertAlmostEqual(self.ellipsoid.volume, (4/3)*np.pi * 0.2*0.15*0.1)

    def test_point_inside(self):
        self.ellipsoid.position_center = [0.5,0.5,0.5]
        with self.subTest("Point inside the ellipsoid"):
            self.assertTrue(self.ellipsoid.point_inside(np.array([0.35, 0.55, 0.5]), self.rve_dims))
        with self.subTest("Point outside the ellipsoid"):
            self.assertTrue(not self.ellipsoid.point_inside(np.array([0.9, 0.9, 0.5]), self.rve_dims))

    def test_generate_points_on_surface(self):
        # Without erosion
        points = self.ellipsoid.generate_points_on_surface(4)
        # location of the points relative to the center of the unrotated ellipsoid
        points_loc = (points - self.ellipsoid.position_center).dot(
            self.ellipsoid.rotation_mat
        )
        surface_eq = (
            points_loc[:, 0] ** 2 / self.ellipsoid.semi_axis_1 ** 2
            + points_loc[:, 1] ** 2 / self.ellipsoid.semi_axis_2 ** 2
            + points_loc[:, 2] ** 2 / self.ellipsoid.semi_axis_3 ** 2
        )
        np.testing.assert_allclose(surface_eq, 1.0, atol=1e-10)


    def test_compute_critical_erosion_thickness(self):
        self.assertAlmostEqual(self.ellipsoid.compute_critical_erosion_thickness(), 0.1**2/0.2)

    def test_rescale(self):
        self.ellipsoid.position_center = np.array([0.3, 0.4, 0.5])
        self.ellipsoid.rescale(2)
        #self.assertAlmostEqual(sphere.radius, 0.4)
        np.testing.assert_allclose(self.ellipsoid.position_center, np.array([0.6, 0.8, 1.0]))

    def test_generate_point_inside(self):
        for _ in range(20):
            self.assertTrue(self.ellipsoid.point_inside(self.ellipsoid.generate_point_inside(), self.rve_dims))


    def test_support_function(self):
        """Check support function."""
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



class EllipsoidIntersection(unittest.TestCase):
    "Test regarding the intersection check, area and length for different scenarios with two ellipsoids."


    def setUp(self):
        "Creates a ellipsoid with position_center assigned."
        self.rve_dims = [1, 1, 1]
        self.rotation_descriptors = {
            "rot_axis_comp_x": 0,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 1,
            "angle": 0,
        }
        self.ellipsoid = Ellipsoid(
                "1",
                {"axis_1": 0.3, "axis_2": 0.2, "axis_3": 0.2, **self.rotation_descriptors},
                self.rve_dims,
            )
        
        self.ellipsoid.position_center = [0.5,0.5,0.5]

    def test_not_intersecting(self):
        other_ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.3,
                "axis_2": 0.2,
                "axis_3": 0.2,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1.0,
                "angle": np.pi,
            },
            self.rve_dims
            )
        other_ellipsoid.position_center = np.array([0.1, 0.2, 0.8])
        with self.subTest("Test intersection check"):
            self.assertTrue(not self.ellipsoid.intersection(other_ellipsoid, self.rve_dims))
        with self.subTest("Test intersection area"):
            self.assertEqual(self.ellipsoid.intersection_area(other_ellipsoid, self.rve_dims), 0)
        with self.subTest("Test intersection length"):
            intersection_length, unit_vector = self.ellipsoid.intersection_length(other_ellipsoid, self.rve_dims)
            self.assertEqual(intersection_length, 0)
            np.testing.assert_allclose( unit_vector, [0,0,0] )


    def test_other_ellipsoid_inside(self):
        "other_ellipsoid is completely inside self.ellipsoid "

        other_ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.1,
                "axis_2": 0.1,
                "axis_3": 0.15,
                "rot_axis_comp_x": np.sqrt(3) / 3,
                "rot_axis_comp_y": np.sqrt(3) / 3,
                "rot_axis_comp_z": np.sqrt(3) / 3,
                "angle": 0,
            },
            self.rve_dims,
        )
        other_ellipsoid.position_center = np.array([0.5, 0.5, 0.5])

        box = self.rve_dims
        
        with self.subTest("Test intersection check"):
            self.assertTrue(
                self.ellipsoid.intersection(other_ellipsoid, self.rve_dims)
            )
        with self.subTest("Test intersection volume"):
            overlap_volume = self.ellipsoid.intersection_area(other_ellipsoid, box)
            self.assertAlmostEqual(overlap_volume, 4/3 * np.pi * 0.05*0.05*0.075, places = 3)
            # The intersection area is obtained via a monte carlo process and, thus, it has some error. Due to this, the intersection volume only has to be equal to the expected volume with places=3. The test, as the code is now, passes some times for places=4, but fails some times as well.

        with self.subTest("Test intersection volume monte carlo"):
            overlap_volume, _ = self.ellipsoid.intersection_area_monte_carlo(other_ellipsoid, box)
            self.assertAlmostEqual(overlap_volume, 4/3 * np.pi * 0.05*0.05*0.075, places = 3)

        with self.subTest("Test intersection length"):
            intersection_length, unit_vector = self.ellipsoid.intersection_length(other_ellipsoid, box)

            other_ellipsoid.position_center = intersection_length * unit_vector
            self.assertTrue(not self.ellipsoid.intersection(other_ellipsoid, box))

    def test_partially_intersecting(self):

        other_ellipsoid = Ellipsoid(
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
        other_ellipsoid.position_center = np.array([0.6, 0.5, 0.5])

        box = self.rve_dims
        with self.subTest("Test intersection check"):
            self.assertTrue(
                self.ellipsoid.intersection(other_ellipsoid, self.rve_dims)
            )
        with self.subTest("Test intersection volume"):
            overlap_volume_1 = self.ellipsoid.intersection_volume_ellipsoid_other(
                other_ellipsoid, self.rve_dims, alg_type="regular")
            overlap_volume_2 = self.ellipsoid.intersection_volume_ellipsoid_other(
                other_ellipsoid, self.rve_dims, alg_type="random")
            self.assertAlmostEqual(overlap_volume_1, 0.0037, places = 2)
            self.assertAlmostEqual(overlap_volume_2, 0.0037, places = 3)

        with self.subTest("Test intersection volume monte carlo"):
            overlap_volume, _ = self.ellipsoid.intersection_area_monte_carlo(other_ellipsoid, box)
            self.assertAlmostEqual(overlap_volume, 0.0037, places = 3)

        with self.subTest("Test intersection length"):
            intersection_length, unit_vector = self.ellipsoid.intersection_length(other_ellipsoid, box)

            self.assertAlmostEqual(intersection_length, 0.17888543)
            np.testing.assert_allclose(
                unit_vector,
                [4.47199053e-01, 4.56811048e-06, 8.94434462e-01],
                rtol=1e-6,
                atol=1e-9,
            )

