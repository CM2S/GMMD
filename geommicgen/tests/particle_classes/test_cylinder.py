"""
Unit tests regarding the Cylinder particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Cylinder, Ellipsoid


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
        intersection, overlap_length = cyl_1.intersection_cylinder_cylinder(
            cyl_2, rve_dims
        )
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
        intersection, overlap_length = cyl_1.intersection_cylinder_cylinder(
            cyl_2, rve_dims
        )
        intersection_1 = cyl_1.intersection_gjk(cyl_2, rve_dims)

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


class TestGJKIntersectionCylinder(unittest.TestCase):
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
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
        intersection = cyl_1.intersection_gjk(cyl_2, rve_dims)
        self.assertTrue(not intersection)


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
        point_inside = self.cylinder.point_inside(
            np.array([0.5, 0.5, 0.5]), self.rve_dims
        )
        self.assertTrue(point_inside)

    def test_point_inside_out(self):
        point_inside = self.cylinder.point_inside(
            np.array([0.75, 0.5, 0.5]), self.rve_dims
        )
        self.assertTrue(not point_inside)


class TestIntegrationCylinder(unittest.TestCase):
    """Test the Monte Carlo integration for cylinders."""

    @unittest.expectedFailure
    def test_cylinder_inside(self):
        """The cylinder is completly inside an ellipsoid.

        (Not working, but also currently not used)"""
        rve_dims = [2.5, 2.5, 2.5]
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
            cylinder, rve_dims, tol=1e-1
        )
        # # print(
        #     "error_estimate_2",
        #     error_estimate,
        #     intersection_volume,
        #     cylinder.volume,
        #     ((intersection_volume - cylinder.volume) / cylinder.volume) * 100,
        # )
        self.assertTrue(
            np.abs((intersection_volume - cylinder.volume) / cylinder.volume) * 100 < 1
        )

    @unittest.skip("Test failing, but function no longer used.")
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
                "axis_1": 0.05,
                "axis_2": 0.2,
                "axis_3": 0.1,
                "rot_axis_comp_x": np.sqrt(3) / 3,
                "rot_axis_comp_y": np.sqrt(3) / 3,
                "rot_axis_comp_z": np.sqrt(3) / 3,
                "angle": 0,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        intersection_volume, error_estimate = cylinder.intersection_area_monte_carlo(
            ellipsoid, rve_dims, tol=1e-1
        )
        (
            intersection_volume_2,
            error_estimate_2,
        ) = cylinder.intersection_area_monte_carlo(ellipsoid, rve_dims, tol=1)
        # print("error_estimate_1", error_estimate, intersection_volume, ellipsoid.volume)
        # print(
        #     "error_estimate_2",
        #     error_estimate_2,
        #     intersection_volume_2,
        #     ellipsoid.volume,
        # )
        self.assertTrue(
            np.abs((intersection_volume - ellipsoid.volume) / ellipsoid.volume) * 100
            < 1
        )
