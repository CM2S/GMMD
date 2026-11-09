"""
Unit tests regarding the Cylinder particle class.
"""
import unittest
from unittest.mock import Mock


import numpy as np
from geommicgen.microstructure.particleclasses import Cylinder, Ellipsoid, Sphere
from geommicgen.microstructure.particleclasses.cylinder import (
    cylinder_size,
    rotation_from_angles,
)


class TestCylinder(unittest.TestCase):
    def test_init(self):
        """Check if the attributes were set correctly in __init__. using different descriptor sets or errors are raised when an attribut has an invalid value"""
        rve_dims = [1, 1, 1]
        phase = "1"

        with self.subTest("Descriptor set 1"):
            descriptors = {
                "r_cyl": 0.1,
                "length": 0.1,
                "n": 2,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }

            cyl = Cylinder(phase, descriptors, rve_dims)
            self.assertEqual(cyl.length, 0.1)
            self.assertEqual(cyl.r_cyl, 0.1)
            self.assertEqual(cyl.azimuth_angle, 0)
            self.assertEqual(cyl.polar_angle, 0)

        with self.subTest("Descriptor set 2"):
            descriptors = {
                "vf": 0.1,
                "length": 0.2,
                "n": 10,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            cyl = Cylinder(phase, descriptors, rve_dims)
            self.assertEqual(cyl.length, 0.2)
            self.assertTrue(np.abs(cyl.r_cyl - np.sqrt(0.1 / (0.2 * 10 * np.pi))) < 1e-4)

        with self.subTest("Descriptor set 3"):
            descriptors = {
                "ratio": 2,
                "length": 0.2,
                "n": 10,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            cyl = Cylinder(phase, descriptors, rve_dims)
            self.assertEqual(cyl.length, 0.2)
            self.assertTrue(np.abs(cyl.r_cyl - 0.1) < 1e-4)

        with self.subTest("Descriptor set 4"):
            descriptors = {
                "ratio": 2,
                "r_cyl": 0.2,
                "n": 10,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            cyl = Cylinder(phase, descriptors, rve_dims)
            self.assertEqual(cyl.length, 0.4)
            self.assertEqual(cyl.r_cyl, 0.2)

        with self.subTest("Descriptor set 5 (invalid r_cyl value)"):
            descriptors = {
                "r_cyl": -0.1,
                "length": 0.1,
                "n": 2,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            with self.assertRaises(ValueError):
                Cylinder(phase, descriptors, rve_dims)

        with self.subTest("Descriptor set 6 (invalid length value)"):
            descriptors = {
                "r_cyl": 0.1,
                "length": -0.1,
                "n": 2,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            with self.assertRaises(ValueError):
                Cylinder(phase, descriptors, rve_dims)

        angles = {"azimuth_angle": 0, "polar_angle": 0}
        for i_case, i_descriptors, i_named in (
            ("zero radius", {"r_cyl": 0, "length": 0.1, "n": 2}, "r_cyl"),
            ("zero length", {"r_cyl": 0.1, "length": 0, "n": 2}, "length"),
            (
                "radius from the ratio too large",
                {"ratio": 0.5, "length": 0.2, "n": 2},
                "r_cyl",
            ),
            (
                "length from the ratio too long",
                {"ratio": 3, "r_cyl": 0.2, "n": 2},
                "length",
            ),
            ("zero ratio", {"ratio": 0.0, "length": 0.2, "n": 2}, "ratio"),
            (
                "negative volume fraction",
                {"vf": -0.1, "length": 0.2, "n": 10},
                "vf",
            ),
        ):
            with self.subTest(i_case):
                with self.assertRaisesRegex(
                    ValueError, "for (descriptor )?{0}".format(i_named)
                ):
                    Cylinder(phase, {**i_descriptors, **angles}, rve_dims)
        # A radius must be below a quarter of the smallest side of the RVE and a length
        # below half of it, and both above zero, whether the descriptors give them or
        # they follow from the ratio of the two. A ratio or a fraction out of bounds is
        # named itself, not through the radius it would give, since a ratio of zero
        # divides by zero and a negative fraction gives a radius that is not a number



    def test_properties(self):
        """Check if the volume, radius and radius_insc properties are correctly specified."""
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
        self.assertAlmostEqual(cyl.radius, 0.14142, places = 5)
        self.assertAlmostEqual(cyl.radius_insc, 0.1, places = 5)

    def test_sym_axis_unit_vec(self):
        """Check the axis of symmetry against the angles, and against the caps."""
        rve_dims = [1, 1, 1]
        rng = np.random.default_rng(0)
        for i_polar, i_azimuth in rng.uniform(0, np.pi, size=(10, 2)):
            cyl = Cylinder(
                "1",
                {
                    "r_cyl": 0.1,
                    "length": 0.2,
                    "n": 1,
                    "azimuth_angle": i_azimuth,
                    "polar_angle": i_polar,
                },
                rve_dims,
            )
            cyl.position_center = np.array([0.5, 0.5, 0.5])
            axis = cyl.sym_axis_unit_vec
            with self.subTest(polar=i_polar, azimuth=i_azimuth):
                np.testing.assert_allclose(
                    axis,
                    [
                        np.sin(i_polar) * np.cos(i_azimuth),
                        np.sin(i_polar) * np.sin(i_azimuth),
                        np.cos(i_polar),
                    ],
                    atol=1e-15,
                )
                self.assertTrue(cyl.point_inside(0.5 + 0.099 * axis, rve_dims))
                self.assertFalse(cyl.point_inside(0.5 + 0.101 * axis, rve_dims))
        # The caps are a half length, 0.1, from the centre along the axis

    def test_negative_angles(self):
        """Check that negative angles give the axis they stand for."""
        cyl = Cylinder(
            "1",
            {
                "r_cyl": 0.1,
                "length": 0.2,
                "n": 1,
                "azimuth_angle": -0.5,
                "polar_angle": -1.0,
            },
            [1, 1, 1],
        )
        self.assertEqual((cyl.polar_angle, cyl.azimuth_angle), (-1.0, -0.5))
        np.testing.assert_allclose(
            cyl.sym_axis_unit_vec,
            [np.sin(-1.0) * np.cos(-0.5), np.sin(-1.0) * np.sin(-0.5), np.cos(-1.0)],
            atol=1e-15,
        )
        # The axis is (-0.74, 0.40, 0.54). The absolute values of the angles would give
        # (0.74, 0.40, 0.54), another line


    def test_contract_and_dilate(self):
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl = Cylinder("1", descriptors, [1,1,1])
        cyl.position_center = np.zeros(3)
        with self.subTest("dilate"):
            cyl.dilate(0.05)
            # The dilation grows the radius, and the length by as much at each cap; the
            # grown cylinder holds every point at 0.05 from the original
            self.assertAlmostEqual(cyl.r_cyl, 0.15)
            self.assertAlmostEqual(cyl.length, 0.3)
            self.assertAlmostEqual(cyl.real_volume, 0.2 * np.pi * 0.1 ** 2)
            self.assertAlmostEqual(cyl.support_function(np.array([1, 0, 0]))[0], 0.15)
            self.assertAlmostEqual(cyl.support_function(np.array([0, 0, 1]))[2], 0.15)
            self.assertAlmostEqual(cyl.volume, 0.3 * np.pi * 0.15 ** 2)
        with self.subTest("contract back to the original size"):
            cyl.contract(0.05)
            self.assertAlmostEqual(cyl.delta, 0)
            self.assertAlmostEqual(cyl.r_cyl, 0.1)
            self.assertAlmostEqual(cyl.length, 0.2)
            self.assertAlmostEqual(cyl.support_function(np.array([1, 0, 0]))[0], 0.1)

    def test_rescale(self):
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl = Cylinder("1", descriptors, [1,1,1])
        cyl.position_center = np.array([0.3, 0.4, 0.5])
        cyl.rescale(2)
        self.assertAlmostEqual(cyl.r_cyl, 0.2)
        np.testing.assert_allclose(cyl.position_center, np.array([0.6, 0.8, 1.0]))



    def test_point_inside(self):
        rve_dims = [1,1,1]
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl = Cylinder("1", descriptors, rve_dims)
        cyl.position_center = np.array([0.6, 0.7, 0.3])
        with self.subTest("Point inside the cylinder"):
            self.assertTrue(cyl.point_inside(np.array([0.56, 0.74, 0.39]), rve_dims))
        with self.subTest("Point outside the cylinder"):
            self.assertTrue(not cyl.point_inside(np.array([0.9, 0.9, 0.5]), rve_dims))
        with self.subTest("Point with a NaN coordinate"):
            self.assertFalse(cyl.point_inside(np.array([0.6, np.nan, 0.3]), rve_dims))
        with self.subTest("Same answers as points_inside"):
            rng = np.random.default_rng(0)
            cyl = Cylinder(
                "1", {**descriptors, "azimuth_angle": 1.1, "polar_angle": 0.7}, rve_dims
            )
            cyl.position_center = np.array([0.95, 0.5, 0.05])
            points = cyl.position_center + rng.uniform(-0.15, 0.15, size=(500, 3))
            np.testing.assert_array_equal(
                [cyl.point_inside(i_point, rve_dims) for i_point in points],
                cyl.points_inside(points, rve_dims),
            )
        # The cylinder crosses two faces of the RVE, so that some points are asked
        # about through their nearest periodic image

    def test_generate_point_inside(self):
        "This test only works if cyl.point_inside function is working"
        rve_dims = [1,1,1]
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 10,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        cyl = Cylinder("1", descriptors, rve_dims)
        cyl.position_center = np.array([0.6, 0.7, 0.3])
        for _ in range(10):
            random_point_inside = cyl.generate_point_inside()
            self.assertTrue( cyl.point_inside(random_point_inside, rve_dims) )



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

    def test_support_function_reaches_the_support(self):
        """Check the support point against the support function of a cylinder."""
        rve_dims = [1, 1, 1]
        rng = np.random.default_rng(0)
        for i_polar, i_azimuth in rng.uniform(0, np.pi, size=(5, 2)):
            cyl = Cylinder(
                "1",
                {
                    "r_cyl": 0.1,
                    "length": 0.3,
                    "n": 1,
                    "azimuth_angle": i_azimuth,
                    "polar_angle": i_polar,
                },
                rve_dims,
            )
            cyl.position_center = np.array([0.5, 0.4, 0.3])
            cyl.dilate(0.02)
            axis = cyl.sym_axis_unit_vec
            directions = np.concatenate(
                [rng.normal(size=(20, 3)), [axis, -2 * axis, np.cross(axis, [1, 0, 0])]]
            )
            for i_direction in directions:
                along = i_direction.dot(axis)
                support = (
                    i_direction.dot(cyl.position_center)
                    + 0.17 * np.abs(along)
                    + 0.12 * np.linalg.norm(i_direction - along * axis)
                )
                with self.subTest(polar=i_polar, azimuth=i_azimuth, d=i_direction):
                    point = cyl.support_function(i_direction)
                    self.assertEqual(point.shape, (3,))
                    self.assertAlmostEqual(i_direction.dot(point), support, places=12)
        # The support function of a cylinder dilated by delta is that of the cylinder
        # it grows into: the reach of the centre, plus half the grown length times the
        # share of the direction along the axis, and the grown radius times the share
        # across it. Directions
        # along the axis and normal to it are among those tried, because rounding can
        # throw the point off the cap for the first

    def test_intersection_calls(self):
        "Check if the function intersection calls the correct function to compute the intersection based on the other particle type"
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
        with self.subTest("Other particle is a cylinder"):
            other_particle = Mock(spec= Cylinder)
            cyl.intersection_gjk = Mock()
            cyl.intersection(other_particle, rve_dims)
            cyl.intersection_gjk.assert_called_once()
            # Two cylinders are checked with GJK too

        with self.subTest("Other particle is other type"):
            other_particle = Mock()
            cyl.intersection_gjk = Mock()
            cyl.intersection(other_particle, rve_dims)
            cyl.intersection_gjk.assert_called_once()



    def test_intersection_lenght_calls(self):
        "Check if the function intersection_lenght calls the correct function to compute the intersection lenght based on the other particle type"
        descriptors = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }
        rve_dims = [1, 1, 1]
        cyl = Cylinder("1", descriptors, rve_dims)
        
        with self.subTest("Other particle is any type"):
            other_particle = Mock()
            cyl.intersection_gjk = Mock(return_value=True)
            cyl.intersection_length_mink_diff = Mock(return_value=["intersection_length", "unit_vector"])
            cyl.intersection_length(other_particle, rve_dims)
            cyl.intersection_length_mink_diff.assert_called_once()

    def test_intersection_length_with_a_sphere(self):
        """Check the overlap of a cylinder and a sphere beside it, measured by GJK."""
        rve_dims = [1, 1, 1]
        cyl = Cylinder(
            "1",
            {"r_cyl": 0.1, "length": 0.2, "n": 1, "azimuth_angle": 0, "polar_angle": 0},
            rve_dims,
        )
        cyl.position_center = np.array([0.5, 0.5, 0.5])
        sphere = Sphere("1", {"r": 0.1, "n": 1}, rve_dims)
        for i_dist_met in ("dist_approx", "dist_exact"):
            with self.subTest("Overlapping", dist_met=i_dist_met):
                sphere.position_center = np.array([0.5, 0.65, 0.5])
                length, direction = cyl.intersection_length(
                    sphere, rve_dims, dist_met=i_dist_met
                )
                self.assertAlmostEqual(length, 0.05)
                np.testing.assert_allclose(direction, [0, 1, 0], atol=1e-12)
        with self.subTest("Apart"):
            sphere.position_center = np.array([0.5, 0.75, 0.5])
            length, direction = cyl.intersection_length(sphere, rve_dims)
            self.assertEqual(length, 0)
            np.testing.assert_array_equal(direction, [0, 0, 0])
        # The side of the cylinder reaches 0.6 along y and the sphere starts at 0.55, so
        # they overlap by 0.05 along the line between their centres. A sphere goes the
        # way of every other particle, through GJK and the Minkowski difference


    def test_intersection_area_calls(self):
        "Check if the function intersection_area calls the correct function to compute the intersection area based on the other particle type"
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
        
        with self.subTest("Other particle is any type"):
            other_particle = Mock()
            cyl.intersection_area_monte_carlo = Mock()
            cyl.intersection_area(other_particle, rve_dims)
            cyl.intersection_area_monte_carlo.assert_called_once()


class TestCylinderSize(unittest.TestCase):
    """Test the radius and the length worked out from the descriptors."""

    def test_each_description(self):
        rve_dims = [1, 2, 3]
        for i_case, i_descriptors, i_size in (
            ("radius and length", {"r_cyl": 0.1, "length": 0.2}, (0.1, 0.2)),
            ("radius and ratio", {"r_cyl": 0.1, "ratio": 3}, (0.1, 0.3)),
            ("length and ratio", {"length": 0.3, "ratio": 3}, (0.1, 0.3)),
            (
                "length, fraction and number",
                {"length": 0.2, "vf": 0.1, "n": 10},
                (np.sqrt(0.1 * 6 / (0.2 * np.pi * 10)), 0.2),
            ),
        ):
            with self.subTest(i_case):
                np.testing.assert_allclose(
                    cylinder_size(i_descriptors, rve_dims), i_size, rtol=1e-12
                )

    def test_order_of_the_cases(self):
        rve_dims = [1, 1, 1]
        with self.subTest("fraction and number before the radius"):
            r_cyl, _ = cylinder_size(
                {"r_cyl": 0.5, "length": 0.2, "vf": 0.1, "n": 10}, rve_dims
            )
            self.assertAlmostEqual(r_cyl, np.sqrt(0.1 / (0.2 * np.pi * 10)))
        with self.subTest("ratio before the radius"):
            self.assertEqual(
                cylinder_size({"r_cyl": 0.5, "length": 0.2, "ratio": 2}, rve_dims),
                (0.1, 0.2),
            )
        # The constructor worked the radius out from the length last, over a radius
        # given with it, and a run that gives all of them still gets the same cylinder

    def test_incomplete_description(self):
        for i_descriptors in ({"r_cyl": 0.1, "n": 2}, {"length": 0.2, "vf": 0.1}):
            with self.subTest(descriptors=i_descriptors):
                with self.assertRaises(ValueError):
                    cylinder_size(i_descriptors, [1, 1, 1])
        with self.subTest("through the constructor"):
            with self.assertRaises(ValueError):
                Cylinder(
                    "1",
                    {"r_cyl": 0.1, "n": 2, "azimuth_angle": 0, "polar_angle": 0},
                    [1, 1, 1],
                )
        # A description that gives one size and not what the other follows from cannot
        # size the cylinder, and says so


class TestRotationFromAngles(unittest.TestCase):
    """Test the rotation from global coordinates to those of a cylinder."""

    def test_rotation(self):
        rng = np.random.default_rng(0)
        for i_polar, i_azimuth in rng.uniform(-np.pi, np.pi, size=(20, 2)):
            rot_mat = rotation_from_angles(i_polar, i_azimuth)
            with self.subTest(polar=i_polar, azimuth=i_azimuth):
                np.testing.assert_allclose(rot_mat.dot(rot_mat.T), np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rot_mat), 1)
                np.testing.assert_allclose(
                    rot_mat[0],
                    [
                        np.sin(i_polar) * np.cos(i_azimuth),
                        np.sin(i_polar) * np.sin(i_azimuth),
                        np.cos(i_polar),
                    ],
                    atol=1e-12,
                )
        # The rows are an orthonormal, right-handed frame, and the first of them is the
        # axis at the polar and the azimuth angle


class TestCylinderIntersection(unittest.TestCase):
    "Class with tests for two cylinder intersection"

    def setUp(self):
        self.rve_dims = [1, 1, 1]
        self.phase_1 = "1"
        self.descriptors_1 = {
            "r_cyl": 0.1,
            "length": 0.2,
            "n": 1,
            "azimuth_angle": 0,
            "polar_angle": 0,
        }       
        self.cyl_1 = Cylinder(self.phase_1, self.descriptors_1, self.rve_dims)
        self.cyl_1.position_center = np.array([0.5, 0.5, 0.5])

    def test_intersection_not_intersecting(self):
        """Test the intersection of non-intersecting cylinder."""
        
        with self.subTest("Case 1: cylinder axis far away"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.15,
                "length": 0.3,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.5, 0.9, 0.5])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            self.assertTrue(not intersection)

        with self.subTest("Case 2: cylinder axis close enough so that intersection could be possible"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.15,
                "length": 0.30,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": -np.pi/2,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.35, 0.35, 0.35])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)

            self.assertTrue(intersection)
            # They do overlap, by about 0.079: the point (0.45, 0.45, 0.45) is in both

    def test_side_side_intersection(self):
        """Test the intersection of cylinders overlapping side-side, that is, the lateral wall of one cylinder is touching the lateral wall of the other"""

        with self.subTest("Case 1"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.05,
                "length": 0.3,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.5, 0.6, 0.5])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            self.assertTrue(intersection)

        with self.subTest("Case 2"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.15,
                "length": 0.3,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": 0,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.65, 0.5, 0.5])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            
            self.assertTrue(intersection)


    def test_side_top_intersection(self):
        """Test the intersection of cylinders overlapping side-top, that is, the lateral wall of one cylinder is touching the top wall of the other"""

        with self.subTest("Case 1"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.05,
                "length": 0.1,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 4,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.5, 0.5, 0.65])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            self.assertTrue(intersection)

        with self.subTest("Case 1"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.05,
                "length": 0.1,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": -np.pi / 4,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.65, 0.5, 0.5])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            self.assertTrue(intersection)

        with self.subTest("Case 3"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.05,
                "length": 0.1,
                "n": 1,
                "azimuth_angle": np.pi/4,
                "polar_angle": -np.pi / 4,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.5, 0.65, 0.65])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)
            self.assertTrue(not intersection)
            # They are about 0.003 apart



    def test_top_top_intersection(self):
        """Test the intersection of cylinders overlapping top-top, that is, the top wall of one cylinder is touching the top wall of the other"""

        with self.subTest("Case 1"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.15,
                "length": 0.30,
                "n": 1,
                "azimuth_angle": np.pi/2,
                "polar_angle": np.pi/2,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.55, 0.7, 0.65])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)

            self.assertTrue(intersection)

        with self.subTest("Case 2"):
            phase_2 = "1"
            descriptors_2 = {
                "r_cyl": 0.15,
                "length": 0.30,
                "n": 1,
                "azimuth_angle": 0,
                "polar_angle": -np.pi/2,
            }
            cyl_2 = Cylinder(phase_2, descriptors_2, self.rve_dims)
            cyl_2.position_center = np.array([0.3, 0.4, 0.45])
            intersection = self.cyl_1.intersection(cyl_2, self.rve_dims)

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


class TestIntegrationCylinder(unittest.TestCase):
    """Test the Monte Carlo integration for cylinders."""

    @unittest.skip("Test sometimes fails. Since the area is computed using a monte carlo process, it has some randomness to it. Thus, sometimes the test fails, eventhough most times is is successful. Since the intersection area is not used in the default options, the test will be skipped.")
    def test_cylinder_inside(self):
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
        self.assertTrue(
            np.abs((intersection_volume - cylinder.volume) / cylinder.volume) * 100 < 1
        )

    @unittest.skip("Test sometimes fails. Since the area is computed using a monte carlo process, it has some randomness to it. Thus, sometimes the test fails, eventhough most times is is successful. Since the intersection area is not used in the default options, the test will be skipped.")
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
        self.assertTrue(
            np.abs((intersection_volume - ellipsoid.volume) / ellipsoid.volume) * 100
            < 1
        )
