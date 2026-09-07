"""
Unit tests regarding particle classes shared methods.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import (
    Ellipsoid,
    Ellipse,
    Cylinder,
    Particle,
)


class TestParticleClass(unittest.TestCase):
    def test_check_size_parameter(self):

        descriptors = {"major_axis": 0.1, "minor_axis": 0.2, "angle": 0.3, "n": -1}
        with self.assertRaises(ValueError):

            _ = Ellipse("1", descriptors, [1, 1])

        descriptors = {"major_axis": 0.1, "minor_axis": 0.2, "angle": 0.3, "n": 1.2}
        with self.assertRaises(ValueError):

            _ = Ellipse("1", descriptors, [1, 1])

        descriptors = {"major_axis": 0.1, "vf": -0.2, "angle": 0.3, "n": -1}
        with self.assertRaises(ValueError):
            _ = Ellipse("1", descriptors, [1, 1])


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


class ParticleRescaleTest(unittest.TestCase):
    def assert_position(self, particle, position):
        for i, j in zip(particle.position_center, position):
            self.assertAlmostEqual(i, j)

    def test_cylinder(self):
        r_cyl_og = 2
        r_cyl_after = 1
        length_og = 1
        length_after = 0.5
        position_og = np.array([1.0, 2, 3])
        position_after = np.array([0.5, 1, 1.5])
        particle = Cylinder(
            "1",
            {
                "r_cyl": r_cyl_og,
                "length": length_og,
                "azimuth_angle": 0,
                "polar_angle": 0,
                "n": 1,
            },
            [10, 10, 10],
        )
        particle.position_center = position_og
        particle.rescale(0.5)
        self.assertAlmostEqual(particle.r_cyl, r_cyl_after)
        self.assertAlmostEqual(particle.length, length_after)
        self.assert_position(particle, position_after)

    def test_ellipse(self):
        major_axis_og = 2
        major_axis_after = 4
        minor_axis_og = 1.5
        minor_axis_after = 3
        position_og = np.array([1.0, 2])
        position_after = np.array([2, 4])
        particle = Ellipse(
            "1",
            {
                "major_axis": major_axis_og,
                "minor_axis": minor_axis_og,
                "angle": 0,
                "n": 1,
            },
            [10, 10],
        )
        particle.position_center = position_og
        particle.rescale(2)
        self.assertAlmostEqual(particle.major_axis, major_axis_after)
        self.assertAlmostEqual(particle.minor_axis, minor_axis_after)
        self.assert_position(particle, position_after)

    def test_ellipsoid(self):
        axis_1_og = 2
        axis_1_after = 4
        axis_2_og = 1.5
        axis_2_after = 3
        axis_3_og = 1.5
        axis_3_after = 3
        position_og = np.array([1.0, 2, 1])
        position_after = np.array([2, 4, 2.0])
        particle = Ellipsoid(
            "1",
            {
                "axis_1": axis_1_og,
                "axis_2": axis_2_og,
                "axis_3": axis_3_og,
                "rot_axis_comp_x": 1.0,
                "rot_axis_comp_y": 1.0,
                "rot_axis_comp_z": 1.0,
                "angle": 0,
                "n": 1,
            },
            [10, 10, 10],
        )
        particle.position_center = position_og
        particle.rescale(2)
        self.assertAlmostEqual(particle.axis_1, axis_1_after)
        self.assertAlmostEqual(particle.axis_2, axis_2_after)
        self.assertAlmostEqual(particle.axis_3, axis_3_after)
        self.assert_position(particle, position_after)
