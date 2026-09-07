"""
Unit tests regarding the Sphere particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Sphere, Cylinder


class TestGJKIntersectionOverlapLengthSphere(unittest.TestCase):
    def test_two_intersecting_spheres(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_1 = Sphere(phase_1, descriptors_1, rve_dims)
        sphere_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_2 = Sphere(phase_2, descriptors_2, rve_dims)
        sphere_2.position_center = np.array([0.5, 0.55, 0.5])
        # plot_particles_2d([sphere_1, sphere_2], rve_dims, "", save=False, show=True)
        intersection = sphere_1.intersection_gjk(sphere_2, rve_dims)
        overlap_length, _ = sphere_1.intersection_length_mink_diff(sphere_2, rve_dims)
        self.assertTrue(intersection)
        self.assertTrue(np.abs(0.05 - overlap_length) / 0.05 < 1e-8)

    def test_two_intersecting_spheres_2(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.1,
            "n": 1,
        }
        sphere_1 = Sphere(phase_1, descriptors_1, rve_dims)
        sphere_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_2 = Sphere(phase_2, descriptors_2, rve_dims)
        sphere_2.position_center = np.array([0.5, 0.55, 0.5])
        # plot_particles_2d([sphere_1, sphere_2], rve_dims, "", save=False, show=True)
        intersection = sphere_1.intersection_gjk(sphere_2, rve_dims)
        overlap_length, _ = sphere_1.intersection_length_mink_diff(sphere_2, rve_dims)
        self.assertTrue(intersection)
        self.assertTrue(np.abs(0.1 - overlap_length) / 0.1 < 1e-8)


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
        self.assertTrue(np.abs(intersection_length - 0.1) < 1e-4)

    @unittest.skip("Incomplete")
    def test_intersect_lateral(self):
        pass

    @unittest.skip("Incomplete")
    def test_intersect_inside(self):
        pass
