"""
Unit tests regarding particle classes shared methods.
"""
import unittest
from unittest.mock import Mock

import numpy as np
from geommicgen.microstructure.particleclasses import (
    Ellipsoid,
    Ellipse,
    Disk,
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


    def test_nearest_simplex(self):

        with self.subTest("triangle in xy plane"):
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

        with self.subTest("tetrahedron"):
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


    def test_nearest_periodic_image(self):

        with self.subTest("In 2D"):
            box = [2, 1]
            point_1 = np.array([0.9, 0.9])
            point_2 = np.array([0.9, 0.1])
            nearest_image_pt_1 = Particle.nearest_periodic_image(point_1, point_2, box)
            self.assertTrue(all(np.abs(nearest_image_pt_1 - np.array([0.9, -0.1])) < 1e-4))

        with self.subTest("In 3D"):
            box = [1, 1, 1]
            point_1 = np.array([0.5, 0.1, 0.9])
            point_2 = np.array([0.9, 0.9, 0.9])
            nearest_image_pt_1 = Particle.nearest_periodic_image(point_1, point_2, box)
            self.assertTrue(
                all(np.abs(nearest_image_pt_1 - np.array([0.5, 1.1, 0.9])) < 1e-4)
            )

    def test_mass(self):
        particle = Disk("1", {"r":0.1, "n": 1}, [1,1])
        with self.subTest("Mass option is radius"):
            mass = particle.mass("radius")
            self.assertEqual(mass, 0.1)
        with self.subTest("Mass option is volume"):
            mass = particle.mass("volume")
            self.assertEqual(mass, np.pi*0.1**2)
        with self.subTest("Mass option is unit"):
            mass = particle.mass("unit")
            self.assertEqual(mass, 1)
        with self.subTest("Mass option is invalid"):
            with self.assertRaises(ValueError):
                mass = particle.mass("invalid")
