"""
Unit tests regarding the Cylindrical Fiber particle class.
"""
import unittest
from unittest.mock import Mock


import numpy as np
from geommicgen.microstructure.particleclasses import CylindricalFiber


class TestFiber(unittest.TestCase):

    def setUp(self):
        phase = "1"
        descriptors = {
            "r": 0.1,
            "n": 1,
            "direction": 2,
        }
        self.rve_dims = [1, 1, 3]
        self.fiber = CylindricalFiber(phase, descriptors, self.rve_dims)

    def test_init(self):
        """Check if the attributes were set correctly in __init__."""

        self.assertEqual(self.fiber.direction_fibers, 2)
        self.assertEqual(self.fiber.length_dir_fibers, self.rve_dims[2])
        self.assertEqual(self.fiber.radius, 0.1)

    def test_volume(self):
        self.assertAlmostEqual(self.fiber.volume, np.pi*0.1**2*3)
        