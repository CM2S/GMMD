"""
Unit tests regarding the Disk particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Disk


class TestGJKIntersectionDisk(unittest.TestCase):
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
        intersection = disk_1.intersection_gjk(disk_2, rve_dims)
        self.assertTrue(intersection)


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
        intersection = disk_1.intersection_gjk(disk_2, rve_dims)
        self.assertTrue(not intersection)


class TestGJKIntersectionOverlapLengthDisk(unittest.TestCase):
    def test_two_intersecting_disks(self):
        rve_dims = [1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.05,
            "n": 1,
        }
        disk_1 = Disk(phase_1, descriptors_1, rve_dims)
        disk_1.position_center = np.array([0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.05,
            "n": 1,
        }
        disk_2 = Disk(phase_2, descriptors_2, rve_dims)
        disk_2.position_center = np.array([0.5, 0.55])
        intersection = disk_1.intersection_gjk(disk_2, rve_dims)
        intersection_length, _ = disk_1.intersection_length_mink_diff(disk_2, rve_dims)
        self.assertTrue(intersection)
        self.assertTrue(np.abs(0.05 - intersection_length) / 0.05 < 1e-8)
