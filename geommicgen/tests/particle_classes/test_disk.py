"""
Unit tests regarding the Disk particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Disk, Ellipse
from unittest.mock import Mock


class TestDisk(unittest.TestCase):

    def _build_disk(self,
            descriptors={"n": 1, "r": 0.1},
            rve_dims = [1,1],
            position_center = None
                    ):
        disk = Disk("phase", descriptors, rve_dims)
        disk.position_center = position_center
        return disk

    def test_init(self):
        "Tests the three ways of describing a disk supported by the constructor."
        rve_dims = [1, 1]
        with self.subTest("Radius supplied directly"):
            disk = Disk("phase", {"n": 1, "r": 0.1}, rve_dims)
            self.assertAlmostEqual(disk.radius, 0.1)

        with self.subTest("Area per particle supplied"):
            disk = Disk("phase", {"n": 1, "area": np.pi * 0.1 ** 2}, rve_dims)
            self.assertAlmostEqual(disk.radius, 0.1)

        with self.subTest("Volume fraction and number of particles supplied"):
            n, vf = 4, 0.1
            disk = Disk("phase", {"n": n, "vf": vf}, rve_dims)
            expected_area = vf * rve_dims[0] * rve_dims[1] / n
            expected_radius = np.sqrt(expected_area / np.pi)
            self.assertAlmostEqual(disk.radius, expected_radius)

    def test_generate_points_on_surface(self):
        disk = self._build_disk(descriptors={"n": 1, "r": 0.1}, position_center=np.array([2.0, 3.0]))
        with self.subTest("Without erosion"):
            points = disk.generate_points_on_surface(4)
            expected = np.array(
                [[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]]
            ) + disk.position_center
            np.testing.assert_allclose(points, expected, atol=1e-10)

        with self.subTest("With erosion"):
            erosion_thick = 0.03
            points = disk.generate_points_on_surface(4, erosion_thick=erosion_thick)
            distances = np.linalg.norm(points - disk.position_center, axis=1)
            np.testing.assert_allclose(distances, disk.radius - erosion_thick)

    def test_intersection_area_calls(self):
        "Tests if the function intersection_area calls the correct functions to compute the intersection area."
        disk = self._build_disk()
        with self.subTest("Other particle is also a disk"):
            other_particle1 = self._build_disk()
            disk.intersection_area_disk_disk = Mock()
            disk.intersection_area(other_particle1, [1,1])
            disk.intersection_area_disk_disk.assert_called_once()

        with self.subTest("Other particle is an ellipse"):
            other_particle2 = Mock(spec = Ellipse)
            other_particle2.intersection_area_ellipse_ellipse = Mock()
            disk.intersection_area(other_particle2, [1,1])
            other_particle2.intersection_area_ellipse_ellipse.assert_called_once()


    def test_intersection_area_disk_disk(self):
        box = [1,1]
        disk = self._build_disk(position_center = np.array([0.5, 0.5]))
        with self.subTest("The other disk is completely inside"):
            other_disk1 = self._build_disk(descriptors = {"n": 1, "r": 0.05}, position_center = np.array([0.5, 0.5]))
            area1 = disk.intersection_area_disk_disk(other_disk1, box)
            self.assertAlmostEqual(area1, np.pi*0.05**2)

        with self.subTest("The other disk is completely outside"):
            other_disk2 = self._build_disk(position_center = np.array([0.3, 0.3]))
            area2 = disk.intersection_area_disk_disk(other_disk2, box)
            self.assertAlmostEqual(area2, 0)

        with self.subTest("The disks are intersecting and other disk is bigger"):
            other_disk3 = self._build_disk(descriptors = {"n": 1, "r": 0.15}, position_center = np.array([0.55, 0.45]))
            area3 = disk.intersection_area_disk_disk(other_disk3, box)
            self.assertAlmostEqual(area3, 0.02881811)


    def test_intersection_check_calls(self):
        "Tests if the function intersection calls the correct functions to do the intersection check"
        disk = self._build_disk()
        with self.subTest("Other particle is also a disk"):
            other_particle1 = self._build_disk()
            disk.intersection_disk_disk = Mock()
            disk.intersection(other_particle1, [1,1])
            disk.intersection_disk_disk.assert_called_once()

        with self.subTest("Other particle is an ellipse"):
            other_particle2 = Mock(spec = Ellipse)
            other_particle2.intersection = Mock()
            disk.intersection(other_particle2, [1,1])
            other_particle2.intersection.assert_called_once()

    def test_intersection_disk_disk(self):
        box = [1,1]
        disk = self._build_disk(position_center = np.array([0.5, 0.5]))
        with self.subTest("The other disk is completely inside"):
            other_disk1 = self._build_disk(descriptors = {"n": 1, "r": 0.05}, position_center = np.array([0.5, 0.5]))
            intersection_flag1 = disk.intersection_disk_disk(other_disk1, box)
            self.assertTrue(intersection_flag1)

        with self.subTest("The other disk is completely outside"):
            other_disk2 = self._build_disk(position_center = np.array([0.3, 0.3]))
            intersection_flag2 = disk.intersection_disk_disk(other_disk2, box)
            self.assertTrue(not intersection_flag2)

        with self.subTest("The disks are intersecting"):
            other_disk3 = self._build_disk(descriptors = {"n": 1, "r": 0.15}, position_center = np.array([0.55, 0.45]))
            intersection_flag3 = disk.intersection_disk_disk(other_disk3, box)
            self.assertTrue(intersection_flag3)

        with self.subTest("The other disk is completely inside, but inside=False"):
            other_disk4 = self._build_disk(descriptors = {"n": 1, "r": 0.05}, position_center = np.array([0.5, 0.5]))
            intersection_flag4 = disk.intersection_disk_disk(other_disk4, box, inside=False)
            self.assertTrue(not intersection_flag4)


    def test_point_inside(self):
        box = [1, 1]
        disk = self._build_disk(descriptors={"n": 1, "r": 0.1}, position_center=np.array([0.5, 0.5]))
        with self.subTest("Point inside the disk"):
            self.assertTrue(disk.point_inside(np.array([0.55, 0.5]), box))

        with self.subTest("Point outside the disk"):
            self.assertTrue(not disk.point_inside(np.array([0.8, 0.8]), box))

        with self.subTest("Point inside the disk only accounting for periodic wrap-around"):
            disk_at_edge = self._build_disk(
                descriptors={"n": 1, "r": 0.1}, position_center=np.array([0.02, 0.5])
            )
            self.assertTrue(disk_at_edge.point_inside(np.array([0.98, 0.5]), box))

    def test_volume(self):
        disk = self._build_disk(descriptors={"n": 1, "r": 0.2}, position_center=np.array([0.5, 0.5]))
        self.assertAlmostEqual(disk.volume, np.pi * 0.2 ** 2)

    def test_compute_critical_erosion_thickness(self):
        disk = self._build_disk(descriptors={"n": 1, "r": 0.2}, position_center=np.array([0.5, 0.5]))
        self.assertAlmostEqual(disk.compute_critical_erosion_thickness(), 0.2)

    def test_support_function(self):
        disk = self._build_disk(descriptors={"n": 1, "r": 0.2}, position_center=np.array([0.5, 0.5]))
        with self.subTest("Direction along the x-axis"):
            support = disk.support_function(np.array([1.0, 0.0, 0.0]))
            np.testing.assert_allclose(support, np.array([0.7, 0.5, 0.0]), atol=1e-10)

        with self.subTest("Direction along the y-axis"):
            support = disk.support_function(np.array([0.0, 1.0, 0.0]))
            np.testing.assert_allclose(support, np.array([0.5, 0.7, 0.0]), atol=1e-10)

    def test_intersection_length_disk_disk(self):
        box = [1, 1]
        disk = self._build_disk(descriptors={"n": 1, "r": 0.1}, position_center=np.array([0.5, 0.5]))

        with self.subTest("The disks do not overlap"):
            other_disk1 = self._build_disk(descriptors={"n": 1, "r": 0.1}, position_center=np.array([0.1, 0.1]))
            self.assertAlmostEqual(disk.intersection_length_disk_disk(other_disk1, box), 0)

        with self.subTest("One disk is completely inside the other"):
            other_disk2 = self._build_disk(descriptors={"n": 1, "r": 0.02}, position_center=np.array([0.5, 0.5]))
            self.assertAlmostEqual(disk.intersection_length_disk_disk(other_disk2, box), 2 * 0.02)

        with self.subTest("The disks partially overlap"):
            other_disk3 = self._build_disk(descriptors={"n": 1, "r": 0.1}, position_center=np.array([0.55, 0.5]))
            distance = 0.05
            expected_length = 0.1 + 0.1 - distance
            self.assertAlmostEqual(disk.intersection_length_disk_disk(other_disk3, box), expected_length)

    def test_intersection_length_calls(self):
        "Tests if the function intersection_length calls the correct functions to compute the intersection length."
        disk = self._build_disk()
        with self.subTest("Other particle is also a disk"):
            other_particle1 = self._build_disk()
            disk.intersection_length_disk_disk = Mock(return_value=0.05)
            disk.intersection_vector = Mock(return_value=np.array([1.0, 0.0]))
            intersection_length, _ = disk.intersection_length(other_particle1, [1, 1])
            disk.intersection_length_disk_disk.assert_called_once()
            disk.intersection_vector.assert_called_once()
            self.assertEqual(intersection_length, 0.05)

        with self.subTest("Other particle is not a disk and they intersect"):
            other_particle2 = Mock(spec=Ellipse)
            disk.intersection_gjk = Mock(return_value=True)
            disk.intersection_length_mink_diff = Mock(return_value=(0.07, np.array([0.0, 1.0])))
            intersection_length, _ = disk.intersection_length(other_particle2, [1, 1])
            disk.intersection_gjk.assert_called_once()
            disk.intersection_length_mink_diff.assert_called_once()
            self.assertEqual(intersection_length, 0.07)

        with self.subTest("Other particle is not a disk and they do not intersect"):
            other_particle3 = Mock(spec=Ellipse)
            disk.intersection_gjk = Mock(return_value=False)
            disk.intersection_length_mink_diff = Mock(return_value=(0.07, np.array([0.0, 1.0])))
            intersection_length, _ = disk.intersection_length(other_particle3, [1, 1])
            self.assertEqual(intersection_length, 0)


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
