"""
Unit tests regarding the Ellipsoid particle class.
"""
import unittest

import numpy as np
import time
from geommicgen.microstructure.particleclasses import Ellipsoid


class TestEllipsoid(unittest.TestCase):
    """Tests concening Ellipsoids"""

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


class EllipsoidTestPartiallyIntersecting(unittest.TestCase):
    """Tests for the Ellipsoid class."""

    def setUp(self):
        self.rve_dims = [1.0, 1.0, 1.0]

        self.ellipsoid_1 = Ellipsoid(
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
        self.ellipsoid_1.position_center = np.array([0.95, 0.5, 0.5])

        self.ellipsoid_2 = Ellipsoid(
            "1",
            {
                "axis_1": 0.3,
                "axis_2": 0.3,
                "axis_3": 0.3,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1.0,
                "angle": 0,
            },
            self.rve_dims,
        )
        self.ellipsoid_2.position_center = np.array([0.05, 0.5, 0.6])

        box = self.rve_dims
        # Saving the array defining the RVE box
        diff_in_box = (
            self.ellipsoid_1.position_center - self.ellipsoid_2.position_center
        )
        self.diff_nearest_other = box * np.round(diff_in_box / box)
        # Computing the difference vector between the centers of the current sphere and

    def test_ellipsoid_intersection(self):
        """Test if the ellipsoids intersect."""

        self.assertTrue(
            self.ellipsoid_1.intersection_ellipsoid_ellipsoid(
                self.ellipsoid_2, self.rve_dims
            )
        )

    def test_ellipsoid_intersection_volume(self):
        """Checking the computed intersection volume.

        Computed using a random distribution of points and a grid."""
        overlap_volume_1 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="random"
        )
        overlap_volume_2 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="regular"
        )

        # v_ellipsoid_2 = ellipsoid_2.volume
        # # print(overlap_volume_1, end_1 - start_1, overlap_volume_2, end_2 - start_2)
        self.assertTrue(np.abs(overlap_volume_1 - overlap_volume_2) < 1e-2)

    def test_ellipsoid_intersection_volume_general_monte_carlo(self):
        """Checking the computed intersection volume.

        Computed using a random distribution of points and a grid."""
        begin_1 = time.time()
        overlap_volume_1 = self.ellipsoid_1.intersection_volume_ellipsoid_other(
            self.ellipsoid_2, self.rve_dims, alg_type="random"
        )
        time_1 = time.time() - begin_1
        begin_2 = time.time()
        (
            overlap_volume_2,
            error_estimate,
        ) = self.ellipsoid_1.intersection_area_monte_carlo(
            self.ellipsoid_2,
            self.rve_dims,
        )
        time_2 = time.time() - begin_2
        # # print("time", time_1, time_2)
        # # print("error_estimate", error_estimate)
        # # print("overlap", overlap_volume_1, overlap_volume_2)
        # v_ellipsoid_2 = ellipsoid_2.volume
        # # print(overlap_volume_1, end_1 - start_1, overlap_volume_2, end_2 - start_2)
        self.assertTrue(np.abs(overlap_volume_1 - overlap_volume_2) < 1e-2)

    def test_intersection_gjk(self):
        intersection = self.ellipsoid_1.intersection_gjk(
            self.ellipsoid_2, self.rve_dims
        )
        self.assertTrue(intersection)


class EllipsoidTestPartiallyIntersecting_1(unittest.TestCase):
    """Tests for the Ellipsoid class."""

    def setUp(self):
        self.rve_dims = [1.0, 1.0, 1.0]

        self.ellipsoid_1 = Ellipsoid(
            "1",
            {
                "axis_1": 0.3367780601921259,
                "axis_2": 0.16838903009606296,
                "axis_3": 0.2245187067947506,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1.0,
                "angle": 2.6157302920449386,
            },
            self.rve_dims,
        )
        self.ellipsoid_1.position_center = np.array(
            [0.08452537, 0.64733004, 0.96206736]
        )

        self.ellipsoid_2 = Ellipsoid(
            "1",
            {
                "axis_1": 0.3367780601921259,
                "axis_2": 0.16838903009606296,
                "axis_3": 0.2245187067947506,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1.0,
                "angle": 4.420185407416334,
            },
            self.rve_dims,
        )
        self.ellipsoid_2.position_center = np.array([0.15250794, 0.3756831, 0.96278858])

        box = self.rve_dims
        # Saving the array defining the RVE box
        diff_in_box = (
            self.ellipsoid_1.position_center - self.ellipsoid_2.position_center
        )
        self.diff_nearest_other = box * np.round(diff_in_box / box)
        # Computing the difference vector between the centers of the current sphere and

    def test_intersection_gjk_2(self):

        intersection = self.ellipsoid_1.intersection_gjk(
            self.ellipsoid_2, self.rve_dims
        )
        self.assertTrue(intersection)

        # previous_mic_path = (
        #     "/home/jose/Documents/code/test_runs/3D/cylindrs_94/mic_0/mic.mic"
        # )
        # with open(previous_mic_path, "rb") as mic:
        #     info_previous_sample = pickle.load(mic)
        #     # No need to generate a new microstructure. Using a previous microstructure.
        #     current_sample = info_previous_sample["microstructure"]
        #     current_mic_generator = info_previous_sample["generation_method"]
        #     trouble_pair = []
        #     for i_particle in current_sample.particles:
        #         if (
        #             i_particle.position_center[0] < 0.25
        #             and 0.25 < i_particle.position_center[1] < 0.75
        #             and i_particle.position_center[2] > 0.75
        #         ):
        #             trouble_pair.append(i_particle)
        #             i_particle.delta = 0
        #             # print(
        #                 i_particle.axis_1,
        #                 i_particle.axis_2,
        #                 i_particle.axis_3,
        #                 i_particle.rotation_axis,
        #                 i_particle.angle,
        #                 i_particle.position_center,
        #             )
        #             # # print(vars(i_particle))
        #     intersection = trouble_pair[0].intersection_gjk(trouble_pair[1], [1, 1, 1])

        # self.assertTrue(intersection)
        # with open(previous_mic_path, "rb") as mic:
        #     info_previous_sample = pickle.load(mic)
        #     # No need to generate a new microstructure. Using a previous microstructure.
        #     current_sample = info_previous_sample["microstructure"]
        #     current_mic_generator = info_previous_sample["generation_method"]
        #     trouble_pair = []
        #     for i_particle in current_sample.particles:
        #         if (
        #             i_particle.position_center[0] < 0.25
        #             and 0.25 < i_particle.position_center[1] < 0.75
        #             and i_particle.position_center[2] > 0.75
        #         ):
        #             trouble_pair.append(i_particle)
        #             # # print(vars(i_particle))
        #     intersection, overlap_length, _ = trouble_pair[0].intersection_gjk(
        #         trouble_pair[1], [1, 1, 1]
        #     )
        #     self.assertTrue(intersection)


class TestGJKIntersectionOverlapLengthEllipsoid(unittest.TestCase):
    def test_two_intersecting_ellipsoids(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "axis_1": 0.1,
            "axis_2": 0.3,
            "axis_3": 0.2,
            "n": 1,
            "rot_axis_comp_x": 1,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 0,
            "angle": 0,
        }
        ellipsoid_1 = Ellipsoid(phase_1, descriptors_1, rve_dims)
        ellipsoid_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "axis_1": 0.1,
            "axis_2": 0.3,
            "axis_3": 0.2,
            "n": 1,
            "rot_axis_comp_x": 1,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 0,
            "angle": 0,
        }
        ellipsoid_2 = Ellipsoid(phase_2, descriptors_2, rve_dims)
        ellipsoid_2.position_center = np.array([0.5, 0.65, 0.5])
        intersection = ellipsoid_1.intersection_gjk(ellipsoid_2, rve_dims)
        overlap_length, unit_vector = ellipsoid_1.intersection_length_mink_diff(
            ellipsoid_2, rve_dims
        )
        self.assertTrue(intersection)
        ellipsoid_2.position_center += overlap_length * unit_vector
        intersection = ellipsoid_1.intersection_gjk(ellipsoid_2, rve_dims)
        self.assertTrue(not intersection)

    def test_two_intersecting_ellipsoids_2(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "axis_1": 0.1,
            "axis_2": 0.3,
            "axis_3": 0.2,
            "n": 1,
            "rot_axis_comp_x": 1,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 0,
            "angle": 0,
        }
        ellipsoid_1 = Ellipsoid(phase_1, descriptors_1, rve_dims)
        ellipsoid_1.position_center = np.array([0.5, 0.5, 0.6])

        phase_2 = "1"
        descriptors_2 = {
            "axis_1": 0.1,
            "axis_2": 0.3,
            "axis_3": 0.2,
            "n": 1,
            "rot_axis_comp_x": 1,
            "rot_axis_comp_y": 0,
            "rot_axis_comp_z": 0,
            "angle": 0,
        }
        ellipsoid_2 = Ellipsoid(phase_2, descriptors_2, rve_dims)
        ellipsoid_2.position_center = np.array([0.5, 0.65, 0.5])
        intersection = ellipsoid_1.intersection_gjk(ellipsoid_2, rve_dims)
        overlap_length, unit_vector = ellipsoid_1.intersection_length_mink_diff(
            ellipsoid_2, rve_dims
        )
        self.assertTrue(intersection)
        ellipsoid_2.position_center += overlap_length * unit_vector
        intersection = ellipsoid_1.intersection_gjk(ellipsoid_2, rve_dims)
        self.assertTrue(not intersection)
