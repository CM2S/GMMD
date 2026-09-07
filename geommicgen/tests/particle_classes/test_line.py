"""
Unit tests regarding the Line particle class.
"""
import unittest

import numpy as np
from geommicgen.microstructure.particleclasses import Sphere, Ellipsoid, Line
from geommicgen.postproc.plotfuncs.plotting_functions import plot_particles_3d


class TestParticleLine(unittest.TestCase):
    """Testing the gjk intersection alg for the Line particle."""

    @unittest.skip("Failing, code incomplete")
    def test_point_sphere(self):
        """Testing intersection between Line and Sphere."""
        rve_dims = [1, 1, 1]
        sphere = Sphere(
            "1",
            {"r": 0.1},
            rve_dims,
        )
        sphere.position_center = np.array([0.5, 0.5, 0.5])
        line = Line("1", 0)
        line.position_center = np.array([0.5, 0.55, 0.5])
        intersection = sphere.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, _ = sphere.intersection_length_mink_diff(
            line,
            rve_dims,
        )

        self.assertTrue(np.abs(overlap_length - 0.05) < 1e-4)

        line.position_center = np.array([0.5, 0.7, 0.5])
        sphere.position_center = np.array([0.5, 0.5, 0.5])
        intersection = sphere.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, _ = sphere.intersection_length_mink_diff(
            line,
            rve_dims,
        )

        self.assertTrue(np.abs(np.abs(overlap_length) - 0.1) < 1e-4)

        line.position_center = np.array([0.7, 0.7, 0.7])
        sphere.position_center = np.array([0.5, 0.5, 0.5])
        intersection = sphere.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, unit_vector = sphere.intersection_length_mink_diff(
            line,
            rve_dims,
        )

        sphere.position_center += overlap_length * unit_vector
        intersection = sphere.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, unit_vector = sphere.intersection_length_mink_diff(
            line,
            rve_dims,
        )

        self.assertTrue(np.abs(overlap_length) < 1e-4)

    @unittest.skip("Failing, code incomplete")
    def test_point_ellipsoid(self):
        """Testing intersection between Point and Ellipsoid."""
        rve_dims = [1, 1, 1]
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.1,
                "axis_2": 0.15,
                "axis_3": 0.05,
                "rot_axis_comp_x": 0,
                "rot_axis_comp_y": 0,
                "rot_axis_comp_z": 1,
                "angle": np.pi / 4,
                "n": 1,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        line = Line("1", 1)
        line.position_center = np.array([0.55, 0.4, 0.5])

        intersection = ellipsoid.intersection_gjk(line, rve_dims)
        overlap_length, unit_vector = ellipsoid.intersection_length_mink_diff(
            line, rve_dims
        )

        # print(intersection, overlap_length, unit_vector)
        ellipsoid.position_center -= overlap_length * unit_vector
        # print(line.position_center + overlap_length * unit_vector)

        intersection = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, unit_vector = ellipsoid.intersection_length_mink_diff(
            line,
            rve_dims,
        )
        # print(intersection, overlap_length)

        self.assertTrue(np.abs(overlap_length) < 1e-4)

        line.position_center = np.array([0.5, 0.72, 0.6])
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        intersection = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        overlap_length, unit_vector = ellipsoid.intersection_length_mink_diff(
            line,
            rve_dims,
        )

        plot_particles_3d(
            [ellipsoid],
            rve_dims,
            "/home/jose/Documents/code/PC_ABS/PC_ABS_S30_V10_R1_5_44/mic_0",
        )
        # print(intersection, overlap_length, unit_vector)
        ellipsoid.position_center += (overlap_length) * unit_vector

        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        import matplotlib.pyplot as plt

        all = []
        all_min = []
        for (i_theta, j_phi) in (
            (i_theta, j_phi)
            for i_theta in np.linspace(-2 * np.pi, 2 * np.pi, 60)
            for j_phi in np.linspace(-2 * np.pi, 2 * np.pi, 120)
        ):
            search_direction = np.array(
                [
                    np.sin(i_theta) * np.cos(j_phi),
                    np.sin(i_theta) * np.sin(j_phi),
                    np.cos(i_theta),
                ]
            )
            mink_diff_point = ellipsoid.support_function(search_direction, proj=1) - (
                line.support_function(-search_direction, proj=1)
            )
            all.append(mink_diff_point.dot(search_direction))
            all_min.append(mink_diff_point)

        plt.scatter

        # print("here", intersection, overlap_length, np.min(all), np.max(all))

        # print(intersection, overlap_length, unit_vector)
        self.assertTrue(np.abs(overlap_length) < 1e-4)

        line.position_center = np.array([0.7, 0.7, 0.7])
        ellipsoid.position_center = np.array([0.5, 0.5, 0.5])
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )

        # print(intersection, overlap_length, (0.2 * np.sqrt(2) - 0.1))
        ellipsoid.position_center += overlap_length * unit_vector
        # print(ellipsoid.position_center)

        all = []
        for (i_theta, j_phi) in (
            (i_theta, j_phi)
            for i_theta in np.linspace(-2 * np.pi, 2 * np.pi, 60)
            for j_phi in np.linspace(-2 * np.pi, 2 * np.pi, 120)
        ):
            search_direction = np.array(
                [
                    np.sin(i_theta) * np.cos(j_phi),
                    np.sin(i_theta) * np.sin(j_phi),
                    np.cos(i_theta),
                ]
            )
            mink_diff_point = ellipsoid.support_function(search_direction) - (
                line.support_function(-search_direction)
            )
            all.append(mink_diff_point.dot(search_direction))

        # print("here", intersection, overlap_length, np.min(all), np.max(all))
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        # print(intersection, overlap_length, (0.2 * np.sqrt(2) - 0.1))
        self.assertTrue(np.abs(overlap_length) < 1e-4)

    @unittest.skip("Failing, code incomplete")
    def test_point_ellipsoid_2(self):
        """Testing intersection between Point and Ellipsoid."""
        rve_dims = [1, 1, 1]
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.17716559698623754,
                "axis_2": 0.11929383740508366,
                "axis_3": 0.1408479066730546,
                "rot_axis_comp_x": 0.0,
                "rot_axis_comp_y": 0.45698723,
                "rot_axis_comp_z": 0.88947326,
                "angle": 0.02669837755257011,
                "n": 1,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.25498361, 0.7627365, 0.40777455])
        line = Line("1", 0)
        line.position_center = np.array([0.58565103, 0.79594423, 0.39845346])
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        plot_particles_3d(
            [ellipsoid],
            rve_dims,
            "/home/jose/Documents/code/PC_ABS/PC_ABS_S30_V10_R1_5_44/mic_0",
        )
        # print(intersection, overlap_length)
        ellipsoid.position_center -= overlap_length * unit_vector
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        # print(intersection, overlap_length)
        self.assertTrue(np.abs(overlap_length) < 1e-4)

    @unittest.skip("Failing, code incomplete")
    def test_point_ellipsoid_3(self):
        """Testing intersection between Point and Ellipsoid."""
        rve_dims = [1, 1, 1]
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.1988998758673037,
                "axis_2": 0.12787941802264405,
                "axis_3": 0.13350690688658415,
                "rot_axis_comp_x": 0.0,
                "rot_axis_comp_y": 0.36900528,
                "rot_axis_comp_z": -0.9294273,
                "angle": 0.14955765822410882,
                "n": 1,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.25508093, 0.90021559, 0.07257134])
        line = Line("1", 1)
        line.position_center = np.array([0.45564597, 0.40471394, 0.35922883])
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        plot_particles_3d(
            [ellipsoid],
            rve_dims,
            "/home/jose/Documents/code/PC_ABS/PC_ABS_S30_V10_R1_5_44/mic_0",
        )
        # print(intersection, overlap_length, unit_vector)
        line.position_center -= overlap_length * unit_vector
        # print(line.position_center)
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        # print(intersection, overlap_length)
        self.assertTrue(np.abs(overlap_length) < 1e-4)

    @unittest.skip("Failing, code incomplete")
    def test_point_ellipsoid_4(self):
        """Testing intersection between Point and Ellipsoid."""
        rve_dims = [1, 1, 1]
        ellipsoid = Ellipsoid(
            "1",
            {
                "axis_1": 0.13515468348411508,
                "axis_2": 0.09561599577868922,
                "axis_3": 0.0971164152587521,
                "rot_axis_comp_x": 0.0,
                "rot_axis_comp_y": -0.80684776,
                "rot_axis_comp_z": -0.59075942,
                "angle": -0.15208305267292058,
                "n": 1,
            },
            rve_dims,
        )
        ellipsoid.position_center = np.array([0.93834649, 0.82398477, 0.91110018])
        line = Line("1", 1)
        line.position_center = np.array([0.19368941, 0.15779493, 0.44256207])
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        plot_particles_3d(
            [ellipsoid],
            rve_dims,
            "/home/jose/Documents/code/PC_ABS/PC_ABS_S30_V10_R1_5_44/mic_0",
        )
        # print(intersection, overlap_length, unit_vector)
        line.position_center += overlap_length * unit_vector
        # print(line.position_center)
        intersection, overlap_length, unit_vector = ellipsoid.intersection_gjk(
            line,
            rve_dims,
        )
        # print(intersection, overlap_length)
        self.assertTrue(np.abs(overlap_length) < 1e-4)
