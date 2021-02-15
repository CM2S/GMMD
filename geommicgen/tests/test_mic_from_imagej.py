import unittest
from unittest.mock import sentinel, Mock, patch

# from microstructure.phase import Phase

import numpy as np
from scipy.optimize import newton_krylov, anderson, root
from micgenmethod.mic_from_imagej_3 import (
    qmc_em_size_param_estimation,
    ellip_func,
    generating_samples,
)
from scipy.stats import lognorm, norm, vonmises


class TestMainFunction(unittest.TestCase):
    """Test class for the Microstructure class."""

    # @patch("micgenmethod.mic_from_imagej.generate_sobol_samples")
    # def test_len(self, mock_generate_sobol_samples):
    #
    #     visible_vars = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def setUp(self):

        self.d_1_func = lambda r_1, r_2, r_3, z_c, p_1, p_2, p_3: np.sqrt(
            (
                8
                * (
                    1
                    - (z_c ** 2)
                    / (
                        r_1 ** 2 * np.sin(p_1) ** 2
                        + r_2 ** 2 * np.sin(p_2) ** 2
                        + r_3 ** 2 * np.sin(p_3) ** 2
                    )
                )
            )
            / (
                (
                    np.cos(p_1) ** 2 / r_1 ** 2
                    + np.cos(p_2) ** 2 / r_2 ** 2
                    + np.cos(p_3) ** 2 / r_3 ** 2
                )
                + np.sqrt(
                    (
                        np.cos(p_1) ** 2 / r_1 ** 2
                        + np.cos(p_2) ** 2 / r_2 ** 2
                        + np.cos(p_3) ** 2 / r_3 ** 2
                    )
                    ** 2
                    - 4
                    * (
                        np.sin(p_1) ** 2 / r_2 ** 2 / r_3 ** 2
                        + np.sin(p_2) ** 2 / r_1 ** 2 / r_3 ** 2
                        + np.sin(p_3) ** 2 / r_1 ** 2 / r_2 ** 2
                    )
                )
            )
        )

        self.d_2_func = lambda r_1, r_2, r_3, z_c, p_1, p_2, p_3: np.sqrt(
            (
                8
                * (
                    1
                    - (z_c ** 2)
                    / (
                        r_1 ** 2 * np.sin(p_1) ** 2
                        + r_2 ** 2 * np.sin(p_2) ** 2
                        + r_3 ** 2 * np.sin(p_3) ** 2
                    )
                )
            )
            / (
                (
                    np.cos(p_1) ** 2 / r_1 ** 2
                    + np.cos(p_2) ** 2 / r_2 ** 2
                    + np.cos(p_3) ** 2 / r_3 ** 2
                )
                - np.sqrt(
                    (
                        np.cos(p_1) ** 2 / r_1 ** 2
                        + np.cos(p_2) ** 2 / r_2 ** 2
                        + np.cos(p_3) ** 2 / r_3 ** 2
                    )
                    ** 2
                    - 4
                    * (
                        np.sin(p_1) ** 2 / r_2 ** 2 / r_3 ** 2
                        + np.sin(p_2) ** 2 / r_1 ** 2 / r_3 ** 2
                        + np.sin(p_3) ** 2 / r_1 ** 2 / r_2 ** 2
                    )
                )
            )
        )

    def test_func_params(self):
        """Test if the functions regarding the ellipsoid sections are working correctly."""

        # The section hits the extremety of the ellipsoid
        z_c = 0.6
        r_1 = 0.4
        r_2 = 0.5
        r_3 = 0.6
        p_1 = 0
        p_2 = 0
        p_3 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_2) ** 2))
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)

        self.assertTrue(np.abs(d_1) < 1e-4)
        self.assertTrue(np.abs(d_2) < 1e-4)

        # The section hits the ellipsoid in the middle
        z_c = 0
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)

        self.assertTrue(np.abs(d_1 - 2 * r_1) / (2 * r_1) < 1e-4)
        self.assertTrue(np.abs(d_2 - 2 * r_2) / (2 * r_2) < 1e-4)

    def test_params(self):
        """Testing solution of non-linear system, regarding the parameters of the ellipsoid.

        Not working for spheres. In general, when two axis are very close.
        """

        def angle_to_0_half_pi(angle):

            angle_0_pi = angle - np.floor(angle / (np.pi)) * np.pi
            angle_0_half_pi = (
                -1 * (angle_0_pi - np.pi) if angle_0_pi > np.pi / 2 else angle_0_pi
            )

            return angle_0_half_pi

        z_c = 0.1
        r_1 = 0.5
        r_2 = 0.4
        r_3 = 0.6
        p_1 = np.pi / 5
        p_2 = np.pi / 6
        p_3 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_2) ** 2))
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)

        curr_ellip_func = ellip_func(r_1, r_2, r_3, p_3, d_1, d_2)
        ellip_adapt = lambda x: curr_ellip_func(x[0], x[1], x[2])
        # sol_p_1, sol_p_2, sol_z_c
        sol = root(
            ellip_adapt,
            [
                np.pi / 4,
                np.pi / 4,
                np.sqrt(r_3 ** 2 * np.sin(p_3) + r_2 ** 2 + r_1 ** 2) / 2,
            ],
            method="hybr",
        )

        [sol_p_1, sol_p_2, sol_z_c] = sol.x
        sol_p_1_f = angle_to_0_half_pi(sol_p_1)
        sol_p_2_f = angle_to_0_half_pi(sol_p_2)
        self.assertTrue(np.abs(sol_p_1_f - p_1) / p_1 < 1e-4)
        self.assertTrue(np.abs(sol_p_2_f - p_2) / p_2 < 1e-4)
        self.assertTrue(np.abs(np.abs(sol_z_c) - z_c) / z_c < 1e-4)

        z_c = 0.15
        r_1 = 0.5
        r_2 = 0.4
        r_3 = 0.6
        p_1 = np.pi / 3
        p_2 = np.pi / 10
        p_3 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_2) ** 2))
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)

        curr_ellip_func = ellip_func(r_1, r_2, r_3, p_3, d_1, d_2)
        ellip_adapt = lambda x: curr_ellip_func(x[0], x[1], x[2])
        # sol_p_1, sol_p_2, sol_z_c
        sol = root(
            ellip_adapt,
            [
                np.pi / 4,
                np.pi / 4,
                np.sqrt(r_3 ** 2 * np.sin(p_3) + r_2 ** 2 + r_1 ** 2) / 2,
            ],
            method="hybr",
        )

        [sol_p_1, sol_p_2, sol_z_c] = sol.x
        sol_p_1_f = angle_to_0_half_pi(sol_p_1)
        sol_p_2_f = angle_to_0_half_pi(sol_p_2)
        self.assertTrue(np.abs(sol_p_1_f - p_1) / p_1 < 1e-4)
        self.assertTrue(np.abs(sol_p_2_f - p_2) / p_2 < 1e-4)
        self.assertTrue(np.abs(np.abs(sol_z_c) - z_c) / z_c < 1e-4)

        z_c = 0.1
        r_1 = 0.2
        r_2 = 0.3
        r_3 = 0.6
        p_1 = np.pi / 3
        p_2 = np.pi / 10
        p_3 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_2) ** 2))
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        print(d_1, d_2)

        curr_ellip_func = ellip_func(r_1, r_2, r_3, p_3, d_1, d_2)
        ellip_adapt = lambda x: curr_ellip_func(x[0], x[1], x[2])
        # sol_p_1, sol_p_2, sol_z_c
        sol = root(
            ellip_adapt,
            [
                np.pi / 4,
                np.pi / 4,
                np.sqrt(r_3 ** 2 * np.sin(p_3) ** 2 + r_2 ** 2 + r_1 ** 2) / 2,
            ],
            method="hybr",
        )

        [sol_p_1, sol_p_2, sol_z_c] = sol.x
        sol_p_1_f = angle_to_0_half_pi(sol_p_1)
        sol_p_2_f = angle_to_0_half_pi(sol_p_2)
        self.assertTrue(np.abs(sol_p_1_f - p_1) / p_1 < 1e-4)
        self.assertTrue(np.abs(sol_p_2_f - p_2) / p_2 < 1e-4)
        self.assertTrue(np.abs(np.abs(sol_z_c) - z_c) / z_c < 1e-4)

    def test_1(self):

        r_1 = 0.5
        r_2 = 0.4
        r_3 = 0.6
        p_1 = np.pi / 3
        p_2 = np.pi / 10
        p_3 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_2) ** 2))
        np.random.seed(42)
        # z_c = np.random.uniform(low=0, high=0.55, 10)
        z_c = 0.15
        d_1 = self.d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        d_2 = self.d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3)
        print(d_1, d_2)
        visible_vars = [[d_1, d_2]]
        b = qmc_em_size_param_estimation(visible_vars, [0.6, 0.1, 10, 3, 10, 3])

        print(b)

    def test_2(self):
        file_path = (
            "/home/jose/Documents/code/paper_results/stat_analysis/3D/Results.csv"
        )
        info = np.genfromtxt(file_path, delimiter=",", skip_header=1)
        visible_vars = info[:, 7:9] / 795
        print(visible_vars)
        angles = info[:, -2] * np.pi / 180
        # for i_ind, i_angle in angles:
        #     if i_angle > n
        visible_vars = np.array([visible_vars[:, 1], visible_vars[:, 0]]).T
        val = np.abs(np.sum([np.exp(1j * i_angle) for i_angle in angles])) / len(angles)
        angle_av = np.angle(np.sum([np.exp(1j * i_angle) for i_angle in angles]))
        print(angle_av, val, np.sqrt(-2 * np.log(val)))
        print(angles)
        import matplotlib.pyplot as plt

        kappa, loc, scale = vonmises.fit(angles, fscale=1 / 2)
        print(kappa, loc, scale)
        x = np.linspace(
            0,
            np.pi,
            100,
        )
        vals = vonmises.pdf(x, kappa, loc=loc, scale=scale)
        print(x)
        print(vals)
        plt.plot(x, vals)
        plt.hist(angles, density=True)
        plt.show()

        plt.hist(visible_vars[:, 1], density=True)
        sigma, loc, scale = lognorm.fit(visible_vars[:, 1], floc=0)
        print("lgonorm_param", sigma, loc, scale)
        x = np.linspace(0, np.max(visible_vars[:, 1]), 50)
        vals = lognorm.pdf(x, sigma, loc=loc, scale=scale)
        plt.plot(x, vals)
        print(lognorm.rvs(sigma, loc=loc, scale=scale, size=20))

        plt.show()

        plt.hist(visible_vars[:, 0] / visible_vars[:, 1], density=True)
        loc, scale = norm.fit(visible_vars[:, 0] / visible_vars[:, 1])
        print(loc, scale)
        x = np.linspace(0, 1, 50)
        vals = norm.pdf(x, loc=loc, scale=scale)
        plt.plot(x, vals)
        plt.show()
        # b = qmc_em_size_param_estimation(visible_vars, [-2, 0.01, 0.5, 0.1, 1, 0.1])

    def test_3(self):
        visible_vars = generating_samples()
        b = qmc_em_size_param_estimation(visible_vars, [-2, 0.1, 0.5, 0.4])
