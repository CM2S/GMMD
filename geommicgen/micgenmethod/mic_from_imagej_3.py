""" Module to import a microstructure form a csv file generated from ImageJ. """
import numpy as np
import os

# Quasi monte carlo integration
import qmcpy as qp
from scipy.optimize import newton_krylov, root, minimize
from scipy.stats import lognorm, beta, norm
import pyswarms as ps
import multiprocessing as mp


# pylint: disable=import-error
import iofuncs.printing as print_funcs
from microstructure.phase import Phase
from microstructure.microstructure import Microstructure


def ellip_func(r_1, r_2, r_3, p_3, d_1, d_2):

    func = lambda p_1, p_2, z_c: [
        d_1 ** 2
        - (
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
                np.max(
                    [
                        0,
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
                        ),
                    ]
                )
            )
        ),
        d_2 ** 2
        - (
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
                np.max(
                    [
                        0,
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
                        ),
                    ]
                )
            )
        ),
        np.sin(p_1) ** 2 + np.sin(p_2) ** 2 + np.sin(p_3) ** 2 - 1,
    ]

    return func


def d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3):
    d_1 = np.sqrt(
        (8 * (1 - (z_c ** 2) / r_2 ** 2))
        / (
            ((1 / r_2 ** 2 + 1 / r_3 ** 2))
            + np.sqrt(((1 / r_2 ** 2 + 1 / r_3 ** 2)) ** 2 - 4 / (r_2 ** 2 * r_3 ** 2))
        )
    )
    # d_1 = np.sqrt(
    #     (
    #         8
    #         * (
    #             1
    #             - (z_c ** 2)
    #             / (
    #                 r_1 ** 2 * np.sin(p_1) ** 2
    #                 + r_2 ** 2 * np.sin(p_2) ** 2
    #                 + r_3 ** 2 * np.sin(p_3) ** 2
    #             )
    #         )
    #     )
    #     / (
    #         (
    #             np.cos(p_1) ** 2 / r_1 ** 2
    #             + np.cos(p_2) ** 2 / r_2 ** 2
    #             + np.cos(p_3) ** 2 / r_3 ** 2
    #         )
    #         + np.sqrt(
    #             (
    #                 np.cos(p_1) ** 2 / r_1 ** 2
    #                 + np.cos(p_2) ** 2 / r_2 ** 2
    #                 + np.cos(p_3) ** 2 / r_3 ** 2
    #             )
    #             ** 2
    #             - 4
    #             * (
    #                 np.sin(p_1) ** 2 / r_2 ** 2 / r_3 ** 2
    #                 + np.sin(p_2) ** 2 / r_1 ** 2 / r_3 ** 2
    #                 + np.sin(p_3) ** 2 / r_1 ** 2 / r_2 ** 2
    #             )
    #         )
    #     )
    # )
    return d_1


def d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3):
    # d_2 = np.sqrt(
    #     (
    #         8
    #         * (
    #             1
    #             - (z_c ** 2)
    #             / (
    #                 r_1 ** 2 * np.sin(p_1) ** 2
    #                 + r_2 ** 2 * np.sin(p_2) ** 2
    #                 + r_3 ** 2 * np.sin(p_3) ** 2
    #             )
    #         )
    #     )
    #     / (
    #         (
    #             np.cos(p_1) ** 2 / r_1 ** 2
    #             + np.cos(p_2) ** 2 / r_2 ** 2
    #             + np.cos(p_3) ** 2 / r_3 ** 2
    #         )
    #         - np.sqrt(
    #             (
    #                 np.cos(p_1) ** 2 / r_1 ** 2
    #                 + np.cos(p_2) ** 2 / r_2 ** 2
    #                 + np.cos(p_3) ** 2 / r_3 ** 2
    #             )
    #             ** 2
    #             - 4
    #             * (
    #                 np.sin(p_1) ** 2 / r_2 ** 2 / r_3 ** 2
    #                 + np.sin(p_2) ** 2 / r_1 ** 2 / r_3 ** 2
    #                 + np.sin(p_3) ** 2 / r_1 ** 2 / r_2 ** 2
    #             )
    #         )
    #     )
    # )
    d_2 = np.sqrt(
        (8 * (1 - (z_c ** 2) / r_2 ** 2))
        / (
            ((1 / r_2 ** 2 + 1 / r_3 ** 2))
            - np.sqrt(((1 / r_2 ** 2 + 1 / r_3 ** 2)) ** 2 - 4 / (r_1 ** 2 * r_3 ** 2))
        )
    )

    return d_2


def compute_weights(sample, vis_vars):
    def jacobian(r_1, r_2, r_3, p_1, p_2, p_3, z_c, d_1, d_2):
        c_1 = (
            r_1 ** 2 * np.sin(p_1) ** 2
            + r_2 ** 2 * np.sin(p_2) ** 2
            + r_3 ** 2 * np.sin(p_3) ** 2
        )
        c_2 = (
            np.cos(p_1) ** 2 / r_1 ** 2
            + np.cos(p_2) ** 2 / r_2 ** 2
            + np.cos(p_3) ** 2 / r_3 ** 2
        )
        c_prime_1 = (r_1 ** 2 - r_2 ** 2) * np.sin(2 * p_1)
        c_prime_2 = (1 / r_2 ** 2 - 1 / r_1 ** 2) * np.sin(2 * p_1)

        jac = (
            (2 * d_1 * d_2 * r_1 ** 2 * r_2 ** 2 * r_3 ** 2 * z_c)
            / ((d_2 ** 2 - d_1 ** 2) * c_1 ** 3)
            * (2 * c_1 * c_prime_2 - c_2 * c_prime_1)
        )
        return jac

    def angle_to_0_half_pi(angle):

        angle_0_pi = angle - np.floor(angle / (np.pi)) * np.pi
        angle_0_half_pi = (
            -1 * (angle_0_pi - np.pi) if angle_0_pi > np.pi / 2 else angle_0_pi
        )

        return angle_0_half_pi

    p_3 = 0
    r_1, r_2, r_3 = sample
    d_1, d_2 = vis_vars
    # curr_ellip_func = ellip_func(r_1, r_2, r_3, p_3, d_1, d_2)
    #
    # ellip_adapt = lambda x: curr_ellip_func(x[0], x[1], x[2])
    # sol = root(
    #     ellip_adapt,
    #     [
    #         np.pi / 4,
    #         np.pi / 4,
    #         np.sqrt(r_3 ** 2 * np.sin(p_3) ** 2 + r_2 ** 2 + r_1 ** 2) / 2,
    #     ],
    #     method="hybr",
    # )
    coeff_2 = (
        d_1 ** 2
        * d_2 ** 2
        * r_1 ** 2
        * r_2 ** 2
        * r_3 ** 4
        / ((d_1 ** 2 + d_2 ** 2) ** 2)
    )
    coeff_1 = -(r_1 ** 2) * r_2 ** 2 * r_3 ** 2
    coeff_0 = (
        np.sin(p_3) ** 2 * ((r_3 ** 2 - r_1 ** 2) * (r_2 ** 2 - r_3 ** 2))
        + r_1 ** 2 * r_2 ** 2
    )
    # coeff_2 = (
    #     r_1 ** 2
    #     * r_2 ** 2
    #     * r_3 ** 2
    #     * (d_1 ** 2 - d_2 ** 2) ** 2
    #     / (4 * (d_1 ** 2 + d_2 ** 2) ** 2)
    # )
    # coeff_1 = r_1 ** 2 * r_2 ** 2 * r_3 ** 2 / 4 - r_2 ** 2 * r_1 ** 2
    # coeff_0 = (
    #     r_1 ** 2 / r_3 ** 2 * (r_3 ** 2 - r_2 ** 2) * np.sin(p_3) ** 2
    #     + r_2 ** 2 / r_3 ** 2 * (r_3 ** 2 - r_1 ** 2)
    #     - (r_3 ** 2 - r_2 ** 2) * np.sin(p_3) ** 2
    #     - r_2 ** 2
    # )
    C_2 = (1 / r_2 ** 2 - 1 / r_3 ** 2) * np.sin(p_3) ** 2 + (
        1 / r_1 ** 2 + 1 / r_3 ** 2
    )
    if 1 / r_2 ** 2 + 1 / r_3 ** 2 <= C_2 <= 1 / r_1 ** 2 + 1 / r_2 ** 2:
        pass
    elif 1 / r_2 ** 2 + 1 / r_3 ** 2 <= C_2 <= 1 / r_1 ** 2 + 1 / r_2 ** 2:
        pass
    else:
        C_2 = np.nan

    C_1 = (
        C_2 ** 2
        * d_1 ** 2
        * d_2 ** 2
        * r_1 ** 2
        * r_2 ** 2
        * r_3 ** 2
        / (d_1 ** 2 + d_2 ** 2) ** 2
    )
    # C_1 = r_1 ** 2 * r_2 ** 2 * r_3 ** 2 / 4 * C_2 + r_1 ** 2 * r_2 ** 2 * r_3 ** 2 * (
    #     d_1 ** 2 - d_2 ** 2
    # ) ** 2 * C_2 ** 2 / (4 * (d_1 ** 2 + d_2 ** 2) ** 2)
    z_c = np.sqrt((1 - np.sqrt(C_1) * d_1 * d_2 / (4 * r_1 * r_2 * r_3)) * C_1)
    p_1 = 0
    # p_2 =
    # p_1 = np.arcsin(
    #     np.sqrt(
    #         (C_1 - r_2 ** 2 - (r_3 ** 2 - r_2 ** 2) * np.sin(p_3) ** 2)
    #         / (r_1 ** 2 - r_2 ** 2)
    #     )
    # )
    p_2 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_3) ** 2))
    # C_1_p = (
    #     r_1 ** 2 * np.sin(p_1) ** 2
    #     + r_2 ** 2 * np.sin(p_2) ** 2
    #     + r_3 ** 2 * np.sin(p_3) ** 2
    # )
    # C_2_p = (
    #     1 / r_1 ** 2 * np.cos(p_1) ** 2
    #     + 1 / r_2 ** 2 * np.cos(p_2) ** 2
    #     + 1 / r_3 ** 2 * np.cos(p_3) ** 2
    # )
    # print("C_1", C_1, C_1_p)
    # print("C_2", C_2, C_2_p)
    if C_2 == np.nan:  # sol.success:
        # sol_p_1, sol_p_2, sol_z_c = sol.x
        # p_1 = angle_to_0_half_pi(sol_p_1)
        # p_2 = angle_to_0_half_pi(sol_p_2)
        # z_c = np.abs(sol_z_c)
        weight = 0
    else:
        # print(p_1, p_2, p_3, d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3), d_1)
        # print(p_1, p_2, p_3, d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3), d_2)
        # weight = (
        #     2
        #     / np.pi
        #     * np.cos(p_1)
        #     / np.sin(p_2)
        #     * np.abs(jacobian(r_1, r_2, r_3, p_1, p_2, p_3, z_c, d_1, d_2)) ** (-1)
        # )
        C_3 = C_2 + np.sqrt(C_2 ** 2 - 4 * C_1 / (r_1 ** 2 * r_2 ** 2 * r_3 ** 2))
        weight = np.abs(-16 * z_c / C_1 / (2 * d_1 * C_3)) ** (-1)
        # print(weight)
    return weight


def generate_sobol_samples(n_sobol_samples, vis_vars, sobol_sequence, curr_k):

    r_max = 2
    weights = []
    sobol_samples = []
    k_sample = curr_k
    n_need = 1
    while len(weights) < n_sobol_samples:
        if k_sample == n_need:
            n_need *= 2
        sobol_sample_try = (
            np.array([0, vis_vars[1] / 2])
            + np.array([1, r_max - vis_vars[1] / 2]) * sobol_sequence[k_sample]
        )
        r_3 = sobol_sample_try[1]
        r_2 = r_3 * sobol_sample_try[0]
        # r_1 = r_2 * sobol_sample_try[0]
        r_1 = r_3 * sobol_sample_try[0]
        # print(r_3, r_2, r_1)
        weight = compute_weights([r_1, r_2, r_3], vis_vars)
        if np.abs(weight) > 1e-4:
            weights.append(weight)
            sobol_samples.append(sobol_sample_try)
            # print(len(weights), n_sobol_samples)
        k_sample += 1

    return weights, sobol_samples, curr_k


def param_dist(curr_param_estimates, samples):
    r_1, r_3 = samples
    mu, sigma, mu_ratio, sigma_ratio = curr_param_estimates
    term_1 = lognorm.pdf(r_3, sigma, scale=np.exp(mu))
    # term_1 = norm.pdf(r_3, loc=mu, scale=sigma)
    # term_2 = beta.pdf(r_1 / r_2, alpha_1, beta_1)
    # term_3 = beta.pdf(r_2 / r_3, alpha_2, beta_2)
    term_2 = norm.pdf(r_1 / r_3, loc=mu_ratio, scale=sigma_ratio)

    complete_pdf = term_1 * term_2
    # print(complete_pdf, term_1, term_2, term_3)
    # print("r", r_3, r_1 / r_3)
    # print("param", mu, sigma)
    return complete_pdf


def expct_log_likelihood(curr_param_estimates, weights, sobol_samples):
    def expected_val_z_u_m_c(curr_param_estimates):

        mu, sigma, mu_ratio, sigma_ratio = curr_param_estimates
        samples_z_u = []
        k_iter = 0
        while k_iter < 10000:
            # p_3_sample = np.arcsin(np.random.rand())
            # p_2_sample = np.arcsin(
            #     np.cos(p_3_sample) * np.sin(np.pi * np.random.rand() / 2)
            # )
            # r_1_r_2_ratio_sample = beta.rvs(a_1, b_1)
            # r_2_r_3_ratio_sample = beta.rvs(a_2, b_2)
            r_2_r_3_ratio_sample = norm.rvs(loc=mu_ratio, scale=sigma_ratio)
            p_3_sample = 0
            samples_z_u.append(
                np.sqrt(
                    (1 - r_2_r_3_ratio_sample ** 2) * np.sin(p_3_sample) ** 2
                    + r_2_r_3_ratio_sample ** 2
                )
            )
            mean_z_u = np.mean(samples_z_u)
            if len(samples_z_u) > 1:
                std_z_u = 1 / np.sqrt(len(samples_z_u)) * np.std(samples_z_u)
                if std_z_u < 1e-4:
                    break
            k_iter += 1

        # lognomal r_3
        expect_z_u = (mu + sigma ** 2 / 2) + np.log(mean_z_u)
        # print(expect_z_u)
        # normal r_3
        # expect_z_u = np.log(mu) + np.log(mean_z_u)

        return expect_z_u

    # print("curr_param_estimates", curr_param_estimates, len(curr_param_estimates))
    pdf = np.full(sobol_samples.shape, 0.0)
    n_particles = sobol_samples.shape[0]
    n_sobol_samples = sobol_samples.shape[1]
    for (i_part, s_sample) in (
        (i_part, s_sample)
        for i_part in range(n_particles)
        for s_sample in range(n_sobol_samples)
    ):
        # Calculate the weights for the qmc integration
        r_3 = sobol_samples[i_part, s_sample][1]
        # print(r_3)
        r_2 = r_3 * sobol_samples[i_part, s_sample][0]
        pdf[i_part, s_sample] = param_dist(curr_param_estimates, [r_2, r_3])

    log_expected_val_z_u = expected_val_z_u_m_c(curr_param_estimates)
    expct_log_likelihood_val = (
        np.sum(weights * np.log(pdf)) - n_particles * log_expected_val_z_u
    )
    # print(np.sum(weights * np.log(pdf)), n_particles * log_expected_val_z_u)
    # print(curr_param_estimates, expct_log_likelihood_val)

    # print(curr_param_estimates, expct_log_likelihood_val)
    return np.min([1e12, -1 * expct_log_likelihood_val])


def expct_log_likelihood_vec(curr_param_estimates_vec, weights, sobol_samples):

    res = []
    for i_param in curr_param_estimates_vec:
        res.append(expct_log_likelihood(i_param, weights, sobol_samples))
    # for i_param in curr_param_estimates_vec:
    # pool = mp.Pool(mp.cpu_count() - 2)
    # result_async = [
    #     pool.apply_async(expct_log_likelihood, args=(i_param, weights, sobol_samples))
    #     for i_param in curr_param_estimates_vec
    # ]
    # res = [r.get() for r in result_async]
    #
    # pool.close()
    return np.array(res)


def qmc_em_size_param_estimation(
    visible_vars, init_param_estimates, max_iter=50, tol=0.015, n_sobol_samples=400
):

    n_particles = len(visible_vars)
    is_weights = np.full((n_particles, n_sobol_samples), 0.0)
    sobol_samples = np.full((n_particles, n_sobol_samples), 0, dtype=object)

    # Generate Sobol samples
    # --------------------------------------------------------------------------------------
    sobol_sequence_obj = qp.Sobol(2)
    n_samp_pt_needed = 2 ** np.ceil(np.log2(n_sobol_samples * n_particles * 10))
    sobol_sequence = sobol_sequence_obj.gen_samples(n_samp_pt_needed)
    curr_k = 0
    for i_part in range(n_particles):
        # Generate Sobol samples \(\pmb h_i^{(s)}\) until \(S\) samples with nonzero weights
        # are obtained
        print(i_part)
        (
            is_weights[i_part, :],
            sobol_samples[i_part, :],
            curr_k,
        ) = generate_sobol_samples(
            n_sobol_samples, visible_vars[i_part], sobol_sequence, curr_k
        )

    curr_param_estimates = list(init_param_estimates)
    # EM algorithm
    # --------------------------------------------------------------------------------------
    bounds = ([-10, 0, 0, 0], [0, 5, 1, 1])
    options = {"c1": 2.025, "c2": 2.025, "w": 0.6}

    weights = np.array(is_weights)
    for (i_part, s_sample) in (
        (i_part, s_sample)
        for i_part in range(n_particles)
        for s_sample in range(n_sobol_samples)
    ):
        # Calculate the weights for the qmc integration
        r_3 = sobol_samples[i_part, s_sample][1]
        r_2 = r_3 * sobol_samples[i_part, s_sample][0]
        weights[i_part, s_sample] *= param_dist([-3, 1, 0.6666, 0.1], [r_2, r_3])
    # Maximization step
    normalized_weights = (
        weights
        / np.array([np.sum(weights, axis=1) for _ in enumerate(weights[0, :])]).T
    )
    print(
        "max",
        [-3, 1, 0.6666, 0.1],
        expct_log_likelihood([-3, 1, 0.6666, 0.1], normalized_weights, sobol_samples),
    )
    optmizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=4,
        options=options,
        bounds=bounds,
    )
    for k_iter in range(15):  # range(max_iter):
        # Estimation step
        weights = np.array(is_weights)
        for (i_part, s_sample) in (
            (i_part, s_sample)
            for i_part in range(n_particles)
            for s_sample in range(n_sobol_samples)
        ):
            # Calculate the weights for the qmc integration
            r_3 = sobol_samples[i_part, s_sample][1]
            r_2 = r_3 * sobol_samples[i_part, s_sample][0]
            weights[i_part, s_sample] *= param_dist(curr_param_estimates, [r_2, r_3])
        normalized_weights = (
            weights
            / np.array([np.sum(weights, axis=1) for _ in enumerate(weights[0, :])]).T
        )
        # Maximization step
        old_param_estimates = list(curr_param_estimates)
        # curr_param_estimates += np.full(6, 0.25) - 0.5 * np.random.rand(6)

        _, curr_param_estimates = optmizer.optimize(
            expct_log_likelihood_vec,
            100,
            weights=normalized_weights,
            sobol_samples=sobol_samples,
            n_processes=15,
        )
        # res = minimize(
        #     expct_log_likelihood,
        #     x0=np.array(old_param_estimates_1),
        #     args=(normalized_weights, sobol_samples),
        #     method="Powell",
        #     options={
        #         "disp": True,
        #         "maxiter": 100,
        #         "ftol": 0.1,
        #         "xtol": 0.1,
        #         # "direc": np.diag([1, 1, 1, 1, 1, 1]),
        #     },
        #     bounds=[(-10, 0), (0, 5), (0, 1), (0, 1), (-1, 1), (0, 1)]
        #     # [-10, 0, 0, 0, -1, 0], [0, 5, 1, 1, 1, 1]
        # )
        # # Stopping criterion
        # # print(old_param_estimates)
        # # print(curr_param_estimates)
        # # print(
        # #     np.max((old_param_estimates - curr_param_estimates) / curr_param_estimates)
        # # )
        # print(res)
        # curr_param_estimates = res.x
        if (
            np.max(
                np.abs(old_param_estimates - curr_param_estimates)
                / np.abs(curr_param_estimates)
            )
            < tol
        ) or k_iter > max_iter:
            break

    return curr_param_estimates


def generating_samples():
    # Statiscal distribution parameters
    # --------------------------------------------------------------------------------------
    mu = -3
    sigma = 1
    mu_ratio = 0.66
    sigma_ratio = 0.1
    mu_dir = 0
    sigma_dir = 0.2

    # Sample sizes
    # --------------------------------------------------------------------------------------
    n_sample_size = 100
    n_particles = 1000

    # Sample vals
    # --------------------------------------------------------------------------------------

    # r_3_sample_vals = lognorm.rvs(sigma, scale=np.exp(mu), size=n_particles)
    # r_2_r_3_sample_vals = beta.rvs(alpha_1, beta_1, size=n_particles)
    # r_2_sample_vals = r_3_sample_vals * r_2_r_3_sample_vals
    # r_1_r_2_sample_vals = beta.rvs(alpha_2, beta_2, size=n_particles)
    # r_1_sample_vals = r_2_sample_vals * r_1_r_2_sample_vals
    # p_3_sample_vals = np.arcsin(np.random.rand(n_particles))
    # p_2_sample_vals = np.arcsin(
    #     np.cos(p_3_sample_vals) * np.sin(np.pi * np.random.rand(n_particles) / 2)
    # )
    # p_1_sample_vals = np.arcsin(
    #     np.sqrt(1 - np.sin(p_2_sample_vals) ** 2 - np.sin(p_3_sample_vals) ** 2)
    # )
    # r_3_sample_vals = norm.rvs(loc=mu, scale=sigma, size=n_particles)
    r_3_sample_vals = lognorm.rvs(sigma, scale=np.exp(mu), size=n_particles)
    r_2_r_3_sample_vals = norm.rvs(loc=mu_ratio, scale=sigma_ratio, size=n_particles)
    r_2_sample_vals = r_3_sample_vals * r_2_r_3_sample_vals
    print(r_3_sample_vals, r_2_sample_vals)
    r_1_sample_vals = r_2_sample_vals
    p_3_sample_vals = np.full(
        (n_particles), 0
    )  # norm.rvs(loc=mu_dir, scale=sigma_dir, size=n_particles)
    p_2_sample_vals = np.arcsin(
        np.cos(p_3_sample_vals) * np.sin(np.pi * np.random.rand(n_particles) / 2)
    )
    p_1_sample_vals = np.arcsin(
        np.sqrt(1 - np.sin(p_2_sample_vals) ** 2 - np.sin(p_3_sample_vals) ** 2)
    )
    z_u = np.sqrt(
        r_1_sample_vals ** 2 * np.sin(p_1_sample_vals) ** 2
        + r_2_sample_vals ** 2 * np.sin(p_2_sample_vals) ** 2
        + r_3_sample_vals ** 2 * np.sin(p_3_sample_vals) ** 2
    )
    # discrete distribution
    particles_samples = np.random.choice(
        n_particles, n_sample_size, p=z_u / np.sum(z_u)
    )

    d_1_sample_vals = []
    d_2_sample_vals = []
    z_c_sample_vals = []
    for i_particle in particles_samples:
        # for each sample sample z_c from uniform of z_u
        z_c_current = np.random.uniform(0, z_u[i_particle])
        z_c_sample_vals.append(z_c_current)
        d_1_sample_vals.append(
            d_1_func(
                r_1_sample_vals[i_particle],
                r_2_sample_vals[i_particle],
                r_3_sample_vals[i_particle],
                z_c_current,
                p_1_sample_vals[i_particle],
                p_2_sample_vals[i_particle],
                p_3_sample_vals[i_particle],
            )
        )
        d_2_sample_vals.append(
            d_2_func(
                r_1_sample_vals[i_particle],
                r_2_sample_vals[i_particle],
                r_3_sample_vals[i_particle],
                z_c_current,
                p_1_sample_vals[i_particle],
                p_2_sample_vals[i_particle],
                p_3_sample_vals[i_particle],
            )
        )
        print(d_1_sample_vals[-1], d_2_sample_vals[-1])

    vis_vars = np.array([d_1_sample_vals, d_2_sample_vals]).T
    return vis_vars
