""" Module to import a microstructure form a csv file generated from ImageJ. """
import numpy as np
import os

# Quasi monte carlo integration
import qmcpy as qp
from scipy.optimize import newton_krylov, root
from scipy.stats import lognorm, beta, norm

# import pyswarms as ps
import multiprocessing as mp


# pylint: disable=import-error
import iofuncs.printing as print_funcs
from microstructure.phase import Phase
from microstructure.microstructure import Microstructure


def generate_microstructure_from_csv(file_path):
    """
    Generate the microstructure for the sample supplied.

    Generate the microstructure for microstructure_sample using the microstructure
    generation method *self*.

    Parameters
    ----------
    file_path: str
        File path of the form "shape_xdim_ydim_vf".
    """
    mic_info = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    # Loading the microstructure info. Assumed to be
    # N,Area,Mean,Min,Max,XM,YM,Major,Minor,Angle

    # Getting info from file name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_name = os.path.basename(file_path)
    try:
        shape, x_dim, y_dim, _ = file_name.split("_")
    except ValueError as error:
        raise ValueError(
            "Filename is not of the form 'shape_xdim_ydim_vf': {0}".format(file_name)
        ) from error
    try:
        x_dim = float(x_dim)
        y_dim = float(y_dim)

    except ValueError as error:
        raise ValueError(
            """The values in the filename for x_dim or y_dim are not numbers.': {0},
            {1}.""".format(
                x_dim, y_dim
            )
        ) from error

    box = [x_dim / np.max([x_dim, y_dim]), y_dim / np.max([x_dim, y_dim])]

    microstructure_sample = Microstructure(box)

    # Building descriptors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if shape == "disks":
        n_particles = len(mic_info)
        area_vals = []
        positions = []
        for i_particle_info in mic_info:
            area_vals.append(i_particle_info[1] / (x_dim * y_dim))
            positions.append(
                np.array(
                    [
                        i_particle_info[5] / np.max([x_dim, y_dim]),
                        box[1] - i_particle_info[6] / np.max([x_dim, y_dim]),
                    ]
                )
            )
            descriptors = {
                "phase_type": 2,
                "n": n_particles,
                "area": area_vals,
                "area_distribution": "specified",
            }
    else:
        n_particles = len(mic_info)
        major_axis_vals = []
        minor_axis_vals = []
        area_vals = []
        angle_vals = []
        positions = []
        for i_particle_info in mic_info:
            major_axis_vals.append(i_particle_info[7])
            minor_axis_vals.append(i_particle_info[8])
            angle_vals.append(i_particle_info[9] / 180 * np.pi)
            positions.append(
                np.array(
                    [
                        i_particle_info[5] / np.max([x_dim, y_dim]),
                        box[1] - i_particle_info[6] / np.max([x_dim, y_dim]),
                    ]
                )
            )
        major_axis_vals = np.array(major_axis_vals) / np.max([x_dim, y_dim])
        minor_axis_vals = np.array(minor_axis_vals) / np.max([x_dim, y_dim])
        ratio_vals = major_axis_vals / minor_axis_vals
        descriptors = {
            "phase_type": 3,
            "n": n_particles,
            "major_axis": major_axis_vals,
            "major_axis_distribution": "specified",
            # "minor_axis": minor_axis_vals,
            # "minor_axis_distribution": "specified",
            "ratio": ratio_vals,
            "ratio_distribution": "specified",
            "angle": angle_vals,
            "angle_distribution": "specified",
        }
    # Generating phases and particles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # microstructure_sample.add_phase(Phase("1", {"phase_type": 1}))
    microstructure_sample.add_phase(Phase("2", descriptors))
    for phase in microstructure_sample.phases.values():
        phase.generate_particles(microstructure_sample.rve_dims)
    # if microstructure_sample.volume_fraction > 1:
    #     raise ValueError(
    #         "The volume fraction goes over 1: {0}".format(
    #             microstructure_sample.volume_fraction
    #         )
    #     )
    # Setting the position of the particles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i_particle_ind, i_particle in enumerate(microstructure_sample.particles):
        i_particle.position_center = positions[i_particle_ind]
        print(positions[i_particle_ind])

    print_funcs.print_microstructure_info(microstructure_sample)

    return microstructure_sample


def d_1_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3):
    d_1 = np.sqrt(
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
    return d_1


def d_2_func(r_1, r_2, r_3, z_c, p_1, p_2, p_3):
    d_2 = np.sqrt(
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
    return d_2


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

    r_1, r_2, r_3, p_3 = sample
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
    roots = np.roots([coeff_2, coeff_1, coeff_0])
    if 1 / r_2 ** 2 + 1 / r_3 ** 2 <= roots[0] <= 1 / r_1 ** 2 + 1 / r_2 ** 2:
        C_2 = roots[0]
    elif 1 / r_2 ** 2 + 1 / r_3 ** 2 <= roots[1] <= 1 / r_1 ** 2 + 1 / r_2 ** 2:
        C_2 = roots[1]
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
    p_1 = np.arcsin(
        np.sqrt(
            (C_1 - r_2 ** 2 - (r_3 ** 2 - r_2 ** 2) * np.sin(p_3) ** 2)
            / (r_1 ** 2 - r_2 ** 2)
        )
    )
    p_2 = np.arcsin(np.sqrt(1 - np.sin(p_1) ** 2 - np.sin(p_3) ** 2))
    z_c = np.sqrt((1 - np.sqrt(C_1) * d_1 * d_2 / (4 * r_1 * r_2 * r_3)) * C_1)

    if C_2 == np.nan:  # sol.success:
        # sol_p_1, sol_p_2, sol_z_c = sol.x
        # p_1 = angle_to_0_half_pi(sol_p_1)
        # p_2 = angle_to_0_half_pi(sol_p_2)
        # z_c = np.abs(sol_z_c)
        weight = 0
    else:
        weight = (
            2
            / np.pi
            * np.cos(p_1)
            / np.sin(p_2)
            * np.abs(jacobian(r_1, r_2, r_3, p_1, p_2, p_3, z_c, d_1, d_2)) ** (-1)
        )

    return weight


def generate_sobol_samples(n_sobol_samples, vis_vars, sobol_sequence):

    r_max = 1
    weights = []
    sobol_samples = []
    k_sample = 0
    n_need = 1
    while len(weights) < n_sobol_samples:
        if k_sample == n_need:
            n_need *= 2
        sobol_sample_try = (
            np.array([0, 0, vis_vars[1] / 2])
            + np.array([1, 1, r_max - vis_vars[1] / 2])
            * sobol_sequence.gen_samples(n=n_need)[k_sample]
        )
        r_3 = sobol_sample_try[2]
        r_2 = r_3 * sobol_sample_try[1]
        r_1 = r_2 * sobol_sample_try[0]
        p_3_sample = np.arcsin(np.random.rand())
        # print(r_3, r_2, r_1)
        weight = compute_weights([r_1, r_2, r_3, p_3_sample], vis_vars)
        if np.abs(weight) > 1e-4:
            weights.append(weight)
            sobol_samples.append(sobol_sample_try)
            # print(len(weights), n_sobol_samples)
        k_sample += 1

    return weights, sobol_samples


def param_dist(curr_param_estimates, r_samples):
    r_1, r_2, r_3 = r_samples
    mu, sigma, alpha_1, beta_1, alpha_2, beta_2 = curr_param_estimates
    term_1 = lognorm.pdf(r_3, sigma, scale=np.exp(mu))
    # term_1 = norm.pdf(r_3, loc=mu, scale=sigma)
    term_2 = beta.pdf(r_1 / r_2, alpha_1, beta_1)
    term_3 = beta.pdf(r_2 / r_3, alpha_2, beta_2)

    complete_pdf = term_1 * term_2 * term_3
    # print(complete_pdf, term_1, term_2, term_3)
    # print("r", r_3, r_1 / r_2, r_2 / r_3)
    # print("param", mu, sigma, alpha_1, beta_1, alpha_2, beta_2)
    return complete_pdf


def expct_log_likelihood(curr_param_estimates, weights, sobol_samples):
    def expected_val_z_u_m_c(curr_param_estimates):

        mu, sigma, a_1, b_1, a_2, b_2 = curr_param_estimates

        samples_z_u = []
        k_iter = 0
        while k_iter < 200:
            p_3_sample = np.arcsin(np.random.rand())
            p_2_sample = np.arcsin(
                np.cos(p_3_sample) * np.sin(np.pi * np.random.rand() / 2)
            )
            r_1_r_2_ratio_sample = beta.rvs(a_1, b_1)
            r_2_r_3_ratio_sample = beta.rvs(a_2, b_2)
            samples_z_u.append(
                np.sqrt(
                    np.sin(p_3_sample) ** 2
                    + r_2_r_3_ratio_sample ** 2 * np.sin(p_2_sample) ** 2
                    + r_2_r_3_ratio_sample ** 2
                    * r_1_r_2_ratio_sample ** 2
                    * (1 - np.sin(p_3_sample) ** 2 - np.sin(p_2_sample) ** 2)
                )
            )
            mean_z_u = np.mean(samples_z_u)
            if len(samples_z_u) > 1:
                std_z_u = 1 / np.sqrt(len(samples_z_u)) * np.std(samples_z_u)
                if std_z_u < 1e-3:
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
        r_3 = sobol_samples[i_part, s_sample][2]
        r_2 = r_3 * sobol_samples[i_part, s_sample][1]
        r_1 = r_2 * sobol_samples[i_part, s_sample][0]
        pdf[i_part, s_sample] = param_dist(curr_param_estimates, [r_1, r_2, r_3])

    log_expected_val_z_u = expected_val_z_u_m_c(curr_param_estimates)
    expct_log_likelihood_val = (
        np.sum(weights * np.log(pdf)) - n_particles * log_expected_val_z_u
    )

    # print(curr_param_estimates, expct_log_likelihood_val)
    return np.min([1e12, -1 * expct_log_likelihood_val])


def expct_log_likelihood_vec(curr_param_estimates_vec, weights, sobol_samples):

    # res = []
    # for i_param in curr_param_estimates_vec:
    #     res.append(expct_log_likelihood(i_param, weights, sobol_samples))
    # for i_param in curr_param_estimates_vec:
    pool = mp.Pool(mp.cpu_count())
    result_async = [
        pool.apply_async(expct_log_likelihood, args=(i_param, weights, sobol_samples))
        for i_param in curr_param_estimates_vec
    ]
    res = [r.get() for r in result_async]

    pool.close()
    return np.array(res)


def qmc_em_size_param_estimation(
    visible_vars, init_param_estimates, max_iter=50, tol=0.015, n_sobol_samples=16384
):

    n_particles = len(visible_vars)
    is_weights = np.full((n_particles, n_sobol_samples), 0.0)
    sobol_samples = np.full((n_particles, n_sobol_samples), 0, dtype=object)

    # Generate Sobol samples
    # --------------------------------------------------------------------------------------
    sobol_sequence = qp.Sobol(3)
    for i_part in range(n_particles):
        # Generate Sobol samples \(\pmb h_i^{(s)}\) until \(S\) samples with nonzero weights
        # are obtained
        print(i_part)
        is_weights[i_part, :], sobol_samples[i_part, :] = generate_sobol_samples(
            n_sobol_samples, visible_vars[i_part], sobol_sequence
        )

    curr_param_estimates = list(init_param_estimates)
    # EM algorithm
    # --------------------------------------------------------------------------------------
    bounds = ([-5, 0, 0, 0, 0, 0], [5, 1, 25, 10, 25, 10])
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    weights = np.array(is_weights)
    for (i_part, s_sample) in (
        (i_part, s_sample)
        for i_part in range(n_particles)
        for s_sample in range(n_sobol_samples)
    ):
        # Calculate the weights for the qmc integration
        r_3 = sobol_samples[i_part, s_sample][2]
        r_2 = r_3 * sobol_samples[i_part, s_sample][1]
        r_1 = r_2 * sobol_samples[i_part, s_sample][0]
        weights[i_part, s_sample] *= param_dist(
            [-3.5, np.sqrt(0.5), 15, 3, 20, 6], [r_1, r_2, r_3]
        )
    # Maximization step
    normalized_weights = (
        weights
        / np.array([np.sum(weights, axis=1) for _ in enumerate(weights[0, :])]).T
    )
    print(
        "max",
        [-3.5, np.sqrt(0.5), 15, 3, 20, 6],
        expct_log_likelihood(
            [-3.5, np.sqrt(0.5), 15, 3, 20, 6], normalized_weights, sobol_samples
        ),
    )
    optmizer = ps.single.GlobalBestPSO(
        n_particles=10, dimensions=6, options=options, bounds=bounds
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
            r_3 = sobol_samples[i_part, s_sample][2]
            r_2 = r_3 * sobol_samples[i_part, s_sample][1]
            r_1 = r_2 * sobol_samples[i_part, s_sample][0]
            weights[i_part, s_sample] *= param_dist(
                curr_param_estimates, [r_1, r_2, r_3]
            )
        # Maximization step
        old_param_estimates = list(curr_param_estimates)
        # curr_param_estimates += np.full(6, 0.25) - 0.5 * np.random.rand(6)

        _, curr_param_estimates = optmizer.optimize(
            expct_log_likelihood_vec,
            10,
            weights=normalized_weights,
            sobol_samples=sobol_samples,
        )
        # Stopping criterion
        # print(old_param_estimates)
        # print(curr_param_estimates)
        # print(
        #     np.max((old_param_estimates - curr_param_estimates) / curr_param_estimates)
        # )
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
    mu = -3.5
    sigma = np.sqrt(0.5)
    alpha_1 = 15
    beta_1 = 3
    alpha_2 = 20
    beta_2 = 6

    # Sample sizes
    # --------------------------------------------------------------------------------------
    n_sample_size = 200
    n_particles = 1000

    # Sample vals
    # --------------------------------------------------------------------------------------

    r_3_sample_vals = lognorm.rvs(sigma, scale=np.exp(mu), size=n_particles)
    r_2_r_3_sample_vals = beta.rvs(alpha_1, beta_1, size=n_particles)
    r_2_sample_vals = r_3_sample_vals * r_2_r_3_sample_vals
    r_1_r_2_sample_vals = beta.rvs(alpha_2, beta_2, size=n_particles)
    r_1_sample_vals = r_2_sample_vals * r_1_r_2_sample_vals
    p_3_sample_vals = np.arcsin(np.random.rand(n_particles))
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

    vis_vars = np.array([d_1_sample_vals, d_2_sample_vals]).T
    return vis_vars
