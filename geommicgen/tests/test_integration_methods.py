""" Unit tests regarding the micgenmethod.integration_methods module """

import unittest

import numpy as np

from geommicgen.micgenmethod.integration_methods import (
    newmark_integration,
    verlet_sync_integration,
)


class NewmarkIntegrationTest(unittest.TestCase):
    " Tests for newmark_integration "

    def setUp(self):
        self.dim = 2
        self.mass = np.eye(self.dim) * 2.0
        self.damping = np.zeros((self.dim, self.dim))
        self.stiffness = np.zeros((self.dim, self.dim))
        self.x_0 = np.array([1.0, 2.0])
        self.x_dot_0 = np.array([0.0, 0.0])
        self.delta_t = 0.1
        self.n_steps = 3

    def test_output_shapes(self):
        " The returned position, velocity and acceleration arrays must have shape (dim, n_steps) "
        f_vec = np.zeros((self.dim, self.n_steps + 1))
        x_vec, x_dot_vec, x_ddot_vec = newmark_integration(
            self.x_0, self.x_dot_0, f_vec, self.mass, self.damping, self.stiffness,
            self.delta_t, self.n_steps, self.dim,
        )
        for result in (x_vec, x_dot_vec, x_ddot_vec):
            self.assertEqual(result.shape, (self.dim, self.n_steps))

    def test_particle_at_rest_stays_at_rest(self):
        " With zero force, zero initial velocity and zero stiffness/damping, the particle does not move "
        f_vec = np.zeros((self.dim, self.n_steps + 1))
        x_vec, x_dot_vec, x_ddot_vec = newmark_integration(
            self.x_0, self.x_dot_0, f_vec, self.mass, self.damping, self.stiffness,
            self.delta_t, self.n_steps, self.dim,
        )
        np.testing.assert_allclose(x_vec, np.tile(self.x_0.reshape(-1, 1), (1, self.n_steps)))
        np.testing.assert_allclose(x_dot_vec, np.zeros((self.dim, self.n_steps)))
        np.testing.assert_allclose(x_ddot_vec, np.zeros((self.dim, self.n_steps)))

    def test_matches_analytical_solution_for_constant_force(self):
        """
        With zero stiffness/damping and a constant force, motion is at constant acceleration
        and must match the closed-form kinematics (Newmark's average-acceleration variant is
        exact for constant acceleration).
        """
        f_const = np.array([4.0, 6.0])
        f_vec = np.tile(f_const.reshape(-1, 1), (1, self.n_steps + 1))
        acceleration = np.linalg.solve(self.mass, f_const)

        x_vec, x_dot_vec, x_ddot_vec = newmark_integration(
            self.x_0, self.x_dot_0, f_vec, self.mass, self.damping, self.stiffness,
            self.delta_t, self.n_steps, self.dim,
        )

        time = np.arange(1, self.n_steps + 1) * self.delta_t
        x_expected = (
            self.x_0.reshape(-1, 1)
            + np.outer(self.x_dot_0, time)
            + 0.5 * np.outer(acceleration, time ** 2)
        )
        x_dot_expected = self.x_dot_0.reshape(-1, 1) + np.outer(acceleration, time)
        x_ddot_expected = np.tile(acceleration.reshape(-1, 1), (1, self.n_steps))

        np.testing.assert_allclose(x_vec, x_expected)
        np.testing.assert_allclose(x_dot_vec, x_dot_expected)
        np.testing.assert_allclose(x_ddot_vec, x_ddot_expected)


class VerletSyncIntegrationTest(unittest.TestCase):
    " Tests for verlet_sync_integration "

    def setUp(self):
        self.dim = 2
        self.m_part = 1.5
        self.x_0 = np.array([1.0, 2.0])
        self.delta_t = 0.1
        self.n_steps = 3

    def test_output_shapes(self):
        " The returned position and velocity arrays must have shape (dim, n_steps) "
        x_dot_0 = np.array([0.5, -0.3])
        f_vec = np.zeros((self.dim, self.n_steps))
        x_vec, x_dot_vec = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim
        )
        self.assertEqual(x_vec.shape, (self.dim, self.n_steps))
        self.assertEqual(x_dot_vec.shape, (self.dim, self.n_steps))

    def test_zero_force_gives_constant_velocity_motion(self):
        " With zero force, the particle must move in a straight line at its initial velocity "
        x_dot_0 = np.array([1.0, -1.0])
        f_vec = np.zeros((self.dim, self.n_steps))
        x_vec, x_dot_vec = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim
        )
        time = np.arange(1, self.n_steps + 1) * self.delta_t
        x_expected = self.x_0.reshape(-1, 1) + np.outer(x_dot_0, time)
        np.testing.assert_allclose(x_vec, x_expected)
        np.testing.assert_allclose(x_dot_vec, np.tile(x_dot_0.reshape(-1, 1), (1, self.n_steps)))

    def test_matches_analytical_position_for_constant_force(self):
        """
        With a constant force, the position must follow the closed-form constant-acceleration
        kinematics.
        """
        x_dot_0 = np.array([0.5, 0.0])
        f_const = np.array([3.0, -2.0])
        f_vec = np.tile(f_const.reshape(-1, 1), (1, self.n_steps))
        acceleration = f_const / self.m_part

        x_vec, _ = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim
        )

        time = np.arange(1, self.n_steps + 1) * self.delta_t
        x_expected = (
            self.x_0.reshape(-1, 1)
            + np.outer(x_dot_0, time)
            + 0.5 * np.outer(acceleration, time ** 2)
        )
        np.testing.assert_allclose(x_vec, x_expected)

    def test_default_dt_old_equals_delta_t(self):
        " Omitting dt_old must give the same result as explicitly passing dt_old=delta_t "
        x_dot_0 = np.array([0.5, -0.3])
        f_vec = np.random.RandomState(0).rand(self.dim, self.n_steps)

        x_default, v_default = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim
        )
        x_explicit, v_explicit = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim,
            dt_old=self.delta_t,
        )

        np.testing.assert_allclose(x_default, x_explicit)
        np.testing.assert_allclose(v_default, v_explicit)

    def test_dt_old_changes_the_result(self):
        " A dt_old different from delta_t must change the computed trajectory "
        x_dot_0 = np.array([0.5, -0.3])
        f_vec = np.random.RandomState(0).rand(self.dim, self.n_steps)

        x_vec, _ = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim
        )
        x_vec_other_dt_old, _ = verlet_sync_integration(
            self.x_0, x_dot_0, f_vec, self.m_part, self.delta_t, self.n_steps, self.dim,
            dt_old=2 * self.delta_t,
        )

        self.assertFalse(np.allclose(x_vec, x_vec_other_dt_old))
