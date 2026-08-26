"""Module for tests related to the random sequential adsorption (RSA) simulation.

This is the RSA counterpart of test_molecular_dynamics_sim.py: it holds behavior specific to
RandomSequentialAdsorption. Behavior shared with other generation methods (e.g. set_box) is
tested in test_generation_method.py instead.
"""

import unittest
from unittest.mock import Mock

import numpy as np

from geommicgen.micgenmethod.rand_seq_adsorption import RandomSequentialAdsorption


class TestRandomSequentialAdsorption(unittest.TestCase):
    """Test class for the RandomSequentialAdsorption class"""

    def setUp(self):
        self.rsa_sim = RandomSequentialAdsorption(500, 0, False, "")

    def test_set_speed_up_scheme(self):
        """Check that setting the speed up scheme links it back to the RSA simulation."""
        speed_up_scheme = Mock()

        self.rsa_sim.set_speed_up_scheme(speed_up_scheme)

        self.assertIs(self.rsa_sim.speed_up_scheme, speed_up_scheme)
        self.assertIs(speed_up_scheme.rsa_sim, self.rsa_sim)

    def test_check_intersection_no_particles_in_microstructure(self):
        """Check that a trial particle is always accepted when the box is still empty."""
        self.rsa_sim.microstructure_sample = Mock(particles=[])
        trial_particle = Mock()

        self.assertFalse(self.rsa_sim.check_intersection(trial_particle, [0, 1]))

    def test_check_intersection_no_overlap(self):
        """Check that the trial particle is accepted when it intersects none of the
        candidates in particles_list."""
        existing_particle_1 = Mock()
        existing_particle_2 = Mock()
        self.rsa_sim.microstructure_sample = Mock(
            particles=[existing_particle_1, existing_particle_2]
        )
        self.rsa_sim.box = [1.0, 1.0]
        trial_particle = Mock()
        trial_particle.intersection.return_value = False

        self.assertFalse(self.rsa_sim.check_intersection(trial_particle, [0, 1]))
        trial_particle.intersection.assert_any_call(existing_particle_1, [1.0, 1.0])
        trial_particle.intersection.assert_any_call(existing_particle_2, [1.0, 1.0])

    def test_check_intersection_overlap_found(self):
        """Check that the trial particle is rejected as soon as an intersection is found."""
        existing_particle_1 = Mock()
        existing_particle_2 = Mock()
        self.rsa_sim.microstructure_sample = Mock(
            particles=[existing_particle_1, existing_particle_2]
        )
        self.rsa_sim.box = [1.0, 1.0]
        trial_particle = Mock()
        trial_particle.intersection.side_effect = [False, True]

        self.assertTrue(self.rsa_sim.check_intersection(trial_particle, [0, 1]))

    def test_check_intersection_only_checks_candidates_in_particles_list(self):
        """Check that only the candidates listed in particles_list are checked."""
        existing_particle_1 = Mock()
        existing_particle_2 = Mock()
        self.rsa_sim.microstructure_sample = Mock(
            particles=[existing_particle_1, existing_particle_2]
        )
        self.rsa_sim.box = [1.0, 1.0]
        trial_particle = Mock()
        trial_particle.intersection.return_value = False

        self.rsa_sim.check_intersection(trial_particle, [1])

        trial_particle.intersection.assert_called_once_with(existing_particle_2, [1.0, 1.0])
