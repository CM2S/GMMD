"""Module for tests related to the random sequential adsorption (RSA) simulation.

This is the RSA counterpart of test_molecular_dynamics_sim.py: it holds behavior specific to
RandomSequentialAdsorption. Behavior shared with other generation methods (e.g. set_box) is
tested in test_generation_method.py instead.
"""

import itertools
import os
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from geommicgen.config.run import read_config
from geommicgen.micgenmethod.rand_seq_adsorption import RandomSequentialAdsorption
from geommicgen.microstructure.microstructure import Microstructure


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


class TestRandomSequentialAdsorptionRun(unittest.TestCase):
    """Test class for whole generations by a random sequential adsorption."""

    def generate(self, input_file):
        """Generate the microstructure an input file describes, seeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "input.yaml")
            with open(path, "w") as file:
                file.write(input_file)
            options = read_config(path).to_options()
        microstructure = Microstructure.from_descriptors(
            options["mic_gen_parameters"]["rve_dimensions"],
            options["mic_gen_descriptors"],
        )
        simulation = RandomSequentialAdsorption.from_options(
            {**options["mic_gen_parameters"], "fixed_seed": 3}
        )
        simulation.generate_microstructure(microstructure)
        self.assertTrue(simulation.status)

        return microstructure

    def test_the_minimum_distance_holds_between_phases(self):
        """Check that disks of two phases are kept the minimum distance apart."""
        microstructure = self.generate(
            "samples: 1\n"
            "rve_dimensions: [1, 1]\n"
            "phases:\n"
            "  0: {type: matrix}\n"
            "  1: {type: disk, r: 0.05, vf: 0.15}\n"
            "  2: {type: disk, r: 0.05, vf: 0.15}\n"
            "generation:\n"
            "  min_distance: 0.02\n"
            "  random_sequential_adsorption: {max_step: 100000}\n"
        )
        box = np.array(microstructure.rve_dims)
        gaps = []
        for i_particle, j_particle in itertools.combinations(microstructure.particles, 2):
            if i_particle.phase == j_particle.phase:
                continue
            centres = i_particle.position_center - j_particle.position_center
            centres = centres - box * np.round(centres / box)
            gaps.append(
                np.linalg.norm(centres) - i_particle.radius - j_particle.radius
            )

        self.assertGreaterEqual(min(gaps), 0.02 - 1e-9)
