"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import os
import tempfile
import unittest
from unittest.mock import sentinel, Mock, patch, call
import subprocess

import sys

import numpy as np
import yaml

from geommicgen.app import run_program


@patch("sys.argv")
class TestMainFromCommandLine(unittest.TestCase):
    """Class for the unit tests regarding the __main__ of the geommicgen package."""

    def test_no_arguments(self, mock_sys_argv):
        """Test if no argumemts raises the correct exception."""

        mock_sys_argv.__len__.return_value = 1
        with self.assertRaises(ValueError):

            run_program()

    def test_too_many_arguments(self, mock_sys_argv):
        """Test if too many argumemts raises the correct exception."""

        mock_sys_argv.__len__.return_value = 4
        with self.assertRaises(ValueError):

            run_program()

    # @patch("microstructure.particleclasses.Disk")
    # @patch("microstructure.phase.FixedValue")
    # def test_generate_particles_number(self, mock_fixed_value, mock_disk):
    #     gen_method = MicGenTest()
    #     rve_dims = [1.0, 1.0]
    #     descriptors = {
    #         "n": mock_fixed_value("n", 10),
    #         "vf": mock_fixed_value("vf", 0.1),
    #     }
    #     phase = sentinel.phase
    #     particles = gen_method.generate_particles(
    #         rve_dims, mock_disk, phase, **descriptors
    #     )
    #     self.assertTrue(len(particles), 10)
    #     for particle in particles:
    #         self.assertTrue(particle.name, "Disk()")

SEEDED_DECK = """
Problem_Type 1
N_DP_Samples 2
RVE_Dimensions [1, 1]
Fixed_Seed 3
Mic_Gen_Descriptors
Phase 0
Phase_Type 1
Phase 1
Phase_Type 2
vf 0.2
n 4
Max_Residue_Per_Particle 0
Max_Step 3
Speed_Up_Scheme Naive
Save_History False
"""


class TestFixedSeedAcrossSamples(unittest.TestCase):
    """Test class for a seeded run of several samples."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def run_deck(self, name):
        """Run the seeded deck under a name of its own, returning its microstructures."""
        deck_path = os.path.join(self.temp_dir.name, name + ".mdsim")
        with open(deck_path, "w") as deck:
            deck.write(SEEDED_DECK)
        with patch("sys.argv", ["geommicgen", deck_path]):
            run_program()
        results_dir = os.path.join(self.temp_dir.name, name)
        documents = []
        for i_sample in range(2):
            with open(os.path.join(results_dir, "mic_{0}".format(i_sample), "mic.yaml")) as f:
                documents.append(yaml.safe_load(f))

        return documents

    def test_each_sample_has_a_seed_of_its_own(self):
        first, second = self.run_deck("a")
        self.assertEqual(first["provenance"]["fixed_seed"], 3)
        self.assertEqual(second["provenance"]["fixed_seed"], 4)
        self.assertNotEqual(first["particles"], second["particles"])
        # One seed for every sample made every sample the same microstructure

    def test_the_samples_are_the_same_in_another_run(self):
        first_run = self.run_deck("a")
        second_run = self.run_deck("b")
        for i_first, i_second in zip(first_run, second_run):
            self.assertEqual(i_first["particles"], i_second["particles"])
