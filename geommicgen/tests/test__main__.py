"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call
import subprocess

import sys

import numpy as np
from app import run_program


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

    # @patch("geommicgen.microstructure.particle_classes.Disk")
    # @patch("geommicgen.microstructure.phase.FixedValue")
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
