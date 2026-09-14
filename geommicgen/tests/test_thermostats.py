"""
Unit tests regarding thermostats.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
import time
from geommicgen.postproc.plotfuncs.plotting_functions import plot_particles_3d
import pickle
from geommicgen.micgenmethod.thermostats import MultiTemperatureIsokineticThermostat, IsokineticThermostat
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation




class TestIsokineticThermostat(unittest.TestCase):

    def test_negative_temperature_raises_error(self):
        reference_temp = -1
        with self.assertRaises(ValueError):
            IsokineticThermostat(reference_temp)

    def test_apply_thermostat(self):
        reference_temp = 1
        thermostat = IsokineticThermostat(reference_temp)
        particle_velocities = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        kin_energy = 0.001

        thermostat.apply_thermostat(particle_velocities, kin_energy)

        dim, n_particles = 2, 2
        resulting_kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)
        # For unit masses, this should equal ½·dim·N·k_b·reference_temp
        expected_kin_energy = 0.5 * dim * n_particles * thermostat.k_b * reference_temp
        self.assertAlmostEqual(resulting_kin_energy, expected_kin_energy)



class TestRatioInOut(unittest.TestCase):
    def test_reached_equilibrium_ratio_in_out_true(self):
        thermostat = MultiTemperatureIsokineticThermostat(
            1e5, "ratio_in_out", max_ratio_osc=2
        )
        overlap_list = []
        for step, overlap in [(0, 0.9), (1, 0.9), (2, 1.1), (3, 0.9)]:
            overlap_list.append(overlap)
            thermostat.molecular_dynamics_sim = Mock(
                step=step, particle_overlap_areas_dict={(1, 2): overlap_list}
            )
            equilibrium_flag = thermostat.reached_equilibrium()
        self.assertTrue(equilibrium_flag)

    def test_reached_equilibrium_ratio_in_out_false(self):
        thermostat = MultiTemperatureIsokineticThermostat(
            1e5, "ratio_in_out", max_ratio_osc=2
        )
        overlap_list = []
        for step, overlap in [(0, 0.9), (1, 0.89), (2, 0.9), (3, 1)]:
            overlap_list.append(overlap)
            thermostat.molecular_dynamics_sim = Mock(
                step=step, particle_overlap_areas_dict={(1, 2): overlap_list}
            )
            equilibrium_flag = thermostat.reached_equilibrium()
        self.assertTrue(not equilibrium_flag)
