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
from postproc.plotfuncs.plotting_functions import plot_particles_3d
import pickle
from micgenmethod.thermostats import MultiTemperatureIsokineticThermostat
from micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation


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
