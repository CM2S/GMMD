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
from geommicgen.micgenmethod.thermostats import (
    BerendsenForceThermostat,
    IsokineticThermostat,
    LOWERING_TEMP_CRITERIA,
    MicroCanonicalEnsemble,
    MultiTemperatureIsokineticThermostat,
    THERMOSTATS,
    thermostat_from_options,
)
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation


class TestThermostatFromOptions(unittest.TestCase):
    """Test class for the thermostat built from the generation parameters of a deck."""

    def test_each_thermostat_is_built_under_its_name(self):
        expected = {
            "micro_canonical": MicroCanonicalEnsemble,
            "isokinetic": IsokineticThermostat,
            "berendsen": BerendsenForceThermostat,
            "multi_temperature": MultiTemperatureIsokineticThermostat,
        }
        self.assertEqual(set(expected), set(THERMOSTATS))
        options = {
            "initial_temp": 1e5,
            "berendsen_coeff": 1e-2,
            "max_ratio_osc": 2,
            "temp_low_ratio": 1 / 4,
        }
        for i_name, i_class in expected.items():
            with self.subTest(thermostat=i_name):
                thermostat = thermostat_from_options(dict(options, thermostat=i_name))
                self.assertIsInstance(thermostat, i_class)

    def test_the_multi_temperature_thermostat_is_the_default(self):
        thermostat = thermostat_from_options(
            {"initial_temp": 1e5, "max_ratio_osc": 2, "temp_low_ratio": 1 / 4}
        )
        self.assertIsInstance(thermostat, MultiTemperatureIsokineticThermostat)
        self.assertEqual(thermostat.criterion, "ratio_in_out")
        # The defaults of the input data file, so that a deck naming neither gets
        # what a deck that reads the documented defaults gets

    def test_each_criterion_takes_its_own_parameter(self):
        options = {"initial_temp": 1e5, "temp_low_ratio": 1 / 4}
        parameters = {
            "ratio_in_out": ("max_ratio_osc", 2),
            "rolling_ave": ("average_window", 25),
            "original": ("min_eq_steps_at_temp", 25),
        }
        self.assertEqual(set(parameters), set(LOWERING_TEMP_CRITERIA))
        for i_criterion, (i_name, i_value) in parameters.items():
            with self.subTest(criterion=i_criterion):
                thermostat = thermostat_from_options(
                    dict(options, lowering_temp_criterion=i_criterion, **{i_name: i_value})
                )
                self.assertEqual(thermostat.criterion, i_criterion)
                self.assertEqual(getattr(thermostat, i_name), i_value)
                with self.assertRaises(ValueError) as context:
                    thermostat_from_options(
                        dict(options, lowering_temp_criterion=i_criterion)
                    )
                self.assertIn(i_name, str(context.exception))

    def test_an_unknown_thermostat_is_refused(self):
        with self.assertRaises(ValueError) as context:
            thermostat_from_options({"thermostat": "isokinetc"})
        self.assertIn("isokinetc", str(context.exception))
        self.assertIn("multi_temperature", str(context.exception))
        # It used to be given the micro canonical ensemble without a word

    def test_an_unknown_criterion_is_refused(self):
        with self.assertRaises(ValueError) as context:
            thermostat_from_options(
                {"initial_temp": 1e5, "lowering_temp_criterion": "rolling"}
            )
        self.assertIn("rolling", str(context.exception))
        self.assertIn("rolling_ave", str(context.exception))

    def test_a_missing_temperature_names_what_needs_it(self):
        with self.assertRaises(ValueError) as context:
            thermostat_from_options({"thermostat": "isokinetic"})
        self.assertIn("initial_temp", str(context.exception))
        self.assertIn("isokinetic", str(context.exception))


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
