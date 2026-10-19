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
from geommicgen.micgenmethod.thermostats import IsokineticThermostat, BerendsenForceThermostat, MultiTemperatureIsokineticThermostat
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation




class TestIsokineticThermostat(unittest.TestCase):

    def test_negative_temperature_raises_error(self):
        reference_temp = -1
        with self.assertRaises(ValueError):
            IsokineticThermostat(reference_temp)

    def test_apply_thermostat(self):
        reference_temp = 200
        thermostat = IsokineticThermostat(reference_temp)
        particle_velocities = [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([2.0, 2.0])]
        kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities) # Considering unit masses

        thermostat.apply_thermostat(particle_velocities, kin_energy)

        dim, n_particles = 2, 3
        # The isokinetic thermostat rescales all velocities by a single factor so that the
        # system's kinetic energy matches the value set by the reference temperature via the
        # equipartition theorem: KE = 1/2 * dim * n_particles * k_b * reference_temp.
        resulting_kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities) # Considering unit masses
        expected_kin_energy = 0.5 * dim * n_particles * thermostat.k_b * reference_temp
        self.assertAlmostEqual(resulting_kin_energy, expected_kin_energy)



class TestBerendsenForceThermostat(unittest.TestCase):

    def test_negative_temperature_raises_error(self):
        reference_temp = -1
        with self.assertRaises(ValueError):
            BerendsenForceThermostat(reference_temp,1)

    def test_apply_thermostat(self):
        reference_temp = 200
        berendsen_coeff = 1
        thermostat = BerendsenForceThermostat(reference_temp,berendsen_coeff)
        particle_velocities = [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([2.0, 2.0])]
        kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities) # Considering unit masses

        thermostat.apply_thermostat(particle_velocities, kin_energy)

        dim, n_particles = 2, 3
        expected_force_coeff = berendsen_coeff * (
            kin_energy - 0.5 * dim * n_particles * thermostat.k_b * reference_temp
        )

        self.assertAlmostEqual(expected_force_coeff, thermostat.force_coeff)





class TestMultiTemperatureIsokineticThermostat(unittest.TestCase):

    def test_apply_thermostat(self):
        initial_temp = 200
        dim, n_particles = 2, 3

        def make_thermostat():
            return MultiTemperatureIsokineticThermostat(
                initial_temp, "original", min_eq_steps_at_temp=2,
            )

        def make_velocities():
            return [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([2.0, 2.0])]

        def assert_rescaled_to_reference_temp(thermostat, particle_velocities):
            # Whatever reference_temp ended up as, apply_thermostat must always finish
            # by calling the inherited isokinetic rescaling for that value (equipartition
            # theorem: KE = 1/2 * dim * n_particles * k_b * reference_temp).
            resulting_kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)
            expected_kin_energy = (
                0.5 * dim * n_particles * thermostat.k_b * thermostat.reference_temp
            )
            self.assertAlmostEqual(resulting_kin_energy, expected_kin_energy)

        with self.subTest("legal configuration: reached_equilibrium is never consulted"):
            thermostat = make_thermostat()
            thermostat.molecular_dynamics_sim = Mock(total_overlap=0, max_residue=1)
            thermostat.reached_equilibrium = Mock()
            particle_velocities = make_velocities()
            kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)

            thermostat.apply_thermostat(particle_velocities, kin_energy)

            thermostat.reached_equilibrium.assert_not_called()
            self.assertEqual(thermostat.reference_temp, initial_temp)
            self.assertEqual(thermostat.temp_change_steps, [0])
            assert_rescaled_to_reference_temp(thermostat, particle_velocities)

        with self.subTest("illegal configuration, equilibrium not reached: no temperature change"):
            thermostat = make_thermostat()
            thermostat.molecular_dynamics_sim = Mock(total_overlap=2, max_residue=1, step=5)
            thermostat.reached_equilibrium = Mock(return_value=False)
            particle_velocities = make_velocities()
            kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)

            thermostat.apply_thermostat(particle_velocities, kin_energy)

            self.assertEqual(thermostat.reference_temp, initial_temp)
            self.assertEqual(thermostat.temp_change_steps, [0])
            assert_rescaled_to_reference_temp(thermostat, particle_velocities)

        with self.subTest("equilibrium reached, kinetic energy diverges: flag set, temperature untouched"):
            thermostat = make_thermostat()
            thermostat.molecular_dynamics_sim = Mock(
                total_overlap=2, max_residue=1, step=5,
                kinetic_energy=10.0, thermic_energy_history=[1.0],
            )
            thermostat.reached_equilibrium = Mock(return_value=True)
            particle_velocities = make_velocities()
            kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)

            thermostat.apply_thermostat(particle_velocities, kin_energy)

            self.assertTrue(thermostat.kin_energy_div)
            self.assertEqual(thermostat.reference_temp, initial_temp)
            self.assertEqual(thermostat.temp_change_steps, [0, 5])
            assert_rescaled_to_reference_temp(thermostat, particle_velocities)

        with self.subTest("equilibrium reached, kinetic energy stable: temperature is lowered"):
            thermostat = make_thermostat()
            thermostat.molecular_dynamics_sim = Mock(
                total_overlap=2, max_residue=1, step=5,
                kinetic_energy=1.5, thermic_energy_history=[1.0],
            )
            thermostat.reached_equilibrium = Mock(return_value=True)
            particle_velocities = make_velocities()
            kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)

            thermostat.apply_thermostat(particle_velocities, kin_energy)

            self.assertFalse(thermostat.kin_energy_div)
            self.assertAlmostEqual(thermostat.reference_temp, initial_temp * thermostat.temp_low_ratio)
            self.assertEqual(thermostat.temp_change_steps, [0, 5])
            assert_rescaled_to_reference_temp(thermostat, particle_velocities)

        with self.subTest("equilibrium reached again once already diverging: temperature lowered unconditionally"):
            thermostat = make_thermostat()
            thermostat.molecular_dynamics_sim = Mock(total_overlap=2, max_residue=1, step=5)
            thermostat.reached_equilibrium = Mock(return_value=True)
            thermostat.kin_energy_div = True
            particle_velocities = make_velocities()
            kin_energy = 0.5 * sum(np.sum(v**2) for v in particle_velocities)

            thermostat.apply_thermostat(particle_velocities, kin_energy)

            self.assertTrue(thermostat.kin_energy_div)
            self.assertAlmostEqual(thermostat.reference_temp, initial_temp * thermostat.temp_low_ratio)
            self.assertEqual(thermostat.temp_change_steps, [0, 5])
            assert_rescaled_to_reference_temp(thermostat, particle_velocities)

    def test_reached_equilibrium_original(self):
        thermostat = MultiTemperatureIsokineticThermostat(
            1e5, "original", min_eq_steps_at_temp=2, 
        )
        
        overlap_list = []
        for step, overlap in [(0, 1), (1, 0.9), (2, 0.9), (3, 0.9), (4,1.1)]:
            overlap_list.append(overlap)
            thermostat.molecular_dynamics_sim = Mock(
                step=step, total_overlap_history= overlap_list
            )
            equilibrium_flag = thermostat.reached_equilibrium()
        self.assertTrue(equilibrium_flag)
        # Check the minumum equlibrium steps at a temperature is updated correctly
        self.assertEqual(thermostat.min_eq_steps_at_temp, 3)
        # Check if the next temperature change step is updated correctly
        self.assertEqual(thermostat._next_temp_change,7)



    def test_reached_equilibrium_rolling_ave(self):
        thermostat = MultiTemperatureIsokineticThermostat(
            1e5, "rolling_ave", average_window=2, 
        )
        
        overlap_list = []
        for step, overlap in [(0, 0.9), (1, 0.9), (2, 0.9), (3, 1.1)]:
            overlap_list.append(overlap)
            thermostat.molecular_dynamics_sim = Mock(
                step=step, total_overlap_history= overlap_list
            )
            equilibrium_flag = thermostat.reached_equilibrium()
        self.assertTrue(equilibrium_flag)


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
