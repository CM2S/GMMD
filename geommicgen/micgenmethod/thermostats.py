"""
Module containing the Thermostat abstract class and its subclasses.

This module contains the Thermostat abstract class, the MicroCanonicalEnsemble subclass,
that has no effect on the DM simulation, the IsokineticThermostat subclass, that enforces an
isokinetic temperature scheme, and the MultiTemperatureIsokineticThermostat subclass, that
enforces a self-calibrating multitemperature isokinetic scheme.
"""

import abc

import numpy as np


class Thermostat(abc.ABC):
    """This is the abstract class for thermostats."""

    def apply_thermostat(self, particle_velocities, kin_energy):
        """Apply the thermostat."""


class MicroCanonicalEnsemble(Thermostat):
    """This is the class for no thermostat, producing a micro canonical ensemble."""

    def __init__(self):
        """Initialize MicroCanonicalEnsemble class object. Do nothing."""
        self.reference_temp = None

    def apply_thermostat(self, particle_velocities, kin_energy):
        """Do nothing."""


class IsokineticThermostat(Thermostat):
    """
    This is the class for the isokinetic thermostat.

    It enforces a strictly constant temperature, diving the velocities of the particles by a
    constant found from the equipartion theorem.

    Attributes
    ----------
    reference_temp: float
        Reference temperature to be maintained by the thermostat.

    k_b: float
        Analog of the Boltzmann constant.
    """

    def __init__(self, reference_temp):
        """
        Initialize an IsokineticThermostat.

        Parameters
        ----------
        reference_temp: float
            Reference temperature to be maintained by the thermostat.
        """
        self.reference_temp = reference_temp
        # Intial temperature
        self.k_b = 1e-15
        # Analog to the Boltzmann constant

    def apply_thermostat(self, particle_velocities, kin_energy):
        """
        Apply the thermostat.

        It enforces a strictly constant temperature, diving the velocities of the particles
        by a constant found from the equipartion theorem.

        Parameters
        ----------
        particle_velocities: list(array)
            List of the particle velocities.

        kin_energy: float
            Kinetic energy of the system of particles.

        Returns
        -------
        particle_velocities: list(array)
            List of the particle velocities after applying the thermostat.
        """
        dim = len(particle_velocities[0])
        number_particles = len(particle_velocities)
        # The thermostate used is the isokinetic with constant temperature
        lambda_vel = np.sqrt(
            2 * dim * number_particles * self.k_b * self.reference_temp / kin_energy
        )
        for i_particle_index in range(number_particles):
            # Running through all the particles
            particle_velocities[i_particle_index] *= lambda_vel
            # Rescalling the velocities
        return particle_velocities


class MultiTemperatureIsokineticThermostat(IsokineticThermostat):
    """
    This is the class for the self-calibrating multi temperature isokinetic thermostat.

    It enforces a strictly constant temperature for a minimum number of steps, dividing the
    velocities of the particles by a constant found from the equipartion theorem.
    After keeping the system at a given temperature for some number of steps, it uses a
    heuristic approach to decide when the temperature is to be decreased again.
    It uses the increase in total overlap as a proxy for the system having reached
    equilibrium.

    Attributes
    ----------
    molecular_dynamics_sim: `.MolecularDynamicsSimulation`
        MD simulatin to which the thermostat is being applied.

    temp_change_steps: list(int)
        Steps at which the temperature was lowered.

    min_eq_steps_at_temp: int
        Minimum number of iterations spent at a the current temperature stage. After this
        number of iterations the temperature may be lowered.

    eq_steps_list: list
        List of the real number of equilibration steps used.

    _next_temp_change: int
        Iteration at which the temperature may be lowered.
    """

    def __init__(self, initial_temp, criterion, **kwargs):
        """
        Initialize an MultiTemperatureIsokineticThermostat.

        Parameters
        ----------
        initial_temp: float
            Initial temperature of the system.

        min_eq_steps_at_temp: int
            Minimum number of steps spent at a given temperature, after which the heuristic
            rule to decide if the temperature is to be decreased is applied.
        """
        self.eq_steps_list = []
        self.molecular_dynamics_sim = None
        self.temp_change_steps = [0]
        if criterion == "original":
            self.criterion = "original"
            self.min_eq_steps_at_temp = kwargs["min_eq_steps_at_temp"]
            self._next_temp_change = kwargs["min_eq_steps_at_temp"]
        elif criterion == "rolling_ave":
            self.criterion = "rolling_ave"
            self.average_window = kwargs["average_window"]
        super().__init__(initial_temp)

    def apply_thermostat(self, particle_velocities, kin_energy):
        """
        Apply the thermostat.

        It enforces a strictly constant temperature for a minimum number of steps, dividing
        the velocities of the particles by a constant found from the equipartion theorem.
        After keeping the system at a given temperature for some number of steps, the
        minimum number of equilibration steps, it uses a heuristic approach to decide when
        the temperature is to be decreased again.
        It uses the increase in total overlap as a proxy for the system having reached
        equilibrium.
        The minimum number of equilibration steps is updated to the real number of
        equilibration steps used, i.e. the number of steps between the two last temperature
        changes.

        Parameters
        ----------
        particle_velocities: list(array)
            List of the particle velocities.

        kin_energy: float
            Kinetic energy of the system of particles.

        Returns
        -------
        particle_velocities: list(array)
            List of the particle velocities after applying the thermostat.
        """
        # The thermostat used is the multi_temperature scheme
        if (
            self.molecular_dynamics_sim.total_overlap
            > self.molecular_dynamics_sim.max_residue
        ):
            # If a legal configuration has not been achieved
            if self.reached_quilibrium():
                # If the total overlap has increased in the previous iterations
                self.reference_temp *= 1 / 4
                # Lowering the temperature
                self.temp_change_steps.append(self.molecular_dynamics_sim.step)
                # Saving minimum equilibration times and times at which the
                # temperature has been lowered
        particle_velocities = super().apply_thermostat(particle_velocities, kin_energy)
        # Compute the rescaling factor only if the kinetic energy is nonzero

        return particle_velocities

    def reached_quilibrium(self) -> bool:
        equilibrium_flag = False
        if self.criterion == "original":
            if self.molecular_dynamics_sim.step > self._next_temp_change:
                # If the end of the equilibration time has been reached
                equilibrium_flag = any(
                    np.array(
                        self.molecular_dynamics_sim.total_overlap_history[
                            -self.min_eq_steps_at_temp // 2 :
                        ]
                    )
                    - np.array(
                        self.molecular_dynamics_sim.total_overlap_history[
                            -self.min_eq_steps_at_temp // 2 - 1 : -1
                        ]
                    )
                    > 0
                )
                if equilibrium_flag:
                    self.min_eq_steps_at_temp += (
                        self.molecular_dynamics_sim.step - self._next_temp_change - 1
                    )
                    # Updating the minimum equilibration time
                    self._next_temp_change = (
                        self.molecular_dynamics_sim.step + self.min_eq_steps_at_temp
                    )
                    # Updating the iteration of the last temperature change
                    self.eq_steps_list.append(self.min_eq_steps_at_temp)
        elif self.criterion == "rolling_ave":
            if (
                self.molecular_dynamics_sim.step - self.temp_change_steps[-1]
                > self.average_window
            ):
                step = self.molecular_dynamics_sim.step
                equilibrium_flag = (
                    self.molecular_dynamics_sim.total_overlap_history[
                        step - self.average_window
                    ]
                    - self.molecular_dynamics_sim.total_overlap_history[step]
                    < 0
                )

        return equilibrium_flag
