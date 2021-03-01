"""
Module containing the MolecularDynamicsSimulation class.

It provides a class whose methods allow the performance of a molecular dynamics simulation
where the forces between the particles are repulsive and propertional to the overlap
area/volume. A given configuration of particles is considered legal when the total overlap
area/volume is smaller than specified, It is accepted when the system remains in a legal
configuration for a specified number of steps as low as one. It requires subclasses of the
Thermostat and SpeedUpScheme abstract classes to work.
"""

from contextlib import contextmanager
import time
import numpy as np
from scipy.stats import hmean

# pylint: disable=import-error
import errors.error_classes as errors
import iofuncs.printing as print_funcs
from micgenmethod.microstructure_gen_method import GenerationMethod
from micgenmethod.integration_methods import VerletSync, Newmark
from microstructure.particle_classes import Matrix, Particle


class MolecularDynamicsSimulation(GenerationMethod):
    """Class for the molecular dynamics simulation class.

    It stores all the options specifying how the simlulation will be run, it contains the
    methods needed to run the simulation and it also stores the relevant details of the
    simulation.

    Attributes
    ----------
    box: list
        List of the dimensions of the simulation box. Almost always equal to the the
        dimensions of the micrrostructure, except for CylindricalFibers.

    particle_velocities: list(array)
        List containing the velocities of the particles in the simulation box.

    particle_forces: list(array)
        List containing the forces applied on the particles in the simulation box.

    particle_overlap_areas: list(float)
        List containing the overlap for each particle in the simulation box.

    particle_total_overlap: float
        Total overlap for all the particles in the simulation box.

    particle_total_overlap_history: list(float)
        History of the total overlap.

    position_center_history: list(list(array))
        It is a list containing the list of the positions of all particles in the simulation
        for each time step.

    thermostat: `.Thermostat`
        Thermostat to be used

    speed_up_scheme: `.SpeedUpScheme`
        Speed up scheme for force computation to be used.

    max_residue_per_particle: float
        Maximum allowable overlap residue between per particle

    max_residue: float
        Maximum allowable total overlap

    max_step: int
        Maxium number of time steps

    max_steps_to_relax: int
        Number of time steps a configuration has to remain legal to be accepted.

    dt: float
        Time step for the intergration of the equations of motion.

    min_distance: float
        Minimum distance between particles.

    type_init_conf: {'random', 'grid'}
        Type of initial configuration used.

    save_history: bool
        Save all the trajectories of the particles, the history of the relative and
        kinetic energy.

    step: int
        Current iteration of the MD simulation.

    time: float
        Time taken by the simulation.

    relative_energy_history: list(float)
        List of the relative energy history.

    kinetic_energy_history: list(float)
        List of the kinetic energy history.
    """

    def __init__(
        self,
        max_residue_per_particle,
        max_step,
        max_steps_to_relax,
        dt,
        min_distance,
        type_init_conf,
        save_history,
        **kwargs
    ):
        """
        Initialize the MolecularDynamicsSimulation class.

        Setting the options governing how the molecular dynamics simulation will be run.

        Parameters
        ----------
        max_residue_per_particle: float
            Maximum allowable overlap residue between particles

        max_step: int
            Maxium number of time steps

        max_steps_to_relax: int
            Number of time steps a configuration has to remain legal to be accepted.

        dt: float
            Time step for the intergration of the equations of motion.

        min_distance: float
            Minimum distance between particles.

        type_init_conf: {'random', 'grid'}
            Type of initial configuration used.

        save_history: bool
            Save all the trajectories of the particles, the history of the relative and
            kinetic energy.

        Keyword Parameters
        ------------------
        damping_coeff: float
            Viscous damping coefficient.

        particle_mass_opt: {'volume', 'radius', 'unit'}
            Consider the mass equal to its volume, its radius or equal to one.

        force_option: {"intersection_area", "intersection_length"}
            Force proportional to the overlap area/volume or the overlap length
        """
        self.box = None
        self.particle_velocities = None
        self.particle_forces = None
        self.particle_overlap_areas = None
        self.particle_overlap_areas_dict = {}
        self.total_overlap = None
        self.total_overlap_history = []
        self.position_center_history = None
        self.relative_energy_history = []
        self.kinetic_energy_history = []
        self.speed_up_scheme = None
        self.thermostat = None
        self.min_distance = min_distance
        self.type_init_conf = type_init_conf
        self.max_residue_per_particle = max_residue_per_particle
        self.max_residue = None
        self.max_step = max_step
        self.max_steps_to_relax = max_steps_to_relax
        self.dt = dt
        self.save_history = save_history
        self.time = None
        self.step = 0
        # Initializing the the time step at 0
        self.max_force = None
        self.damping_coeff = kwargs.get("damping_coeff", 0)
        self.particle_mass_opt = kwargs.get("particle_mass_opt", "volume")
        self.force_option = kwargs.get("force_option", "intersection_area")
        self.force_rescale = kwargs.get("force_rescale", False)
        self.dt_adapt = kwargs.get("dt_adapt", "const")
        self.offset = kwargs.get("offset", True)
        self.fixed_seed = kwargs.get("fixed_seed", None)
        self.force_rescale_coeff = 1
        self.coord_number = None
        self.thermic_enegy_history = []
        self.dist_met = "dist_approx"
        self.all_dt = []

    def generate_microstructure(self, microstructure_sample):
        """
        Generate the microstructure for the sample supplied.

        Generate the microstructure for microstructure_sample using the microstructure
        generation method *self*.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure sample to be generated
        """
        self.microstructure_sample = microstructure_sample
        for phase in microstructure_sample.phases.values():
            if phase.type is not Matrix and not phase.inner_phase:
                phase.generate_particles(microstructure_sample.rve_dims)
        if microstructure_sample.volume_fraction > 1:
            raise ValueError(
                "The volume fraction goes over 1: {0}".format(
                    microstructure_sample.volume_fraction
                )
            )
        # Generating the particles
        self.set_box(microstructure_sample.particles, microstructure_sample.rve_dims)
        self.generate_initial_configuration(microstructure_sample.particles)
        print_funcs.print_microstructure_info(microstructure_sample)
        try:
            start = time.time()
            self.run_molecular_dynamics_simulation(microstructure_sample.particles)
            self.time = time.time() - start
            microstructure_sample.total_overlap = self.total_overlap
            # Placing inner phases
            # ------------------------------------------------------------------------------
            for phase in microstructure_sample.phases.values():
                if phase.inner_phase:
                    outer_phase = microstructure_sample.phases[str(phase.outer_phase)]
                    phase.generate_particles(microstructure_sample.rve_dims)
                    if phase.volume_fraction > outer_phase.volume_fraction:
                        raise ValueError(
                            "The volume fraction goes over {0}: {1}".format(
                                outer_phase.volume_fraction, phase.volume_fraction
                            )
                        )
                    print_funcs.print_microstructure_info(microstructure_sample)
                    self.place_inner_phase_rsa(phase, outer_phase)
        finally:
            print_funcs.print_final_message(
                self.time,
                microstructure_sample.total_overlap,
                len(self.total_overlap_history),
                self.max_residue,
            )

    def set_box(self, particles, rve_dims):
        """
        Set the dimensions of the simulation box.

        Set the dimensions of the box according to the particles present and the
        dimensions of the microstructure.

        It is assumed that there are no incompatible particles.

        Parameters
        ----------
        particles: list(`.Particle`)
            List of particles in the simulation.

        rve_dims: list(floats)
            Dimensions of the microstructure in each spatial direction.
        """
        if any(
            [
                particle.__class__.__name__ == "CylindricalFiber"
                for particle in particles
            ]
        ):
            self.box = list(rve_dims)
            del self.box[particles[0].direction_fibers]
        else:
            self.box = list(rve_dims)

    def set_thermostat(self, thermostat):
        """Set the thermostat for the moleuclar dynamics simulation."""
        self.thermostat = thermostat
        thermostat.molecular_dynamics_sim = self

    def set_speed_up_scheme(self, speed_up_scheme):
        """Set speed up scheme for the molecular dynamics simulation."""
        self.speed_up_scheme = speed_up_scheme
        speed_up_scheme.molecular_dynamics_sim = self

    def generate_initial_configuration(self, particles):
        """
        Generate the initial configuration for particles.

        It sets the position of the center of all the particles in the simulation box, as
        well the velocity of the particles. The initial configuration is stored.

        Parameters
        ----------
        particles: list(particles)
            Particles in the simulation box.
        """
        self.position_center_history = [[None] for _ in particles]
        self.particle_velocities = [None for _ in particles]
        if self.type_init_conf == "random":
            # Random configuration for the particle centers and the zero velocity
            if self.fixed_seed is not None:
                np.random.seed(self.fixed_seed)
                # Generating the same initial configuration in different runs.
            for i_ind, i_particle in enumerate(particles):
                # Running through all the particles
                i_particle.position_center = self.box * np.random.uniform(
                    size=i_particle.dim
                )
                self.particle_velocities[i_ind] = np.zeros(i_particle.dim)
                # Generating the positions from a random uniform distribution
                self.position_center_history[i_ind][0] = i_particle.position_center
                # Saving initial configuration
        elif self.type_init_conf == "grid":
            # Particles randomly assigned to a place in a grid constructed to have an equal
            # number of cells in each direction and a total number of cells larger than the
            # number of particles
            if particles[0].dim == 3:
                # n_cells_side = np.int(np.ceil(np.cbrt(len(particles))))
                n_cells_side = 6
                # Number of cells in each direction
                cell_length = np.array(self.box) / n_cells_side
                # Length of the cells in each direction
                k_counter = 0
                # Initializing the counter
                grid_places = np.arange(n_cells_side ** 3)
                # Label of each grid place
                np.random.shuffle(grid_places)
                # Distributing the particles randomly to different cells of the grid
                for x_cell in range(n_cells_side):
                    for y_cell in range(n_cells_side):
                        for z_cell in range(n_cells_side):
                            if grid_places[k_counter] < len(particles):
                                particles[
                                    grid_places[k_counter]
                                ].position_center = np.array(
                                    [
                                        x_cell * cell_length[0] + cell_length[0] / 2,
                                        y_cell * cell_length[1] + cell_length[1] / 2,
                                        z_cell * cell_length[2] + cell_length[2] / 2,
                                    ]
                                )

                                # Generating the positions from a random uniform
                                # distribution between
                                self.particle_velocities[
                                    grid_places[k_counter]
                                ] = np.random.uniform(low=-0.1, high=0.1, size=3)
                                self.position_center_history[grid_places[k_counter]][
                                    0
                                ] = particles[grid_places[k_counter]].position_center
                                # Saving particle history
                            k_counter += 1
            elif particles[0].dim == 2:
                n_cells_side = np.int(np.ceil(np.sqrt(len(particles))))
                # Number of cells in each direction
                cell_length = self.box / n_cells_side
                # Length of the cells in each direction
                k_counter = 0
                # Initializing the counter
                grid_places = np.arange(n_cells_side ** 2)
                # Label of each grid place
                np.random.shuffle(grid_places)
                # Distributing the particles randomly to different cells of the grid
                for x_cell in range(n_cells_side):
                    for y_cell in range(n_cells_side):
                        if grid_places[k_counter] < len(particles):
                            particles[
                                grid_places[k_counter]
                            ].position_center = np.array(
                                [
                                    x_cell * cell_length[0] + cell_length[0] / 2,
                                    y_cell * cell_length[1] + cell_length[1] / 2,
                                ]
                            )
                            # Generating the positions from a random uniform distribution
                            self.particle_velocities = np.random.uniform(
                                low=-0.1, high=0.1, size=2
                            )
                            self.position_center_history[grid_places[k_counter]][
                                0
                            ] = particles[grid_places[k_counter]].position_center
                        # # Saving particle history
                        k_counter += 1
        elif self.type_init_conf == "bcc":
            step = self.box[0] / 4
            ind_part = 0
            for k_layer in range(8):
                if np.mod(k_layer, 2) == 0:
                    for (i_row, j_column) in (
                        (i_row, j_column) for i_row in range(4) for j_column in range(4)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [i_row, j_column, k_layer * 0.5]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        ind_part += 1
                else:
                    for (i_row, j_column) in (
                        (i_row, j_column) for i_row in range(4) for j_column in range(4)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [0.5, 0.5, 0]
                        ) + step * np.array(
                            [
                                i_row,
                                j_column,
                                k_layer * 0.5,
                            ]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        # Saving initial configuration
                        ind_part += 1
            print(ind_part, "\n\n\n")

        elif self.type_init_conf == "fcc":
            step = self.box[0] / 4
            ind_part = 0
            for k_layer in range(8):
                if np.mod(k_layer, 2) == 0:
                    for (i_row, j_column) in (
                        (i_row, j_column) for i_row in range(4) for j_column in range(4)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [i_row, j_column, k_layer * 0.5]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        ind_part += 1
                    for (i_row, j_column) in (
                        (i_row, j_column)
                        for i_row in range(1, 5)
                        for j_column in range(4)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [0.5, 0.5, 0]
                        ) + step * np.array(
                            [
                                i_row,
                                j_column,
                                k_layer * 0.5,
                            ]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        # Saving initial configuration
                        ind_part += 1
                else:
                    for (i_row, j_column) in (
                        (i_row, j_column)
                        for i_row in range(1, 5)
                        for j_column in range(4)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [-0.5, 0, 0]
                        ) + step * np.array(
                            [
                                i_row,
                                j_column,
                                k_layer * 0.5,
                            ]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        # Saving initial configuration
                        ind_part += 1
                    for (i_row, j_column) in (
                        (i_row, j_column)
                        for i_row in range(4)
                        for j_column in range(1, 5)
                    ):
                        particles[ind_part].position_center = step * np.array(
                            [0, -0.5, 0]
                        ) + step * np.array(
                            [
                                i_row,
                                j_column,
                                k_layer * 0.5,
                            ]
                        )
                        self.particle_velocities[ind_part] = np.zeros(particles[0].dim)
                        self.position_center_history[ind_part][0] = particles[
                            ind_part
                        ].position_center
                        # Saving initial configuration
                        ind_part += 1
        else:
            try:
                raise errors.UnsupportedInitialConfigurationType(self.type_init_conf)
            except errors.UnsupportedInitialConfigurationType as error:
                error.message()

    @contextmanager
    def virtual_particle_sizes(self, particles):
        """
        Ensure a minimum distance using a virtual particle size larger than the real size.

        Dilate all the particles so that a minimum distance is ensured after contraction at
        the begin of the simulation and contract all the particles so that a minimum
        distance is ensured.

        Parameters
        ----------
        particles: list(`.Particle`)
            List of the particles inside the simulation box.
        """
        for i_particle in particles:
            # Running through all the particles
            i_particle.dilate(self.min_distance / 2)
            # Dilate i_particle
        print(self.microstructure_sample.volume_fraction, "\n\n\n")
        try:
            yield
        finally:
            for i_particle in particles:
                # Running through all the particles
                i_particle.contract(self.min_distance / 2)
                # contract i_particle
            if self.thermostat.__class__.__name__ == "MultiTemperatureIsokineticScheme":
                self.thermostat.equilibration_steps.append(self.thermostat.jump_list)
            if not self.save_history:
                # If the complete motion was not saved
                for i_particle_ind, _ in enumerate(particles):

                    self.position_center_history[i_particle_ind].append(
                        i_particle.position_center.flatten()
                    )
                    # Saving the final configuration
            if self.offset:
                offset = self.compute_rve_offset(particles, self.box)
                for i_particle in particles:
                    # Running through all the particles
                    i_particle.position_center -= np.array(offset)[: len(self.box)]
                    # Applying the offset to the particles

    def run_molecular_dynamics_simulation(self, particles):
        """
        Run the Molecular Dynamics simulation for the system of particles given.

        This is the main function of the Molecular Dynamics simulation. It consists of the
        initialization of the sytem, and the loop that contains the dynamics of the system:
        computation of the forces and integration of the equations of motion.

        Parameters
        ----------
        particles : list(`.Particle`)
            Array containing the Particle objects to be placed inside the RVE.
        """
        with self.virtual_particle_sizes(particles):
            number_particles = len(particles)
            # Saving the number of particles
            self.max_residue = self.max_residue_per_particle * number_particles
            # Maximum total overlap residue
            n_steps_relax = 0
            # Initializing the number of steps that a microstructure was complying with the
            # maximum overlap residue
            self.compute_forces(particles)
            # Computing the forces in the initial configuration to obtain the initial
            # relative potential energy (related to the overlap)
            self.compute_relative_energy()
            # Computing the relative
            self.compute_kinetic_energy(particles)
            # Computing the kinetic energy

            print_funcs.print_to_terminal_refresh(
                self.step,
                self.total_overlap,
                self.relative_energy,
                self.kinetic_energy,
                temp=self.thermostat.reference_temp,
                first=True,
            )
            # # Print info about the iteration
            if self.thermostat.reference_temp < 0:
                vel = (
                    0.25
                    * 10 ** (1 + self.thermostat.reference_temp)
                    * particles[0].radius
                    / self.dt
                )
                # (
                #     1
                #     / np.abs(self.thermostat.reference_temp)
                #     * (2 * particles[0].radius / self.dt - self.dt)
                # )
                # print("{0:e}".format(self.thermostat.reference_temp), "\n\n")
                self.thermostat.reference_temp = (
                    vel ** 2
                    * np.sum(
                        [
                            particle.mass(self.particle_mass_opt)
                            for particle in particles
                        ]
                    )
                    / (particles[0].dim * self.thermostat.k_b * len(particles))
                )

            while (
                self.step < self.max_step
            ) and n_steps_relax <= self.max_steps_to_relax:
                # Run the simulation while the number of steps the overlap has been smaller
                # than the  allowed maximum residue is larger than
                # options['max_steps_to_relax'], so that the  particles have time to get
                # away from each other.
                self.integrate(particles)
                # Integrating the equations of motion
                self.step += 1
                # # Moving to the next time step
                self.compute_forces(particles)
                # Computing the forces on all particles
                self.compute_relative_energy()
                # Computing the relative energy
                self.compute_kinetic_energy(particles)
                self.thermic_enegy_history.append(
                    1
                    / 2
                    * self.thermostat.k_b
                    * self.thermostat.reference_temp
                    * particles[0].dim
                    * len(particles)
                )
                # Computing the kinetic energy
                self.thermostat.apply_thermostat(
                    self.particle_velocities, self.kinetic_energy
                )
                Particle.dist_met = self.dist_met
                # Particle.dist_met = "dist_exact"
                # print(self.dist_met, "\n\n")

                if self.total_overlap <= self.max_residue + 1e-12:
                    # If the configuration has an overlap area smaller than the tolerance
                    n_steps_relax += 1
                else:
                    n_steps_relax = 0
                    # Restarting the count
                print_funcs.print_to_terminal_refresh(
                    self.step,
                    self.total_overlap,
                    self.relative_energy,
                    self.kinetic_energy,
                )
                if self.step > 500 and all(
                    (
                        np.abs(
                            np.array(self.total_overlap_history[-500:])
                            - np.array(self.total_overlap_history[-500 - 1 : -1])
                        )
                    )
                    / np.array(self.total_overlap_history[-500 - 1 : -1])
                    * 100
                    < 1e-2
                ):
                    # if self.step > 50:
                    #     print(
                    #         np.log10(
                    #             np.abs(
                    #                 np.array(self.total_overlap_history[-50])
                    #                 / np.array(self.total_overlap_history[-1])
                    #             )
                    #         ),
                    #         "\n\n",
                    #     )
                    # if self.step > step_rep + 150 and (
                    #     np.log10(
                    #         np.abs(
                    #             np.array(self.total_overlap_history[-50])
                    #             / np.array(self.total_overlap_history[-1])
                    #         )
                    #     )
                    #     < 0.5
                    # ):
                    # If after 500 iterations all the iterations produced a relative change
                    # smaller than 1e-5% assume it is not possible to find a legal
                    # configuration
                    print_funcs.print_to_file("Failed sample")
                    break
                    # self.thermostat.reference_temp *= 10 ** 4
                    # step_rep += 150
                    # # import pdb
                    # #
                    # # pdb.set_trace()
            self.total_overlap = 0
            for i_particle_index, i_particle in enumerate(particles):
                # Running though all the particles
                for j_particle_index, j_particle in enumerate(particles):
                    if j_particle_index is None:
                        continue
                    j_particle = particles[j_particle_index]
                    if j_particle_index > i_particle_index:
                        # Running through the particle pairs that have not been considered yet
                        intersection_area, unit_vector_i_j = getattr(
                            i_particle, self.force_option
                        )(j_particle, self.box)
                        self.total_overlap += intersection_area
                        # Updating the overlap area

    def compute_forces(self, particles):
        """
        Compute the forces between all the particle pairs in the system.

        Parameters
        ----------
        particles : list(`.Particle`)
            Array containing the Particle objects to be placed inside the RVE
        """
        self.speed_up_scheme.new_list(particles)
        # Computing a new list for force computation
        self.particle_forces = [np.zeros(particles[0].dim) for _ in particles]
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
        self.compute_forces_overlap(particles)
        # Computing forces due to particle overlap
        self.compute_forces_thermostat(particles)
        # Computing forces due to the thermostat
        self.compute_forces_damping(particles)
        # Computing forces due to damping
        if self.force_rescale:
            # if self.max_force is None:
            # self.max_force = np.max(
            #     [np.linalg.norm(force) for force in self.particle_forces]
            # )
            #     self.max_force = np.max([i_particle.volume for i_particle in particles])
            # self.current_max_force = np.max(
            #     [np.linalg.norm(force) for force in self.particle_forces]
            # )
            self.particle_forces = [
                force * self.force_rescale_coeff
                # force * self.max_force / self.current_max_force
                if self.current_max_force != 0 else force
                for force in self.particle_forces
            ]

    def compute_forces_overlap(self, particles):
        self.total_overlap = 0
        # Setting the total overlap to zero as it will computed again
        self.particle_overlap_areas = [0 for _ in particles]
        # Setting all the overlap areas to zero at the beginning of the iteration as
        # they are added sequentially as each pair is considered
        coord_number_list = [0 for _ in particles]
        for i_particle_index, i_particle in enumerate(particles):
            # Running though all the particles
            for j_particle_index in set(
                self.speed_up_scheme.particle_list[i_particle_index]
            ):
                if j_particle_index is None:
                    continue
                j_particle = particles[j_particle_index]
                if j_particle_index > i_particle_index:
                    # Running through the particle pairs that have not been considered yet
                    intersection_area, unit_vector_i_j = getattr(
                        i_particle, self.force_option
                    )(j_particle, self.box)
                    self.particle_overlap_areas_dict.setdefault(
                        (i_particle_index, j_particle_index),
                        [0 for _ in range(self.step - 1)],
                    )
                    self.particle_overlap_areas_dict[
                        (i_particle_index, j_particle_index)
                    ] += [intersection_area]
                    # Intersection area between particle i and j
                    self.particle_overlap_areas[i_particle_index] += intersection_area
                    self.particle_overlap_areas[j_particle_index] += intersection_area
                    self.total_overlap += intersection_area
                    # Updating the overlap area
                    force_i_j = -intersection_area * unit_vector_i_j
                    # Computing the force on particle_i due to particle_j proportional to
                    # their intersection area/volume
                    self.particle_forces[i_particle_index] += force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 1
                    self.particle_forces[j_particle_index] -= force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 2
                    if self.dt_adapt == "sqrt":
                        # Computing the number of particles effectively touching
                        # diff_center = (
                        #     i_particle.position_center - j_particle.position_center
                        # )
                        # diff_center = diff_center - self.box * np.round(
                        #     diff_center / self.box
                        # )
                        # # Vector between the centers of the current disk and the nearest image of the other
                        # # disk
                        # distance_spheres = np.sqrt(diff_center.dot(diff_center))
                        # # Distance between the disks
                        # if distance_spheres < (i_particle.radius + j_particle.radius):
                        if any(np.abs(force_i_j) > 0):
                            coord_number_list[i_particle_index] += 1
                            coord_number_list[j_particle_index] += 1

        if self.dt_adapt == "sqrt":
            # Computing the mean coordination number
            self.coord_number = np.mean(coord_number_list)
        # print(np.sum(coord_number_list), np.sum(times), len(times), "\n\n")
        self.total_overlap_history.append(self.total_overlap)

    def compute_forces_thermostat(self, particles):
        if self.thermostat.force_coeff is None:
            pass
        else:
            for i_particle_ind, _ in enumerate(particles):
                self.particle_forces[i_particle_ind] -= (
                    self.thermostat.force_coeff
                    * self.particle_velocities[i_particle_ind]
                )

    def compute_forces_damping(self, particles):
        if self.damping_coeff == 0:
            pass
        else:
            for i_particle_ind, _ in enumerate(particles):
                self.particle_forces[i_particle_ind] -= (
                    self.damping_coeff * self.particle_velocities[i_particle_ind]
                )

    def integrate(self, particles, **kwargs):
        """Integrate the equations of motion."""
        dim = particles[0].dim
        # Dimension of the problem
        if self.dt_adapt == "log":
            self.dt = -0.08 * np.log(
                self.kinetic_energy
                + np.sum(
                    [
                        np.sum(i_particle_force ** 2)
                        for i_particle_force in self.particle_forces
                    ]
                )
            )
        elif self.dt_adapt == "quad":
            max_vel = np.mean(
                [
                    np.linalg.norm(i_particle_vel)
                    for i_particle_vel in self.particle_velocities
                ]
            )
            self.dt = (
                1
                / (
                    np.sqrt(2)
                    * len(particles)
                    * np.pi
                    * 2
                    * (particles[0].radius) ** 2
                    * max_vel
                )
                if max_vel != 0
                else self.dt
            )
        elif self.dt_adapt == "const":
            pass
        elif self.dt_adapt == "sqrt":
            self.all_dt.append(self.dt)
            # max_vel = np.max(
            #     [
            #         np.linalg.norm(i_particle_vel)
            #         for i_particle_vel in self.particle_velocities
            #     ]
            # )
            harm_r = hmean([particle.radius_insc for particle in particles])
            harm_r_2 = hmean([particle.radius for particle in particles])
            adim_mean_path = 1 / (
                np.sqrt(2)
                * particles[0].dim
                * self.microstructure_sample.volume_fraction
            )
            # print("adim_mean_path", adim_mean_path)
            if self.force_option == "force_spring":
                # self.dt = 2 * (
                #     np.min([1.2, adim_mean_path])
                #     * (
                #         np.sqrt(
                #             harm_m
                #             / (
                #                 4
                #                 * (2 * harm_r - self.dt * max_vel)
                #                 * harm_r
                #                 / (2 * harm_r) ** 2
                #             )
                #         )
                #         if max_vel != 0 and 2 * harm_r > self.dt * max_vel
                #         else 0.05
                #     )
                # )
                max_vel = np.max(
                    [
                        np.linalg.norm(i_particle_vel)
                        for i_particle_vel in self.particle_velocities
                    ]
                )
                k_eff = (
                    2 * (2 * harm_r - max_vel * self.dt) / (2 * harm_r)
                    if max_vel != 0 and 2 * harm_r > self.dt * max_vel
                    else 1
                )
                self.dt = np.sqrt(2 / max(1, self.coord_number)) * np.sqrt(
                    harm_r / k_eff
                )

            elif self.force_option == "intersection_length":
                # dt_2 = 2 * (np.min([1, adim_mean_path]) * (np.sqrt(harm_m)))
                # self.dt = np.sqrt(adim_mean_path) * (np.sqrt(harm_r))
                # self.dt = 0.005
                # if self.step == 0:
                #     self.max_vel = 0
                # else:
                #     pass
                # self.max_vel = max(self.max_vel, max_vel)
                # self.dt = (
                #     np.min(
                #         [
                #             1
                #             / 2
                #             / np.sqrt(2)
                #             * np.sqrt(adim_mean_path * np.sqrt(harm_r) / self.max_vel),
                #             np.sqrt(2),
                #         ]
                #     )
                #     * (np.sqrt(harm_m))
                #     if self.max_vel != 0 and 2 * harm_r > self.dt * self.max_vel
                #     else 0.05
                # )
                # print(self.coord_number, "\n\n")
                self.dt = np.sqrt(2 / max(1, self.coord_number)) * np.sqrt(harm_r_2)
                # dt_2 = np.sqrt(2 / max(1, self.coord_number)) * np.sqrt(harm_r)
                # self.dt = 0.05
                # print(self.dt, dt_2, self.coord_number, "\n\n")
                # self.dt = np.sqrt(harm_r)
                # print(
                #     self.dt,
                #     np.sqrt(adim_mean_path) * (np.sqrt(harm_m)),
                #     4
                #     * np.sqrt(2)
                #     * (adim_mean_path * harm_r * np.sqrt(harm_r) / self.max_vel) ** -1,
                #     2 * (adim_mean_path) ** -1,
                #     adim_mean_path,
                #     self.max_vel * np.sqrt(harm_r),
                #     harm_r,
                #     "\n\n\n",
                # )
            # print(
            #     2
            #     * (
            #         np.min([1.2, adim_mean_path])
            #         * (
            #             np.sqrt(
            #                 harm_m
            #                 / (
            #                     2
            #                     * (2 * harm_r - self.dt * max_vel)
            #                     * harm_r
            #                     / (2 * harm_r) ** 2
            #                 )
            #             )
            #             if max_vel != 0 and 2 * harm_r > self.dt * max_vel
            #             else 0.05
            #         )
            #     ),
            #     2 * (np.min([2, adim_mean_path]) * (np.sqrt(harm_m * 2))),
            #     "\n\n\n",
            # )
        for i_particle_index, i_particle in enumerate(particles):
            # Running through all the particles
            if False:  # self.integration_scheme == "newmark":
                # The integration scheme chosen was Newmark
                damping_constant = kwargs.get("damping_coeff", 0)
                [new_position, new_velocity, _] = Newmark(
                    i_particle.position_center,
                    self.particle_velocities[i_particle_index],
                    np.array([self.particle_forces[i_particle_index]], dtype="float").T,
                    i_particle.volume
                    * np.eye(dim, dtype="float"),  # 10e-6*np.eye(2,dtype='float'),#
                    damping_constant * np.eye(dim, dtype="float"),
                    np.zeros(
                        (dim, dim),
                        dtype="float",
                    ),
                    self.dt,
                    1,
                    dim,
                )
                # Obtaining the new position and velocity of particle i

            elif True:  # self.integration_scheme == "verlet":
                # The integration scheme chosen was Verlet
                [new_position, new_velocity] = VerletSync(
                    i_particle.position_center,
                    self.particle_velocities[i_particle_index],
                    np.array([self.particle_forces[i_particle_index]], dtype="float").T,
                    i_particle.mass(self.particle_mass_opt),
                    self.dt,
                    1,
                    dim,
                    # dt_old=self.dt_old,
                )
            new_position[:, 0] = new_position[:, 0] - self.box * np.floor(
                new_position[:, 0] / self.box
            )
            # New position enforcing boundary conditions
            i_particle.position_center = new_position[:, 0]
            self.particle_velocities[i_particle_index] = new_velocity[:, 0]
            # Updating the position and velocity of particle i
            if self.save_history:
                # The history of the particle's motion is required
                self.position_center_history[i_particle_index].append(
                    new_position.flatten()
                )
            # all_vel = np.sum(
            #     [
            #         i_particle.mass(option=self.particle_mass_opt) * i_particle_vel
            #         for i_particle, i_particle_vel in zip(
            #             particles, self.particle_velocities
            #         )
            #     ]
            # )
            # for particle_vel in self.particle_velocities:
            #     particle_vel = particle_vel - all_vel
            # Putting the systemas a whole at rest

    def compute_relative_energy(self):
        """Relative energy computed from the forces between particles."""
        norm_force_vec = np.array(
            [np.linalg.norm(i_force) for i_force in self.particle_forces], dtype="float"
        )
        # Obtaining a list with the norms of the vector forces
        self.relative_energy = norm_force_vec.dot(norm_force_vec)
        # Computing the relative energy

    @property
    def relative_energy(self):
        """Relative energy computed from the forces between particles."""
        return self._relative_energy

    @relative_energy.setter
    def relative_energy(self, relative_energy):
        """Set the relative energy. Saving it in history if the option is enabled."""
        self._relative_energy = relative_energy
        if self.save_history:
            self.relative_energy_history.append(relative_energy)

    def compute_kinetic_energy(self, particles):
        """Kinetic energy of the system of particles."""
        self.kinetic_energy = 0.5 * np.sum(
            [
                i_particle.mass(self.particle_mass_opt) * np.sum(velocity ** 2)
                for i_particle, velocity in zip(particles, self.particle_velocities)
            ]
        )

    @property
    def kinetic_energy(self):
        """Kinetic energy of the system of particles."""
        return self._kinetic_energy

    @kinetic_energy.setter
    def kinetic_energy(self, kinetic_energy):
        """Set the kinetic energy. Saving it in history if the option is enabled."""
        self._kinetic_energy = kinetic_energy
        if self.save_history:
            self.kinetic_energy_history.append(kinetic_energy)

    def place_inner_phase_rsa(self, inner_phase, outer_phase):
        """Place inner phase."""
        rve_dims = np.array(inner_phase.microstructure.rve_dims)
        # Pairing inner phase particles w/ outer phase particles
        # ----------------------------------------------------------------------------------
        out_part_in_part = [[] for _ in outer_phase.particles]
        available_volume = [particle.volume for particle in outer_phase.particles]
        for i_particle_ind, i_inner_particle in enumerate(inner_phase.particles):
            while True:
                outer_particle_ind = np.random.choice(
                    list(range(len(outer_phase.particles))),
                    p=[
                        particle.volume
                        / (
                            outer_phase.volume_fraction
                            * inner_phase.microstructure.volume
                        )
                        for particle in outer_phase.particles
                    ],
                )
                if i_inner_particle.volume < 0.5 * available_volume[outer_particle_ind]:
                    available_volume[outer_particle_ind] -= i_inner_particle.volume
                    out_part_in_part[outer_particle_ind].append(i_inner_particle)
                    print(i_particle_ind)
                    break
        # Placing inner phase particles in each outer phase particle 1-by-1
        # ----------------------------------------------------------------------------------
        k_part = 0
        for i_outer_particle, i_list_inner_particles in zip(
            outer_phase.particles, out_part_in_part
        ):
            placed_particles = []
            for j_inner_particle in i_list_inner_particles:
                j_inner_particle.dilate(0.01 * j_inner_particle.radius)
                i_outer_particle.contract(j_inner_particle.radius * 1.05)
                p_iter = 0
                while p_iter < 200:
                    p_iter += 1
                    j_inner_particle.position_center = (
                        i_outer_particle.generate_point_inside()
                    )
                    j_inner_particle.position_center = np.squeeze(
                        j_inner_particle.position_center
                        - rve_dims
                        * np.floor(j_inner_particle.position_center / rve_dims)
                    )
                    intersect = False
                    for k_particle in placed_particles:
                        if k_particle.intersection(j_inner_particle, rve_dims):
                            intersect = True
                            break
                    if not intersect:
                        placed_particles.append(j_inner_particle)
                        k_part += 1
                        print(k_part, "\n\n")
                        break

                i_outer_particle.dilate(j_inner_particle.radius * 1.05)
                j_inner_particle.contract(0.01 * j_inner_particle.radius)
