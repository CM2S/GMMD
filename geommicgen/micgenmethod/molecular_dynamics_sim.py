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

# pylint: disable=import-error
import errors.error_classes as errors
import iofuncs.printing as print_funcs
from micgenmethod.microstructure_gen_method import GenerationMethod
from micgenmethod.integration_methods import VerletSync, Newmark
from microstructure.particle_classes import Matrix


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

        time: float
            Time taken by the simulation.
        """
        self.box = None
        self.particle_velocities = None
        self.particle_forces = None
        self.particle_overlap_areas = None
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

    def generate_microstructure(self, microstructure_sample):
        """
        Generate the microstructure for the sample supplied.

        Generate the microstructure for microstructure_sample using the microstucutre
        generation method *self*.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure sample to be generated
        """
        for phase in microstructure_sample.phases.values():
            if phase.type is not Matrix:
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
            Dimensions of the microstucutre in each spatial direction.
        """
        if any(
            [
                particle.__class__.__name__ == "CylindricalFiber"
                for particle in particles
            ]
        ):
            ax_ind = {"x": 0, "y": 1, "z": 2}
            self.box = rve_dims
            del self.box[ax_ind[particles[0].direction_fibers]]
        else:
            self.box = rve_dims

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
            # np.random.seed(42)
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
                n_cells_side = np.int(np.ceil(np.cbrt(len(particles))))
                # Number of cells in each direction
                cell_length = self.box / n_cells_side
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
                                self.position_center_history[0][
                                    grid_places[k_counter]
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
                            self.position_center_history[0][
                                grid_places[k_counter]
                            ] = particles[grid_places[k_counter]].position_center
                            # # Saving particle history
                        k_counter += 1
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

        min_distance: float
            Minimum distance between two particles.
        """
        for i_particle in particles:
            # Running through all the particles
            i_particle.dilate(self.min_distance)
            # Dilate i_particle
        try:
            yield
        finally:
            for i_particle in particles:
                # Running through all the particles
                i_particle.contract(self.min_distance)
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
        self.integration_scheme = "verlet"
        with self.virtual_particle_sizes(particles):
            number_particles = len(particles)
            # Saving the number of particles
            self.max_residue = self.max_residue_per_particle * number_particles
            # Maximum total overlap residue
            self.step = 0
            # Initializing the the time step at 0
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
            while (
                self.step < self.max_step
            ) and n_steps_relax < self.max_steps_to_relax:
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
                # Computing the kinetic energy
                self.thermostat.apply_thermostat(
                    self.particle_velocities, self.kinetic_energy
                )
                # Contract all particles
                if self.total_overlap <= self.max_residue:
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
                    < 1e-5
                    # If after 500 iterations all the iterations produced a relative change
                    # smaller than 1e-5% assume it is not possible to find a legal
                    # configuration
                ):
                    print_funcs.printToFile("Failed sample")
                    break

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
        self.total_overlap = 0
        # Setting the total overlap to zero as it will computed again
        self.particle_overlap_areas = [0 for _ in particles]
        # Setting all the overlap areas to zero at the beginning of the iteration as
        # they are added sequentially as each pair is considered
        self.particle_forces = [np.zeros(particles[0].dim) for _ in particles]
        # Setting all forces to zero at the beginning of the iteration as they are added
        # sequentially as each pair is considered
        for i_particle_index, i_particle in enumerate(particles):
            # Running though all the particles
            for j_particle_index in self.speed_up_scheme.particle_list[
                i_particle_index
            ]:
                if j_particle_index is None:
                    continue
                j_particle = particles[j_particle_index]
                if j_particle_index > i_particle_index:
                    # Running through the particle pairs that have not been considered yet
                    intersection_area = i_particle.intersection_area(
                        j_particle, self.box
                    )
                    # Intersection area between particle i and j
                    self.particle_overlap_areas[i_particle_index] += intersection_area
                    self.particle_overlap_areas[j_particle_index] += intersection_area
                    self.total_overlap += intersection_area
                    # Updating the overlap area
                    unit_vector_i_j = i_particle.intersection_vector(
                        j_particle, self.box
                    )
                    # Unit vector from particle i to particle j
                    force_i_j = -intersection_area * unit_vector_i_j
                    # Computing the force on particle_i due to particle_j proportional to
                    # their intersection area/volume
                    self.particle_forces[i_particle_index] += force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 1
                    self.particle_forces[j_particle_index] -= force_i_j
                    # Adding the force due to the interaction between particle 1 and 2 to
                    # the total force acting on particle 2
        self.total_overlap_history.append(self.total_overlap)

    def integrate(self, particles, **kwargs):
        """Integrate the equations of motion."""
        dim = particles[0].dim
        # Dimension of the problem
        for i_particle_index, i_particle in enumerate(particles):
            # Running through all the particles
            if self.integration_scheme == "newmark":
                # The integration scheme chosen was Newmark
                damping_constant = kwargs.get("damping_constant", 0)
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

            elif self.integration_scheme == "verlet":
                # The integration scheme chosen was Verlet
                [new_position, new_velocity] = VerletSync(
                    i_particle.position_center,
                    self.particle_velocities[i_particle_index],
                    np.array([self.particle_forces[i_particle_index]], dtype="float").T,
                    i_particle.volume,
                    self.dt,
                    1,
                    dim,
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
            # putSystemAtRest(particles)
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
        self.kinetic_energy = np.sum(
            [
                i_particle.volume * np.sum(velocity ** 2)
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
