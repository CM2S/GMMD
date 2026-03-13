"""
Alterar descrição mais tarde
Module containing the RandomSequentialAdsorption class.

It provides a class whose methods allow the performance of a random sequential adsorption method. In this method, particles are generated one by one and added to the system if they do not intersect with any of the particles already present in the system. The process is repeated until a certain number of particles is reached or until a certain volume fraction is reached.

"""

from contextlib import contextmanager
import time
import numpy as np
from scipy.stats import hmean

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
import geommicgen.errors.error_classes as errors
import geommicgen.iofuncs.file_handling as fileio
import geommicgen.iofuncs.printing as print_funcs
from geommicgen.microstructure.particleclasses import Matrix
from geommicgen.micgenmethod.microstructure_gen_method import GenerationMethod
from geommicgen.micgenmethod.integration_methods import verlet_sync_integration


class RandomSequentialAdsorption(GenerationMethod):
    """Class for the random sequential adsorption simulation class.

    It stores all the options specifying how the simlulation will be run, it contains the
    methods needed to run the simulation and it also stores the relevant details of the
    simulation.


    Attributes
    ----------
    box: list
        List of the dimensions of the simulation box. Almost always equal to the the
        dimensions of the microstructure, except for CylindricalFibers.

    microstructure_sample: `.Microstructure`
        Microstructure to be generated.

    to be added
    """

    def __init__(self,max_step):
        """Initialize the random sequential adsorption simulation class."""
        #self.mic_gen_parameters = mic_gen_parameters
        #self.mic_gen_descriptors = mic_gen_descriptors
        #self.microstructure_sample = None
        self.time= None
        self.microstructure_sample = None
        self.max_step = max_step

    def generate_microstructure(self, microstructure_sample):
        """Generate microstructure."""
        self.microstructure_sample = microstructure_sample
        self.set_box(microstructure_sample.particles, microstructure_sample.rve_dims)
        start = time.time()
        n_particles_total = 0
        n_iterations = 0
        print_funcs.print_to_terminal_refresh_rsa(
            n_iterations,
            n_particles_total,
            first=True
            )
        
        for i_phase in microstructure_sample.phases.values():
            if i_phase.type is not Matrix and not i_phase.inner_phase:
                try:
                    n_particles_target = i_phase.descriptors["n"].value
                except KeyError:
                    pass
                else:                   
                    while len(i_phase.particles) < n_particles_target:
                        # if number of iterations is higher than the maximum number of iterations
                        if n_iterations > self.max_step:
                            print("Simulation reached the maximum number of iterations.")
                            break
                        n_iterations += 1
                        new_particle = i_phase.generate_single_particle(microstructure_sample.rve_dims)
                        if not self.check_intersection(new_particle, microstructure_sample.phases):
                            i_phase.particles.append(new_particle)
                            n_particles_total += 1
                        print_funcs.print_to_terminal_refresh_rsa(
                        n_iterations,
                        n_particles_total
                        )
                if "vf" in i_phase.descriptors and "n" not in i_phase.descriptors:
                    vf_target= i_phase.descriptors["vf"].value             
                    while i_phase.volume_fraction < vf_target:
                        # if number of iterations is higher than the maximum number of iterations
                        if n_iterations > self.max_step:
                            print("Simulation reached the maximum number of iterations.")
                            break
                        n_iterations += 1
                        new_particle = i_phase.generate_single_particle(microstructure_sample.rve_dims)
                        if not self.check_intersection(new_particle, microstructure_sample.phases):
                            i_phase.particles.append(new_particle)
                            n_particles_total += 1
                        print_funcs.print_to_terminal_refresh_rsa(
                        n_iterations,
                        n_particles_total
                        )
                                
        self.time = time.time() - start
        print("\n\n")
        print_funcs.print_microstructure_info(microstructure_sample)
        print_funcs.print_final_message_rsa(self.time, n_iterations)
        


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


    def check_intersection(self, particle, particles):
        """Check if the particle intersects any of the particles in the list.
        True if it intersects, False otherwise."""
        # check if the particle intersects any of the particles in the list
        for i_phase in particles.values():
            for i_particle in i_phase.particles:
                if particle.intersection(i_particle, self.microstructure_sample.rve_dims):
                    return True
        return False


