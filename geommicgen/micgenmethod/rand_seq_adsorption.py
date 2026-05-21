"""
Alterar descrição mais tarde
Module containing the RandomSequentialAdsorption class.

It provides a class whose methods allow the performance of a random sequential adsorption method. In this method, particles are generated one by one and added to the system if they do not intersect with any of the particles already present in the system. The process is repeated until a certain number of particles is reached or until a certain volume fraction is reached.

"""

from contextlib import contextmanager
import os
import time
from tracemalloc import start
import numpy as np
from scipy.stats import hmean
import geommicgen.postproc.plotfuncs.plotting_functions as plot_funcs

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
import geommicgen.errors.error_classes as errors
import geommicgen.iofuncs.file_handling as fileio
import geommicgen.iofuncs.printing as print_funcs
from geommicgen.microstructure.particleclasses import Matrix
from geommicgen.micgenmethod.microstructure_gen_method import GenerationMethod
from geommicgen.micgenmethod import speed_up_schemes


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
    
    max_step: int
        Maxium number of steps

    min_distance: float
        Minimum distance between particles.

    status: bool
        Status of the simulation. Successfull or failed.

    speed_up_scheme: `.SpeedUpScheme`
        Speed up scheme for intersection computation.

    save_history: bool
        Save all the trajectories of the particles, the history of the relative and
        kinetic energy.

    RSA_vf_history: list
        history of the volume fraction

    intersection_check_history: list
        Number of intersection checks done at every step

    accepted_particles_history: list
        Number of particles accepeted in the microstructure per step
    
    Keyword Parameters
        ------------------

    simulation_gif: bool
            Make a gif of the simulation.

    """

    def __init__(self,max_step,min_distance,save_history, sample_dir, **kwargs):
        """Initialize the random sequential adsorption simulation class.
        
        Parameters
        ----------
        
        max_step: int
            Maxium number of steps

        min_distance: float
            Minimum distance between particles.

        save_history: bool
            Save all the trajectories of the particles, the history of the relative and
            kinetic energy.

        sample_dir: str
            Directory of the sample for which the microstructure is being generated.

        Keyword Parameters
            ------------------
            
        simulation_gif: bool
                Make a gif of the simulation.
        """
        #self.mic_gen_parameters = mic_gen_parameters
        #self.mic_gen_descriptors = mic_gen_descriptors
        #self.microstructure_sample = None
        self.time= None
        self.microstructure_sample = None
        self.max_step = max_step
        self.min_distance = min_distance
        self.status = False
        self.RSA_vf_history = []
        self.speed_up_scheme = None
        self.save_history = save_history
        self.intersection_check_history = []
        self.accepted_particles_history = []
        self.sample_dir = sample_dir
        self.make_gif = kwargs.get("simulation_gif")


    def generate_microstructure(self, microstructure_sample):
        """Generate microstructure."""
        self.microstructure_sample = microstructure_sample
        self.set_box(microstructure_sample.particles, microstructure_sample.rve_dims)
        start = time.time()
        n_particles_total = 0
        n_iteration = 0
        print_funcs.print_to_terminal_refresh_rsa(
            n_iteration,
            n_particles_total,
            first=True
            )
        
        if self.make_gif:
            # Create folder for the gif if it does not exist
            gif_dir = os.path.join(self.sample_dir, "Simulation_gif")
            os.makedirs(gif_dir)
        
        self.particle_overlap_areas = [0 for _ in microstructure_sample.particles]
        # Setting all the overlap areas to zero at the beginning of the iteration as
        # they are added sequentially as each pair is considered
        for i_phase in microstructure_sample.phases.values():
            if i_phase.type is not Matrix and not i_phase.inner_phase:
                # Determine which target to use
                if "n" in i_phase.descriptors:
                    target_value = i_phase.descriptors["n"].value
                    condition = lambda: len(i_phase.particles) < i_phase.descriptors["n"].value
                elif "vf" in i_phase.descriptors:
                    target_value = i_phase.descriptors["vf"].value
                    condition = lambda: i_phase.real_volume_fraction < target_value
                else:
                    raise errors.MicrostructureGenerationError(
                        "Neither number of particles nor volume fraction specified for phase {0}".format(i_phase)
                    )

                while condition():
                    # if number of iterations is higher than the maximum number of iterations
                    if n_iteration > self.max_step:
                        print("Simulation reached the maximum number of iterations.")
                        break
                    n_iteration += 1
                        
                    new_particle = i_phase.generate_single_particle(microstructure_sample.rve_dims)
                    new_particle.dilate(self.min_distance / 2)
                    # Dilate the particle if there is a minimum distance imposed. It will later be contracted back to its original size. If there is no minimum distance, min_distance is 0, so the particle will not be dilated.

                    # Get list of particles
                    if len(microstructure_sample.particles) > 0:
                        self.speed_up_scheme.new_list(microstructure_sample.particles,new_particle)
                        particles_list = self.speed_up_scheme.particle_list
                        if self.save_history:
                            self.intersection_check_history.append( len(particles_list) )
                    else:
                        particles_list = []

                    if not self.check_intersection(new_particle,particles_list):
                        i_phase.particles.append(new_particle)
                        n_particles_total += 1                       
                    if self.save_history:
                        self.RSA_vf_history.append(microstructure_sample.volume_fraction)
                        self.accepted_particles_history.append(n_particles_total)
                    print_funcs.print_to_terminal_refresh_rsa(
                    n_iteration,
                    n_particles_total
                    )

                    # save image of every step of simulation
                    if self.make_gif:
                        kwargs = {"simulation_gif": True, "simulation_gif_dir" : gif_dir, "iteration": n_iteration, "save": False}
                        plot_funcs.plot_particles(microstructure_sample.particles, microstructure_sample.rve_dims, self.sample_dir, **kwargs)
 
                # Contract all the particles back to their original. If there is no minimum distance, the particles will not be contracted, since min_distance is 0.
                real_vf = self.microstructure_sample.real_volume_fraction
                virtual_vf = self.microstructure_sample.volume_fraction
                self.contract_all_particles(i_phase.particles)
        
                    
        # if the simulation stopped because it reached the maximum number of iterations, status is Flase, otherwise it is True.
        if n_iteration < self.max_step:
            self.status = True
           
        self.time = time.time() - start
        print("\n\n")
        print_funcs.print_microstructure_info(microstructure_sample)
        if self.min_distance != 0:
            print_funcs.print_virtual_total_volume_fraction(
                real_vf, virtual_vf, self.min_distance
            )
        print_funcs.print_final_message_rsa(self.time, n_iteration)
                


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

    def set_speed_up_scheme(self, speed_up_scheme):
        """Set speed up scheme for the random sequential adsorption simulation."""
        self.speed_up_scheme = speed_up_scheme
        speed_up_scheme.rsa_sim = self


    def check_intersection(self, new_particle, particles_list):
        """Check if the new_particle intersects any of the particles in the list.

        Input:
            - new_particle: an object from class particle
            - particles_list: list of numbers. Each number is the index of the particles that must be checked for intersection

        Output:
            -True if it intersects with any particle, False otherwise.
        """
        start = time.perf_counter()

        if self.microstructure_sample.particles ==[]:
            duration = time.perf_counter() - start
            with open("check_intersection_times.txt", "a") as f:
                f.write(f"{duration}, ")
            return False
        else:
            for i in particles_list:
                i_particle = self.microstructure_sample.particles[i]
                if new_particle.intersection(i_particle, self.microstructure_sample.rve_dims):
                    duration = time.perf_counter() - start
                    with open("check_intersection_times.txt", "a") as f:
                        f.write(f"{duration}, ")
                        return True
        
        duration = time.perf_counter() - start
        with open("check_intersection_times.txt", "a") as f:
            f.write(f"{duration}, ")
           
        return False