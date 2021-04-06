"""Loading micorstructures from files."""

import numpy as np
from microstructure.microstructure import Microstructure
from microstructure.phase import Phase
import iofuncs.printing as print_funcs


def generate_microstructure_from_txt(file_path):
    """
    Generate the microstructure for the sample supplied.

    Generate the microstructure for microstructure_sample using the microstructure
    generation method *self*.

    Parameters
    ----------
    file_path: str
        File path of the form "xdim_ydim".
    """
    with open(file_path, "r") as input_file:
        input_file.readline()
        box = [int(dim) for dim in input_file.readline().split(",")]
        input_file.readline()
        phase_type = input_file.readline()

    mic_info = np.genfromtxt(file_path, delimiter=",", names=True, skip_header=4)
    if mic_info.shape == ():
        mic_info = mic_info.reshape((1,))
    # Loading the microstructure info. Assumed to be
    # N,Area,Mean,Min,Max,XM,YM,Major,Minor,Angle

    # Getting info from file name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    microstructure_sample = Microstructure(box)

    # Building descriptors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    descriptors = {}
    for descriptor in mic_info.dtype.names:
        if not descriptor.startswith("pos_"):
            descriptors[descriptor] = []
            descriptors["{0}_distribution".format(descriptor)] = "specified"

    positions = []
    for i_part_ind, i_particle_info in enumerate(mic_info):
        positions.append([])
        for descriptor_value, descriptor in zip(i_particle_info, mic_info.dtype.names):
            if descriptor.startswith("pos_"):
                if descriptor_value < 0:
                    raise ValueError("The position of the particle must be positive.")
                # assumes that the order pos_x, pos_y, pos_z
                positions[-1].append(descriptor_value)
            else:
                descriptors[descriptor].append(descriptor_value)
        if len(positions[-1]) != len(box):
            raise ValueError("Too many positions supplied")
        if np.any(np.array(positions[-1]) > np.array(box)):
            raise ValueError("The particle {0} is outside the box.".format(i_part_ind))

    descriptors["n"] = len(mic_info)
    descriptors["phase_type"] = phase_type
    # Generating phases and particles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    microstructure_sample.add_phase(Phase("2", descriptors))
    for phase in microstructure_sample.phases.values():
        phase.generate_particles(microstructure_sample.rve_dims)
    microstructure_sample.add_phase(Phase("1", {"phase_type": 1}))
    # if microstructure_sample.volume_fraction > 1:
    #     raise ValueError(
    #         "The volume fraction goes over 1: {0}".format(
    #             microstructure_sample.volume_fraction
    #         )
    #     )
    # Setting the position of the particles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i_particle_ind, i_particle in enumerate(microstructure_sample.particles):
        i_particle.position_center = positions[i_particle_ind]

    print_funcs.print_microstructure_info(microstructure_sample)

    return microstructure_sample
