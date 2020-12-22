""" Module to import a microstructure form a csv file generated from ImageJ. """
import numpy as np
import os


# pylint: disable=import-error
import iofuncs.printing as print_funcs
from microstructure.phase import Phase
from microstructure.microstructure import Microstructure


def generate_microstructure_from_csv(file_path):
    """
    Generate the microstructure for the sample supplied.

    Generate the microstructure for microstructure_sample using the microstructure
    generation method *self*.

    Parameters
    ----------
    file_path: str
        File path of the form "shape_xdim_ydim_vf".
    """
    mic_info = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    # Loading the microstructure info. Assumed to be
    # N,Area,Mean,Min,Max,XM,YM,Major,Minor,Angle

    # Getting info from file name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_name = os.path.basename(file_path)
    try:
        shape, x_dim, y_dim, _ = file_name.split("_")
    except ValueError as error:
        raise ValueError(
            "Filename is not of the form 'shape_xdim_ydim_vf': {0}".format(file_name)
        ) from error
    try:
        x_dim = float(x_dim)
        y_dim = float(y_dim)

    except ValueError as error:
        raise ValueError(
            """The values in the filename for x_dim or y_dim are not numbers.': {0},
            {1}.""".format(
                x_dim, y_dim
            )
        ) from error

    box = [x_dim / np.max([x_dim, y_dim]), y_dim / np.max([x_dim, y_dim])]

    microstructure_sample = Microstructure(box)

    # Building descriptors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if shape == "disks":
        n_particles = len(mic_info)
        area_vals = []
        positions = []
        for i_particle_info in mic_info:
            area_vals.append(i_particle_info[1] / (x_dim * y_dim))
            positions.append(
                np.array(
                    [
                        i_particle_info[5] / np.max([x_dim, y_dim]),
                        box[1] - i_particle_info[6] / np.max([x_dim, y_dim]),
                    ]
                )
            )
            descriptors = {
                "phase_type": 2,
                "n": n_particles,
                "area": area_vals,
                "area_distribution": "specified",
            }
    else:
        n_particles = len(mic_info)
        major_axis_vals = []
        minor_axis_vals = []
        area_vals = []
        angle_vals = []
        positions = []
        for i_particle_info in mic_info:
            major_axis_vals.append(i_particle_info[7])
            minor_axis_vals.append(i_particle_info[8])
            angle_vals.append(i_particle_info[9] / 180 * np.pi)
            positions.append(
                np.array(
                    [
                        i_particle_info[5] / np.max([x_dim, y_dim]),
                        box[1] - i_particle_info[6] / np.max([x_dim, y_dim]),
                    ]
                )
            )
        major_axis_vals = np.array(major_axis_vals) / np.max([x_dim, y_dim])
        minor_axis_vals = np.array(minor_axis_vals) / np.max([x_dim, y_dim])
        ratio_vals = major_axis_vals / minor_axis_vals
        descriptors = {
            "phase_type": 3,
            "n": n_particles,
            "major_axis": major_axis_vals,
            "major_axis_distribution": "specified",
            # "minor_axis": minor_axis_vals,
            # "minor_axis_distribution": "specified",
            "ratio": ratio_vals,
            "ratio_distribution": "specified",
            "angle": angle_vals,
            "angle_distribution": "specified",
        }
    # Generating phases and particles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # microstructure_sample.add_phase(Phase("1", {"phase_type": 1}))
    microstructure_sample.add_phase(Phase("2", descriptors))
    for phase in microstructure_sample.phases.values():
        phase.generate_particles(microstructure_sample.rve_dims)
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
        print(positions[i_particle_ind])

    print_funcs.print_microstructure_info(microstructure_sample)

    return microstructure_sample
