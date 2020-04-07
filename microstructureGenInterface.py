"""
# Microstructure Generation Interface (DATAGEM Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
"""
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Dumping files in a binary format
import pickle
# Paht commands
import os
#
import main
#                                                        Microstructure Generation Interface
# ==========================================================================================
# The following function is essentially an interface between the DATAGEM program and a
# given program to generate the microstructure(s) related to a given design point and the
# associated spatial discretization files. The following Q&A summarize the main steps to
# implement a given program to automatically generate microstructures for a given set of
# descriptors.
#
# Q1. What are the input arguments of the microstructure generation interface function
#     (i.e. what is the data available to generate the microstructure(s) associated to
#     a given design point)?
#
# A1. The input arguments are:
#
#     dp_dir - Directory where the microstructure spatial discretization file(s) associated
#              with the given design point are to be stored
#
#     mic_gen_program - Integer variable (read from the user input data file) which
#                       specifies an available program to generate the microstructure(s)
#                       and associated discretization file(s) of a given design point
#
#     mic_gen_parameters - An dictionary which contains all the required parameters (or
#                   options) for the selected program to generate the microstructure(s) and
#                          and associated discretization file(s) of a given design point
#                          (to be discussed...)
#
#     problem_type - Problem type | 1. 2D problem (plain strain)
#                                 | 2. 2D problem (plain stress)
#                                 | 3. 2D problem (axisymmetric)
#                                 | 4. 3D problem
#
#    n_dp_samples - Number of microstructures (samples) to be generated, associated to
#                   the given design point
#
#    mic_gen_descriptors_array - A dictionary which contains all the microstructure
#                                descriptor-related information required to generate the
#                                given design point microstructure(s) automatically,
#                                stored as
#
#                                                     Microstructure Descriptors
#                                               _                                    _
#                     dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
#                                              |_  < value >     < value >      ...  _|
#
#   phase_types - Dictionary which contains each material phase type, stored as
#
#                    dictionary['phase_id'] = phase_type
#
#   discret_file_ext - List which contains the required spatial discretization file(s),
#                      stored as
#
#                             array = [ < discret_type > < discret_type >  ... ]
#
#   discret_spec_array - Dictionary which contains the required parameters to generate
#                        each type of specified discretization file, stored as
#
#                            dictionary['disc_ext']['parameter'] = [ ... ]
#
#                                            -
#
# Q2. How can a microstructure generation program be implemented/used in this program?
#
# A2. The implementation/use of a given microstructure generation program follows the main
#     steps described below:
#
#     a. Add the microstructure generation program option in the user input data file and
#        in the readInputData.py module - stored in mic_gen_program
#     b. Define the microstructure generation program parameters which must be read from the
#        user input data file and implement the associated reading procedure in the
#        readInputData.py - stored in mic_gen_parameters
#     c. Implement the microstructure generation program in the interface according to one
#        of two options:
#
#        c1. Option A - Implement the microstructure generation program as a function so
#                       that the input parameters can be passed as arguments
#                       1. Convert the microstructure generation input data (stored in
#                          mic_gen_descriptors_array and mic_gen_parameters) to the
#                          microstructure generation program required format
#                       2. Call the program to generate the required microstructure(s) and
#                          associated spatial discretization file(s) (the last one based on
#                          the input data stored in the discret_file_ext and
#                          discret_spec_array)
#
#        c2. Option B - Implement the microstructure generation program as a script
#                       (independent program) so that the input parameters are passed
#                       through an input data file (mic_gen_input_data.dat)
#                       1. Write the microstructure generation input data (stored in
#                          mic_gen_descriptors_array and mic_gen_parameters) in a
#                          microstructure generation input data file
#                          (mic_gen_input_data.dat) according to the microstructure
#                          generation program required format
#                       2. Run the program to generate the required microstructure(s)
#                          and associated spatial discretization file(s) (the last one based
#                          on the input data stored in the discret_file_ext and
#                          discret_spec_array)
#
# Q3. What is the required format of the spatial discretization file(s) and where should
#     they be stored?
#
# A3. The required format of the spatial discretization file(s) is intrinsically related
#     with the selected solution method (program) specified in the user input data file and
#     based on the input data stored in the discret_file_ext and discret_spec_array.
#     Nonetheless, the spatial discretization file(s) associated to the given design point
#     must be stored in the dp_dir directory and named as
#     DPX_SY_Microstructure.< discret_file_ext > (where X is the design point number,
#     omitted in single analysis, and Y is the sample number).
#
#     Note: For the sake of memory usage, all the auxiliary files which are created in
#           dp_dir in order to generate the required spatial discretization file(s) shall
#           be deleted afterwards.
#


def generateMicrostructures(dp_dir, mic_gen_program, mic_gen_parameters, problem_type,
                            n_dp_samples, mic_gen_descriptors_array, phase_types,
                            discret_file_ext, discret_spec_array):
    """
    Send descriptors and options to a program in order to generate microstructures.

    This function is used to interface with the program that generates the microstructures.

    Parameters
    ----------
    dp_dir: string
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored

    mic_gen_program: integer
        Integer variable (read from the user input data file) which specifies an
        available program to generate the microstructure(s) and associated
        discretization file(s) of a given design point

    mic_gen_parameters: array
        An array which contains all the required parameters (or options)
        for the selected program to generate the microstructure(s) and
        and associated discretization file(s) of a given design point.

        ================================ ======================================
        Option                           Description
        ================================ ======================================
        "max_residue_per_particle"       Maximum overlap residue per particle.
        "max_step"                       Maximum number of iterations.
        "integration_scheme"             Optional. {'Newmark'}. Integration scheme
                                         for the equations of motion.
        "speed_up_scheme"                Optional. {'Naive', 'Cell', 'Verlet'}.
                                         Speed up scheme used for force computation
        ================================ ======================================

    problem_type: integer
        Problem type    | 1. 2D problem (plain strain)
                        | 2. 2D problem (plain stress)
                        | 3. 2D problem (axisymmetric)
                        | 4. 3D problem

    n_dp_samples: integer
        Number of microstructures (samples) to be generated, associated to
        the given design point

    mic_gen_descriptors_array: array
        A dictionary which contains all the microstructure
        descriptor-related information required to generate the
        given design point microstructure(s) automatically stored as:

                                        Microstructure Descriptors
                                  _                                    _
        dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
                                 |_  < value >     < value >      ...  _|.

        See notes_.

    phase_types: dictionary
        Dictionary which contains each material phase type, stored as
                       dictionary['phase_id'] = phase_type
    discret_file_ext: list
        List which contains the required spatial discretization file(s), stored as:

                        array = [ < discret_type > < discret_type >  ... ]

    discret_spec_array: dictionary
        Dictionary which contains the required parameters to generate
        each type of specified discretization file, stored as:

                               dictionary['disc_ext']['parameter'] = [ ... ]
    Notes
    -----
    The parameters for microstructure generation depend on the shape of the particle. They
    are detailed in the following tables. Particular choices of their values may lead to
    incompatibilities.

        ================================ ======================================
        Disk: Choose 2 of the parameters
        -----------------------------------------------------------------------
        'r'                              Radius of the disk
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

        ================================ ======================================
        Ellipse: Choose 4 of the parameters, including 'angle'
        -----------------------------------------------------------------------
        'major_axis'                     Radius of the disk
        'minor_axis'                     Number of particles
        'angle'                          Volume fraction
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

    Any parameter may have a chosen distribution, specified as detailed below:
    - Fixed distribution: The parameters are fixed. Simply specify the parameter.
    - Discrete distribution: There parameters follow a discrete distribution, where the
    parameters take only the given values with the given probability.
        1. Specify the distribution of parameter *param* as::

                    np.array([['distribution_param']['fixed'], dtype=obj)

        2. Specify the value of the parameter and the probability of that value occuring::

                (np.array([['param_1', 'prob_param_1', 'param_2', 'prob_param_2'],
                        [1, 0.4, 2, 0.6]], dtype=obj))

    - Uniform distribution:
    - Gaussian distribution:
    """
    if mic_gen_program == 1:
        # My program
        mic_gen_descriptors_dict = {}
        # Initializing the dictionary containing the microstructure descriptors
        for i_phase in mic_gen_descriptors_array:
            # Running through all the phases
            mic_gen_descriptors_dict[i_phase] = (
                {mic_gen_descriptors_array[i_phase][0, i]:
                    mic_gen_descriptors_array[i_phase][1, i]
                    for i in range(len(mic_gen_descriptors_array[i_phase][0]))})
        info_dict = {
            "dp_dir": dp_dir,
            "mic_gen_parameters": mic_gen_parameters,
            "problem_type": problem_type,
            "n_dp_samples": n_dp_samples,
            "mic_gen_descriptors": mic_gen_descriptors_dict,
            "phase_types": phase_types,
            "discret_file_ext": discret_file_ext,
            "discret_spec_array": discret_spec_array
            }
        # Building a dictionary to be pickled with all the information coming from the
        # interfacing program
        pickle.dump(info_dict, open("input_data\\info_micro.p", "wb"))
        # Dumping the info_dict dictionary into info_micro.p to be loaded in the program
        # that generates microstructures
        main.main()
        # Executing the script for microstructure generation


if __name__ == '__main__':

    # ======================================================================================
    # dp_dir: string
    #     Directory where the microstructure spatial discretization file(s) associated
    #     with the given design point are to be stored
    dp_dir = ("C:\\Users\\José\\Notebooks\\Database"
              + "\\Universidade\\Dissertacao\\programa\\results")
    # ======================================================================================
    # mic_gen_program: integer
    #     Integer variable (read from the user input data file) which specifies an
    #     available program to generate the microstructure(s) and associated
    #     discretization file(s) of a given design point
    mic_gen_program = 1
    # ======================================================================================
    # mic_gen_parameters: array
    #     An array which contains all the required parameters (or options)
    #     for the selected program to generate the microstructure(s) and
    #     and associated discretization file(s) of a given design point
    #     (to be discussed...)
    mic_gen_parameters = {}
    # Initializing the dictionary containing the options
    #                                                                    Stopping criteria
    # --------------------------------------------------------------------------------------
    mic_gen_parameters['max_residue_per_particle'] = 1e-8
    mic_gen_parameters['max_step'] = 10000
    mic_gen_parameters['max_steps_to_relax'] = 250
    mic_gen_parameters['speed_up_scheme'] = 'Verlet'
    mic_gen_parameters['verlet_factor'] = 1.2
    # Maximum number of steps
    problem_type = 1
    # n_dp_samples: integer
    #     Number of microstructures (samples) to be generated, associated to
    #     the given design point
    n_dp_samples = 1
    # mic_gen_descriptors_array: dictionary
    #     A dictionary which contains all the microstructure
    #     descriptor-related information required to generate the
    #     given design point microstructure(s) automatically,
    #     stored as
    #                                     Microstructure Descriptors
    #                               _                                    _
    #     dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
    #                              |_  < value >     < value >      ...  _|
    #
    mic_gen_descriptors_array = {}

    mic_gen_descriptors_array['4'] = np.array([['rve_dims'], [[1.0, 1.0, 1.0]]])
    mic_gen_descriptors_array['2'] = np.array([['n', 'vf'], [20, 0.65]], dtype=object)

    # Types of particles
    # 1 - Matrix
    # 2 - Circular particle (disk)
    # 3 - Elliptical particle
    # 4 - Spherical particle
    phase_types = {}
    phase_types['4'] = 1  # Matrix
    phase_types['2'] = 2  # Elliptical particle
    # discret_file_ext: list
    #     List which contains the required spatial discretization file(s), stored as
    #                     array = [ < discret_type > < discret_type >  ... ]

    # discret_spec_array: dictionary
    #     Dictionary which contains the required parameters to generate
    #     each type of specified discretization file, stored as
    #                            dictionary['disc_ext']['parameter'] = [ ... ]

    discret_file_ext = []

    discret_spec_array = {}
    discret_spec_array['rgmsh'] = {}
    discret_spec_array['rgmsh']['rve_dims'] = np.array([1.0, 1.0])
    discret_spec_array['rgmsh']['n_voxels_dims'] = np.array([50, 50])
    # discret_spec_array['femsh'] = {}
    # discret_spec_array['femsh']['rve_dims'] = np.array([1.0, 1.0, 1.0])
    # discret_spec_array['femsh']['mesh_size'] = 0.1

    generateMicrostructures(dp_dir, mic_gen_program, mic_gen_parameters, problem_type,
                            n_dp_samples, mic_gen_descriptors_array, phase_types,
                            discret_file_ext, discret_spec_array)
