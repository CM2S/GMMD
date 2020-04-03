#
# Microstructure Generation Interface (DATAGEM Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Dumping files in a binary format
import pickle
#
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
#     mic_gen_parameters - An array which contains all the required parameters (or options)
#                          for the selected program to generate the microstructure(s) and
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
def generateMicrostructures(dp_dir,mic_gen_program,mic_gen_parameters,problem_type,
                            n_dp_samples,mic_gen_descriptors_array,phase_types,
                            discret_file_ext,discret_spec_array):
    '''
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
        and associated discretization file(s) of a given design point
        (to be discussed...)

    problem_type: integer
        Problem type    | 1. 2D problem (plain strain)
                        | 2. 2D problem (plain stress)
                        | 3. 2D problem (axisymmetric)
                        | 4. 3D problem

    n_dp_samples: integer
        Number of microstructures (samples) to be generated, associated to
        the given design point

    mic_gen_descriptors_array: dictionary
        A dictionary which contains all the microstructure
        descriptor-related information required to generate the
        given design point microstructure(s) automatically,
        stored as:

                                        Microstructure Descriptors
                                  _                                    _
        dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
                                 |_  < value >     < value >      ...  _|

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
    '''

    if 'mic_gen_program' == 1:
    # My program
        info_dict = { \
            "dp_dir":dp_dir, \
            "mic_gen_parameters":mic_gen_parameters, \
            "problem_type":problem_type, \
            "n_dp_samples":n_dp_samples, \
            "mic_gen_descriptors_array":mic_gen_descriptors_array, \
            "phase_types":phase_types, \
            "discret_file_ext":discret_file_ext, \
            "discret_spec_array":discret_spec_array \
            }
        # Building a dictionary to be pickled with all the information coming from the
        # interfacing program
        pickle.dump(info_dict, open("info_micro.p", "wb"))
        # Dumping the info_dict dictionary into info_micro.p to be loaded in the program
        # that generates microstructures
        execfile('main.py')
        # Executing the script for microstructure generation
