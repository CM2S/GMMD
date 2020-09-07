class Simulation:
    """Class for the simulations that generate the microstructures."""

    def __init__(self, working_directory):
        """Initizalizer for the Simulation Class."""

        self.dp_dir = working_directory
        #     Directory where the microstructure spatial discretization file(s) associated
        #     with the given design point are to be stored
        self.mic_gen_parameters = {}
        # Dictionayr with generation parameters
        self.mic_gen_descriptors = {}
        # Dictionaty containing the descriptors for the phases
        self.phase_types = {}
        # Dictionary containing the phase types
        self.discret_file_ext = []
        # list containing the extensions for the output mesh files
        self.discret_spec_array = {}
        # Parameters for the generation of the meshes
        self.problem_type = 0
        # Problem type
        self.n_dp_samples = 0
        # Number of samples to be generated

    def parametersChecks(self):

        try:
            if self.n_dp_samples < 1:
                # The number of samples must be an integer larger or equal to 1
                raise errors.NumberSamples(self.n_dp_samples)
        except errors.NumberSamples() as error:
            error.message()
            quit()

        try:
            if set(self.phase_types.keys()) != set(self.mic_gen_descriptors.keys()):
                # There are phases which not have descriptors or a phase type
                for phase in self.mic_gen_descriptors:
                    if phase not in self.phase_types:
                        # If there is a phase that has descriptors but no phase type
                        raise errors.PhaseDescriptorsMatch(phase)
        except errors.PhaseDescriptorsMatch as error:
            error.message()
            quit()
        # try:
        #     for phase in phase_types:
        #         if not RepresentsInt(phase) or not isinstance(phase, str):
        #             raise errors.UnexpectedValue(phase, 'key of phase_types',
        #                                          'string containing an integer')
        # except errors.UnexpectedValue as error:
        #     error.message()
        #     quit()

    def setOptionsSimulation(self, options):

        self.mic_gen_parameters = options["Mic_Gen_Parameters"]
        self.problem_type = options["Problem_Type"]
        self.n_dp_samples = options["N_DP_Samples"]
        self.mic_gen_descriptors = options["Mic_Gen_Descriptors"]
        self.phase_types = {
            phase_name: descriptors["Phase_Type"]
            for phase_name, descriptors in options["Mic_Gen_Descriptors"].items()
        }
        self.mesh_options = options.get("Mesh_Options", [])
        self.mesh_ext = [ext for ext in options.get("Mesh_Options", [])]

        self.parametersChecks()
