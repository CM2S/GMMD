"""
This module contains the class for errors.

Mostly unused at this point.
"""


class Error(Exception):
    """Base class for exceptions in this module."""


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        """Initizalize InputError instance."""
        self.expression = expression
        self.message = message
        super().__init__()


class ParameterMissing(Error):
    """Raised when a parameter is missing."""

    def __init__(self, missing_parameter, phase):
        """Initizalize ParameterMissing instance."""
        self.missing_parameter = missing_parameter
        self.phase = phase
        super().__init__()

    def message(self):
        """Print message."""
        print(
            "The parameter {0} of phase {1} is missing.".format(
                self.missing_parameter, self.phase
            )
        )


class UnacceptableParameters(Error):
    """Raised when the combination of parameters given are not acceptable."""

    def __init__(self, used_parameters, phase, acceptable_descriptions):
        """Initizalize UnacceptableParameters instance."""
        self.used_parameters = used_parameters
        self.phase = phase
        self.acceptable_descriptions = acceptable_descriptions
        super().__init__()

    def message(self):
        """Print message."""
        print(
            "The set of parameters {0} for phase {1} are not acceptable.".format(
                self.used_parameters, self.phase
            )
        )
        print("Acceptable configurations are {0}.".format(self.acceptable_descriptions))


class UnexpectedValue(Error):
    """Raised when a parameter has an unexpected value."""

    def __init__(self, value, name_var, acceptable_values):
        """Initizalize UnexpectedValue instance."""
        self.value = value
        self.name_var = name_var
        self.acceptable_values = acceptable_values
        super().__init__()

    def message(self):
        """Print message."""
        print("The value given for {0} is not valid.".format(self.name_var))
        print(
            "It must be {0}, but {1} was the value given.".format(
                self.acceptable_values, self.value
            )
        )


class PhaseDescriptorsMatch(Error):
    """Raised when there are phases with descriptors but no phase type."""

    def __init__(self, phase):
        """Initizalize PhaseDescriptorsMatch instance."""
        self.phase = phase
        super().__init__()

    def message(self):
        """Print message."""
        print("Phase {0} has descriptors but no phase type.".format(self.phase))


class InsufficientInfoMesh(Error):
    """Raised when the info about the mesh required is insufficient."""

    def __init__(self, info, necessary_info, disc_ext):
        """Initizalize InsufficientInfoMesh instance."""
        self.info = info
        self.necessary_info = necessary_info
        self.disc_ext = disc_ext
        super().__init__()

    def message(self):
        """Print message."""
        print(
            "The specifications for the mesh {0} given were insufficient: {1}".format(
                self.disc_ext, self.info
            )
        )
        print("The necessary specifications are: {0}".format(self.necessary_info))


class UnsupportedMesh(Error):
    """Raised when an unsupported mesh is required."""

    def __init__(self, disc_ext):
        """Initizalize UnsupportedMesh instance."""
        self.disc_ext = disc_ext
        super().__init__()

    def message(self):
        """Print message."""
        print("The mesh with extension {0} is not supported.".format(self.disc_ext))


class UnsupportedInitialConfigurationType(Error):
    """Raised when an unsupported initial configuration is specified."""

    def __init__(self, init_conf):
        """Initizalize UnsupportedInitialConfigurationType instance."""
        self.init_conf = init_conf
        super().__init__()

    def message(self):
        """Print message."""
        print(
            "The initial configuration type {0} is not supported.".format(
                self.init_conf
            )
        )


class UnsupportedDistribution(Error):
    """Raised when an unsupported distribution is required."""

    def __init__(self, distibution, parameter, phase):
        """Initizalize UnsupportedDistribution instance."""
        self.parameter = parameter
        self.phase = phase
        self.distibution = distibution
        super().__init__()

    def message(self):
        """Print message."""
        print(
            "The distribution {0} for {1} in phase {2} is not supported.".format(
                self.distibution, self.parameter, self.phase
            )
        )


class IncompatibleDimension(Error):
    """Raised when two variables have incompatible dimensions."""

    def __init__(self, *name_vars):
        """Initizalize IncompatibleDimension instance."""
        self.name_vars = name_vars
        super().__init__()

    def message(self):
        """Print message."""
        print("The variables {} have incompatible dimensions.".format(self.name_vars))


class DangerousValueNormal(Error):
    """
    Error raised when dangerous parameters are chosen in a Normal distriburion.

    Raised when a chosen normal distribution for a geometrical size paramter generates
    unacceptable parameters with a probability that is too high.
    """

    def __init__(self, parameter, phase, tail):
        """Initizalize DangerousValueNormal instance."""
        self.parameter = parameter
        self.phase = phase
        self.tail = tail
        super().__init__()

    def message(self):
        """Print message."""
        if self.tail == "low":
            print(
                """The chosen parameters for the normal distribution of the {1} parameter
in phase {0} produce smaller values than 0 with probability greater
than 2.5%.""".format(
                    self.phase, self.parameter
                )
            )
        elif self.tail == "high":
            print(
                """The chosen parameters for the normal distribution of the {1} parameter
in phase {0} produce larger values than the smallest dimension of the
RVE with probability greater
than 2.5%.""".format(
                    self.phase, self.parameter
                )
            )


class UnableToGenerateSample(Error):
    """
    Raise when unable to generate sample.

    Raised when a sample of geometrical size parameters could not be generate with the
    supplied statistical parameters.
    """

    def __init__(self, parameter, phase, max_sample):
        """Initizalize UnableToGenerateSample instance."""
        self.parameter = parameter
        self.phase = phase
        self.max_sample = max_sample
        super().__init__()

    def message(self):
        """Print message."""
        print(
            """From the {0} samples generated for {1} in phase {2} none was acceptable.
 Try changing the statistical parameters defining its statistical
 distribution.""".format(
                self.max_sample, self.parameter, self.phase
            )
        )


class VolumeFractionLargerOne(Error):
    """Raise when the global volume fraction is larger than 1."""

    def __init__(self, phase):
        """Initizalize VolumeFractionLargerOne instance."""
        self.phase = phase
        super().__init__()

    def message(self):
        """Print message."""
        print(
            """While adding particles to phase {0} the global volume fraction went
over 1.""".format(
                self.phase
            )
        )


class NoMesh(Error):
    """Raised when no mesh was specified."""

    def message(self):
        """Print message."""
        print("No mesh was specified.")


class MissingInfoExtension(Error):
    """Raised when no specifications were given for some mesh."""

    def __init__(self, ext):
        """Initizalize MissingInfoExtension instance."""
        self.ext = ext
        super().__init__()

    def message(self):
        """Print message."""
        print("The mesh {0} has no specifications.".format(self.ext))


class IncompatibleDimensionsRVEphase(Error):
    """Raised when the RVE dimension is incompatible with the phase."""

    def __init__(self, phase_type_name, dim_phase, rve_dim, phase):
        """Initizalize IncompatibleDimensionsRVEphase instance."""
        self.phase_type_name = phase_type_name
        self.dim_phase = dim_phase
        self.rve_dim = rve_dim
        self.phase = phase
        super().__init__()

    def message(self):
        """Print message."""
        print(
            """The phase type '{0}'' of phase {3} requires a {1}D RVE, but the RVE given
is {2}D.""".format(
                self.phase_type_name, self.dim_phase, self.rve_dim, self.phase
            )
        )


class OnlyCylindricalFibers(Error):
    """Raised when cylindrical fibers are specified with other particles."""

    def message(self):
        """Print message."""
        print(
            "Cylindrical fibers are not supported with other phase types in the same RVE."
        )


class UnsupportedPhaseType(Error):
    """Raised when a required phase type is not supported."""

    def __init__(self, phase_type, phase):
        """Initizalize UnsupportedPhaseType instance."""
        self.phase_type = phase_type
        self.phase = phase
        super().__init__()

    def message(self):
        """Print message."""
        print(
            """The phase type {0} specified for phase {1}
                 is not supproted""".format(
                self.phase_type, self.phase
            )
        )
