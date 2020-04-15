"""
This module contains the class of errors.
"""


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class ParameterMissing(Error):
    """Raised when an operation attempts a state transition that's not
    allowed.

    Attributes:
        previous -- state at beginning of transition
        next -- attempted new state
        message -- explanation of why the specific transition is not allowed
    """

    def __init__(self, missing_parameter, phase):
        self.missing_parameter = missing_parameter
        self.phase = phase

    def message(self):
        print("The parameter {0} of phase {1} is missing.".format(self.missing_parameter,
              self.phase))


class UnacceptableParameters(Error):
    """Raised when the combination of parameters given are not acceptable."""

    def __init__(self, used_parameters, phase, acceptable_descriptions):
        self.used_parameters = used_parameters
        self.phase = phase
        self.acceptable_descriptions = acceptable_descriptions

    def message(self):
        print("The set of parameters {0} for phase {1} are not acceptable.".format(
            self.used_parameters, self.phase))
        print("Acceptable configurations are {0}.".format(self.acceptable_descriptions))
