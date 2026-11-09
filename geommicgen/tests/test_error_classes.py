"""Test module for the errors of the package."""

import inspect
import pickle
import unittest

from geommicgen.errors import error_classes
from geommicgen.errors.error_classes import Error, MissingOptionalDependency


class TestPickling(unittest.TestCase):
    """Test class for the errors of the package, taken to another process and back."""

    def test_every_error_comes_back_as_it_was(self):
        errors = [
            i_class
            for _, i_class in inspect.getmembers(error_classes, inspect.isclass)
            if issubclass(i_class, Error)
        ]
        self.assertGreater(len(errors), 20)
        for i_class in errors:
            parameters = [
                j_parameter
                for j_parameter in list(
                    inspect.signature(i_class.__init__).parameters.values()
                )[1:]
                if j_parameter.default is inspect.Parameter.empty
                and j_parameter.kind is not inspect.Parameter.VAR_POSITIONAL
            ]
            error = i_class(*["x{0}".format(j) for j in range(len(parameters))])
            with self.subTest(error=i_class.__name__):
                restored = pickle.loads(pickle.dumps(error))
                self.assertIs(type(restored), i_class)
                self.assertEqual(str(restored), str(error))
                self.assertEqual(vars(restored), vars(error))
        # An error was unpickled by calling its class with its message, which it took
        # for what it builds its message from: the message came back inside a second
        # one, or the error could not be built at all

    def test_an_argument_given_by_name_is_kept(self):
        error = MissingOptionalDependency("gmsh", "pip install gmsh", reason="why")
        self.assertEqual(str(pickle.loads(pickle.dumps(error))), str(error))


if __name__ == "__main__":
    unittest.main()
