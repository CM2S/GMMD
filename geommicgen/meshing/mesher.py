"""
Module containing the interface of the meshers.

A mesher takes a `.Microstructure` and returns the `.Mesh` that discretises it. The
meshers are kept in a registry so that a new one becomes available to the command line
and to the input file by being registered, without any other part of the package having
to know about it.

A mesher never writes a file. Writing is the concern of `geommicgen.meshing.writers`,
for the standard formats, and of `geommicgen.translators`, for the formats that solvers
read.
"""

import abc
import importlib

MESHERS = {}
# Correspondence between the name of a mesher and the class that implements it

BUILTIN_MESHER_MODULES = (
    "geommicgen.meshing.gmsh_mesher",
    "geommicgen.meshing.voxel_mesher",
)
# Modules holding the meshers that ship with the package. Importing one is what
# registers the mesher in it, and that is done the first time a mesher is looked up
# rather than when this module is imported, so that a caller that only wants to read a
# mesh does not pay for the geometry kernel


def load_builtin_meshers():
    """Import the meshers that ship with the package, so that they register."""
    for i_module in BUILTIN_MESHER_MODULES:
        importlib.import_module(i_module)


class Mesher(abc.ABC):
    """
    Class for the meshers of microstructures.

    Class Attributes
    ----------------
    name: str
        Name by which the mesher is requested.

    description: str
        Name the mesher is reported under in the summary of a run.

    warnings: list
        Messages about the run, for the caller to report. A mesher that collects them
        rebinds this in its initializer.
    """

    name = None
    description = None
    warnings = ()

    @abc.abstractmethod
    def mesh(self, microstructure, report=None):
        """
        Mesh a microstructure.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        report: callable
            Called with the index of the particle that has been dealt with and the
            total number of particles, so that a caller can report the progress of a
            long run. Meshing is a library call, so a mesher prints nothing itself.

        Returns
        -------
        `.Mesh`
            The mesh discretising the microstructure.
        """


def register_mesher(mesher_class):
    """
    Register a mesher, so that it can be requested by name.

    Parameters
    ----------
    mesher_class: class
        Class of the mesher, deriving from `.Mesher`.

    Returns
    -------
    class
        The class that was registered, so that this can be used as a decorator.
    """
    MESHERS[mesher_class.name] = mesher_class

    return mesher_class


def get_mesher(name):
    """
    Get the class of a mesher from its name.

    Parameters
    ----------
    name: str
        Name of the mesher.

    Returns
    -------
    class
        Class of the mesher.

    Raises
    ------
    ValueError:
        If no mesher is registered under that name.
    """
    if name not in MESHERS:
        load_builtin_meshers()
    if name not in MESHERS:
        raise ValueError(
            "There is no mesher named {0}. The available meshers are {1}.".format(
                name, ", ".join(available_meshers())
            )
        )

    return MESHERS[name]


def available_meshers():
    """
    Get the names of every registered mesher.

    Returns
    -------
    list
        Sorted names of the meshers.
    """
    load_builtin_meshers()

    return sorted(MESHERS)
