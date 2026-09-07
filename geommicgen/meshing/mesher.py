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

MESHERS = {}
# Correspondence between the name of a mesher and the class that implements it


class Mesher(abc.ABC):
    """
    Class for the meshers of microstructures.

    Class Attributes
    ----------------
    name: str
        Name by which the mesher is requested.
    """

    name = None

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
    return sorted(MESHERS)
