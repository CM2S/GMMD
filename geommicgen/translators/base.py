"""
Module containing the interface of the solver writers.

A solver writer takes a `.Mesh` and writes it in the format one solver reads. The
writers are kept in a registry so that a new one becomes available to the command line
and to the input file by being registered, without any other part of the package having
to know about it.
"""

import abc

WRITERS = {}
# Correspondence between the name of a format and the class that writes it


class SolverWriter(abc.ABC):
    """
    Class for the writers of solver specific mesh formats.

    Class Attributes
    ----------------
    name: str
        Name by which the format is requested.

    extension: str
        Extension of the file the writer produces.

    needs_cells: bool
        Whether the writer needs the explicit cells of the mesh. A writer that only
        reads the grid of a structured mesh sets this to False and is therefore usable
        with grids that are too large to be expressed as cells.
    """

    name = None
    extension = None
    needs_cells = True

    @abc.abstractmethod
    def write(self, mesh, file_path):
        """
        Write a mesh in the format of the solver.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        file_path: str
            Path of the file to be written.

        Returns
        -------
        list
            Paths of the files that were written.
        """


def register_writer(writer_class):
    """
    Register a solver writer, so that it can be requested by name.

    Parameters
    ----------
    writer_class: class
        Class of the writer, deriving from `.SolverWriter`.

    Returns
    -------
    class
        The class that was registered, so that this can be used as a decorator.
    """
    WRITERS[writer_class.name] = writer_class

    return writer_class


def get_writer(name):
    """
    Get the class of a solver writer from its name.

    Parameters
    ----------
    name: str
        Name of the format.

    Returns
    -------
    class
        Class of the writer.

    Raises
    ------
    ValueError:
        If no writer is registered under that name.
    """
    if name not in WRITERS:
        raise ValueError(
            "There is no writer for the format {0}. The available formats are "
            "{1}.".format(name, ", ".join(available_writers()))
        )

    return WRITERS[name]


def available_writers():
    """
    Get the names of every registered writer.

    Returns
    -------
    list
        Sorted names of the formats that can be written.
    """
    return sorted(WRITERS)
