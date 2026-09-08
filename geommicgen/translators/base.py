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

EXPLANATIONS = {}
# Reasons why a format that a reader might expect is deliberately not written

_LOADERS = []
# Callables that register further writers the first time one is looked up


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

    needs_grid: bool
        Whether the writer needs the grid of a structured mesh.

    requires_periodic: bool
        Whether the writer refuses a mesh whose opposite faces are not discretised
        alike, because the solver it writes for would refuse it in turn.
    """

    name = None
    extension = None
    needs_cells = True
    needs_grid = False
    requires_periodic = False

    @classmethod
    def from_options(cls, options):
        """
        Build the writer from the options a deck or a command line gave.

        A writer that takes no options is built from any options at all, which is why
        this is not abstract: only the writers that have something to configure say so.

        Parameters
        ----------
        options: dict
            Options given for the discretisation, keyed by the name of the keyword.

        Returns
        -------
        `.SolverWriter`
            The writer.
        """
        return cls()

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

        Raises
        ------
        ValueError:
            If the mesh does not carry what the format needs.

        PeriodicityError:
            If the format needs a periodic mesh and the faces do not match.
        """
        if self.needs_grid and mesh.structured is None:
            raise ValueError(
                "The {0} format is written from the grid of a structured mesh; mesh "
                "the microstructure with a structured mesher to obtain one.".format(
                    self.name
                )
            )
        if self.requires_periodic:
            mesh.check_periodic_conformity()
        # Declaring what a format needs is what keeps every writer from having to
        # remember the same checks

        return self._write(mesh, file_path)

    @abc.abstractmethod
    def _write(self, mesh, file_path):
        """
        Write a mesh once it is known to carry what the format needs.

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


def register_loader(loader):
    """
    Register a callable that adds further writers the first time one is looked up.

    This keeps the cost of the formats that need a heavy import out of the import of
    this package, which the writers that need nothing beyond numpy should not pay.

    Parameters
    ----------
    loader: callable
        Callable that registers writers when it is invoked.
    """
    _LOADERS.append(loader)


def _run_loaders():
    """Invoke every loader that has not run yet."""
    while _LOADERS:
        _LOADERS.pop()()


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
        _run_loaders()
    if name not in WRITERS:
        if name in EXPLANATIONS:
            raise ValueError(
                "The format {0} is deliberately not written because {1}.".format(
                    name, EXPLANATIONS[name]
                )
            )
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
    _run_loaders()

    return sorted(WRITERS)
