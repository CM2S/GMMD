"""
Module containing the interface of the meshers.

A mesher takes a `.Microstructure` and returns the `.Mesh` that discretises it. The
meshers are kept in a registry so that a new one becomes available to the command line
and to the input file by being registered, without any other part of the package having
to know about it.

A mesher never writes a file. Writing is the concern of `geommicgen.meshing.writers`,
for the standard formats, and of `geommicgen.translators`, for the formats that solvers
read.

Adding a mesher
---------------
Subclass `.Mesher`, decorate it with `register_mesher`, and add its module to
`BUILTIN_MESHER_MODULES`. It needs:

- ``name``, the word a deck and ``geommicgen-mesh --mesher`` ask for it by;
  ``description``, the line it is reported under; ``default_formats``, the formats a
  mesh is written in when none is named.
- ``options``, one entry per parameter of its initializer, keyed by the deck keyword
  and describing its ``type`` and ``help``. The deck reader and the command line are
  built from it. `from_options` then builds the mesher without more; override it only
  when one set of options means several meshers.
- ``label`` set by the initializer: what tells this discretisation apart from another
  of the same microstructure, since the files are named after it.
- ``warnings``, a list the initializer binds when the mesher has anything to report;
  the empty tuple of the base class serves one that never does.
- `mesh`, returning a `.Mesh`. Whatever the tool underneath produces, the mesh handed
  back holds: ``points`` of shape *(n_nodes, 3)*, three coordinates even in two
  dimensions; ``cells`` as *(type, connectivity)* blocks with the type named as meshio
  names it and the nodes in VTK order, since every writer reorders from that;
  ``phase`` with one integer array per block whose values are keys of
  ``phase_names``; ``matrix_phase``; ``periodic`` True when the mesher meant the mesh
  to be periodic, which `Mesh.check_periodic_conformity` then verifies; and
  ``source`` naming the mesher and what set its resolution. A structured mesher
  gives ``structured`` in place of the points and cells, and the mesh builds them on
  demand. `geommicgen.tests.test_pipeline` runs every registered mesher against these
  requirements.
- Nothing printed: progress goes through ``report``, messages through ``warnings``.
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

    options: dict
        Options the mesher is built with, keyed by the name of the keyword an input
        data file gives them under. Each is described by its *type*, one of the types
        an input data file reads, and a *help* line for the command line. The name in
        lower case is the parameter of the initializer the option is passed as, which
        is how `from_options` builds the mesher without the mesher saying more.

    default_formats: tuple
        Names of the formats the mesh is written in when neither an input data file
        nor a command line names any.

    label: str
        What tells a discretisation by this mesher apart from another of the same
        microstructure, such as the element or the number of voxels. Set by the
        initializer, and used to name the files.
    """

    name = None
    description = None
    warnings = ()
    options = {}
    default_formats = ()
    label = None

    @classmethod
    def from_options(cls, options):
        """
        Build the meshers the options ask for.

        Each option the mesher declares that the options hold is passed to the
        initializer under its own name, so a mesher whose options are its parameters
        needs nothing beyond declaring them. A mesher that is asked for several
        discretisations at once, such as a grid at several resolutions, overrides this
        and returns one mesher for each.

        Parameters
        ----------
        options: dict
            Options given for the discretisation, keyed by the name of the keyword in
            lower case, the way an input data file and the command line both key them.

        Returns
        -------
        list
            The meshers, one for each discretisation to be produced.
        """
        kwargs = {}
        for i_name in cls.options:
            value = options.get(i_name.lower())
            if value is not None:
                kwargs[i_name.lower()] = value
        # An option that was not given is left to the default of the initializer

        return [cls(**kwargs)]

    # TODO: multi-resolution meshing. A mesher is built for one resolution and one
    # element, and `mesh` gives one mesh; a deck that wants a microstructure at
    # several sizes or with several elements has to name the mesher once per one,
    # and gmsh then rebuilds the CAD model, fragmenting every particle and its images
    # into the box, for each. Measured on 60 spheres: the model is 4.5 s of a 4.5 s
    # mesh, re-meshing it at another size 0.5 s, and setOrder(2) on the linear mesh
    # 0.2 s. So the shape to move to is one mesher holding lists of sizes and
    # elements whose `mesh` builds the model once and yields one `.Mesh` per
    # combination, each with its own label; `MeshJob`, `run_mesh_jobs` and the mesh
    # command then take several meshes from one job. The voxel mesher, which already
    # takes a list of resolutions and returns one mesher each, would follow the same
    # contract. The gmsh signature fixtures guard that setOrder gives the same
    # second order mesh as generating at order two
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


def mesher_options():
    """
    Get every option a registered mesher declares.

    Returns
    -------
    dict
        Dictionary of the form *{option_name: description}*, over every mesher.

    Raises
    ------
    ValueError:
        If two meshers describe an option of the same name differently.
    """
    load_builtin_meshers()
    options = {}
    for i_mesher in MESHERS.values():
        for j_name, j_description in i_mesher.options.items():
            if options.get(j_name, j_description) != j_description:
                raise ValueError(
                    "The option {0} is described differently by two meshers.".format(
                        j_name
                    )
                )
            options[j_name] = j_description

    return options
    # An option shared by two meshers has to mean the same to both, since the command
    # line offers it once
