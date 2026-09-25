"""
Module containing the writers of the mesh formats meshio supports.

This is what the separation between the mesh and the solver formats buys: a format
meshio can write becomes available without a writer of its own. XDMF serves FEniCS, and
the VTK formats and gmsh's serve the viewers and the other meshers.

Only the formats that carry the phase of every cell are offered. meshio can write about
thirty, but many of them are surface formats, or drop the cell data without saying so,
and a mesh of a microstructure written in one of those has quietly lost the only thing
that distinguishes its phases. Offering a format is therefore a deliberate act, and the
ones a reader might expect to find carry a reason instead. A format a solver reads is
offered once that solver has been run on what is written of it, which is what the
solver tests do; carrying the phases is not enough on its own.
"""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import MissingOptionalDependency
from geommicgen.translators.base import (
    EXPLANATIONS,
    SolverWriter,
    register_loader,
    register_writer,
)

SUPPORTED_FORMATS = {
    "vtu": (".vtu", None, False),
    "vtk": (".vtk", None, False),
    "gmsh": (".msh", None, False),
    "xdmf": (".xdmf", "h5py", True),
}
# Formats that carry the phase of every cell, with the extension each one uses, the
# package meshio needs in order to write it when it needs one beyond its own, and
# whether the points are written with as many coordinates as the mesh has dimensions.
# The VTK family and gmsh always take three; the others take two for a two dimensional
# mesh, and a solver that reads them takes the number of coordinates as the dimension
# of the problem -- FEniCS builds a surface in space from a triangle mesh with three

UNSUPPORTED_FORMATS = {
    "ansys": "the meshio writer does not carry the phase of the cells",
    "permas": "the meshio writer does not carry the phase of the cells",
    "off": "the meshio writer does not carry the phase of the cells",
    "svg": "the format is a drawing and carries no cell data",
    "obj": "the format holds surfaces only and cannot describe a volume mesh",
    "ply": "the format holds surfaces only and cannot describe a volume mesh",
    "stl": "the format holds surfaces only and cannot describe a volume mesh",
    "dolfin-xml": "the format is legacy, and meshio itself recommends xdmf instead",
    "med": "no Code_Aster run has been made on what meshio writes of it, and the "
    "periodicity of the cell would be left to the solver's own setup",
    "exodus": "no MOOSE run has been made on what meshio writes of it, and the "
    "periodicity of the cell would be left to the solver's own setup",
}
# Formats a reader might expect, with the reason each one is not offered


class MeshioWriter(SolverWriter):
    """
    Class for the writers of the mesh formats meshio supports.

    Attributes
    ----------
    file_format: str
        Name of the format, as meshio knows it.

    requires_package: str
        Name of the package meshio needs in order to write the format, when it needs
        one beyond its own dependencies.

    flat_points: bool
        Whether the points are written with as many coordinates as the mesh has
        dimensions, rather than always three.
    """

    requires_package = None
    flat_points = False

    def __init__(self, file_format=None):
        """Initizalizer for the MeshioWriter Class."""
        self.file_format = file_format if file_format is not None else type(self).name

    def _write(self, mesh, file_path):
        """
        Write a mesh in one of the formats meshio supports.

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
        MissingOptionalDependency:
            If meshio needs a package that is not installed to write the format.
        """
        import meshio

        if self.requires_package is not None:
            try:
                __import__(self.requires_package)
            except ImportError:
                raise MissingOptionalDependency(
                    self.requires_package,
                    "pip install 'geommicgen[{0}]'".format(self.file_format),
                    reason="meshio needs it to write the {0} format and it is not "
                    "installed".format(self.file_format),
                ) from None
        # Checking first turns a bare import error raised inside meshio into a sentence
        # naming the package and the extra that installs it, which is named after the
        # format so that the command can be read off the message

        meshio_mesh = mesh.to_meshio()
        if self.flat_points:
            meshio_mesh.points = meshio_mesh.points[:, : mesh.dim]
        meshio.write(file_path, meshio_mesh, file_format=self.file_format)

        return [file_path]


def register_meshio_writers():
    """
    Register a writer for every supported meshio format.

    Returns
    -------
    list
        Names of the formats that were registered.
    """
    for i_format, (i_extension, i_package, i_flat) in sorted(
        SUPPORTED_FORMATS.items()
    ):
        register_writer(
            type(
                "Meshio{0}Writer".format(i_format.title().replace("-", "")),
                (MeshioWriter,),
                {
                    "name": i_format,
                    "extension": i_extension,
                    "requires_package": i_package,
                    "flat_points": i_flat,
                    "__doc__": "Class for the writer of the {0} format.".format(
                        i_format
                    ),
                },
            )
        )
    # One class per format, because the registry holds classes the caller instantiates
    # and the class attributes are what carry the name and the extension

    return sorted(SUPPORTED_FORMATS)


EXPLANATIONS.update(UNSUPPORTED_FORMATS)
register_loader(register_meshio_writers)
# Registering is deferred until a writer is actually looked up, so that importing this
# package does not import meshio
