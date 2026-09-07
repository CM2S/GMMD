"""
Module containing the writers of the mesh formats meshio supports.

This is what the separation between the mesh and the solver formats buys: a format
meshio can write becomes available without a writer of its own. XDMF serves FEniCS,
Exodus serves MOOSE, MED serves Code_Aster, and the VTK formats serve the viewers.

Only the formats that carry the phase of every cell are offered. meshio can write about
thirty, but many of them are surface formats, or drop the cell data without saying so,
and a mesh of a microstructure written in one of those has quietly lost the only thing
that distinguishes its phases. Offering a format is therefore a deliberate act, and the
ones a reader might expect to find carry a reason instead.
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
    "vtu": (".vtu", None),
    "vtk": (".vtk", None),
    "gmsh": (".msh", None),
    "xdmf": (".xdmf", "h5py"),
    "med": (".med", "h5py"),
    "exodus": (".e", "netCDF4"),
}
# Formats that carry the phase of every cell, with the extension each one uses and the
# package meshio needs in order to write it, when it needs one beyond its own

UNSUPPORTED_FORMATS = {
    "abaqus": (
        "the meshio writer names the elements after rigid and shell types and cannot "
        "write the periodic constraints an RVE needs"
    ),
    "ansys": "the meshio writer does not carry the phase of the cells",
    "permas": "the meshio writer does not carry the phase of the cells",
    "off": "the meshio writer does not carry the phase of the cells",
    "svg": "the format is a drawing and carries no cell data",
    "obj": "the format holds surfaces only and cannot describe a volume mesh",
    "ply": "the format holds surfaces only and cannot describe a volume mesh",
    "stl": "the format holds surfaces only and cannot describe a volume mesh",
    "dolfin-xml": "the format is legacy, and meshio itself recommends xdmf instead",
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
    """

    requires_package = None

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
                    "pip install {0}".format(self.requires_package),
                    reason="meshio needs it to write the {0} format and it is not "
                    "installed".format(self.file_format),
                ) from None
        # Checking first turns a bare import error raised inside meshio into a sentence
        # naming the package and the command that installs it

        meshio.write(file_path, mesh.to_meshio(), file_format=self.file_format)

        return [file_path]


def register_meshio_writers():
    """
    Register a writer for every supported meshio format.

    Returns
    -------
    list
        Names of the formats that were registered.
    """
    for i_format, (i_extension, i_package) in sorted(SUPPORTED_FORMATS.items()):
        register_writer(
            type(
                "Meshio{0}Writer".format(i_format.title().replace("-", "")),
                (MeshioWriter,),
                {
                    "name": i_format,
                    "extension": i_extension,
                    "requires_package": i_package,
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
