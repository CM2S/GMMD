"""
Module containing the writer of every mesh format meshio supports.

This is what the separation between the mesh and the solver formats buys: any format
meshio can write becomes available without a writer of its own. XDMF serves FEniCS,
Exodus serves MOOSE, MED serves Code_Aster, and the VTK formats serve the viewers.

Not every format carries everything. The phase of each cell is written as cell data
wherever the format supports it, and is silently lost where it does not. The formats
that drop cell data altogether are refused rather than written incorrectly, and Abaqus
is redirected to the writer in this package, because the one in meshio names the
elements after rigid and shell types that no continuous RVE can use.
"""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import SolverWriter, register_writer

REFUSED_FORMATS = {
    "abaqus": (
        "the meshio writer for Abaqus names the elements after rigid and shell types "
        "and cannot write the periodic constraints; use the abaqus writer of this "
        "package instead"
    ),
    "ansys": "the meshio writer for ANSYS does not carry the phase of the cells",
    "permas": "the meshio writer for PERMAS does not carry the phase of the cells",
    "dolfin-xml": "the format is legacy; write xdmf instead",
}
# Formats that would silently lose the phase, or write elements that cannot be used

RECOMMENDED_FORMATS = ("vtu", "vtk", "xdmf", "gmsh", "med", "exodus")
# Formats that carry the phase of every cell and are worth advertising


class MeshioWriter(SolverWriter):
    """
    Class for the writers of the mesh formats meshio supports.

    Attributes
    ----------
    file_format: str
        Name of the format, as meshio knows it.
    """

    needs_cells = True

    def __init__(self, file_format):
        """Initizalizer for the MeshioWriter Class."""
        self.file_format = file_format
        self.name = file_format

    def write(self, mesh, file_path):
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
        ValueError:
            If the format would lose the phase of the cells or write unusable elements.
        """
        import meshio

        if self.file_format in REFUSED_FORMATS:
            raise ValueError(
                "The format {0} is not written by geommicgen because {1}.".format(
                    self.file_format, REFUSED_FORMATS[self.file_format]
                )
            )
        meshio.write(file_path, mesh.to_meshio(), file_format=self.file_format)

        return [file_path]


def register_meshio_writers():
    """
    Register a writer for every format meshio can write.

    Returns
    -------
    list
        Names of the formats that were registered.
    """
    import meshio

    try:
        formats = sorted(meshio._helpers._writer_map)
    except AttributeError:
        formats = list(RECOMMENDED_FORMATS)
    # The private map is the only complete list; the recommended formats are used when
    # a future version of meshio renames it

    registered = []
    for i_format in formats:
        if i_format in REFUSED_FORMATS:
            continue

        writer_class = type(
            "Meshio{0}Writer".format(i_format.title().replace("-", "")),
            (MeshioWriter,),
            {
                "name": i_format,
                "extension": "." + i_format,
                "__init__": _make_initializer(i_format),
                "__doc__": "Class for the writer of the {0} format.".format(i_format),
            },
        )
        register_writer(writer_class)
        registered.append(i_format)

    return registered


def _make_initializer(file_format):
    """Build the initializer of the writer of one format."""

    def __init__(self):
        """Initizalize the writer."""
        MeshioWriter.__init__(self, file_format)

    return __init__
