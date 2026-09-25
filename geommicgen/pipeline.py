"""
Module containing the pipeline that joins the meshers to the writers.

A `.MeshJob` holds the mesher that produces a mesh and the writers the mesh is then
handed to, and runs them together, so that this is the one place a mesh passes from
the second stage to the third. The meshers and the writers know nothing of each other,
and nothing of the input file: an input data file asks for a discretisation by naming
the mesher that produces it, gmsh or voxel, and `build_mesh_jobs` turns that request
into the jobs; the command line builds the same jobs from its arguments. A mesher that
is registered can be named by either without this module knowing it.

Everything the input file can get wrong is settled while it is being read: the mesher
and every writer are resolved before the first sample is generated, so a name that does
not exist is refused straight away rather than after a mesh has been built.
"""

import os
import time
import traceback

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import get_mesher
from geommicgen.meshing.writers import standard_mesh_path, write_standard_mesh
from geommicgen.translators.base import get_writer

MESH_DIRECTORY = "meshes"
# Directory of a sample the meshes are written into


class MeshJob:
    """
    Class for one discretisation an input data file asks for.

    A job owns the mesher that produces the mesh and the writers the mesh is handed to.
    It runs them together and keeps the outcome, so that one discretisation failing does
    not stop the others: the program reports at the end which of them could not be
    produced, and ends unsuccessfully.

    Attributes
    ----------
    mesher: `.Mesher`
        Mesher that produces the mesh.

    writers: list
        Solver writers the mesh is handed to, beyond the standard output.

    base_name: str
        Name of the files that are written, without any extension.

    description: str
        Name the job is reported under in the summary of execution times.

    time: float
        Seconds the job took, or None if it has not run.

    files: list
        Paths of the files that were written.

    error: Exception
        Error that stopped the job, or None if it succeeded or has not run.

    trace: str
        Traceback of that error, as text.
    """

    def __init__(self, mesher, writers, base_name):
        """
        Initizalizer for the MeshJob Class.

        Parameters
        ----------
        mesher: `.Mesher`
            Mesher that produces the mesh.

        writers: list
            Solver writers the mesh is handed to.

        base_name: str
            Name of the files that are written, without any extension.
        """
        self.mesher = mesher
        self.writers = list(writers)
        self.base_name = base_name
        self.description = mesher.description
        self.time = None
        self.files = []
        self.error = None
        self.trace = None

    def run(self, microstructure, result_dir, report=None):
        """
        Mesh a microstructure and write it in every format the job asks for.

        Whether the job succeeded is told by its *error* afterwards.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        result_dir: str
            Directory the files are written into. The caller chooses it, so that a run
            of the program can put them under the sample and the command line can put
            them where it was told to.

        report: callable
            Called with the index of the particle that has been dealt with and the
            total number of particles.
        """
        start = time.time()
        self.files = []
        os.makedirs(result_dir, exist_ok=True)
        base_path = os.path.join(result_dir, self.base_name)
        try:
            mesh = self.mesher.mesh(microstructure, report=report)
            if not isinstance(mesh, Mesh):
                raise TypeError(
                    "The mesher {0} returned {1} rather than a Mesh, which is what "
                    "the writers take.".format(self.mesher.name, type(mesh).__name__)
                )
            self.write(mesh, base_path)
        except Exception as error:  # pylint: disable=broad-except
            self.trace = traceback.format_exc()
            self.error = error.with_traceback(None)
            # The error is kept rather than raised, so that the discretisations asked
            # for beside this one still get their chance. The traceback is kept as text
            # and taken off the exception, because a live one holds every frame between
            # here and the raise, and through them the whole mesh, for as long as the
            # job is kept -- and the failed jobs are kept until the batch ends
        finally:
            self.time = time.time() - start

    def summary(self):
        """
        Say in one line what became of the discretisation.

        Returns
        -------
        str
            The name of the files, what discretised them, and the outcome.
        """
        if self.time is None:
            outcome = "not run"
        elif self.error is None:
            outcome = "ok"
        else:
            outcome = "failed: {0}: {1}".format(
                type(self.error).__name__, self.error
            )
        # A job that has not run has no error either, so the absence of one is not on
        # its own the same as having succeeded

        return "{0} ({1}): {2}".format(self.base_name, self.description, outcome)

    def write(self, mesh, base_path):
        """
        Write a mesh in the standard format and with every writer the job asks for.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        base_path: str
            Path of the files to be written, without any extension.

        Every file is recorded on the job as soon as it is written, so that a format
        failing part of the way through still reports what did reach the disk.

        Raises
        ------
        ValueError:
            If a writer would write over the standard output.
        """
        refuse_overwrites(
            base_path, self.writers, [standard_mesh_path(mesh, base_path)]
        )
        self.files += write_standard_mesh(mesh, base_path)
        # Every mesh is written in a standard format that a viewer reads, whichever
        # other formats were asked for. With its sidecar that file is the whole of the
        # second stage: it reads back as the same kind of mesh it was written from, so
        # the third stage runs off it alone, later or somewhere else

        write_formats(mesh, base_path, self.writers, written=self.files)


def job_base_name(deck_name, label):
    """
    Name the files of a job after the input data file and what discretises it.

    Parameters
    ----------
    deck_name: str
        Name of the input data file the microstructure was generated from.

    label: str
        What tells this discretisation apart from another of the same microstructure:
        the element for a mesh, the number of voxels for a grid.

    Returns
    -------
    str
        Name of the files, for a writer to put its own extension on.
    """
    if deck_name:
        return "{0}_{1}".format(os.path.splitext(deck_name)[0], label)

    return label
    # Naming the files after the deck and the discretisation keeps two runs of the same
    # microstructure apart, which neither does on its own


def refuse_overwrites(base_path, writers, protected):
    """
    Refuse a writer aimed at a file this run needs as it is.

    Parameters
    ----------
    base_path: str
        Path of the files to be written, without an extension.

    writers: list
        The writers.

    protected: list
        Paths no writer may be aimed at: the standard output of the mesh, or the file
        the mesh was read from.

    Raises
    ------
    ValueError:
        If a writer would write over one of them.
    """
    guarded = {os.path.abspath(i_path) for i_path in protected}
    over = [
        (i_writer.name, base_path + i_writer.extension)
        for i_writer in writers
        if os.path.abspath(base_path + i_writer.extension) in guarded
    ]
    if over:
        raise ValueError(
            "The format {0} writes {1}, which this run needs as it is. Ask for it "
            "under another name, or somewhere else.".format(
                ", ".join(i_name for i_name, _ in over),
                ", ".join(os.path.basename(i_path) for _, i_path in over),
            )
        )
    # Asked before anything is written, so that a request that cannot be honoured
    # leaves nothing behind rather than half of it


def write_formats(mesh, base_path, writers, protected=(), written=None):
    """
    Write a mesh with every writer.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh to be written.

    base_path: str
        Path of the files, without an extension.

    writers: list
        The writers.

    protected: list
        Paths no writer may be aimed at.

    written: list
        List the paths are appended to as they are written, so that a format failing
        part of the way through still leaves the caller with what did reach the disk.

    Returns
    -------
    list
        Paths of the files that were written.
    """
    refuse_overwrites(base_path, writers, protected)
    written = [] if written is None else written
    for i_writer in writers:
        written += i_writer.write(mesh, base_path + i_writer.extension)

    return written


def writers_from_options(options, defaults):
    """
    Resolve the writers a discretisation is to be written with.

    Parameters
    ----------
    options: dict
        Options given for the discretisation in the input data file.

    defaults: tuple
        Formats used when the keyword is absent. An empty list is not absent: it asks
        for the standard output and nothing else, which is the second stage on its own.

    Returns
    -------
    list
        The writers, each built from the options the discretisation was given.

    Raises
    ------
    ValueError:
        If a format is named that there is no writer for, or if an option a writer does
        understand is not a value it accepts.
    """
    names = options.get("formats", None)
    names = list(defaults) if names is None else list(names)

    return [get_writer(i_name).from_options(options) for i_name in names]
    # Each writer takes from the options what it understands, so a keyword meant for
    # one format costs the others nothing


def refuse_repeated_files(jobs):
    """
    Refuse discretisations that would be written over each other.

    Two discretisations can share a name and come to no harm: a mesh and a grid of the
    same microstructure are each written in formats of their own, so File_Name my_mesh
    under both gives my_mesh.vtu beside my_mesh.vti. What cannot be honoured is one
    name given to two discretisations that write a format in common.

    Parameters
    ----------
    jobs: list
        The `.MeshJob` objects built.

    Raises
    ------
    ValueError:
        If two of them would write the same file.
    """
    written = {}
    for i_job in jobs:
        for j_writer in i_job.writers:
            path = i_job.base_name + j_writer.extension
            if path in written:
                raise ValueError(
                    "{0} and {1} would both be written as {2}, so one would be written "
                    "over the other. Give each of them a File_Name of its own, or "
                    "neither of them one.".format(
                        written[path], i_job.description.lower(), path
                    )
                )
            written[path] = i_job.description
    # The standard output of a job is not among these: whether it is written as a mesh
    # or as an image is known only once the mesh exists, and `MeshJob.write` guards it
    # there, where the two kinds of the same name do not collide anyway


def build_mesh_jobs(mesh_options, deck_name=None):
    """
    Build the meshing jobs an input data file asks for.

    Parameters
    ----------
    mesh_options: dict
        Mesh options read from the input data file, keyed by the name of the mesher.

    deck_name: str
        Name of the input data file, used to name the files of a job.

    Returns
    -------
    list
        The `.MeshJob` objects to run.

    Raises
    ------
    ValueError:
        If a discretisation is asked for that there is no mesher for, if a format is
        named that there is no writer for, or if one name is given for several files.
    """
    jobs = []
    for i_name, i_options in mesh_options.items():
        mesher_class = get_mesher(i_name)
        meshers = mesher_class.from_options(i_options)
        writers = writers_from_options(i_options, mesher_class.default_formats)
        # The mesher takes from the options what it declares, and so does each writer,
        # so nothing here knows what either is built from

        file_name = i_options.get("file_name")
        if file_name and len(meshers) > 1:
            raise ValueError(
                "File_Name {0} would be given to each of the {1} discretisations asked "
                "for under {2} ({3}), so each would be written over the one before. "
                "Remove it and they are named {4} instead.".format(
                    file_name,
                    len(meshers),
                    i_name,
                    ", ".join(j_mesher.label for j_mesher in meshers),
                    ", ".join(
                        job_base_name(deck_name, j_mesher.label) for j_mesher in meshers
                    ),
                )
            )
        # Refused rather than resolved, because the run would otherwise end with the
        # last one alone

        for j_mesher in meshers:
            jobs.append(
                MeshJob(
                    j_mesher,
                    writers,
                    file_name or job_base_name(deck_name, j_mesher.label),
                )
            )

    refuse_repeated_files(jobs)
    # The name of one discretisation is only known to collide with the name of another
    # once all of them are built, so this is asked of the whole set

    return jobs
