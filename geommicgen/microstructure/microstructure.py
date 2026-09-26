"""
This module contains the Microstructure class.

Each instance of the Microstructure class is a microstructure sample, composed of instances
of the Phase class, in turn described by the adequate phase descriptors.
"""

import copy

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.micgenmethod.speed_up_schemes import CellList
from geommicgen.microstructure.phase import Phase


def unit_scale(lengths):
    """
    Give the factor that brings the shortest of some lengths to one.

    The stages that work with tolerances, gmsh's and the particles' own among them,
    work on the microstructure multiplied by it, so that a tolerance means the same
    thing whatever the units of the RVE are, and the same microstructure written in
    other units is treated alike. When the shortest length is a power of two the
    factor is one as well, and it is multiplied out exactly, so a microstructure in
    units a power of two apart is treated alike to the last bit; otherwise what comes
    back is rounded in its last digit.

    Parameters
    ----------
    lengths: list(float)
        Positive lengths, the dimensions of an RVE.

    Returns
    -------
    float
        The factor, 1 for a unit RVE.
    """
    return 1 / min(lengths)
    # The shortest side itself, and not the power of two nearest it, which kept the
    # mantissa of the side: the simulation, whose time step and constants are not
    # scaled with the box, then ran a deck in millimetres and the same deck in
    # micrometres in boxes of sides 1.95 and 1.05, and gave two microstructures


class Microstructure:
    """
    Class for the Microstructure.

    Attributes
    ----------
    rve_dims: array
        Array containing the dimensions of the microstructure in each spatial direction.

    matrix_phase: str
        Name of the matrix phase.

    volume: float
        Volume/area of the microstructure.

    phases: dict
        Dictionary whose keys are the name of the phases and whose values are the
        corresponding instance of `.Phase`.

    total_overlap: float
        Measure of the overlap. Zero when there is no overlap.
    """

    def __init__(self, rve_dims):
        """Initizalizer for the Microstructure Class.

        Parameters
        ----------
        rve_dims: array
            Array containing the dimnesions of the microstructure in each spatial direction.
        """
        self.matrix_phase = None
        self.rve_dims = rve_dims
        self.dim = len(rve_dims)
        if self.dim != 2 and self.dim != 3:
            # Only 2D and 3D microstructures allowed
            raise ValueError("Only 2D and 3D microstructures are supproted.")
        if any([rve_dim <= 0 for rve_dim in self.rve_dims]):
            # The dimnesion of the microstrucutre must be positive
            raise ValueError(
                "The dimensions of the microstructure must be positive values."
            )
        self.phases = {}
        self.total_overlap = None

    @property
    def volume(self):
        """Volume/area of the microstructure."""
        return np.prod(self.rve_dims)
        # Worked out from the dimensions rather than kept beside them, so that a copy
        # given other dimensions cannot carry the volume of the one it was copied from

    def scaled(self, factor):
        """
        Give a copy of the microstructure with every length multiplied by a factor.

        The copy is the same microstructure in other units: its dimensions, the
        positions and the sizes of its particles are multiplied, and nothing else is.
        The descriptors of its phases are not, so particles are not to be generated
        for it.

        Parameters
        ----------
        factor: float
            Factor every length is multiplied by.

        Returns
        -------
        `.Microstructure`
            The copy.
        """
        scaled = copy.deepcopy(self)
        scaled.rve_dims = [i_dim * factor for i_dim in self.rve_dims]
        for i_particle in scaled.particles:
            i_particle.rescale(factor)

        # TODO: the overlap is left as it is: a length, or an area or a volume,
        # depending on how the simulation measured it, which the microstructure does
        # not know

        return scaled

    @classmethod
    def from_descriptors(cls, rve_dims, descriptors):
        """
        Build the microstructure an input data file describes, with its phases.

        Parameters
        ----------
        rve_dims: array
            Dimensions of the microstructure in each spatial direction.

        descriptors: dict
            Descriptors of each phase, keyed by the name of the phase, as the input data
            file gives them.

        Returns
        -------
        `.Microstructure`
            The microstructure, with no particles yet.

        Raises
        ------
        ValueError:
            If no phase is a matrix, or if two are.
        """
        microstructure = cls(rve_dims)
        for i_name, i_descriptors in descriptors.items():
            microstructure.add_phase(Phase(i_name, i_descriptors))
        if microstructure.matrix_phase is None:
            raise ValueError("No matrix phase was specified.")
        # Refused here, where the phases are declared, rather than by whichever
        # mesher first fills the RVE and finds nothing to fill it with

        return microstructure

    def add_phase(self, phase):
        """
        Add a phase to the microstructure.

        Parameters
        ----------
        phase: `.Phase`
        """
        self.phases[phase.name] = phase
        phase.microstructure = self
        if self.phases[phase.name].type.__name__ == "Matrix":
            if self.matrix_phase is not None:
                raise ValueError(
                    "The matrix for Phase {0} was specified twice.".format(phase.name)
                )
            self.matrix_phase = phase.name
        else:
            if self.dim != self.phases[phase.name].type.dim:
                raise ValueError(
                    "The particle type chosen is not compatible "
                    + "with the dimensions of the microstructure."
                )
            if self.phases[phase.name].type.__name__ == "CylindricalFiber":
                for i_phase in self.phases.values():
                    if i_phase.type.__name__ not in ("CylindricalFiber", "Matrix"):
                        raise ValueError(
                            "The CylindricalFiber particles are only compatible with"
                            + " each other."
                        )

    def inside_particle_phase(self, pts):
        """
        Say, for each of a set of points, whether it lies inside a particle.

        Parameters
        ----------
        pts: list(array)
            Positions of the points, which may lie outside the RVE and are wrapped
            into it.

        Returns
        -------
        list(int)
            1 for a point inside a particle, 0 otherwise, in the order given.
        """
        dim = len(pts[0])
        box = np.asarray(self.rve_dims, dtype=float)[:dim]
        particles = self.particles
        cell_list = CellList()
        cell_list.box = box
        cell_list.max_radius = np.max([i_particle.radius for i_particle in particles])
        centers = np.array(
            [i_particle.position_center[:dim] for i_particle in particles], dtype=float
        )
        particle_cells = cell_list.cells_of(centers - box * np.floor(centers / box))
        # The cells are sized by the largest particle, so a point inside one lies in the
        # cell of its centre or in a neighbour of it: the guarantee the simulation relies
        # on for two particles overlapping, with a point in place of the second. The
        # points themselves are only looked up in the list; putting them in it, as used
        # to be done, tested every point against every other point in its cell. The
        # centres are wrapped into the RVE, as the points are. They were put in the
        # cells as they were, so a centre outside the RVE, which a microstructure read
        # from a file may have, was counted in the last cell along a direction, or in a
        # wrong one, and the points around it were not tested against it

        # TODO: the cell list is built anew on every call, and the `cells_around` of
        # every cell with it, although the particles have not moved between the calls:
        # the two point correlation function asks 401 times over one microstructure and
        # spends a third of its time on the rebuilds. The index belongs to the
        # microstructure, held until the particles move, or to a grid of its own that
        # `CellList` and this both read -- which would also give the analyses a point
        # index that does not carry the force computation machinery of a speed up
        # scheme, and let `cells_around` be worked out once per grid rather than once
        # per instance
        positions = np.asarray(pts, dtype=float)
        positions = positions - box * np.floor(positions / box)
        cells = cell_list.cells_of(positions)
        order = np.argsort(cells, kind="stable")
        n_cells = int(np.prod(cell_list.n_cell_dim))
        starts = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=n_cells))])
        # The points in the order of the cell they fall in, and where the points of
        # each cell begin, since the cell of a particle is what says which points can
        # be inside it

        inside = np.zeros(len(positions), dtype=bool)
        for i_particle, i_cell in zip(particles, particle_cells):
            neighborhood = np.concatenate(
                [
                    order[starts[j_cell]:starts[j_cell + 1]]
                    for j_cell in cell_list.cells_around[i_cell]
                ]
            )
            if neighborhood.size > 0:
                inside[neighborhood] |= i_particle.points_inside(
                    positions[neighborhood], self.rve_dims
                )
        # Every point that can be inside a particle is tested against it in one array
        # operation, where the test used to be a call into a particle for each point
        # and each particle around it -- which is the whole of what the two point
        # correlation function costs. The size is asked for so that a particle with no
        # point near it is not called at all

        return inside.astype(int).tolist()

    @property
    def particles(self):
        """Particles in the microstructure."""
        particles = []
        for i_phase in self.phases.values():
            particles += i_phase.particles

        return particles

    @property
    def volume_fraction(self):
        """Volume fraction of particles in the microstructure."""
        vf = 0
        for i_phase in self.phases.values():
            vf += i_phase.volume_fraction

        return vf

    @property
    def volume_fraction_circ(self):
        """Volume fraction of circumscribed sphesres/disks to the particles."""
        vf = 0
        for i_phase in self.phases.values():
            vf += i_phase.volume_fraction_circ

        return vf
