"""
Module containing all the Particle abstract class and all its subclasses.

Each subclass of the Particle class is a type of particle. This module includes the Ellipse,
Disk, CylindricalFiber, Ellipsoid and Shpere classes.
"""
from __future__ import annotations

import abc
import time


from itertools import cycle

import numpy as np

from scipy import integrate
from scipy.optimize import fmin


class Particle(abc.ABC):
    """
    This is the class for particles.

    Each particle in the microstructure is an instance of this class.

    Attributes
    ----------
    position_center: array
        The position vector of the center of mass of the particle

    dim: int
        Number of the dimensions of the space where the particle "lives"

    phase: str
        Phase to which the particle belongs.

    Class Atributes
    ---------------
    possible_parameters: dict
        Possible parameters characterizing a phase containing this of particle.

    acceptable_descriptions: list(set)
        List of acceptable descriptions.
    """

    possible_parameters = {
        "n": ("Number of particles", "int"),
        "vf": ("Volume fraction", "float"),
    }
    acceptable_descriptions = [set()]
    dist_met = "dist_approx"

    def __init__(self, dim: int, phase: str):
        """
        Initialize a Particle class object.

        Parameters
        ----------
        dim: int
            Number of the dimensions of the space where the particle "lives".

        phase: str
            Name of the phase to which the particle belongs.

        position_center: array
            Position of the center of mass of the particle.
        """
        self.dim = dim
        # Setting the the dimension where the particle "lives"
        self.phase = phase
        # Phase to which the particle belongs
        self.position_center = None
        self.delta = 0

    @staticmethod
    def nearest_periodic_image(point_1, point_2, box):
        """Get the nearest periodic image of *point_1* to *point_2* in *box* with pbcs."""
        diff_in_box = point_2 - point_1
        # Difference vector between the two points
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the nearest image of point_2 relative point_1 to point_1
        point_1_nearest_image = point_1 + diff_nearest_other

        return point_1_nearest_image

    @classmethod
    def check_acceptable_description(cls, descriptors):
        """Check if descriptors are an acceptable description.

        Depending on the type of particle different sets of descriptors are sufficient to
        describe the particles in a phase.

        Parameters
        ----------
        descriptors: set(str)
            Set of descriptors.
        """
        if any(
            [
                descriptors == acceptable_description
                for acceptable_description in cls.acceptable_descriptions
            ]
        ):
            # Checking acceptable sets of parameters
            pass
        else:
            raise ValueError(
                "{0} is not an acceptable description of an {1}.".format(
                    descriptors, cls.__name__
                )
            )

    def intersection_gjk(
        self,
        particle_2: Particle,
        box: list,
        tol: float = 1e-8,
        inside=True,
        out_dist=False,
        int_only=False,
    ) -> tuple[bool, float]:
        """Check using a version of the GJK intersection method if the particles intersect.

        At the core of the algorithm is the theorem stating that if the two convex shapes
        intersect, their Minkowski difference, i.e. the difference of all the points in both
        sets, contains the origin. The distance between the two shapes will also be the
        smallest distance between the origin and the boundary of the Minkowski difference.

        We start with a random direction, get the corresponding point from the support
        function of the Minkowski difference. After this first step we try to find points
        belonging to the Minkowski difference that are closer to the origin. This always
        motivates the direction of the search direction.

        We conclude that the two shapes do not intersect if the new point from the Minkowski
        difference is behind the origin as seen from the perspective of the search
        direction, i.e. their dot product is negative.

        Instead of computing the complete Minkowski difference of the shapes we limit
        ourselves to simplices, and try to enclose the origin with the simplex of hightest
        dimension corresponding to the problem, a triangle for 2D and a tetrahedron for 3D.

        Parameters
        ----------
        particle_2: `.Particle`
            Particle that will be checked for intersection with *self*.

        box: list
            Simulation box containing the particles.

        tol: float
            Tolerance for the minimum distance

        inside: bool
            Consider self completly inside particle_2 as an intersection or not.

        Returns
        -------
        intersection: bool
            True if the particles intersect, False otherwise.

        overlap_length: float
            The penetratin length if an intersection is detected, *None* otherwise.

        minimum_dist_rem: np.array
            Unit vector such a displacement of magnitude *overlap_length* removes completly
            the overlap.
        """
        intersection = None
        diff_in_box = self.position_center - particle_2.position_center
        # Difference vector between the center of the two particles
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current
        # ellipse
        if self.dim == 2:
            diff_nearest_other = np.append(diff_nearest_other, [0])

        random_dir = np.array(
            [
                1,
                0.5,
                0.2,
            ]
        )
        simplex = [
            self.support_function(random_dir)
            - (particle_2.support_function(-random_dir) + diff_nearest_other)
        ]
        search_direction = -simplex[0]
        k_iter = 0
        while True:
            if k_iter == 100:
                # The algorithm diverged. The conservative posture is taken and it is
                # assumed that the shapes intersect
                intersection = True
                break
            k_iter += 1
            new_mink_diff_point = self.support_function(search_direction) - (
                particle_2.support_function(-search_direction) + diff_nearest_other
            )
            if (
                new_mink_diff_point.dot(np.array(search_direction)) < 0
                and intersection is None
            ):
                # If moving towards the origin we have no gone past it
                intersection = False
                break
            # We've gone past the origin
            simplex.append(new_mink_diff_point)
            simplex, search_direction = self.nearest_simplex(simplex)

            # We have found a simplex of the hightest dimension of the problem containing
            # the origin
            if len(simplex) == self.dim + 1:
                intersection = True
                break

        return intersection

    def intersection_length(
        self,
        particle_2: Particle,
        box: list,
        tol: float = 1e-8,
        dist_met: str = "dist_exact",
    ) -> tuple[float, np.array]:
        """
        Intersection length of *self* with *particle_2*, assuming they intersect.

        The intersection length is here defined as the smallest distance from the origin to
        the boundary of the Minkowski difference of the shapes of the two particles, *self*
        and *particle_2*.

        It is computed to a prescribed accuracy using a minimization algorithm (Nelder-Mead)
        or an approximation is supplie whose error is not estimated.

        Parameters
        ----------
        particle_2: `.Particles`
            Particle that intersects *self*.

        box: list(float)
            Dimensions of the simulation box.

        tol: float (optional)
            Tolerance used for the computation of the intersection length.

        dist_met: {"dist_exact", "dist_approx"} (optional)
            Flag for the exact computation of the intersection length and direction, or the
            use of an approximation.

        Returns
        -------
        intersection_length: float
            Intersection legnth between *self* and *particle_2*.

        intersection_dir: float
            Direction along which a displacement of magnitude *intersection_length* would
            lead to tangent particles.
        """
        diff_in_box = self.position_center - particle_2.position_center
        # Difference vector between the center of the two particles
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current
        # ellipse
        if self.dim == 2:
            diff_nearest_other = np.append(diff_nearest_other, [0])

        if Particle.dist_met == "dist_approx":
            unit_vector = self.intersection_vector(particle_2, box)
            mink_diff_point = self.support_function(unit_vector) - (
                particle_2.support_function(-unit_vector) + diff_nearest_other
            )
            intersection_length = mink_diff_point[0 : self.dim].dot(unit_vector)
            # intersection_length = np.linalg.norm(mink_diff_point)
            # intersection_dir = mink_diff_point / np.linalg.norm(mink_diff_point)
            intersection_dir = unit_vector

        elif Particle.dist_met == "dist_exact":
            # --------------------------------------------------------------------------
            unit_vector = self.intersection_vector(particle_2, box)
            if self.dim == 2:
                first_guess = np.array([np.arctan2(unit_vector[1], unit_vector[0])])
            elif self.dim == 3:
                first_guess = np.array(
                    [
                        np.arctan2(unit_vector[1], unit_vector[0]),
                        np.arctan(
                            np.sqrt(unit_vector[0] ** 2 + unit_vector[1] ** 2)
                            / unit_vector[2]
                        ),
                    ]
                )
            # Using the unit vector of the straight line going throught the center of
            # the particles as a first guess.
            # The search for the minimum distance is done on the space of the polar and
            # spherical coordinates
            search_direction_angles, intersection_length, *_ = fmin(
                Particle.minimum_dist_to_diff_sup,
                first_guess[0 : self.dim - 1],
                args=(self, particle_2, box),
                # xtol=1e-3,
                ftol=tol,
                maxiter=1000,
                full_output=1,
                disp=0,
            )
            if intersection_length <= 0:
                intersection_length = 0
                intersection_dir = unit_vector
            if self.dim == 2:
                i_theta = search_direction_angles[0]
                intersection_dir = np.array([np.cos(i_theta), np.sin(i_theta)])

            elif self.dim == 3:
                i_theta, j_phi = search_direction_angles
                intersection_dir = np.array(
                    [
                        np.sin(i_theta) * np.cos(j_phi),
                        np.sin(i_theta) * np.sin(j_phi),
                        np.cos(i_theta),
                    ]
                )
        return intersection_length, intersection_dir

    @staticmethod
    def minimum_dist_to_diff_sup(search_direction_unit, particle_1, particle_2, box):
        """Distance from the origin to the suppport function of the Minkowski difference.

        Gives the distance along the search direction of the corresponding point in the
        support function of the Minkowski difference of the two particles.
        The *search_direction_unit* contains the angles necessary to define a unit vector.


        Parameters
        ---------
        search_direction: np.array
            Array containing the angles defining the unit vector, using polar coordinates
            for 2D and spherical coordinates for 3D.

        particle_1: `.Particle`
            Particle

        particle_2: `.Particle`
            Particle

        box: list(float)
            Dimension of the simulation box in each direction.

        Returns
        -------
        dist: float
            Distance from the origin to the point on the support function of the Minkowski
            difference of the two particles corresponding to the search direction.
        """
        diff_in_box = particle_1.position_center - particle_2.position_center
        # Difference vector between the center of the two particles
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current
        # ellipse
        # diff = diff_in_box + diff_nearest_other
        if particle_1.dim == 3:
            i_theta, j_phi = search_direction_unit
            search_direction = np.array(
                [
                    np.sin(i_theta) * np.cos(j_phi),
                    np.sin(i_theta) * np.sin(j_phi),
                    np.cos(i_theta),
                ]
            )
        elif particle_1.dim == 2:
            diff_nearest_other = np.append(diff_nearest_other, [0])
            i_theta = search_direction_unit[0]
            search_direction = np.array([np.cos(i_theta), np.sin(i_theta), 0])

        mink_diff_point = particle_1.support_function(search_direction) - (
            particle_2.support_function(-search_direction) + diff_nearest_other
        )
        dist = mink_diff_point.dot(search_direction)
        # dist = np.linalg.norm(mink_diff_point)
        # diff / np.linalg.norm(diff))
        return dist

    @staticmethod
    def nearest_simplex(simplex):
        """
        Get the nearest simplex to the origin and the corresponding search direction.

        This function computes from the simplex given the subset of points comprising a the
        closest simplex to the origin.
        """

        def nearest_triangle(vec_to_origin, vec_1, vec_2, sub_simplex):

            normal_tri = np.cross(vec_1, vec_2)
            if np.cross(vec_1, normal_tri).dot(vec_to_origin) > 0:
                if vec_1.dot(vec_to_origin) > 0:
                    del sub_simplex[1]
                    search_direction = np.cross(
                        vec_1,
                        np.cross(vec_to_origin, vec_1),
                    )
                elif vec_2.dot(vec_to_origin) > 0:
                    del sub_simplex[0]
                    search_direction = np.cross(
                        vec_2,
                        np.cross(vec_to_origin, vec_2),
                    )
                else:
                    del sub_simplex[0:2]
                    search_direction = vec_to_origin
            elif np.cross(normal_tri, vec_2).dot(vec_to_origin) > 0:
                if vec_2.dot(vec_to_origin) > 0:
                    del sub_simplex[0]
                    search_direction = np.cross(
                        vec_2,
                        np.cross(vec_to_origin, vec_2),
                    )
                else:
                    del sub_simplex[0:2]
                    search_direction = vec_to_origin
            else:

                if normal_tri.dot(vec_last_to_origin) > 0:
                    search_direction = normal_tri
                    sub_simplex.reverse()
                else:
                    search_direction = -normal_tri

            return sub_simplex, search_direction

        if len(simplex) == 2:

            vec_last_to_previous = simplex[0] - simplex[1]
            vec_last_to_origin = -simplex[1]
            if vec_last_to_previous.dot(vec_last_to_origin) > 0:
                search_direction = (
                    vec_last_to_origin
                    - vec_last_to_previous.dot(vec_last_to_origin)
                    * vec_last_to_previous
                    / np.linalg.norm(vec_last_to_previous) ** 2
                )
                #     np.linalg.norm(vec_last_to_previous) * vec_last_to_origin
                #     - vec_last_to_previous.dot(vec_last_to_origin) * vec_last_to_origin
                # )
            else:
                search_direction = simplex[1]
                del simplex[0]
        elif len(simplex) == 3:

            vec_last_to_origin = -simplex[2]
            vec_last_to_previous_1 = simplex[0] - simplex[2]
            vec_last_to_previous_2 = simplex[1] - simplex[2]
            simplex, search_direction = nearest_triangle(
                vec_last_to_origin,
                vec_last_to_previous_1,
                vec_last_to_previous_2,
                [simplex[0], simplex[1], simplex[2]],
            )

        elif len(simplex) == 4:
            vec_last_to_origin = -simplex[3]
            vec_last_to_previous_1 = simplex[0] - simplex[3]
            vec_last_to_previous_2 = simplex[1] - simplex[3]
            vec_last_to_previous_3 = simplex[2] - simplex[3]
            normal_tri_12 = np.cross(vec_last_to_previous_1, vec_last_to_previous_2)
            normal_tri_23 = np.cross(vec_last_to_previous_2, vec_last_to_previous_3)
            normal_tri_31 = np.cross(vec_last_to_previous_3, vec_last_to_previous_1)
            # Outward normals
            if normal_tri_12.dot(vec_last_to_origin) < 0:
                simplex, search_direction = nearest_triangle(
                    vec_last_to_origin,
                    vec_last_to_previous_1,
                    vec_last_to_previous_2,
                    [simplex[0], simplex[1], simplex[3]],
                )

            elif normal_tri_23.dot(vec_last_to_origin) < 0:
                simplex, search_direction = nearest_triangle(
                    vec_last_to_origin,
                    vec_last_to_previous_2,
                    vec_last_to_previous_3,
                    [simplex[1], simplex[2], simplex[3]],
                )
            elif normal_tri_31.dot(vec_last_to_origin) < 0:
                simplex, search_direction = nearest_triangle(
                    vec_last_to_origin,
                    vec_last_to_previous_3,
                    vec_last_to_previous_1,
                    [simplex[2], simplex[0], simplex[3]],
                )
            else:
                search_direction = np.array([0, 0, 0])

        return simplex, search_direction

    def intersection_vector(self, other_particle, box):
        """Compute the unit vector from the center of masss of particle i to particle j.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle.

        box: list
            Dimensions of the simulation box in each spatial direction

        Returns
        -------
        unit_vector_i_j: array
            Unit vector from the center of *self* to *other_particle*
        """
        vector_centers = other_particle.position_center - self.position_center
        vector_centers = vector_centers - box * np.round(vector_centers / box)
        # Vector connecting the centers of the current particle and the nearest image of
        # the other particle
        if np.linalg.norm(vector_centers) != 0:
            unit_vector_i_j = vector_centers / np.linalg.norm(vector_centers)
            # unit_vector_i_j = vector_centers/np.linalg.norm(vector_centers)
        else:
            random_vector = np.random.uniform(size=self.dim)
            unit_vector_i_j = random_vector / np.linalg.norm(random_vector)

        return unit_vector_i_j

    @abc.abstractmethod
    def intersection(self, other_particle, box) -> bool:
        """Check if the two particles intersect."""

    @abc.abstractmethod
    def intersection_area(self, other_particle, box) -> float:
        """Compute the interesection area/volume between two particles."""

    @abc.abstractmethod
    def intersection_length(self, other_particle, box) -> tuple[float, np.array]:
        """Compute the interesection length between two particles."""

    @abc.abstractmethod
    def support_function(self, direction: np.array) -> np.array:
        """Compute the interesection length between two particles."""

    @abc.abstractmethod
    def point_inside(self, point: np.array) -> bool:
        """Check if some point is inside the particle."""

    @abc.abstractmethod
    def generate_point_inside(self):
        """Generate a random point inside the particle."""

    def intersection_area_monte_carlo(
        self, other_particle, box, tol=1, max_it=20000, min_it=100
    ):
        """Integrate the intersection area/volume using a Monte Carlo technique."""
        points_in = []
        point = self.generate_point_inside()
        if other_particle.point_inside(point, box):
            points_in.append(1)
        else:
            points_in.append(0)
        while True:
            point = self.generate_point_inside()
            if other_particle.point_inside(point, box):
                points_in.append(1)
            else:
                points_in.append(0)
            current_std = np.std(points_in)
            if len(points_in) > min_it:
                if (
                    (self.volume / other_particle.volume)
                    * 100
                    * 2
                    * current_std
                    / np.sqrt(len(points_in))
                    < tol
                    and current_std != 0
                ) or len(points_in) > max_it:
                    overlap_area = np.mean(points_in) * self.volume
                    error_estimate = (
                        2 * current_std / np.sqrt(len(points_in)) * self.volume
                    )
                    break

        return overlap_area, error_estimate

    def mass(self, option="volume"):
        """Return the mass of the particle according to the *option* selected.

        Parameters
        ----------
        option: {"volume", "radius", "unit"}
            Consider the mass equal to its volume, its radius or equal to one.
        """

        if option == "volume":
            mass = self.volume
        elif option == "radius":
            mass = self.radius
        elif option == "unit":
            mass = 1
        else:
            raise ValueError("Unsupported option for the particle mass.")

        return mass

    def force_spring(self, other_particle, box, degree=2, tol=1e-8):
        """Compute force due to non-linear spring at the intersection of degree *degree*."""
        disp, unit_vector = self.intersection_length(other_particle, box, tol=tol)
        dist = self.radius + other_particle.radius - disp
        # Distance between the current sphere and the nearest image of the other sphere
        r_min = (
            self.radius
            if self.radius < other_particle.radius
            else other_particle.radius
        )
        r_max = (
            other_particle.radius
            if other_particle.radius > self.radius
            else self.radius
        )
        if disp <= 0:
            force = 0
        elif disp >= r_min + r_max:
            force = r_min + r_max
        else:
            force = (r_max + r_min) * (1 - (dist / (r_max + r_min)) ** degree)
        return force, unit_vector

    @property
    def volume_circ(self):
        """Volme of the corresponding circumscribed spheres/disk."""
        if self.dim == 2:
            volume = np.pi * self.radius ** 2
        elif self.dim == 3:
            volume = 4 / 3 * np.pi * self.radius ** 3

        return volume


class Ellipse(Particle):
    """
    This is the class for Ellipse.

    Attributes
    ----------
    major_axis: float
        Major axis of the ellipse.

    semi_major_axis: float
        Half the major axis.

    minor_axis: float
        Minor axis of the ellipse.

    semi_minor_axis: float
        Half the minor axis.

    eccentricity: float
        Eccentricity of the ellipse.

    radius: float
        Radius of the circunscribed circle

    radius_insc: float
        Radius of the inscribed circle

    rot_mat: 2-array(floats)
        Rotation matrix from the global to the local coordinates.

    angle: float
        Angle in radians that the major axis forms with the x-axis.

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe an ellipse,
        and their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing ellipses.

    dim: int
        Dimension that the particle inhabits.
    """

    possible_parameters = {
        **{
            "major_axis": ("Major axis", "float"),
            "minor_axis": ("Minor axis", "float"),
            "angle": ("Angle", "float"),
            "eccentricity": ("Eccentricity", "float"),
            "ratio": ("Ratio a/b", "float"),
        },
        **Particle.possible_parameters,
    }
    # all possible_parameters
    acceptable_descriptions = [
        {"major_axis", "minor_axis", "angle", "n"},
        {"major_axis", "minor_axis", "angle", "vf"},
        {"major_axis", "angle", "n", "vf"},
        {"minor_axis", "angle", "n", "vf"},
        {"ratio", "angle", "n", "vf"},
        {"major_axis", "ratio", "angle", "vf"},
        {"major_axis", "ratio", "angle", "n"},
    ]
    dim = 2
    # List of acceptable collections of parameters

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Ellipse obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        if "major_axis" in descriptors and "minor_axis" in descriptors:
            # Both major and minor axis were supplied
            major_axis = np.max(
                [descriptors["major_axis"], descriptors["minor_axis"]], axis=0
            )
            minor_axis = np.min(
                [descriptors["major_axis"], descriptors["minor_axis"]], axis=0
            )
            # Ensuring that the major axis is greater than the minor axis
        elif "major_axis" in descriptors and "vf" in descriptors and "n" in descriptors:
            # The major_axis, the volume faction and the number of particles were supplied
            volume_part = (
                descriptors["vf"] * rve_dims[0] * rve_dims[1] / descriptors["n"]
            )
            # All particles will have the same volume
            aux_minor_axis = volume_part / (np.pi * descriptors["major_axis"] * 1 / 4)
            # Minor axis computed assuming that all particles have the same area
            major_axis = np.max([descriptors["major_axis"], aux_minor_axis], axis=0)
            minor_axis = np.min([descriptors["major_axis"], aux_minor_axis], axis=0)
            # Ensuring that the major axis is greater than the minor axis
        elif "minor_axis" in descriptors and "vf" in descriptors and "n" in descriptors:
            # The minor axis, the volume faction and the number of particles were supplied
            volume_part = (
                descriptors["vf"] * rve_dims[0] * rve_dims[1] / descriptors["n"]
            )
            aux_major_axis = volume_part / (np.pi * descriptors["minor_axis"] * 1 / 4)
            # Minor axis computed assuming that all particles have the same area
            major_axis = np.max([aux_major_axis, descriptors["minor_axis"]], axis=0)
            minor_axis = np.min([aux_major_axis, descriptors["minor_axis"]], axis=0)
            # Ensuring that the major axis is greater than the minor axis
        elif "ratio" in descriptors and "vf" in descriptors and "n" in descriptors:
            volume_part = (
                descriptors["vf"] * rve_dims[0] * rve_dims[1] / descriptors["n"]
            )
            minor_axis = np.sqrt(volume_part / (np.pi * descriptors["ratio"] * 1 / 4))
            major_axis = descriptors["ratio"] * minor_axis
        elif "ratio" in descriptors and "major_axis" in descriptors:
            # Ratio and major axis were supplied
            major_axis = descriptors["major_axis"]
            minor_axis = major_axis / descriptors["ratio"]
        if "angle" in descriptors:
            angle = descriptors["angle"]

        if major_axis <= 0 or minor_axis <= 0:
            raise ValueError("Major and minor axis must be positive values.")

        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.angle = angle
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), np.sin(self.angle)],
                [-np.sin(self.angle), np.cos(self.angle)],
            ]
        )
        super().__init__(2, phase)

    @property
    def volume(self):
        """Area(volume) of the ellipse."""
        volume = np.pi * self.semi_major_axis * self.semi_minor_axis

        return volume

    @property
    def semi_major_axis(self):
        """Semi major axis of the ellipse."""
        semi_major_axis = self.major_axis / 2

        return semi_major_axis

    @property
    def semi_minor_axis(self):
        """Semi minor axis of the ellipse."""
        semi_minor_axis = self.minor_axis / 2

        return semi_minor_axis

    @property
    def eccentricity(self):
        """Eccentricity of the ellipse."""
        eccentricity = np.sqrt(1 - self.minor_axis ** 2 / self.major_axis ** 2)

        return eccentricity

    @property
    def radius(self):
        """Radius of the circumscribed circle to the ellipse."""
        radius = self.semi_major_axis + self.delta

        return radius

    @property
    def radius_insc(self):
        """Radius of the inscribed circle to the ellipse."""
        radius_insc = self.semi_minor_axis

        return radius_insc

    def contract(self, distance):
        """Contract the particle."""
        self.delta -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        self.delta += distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def point_inside(self, point, box, tol=1e-4, position="inside"):
        """
        Check if the point is inside, outside or on the ellipse given a tolerance.

        Parameters
        ---------
        self: `.Ellipse`
            Ellipse under analysis

        point: array
            Point under analysis

        tol: float
            Tolerance

        position: string
            'inside' or 'on'

        Returns
        -------
        point_in: bool
            True if the point is inside the ellipse and False otherwise.
        """
        ## FIXME: new erosion not considered
        if self.delta != 0:
            print("WARNING!!!")
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        r_vector = self.rot_mat.dot(point_nearest_pbc - self.position_center)
        # Defininig the radius vector relative to the coordinate system of the ellipse
        # r_point = np.linalg.norm(r_vector)
        # # Distance from the point to the center of the ellipse
        # angle_pt_major = np.arctan2(r_vector[1], r_vector[0])
        # Angle that the vector connecting the center of the ellipse and the point makes
        # with the major axis
        if position == "inside":
            # Checking if the point is inside the ellipse
            # point_in = r_point <= tol + self.semi_minor_axis / np.sqrt(
            #     1 - (self.eccentricity * np.cos(angle_pt_major)) ** 2
            # )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
            point_in = (r_vector[0] / self.semi_major_axis) ** 2 + (
                r_vector[1] / self.semi_minor_axis
            ) ** 2 <= 1 + tol
        elif position == "on":
            # Checking if the point is on the ellipse
            # point_in = (
            #     np.abs(
            #         r_point
            #         - self.semi_minor_axis
            #         / np.sqrt(1 - (self.eccentricity * np.cos(angle_pt_major)) ** 2)
            #     )
            #     < tol
            # )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
            point_in = (
                np.abs(
                    (r_vector[0] / self.semi_major_axis) ** 2
                    + (r_vector[1] / self.semi_minor_axis) ** 2
                    - 1
                )
                < tol
            )
        return point_in

    def intersection_area_ellipse_ellipse(self, other_ellipse, box):
        """Compute the orverlap area between the current and the other ellipse."""
        diff_in_box = self.position_center - other_ellipse.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector from the position of the other ellipse to its nearest image to the current
        # ellipse
        intersect_pts = intersection_points_ellipses(
            self.major_axis / 2,
            self.minor_axis / 2,
            self.position_center,
            self.angle,
            other_ellipse.major_axis / 2,
            other_ellipse.minor_axis / 2,
            other_ellipse.position_center + diff_nearest_other,
            other_ellipse.angle,
        )
        # Computing the intersection points of the two ellipses
        intersection_area = 0
        if len(intersect_pts) == 0:
            # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume >= other_ellipse.volume:
                # The current ellipse is larger than the other ellipse
                if self.point_inside(other_ellipse.position_center, box):
                    # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.point_inside(self.position_center, box):
                    # The current ellipse is completly inside the other ellipse
                    intersection_area = self.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
        elif len(intersect_pts) == 1:
            # Either the ellipses are disjoint or one of them is completly inside the other,
            # except for the intersection point
            if self.volume >= other_ellipse.volume:
                # The current ellipse is larger than the other ellipse
                if self.point_inside(other_ellipse.position_center, box):
                    # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.point_inside(self.position_center, box):
                    # The current ellipse is completly inside the other ellipse
                    intersection_area = self.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
        elif len(intersect_pts) == 2:
            # The ellipses intersect in two points. The case where one of the ellipses is
            # inside the other and both are tangent at the intersection points is
            # disregarded
            intersection_area = 0
            # Initializing the intersection area
            intersect_pts_ord = self.sort_points_on_ellipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            ellipses = [self, other_ellipse]
            # Saving the ellipses in a list
            midpoint = self.midpoint_on_ellipse(
                intersect_pts_ord[0], intersect_pts_ord[1]
            )
            # Midpoint between the first two intersection points in the current ellipse
            if other_ellipse.point_inside(midpoint - diff_nearest_other, box):
                # If the midpoint is on the other ellipse
                intersection_area += self.area_ellipse_section(
                    intersect_pts_ord[0], intersect_pts_ord[1]
                )
                # The correct segment belongs to the current ellipse
                k_ellipse = 0
                # Index of the current ellipse
            else:
                intersection_area += other_ellipse.area_ellipse_section(
                    intersect_pts_ord[0] - diff_nearest_other,
                    intersect_pts_ord[1] - diff_nearest_other,
                )
                # The correct segment belongs to the other ellipse
                k_ellipse = 1
                # Index of the other ellipse
            for i_segment in range(1, 2):
                # Running through each segment
                k_ellipse = np.mod(k_ellipse + 1, 2)
                # Index of the ellipse whose area segment needs to calculated
                intersection_area += ellipses[k_ellipse].area_ellipse_section(
                    intersect_pts_ord[np.mod(i_segment, 2)]
                    - k_ellipse * diff_nearest_other,
                    intersect_pts_ord[np.mod(i_segment + 1, 2)]
                    - k_ellipse * diff_nearest_other,
                )
                # Computing the area of the segment
        elif len(intersect_pts) == 3:
            # This case is disregarded
            intersection_area = 0
        elif len(intersect_pts) == 4:
            # One of the ellipses goes through the other
            intersection_area = 0
            # Initializing the intersection area
            intersect_pts_ord = self.sort_points_on_ellipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            intersection_area += 0.5 * np.abs(
                (intersect_pts_ord[2][0] - intersect_pts_ord[0][0])
                * (intersect_pts_ord[3][1] - intersect_pts_ord[1][1])
                - (intersect_pts_ord[3][0] - intersect_pts_ord[1][0])
                * (intersect_pts_ord[2][1] - intersect_pts_ord[0][1])
            )
            # Computing the area of the quadrilateral inscribed in the overlap of the
            # two ellipses
            ellipses = [self, other_ellipse]
            # List of the ellipse objects to iterate over
            midpoint = self.midpoint_on_ellipse(
                intersect_pts_ord[0], intersect_pts_ord[1]
            )
            # Obtaining the midpoint between the first two intersection points to decide
            # to which ellipses belong to the area sections to be calculated
            if other_ellipse.point_inside(midpoint - diff_nearest_other, box):
                intersection_area += self.area_ellipse_section(
                    intersect_pts_ord[0], intersect_pts_ord[1]
                )
                k_ellipse = 0
            else:
                intersection_area += other_ellipse.area_ellipse_section(
                    intersect_pts_ord[0] - diff_nearest_other,
                    intersect_pts_ord[1] - diff_nearest_other,
                )
                k_ellipse = 1
            for i_segment in range(1, 4):
                # Running through each segment
                k_ellipse = np.mod(k_ellipse + 1, 2)
                intersection_area += ellipses[k_ellipse].area_ellipse_section(
                    intersect_pts_ord[np.mod(i_segment, 4)]
                    - k_ellipse * diff_nearest_other,
                    intersect_pts_ord[np.mod(i_segment + 1, 4)]
                    - k_ellipse * diff_nearest_other,
                )

        return intersection_area

    def midpoint_on_ellipse(self, point_1, point_2):
        """
        Return the point midway between point_1 and point_2, anti clockwise.

        The midpoint is determined using the parameteric angles of the points relative to
        the center of the ellipse and its major axis.
        """
        angle = []
        for i_point in [point_1, point_2]:
            # Running through all the points
            radius_vector = self.rot_mat.dot(i_point - self.position_center)
            # Obtaining the radius vector corresponding to the i_point in the coordinate
            # system of the ellipse
            i_angle = np.arctan2(radius_vector[1], radius_vector[0])
            # Angle the radius vector of the point makes with the major axis of the ellipse
            # between 0 and pi
            if i_angle < 0:
                # If the y-coordinate of the radius vector is negative
                i_angle = i_angle + 2 * np.pi
                # Accounting for the fact that arccos only gives values between 0 and pi
            angle.append(i_angle)
        angle[1] = angle[1] + 2 * np.pi if angle[0] > angle[1] else angle[1]
        angle_mid = (angle[0] + angle[1]) / 2
        # Angle of the midpoint
        radius_mid = self.semi_minor_axis / np.sqrt(
            1 - (self.eccentricity * np.cos(angle_mid)) ** 2
        )
        # Radius of the midpoint
        midpoint_loc = radius_mid * np.array([np.cos(angle_mid), np.sin(angle_mid)])
        # Cartesian coordinates of the midpoint in the coordinate system of the ellipse
        midpoint = self.position_center + self.rot_mat.T.dot(midpoint_loc)
        # Cartesian coordinates of the midpoint in the global coordinate system
        return midpoint

    def sort_points_on_ellipse(self, points):
        """
        Sort the points given in the ellipse clockwise.

        The points are sorted using their parameteric angles, measured relative to the
        center of the ellipse and its major axis. All angles are assumed to be positive.
        """
        angle = []
        for i_point in points:
            # Running through all the points
            radius_vector = self.rot_mat.dot(i_point - self.position_center)
            # Obtaining the radius vector corresponding to the i_point in the coordinate
            # system of the ellipse
            angle_i = np.arctan2(radius_vector[1], radius_vector[0])
            # Angle the radius vector of the point makes with the major axis of the ellipse
            # between 0 and pi
            if angle_i < 0:
                # If the y-coordinate of the radius vector is negative
                angle_i = angle_i + 2 * np.pi
                # Accounting for the fact that arccos only gives values between 0 and pi
            angle.append(angle_i)
            # Appending the angle
        y_ordered = [points[i] for i in np.argsort(angle)]
        # Obtaining the list of points with angles sorted counter clockwise
        return y_ordered

    def area_ellipse_section(self, intersect_pt_1, intersect_pt_2):
        """
        Compute the area of the section defined by two points.

        Compute the area of the segment defined by the secant line drawn between the two
        points given and the ellipse, anti clockwise from point 1 to point 2.

        Parameters
        ----------
        self: `.Ellipse`
            Ellipse under analysis.

        intersect_pt_1: array
            Array containing the coordinates of the first intersection point. The
            funtion does not check if the point is indeed on the ellipse

        intersect_pt_2: array
            Array containing the coordinates of the second intersection point. The
            funtion does not check if the point is indeed on the ellipse

        Returns
        --------
        area_segment: float
            Area of the segment defined by the secant line drawn between the two
            points given and the ellipse
        """
        pt_1 = self.rot_mat.dot(intersect_pt_1 - self.position_center)
        pt_2 = self.rot_mat.dot(intersect_pt_2 - self.position_center)
        # Translation and rotation of the ellipse to the origin aligning with the xy axis
        if pt_1[1] > 0:
            theta_1 = np.arccos(
                np.max([np.min([pt_1[0] / self.semi_major_axis, 1]), -1])
            )
        else:
            theta_1 = 2 * np.pi - np.arccos(
                np.max([np.min([pt_1[0] / self.semi_major_axis, 1]), -1])
            )
        # Computing the parametric angle corresponding to the first intersection point
        # ensuring that there are no errors using the trigonometric functions
        if pt_2[1] > 0:
            theta_2 = np.arccos(
                np.max([np.min([pt_2[0] / self.semi_major_axis, 1]), -1])
            )
        else:
            theta_2 = 2 * np.pi - np.arccos(
                np.max([np.min([pt_2[0] / self.semi_major_axis, 1]), -1])
            )
        # Computing the parametric angle corresponding to the second intersection point
        if theta_1 <= theta_2:
            theta_1_hat = theta_1
        else:
            theta_1_hat = theta_1 - 2 * np.pi
        # Ensuring that the angle theta_1 is always smaller than theta_2 as the area
        # is computed in an anti-clockwise manner from point 1 to 2
        area_sector = (
            (theta_2 - theta_1_hat) * self.semi_major_axis * self.semi_minor_axis / 2
        )
        # Area of the ellipse sector defined by the two points
        area_triangle_sgn = (
            np.sign(theta_2 - theta_1_hat - np.pi)
            / 2
            * np.abs(pt_1[0] * pt_2[1] - pt_2[0] * pt_1[1])
        )
        # Signed area of the triangle defined by the two point and the center of the
        # ellipse
        area_segment = area_sector + area_triangle_sgn
        # Area of the ellipse segment
        return area_segment

    def intersection_area(self, other_particle, box):
        """
        Compute the intersection area between the ellipse and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.
        """
        intersection_area = self.intersection_area_ellipse_ellipse(other_particle, box)
        # Computing the intersection area
        return intersection_area
        # Returning the intersection area

    def intersection(self, other_ellipse, box, inside=True):
        """Check if two ellipses intersect."""
        intersection_bool, _, _ = self.intersection_gjk(
            other_ellipse, box, int_only=True
        )
        # diff_in_box = self.position_center - other_ellipse.position_center
        # # Difference vector between the center of the two ellipses
        # diff_nearest_other = box * np.round(diff_in_box / box)
        # # Difference vector to the nearest image of the other particle
        # y_inter_sect = intersection_points_ellipses(
        #     self.semi_major_axis,
        #     self.semi_minor_axis,
        #     self.position_center,
        #     self.angle,
        #     other_ellipse.semi_major_axis,
        #     other_ellipse.semi_minor_axis,
        #     other_ellipse.position_center + diff_nearest_other,
        #     other_ellipse.angle,
        # )
        # if len(y_inter_sect) > 0:
        #     # There are intersection points betweeen the two neighborhoods
        #     intersection_bool = True
        # else:
        #     if inside:
        #         # Either the ellipses are disjoint or one of them is completly inside the other
        #         if self.volume >= other_ellipse.volume:
        #             # The current ellipse is larger than the other ellipse
        #             intersection_bool = self.point_inside(
        #                 other_ellipse.position_center, box
        #             )
        #         else:
        #             intersection_bool = other_ellipse.point_inside(
        #                 self.position_center, box
        #             )
        #     else:
        #         intersection_bool = False
        # intersection_bool, overlap_length, unit_vector = self.intersection_gjk(
        #     other_ellipse, box
        # )
        # # unit_vector_norm = self.intersection_vector(other_particle, box)
        # # if unit_vector_norm.dot(unit_vector) <= 0:
        # #     unit_vector *= -1
        # intersection_length = overlap_length if intersection_bool else 0
        return intersection_bool

    def generate_points_on_surface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the ellipse."""
        points_loc = np.array(
            [
                [
                    self.semi_major_axis * np.cos(theta),
                    self.semi_minor_axis * np.sin(theta),
                ]
                for theta in np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            ]
        )
        # Generating the points in the Disk's local coordinates
        if erosion_thick > 0:
            # If erosion was sepcified
            for point_ind, _ in enumerate(points_loc):
                # For each point on the surface with its corresponding homogeneous angle
                angle_normal = np.arctan2(
                    self.semi_major_axis
                    / self.semi_minor_axis
                    * points_loc[point_ind][1],
                    points_loc[point_ind][0],
                )
                # Computing the angle of the normal at the current point
                points_loc[point_ind] -= erosion_thick * np.array(
                    [np.cos(angle_normal), np.sin(angle_normal)]
                )
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        points_glob = np.array(
            [
                self.rot_mat.T.dot(point_loc) + self.position_center
                for point_loc in points_loc
            ]
        )
        # Transforming local in global coordinates
        return points_glob

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for an ellipse."""
        erosion_thickness = self.semi_minor_axis ** 2 / self.semi_major_axis
        # Semi-latus rectum
        return erosion_thickness

    def uniform_sample_ellipse(self, n_samples=1):
        """Generate uniform random sample of points inside an ellipse."""
        if self.delta != 0:
            # FIXME: NEW DELTA
            # print("WARNING!!")
            pass
        points = []
        for _ in range(n_samples):
            z = np.array([0.0, 0.0])
            z[0] = np.random.normal()
            z[1] = np.random.normal()
            r = np.random.uniform() ** (1 / 2)
            R = np.linalg.norm(z)
            x_loc = r * self.semi_major_axis * z[0] / R
            y_loc = r * self.semi_minor_axis * z[1] / R
            [x_glob, y_glob] = self.rot_mat.T.dot([x_loc, y_loc]) + self.position_center
            points.append(np.array([x_glob, y_glob]))

        return points

    def regular_sample_ellipse(self, n_samples=1):
        """Generate a regular grid of points inside the ellipse."""
        n_theta = int(np.sqrt(n_samples ** (1)))
        n_r = int(np.round(n_samples / n_theta))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r = np.linspace(0.01, 1, n_r, endpoint=True)
        points = []
        for i_theta, j_radius in [
            (i_theta, j_radius) for i_theta in theta for j_radius in r
        ]:
            [x_loc, y_loc] = [
                j_radius * self.semi_major_axis * np.cos(i_theta),
                j_radius * self.semi_minor_axis * np.sin(i_theta),
            ]
            [x_glob, y_glob] = self.rot_mat.dot([x_loc, y_loc]) + self.position_center
            points.append(np.array([x_glob, y_glob]))

        return points

    def generate_point_inside(self):
        return self.uniform_sample_ellipse()

    # def intersection_length_ellipse_ellipse(
    #     self, other_ellipse: Ellipse, box: list
    # ) -> np.array:
    #     """Intersection length between two ellipses."""
    #
    #     # Generating 3 poits belong to the support function
    #     # ----------------------------------------------------------------------------------
    #     pts = [None for _ in range(3)]
    #     for i_ind in range(3):
    #         random_dir = None
    #         pts[i_ind] = self.support_function(random_dir) - self.support_function(
    #             -random_dir
    #         )
    #
    #     # Compute coefficients for the eelipse corresponding to the support function
    #     # ----------------------------------------------------------------------------------
    #     mat = np.array(
    #         [
    #             [pts[0][0] ** 2, pts[0][1] * pts[0][0], pts[0][1] ** 2],
    #             [pts[1][0] ** 2, pts[1][1] * pts[1][0], pts[1][1] ** 2],
    #             [pts[2][0] ** 2, pts[2][1] * pts[0][0], pts[2][1] ** 2],
    #         ]
    #     )
    #     aa, bb, cc = None
    #
    #     # Compute the principals directions of the ellipse
    #     # ----------------------------------------------------------------------------------
    #     ellip_mat = np.array(
    #         [
    #             [aa, 0.5 * bb],
    #             [0.5 * bb, cc],
    #         ]
    #     )
    #     vals = np.linalg.eigh(ellip_mat)
    #
    #     # Compute erosion needed for the origin to be on the surface of the support function
    #     # ----------------------------------------------------------------------------------
    #     cos_o = vals.dot(-vec_diff)
    #     erosion_thick = a - vec_diff[0] / cos_0
    #     dir_norm = np.vec_diff[0] / (a - erosion_thick)
    #
    #     # Computing normal direction to the eroded support function at the origin
    #     # ----------------------------------------------------------------------------------
    #
    #     # Computing the intersection length
    #     # ----------------------------------------------------------------------------------
    #
    #     return None

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8
    ) -> tuple[float, np.array]:
        """
        Compute the intersection length between the Ellipse and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        Returns
        -------
        intersection_length: float
            Minimum distance allowing for the removal of the intersection.

        unit_vector: np.array
            Direction of the minimum displacement allowing for the removal of the
            intersection.
        """
        if True:
            intersection, overlap_length, unit_vector = self.intersection_gjk(
                other_particle, box, tol=tol
            )
            intersection_length = overlap_length if intersection else 0
        elif isinstance(other_particle, Ellipse):
            start = time.time()
            # The other particle is also a Ellipse
            # (
            #     _,
            #     intersection_length,
            # ) = self.intersection_length_ellipse_ellipse(other_particle, box)
            intersection = self.intersection(other_particle, box)
            unit_vector = self.intersection_vector(other_particle, box)
            # Computing the intersection length
            if intersection:
                # There is overlap
                # overlap_volume = self.intersection_volume_ellipsoid_other(
                #     other_particle, box, max_it=50, seq_size=100
                # )
                diff_in_box = self.position_center - other_particle.position_center
                # Difference vector between the center of the two particles
                diff_nearest_other = box * np.round(diff_in_box / box)
                # Vector from the position of the other ellipse to its nearest image to the current
                # ellipse
                search_direction = self.intersection_vector(other_particle, box)
                mink_diff_point = self.support_function(search_direction)[0:2] - (
                    other_particle.support_function(-search_direction)[0:2]
                    + diff_nearest_other
                )
                intersection_length = mink_diff_point.dot(search_direction)
                # _, intersection_length_2, unit_vector_2 = self.intersection_gjk(
                #     other_particle, box, tol=tol
                # )
                unit_vector = search_direction
                if intersection_length < 0:
                    intersection_length = 0
                    unit_vector = np.array([0.0, 0.0])
                # Computing the intersection area
            else:
                # There is no overlap
                intersection_length = 0
                unit_vector = np.array([0.0, 0.0])
            time_1 = time.time() - start
            start = time.time()
            _, intersection_length_2, unit_vector_2 = self.intersection_gjk(
                other_particle, box, tol=tol
            )
            time_2 = time.time() - start
            # print(time_1, time_2)
            # print(
            #     "error",
            #     intersection_length_2,
            #     intersection_length,
            #     np.abs(intersection_length_2 - intersection_length)
            #     / intersection_length_2
            #     * 100,
            # )

        else:
            intersection, overlap_length, unit_vector = self.intersection_gjk(
                other_particle, box, tol=tol
            )
            intersection_length = overlap_length if intersection else 0
        return intersection_length, unit_vector
        # Returning the intersection length

    def intersection_length_ellipse_ellipse(
        self, other_ellipse: Ellipse, box: list
    ) -> float:
        """
        Compute the intersection length between two Ellipses.

        Parameters
        ----------
        other_ellipse: `.Ellipse`
            Other ellipse whose intersection length with the current ellipse we want to know
        """
        other_ellipse_position_center_nearest_pbc = Particle.nearest_periodic_image(
            other_ellipse.position_center, self.position_center, box
        )

        intersect_pts = intersection_points_ellipses(
            self.major_axis / 2,
            self.minor_axis / 2,
            self.position_center,
            self.angle,
            other_ellipse.major_axis / 2,
            other_ellipse.minor_axis / 2,
            other_ellipse_position_center_nearest_pbc,
            other_ellipse.angle,
        )
        if len(intersect_pts) == 0 or len(intersect_pts) == 1:
            # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume >= other_ellipse.volume:
                # The current ellipse is larger than the other ellipse
                if self.point_inside(other_ellipse.position_center, box):
                    # The other ellipse is completly inside the current ellipse
                    intersection_length = other_ellipse.major_axis
                    intersection = True
                else:
                    # The ellipses are disjoint
                    intersection_length = 0
                    intersection = False
            else:
                if other_ellipse.point_inside(self.position_center, box):
                    # The current ellipse is completly inside the other ellipse
                    intersection_length = self.major_axis
                    intersection = True
                else:
                    # The ellipses are disjoint
                    intersection_length = 0
                    intersection = False
                    # The intersection area is 0
        elif len(intersect_pts) == 2:
            # The ellipses intersect in two points. The case where one of the ellipses is
            # inside the other and both are tangent at the intersection points is
            # disregarded
            intersect_pts_ord = self.sort_points_on_ellipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            midpoint_1 = self.midpoint_on_ellipse(
                intersect_pts_ord[0], intersect_pts_ord[1]
            )
            # Midpoint between the first two intersection points in the current ellipse
            if other_ellipse.point_inside(midpoint_1, box):
                midpoint_2 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[1], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[0], other_ellipse.position_center, box
                    ),
                )
                # Midpoint between the first two intersection points in the other ellipse
            else:
                midpoint_1 = self.midpoint_on_ellipse(
                    intersect_pts_ord[1], intersect_pts_ord[0]
                )
                # The midpoint we are looking for is opposite from the one computed
                midpoint_2 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[0], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[1], other_ellipse.position_center, box
                    ),
                )
                # Midpoint between the first two intersection points in the other ellipse
            # opts = [
            #     midpoint_2 - midpoint_1,
            #     intersect_pts_ord[0] - intersect_pts_ord[1],
            # ]
            # intersection_arg = np.minarg([np.linalg.norm(vec) for vec in opts])
            # intersection_length = np.linalg.norm(opts[intersection_arg])
            # unit_vector = opts[intersection_arg]
            intersection_length = np.linalg.norm(midpoint_2 - midpoint_1)
            intersection = True
        elif len(intersect_pts) == 3:
            intersection_length = 0
            # FIXME: Inconrrect result. Not very important as it almost never happens
            intersection = True
        elif len(intersect_pts) == 4:
            # One of the ellipses goes through the other
            intersect_pts_ord = self.sort_points_on_ellipse(intersect_pts)
            # Ordering the intersection points according to their angle relative to the
            # major axis of the current ellipse counter clockwise
            midpoint_1 = self.midpoint_on_ellipse(
                intersect_pts_ord[0], intersect_pts_ord[1]
            )
            # Midpoint between the first two intersection points in the current ellipse
            if other_ellipse.point_inside(midpoint_1, box):
                midpoint_2 = self.midpoint_on_ellipse(
                    intersect_pts_ord[2], intersect_pts_ord[3]
                )
                # Midpoint between the first two intersection points in the other ellipse
                midpoint_3 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[1], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[2], other_ellipse.position_center, box
                    ),
                )
                # The midpoint we are looking for is opposite from the one computed
                midpoint_4 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[3], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[0], other_ellipse.position_center, box
                    ),
                )
                # Midpoint between the first two intersection points in the other ellipse
            else:
                midpoint_1 = self.midpoint_on_ellipse(
                    intersect_pts_ord[1], intersect_pts_ord[2]
                )
                midpoint_2 = self.midpoint_on_ellipse(
                    intersect_pts_ord[3], intersect_pts_ord[0]
                )
                # Midpoint between the first two intersection points in the other ellipse
                midpoint_3 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[0], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[1], other_ellipse.position_center, box
                    ),
                )
                # The midpoint we are looking for is opposite from the one computed
                midpoint_4 = other_ellipse.midpoint_on_ellipse(
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[2], other_ellipse.position_center, box
                    ),
                    Particle.nearest_periodic_image(
                        intersect_pts_ord[3], other_ellipse.position_center, box
                    ),
                )
                # Midpoint between t
            intersection_length = np.min(
                [
                    np.linalg.norm(midpoint_2 - midpoint_1),
                    np.linalg.norm(midpoint_3 - midpoint_4),
                ]
            )
            intersection = True

        return intersection, intersection_length  # , unit_vector

    def support_function(self, direction: np.array) -> np.array:
        """Support funciton for the ellipse."""
        dir_local = self.rot_mat.dot(direction[0:2])
        dir_normal = np.array(
            [
                dir_local[0] * (self.semi_major_axis ** 2),
                dir_local[1] * (self.semi_minor_axis ** 2),
            ]
        )
        rescale_factor = np.sqrt(
            1
            / (
                (dir_normal[0] / self.semi_major_axis) ** 2
                + (dir_normal[1] / self.semi_minor_axis) ** 2
            )
        )
        dir_nomal_unit = dir_normal / np.linalg.norm(dir_normal)
        point_on_ellipse_loc = rescale_factor * dir_normal + self.delta * dir_nomal_unit
        point_on_ellipse_glob = (
            self.rot_mat.T.dot(point_on_ellipse_loc) + self.position_center
        )
        return np.append(point_on_ellipse_glob, [0])


class Disk(Ellipse):
    """
    This is the subclass of particles with the form of a circular disk.

    Attributes
    ----------
    radius: float
        Radius of the disk

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a disk, and
        their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing disks.
    """

    possible_parameters = {
        **Particle.possible_parameters,
        **{"r": ("Radius", "float"), "area": ("Area per particle", "float")},
    }
    #
    # )
    # all possible_parameters
    acceptable_descriptions = [
        {"r", "n"},
        {"r", "vf"},
        {"n", "vf"},
        {"area", "vf"},
        {"area", "n"},
    ]
    # List of acceptable collections of parameters
    dim = 2

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Disk obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        if "r" in descriptors:
            # The radius was supplied
            r = descriptors["r"]
        elif "area" in descriptors:
            # The area of each particle was supplied
            r = np.sqrt(descriptors["area"] / np.pi)
        elif "vf" in descriptors and "n" in descriptors:
            # Both the volume fraction and the number of particles was supplied
            area = descriptors["vf"] * rve_dims[0] * rve_dims[1] / descriptors["n"]
            # Area of each particle (all the same)
            r = np.sqrt(area / np.pi)
        descriptors_ellipse = {"major_axis": 2 * r, "minor_axis": 2 * r, "angle": 0}
        super().__init__(phase, descriptors_ellipse, rve_dims)

    def generate_points_on_surface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the Disk."""
        points_loc = np.array(
            [
                [self.radius * np.cos(theta), self.radius * np.sin(theta)]
                for theta in np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            ]
        )
        # Generating the points in the Disk's local coordinates
        points_glob = points_loc + self.position_center
        # Transforming local in global coordinates
        if erosion_thick > 0:
            # If erosion was sepcified
            for (point_ind, _), theta in zip(
                enumerate(points_glob),
                np.linspace(0, 2 * np.pi, n_points, endpoint=False),
            ):
                # For each point on the surface with its corresponding homogeneous angle
                points_glob[point_ind] -= erosion_thick * np.array(
                    [np.cos(theta), np.sin(theta)]
                )
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        return points_glob

    def intersection_area(self, other_particle: Particle, box: list) -> float:
        """
        Compute the intersection area between the disk and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        """
        if isinstance(other_particle, (Disk, CylindricalFiber)):
            # The other particle is also a Disk
            intersection_area = self.intersection_area_disk_disk(other_particle, box)
            # Computing the intersection area
        elif isinstance(other_particle, Ellipse):
            other_particle: Ellipse
            # The other particle is an Ellipse
            intersection_area = other_particle.intersection_area_ellipse_ellipse(
                self, box
            )
            # Computing the intersection area
        return intersection_area
        # Returning the intersection area

    def intersection_area_disk_disk(self, other_disk, box):
        """Compute the intersection area between two disks."""
        diff_center = self.position_center - other_disk.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector from the current disk to the nearest image of the other disk
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the center of the disks
        if self.radius >= other_disk.radius:
            # The radius of the self is larger than the radius of the other disk
            r_1 = self.radius
            # Disk 1 is the disk with the larger radius
            r_2 = other_disk.radius
            # Disk 2 is the disk with the smaller radius
        else:
            # The radius of the other disk is larger than the radius of the self
            r_1 = other_disk.radius
            # Disk 1 is the disk with the larger radius
            r_2 = self.radius
            # Disk 2 is the disk with the smaller radius
        if d >= (r_1 + r_2):
            # The disks intersect at most at one point
            intersection_area = 0
            # The intersection area of the disks is zero
        elif d <= r_1 - r_2:
            # Disk 2 is interely contained within Disk 1
            intersection_area = np.pi * r_2 ** 2
            # The intersection area is equal to the area of the smaller disk, Disk 2
        else:
            d_1 = (r_1 ** 2 - r_2 ** 2 + d ** 2) / (2 * d)
            # x coordinate of the intersection point of the two disks if the the origin is
            # at disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_area = (
                r_1 ** 2 * np.arccos(d_1 / r_1)
                - d_1 * np.sqrt(r_1 ** 2 - d_1 ** 2)
                + r_2 ** 2 * np.arccos(d_2 / r_2)
                - d_2 * np.sqrt(r_2 ** 2 - d_2 ** 2)
            )
            # Computing the intersection area
        return intersection_area
        # Returning the intersection area

    def intersection_sqrt(self, other_sphere, box):
        """
        Compute the intersection volume between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
            Other sphere whose intersection volume with the current sphere we want to know
        """
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        d = np.linalg.norm(diff_center)
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_sphere.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = other_sphere.radius
            # Sphere 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_sphere.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = self.radius
            # Sphere 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_volume = 0
            # The intersection area of the spheres is zero
        elif d <= r_1 - r_2:
            # Sphere 2 is interely contained within Sphere 1
            # intersection_volume = r_1 + r_2  # 4 / 3 * np.pi * r_2 ** 3
            # intersection_volume = 4 / 3 * np.pi * r_2 ** 3
            intersection_volume = r_2
            # The intersection area is equal to the area of the smaller sphere, Sphere 2
        else:
            d_1 = (r_1 ** 2 - r_2 ** 2 + d ** 2) / (2 * d)
            # x coordinate of the intersection point of the two disks if the the origin is
            # at disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_volume = r_2 * (
                1 - (d) ** 2 / (r_1 + r_2) ** 2
            )  # / (2 * r_2) * 4 / 3 * np.pi * r_2 ** 3
            # intersection_volume = (
            #     r_1 ** 3
            #     / 3
            #     * 2
            #     * np.pi
            #     * (1 - d_1 / r_1)  # Volume of spherical sector (Sphere 1)
            #     - d_1 * (r_1 ** 2 - d_1 ** 2) * np.pi / 3  # Volume of cone (Sphere 1)
            #     + r_2 ** 3
            #     / 3
            #     * 2
            #     * np.pi
            #     * (1 - d_2 / r_2)  # Volume of shperical sector (Sphere 2)
            #     - d_2 * (r_2 ** 2 - d_2 ** 2) * np.pi / 3
            # )  # Volume of cone (Sphere 2)
            # Computing the intersection area as the sum of the spherical caps minus the
            # corresponding cones
            # intersection_volume = 0.01
        return intersection_volume
        # Returning the intersection area

    def intersection(self, other_particle: Particle, box: list, inside=True) -> bool:
        """Check if the Disk intersects the other_particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.
        """
        if isinstance(other_particle, (Disk, CylindricalFiber)):
            # The other particle is also a Disk
            intersection = self.intersection_disk_disk(
                other_particle, box, inside=inside
            )
            # Computing the intersection area
        elif isinstance(other_particle, Ellipse):
            # The other particle is an Ellipse
            other_particle: Ellipse
            intersection = other_particle.intersection(self, box)
            # Computing the intersection area
        return intersection
        # Returning the intersection area

    def point_inside(self, point, box):
        """Check if some point is inside the Disk."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        point_in = (
            np.linalg.norm(self.position_center - point_nearest_pbc) <= self.radius
        )

        return point_in

    def intersection_disk_disk(self, other_disk: Disk, box: list, inside=True) -> bool:
        """Check if two Disks intersect."""
        diff_center = self.position_center - other_disk.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        distance_disks = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        if inside:
            # Being completly inside is considered as an intersection
            intersection_bool = distance_disks < (self.radius + other_disk.radius)
        else:
            # Being completly inside is not considered as an intersection
            intersection_bool = distance_disks < (
                self.radius + other_disk.radius
            ) and distance_disks > np.abs(self.radius - other_disk.radius)

        return intersection_bool

    @property
    def volume(self):
        """Volume/area of the disk."""
        volume = np.pi * self.radius ** 2

        return volume

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for a disk."""
        erosion_thickness = self.radius
        return erosion_thickness

    def support_function(self, direction):
        radius_vec = direction[0:2] / np.linalg.norm(direction[0:2]) * self.radius
        return np.append(self.position_center + radius_vec, [0])
        # GJK algorithm is written for 3D

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8
    ) -> tuple[float, np.array]:
        """
        Compute the intersection length between the Disk and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        Returns
        -------
        intersection_length: float
            Minimum distance allowing for the removal of the intersection.

        unit_vector: np.array
            Direction of the minimum displacement allowing for the removal of the
            intersection.
        """
        if isinstance(other_particle, Disk):
            # The other particle is also a Disk
            intersection_length = self.intersection_length_disk_disk(
                other_particle, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
            # Computing the intersection length
        elif isinstance(other_particle, Ellipse) and False:
            # The other particle is a ellipse
            other_particle: Ellipse
            _, intersection_length = other_particle.intersection_ellipse_ellipse(
                other_particle, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
        else:
            intersection, overlap_length, unit_vector = self.intersection_gjk(
                other_particle, box, tol=tol
            )
            intersection_length = overlap_length if intersection else 0
        return intersection_length, unit_vector
        # Returning the intersection length

    def intersection_length_disk_disk(self, other_disk: Disk, box: list) -> float:
        """
        Compute the intersection length between two Disks.

        Parameters
        ----------
        other_disk: `.Disk`
            Other sphere whose intersection length with the current sphere we want to know
        """
        d = np.linalg.norm(
            Particle.nearest_periodic_image(
                self.position_center, other_disk.position_center, box
            )
            - other_disk.position_center
        )
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_disk.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Disk 1 is the sphere with the larger radius
            r_2 = other_disk.radius
            # Disk 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_disk.radius
            # Disk 1 is the sphere with the larger radius
            r_2 = self.radius
            # Disk 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_length = 0
            # The intersection length of the spheres is zero
        elif d <= r_1 - r_2:
            # Disk 2 is interely contained within Disk 1
            intersection_length = 2 * r_2
            # The intersection length is the diameter of the smaller sphere
        else:
            intersection_length = r_1 + r_2 - d
            # intersection length
        return intersection_length
        # Returning the intersection length


class CylindricalFiber(Disk):
    """
    This is the subclass of particles with the form of a circular disk.

    Attributes
    ----------
    direction_fibers: {0, 1, 2}
        Direction in which the fibers run. 'x':0, 'y':1 and 'z':2

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a disk, and
        their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing disks.

    dim: int
        Dimension that the particle inhabits
    """

    possible_parameters = {
        **{
            "r": ("Radius", "float"),
            "area": ("Area per particle", "float"),
            "direction": ("Fiber direction", "int"),
        },
        **Particle.possible_parameters,
    }
    # all possible_parameters
    acceptable_descriptions = [
        {"r", "n", "direction"},
        {"r", "vf", "direction"},
        {"n", "vf", "direction"},
        {"area", "vf", "direction"},
        {"area", "n", "direction"},
    ]
    dim = 3
    # List of acceptable collections of parameters

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Ellipse obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        self.direction_fibers = descriptors.pop("direction")
        # Integer giving the direction of the fibers
        self.length_dir_fibers = rve_dims[self.direction_fibers]
        # Setting the size of the simulation box
        box = list(rve_dims)
        del box[self.direction_fibers]
        super().__init__(phase, descriptors, box)
        # Using the constructor of the parent class

    @property
    def volume(self):
        """Volume of the cylindrical fiber."""
        volume = np.pi * self.radius ** 2 * self.length_dir_fibers

        return volume


class Ellipsoid(Particle):
    """
    This is the subclass of particles with the form of an ellipsoid.

    Attributes
    ----------
    axis_1: float
        Principal axis along xx before aplying the rotation.

    axis_2: float
        Principal axis along yy before aplying the rotation.

    axis_3: float
        Principal axis along zz before aplying the rotation.

    rotation_axis: array
        Rotation axis used to characterize the orientation of the Ellipsoid.

    angle: float
        Angle of rotation around the rotation axis.

    rot_quad: array
        Rotation quaternion

    rotation_mat: array
        Rotation matrix from local to global coordinates

    radius: float
        Radius of the circumscribed sphere.

    radius_insc: float
        Radius of the inscribed sphere.

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a disk, and
        their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing disks.

    dim: int
        Dimension that the particle inhabits
    """

    possible_parameters = {
        **{
            "axis_1": ("Axis 1", "float"),
            "semi_axis_1": ("Semi-Axis 1", "float"),
            "axis_2": ("Axis 2", "float"),
            "axis_3": ("Axis 3", "float"),
            "rot_axis_comp_x": ("x-component rotation axis", "float"),
            "rot_axis_comp_y": ("y-component rotation axis", "float"),
            "rot_axis_comp_z": ("z-component rotation axis", "float"),
            "angle": ("Rotation angle", "float"),
            "ratio_12": ("Ratio a1/a2", "float"),
            "ratio_21": ("Ratio a2/a1", "float"),
            "ratio_13": ("Ratio a1/a3", "float"),
            "ratio_32": ("Ratio a3/a2", "float"),
            "ratio_321": ("Ratio a3/a1 and a2/a1", "float"),
            "p_3": ("Angle a1 makes with XY", "float"),
            "phi_z": ("Angle that the projection of a1 in XY makes with Y", "float"),
        },
        **Particle.possible_parameters,
    }
    # all possible_parameters
    acceptable_descriptions = [
        {
            "axis_1",
            "axis_2",
            "axis_3",
            "rot_axis_comp_x",
            "rot_axis_comp_y",
            "rot_axis_comp_z",
            "angle",
            "n",
        },
        {
            "axis_1",
            "axis_2",
            "axis_3",
            "rot_axis_comp_x",
            "rot_axis_comp_y",
            "rot_axis_comp_z",
            "angle",
            "vf",
        },
        {
            "axis_1",
            "ratio_12",
            "ratio_13",
            "rot_axis_comp_x",
            "rot_axis_comp_y",
            "rot_axis_comp_z",
            "angle",
            "vf",
        },
        {
            "vf",
            "n",
            "ratio_12",
            "ratio_13",
            "rot_axis_comp_x",
            "rot_axis_comp_y",
            "rot_axis_comp_z",
            "angle",
        },
        {"vf", "semi_axis_1", "ratio_321", "p_3", "phi_z"},
        {"vf", "semi_axis_1", "ratio_32", "ratio_21", "p_3", "phi_z"},
        {"vf", "axis_2", "ratio_32", "ratio_21", "p_3", "phi_z"},
        {"vf", "axis_1", "ratio_321", "p_3", "phi_z"},
    ]
    # List of acceptable collections of parameters
    dim = 3

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Ellipse obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        if "axis_1" in descriptors and "axis_2" in descriptors and "axis_3":
            # All axis were supplied
            axis_1 = descriptors["axis_1"]
            axis_2 = descriptors["axis_2"]
            axis_3 = descriptors["axis_3"]
        if (
            "ratio_12" in descriptors
            and "ratio_13" in descriptors
            and "vf" in descriptors
            and "n" in descriptors
        ):
            volume = (
                descriptors["vf"]
                * rve_dims[0]
                * rve_dims[1]
                * rve_dims[2]
                / descriptors["n"]
            )
            axis_1 = np.cbrt(
                volume
                * descriptors["ratio_12"]
                * descriptors["ratio_13"]
                * 8
                / (np.pi * 4 / 3)
            )
            axis_2 = axis_1 / descriptors["ratio_12"]
            axis_3 = axis_1 / descriptors["ratio_13"]
        if (
            "ratio_12" in descriptors
            and "ratio_13" in descriptors
            and "axis_1" in descriptors
        ):
            axis_1 = descriptors["axis_1"]
            axis_2 = axis_1 / descriptors["ratio_12"]
            axis_3 = axis_1 / descriptors["ratio_13"]
        if "ratio_321" in descriptors and "semi_axis_1" in descriptors:
            axis_1 = 2 * descriptors["semi_axis_1"]
            axis_2 = descriptors["ratio_321"] * axis_1
            axis_3 = descriptors["ratio_321"] * axis_1
        if "ratio_321" in descriptors and "axis_1" in descriptors:
            axis_1 = descriptors["axis_1"]
            axis_2 = descriptors["ratio_321"] * axis_1
            axis_3 = descriptors["ratio_321"] * axis_1
        if (
            "ratio_32" in descriptors
            and "ratio_21" in descriptors
            and "semi_axis_1" in descriptors
        ):
            axis_1 = 2 * descriptors["semi_axis_1"]
            axis_2 = descriptors["ratio_21"] * axis_1
            axis_3 = descriptors["ratio_32"] * axis_2
        if (
            "ratio_32" in descriptors
            and "ratio_21" in descriptors
            and "axis_2" in descriptors
        ):
            # file_path = (
            #     "/home/jose/Documents/code/paper_results/stat_analysis/3D/Results.csv"
            # )
            # info = np.genfromtxt(file_path, delimiter=",", skip_header=1)
            # visible_vars = info[:, 7:9] / 795
            # # print(visible_vars)
            # angles = info[:, -2] * np.pi / 180
            # # for i_ind, i_angle in angles:
            # #     if i_angle > n
            # visible_vars = np.array([visible_vars[:, 1], visible_vars[:, 0]]).T
            # ind = np.random.choice(np.arange(len(visible_vars[:, 1])))
            # axis_2 = visible_vars[:, 1][ind]
            # axis_3 = visible_vars[:, 0][ind]

            axis_2 = min(descriptors["axis_2"], 0.2)
            axis_1 = axis_2 / max(min(descriptors["ratio_21"], 1), 0.4)
            axis_3 = max(min(descriptors["ratio_32"], 1), 0.4) * axis_2
        if "angle" in descriptors:
            angle = descriptors["angle"]
        if (
            "rot_axis_comp_x" in descriptors
            and "rot_axis_comp_y" in descriptors
            and "rot_axis_comp_z" in descriptors
        ):
            # Euler angles
            rot_axis_comp_x = descriptors["rot_axis_comp_x"]
            rot_axis_comp_y = descriptors["rot_axis_comp_y"]
            rot_axis_comp_z = descriptors["rot_axis_comp_z"]
        if "p_3" in descriptors and "phi_z" in descriptors:
            p_3 = descriptors["p_3"]
            phi_z = descriptors["phi_z"]
            # phi_z = angles[ind]

            rot_mat_y = np.array(
                [
                    [np.cos(p_3), 0, -np.sin(p_3)],
                    [
                        0,
                        1,
                        0,
                    ],
                    [np.sin(p_3), 0, np.cos(p_3)],
                ],
            )
            rot_mat_z = np.array(
                [
                    [np.cos(phi_z), -np.sin(phi_z), 0],
                    [np.sin(phi_z), np.cos(phi_z), 0],
                    [0, 0, 1],
                ],
            )
            complete_rot_mat = rot_mat_z.dot(rot_mat_y)
            rot_axis_comp_x = complete_rot_mat[2, 1] - complete_rot_mat[1, 2]
            rot_axis_comp_y = complete_rot_mat[0, 2] - complete_rot_mat[2, 0]
            rot_axis_comp_z = complete_rot_mat[1, 0] - complete_rot_mat[0, 1]
            angle = np.arccos((np.trace(complete_rot_mat) - 1) / 2)

        self.axis_1 = np.abs(axis_1)
        self.axis_2 = np.abs(axis_2)
        self.axis_3 = np.abs(axis_3)
        self.rotation_axis = np.array(
            [rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z]
        ) / np.linalg.norm(
            np.array([rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z])
        )

        self.angle = angle
        self.rot_quat = np.array(
            [
                np.cos(angle / 2),
                np.sin(angle / 2) * self.rotation_axis[0],
                np.sin(angle / 2) * self.rotation_axis[1],
                np.sin(angle / 2) * self.rotation_axis[2],
            ]
        )

        q = self.rot_quat
        self.rotation_mat = np.array(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                    2 * (q[1] * q[2] - q[3] * q[0]),
                    2 * (q[1] * q[3] + q[2] * q[0]),
                ],
                [
                    2 * (q[1] * q[2] + q[3] * q[0]),
                    1 - 2 * (q[1] ** 2 + q[3] ** 2),
                    2 * (q[2] * q[3] - q[1] * q[0]),
                ],
                [
                    2 * (q[1] * q[3] - q[2] * q[0]),
                    2 * (q[2] * q[3] + q[1] * q[0]),
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                ],
            ]
        )
        # Rotation matrix from local to global coordinates
        super().__init__(3, phase)

    @property
    def volume(self):
        """Volume of the ellipsoid."""
        volume = 4 / 3 * np.pi * self.semi_axis_1 * self.semi_axis_2 * self.semi_axis_3

        return volume

    @property
    def radius(self):
        """Radius of the circumscribed sphere to the ellipsoid."""
        radius = (
            np.max([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3]) + self.delta
        )
        # Radius of the circunscribed sphere

        return radius

    @property
    def radius_insc(self):
        """Radius of the inscribed circle to the ellipsoid."""
        radius_insc = np.min([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3])

        return radius_insc

    @property
    def semi_axis_1(self):
        """Semi principal axis along xx before aplying the rotation."""
        semi_axis_1 = self.axis_1 / 2
        # Radius of the circunscribed sphere

        return semi_axis_1

    @property
    def semi_axis_2(self):
        """Semi principal axis along yy before aplying the rotation."""
        semi_axis_2 = self.axis_2 / 2
        # Radius of the circunscribed sphere

        return semi_axis_2

    @property
    def semi_axis_3(self):
        """Semi principal axis along zz before aplying the rotation."""
        semi_axis_3 = self.axis_3 / 2
        # Radius of the circunscribed sphere

        return semi_axis_3

    def contract(self, distance):
        """Contract the particle."""
        # self.axis_1 -= 2 * distance
        # self.axis_2 -= 2 * distance
        # self.axis_3 -= 2 * distance
        self.delta -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        # self.axis_1 += 2 * distance
        # self.axis_2 += 2 * distance
        # self.axis_3 += 2 * distance
        self.delta += distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def M(self):
        """Get transformation matrix in homogeneous coordinates."""
        M = np.concatenate(
            (
                np.concatenate(
                    (self.rotation_mat, np.array([self.position_center]).T), axis=1
                ),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
            ),
            axis=0,
        )
        return M

    def M_inv(self, diff_nearest=np.array([0.0, 0.0, 0.0])):
        """Get the inverse of the transformation matrix in homogeneous coordinates."""
        M_inv = np.concatenate(
            (
                np.concatenate(
                    (
                        self.rotation_mat.T,
                        np.array(
                            [
                                -self.rotation_mat.T.dot(
                                    self.position_center + diff_nearest
                                )
                            ]
                        ).T,
                    ),
                    axis=1,
                ),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
            ),
            axis=0,
        )
        return M_inv

    def A_glob(self, diff_nearest=np.array([0.0, 0.0, 0.0])):
        """Auxiliar matrix to determine intersection points."""
        A_loc = np.array(
            [
                [1.0 / self.semi_axis_1 ** 2, 0.0, 0.0, 0.0],
                [0.0, 1.0 / self.semi_axis_2 ** 2, 0.0, 0.0],
                [0.0, 0.0, 1.0 / self.semi_axis_3 ** 2, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        A_glob = self.M_inv(diff_nearest).T.dot(A_loc.dot(self.M_inv(diff_nearest)))
        return A_glob

    def point_inside(self, point, box, tol=1e-6, position="inside"):
        """
        Check if a given point is inside the ellipsoid.

        This function determines if the point is inside, outside or on the ellipse given a
        tolerance.

        Parameters
        ----------
        self: `.Ellipsoid`
            Ellipsoid under analysis

        point: array
            Point under analysis

        tol: float
            Tolerance

        position: string
            'inside' or 'on'

        Returns
        -------
        point_in: bool
            True if the point is inside the ellipse and False otherwise.
        """
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        # Nearest periodic imagine of *point* to *self.position_center*
        rot_mat_l_g = self.rotation_mat
        # Rotation matrix from local to global coordinates
        point_loc = rot_mat_l_g.T.dot(point_nearest_pbc - self.position_center)
        # Point in local coordinates
        if position == "inside":
            # Checking if the point is inside the ellipse
            point_in = (
                point_loc[0] ** 2 / self.semi_axis_1 ** 2
                + point_loc[1] ** 2 / self.semi_axis_2 ** 2
                + point_loc[2] ** 2 / self.semi_axis_3 ** 2
                - 1
                <= tol
            )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
        elif position == "on":
            # Checking if the point is on the ellipse
            point_in = (
                np.abs(
                    point_loc[0] ** 2 / self.semi_axis_1 ** 2
                    + point_loc[1] ** 2 / self.semi_axis_2 ** 2
                    + point_loc[2] ** 2 / self.semi_axis_3 ** 2
                    - 1
                )
                <= tol
            )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
        return point_in

    def intersection_volume_ellipsoid_other(
        self, other_particle, box, alg_type="random", tol=1, max_it=1000, seq_size=50
    ):
        """
        Compute the overlap volume between this ellipsoid and another particle.

        This function uses Monte Carlo method to obtain the overlap volume, generating
        random points inside the current ellipsoid and checking if they also belong to the
        other particle. The overlap volume is proportional to the probability of the point
        being inside both particles.

        Parameters
        ----------
        self: `.Ellipsoid`
            Current ellipsoid

        other_particle: `.Particle`
            Other particle.

        alg_type: {'random', 'regular'}, optional
            Integration method.
            'random' - Monte Carlo method
            'regular' - Quadrature (scipy)

        tol: float, optional
            Tolerance for the error estimate.

        max_it: integer, optional
            Maximum number of iterations.

        seq_size: integer, optional
            Number of points used to estimate the overlap volume at each iteration

        Returns
        -------
        overlap_volume: float
            Overlap volume of the two particles.
        """
        diff_in_box = self.position_center - other_particle.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Vector between the other particle in the RVE to its nearest image to the current
        # ellipsoid
        if alg_type == "random":
            k_iteration = 0
            # Initializing the iteration counter
            overlap_volume_est = []
            # Vector of the overlap volume estimates
            error = 10
            # Initializing the error
            while (error > tol) and (k_iteration < max_it):
                # Run the Monte Carlo method while the error estimate is larger than the
                # tolerance
                # and the number of iterations is smaller than the maximm allowed number of
                # iterations
                total_n_points = 0
                points_inside = 0
                # Initializing the counters for the number of points generated and the
                # number of points inside both volumes
                for _ in range(seq_size):
                    # Generating seq_size points
                    total_n_points += 1
                    # Counting the generated points
                    point = self.generate_point_inside()
                    # Generating a random point inside the current ellipsoid
                    if other_particle.point_inside(point - diff_nearest_other, box):
                        # If the generated point is inside the volume of the other particle
                        points_inside += 1
                        # Counting the points inside both particles
                overlap_volume_est.append(self.volume * points_inside / total_n_points)
                # Estimation for the overlap volume
                k_iteration += 1
                # Increasing the iteration couter
                if k_iteration > 2:
                    # If there are more than 2 estimations
                    overlap_volume = np.mean(overlap_volume_est)
                    error = (
                        np.std(overlap_volume_est)
                        / np.sqrt(len(overlap_volume_est))
                        / overlap_volume
                        * 100
                    )
                    # Estimation and error computed assuming that each iteration is
                    # independent from the last and follow a normal distribution
        elif alg_type == "regular":
            A = self.semi_axis_1
            B = self.semi_axis_2
            C = self.semi_axis_3

            def pointsInside(x, y, z):
                pointIn = other_particle.point_inside(
                    self.rotation_mat.dot([x, y, z])
                    + self.position_center
                    - diff_nearest_other,
                    box,
                )
                if pointIn:
                    value = 1
                else:
                    value = 0
                return value

            (overlap_volume, _) = integrate.tplquad(
                pointsInside,
                -A,
                A,
                lambda x: -B * np.sqrt(1 - x ** 2 / A ** 2),
                lambda x: B * np.sqrt(1 - x ** 2 / A ** 2),
                lambda x, y: -C * np.sqrt(1 - x ** 2 / A ** 2 - y ** 2 / B ** 2),
                lambda x, y: C * np.sqrt(1 - x ** 2 / A ** 2 - y ** 2 / B ** 2),
                epsrel=0.1,
            )

        return np.round(overlap_volume, decimals=5)

    def generate_regular_grid(self, n_samples):
        """Generate a regular sample of points in the ellipsoid."""
        n_theta = int(np.sqrt(n_samples ** (1)))
        n_phi = int(np.cbrt(n_samples ** (1)))
        # Number of sample points for the angle
        n_r = int(np.round(n_samples / n_theta / n_phi))
        # Number of sample points for the radius. Muliplied by the number of points for the
        # angle gives the number of sample points
        radius = (np.linspace(0.01, 1, n_r, endpoint=True)) ** (1 / 3)
        theta = np.linspace(0, np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        # Regularly and uniformly sampling the angle and the radius
        x_samples = []
        for i_theta in theta:
            for j_phi in phi:
                for k_radius in radius:
                    x_loc = np.array(
                        [
                            k_radius
                            * self.semi_axis_1
                            * np.sin(i_theta)
                            * np.cos(j_phi),
                            k_radius
                            * self.semi_axis_2
                            * np.sin(i_theta)
                            * np.sin(j_phi),
                            k_radius * self.semi_axis_3 * np.cos(i_theta),
                        ]
                    )
                    x_glob = self.rotation_mat.dot(x_loc) + self.position_center
                    x_samples.append(x_glob)
        return x_samples

    def generate_point_inside(self):
        """Generate a random point inside the ellipsoid."""
        if self.delta != 0:
            # FIXME:new delta
            # print("Warning")
            pass
        w = np.random.normal(size=3)
        # Generating 3 independent random points from the standard Gaussian distribution
        r = np.random.uniform() ** (1 / 3)
        # Sampling the "radius"
        R = np.linalg.norm(w)
        x_loc = np.array(
            [
                r * (self.semi_axis_1 + self.delta) * w[0] / R,
                r * (self.semi_axis_2 + self.delta) * w[1] / R,
                r * (self.semi_axis_3 + self.delta) * w[2] / R,
            ]
        )

        x_glob = self.rotation_mat.dot(x_loc) + self.position_center
        return x_glob

    def intersection_area(self, other_particle: Particle, box: list) -> float:
        """Compute the intersection between the ellipsoid and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.

        Returns
        -------
        overlap_volume: float
            Overlap volume(area) between the ellipsoid and the other particle.
        """

        if False:  # isinstance(other_particle, Ellipsoid):
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
            # Saving the class name of the other particle as a string
            if intersection:
                # There is overlap
                overlap_volume = self.intersection_volume_ellipsoid_other(
                    other_particle, box, max_it=50, seq_size=100
                )
                # Computing the intersection area
            else:
                # There is no overlap
                overlap_volume = 0
        elif isinstance(other_particle, (Cylinder, Ellipsoid)):
            intersection, overlap_length = self.intersection_gjk(other_particle, box)
            overlap_volume = overlap_length if intersection else 0
        return overlap_volume

    def intersection_ellipsoid_ellipsoid(self, other_ellipsoid, box):
        """Check if the current and the other ellipsoid intersect."""

        def coefficients_characteristic_equation(M_i, axis_lengths, A_j):
            """
            Compute coefficients of the characteristic equation for ellipsoids i and j.

            Parameters
            ----------
            M_i: 4-array
            Rotation and translation matrix from local to global homogeneous coordinates of
            ellipsoid 1.

            axis_lengths: list
            List containing the length of the semi principal axes of ellipsoid 1.

            A_j: 4-array
            Global homogeneous characteristic matrix of ellipsoid j.

            Returns
            -------
            p: list
                Coefficients of the characteristic equation with p[0] the coefficient
                relative to the term of 4th order.
            """
            C = M_i.T.dot(A_j.dot(M_i))
            # Saving the auxiliar matrix C
            [a, b, c] = axis_lengths
            delta_1 = (1 / a) ** 2
            delta_2 = (1 / b) ** 2
            delta_3 = (1 / c) ** 2
            # Defining the auxiliar parameters delta_1, delta_2 and delta_3
            p_1 = -delta_1 * delta_2 * delta_3
            p_2 = -(
                delta_2 * delta_3 * C[0, 0]
                + delta_1 * delta_3 * C[1, 1]
                + delta_1 * delta_2 * C[2, 2]
                - delta_1 * delta_2 * delta_3 * C[3, 3]
            )
            p_3 = (
                delta_1 * delta_2 * (C[2, 2] * C[3, 3] - C[2, 3] * C[3, 2])
                + delta_2 * delta_3 * (C[0, 0] * C[3, 3] - C[0, 3] * C[3, 0])
                + delta_1 * delta_3 * (C[1, 1] * C[3, 3] - C[1, 3] * C[3, 1])
                + delta_1 * (C[1, 2] * C[2, 1] - C[1, 1] * C[2, 2])
                + delta_2 * (C[0, 2] * C[2, 0] - C[0, 0] * C[2, 2])
                + delta_3 * (C[0, 1] * C[1, 0] - C[0, 0] * C[1, 1])
            )
            p_4 = (
                delta_1
                * (
                    C[1, 1] * C[2, 2] * C[3, 3]
                    - C[1, 1] * C[2, 3] * C[3, 2]
                    - C[2, 2] * C[3, 1] * C[1, 3]
                    - C[3, 3] * C[2, 1] * C[1, 2]
                    + C[2, 1] * C[1, 3] * C[3, 2]
                    + C[3, 1] * C[1, 2] * C[2, 3]
                )
                + delta_2
                * (
                    C[0, 0] * C[2, 2] * C[3, 3]
                    - C[0, 0] * C[2, 3] * C[3, 2]
                    - C[2, 2] * C[0, 3] * C[3, 0]
                    - C[3, 3] * C[0, 2] * C[2, 0]
                    + C[2, 0] * C[0, 3] * C[3, 2]
                    + C[3, 0] * C[0, 2] * C[2, 3]
                )
                + delta_3
                * (
                    C[0, 0] * C[1, 1] * C[3, 3]
                    - C[0, 0] * C[1, 3] * C[3, 1]
                    - C[1, 1] * C[0, 3] * C[3, 0]
                    - C[3, 3] * C[0, 1] * C[1, 0]
                    + C[1, 0] * C[0, 3] * C[3, 1]
                    + C[3, 0] * C[0, 1] * C[1, 3]
                )
                + C[0, 0] * C[1, 2] * C[2, 1]
                + C[1, 1] * C[0, 2] * C[2, 0]
                + C[2, 2] * C[0, 1] * C[1, 0]
                - C[0, 0] * C[1, 1] * C[2, 2]
                - C[1, 0] * C[0, 2] * C[2, 1]
                - C[2, 0] * C[0, 1] * C[1, 2]
            )
            p_5 = np.linalg.det(A_j)
            # Obtaining the coefficients
            return [p_1, p_2, p_3, p_4, p_5]

        def coefficients_eta(p_1, p_2, p_3, p_4, p_5):
            p_1_bar = p_2 / (4 * p_1)
            p_2_bar = p_3 / (6 * p_1)
            p_3_bar = -p_4 / (4 * p_1)
            p_4_bar = p_5 / p_1

            beta_1 = (p_4_bar - p_1_bar * p_3_bar) + 3 * (
                p_2_bar ** 2 - p_1_bar * p_3_bar
            )
            beta_2 = (
                -p_3_bar * (p_3_bar - p_1_bar * p_2_bar)
                - p_4_bar * (p_1_bar ** 2 - p_2_bar)
                - p_2_bar * (p_2_bar ** 2 - p_1_bar * p_3_bar)
            )

            eta_1 = beta_1 ** 3 - 27 * beta_2 ** 2
            eta_2 = (
                -9 * (p_3_bar - p_1_bar * p_2_bar) ** 2
                + 27 * (p_1_bar ** 2 - p_2_bar) * (p_2_bar ** 2 - p_1_bar * p_3_bar)
                - 3 * (p_4_bar - p_1_bar * p_3_bar) * (p_1_bar ** 2 - p_2_bar)
            )
            eta_3 = beta_1 * (p_3_bar - p_1_bar * p_2_bar) - 3 * p_1_bar * beta_2
            eta_4 = -(p_4_bar - p_1_bar * p_3_bar)
            eta_5 = p_1_bar ** 2 - p_2_bar
            return [eta_1, eta_2, eta_3, eta_4, eta_5]

        diff_in_box = self.position_center - other_ellipsoid.position_center
        diff_nearest = box * np.round(diff_in_box / box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere

        p = coefficients_characteristic_equation(
            self.M(),
            [self.semi_axis_1, self.semi_axis_2, self.semi_axis_3],
            other_ellipsoid.A_glob(diff_nearest),
        )
        # Obtaining the coefficients of the characteristic equation
        # det(\lambda*A_i + A_j) = 0
        eta = coefficients_eta(p[0], p[1], p[2], p[3], p[4])
        # Obtaining the related coefficients eta
        cond_sep_1 = eta[0] == 0 and eta[1] > 0 and eta[2] > 0 and eta[4] > 0
        cond_sep_2 = eta[0] > 0 and eta[1] > 0 and eta[4] > 0
        cond_tan_1 = eta[0] == 0 and eta[1] > 0 and eta[2] < 0 and eta[4] > 0
        cond_tan_2 = eta[0] == 0 and eta[1] == 0 and eta[3] < 0 and eta[4] > 0
        # Computing the separation and tangent conditions from the eta coefficients
        intersect = not (cond_sep_1 or cond_sep_2 or cond_tan_1 or cond_tan_2)

        return intersect

    def generate_points_on_surface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the ellipse."""
        theta = np.linspace(np.pi / (n_points + 1), np.pi, n_points, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Using the convention from physics for the angles
        points_loc = []
        for i_theta in theta:
            for j_phi in phi:
                points_loc.append(
                    [
                        self.semi_axis_1 * np.sin(i_theta) * np.cos(j_phi),
                        self.semi_axis_2 * np.sin(i_theta) * np.sin(j_phi),
                        self.semi_axis_3 * np.cos(i_theta),
                    ]
                )
        # Generating the points in the Disk's local coordinates
        if erosion_thick > 0:
            # If erosion was sepcified
            for point_ind, i_point in enumerate(points_loc):
                # For each point on the surface with its corresponding homogeneous angle
                normal_vec = np.array(
                    [
                        i_point[0] / self.semi_axis_1 ** 2,
                        i_point[1] / self.semi_axis_2 ** 2,
                        i_point[2] / self.semi_axis_3 ** 2,
                    ]
                )
                unit_normal = normal_vec / np.linalg.norm(normal_vec)
                # Outward unit normal
                points_loc[point_ind] -= erosion_thick * unit_normal
                # Translation of the point in the normal direction to the surface by the
                # specified thickness (erosion)
        points_glob = np.array(
            [
                self.rotation_mat.dot(point_loc) + self.position_center
                for point_loc in points_loc
            ]
        )
        # Transforming local in global coordinates
        return points_glob

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for an ellipse."""
        smallest_semi_axis = np.min(
            [self.semi_axis_1, self.semi_axis_2, self.semi_axis_3]
        )
        largest_semi_axis = np.max(
            [self.semi_axis_1, self.semi_axis_2, self.semi_axis_3]
        )
        erosion_thickness = smallest_semi_axis ** 2 / largest_semi_axis
        # Semi-latus rectum
        return erosion_thickness

    def intersection(self, other_particle: Particle, box: list, inside=False) -> bool:
        """
        Check if the Ellipsoid intersects the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list
            Dimensions of the simulation box.

        inside: bool
            Consider self completly inside other_particle as an intersection. The opposite
            is not considered.

        Returns
        -------
        intersection: bool
            True if the particles intersect.
        """
        if isinstance(other_particle, (Sphere, Ellipsoid)) and inside:
            other_particle: Ellipsoid
            # The other particle is also an Ellipsoid or subclass
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
        elif isinstance(other_particle, Cylinder) or not inside:
            intersection, _, _ = self.intersection_gjk(
                other_particle, box, inside=inside, int_only=True
            )
        else:
            raise ValueError("Incompatible particles.")
        return intersection
        # Returning the intersection area

    def support_function(self, direction):
        dir_local = self.rotation_mat.T.dot(direction)
        dir_normal = np.array(
            [
                dir_local[0] * (self.semi_axis_1 ** 2),
                dir_local[1] * (self.semi_axis_2 ** 2),
                dir_local[2] * (self.semi_axis_3 ** 2),
            ]
        )
        rescale_factor = np.sqrt(
            1
            / (
                (dir_normal[0] / self.semi_axis_1) ** 2
                + (dir_normal[1] / self.semi_axis_2) ** 2
                + (dir_normal[2] / self.semi_axis_3) ** 2
            )
        )
        dir_normal_unit = dir_normal / np.linalg.norm(dir_normal)
        point_on_ellipsoid_loc = (
            rescale_factor * dir_normal + self.delta * dir_normal_unit
        )
        point_on_ellipsoid_glob = (
            self.rotation_mat.dot(point_on_ellipsoid_loc) + self.position_center
        )
        return point_on_ellipsoid_glob

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8
    ) -> tuple[float, np.array]:
        """Intersection length between *self* and *other_particle* on *box*."""
        if isinstance(other_particle, Ellipsoid) and False:
            # start = time.time()
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
            # Saving the class name of the other particle as a string
            if intersection:
                # There is overlap
                # overlap_volume = self.intersection_volume_ellipsoid_other(
                #     other_particle, box, max_it=50, seq_size=100
                # )
                diff_in_box = self.position_center - other_particle.position_center
                # Difference vector between the center of the two particles
                diff_nearest_other = box * np.round(diff_in_box / box)
                # Vector from the position of the other ellipse to its nearest image to the current
                # ellipse
                search_direction = self.intersection_vector(other_particle, box)
                mink_diff_point = self.support_function(search_direction) - (
                    other_particle.support_function(-search_direction)
                    + diff_nearest_other
                )
                intersection_length = mink_diff_point.dot(search_direction)
                # _, intersection_length_2, unit_vector_2 = self.intersection_gjk(
                #     other_particle, box, tol=tol
                # )
                unit_vector = search_direction
                if intersection_length < 0:
                    intersection_length = 0
                    unit_vector = np.array([0.0, 0.0, 0.0])
                # Computing the intersection area
            else:
                # There is no overlap
                intersection_length = 0
                unit_vector = np.array([0.0, 0.0, 0.0])
            # time_1 = time.time() - start
            # start = time.time()
            # _, intersection_length, unit_vector = self.intersection_gjk(
            #     other_particle, box, tol=tol
            # )
            # time_2 = time.time() - start
            # # print(time_1, time_2)
        elif isinstance(other_particle, (Cylinder, Ellipsoid)):
            _, intersection_length, unit_vector = self.intersection_gjk(
                other_particle, box, tol=tol
            )
        return intersection_length, unit_vector


class Sphere(Ellipsoid):
    """
    This is the subclass of particles with the form of a sphere.

    Attributes
    ----------
    radius: float
        Radius of the disk

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a sphere,
        and their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing spheres.
    """

    possible_parameters = {
        **{"r": ("Radius", "float"), "volume": ("Volume per particle", "float")},
        **Particle.possible_parameters,
    }

    # all possible_parameters
    acceptable_descriptions = [
        {"r", "n"},
        {"r", "vf"},
        {"n", "vf"},
        {"volume", "vf"},
        {"volume", "n"},
    ]
    # List of acceptable collections of parameters

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Sphere obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        if "r" in descriptors:
            # The radius was supplied
            radius = descriptors.pop("r")
        elif "volume" in descriptors:
            # The area of each particle was supplied
            radius = np.cbrt(descriptors.pop("volume") / (4 / 3 * np.pi))
        elif "vf" in descriptors and "n" in descriptors:
            # Both the volume fraction and the number of particles was supplied
            volume = (
                descriptors["vf"]
                * rve_dims[0]
                * rve_dims[1]
                * rve_dims[2]
                / descriptors["n"]
            )
            # Area of each particle (all the same)
            radius = np.cbrt(volume / (4 / 3 * np.pi))

        ellipsoid_descriptors = {
            "axis_1": 2 * radius,
            "axis_2": 2 * radius,
            "axis_3": 2 * radius,
            "angle": 0.0,
            "rot_axis_comp_x": 0.0,
            "rot_axis_comp_y": 0.0,
            "rot_axis_comp_z": 1.0,
        }
        super().__init__(phase, ellipsoid_descriptors, rve_dims)

    def intersection_area(self, other_particle, box):
        """
        Compute the intersection volume (area) between the Sphere and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        """
        if isinstance(other_particle, Sphere):
            # The other particle is also a Sphere
            intersection_volume = self.intersection_volume_sphere_sphere(
                other_particle, box
            )
            # Computing the intersection area
        elif isinstance(other_particle, Ellipsoid):
            # The other particle is an Ellipsoid
            other_particle: Ellipsoid
            intersection_volume = other_particle.intersection_area(self, box)
            # Computing the intersection area
        else:
            intersection, overlap_length = self.intersection_gjk(other_particle, box)
            intersection_volume = overlap_length if intersection else 0
        return intersection_volume
        # Returning the intersection area

    def intersection_volume_sphere_sphere(self, other_sphere, box):
        """
        Compute the intersection volume between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
            Other sphere whose intersection volume with the current sphere we want to know
        """
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        d = np.linalg.norm(diff_center)
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_sphere.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = other_sphere.radius
            # Sphere 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_sphere.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = self.radius
            # Sphere 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_volume = 0
            # The intersection area of the spheres is zero
        elif d <= r_1 - r_2:
            # Sphere 2 is interely contained within Sphere 1
            intersection_volume = 4 / 3 * np.pi * r_2 ** 3
            # The intersection area is equal to the area of the smaller sphere, Sphere 2
        else:
            d_1 = (r_1 ** 2 - r_2 ** 2 + d ** 2) / (2 * d)
            # x coordinate of the intersection point of the two disks if the the origin is
            # at disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_volume = (
                r_1 ** 3
                / 3
                * 2
                * np.pi
                * (1 - d_1 / r_1)  # Volume of spherical sector (Sphere 1)
                - d_1 * (r_1 ** 2 - d_1 ** 2) * np.pi / 3  # Volume of cone (Sphere 1)
                + r_2 ** 3
                / 3
                * 2
                * np.pi
                * (1 - d_2 / r_2)  # Volume of shperical sector (Sphere 2)
                - d_2 * (r_2 ** 2 - d_2 ** 2) * np.pi / 3
            )  # Volume of cone (Sphere 2)
            # Computing the intersection area as the sum of the spherical caps minus the
            # corresponding cones
            # intersection_volume = 0.01
        return intersection_volume
        # Returning the intersection area

    def intersection_sqrt(self, other_sphere, box):
        """
        Compute the intersection volume between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
            Other sphere whose intersection volume with the current sphere we want to know
        """
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Computing the difference vector between the centers of the current sphere and
        # the nearest image of the other sphere
        d = np.linalg.norm(diff_center)
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_sphere.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = other_sphere.radius
            # Sphere 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_sphere.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = self.radius
            # Sphere 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_volume = 0
            # The intersection area of the spheres is zero
        elif d <= r_1 - r_2:
            # Sphere 2 is interely contained within Sphere 1
            # intersection_volume = r_1 + r_2  # 4 / 3 * np.pi * r_2 ** 3
            # intersection_volume = 4 / 3 * np.pi * r_2 ** 3
            intersection_volume = r_2
            # The intersection area is equal to the area of the smaller sphere, Sphere 2
        else:
            d_1 = (r_1 ** 2 - r_2 ** 2 + d ** 2) / (2 * d)
            # x coordinate of the intersection point of the two disks if the the origin is
            # at disk 1 and the x axis goes through the center of both disks
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to disk 2
            intersection_volume = r_2 * (
                1 - (d) ** 2 / (r_1 + r_2) ** 2
            )  # / (2 * r_2) * 4 / 3 * np.pi * r_2 ** 3
            # intersection_volume = (
            #     r_1 ** 3
            #     / 3
            #     * 2
            #     * np.pi
            #     * (1 - d_1 / r_1)  # Volume of spherical sector (Sphere 1)
            #     - d_1 * (r_1 ** 2 - d_1 ** 2) * np.pi / 3  # Volume of cone (Sphere 1)
            #     + r_2 ** 3
            #     / 3
            #     * 2
            #     * np.pi
            #     * (1 - d_2 / r_2)  # Volume of shperical sector (Sphere 2)
            #     - d_2 * (r_2 ** 2 - d_2 ** 2) * np.pi / 3
            # )  # Volume of cone (Sphere 2)
            # Computing the intersection area as the sum of the spherical caps minus the
            # corresponding cones
            # intersection_volume = 0.01
        return intersection_volume
        # Returning the intersection area

    @property
    def volume(self):
        """Volume of the sphere."""
        volume = 4 * np.pi / 3 * self.radius ** 3

        return volume

    def intersection(self, other_particle: Particle, box: list, inside=True) -> bool:
        """
        Check if the Sphere intersects the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        intersection: bool
            True if the particles intersect.
        """
        if isinstance(other_particle, Sphere):
            # The other particle is also a Disk
            intersection = self.intersection_sphere_sphere(
                other_particle, box, inside=inside
            )
            # Computing the intersection area
        elif isinstance(other_particle, Ellipsoid):
            other_particle: Ellipsoid
            intersection = other_particle.intersection(self, box)
        elif isinstance(other_particle, Cylinder):
            other_particle: Cylinder
            intersection, _ = self.intersection_sphere_cylinder(other_particle, box)
        else:
            intersection, _ = self.intersection_gjk(other_particle, box, int_only=True)

        return intersection
        # Returning the intersection area

    def intersection_sphere_sphere(
        self, other_sphere: Sphere, box: list, inside=True
    ) -> bool:
        """Check if the two spheres intersect."""
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        distance_spheres = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        if inside:
            # Being completly inside is considered as an intersection
            intersection_bool = distance_spheres < (self.radius + other_sphere.radius)
        else:
            # Being completly inside is not considered as an intersection
            intersection_bool = distance_spheres < (
                self.radius + other_sphere.radius
            ) and distance_spheres > np.abs(self.radius - other_sphere.radius)

        return intersection_bool

    def intersection_sphere_cylinder(
        self: Sphere, cylinder: Cylinder, box: list
    ) -> tuple[bool, float]:
        cylinder_position_center_pbc = Particle.nearest_periodic_image(
            cylinder.position_center, self.position_center, box
        )
        dist_on_axis = cylinder.sym_axis_unit_vec.dot(
            self.position_center - cylinder_position_center_pbc
        )
        if np.abs(dist_on_axis) > cylinder.length / 2 + self.radius:
            if (
                np.linalg.norm(self.position_center - cylinder_position_center_pbc)
                < self.radius
            ):
                intersection = True
                overlap_length = cylinder.length

                return intersection, overlap_length

            intersection = False
            overlap_length = 0

            return intersection, overlap_length

        if np.abs(dist_on_axis) < cylinder.length / 2:
            L = np.sqrt(
                np.sum((self.position_center - cylinder_position_center_pbc) ** 2)
                - dist_on_axis ** 2
            )
            if L < self.radius + cylinder.r_cyl:
                intersection = True
                overlap_length = np.sqrt((self.radius + cylinder.r_cyl) ** 2 - L ** 2)
                # sc2
                return intersection, overlap_length

            intersection = False
            overlap_length = 0

            return intersection, overlap_length

        if (
            cylinder.length / 2
            <= np.abs(dist_on_axis)
            <= cylinder.length / 2 + self.radius
        ):
            L = np.sqrt(
                np.sum((self.position_center - cylinder_position_center_pbc) ** 2)
                - dist_on_axis ** 2
            )
            if (
                L
                < np.sqrt(
                    self.radius ** 2 - (np.abs(dist_on_axis) - cylinder.length / 2) ** 2
                )
                + cylinder.r_cyl
            ):
                if np.abs(L) < cylinder.r_cyl:
                    intersection = True
                    overlap_length = (
                        cylinder.length / 2 + self.radius - np.abs(dist_on_axis)
                    )
                    # sc3

                    return intersection, overlap_length

                if np.abs(L) >= cylinder.r_cyl:
                    intersection = True
                    overlap_length = (
                        np.sqrt(
                            self.radius ** 2
                            - (np.abs(dist_on_axis) - cylinder.length / 2) ** 2
                        )
                        + cylinder.r_cyl
                        - L
                    )
                    # sc4

                    return intersection, overlap_length

                intersection = False
                overlap_length = 0

                return intersection, overlap_length
        intersection = False
        overlap_length = 0

        return intersection, overlap_length

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8
    ) -> tuple[float, np.array]:
        """
        Compute the intersection length between the Sphere and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        """
        if isinstance(other_particle, Sphere):
            # The other particle is also a Sphere
            intersection_length = self.intersection_length_sphere_sphere(
                other_particle, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
            # Computing the intersection length
        # elif isinstance(other_particle, Cylinder):
        #     # The other particle is a cylinder
        #     other_particle: Cylinder
        #     _, intersection_length = self.intersection_sphere_cylinder(
        #         other_particle, box
        #     )
        #     unit_vector = self.intersection_vector(other_particle, box)
        else:
            intersection, overlap_length, unit_vector = self.intersection_gjk(
                other_particle, box, tol=tol
            )
            intersection_length = overlap_length if intersection else 0
        return intersection_length, unit_vector
        # Returning the intersection area

    def intersection_length_sphere_sphere(
        self, other_sphere: Sphere, box: list
    ) -> float:
        """
        Compute the intersection length between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
            Other sphere whose intersection length with the current sphere we want to know
        """
        d = np.linalg.norm(
            Particle.nearest_periodic_image(
                self.position_center, other_sphere.position_center, box
            )
            - other_sphere.position_center
        )
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_sphere.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = other_sphere.radius
            # Sphere 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_sphere.radius
            # Sphere 1 is the sphere with the larger radius
            r_2 = self.radius
            # Sphere 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_length = 0
            # The intersection length of the spheres is zero
        elif d <= r_1 - r_2:
            # Sphere 2 is interely contained within Sphere 1
            intersection_length = 2 * r_2
            # The intersection length is the diameter of the smaller sphere
        else:
            intersection_length = r_1 + r_2 - d
            # intersection length
        return intersection_length
        # Returning the intersection length

    def point_inside(self, point, box, tol=1e-3):
        """Check if point is inside the particle."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        point_in = (
            np.linalg.norm(self.position_center - point_nearest_pbc) - self.radius
            <= tol
        )

        return point_in

    def generate_points_on_surface(self, n_points, erosion_thick=0):
        """Generate *n_points* on the surface of the sphere."""
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Convention from physics
        if erosion_thick > 0:
            # If erosion was sepcified
            radius = self.radius - erosion_thick
            # Eroding the radius
        else:
            radius = self.radius
        points_loc = []
        for i_theta in theta:
            for j_phi in phi:
                points_loc.append(
                    [
                        radius * np.sin(i_theta) * np.cos(j_phi),
                        radius * np.sin(i_theta) * np.sin(j_phi),
                        radius * np.cos(i_theta),
                    ]
                )
        # Generating the points in the Sphere's local coordinates
        points_glob = points_loc + self.position_center
        # Transforming local in global coordinates
        return points_glob

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for a sphere."""
        erosion_thickness = 0.9 * self.radius
        # Semi-latus rectum
        return erosion_thickness

    def support_function(self, direction):
        return (
            self.position_center + direction / np.linalg.norm(direction) * self.radius
        )


class Cylinder(Particle):
    """This is the class for short cylinders.

    Attributes
    ----------
    r_cyl: float
        Radius of the cylinder

    length: float
        Length of the particle

    azimuth_angle: float
        Azimuth angle of the cylinder, i.e. the angle that the axis of the cylinder forms
        with the positive x semi-axis when project onto the xy plane.

    polar_angle: float
        Polar angle of the cylinder, i.e. the angle that the axis of the
        cylinder forms with the positive z semi-axis.

    rot_mat: array
        Rotation matrix from local to global coordinates.

    radius: float
        Radius of the circumscribed sphere.

    radius_insc: float
        Radius of the inscribed sphere.

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a sphere,
        and their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing spheres.
    """

    possible_parameters = {
        **{
            "r_cyl": ("Cylinder Radius", "float"),
            "length": ("Cylinder Length", "float"),
            "azimuth_angle": ("Azimuthal angle", "float"),
            "polar_angle": ("Polar angle", "float"),
        },
        **Particle.possible_parameters,
    }

    # all possible_parameters
    acceptable_descriptions = [
        {"r_cyl", "length", "n", "azimuth_angle", "polar_angle"},
        {"r_cyl", "length", "vf", "azimuth_angle", "polar_angle"},
        {"r_cyl", "ratio", "n", "azimuth_angle", "polar_angle"},
        {"ratio", "length", "vf", "azimuth_angle", "polar_angle"},
        {"n", "length", "vf", "azimuth_angle", "polar_angle"},
    ]
    dim = 3
    # List of acceptable collections of parameters

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Cylinder obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction.

        Raises
        ------
        ValueError:
            When the cylinder radius or the length are nonpositive numbers.
        """
        if "r_cyl" in descriptors:
            if descriptors["r_cyl"] <= 0:
                raise ValueError(
                    "In Phase {0}:".format(phase)
                    + "The radius of a cylinder particle must be a positive number."
                )
            self.r_cyl = descriptors["r_cyl"]
            if "ratio" in descriptors:
                self.length = self.r_cyl * descriptors["ratio"]
        if "length" in descriptors:
            if descriptors["length"] <= 0:
                raise ValueError(
                    "In Phase {0}:".format(phase)
                    + "The length of a cylinder particle must be a positive number."
                )
            self.length = descriptors["length"]
            if "vf" in descriptors and "n" in descriptors:
                self.r_cyl = np.sqrt(
                    descriptors["vf"]
                    * np.prod(rve_dims)
                    / (self.length * np.pi * descriptors["n"])
                )
            elif "ratio" in descriptors:
                self.r_cyl = self.length / descriptors["ratio"]
        if "azimuth_angle" in descriptors:
            self.azimuth_angle = np.abs(descriptors["azimuth_angle"])
        if "polar_angle" in descriptors:
            self.polar_angle = np.abs(descriptors["polar_angle"])
        self.rot_mat = np.array(
            [
                [
                    np.sin(self.polar_angle) * np.cos(self.azimuth_angle),
                    np.sin(self.polar_angle) * np.sin(self.azimuth_angle),
                    np.cos(self.polar_angle),
                ],
                [
                    np.cos(self.polar_angle) * np.cos(self.azimuth_angle),
                    np.cos(self.polar_angle) * np.sin(self.azimuth_angle),
                    -np.sin(self.polar_angle),
                ],
                [
                    -np.sin(self.azimuth_angle),
                    np.cos(self.azimuth_angle),
                    0,
                ],
            ]
        )
        super().__init__(3, phase)

    @property
    def volume(self):
        """Particle volume."""
        volume = self.length * np.pi * self.r_cyl ** 2
        return volume

    @property
    def sym_axis_unit_vec(self):
        """Get unit vector along the cylinder's symmetry axis."""
        sym_axis_unit_vec = np.array(
            [
                np.cos(self.azimuth_angle) * np.sin(self.polar_angle),
                np.sin(self.azimuth_angle) * np.sin(self.polar_angle),
                np.cos(self.polar_angle),
            ]
        )
        return sym_axis_unit_vec

    @property
    def radius(self):
        radius = np.sqrt((self.length / 2) ** 2 + self.r_cyl ** 2) + self.delta
        return radius

    @property
    def radius_insc(self):
        """Radius of the inscribed sphere to the cylinder."""
        radius_insc = np.min([self.length / 2, self.r_cyl])

        return radius_insc

    def intersection(self, other_particle, box, inside=True):
        intersection, *_ = self.intersection_gjk(
            other_particle, box, inside=inside, int_only=True
        )
        return intersection

    def intersection_area(self, other_particle, box):
        intersection_volume = self.intersection_area_monte_carlo(other_particle, box)
        return intersection_volume

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8
    ) -> tuple[float, np.array]:
        """Compute the intersection length between *self* and *other_particle* in *box*.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        Returns
        -------
        intersection_length: float
            Minimum distance allowing for the removal of the intersection.

        unit_vector: np.array
            Direction of the minimum displacement allowing for the removal of the
            intersection.
        """

        if False and isinstance(other_particle, Cylinder):
            other_particle: Cylinder
            _, intersection_length = self.intersection_cylinder_cylinder(
                other_particle, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
        elif False and isinstance(other_particle, Sphere):
            other_particle: Sphere
            _, intersection_length = other_particle.intersection_sphere_cylinder(
                self, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
        else:
            _, intersection_length, unit_vector = self.intersection_gjk(
                other_particle, box
            )

        return intersection_length, unit_vector

    def support_function(self, direction: np.array) -> np.array:

        dir_parallel_comp = (
            direction.dot(self.sym_axis_unit_vec) * self.sym_axis_unit_vec
        )
        dir_normal_comp = direction - dir_parallel_comp
        axial_vec_local = (
            self.length
            / 2
            * self.sym_axis_unit_vec
            * np.sign(self.sym_axis_unit_vec.dot(dir_parallel_comp))
            if np.sign(self.sym_axis_unit_vec.dot(dir_parallel_comp)) != 0
            else self.length / 2 * self.sym_axis_unit_vec
        )
        trans_vec_local = (
            self.r_cyl * dir_normal_comp / np.linalg.norm(dir_normal_comp)
            if np.linalg.norm(dir_normal_comp) != 0
            else 0
        )

        dir_unit = direction / np.linalg.norm(direction)
        point_global = (
            self.position_center
            + axial_vec_local
            + trans_vec_local
            + self.delta * dir_unit
        )

        return point_global

    def contract(self, distance):
        """Contract the particle."""
        # self.r_cyl -= distance
        # self.length -= distance
        self.delta -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        # self.r_cyl += distance
        # self.length += distance
        self.delta += distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def intersection_cylinder_cylinder(
        self: Cylinder, other_cylinder: Cylinder, box: list
    ) -> tuple[bool, float]:
        """Check if two cylinders intersect. If so give overlap length.

        This method used analytical means to detect the intersection of two cylinders and
        give their overlap length.

        Parameters
        ----------
        other_cylinder: `.Cylinder`
            Other cylinder.

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        intersection: bool
            True if the cylinders intersect, False otherwise.

        overlap_length: float
            Equal to the overlap_length if the cylinders overlap, 0 otherwise.
        """
        other_cylinder_position_center_pbc = Particle.nearest_periodic_image(
            other_cylinder.position_center, self.position_center, box
        )
        normal = (
            np.cross(self.sym_axis_unit_vec, other_cylinder.sym_axis_unit_vec)
            if np.abs(self.sym_axis_unit_vec.dot(other_cylinder.sym_axis_unit_vec) - 1)
            > 1e-4
            else np.cross(self.sym_axis_unit_vec, np.array([1, 0, 0]))
        )
        normal_unit = normal / np.linalg.norm(normal)
        # Normal vector to both symmetry axis of the cylinders
        shortest_dist = np.abs(
            normal_unit.dot(self.position_center - other_cylinder_position_center_pbc)
        )
        if shortest_dist > self.r_cyl + other_cylinder.r_cyl:
            intersection = False
            overlap_length = 0
            return intersection, overlap_length
            # If the shortest distance between axis is greater than the sum of the radii of
            # the cylinders they do not intersect

        normal_1 = np.cross(normal_unit, self.sym_axis_unit_vec * self.length / 2)
        normal_2 = np.cross(
            normal_unit,
            other_cylinder.sym_axis_unit_vec * other_cylinder.length / 2,
        )
        # Normals to each of the axis and the common normal
        normalized_dist_1 = (
            normal_2.dot(other_cylinder_position_center_pbc - self.position_center)
            / normal_2.dot(self.sym_axis_unit_vec * self.length / 2)
            if np.abs(normal_2.dot(self.sym_axis_unit_vec)) > 1e-5
            else 0
        )
        normalized_dist_2 = (
            normal_1.dot(self.position_center - other_cylinder_position_center_pbc)
            / normal_1.dot(other_cylinder.sym_axis_unit_vec * other_cylinder.length / 2)
            if np.abs(normal_1.dot(other_cylinder.sym_axis_unit_vec)) > 1e-5
            else 0
        )
        # Normalized distance relative to the length of each cylinder from the center
        # point to the closest point on the axis to the other axis.
        if np.abs(normalized_dist_1) <= 1 and np.abs(normalized_dist_2) <= 1:
            # Intersection located in the laterals of the cylinders
            intersection = True
            overlap_length = self.r_cyl + other_cylinder.r_cyl - shortest_dist
            return intersection, overlap_length

        (
            intersection,
            overlap_length,
        ) = self.intersection_top_disk_lateral_cylinder(other_cylinder, box)
        if intersection is True:
            return intersection, overlap_length
        intersection, overlap_length = self.intersection_top_disks(other_cylinder, box)
        if intersection is True:
            return intersection, overlap_length
        intersection = False
        overlap_length = 0
        return intersection, overlap_length

    def intersection_top_disks(
        self: Cylinder, other: Cylinder, box: list
    ) -> tuple[bool, float]:
        """Check if the top of two cylinders intersect.

        Parameters
        ----------
        other: `.Cylinder`
            Other cylinder.

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        intersection: bool
            True if the cylinders intersect, False otherwise.

        overlap_length: float
            Equal to the overlap_length if the cylinders overlap, 0 otherwise.
        """

        def intersection_disk_disk(
            pos_d1: np.array,
            r_d1: float,
            normal_d1: np.array,
            pos_d2: np.array,
            r_d2: float,
            normal_d2: np.array,
        ) -> tuple[bool, float]:
            """Check if top two disks of two cylinders intersect.

            Parameters
            ----------
            pos_d1: array
                Center of disk 1.

            r_d1: float
                Radius of disk 1.

            normal_d1: array
                Outer normal to disk 1.

            pos_d2: array
                Center of disk 2.

            r_d2: float
                Radius of disk 2.

            normal_d2: array
                Outer normal to disk 2.

            Returns
            -------
            intersection: bool
                True if the cylinders intersect, False otherwise.

            overlap_length: float
                Equal to the overlap_length if the cylinders overlap, 0 otherwise.
            """
            normal = np.cross(normal_d1, normal_d2)
            normal_to_int = np.cross(normal, normal_d1)
            common_pt = (
                pos_d1
                + normal_d2.dot(pos_d2 - pos_d1)
                / normal_d2.dot(normal_to_int)
                * normal_to_int
            )
            if (
                np.linalg.norm(common_pt - pos_d1) <= r_d1
                and np.linalg.norm(common_pt - pos_d2) <= r_d2
            ):
                intersection = True
                if (
                    r_d1 ** 2 - np.linalg.norm(common_pt - pos_d1) ** 2
                    > r_d2 ** 2 - np.linalg.norm(common_pt - pos_d2) ** 2
                ):
                    overlap_length = np.abs(
                        -(r_d2 - np.linalg.norm(common_pt - pos_d2))
                    )
                    # d1
                else:
                    overlap_length = np.abs(
                        -(r_d1 - np.linalg.norm(common_pt - pos_d1))
                    )
                    # d2
            else:
                intersection = False
                overlap_length = 0
            return intersection, overlap_length

        other_position_center_pbc = Particle.nearest_periodic_image(
            other.position_center, self.position_center, box
        )

        intersection, overlap_length = intersection_disk_disk(
            self.position_center + self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            self.sym_axis_unit_vec,
            other_position_center_pbc + other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_disk(
            self.position_center + self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            self.sym_axis_unit_vec,
            other_position_center_pbc - other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            -other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_disk(
            self.position_center - self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            -self.sym_axis_unit_vec,
            other_position_center_pbc + other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_disk(
            self.position_center - self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            -self.sym_axis_unit_vec,
            other_position_center_pbc - other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            -other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection = False
        overlap_length = 0
        return intersection, overlap_length

    def intersection_top_disk_lateral_cylinder(
        self: Cylinder, other: Cylinder, box: list
    ) -> tuple[bool, float]:
        """Check if the top of one one of the cylinders intersects the lateral of the other.

        Parameters
        ----------
        other: `.Cylinder`
            Other cylinder.

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        intersection: bool
            True if the cylinders intersect, False otherwise.

        overlap_length: float
            Equal to the overlap_length if the cylinders overlap, 0 otherwise.
        """

        def intersection_disk_cylinder(
            pos_c: np.array,
            r_c: float,
            len_c: np.array,
            pos_d: np.array,
            r_d: float,
            normal_d: np.array,
        ):
            """Check if the top of a cylinder intersect the lateral of another cylinder.

            Parameters
            ----------
            pos_c: array
                Center of the cylinder.

            r_c: float
                Radius of the cylinder.

            len_c: array
                Vector parallel to the symmetry axis, whose norm is equal to half its
                length.

            pos_d: array
                Center of the disk.

            r_d: float
                Radius of the disk.

            normal_d: array
                Outer normal the disk.

            Returns
            -------
            intersection: bool
                True if the cylinders intersect, False otherwise.

            overlap_length: float
                Equal to the overlap_length if the cylinders overlap, 0 otherwise.
            """
            intrsct_pt_axis_disk = (
                pos_c + normal_d.dot(pos_d - pos_c) / normal_d.dot(len_c) * len_c
            )
            pt_bound_disk = pos_d + r_d * (
                intrsct_pt_axis_disk - pos_d
            ) / np.linalg.norm(intrsct_pt_axis_disk - pos_d)
            dist = len_c.dot(pt_bound_disk - pos_c) / np.linalg.norm(len_c)
            proj_on_axis_bound_pt = pos_c + dist * len_c / np.linalg.norm(len_c)
            if np.abs(dist) < np.linalg.norm(len_c):
                intersection = True
                if (
                    np.linalg.norm(pos_d - intrsct_pt_axis_disk) > r_d
                    and np.linalg.norm(proj_on_axis_bound_pt - pt_bound_disk) < r_c
                ):
                    overlap_length = np.abs(
                        -(np.linalg.norm(proj_on_axis_bound_pt - pt_bound_disk) - r_c)
                    )
                    # cd1
                elif (
                    np.linalg.norm(pos_d - intrsct_pt_axis_disk) < r_d
                    and np.linalg.norm(proj_on_axis_bound_pt - pt_bound_disk) < r_c
                ):
                    overlap_length = np.abs(
                        -(
                            np.linalg.norm(proj_on_axis_bound_pt - pt_bound_disk)
                            - 2 * r_c
                        )
                    )
                    # cd1
                else:
                    overlap_length = 0
            elif np.linalg.norm(intrsct_pt_axis_disk - pos_d) < r_d:
                intersection = True
                overlap_length = 2 * r_c
                # cd3
            else:
                intersection = False
                overlap_length = 0

            return intersection, overlap_length

        other_position_center_pbc = Particle.nearest_periodic_image(
            other.position_center, self.position_center, box
        )

        intersection, overlap_length = intersection_disk_cylinder(
            self.position_center,
            self.r_cyl,
            self.sym_axis_unit_vec * self.length / 2,
            other_position_center_pbc + other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_cylinder(
            self.position_center,
            self.r_cyl,
            self.sym_axis_unit_vec * self.length / 2,
            other_position_center_pbc - other.sym_axis_unit_vec * other.length / 2,
            other.r_cyl,
            -other.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_cylinder(
            other_position_center_pbc,
            other.r_cyl,
            other.sym_axis_unit_vec * self.length / 2,
            self.position_center + self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            self.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection, overlap_length = intersection_disk_cylinder(
            other_position_center_pbc,
            other.r_cyl,
            other.sym_axis_unit_vec * self.length / 2,
            self.position_center - self.sym_axis_unit_vec * self.length / 2,
            self.r_cyl,
            -self.sym_axis_unit_vec,
        )
        if intersection is True:
            return intersection, overlap_length

        intersection = False
        overlap_length = 0
        return intersection, overlap_length

    def point_inside(self, point: np.array, box: list) -> bool:
        """Check if *point* is inside *self*."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        dist_on_axis = self.sym_axis_unit_vec.dot(
            self.position_center - point_nearest_pbc
        )
        if np.abs(dist_on_axis) > self.length / 2:
            point_inside = False

        elif np.abs(dist_on_axis) <= self.length / 2:
            L = np.sqrt(
                np.sum((self.position_center - point_nearest_pbc) ** 2)
                - dist_on_axis ** 2
            )
            if L < self.r_cyl:
                point_inside = True
            else:
                point_inside = False

        return point_inside

    def generate_point_inside(self):
        """Generate a random point inside the cylinder."""
        w = np.random.normal(size=2)
        # Generating 3 independent random points from the standard Gaussian distribution
        r = np.random.uniform() ** (1 / 2)
        # Sampling the "radius"
        R = np.linalg.norm(w)
        x_loc = np.array(
            [
                r * self.r_cyl * w[0] / R,
                r * self.r_cyl * w[1] / R,
            ]
        )
        x_loc = np.append(x_loc, np.random.uniform(-self.length / 2, self.length / 2))
        x_glob = self.rot_mat.dot(x_loc) + self.position_center
        return x_glob


class Matrix(Particle):
    """
    Class for the "matrix" particle.

     Created just to make polymorphism work in the rest of the code.
    """

    possible_parameters = {}
    acceptable_descriptions = [set()]

    def intersection(self, other_particle, box):
        """Do nothing."""
        return None

    def intersection_area(self, other_particle, box):
        """Do nothing."""
        return None


class Point(Particle):

    radius = 0

    def intersection(self, other_particle, box) -> bool:
        """Check if the two particles intersect."""

    def intersection_area(self, other_particle, box) -> float:
        """Compute the interesection area/volume between two particles."""

    def intersection_length(self, other_particle, box) -> tuple[float, np.array]:
        """Compute the interesection length between two particles."""

    def support_function(self, direction: np.array) -> np.array:
        """Compute the interesection length between two particles."""

        return (
            self.position_center
            if self.dim == 3
            else np.append(self.position_center, 0)
        )

    def point_inside(self, point: np.array, box: list) -> bool:
        """Check if some point is inside the particle."""
        return False

    def generate_point_inside(self):
        """Generate a random point inside the particle."""


class Line(Particle):
    def __init__(self, phase, dir_ind):
        self.dir_ind = dir_ind
        super().__init__(3, phase)

    def intersection(self, other_particle, box) -> bool:
        """Check if the two particles intersect."""

    def intersection_area(self, other_particle, box) -> float:
        """Compute the interesection area/volume between two particles."""

    def intersection_length(self, other_particle, box) -> tuple[float, np.array]:
        """Compute the interesection length between two particles."""

    def support_function(self, direction: np.array) -> np.array:
        """Compute point from the support function along *direction*."""

        point = list(self.position_center)
        if direction[self.dir_ind] >= 0:
            point[self.dir_ind] = 1
        else:
            point[self.dir_ind] = 0
        # point[self.dir_ind] = direction[self.dir_ind]
        # Point in the local coordinate system

        return np.array(point)

    def point_inside(self, point: np.array) -> bool:
        """Check if some point is inside the particle."""

    def generate_point_inside(self):
        """Generate a random point inside the particle."""


def intersection_points_ellipses(
    A1, B1, center_1, angle_1, A2, B2, center_2, angle_2, tol=1e-10
):
    """
    Return the y coordinates of the intersection points between two ellipses.

    Parameters
    ----------
    A1: float
        Semi-major axis of ellipse 1.

    B1: float
        Semi-minor axis of ellipse 1.

    center_1: array
        Coordinates of the center of ellipse 1.

    angle_1: float
        Angle in radians that the major axis of ellipse 1 forms with the x-axis.

    A2: float
        Semi-major axis of ellipse 2.

    B2: float
        Semi-minor axis of ellipse 2.

    center_2: array
        Coordinates of the center of ellipse 2.

    angle_2: float
        Angle in radians that the major axis of ellipse 2 forms with the x-axis

    Returns
    -------
    intersect_points: list(array)
        List of arrays containing the intersection points of the two ellipses in the
        original coordinate system
    """
    intersect_pts = []
    # Initializing the array containing the intersection points
    rot_mat = np.array(
        [[np.cos(angle_1), np.sin(angle_1)], [-np.sin(angle_1), np.cos(angle_1)]]
    )
    rot_mat_back = rot_mat.T
    # Rotation matrix that alignes ellipse 1 with the xy-axis
    center_2_TR = rot_mat.dot(center_2 - center_1)
    # Translation and rotation of ellipse 2 with the origin at the center of ellipse 1
    # aligning with the xy axis
    theta = angle_2 - angle_1
    # Saving the angle between the axis of both ellipses
    AA = A2 ** 2 * np.sin(theta) ** 2 + B2 ** 2 * np.cos(theta) ** 2
    BB = 2 * (B2 ** 2 - A2 ** 2) * np.sin(theta) * np.cos(theta)
    CC = A2 ** 2 * np.cos(theta) ** 2 + B2 ** 2 * np.sin(theta) ** 2
    DD = -2 * AA * center_2_TR[0] - BB * center_2_TR[1]
    EE = -BB * center_2_TR[0] - 2 * CC * center_2_TR[1]
    FF = (
        AA * center_2_TR[0] ** 2
        + BB * center_2_TR[0] * center_2_TR[1]
        + CC * center_2_TR[1] ** 2
        - A2 ** 2 * B2 ** 2
    )
    # Coefficients defining ellipse 2 on the coordinate system of ellipse 1
    p = np.zeros(5)
    # Initializing the vector of the coefficients
    p[0] = (
        -(CC ** 2) * B1 ** 4
        + 2 * (AA * CC - BB ** 2 / 2) * A1 ** 2 * B1 ** 2
        - A1 ** 4 * AA ** 2
    )
    p[1] = (
        -((-A1 * BB + EE) * CC + CC * (A1 * BB + EE)) * B1 ** 4
        + 2 * (AA * EE - BB * DD) * A1 ** 2 * B1 ** 2
    )
    p[2] = (
        -(
            (A1 ** 2 * AA - A1 * DD + FF) * CC
            + (-A1 * BB + EE) * (A1 * BB + EE)
            + CC * (A1 ** 2 * AA + A1 * DD + FF)
        )
        * B1 ** 4
        + 2 * (AA ** 2 * A1 ** 2 + AA * FF - 1 / 2 * DD ** 2) * A1 ** 2 * B1 ** 2
    )
    p[3] = (
        -(
            (A1 ** 2 * AA - A1 * DD + FF) * (A1 * BB + EE)
            + (-A1 * BB + EE) * (A1 ** 2 * AA + A1 * DD + FF)
        )
        * B1 ** 4
    )
    p[4] = -(A1 ** 2 * AA - A1 * DD + FF) * (A1 ** 2 * AA + A1 * DD + FF) * B1 ** 4
    # Coefficients of the polynomial expressing the intersection of the two ellipses
    y_pts = []
    roots = set(np.roots(p))
    # Roots of the polynomial, with positive values giving the y values of the
    # intersection points in the coordinate system of ellipse 1
    for i_root in roots:
        # Running through all the roots
        if np.abs(np.imag(i_root)) < tol:
            # if the root is real, then it is the y-coordinate of an intersection point
            y_pt = np.real(i_root)
            disc = 1 - y_pt ** 2 / B1 ** 2
            if disc < -1e-4:
                continue
            if np.abs((np.abs(y_pt) - B1) / B1) < 1e-4:
                intersect_pts.append(rot_mat_back.dot(np.array([0, y_pt])) + center_1)
            elif (
                not np.any(np.isclose(y_pt * np.ones(len(y_pts)), y_pts))
                or len(y_pts) == 0
            ):
                x_pt = A1 * np.sqrt(disc)
                y_pts.append(y_pt)
                # (x_pt, y_pt) and (-x_pt,y_pt) are the coordinates of the potential
                # intersection points obtained assuming that they are on ellipse 1
                on_ellipse_2_1 = (
                    np.abs(
                        AA * x_pt ** 2
                        + BB * x_pt * y_pt
                        + CC * y_pt ** 2
                        + DD * x_pt
                        + EE * y_pt
                        + FF
                    )
                    < tol
                )
                # Checking if (x_pt, y_pt) is also on ellispe 2 and so it's a real
                # intersection point
                on_ellipse_2_2 = (
                    np.abs(
                        AA * x_pt ** 2
                        - BB * x_pt * y_pt
                        + CC * y_pt ** 2
                        - DD * x_pt
                        + EE * y_pt
                        + FF
                    )
                    < tol
                )
                # Checking if (-x_pt, y_pt) is also on ellispe 2 and so it's a real
                # intersection point
                if on_ellipse_2_1:
                    # (x_pt, y_pt) is a true intersection point
                    intersect_pts.append(
                        rot_mat_back.dot(np.array([x_pt, y_pt])) + center_1
                    )
                    # Append the point to the list of intersection points in the original
                    # coordinate system
                if on_ellipse_2_2:
                    # (-x_pt, y_pt) is a true intersectio point
                    intersect_pts.append(
                        rot_mat_back.dot(np.array([-x_pt, y_pt])) + center_1
                    )
                    # Append the point to the list of intersection points in the original
                    # coordinate system
                # if on_ellipse_2_1 and on_ellipse_2_2 and np.abs(x_pt)<0.05:
                #     intersect_pts.pop()

    return intersect_pts
