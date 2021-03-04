"""
Module containing all the Particle abstract class and some particular subclasses.

The subclasses include are the the Matrix, Point and Line subclasses.
"""
from __future__ import annotations

import abc

import numpy as np

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

        Returns
        -------
        intersection: bool
            True if the particles intersect, False otherwise.
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
            if k_iter == 1000:
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

    def intersection_length_mink_diff(
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
    def point_inside(self, point: np.array, box: list) -> bool:
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
