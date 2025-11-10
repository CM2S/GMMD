"""Module containing the Square particle class."""

from __future__ import annotations
from typing import Union

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import Ellipse, Particle, MINIMUM_SIZE


class Square(Particle):
    """
    This is the subclass of particles with the form of a circular square.

    Attributes
    ----------
    radius: float
        Radius of the square

    Class Attributes
    ----------------
    possible_parameters: dict
        Dictionary containing as keys the possible parameters used to describe a square, and
        their names for printing

    acceptable_descriptions: list(set(strings))
        Acceptable sets of parameters that fully describe a phase containing squares.
    """

    possible_parameters = {
        **Particle.possible_parameters,
        **{
            "side": (
                "Side",
                lambda side, rve_dims: MINIMUM_SIZE / 2 < side < min(rve_dims) / 4,
                "float",
            ),
            "area": (
                "Area per particle",
                lambda area, rve_dims: np.pi * (MINIMUM_SIZE / 2) ** 2
                < area
                < np.pi * (min(rve_dims) / 4) ** 2,
                "float",
            ),
        },
    }
    #
    # )
    # all possible_parameters
    acceptable_descriptions = [
        {"r", "n"},
        {"area", "n"},
    ]
    # List of acceptable collections of parameters
    dim = 2

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Square obejct.

        Parameters
        ----------
        phase: string
            Phase to which the ellipse belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
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
        """Generate *n_points* on the surface of the Square."""
        points_loc = np.array(
            [
                [self.radius * np.cos(theta), self.radius * np.sin(theta)]
                for theta in np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            ]
        )
        # Generating the points in the Square's local coordinates
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
        Compute the intersection area between the square and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        """
        if isinstance(other_particle, Square):
            # The other particle is also a Square
            intersection_area = self.intersection_area_square_square(other_particle, box)
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

    def intersection_area_square_square(self, other_square, box):
        """Compute the intersection area between two squares."""
        diff_center = self.position_center - other_square.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector from the current square to the nearest image of the other square
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the center of the squares
        if self.radius >= other_square.radius:
            # The radius of the self is larger than the radius of the other square
            r_1 = self.radius
            # Square 1 is the square with the larger radius
            r_2 = other_square.radius
            # Square 2 is the square with the smaller radius
        else:
            # The radius of the other square is larger than the radius of the self
            r_1 = other_square.radius
            # Square 1 is the square with the larger radius
            r_2 = self.radius
            # Square 2 is the square with the smaller radius
        if d >= (r_1 + r_2):
            # The squares intersect at most at one point
            intersection_area = 0
            # The intersection area of the squares is zero
        elif d <= r_1 - r_2:
            # Square 2 is interely contained within Square 1
            intersection_area = np.pi * r_2 ** 2
            # The intersection area is equal to the area of the smaller square, Square 2
        else:
            d_1 = (r_1 ** 2 - r_2 ** 2 + d ** 2) / (2 * d)
            # x coordinate of the intersection point of the two squares if the the origin is
            # at square 1 and the x axis goes through the center of both squares
            d_2 = d - d_1
            # Distance in the x axis from the intersection point to square 2
            intersection_area = (
                r_1 ** 2 * np.arccos(d_1 / r_1)
                - d_1 * np.sqrt(r_1 ** 2 - d_1 ** 2)
                + r_2 ** 2 * np.arccos(d_2 / r_2)
                - d_2 * np.sqrt(r_2 ** 2 - d_2 ** 2)
            )
            # Computing the intersection area
        return intersection_area
        # Returning the intersection area

    def intersection(self, other_particle: Particle, box: list, inside=True) -> bool:
        """Check if the Square intersects the other_particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.
        """
        if isinstance(other_particle, Square):
            # The other particle is also a Square
            intersection = self.intersection_square_square(
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
        """Check if some point is inside the Square."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        point_in = (
            np.linalg.norm(self.position_center - point_nearest_pbc) <= self.radius
        )

        return point_in

    def intersection_square_square(self, other_square: Square, box: list, inside=True) -> bool:
        """Check if two Squares intersect."""
        diff_center = self.position_center - other_square.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector between the centers of the current square and the nearest image of the other
        # square
        distance_squares = np.sqrt(diff_center.dot(diff_center))
        # Distance between the squares
        if inside:
            # Being completly inside is considered as an intersection
            intersection_bool = distance_squares < (self.radius + other_square.radius)
        else:
            # Being completly inside is not considered as an intersection
            intersection_bool = distance_squares < (
                self.radius + other_square.radius
            ) and distance_squares > np.abs(self.radius - other_square.radius)

        return intersection_bool

    @property
    def volume(self):
        """Volume/area of the square."""
        volume = np.pi * (self.radius + self.delta) ** 2

        return volume

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for a square."""
        erosion_thickness = self.radius
        return erosion_thickness

    def support_function(self, direction):
        """Support function for the square."""
        radius_vec = direction[0:2] / np.linalg.norm(direction[0:2]) * self.radius
        return np.append(self.position_center + radius_vec, [0])
        # GJK algorithm is written for 3D

    def intersection_length(
        self, other_particle: Particle, box: list, tol: float = 1e-8, **kwargs
    ) -> Union[float, np.array]:
        """
        Compute the intersection length between the Square and the other particle.

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
        dist_met = kwargs.get("dist_met", "dist_exact")
        if isinstance(other_particle, Square):
            # The other particle is also a Square
            intersection_length = self.intersection_length_square_square(
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
            intersection = self.intersection_gjk(other_particle, box, tol=tol)
            overlap_length, unit_vector = self.intersection_length_mink_diff(
                other_particle, box, dist_met=dist_met
            )
            intersection_length = overlap_length if intersection else 0
        return intersection_length, unit_vector
        # Returning the intersection length

    def intersection_length_square_square(self, other_square: Square, box: list) -> float:
        """
        Compute the intersection length between two Squares.

        Parameters
        ----------
        other_square: `.Square`
            Other sphere whose intersection length with the current sphere we want to know
        """
        d = np.linalg.norm(
            Particle.nearest_periodic_image(
                self.position_center, other_square.position_center, box
            )
            - other_square.position_center
        )
        # Distance between the current sphere and the nearest image of the other sphere
        if self.radius >= other_square.radius:
            # The radius of the self is larger than the radius of the other sphere
            r_1 = self.radius
            # Square 1 is the sphere with the larger radius
            r_2 = other_square.radius
            # Square 2 is the sphere with the smaller radius
        else:
            # The radius of the other sphere is larger than the radius of the self
            r_1 = other_square.radius
            # Square 1 is the sphere with the larger radius
            r_2 = self.radius
            # Square 2 is the sphere with the smaller radius
        if d >= (r_1 + r_2):
            # The spheres intersect at most at one point
            intersection_length = 0
            # The intersection length of the spheres is zero
        elif d <= r_1 - r_2:
            # Square 2 is interely contained within Square 1
            intersection_length = 2 * r_2
            # The intersection length is the diameter of the smaller sphere
        else:
            intersection_length = r_1 + r_2 - d
            # intersection length
        return intersection_length
        # Returning the intersection length
