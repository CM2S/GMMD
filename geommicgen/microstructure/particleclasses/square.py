"""Module containing the Square particle class."""

from __future__ import annotations
from typing import Union

import numpy as np

# pylint: disable=import-errorpython
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.particleclasses import  Particle, MINIMUM_SIZE


class Square(Particle):
    """
    This is the subclass of particles with the form of square.

    Attributes
    ----------
    side: float
        Side length of the square

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
                lambda side, rve_dims: MINIMUM_SIZE < side < min(rve_dims) / 4,
                "float",
            ),
            "area": (
                "Area per particle",
                lambda area, rve_dims: MINIMUM_SIZE ** 2
                < area
                <  (min(rve_dims) / 4) ** 2,
                "float",
            ),
        },
    }
    # all possible_parameters
    #
    acceptable_descriptions = [
        {"side", "n"},
        # Implement later
        #{"area", "n"},
        #{"side", "vf"},
        #{"area", "vf"}
    ]
    # List of acceptable collections of parameters
    dim = 2

    def __init__(self, phase, descriptors, rve_dims):
        """
        Initialize a classe Square object.

        Parameters
        ----------
        phase: square
            Phase to which the square belongs

        descriptors: dict
            Dictionary of the form *{descriptor_name: value}*

        rve_dims: list
            List containing the dimensions of the microstructure in each direction
        """
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
        if "side" in descriptors:
            # The side was supplied
            side = descriptors["side"]
        elif "area" in descriptors:
            # The area of each particle was supplied
            side = np.sqrt(descriptors["area"])
        elif "vf" in descriptors and "n" in descriptors:
            # Both the volume fraction and the number of particles was supplied
            area = descriptors["vf"] * rve_dims[0] * rve_dims[1] / descriptors["n"]
            # Area of each particle (all the same)
            side = np.sqrt(descriptors["area"])
        self.side = side
        super().__init__(2, phase)

    def generate_points_on_surface(self):
        """Generate *n_points* on the surface of the square."""
        raise NotImplementedError("To be implemented later.")

    def intersection_area(self, other_particle: Particle, box: list) -> float:
        """
        Compute the intersection area between the square and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle
        """
        # For now, only square-square intersection area is implemented
        if isinstance(other_particle, Square):
            # The other particle is also a Square
          intersection_area = self.intersection_area_square_square(other_particle, box)
            # Computing the intersection area
        #elif isinstance(other_particle, Disk):
            #other_particle: Disk
            # The other particle is an Disk
            #intersection_area = other_particle.intersection_area_square_disc(
                #self, box
            #)
        #lif isinstance(other_particle, Ellipse):
            #other_particle: Ellipse
            # The other particle is an Ellipse
            #intersection_area = other_particle.intersection_area_square_ellipse( self, box)
            # Computing the intersection area
        else:
            raise NotImplementedError(
                "Intersection area between Square and {0} not implemented.".format(
                    other_particle.__class__.__name__
                )
            )
        return intersection_area
        # Returning the intersection area

    def intersection_area_square_square(self, other_square, box):
        """Compute the intersection area between two squares."""
        self.position_center
        other_square.position_center
        x_min = max(self.position_center[0] - self.side / 2, other_square.position_center[0] - other_square.side / 2)
        x_max = min(self.position_center[0] + self.side / 2, other_square.position_center[0] + other_square.side / 2)
        y_min = max(self.position_center[1] - self.side / 2, other_square.position_center[1] - other_square.side / 2)
        y_max = min(self.position_center[1] + self.side / 2, other_square.position_center[1] + other_square.side / 2)
        overlap_x = max(0, x_max - x_min)
        overlap_y = max(0, y_max - y_min)
        intersection_area = overlap_x * overlap_y
        return intersection_area
        # Returning the intersection area


    def point_inside(self, point, box):
        """Check if some point is inside the Square."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        point_in = (
            (np.array(self.position_center) - np.array(point_nearest_pbc))[0] <= self.side / 2 and (np.array(self.position_center) - np.array(point_nearest_pbc))[1]<= self.side / 2 
        )
        
        return point_in

    def intersection_square_square(self, other_square: Square, box: list, inside=True) -> bool:
        """Check if two Squares intersect."""
        # Vertices of self
        x_min_self = self.position_center[0] - self.side / 2
        x_max_self = self.position_center[0] + self.side / 2
        y_min_self = self.position_center[1] - self.side / 2
        y_max_self = self.position_center[1] + self.side / 2
        # Vertices of other_square
        x_min_other = other_square.position_center[0] - other_square.side / 2
        x_max_other = other_square.position_center[0] + other_square.side / 2
        y_min_other = other_square.position_center[1] - other_square.side / 2
        y_max_other = other_square.position_center[1] + other_square.side / 2
        # Check for intersection
        intersection_bool = (
            (x_min_self <= x_min_other < x_max_self or x_min_self < x_max_other <= x_max_self)
            and
            (y_min_self <= y_min_other < y_max_self or y_min_self < y_max_other <= y_max_self)
             )
        return intersection_bool
    
    def intersection(self, other_particle: Particle, box: list) -> bool:
        """Check this square intersects the other particle."""
        if isinstance(other_particle, Square):
            other_particle: Square
            intersection_bool = self.intersection_square_square(other_particle, box)
        else:
            raise NotImplementedError("To be implemented later.")
            # intersection_bool = self.intersection_gjk(other_particle, box)
        return intersection_bool

    @property
    def radius(self):
        """Radius of the circumscribed circle to the square."""
        radius = self.side / 2

        return radius

    @property
    def volume(self):
        """Volume/area of the square."""
        volume =  (self.side) ** 2

        return volume
    

    def contract(self, distance):
        """Contract the particle."""
        self.delta -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        self.delta += distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def compute_critical_erosion_thickness(self):
        """Compute the critical erosion thickness for a square."""
        raise NotImplementedError("To be implemented later.")
        erosion_thickness = self.side/2
        return erosion_thickness

    def support_function(self, direction):
        """Support function for GJK algorithm for the square."""  
        if direction[0] >= 0 and direction[1] >= 0:
            vec = np.array([self.side / 2, self.side / 2])
        elif direction[0] < 0 and direction[1] >= 0:
            vec = np.array([-self.side / 2, self.side / 2])
        elif direction[0] < 0 and direction[1] < 0:
            vec = np.array([-self.side / 2, -self.side / 2])
        else:
            vec = np.array([self.side / 2, -self.side / 2]) 
        return np.append(self.position_center + vec, [0])
        # GJK algorithm is written for 3D


    def uniform_sample_square(self, n_samples: int = 1) -> list(np.array):
        """Generate uniform random sample of points inside a square."""
        points = []
        half_side = self.side / 2
        for _ in range(n_samples):
            x_loc = np.random.uniform(-half_side, half_side)
            y_loc = np.random.uniform(-half_side, half_side)
            [x_glob, y_glob] = self.rot_mat.T.dot([x_loc, y_loc]) + self.position_center
            points.append(np.array([x_glob, y_glob]))

        return points
    
    def generate_point_inside(self):
        """Generate a random point inside the square."""

        return self.uniform_sample_square()[0]



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

        else:
            raise NotImplementedError("To be implemented later.")
        # elif isinstance(other_particle, Ellipse) and False:
        #     # The other particle is a ellipse
        #     other_particle: Ellipse
        #     _, intersection_length = other_particle.intersection_ellipse_ellipse(
        #         other_particle, box
        #     )
        #     unit_vector = self.intersection_vector(other_particle, box)
        # else:
        #     intersection = self.intersection_gjk(other_particle, box, tol=tol)
        #     overlap_length, unit_vector = self.intersection_length_mink_diff(
        #         other_particle, box, dist_met=dist_met
        #     )
        #     intersection_length = overlap_length if intersection else 0
        return intersection_length, unit_vector
        # Returning the intersection length

    def intersection_length_square_square(self, other_square: Square, box: list) -> float:
        """
        Compute the intersection length between two Squares.

        Parameters
        ----------
        other_square: `.Square`
            Other square whose intersection length with the current square we want to know
        """
        x_min = max(self.position_center[0] - self.side / 2, other_square.position_center[0] - other_square.side / 2)
        x_max = min(self.position_center[0] + self.side / 2, other_square.position_center[0] + other_square.side / 2)

        y_min = max(self.position_center[1] - self.side / 2, other_square.position_center[1] - other_square.side / 2)
        y_max = min(self.position_center[1] + self.side / 2, other_square.position_center[1] + other_square.side / 2)
        overlap_x = max(0, x_max - x_min)
        overlap_y = max(0, y_max - y_min)
        intersection_length = min(overlap_x, overlap_y)
        return intersection_length
        # Returning the intersection length

    def rescale(self, rescale_parameter):
        """Rescale all size parameters and the position according to *rescale_parameter*."""
        self.side *= rescale_parameter