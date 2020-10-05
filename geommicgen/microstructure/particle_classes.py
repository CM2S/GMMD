"""
Module containing all the Particle abstract class and all its subclasses.

Each subclass of the Particle class is a type of particle. This module includes the Ellipse,
Disk, CylindricalFiber, Ellipsoid and Shpere classes.
"""
from __future__ import annotations

import abc

import numpy as np

from scipy import integrate


class Particle(abc.ABC):
    """
    This is the class for particles.

    Each particle in the microstucutre is an instance of this class.

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
        radius = self.semi_major_axis

        return radius

    def contract(self, distance):
        """Contract the particle."""
        self.major_axis -= 2 * distance
        self.minor_axis -= 2 * distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        self.major_axis += 2 * distance
        self.minor_axis += 2 * distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def point_inside(self, point, tol=1e-4, position="inside"):
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
        r_vector = self.rot_mat.dot(point - self.position_center)
        # Defininig the radius vector relative to the coordinate system of the ellipse
        r_point = np.linalg.norm(r_vector)
        # Distance from the point to the center of the ellipse
        angle_pt_major = np.arctan2(r_vector[1], r_vector[0])
        # Angle that the vector connecting the center of the ellipse and the point makes
        # with the major axis
        if position == "inside":
            # Checking if the point is inside the ellipse
            point_in = r_point <= tol + self.semi_minor_axis / np.sqrt(
                1 - (self.eccentricity * np.cos(angle_pt_major)) ** 2
            )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
        elif position == "on":
            # Checking if the point is on the ellipse
            point_in = (
                np.abs(
                    r_point
                    - self.semi_minor_axis
                    / np.sqrt(1 - (self.eccentricity * np.cos(angle_pt_major)) ** 2)
                )
                < tol
            )
            # Using the polar form of the ellipse checking if the point is inside the
            # ellipse
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
                if self.point_inside(other_ellipse.position_center):
                    # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.point_inside(self.position_center):
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
                if self.point_inside(other_ellipse.position_center):
                    # The other ellipse is completly inside the current ellipse
                    intersection_area = other_ellipse.volume
                    # The intersection area is the area of the smaller ellipse
                else:
                    # The ellipses are disjoint
                    intersection_area = 0
                    # The intersection area is 0
            else:
                if other_ellipse.point_inside(self.position_center):
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
            if other_ellipse.point_inside(midpoint - diff_nearest_other):
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
            if other_ellipse.point_inside(midpoint - diff_nearest_other):
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

    def midpoint_on_ellipse(self, *args):
        """
        Return the point midway between point_1 and point_2, anti clockwise.

        The midpoint is determined using the parameteric angles of the points relative to
        the center of the ellipse and its major axis.
        """
        angle = []
        for i_point in args:
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

    def intersection(self, other_ellipse, box):
        """Check if two ellipses intersect."""
        diff_in_box = self.position_center - other_ellipse.position_center
        # Difference vector between the center of the two ellipses
        diff_nearest_other = box * np.round(diff_in_box / box)
        # Difference vector to the nearest image of the other particle
        y_inter_sect = intersection_points_ellipses(
            self.semi_major_axis,
            self.semi_minor_axis,
            self.position_center,
            self.angle,
            other_ellipse.semi_major_axis,
            other_ellipse.semi_minor_axis,
            other_ellipse.position_center + diff_nearest_other,
            other_ellipse.angle,
        )
        if len(y_inter_sect) > 0:
            # There are intersection points betweeen the two neighboorhoods
            intersection_bool = True
        else:
            # Either the ellipses are disjoint or one of them is completly inside the other
            if self.volume >= other_ellipse.volume:
                # The current ellipse is larger than the other ellipse
                intersection_bool = self.point_inside(other_ellipse.position_center)
            else:
                intersection_bool = other_ellipse.point_inside(self.position_center)
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
        points = []
        for _ in range(n_samples):
            z = np.array([0.0, 0.0])
            z[0] = np.random.normal()
            z[1] = np.random.normal()
            r = np.random.uniform() ** (1 / 2)
            R = np.linalg.norm(z)
            x_loc = r * self.semi_major_axis * z[0] / R
            y_loc = r * self.semi_minor_axis * z[1] / R
            [x_glob, y_glob] = self.rot_mat.dot([x_loc, y_loc]) + self.position_center
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

    def intersection(self, other_particle: Particle, box: list) -> bool:
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
            intersection = self.intersection_disk_disk(other_particle, box)
            # Computing the intersection area
        elif isinstance(other_particle, Ellipse):
            # The other particle is an Ellipse
            other_particle: Ellipse
            intersection = other_particle.intersection(self, box)
            # Computing the intersection area
        return intersection
        # Returning the intersection area

    def point_inside(self, point):
        """Check if some point is inside the Disk."""
        point_in = np.linalg.norm(self.position_center - point) <= self.radius

        return point_in

    def intersection_disk_disk(self, other_disk: Disk, box: list) -> bool:
        """Check if two Disks intersect."""
        diff_center = self.position_center - other_disk.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        distance_disks = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        intersection_bool = distance_disks < (self.radius + other_disk.radius)

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
            "axis_2": ("Axis 2", "float"),
            "axis_3": ("Axis 3", "float"),
            "rot_axis_comp_x": ("x-component rotation axis", "float"),
            "rot_axis_comp_y": ("y-component rotation axis", "float"),
            "rot_axis_comp_z": ("z-component rotation axis", "float"),
            "angle": ("Rotation angle", "float"),
            "ratio_12": ("Ratio a1/a2", "float"),
            "ratio_13": ("Ratio a1/a3", "float"),
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
            "vf",
            "n",
            "ratio_12",
            "ratio_13",
            "rot_axis_comp_x",
            "rot_axis_comp_y",
            "rot_axis_comp_z",
            "angle",
        },
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

        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.axis_3 = axis_3
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
        radius = np.max([self.semi_axis_1, self.semi_axis_3, self.semi_axis_3])
        # Radius of the circunscribed sphere

        return radius

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
        self.axis_1 -= 2 * distance
        self.axis_2 -= 2 * distance
        self.axis_3 -= 2 * distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        self.axis_1 += 2 * distance
        self.axis_2 += 2 * distance
        self.axis_3 += 2 * distance
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

    def point_inside(self, point, tol=1e-6, position="inside"):
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
        rot_mat_l_g = self.rotation_mat
        # Rotation matrix from local to global coordinates
        point_loc = rot_mat_l_g.T.dot(point - self.position_center)
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
                    if other_particle.point_inside(point - diff_nearest_other):
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
                    - diff_nearest_other
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
        w = np.random.normal(size=3)
        # Generating 3 independent random points from the standard Gaussian distribution
        r = np.random.uniform() ** (1 / 3)
        # Sampling the "radius"
        R = np.linalg.norm(w)
        x_loc = np.array(
            [
                r * self.semi_axis_1 * w[0] / R,
                r * self.semi_axis_2 * w[1] / R,
                r * self.semi_axis_3 * w[2] / R,
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

        if isinstance(other_particle, Ellipsoid):
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
        theta = np.linspace(0, np.pi, n_points)
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

    def intersection(self, other_particle: Particle, box: list) -> bool:
        """
        Check if the Ellipsoid intersects the other particle.

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
        if isinstance(other_particle, (Sphere, Ellipsoid)):
            other_particle: Ellipsoid
            # The other particle is also an Ellipsoid or subclass
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
        else:
            raise ValueError("Incompatible particles.")
        return intersection
        # Returning the intersection area


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

    @property
    def volume(self):
        """Volume of the sphere."""
        volume = 4 * np.pi / 3 * self.radius ** 3

        return volume

    def intersection(self, other_particle: Particle, box: list) -> bool:
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
            intersection = self.intersection_sphere_sphere(other_particle, box)
            # Computing the intersection area
        elif isinstance(other_particle, Ellipsoid):
            other_particle: Ellipsoid
            intersection = other_particle.intersection(self, box)
        return intersection
        # Returning the intersection area

    def point_inside(self, point, tol=1e-3):
        """Check if point is inside the particle."""
        point_in = np.linalg.norm(self.position_center - point) - self.radius <= tol

        return point_in

    def intersection_sphere_sphere(self, other_sphere: Sphere, box: list) -> bool:
        """Check if the two spheres intersect."""
        diff_center = self.position_center - other_sphere.position_center
        diff_center = diff_center - box * np.round(diff_center / box)
        # Vector between the centers of the current disk and the nearest image of the other
        # disk
        d = np.sqrt(diff_center.dot(diff_center))
        # Distance between the disks
        intersection = d < (self.radius + other_sphere.radius)
        # The disks are in eachothers neighboorhoods
        return intersection

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
            self.azimuth_angle = descriptors["azimuth_angle"]
        if "polar_angle" in descriptors:
            self.polar_angle = descriptors["polar_angle"]
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

    def intersection(self):
        pass

    def intersection_area(self):
        pass

    def support_function(self, direction: np.array) -> list:

        dir_parallel_comp = (
            direction.dot(self.sym_axis_unit_vec) * self.sym_axis_unit_vec
        )
        dir_normal_comp = direction - dir_parallel_comp
        dir_unit_normal_comp = dir_normal_comp / np.linalg.norm(dir_normal_comp)
        axial_vec_local = (
            self.length
            / 2
            * self.sym_axis_unit_vec
            * np.sign(self.sym_axis_unit_vec.dot(dir_parallel_comp))
            if np.sign(self.sym_axis_unit_vec.dot(dir_parallel_comp)) != 0
            else self.length / 2 * self.sym_axis_unit_vec
        )
        trans_vec_local = self.r_cyl * dir_unit_normal_comp
        point_global = self.position_center + axial_vec_local + trans_vec_local

        return point_global


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
            if (
                not np.any(np.isclose(y_pt * np.ones(len(y_pts)), y_pts))
                or len(y_pts) == 0
            ):
                y_pts.append(y_pt)
                x_pt = A1 * np.sqrt(1 - y_pt ** 2 / B1 ** 2)
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
