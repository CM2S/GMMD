"""Module containing the Ellipse particle class."""
from __future__ import annotations

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from .particle import Particle, MINIMUM_SIZE


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
            "major_axis": (
                "Major axis",
                lambda major_axis, rve_dims: min(rve_dims) / 2
                > major_axis
                > MINIMUM_SIZE,
                "float",
            ),
            "minor_axis": (
                "Minor axis",
                lambda minor_axis, rve_dims: min(rve_dims) / 2
                > minor_axis
                > MINIMUM_SIZE,
                "float",
            ),
            "angle": ("Angle", lambda angle, rve_dims: True, "float"),
            "eccentricity": (
                "Eccentricity",
                lambda eccentricity, rve_dims: eccentricity >= 0,
                "float",
            ),
            "ratio": ("Ratio a/b", lambda ratio, rve_dims: ratio >= 1, "float"),
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
        Initialize a classe Ellipse object.

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
        """Area(volume) of the real ellipse."""
        volume = (
            np.pi
            * (self.semi_major_axis)
            * (self.semi_minor_axis)
        )
        return volume


    @property
    def real_volume(self):
        """Real area(volume) of the ellipse."""
        volume = (
            np.pi
            * (self.semi_major_axis - self.delta)
            * (self.semi_minor_axis - self.delta)
        )

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

    @property
    def radius_insc(self):
        """Radius of the inscribed circle to the ellipse."""
        radius_insc = self.semi_minor_axis

        return radius_insc

    def contract(self, distance):
        """Contract the particle."""
        self.delta -= distance
        self.major_axis -= 2 * distance
        self.minor_axis -= 2 * distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance):
        """Dilate the particle."""
        self.delta += distance
        self.major_axis += 2 * distance
        self.minor_axis += 2 * distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def point_inside(self, point: np.array, box: list, **kwargs) -> bool:
        """
        Check if the point is inside, outside or on the ellipse given a tolerance.

        Only an approximation if the ellipse is dilate (self.delta != 0).

        Parameters
        ---------
        self: `.Ellipse`
            Ellipse under analysis

        point: array
            Point under analysis

        Returns
        -------
        point_in: bool
            True if the point is inside the ellipse and False otherwise.

        Keyword Arguments
        -----------------
        tol: float
            Tolerance

        position: string
            'inside' or 'on'
        """
        # Collecting keyword arguments
        position = kwargs.get("position", "inside")
        tol = kwargs.get("position", 1e-8)
        # Defininig the radius vector relative to the coordinate system of the ellipse
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        r_vector = self.rot_mat.dot(point_nearest_pbc - self.position_center)
        if position == "inside":
            # Checking if the point is inside the ellipse
            point_in = (r_vector[0] / self.semi_major_axis) ** 2 + (
                r_vector[1] / self.semi_minor_axis
            ) ** 2 <= 1 + tol
        elif position == "on":
            # Checking if the point is on the ellipse
            point_in = (
                np.abs(
                    (r_vector[0] / self.semi_major_axis) ** 2
                    + (r_vector[1] / self.semi_minor_axis) ** 2
                    - 1
                )
                < tol
            )
        return point_in

    def intersection_area_ellipse_ellipse(
        self, other_ellipse: Ellipse, box: list
    ) -> float:
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

    def midpoint_on_ellipse(self, point_1: np.array, point_2: np.array) -> np.array:
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

    def sort_points_on_ellipse(self, points: list(np.array)) -> list(float):
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

    def area_ellipse_section(
        self, intersect_pt_1: np.array, intersect_pt_2: np.array
    ) -> float:
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

    def intersection_area(self, other_particle: Particle, box: list) -> float:
        """
        Compute the intersection area between the ellipse and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.
        """
        if isinstance(other_particle, Ellipse):
            other_particle: Ellipse
            intersection_area = self.intersection_area_ellipse_ellipse(
                other_particle, box
            )
        else:
            raise ValueError(
                "Compatible particle pair: {0} and {1}".format(self, other_particle)
            )
        return intersection_area

    def intersection_ellipse_ellipse(
        self, other_ellipse: Ellipse, box: list, inside=True
    ) -> bool:
        """
        Check if this ellipse intersects the other ellipse.

        Parameters
        ----------
        other_ellipse: `.Ellipse`
            Other ellipse that may intersect *self*.

        box: list(float)
            Dimensions of the simulation box.

        inside: optional, bool
            If inside is True, when one ellipse is completly inside the other is counted as
            an interseciton.

        Returns
        -------
        intersection_bool: bool
            True if the two ellipses intersect.
        """
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
            # There are intersection points betweeen the two neighborhoods
            intersection_bool = True
        else:
            if inside:
                # Either the ellipses are disjoint or one of them is completly inside the
                # other
                if self.volume >= other_ellipse.volume:
                    # The current ellipse is larger than the other ellipse
                    intersection_bool = self.point_inside(
                        other_ellipse.position_center, box
                    )
                else:
                    intersection_bool = other_ellipse.point_inside(
                        self.position_center, box
                    )
            else:
                intersection_bool = False

        return intersection_bool

    def intersection(self, other_particle: Particle, box: list) -> bool:
        """Check this ellipses intersects the other particle."""
        if isinstance(other_particle, Ellipse) and False:
            other_particle: Ellipse
            intersection_bool = self.intersection_ellipse_ellipse(other_particle, box)
        else:
            intersection_bool = self.intersection_gjk(other_particle, box)
        return intersection_bool

    def generate_points_on_surface(
        self, n_points: int, erosion_thick: float = 0
    ) -> list(np.array):
        """
        Generate *n_points* on the surface of the ellipse.

        Parameters
        ----------
        n_points: int
            Numbers of points to generated on the boundary of the ellipse.

        erosion_thick: optional, float
            Erosion to be applied to the ellipse, i.e., perpendicular distance to the
            boundary by which the ellipse will be shrinked.

        Returns
        -------
        points_glob: list(np.array), len (n_points)
            List of the points on the boundary of the elllipse.
        """
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
        # Semi-latus rectum, or the smallest radius of curvature
        return erosion_thickness

    def uniform_sample_ellipse(self, n_samples: int = 1) -> list(np.array):
        """Generate uniform random sample of points inside an ellipse.

        Only approximate if the ellipse is dilated (self.delta != 0).
        """
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

    def regular_sample_ellipse(self, n_samples: int = 1) -> list(np.array):
        """Generate a regular grid of points inside the ellipse."""
        n_theta = int(np.sqrt(n_samples ** (1)))
        n_r = int(np.round(n_samples / n_theta))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        radii = np.linspace(0.01, 1, n_r, endpoint=True)
        points = []
        for i_theta, j_radius in [
            (i_theta, j_radius) for i_theta in theta for j_radius in radii
        ]:
            [x_loc, y_loc] = [
                j_radius * self.semi_major_axis * np.cos(i_theta),
                j_radius * self.semi_minor_axis * np.sin(i_theta),
            ]
            [x_glob, y_glob] = self.rot_mat.dot([x_loc, y_loc]) + self.position_center
            points.append(np.array([x_glob, y_glob]))

        return points

    def generate_point_inside(self):
        """Generate a random point inside the ellipse.

        Only approximate if the ellipse is dilated (self.delta !=0).
        """
        return self.uniform_sample_ellipse()[0]

    def intersection_length(
        self, other_particle: Particle, box: list, **kwargs
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

        Keyword Parameters
        ------------------
        dist_met: {"dist_approx", "dist_exact"}
            Method used for the intersection length computation. Exact or approximate.
        """
        dist_met = kwargs.get("dist_met", "dist_exact")
        if True:
            intersection = self.intersection_gjk(other_particle, box)
        elif isinstance(other_particle, Ellipse):
            other_particle: Ellipse
            intersection = self.intersection_ellipse_ellipse(other_particle, box)
        if intersection:
            intersection_length, unit_vector = self.intersection_length_mink_diff(
                other_particle, box, dist_met=dist_met
            )
        else:
            intersection_length = 0
            unit_vector = np.array([0, 0])

        return intersection_length, unit_vector
        # Returning the intersection length

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
