"""Module containing the cylinder particle class."""
from __future__ import annotations
from typing import Union

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from .particle import Particle, MINIMUM_SIZE


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
        Rotation matrix from global to local coordinates; see `rotation_from_angles`.

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
            "r_cyl": (
                "Cylinder Radius",
                lambda r_cyl, rve_dims: min(rve_dims) / 4 > r_cyl > MINIMUM_SIZE / 2,
                "float",
            ),
            "length": (
                "Cylinder Length",
                lambda length, rve_dims: min(rve_dims) / 2 > length > MINIMUM_SIZE,
                "float",
            ),
            "azimuth_angle": (
                "Azimuthal angle",
                lambda azimuth_angle, rve_dims: True,
                "float",
            ),
            "polar_angle": ("Polar angle", lambda polar_angle, rve_dims: True, "float"),
            "ratio": (
                "Ratio length/radius",
                lambda ratio, rve_dims: ratio > 0,
                "float",
            ),
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
            When the radius or the length of the cylinder, given or derived, is out of
            its bounds.
        """
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
        self.r_cyl, self.length = cylinder_size(descriptors, rve_dims)
        for i_name, i_value in (("r_cyl", self.r_cyl), ("length", self.length)):
            if not Cylinder.possible_parameters[i_name][1](i_value, rve_dims):
                raise ValueError(
                    "{0} is not an allowable value for {1}, which the descriptors "
                    "give.".format(i_value, i_name)
                )
        # The bounds of the radius and the length given as descriptors hold for the
        # ones the descriptors give through the ratio or a volume. The ratio was used
        # here and in the descriptions, and not declared, so no description by it was
        # accepted. The bounds refuse a radius or a length of zero or less, so neither
        # needs a check of its own
        self.azimuth_angle = descriptors["azimuth_angle"]
        self.polar_angle = descriptors["polar_angle"]
        self.rot_mat = rotation_from_angles(self.polar_angle, self.azimuth_angle)
        # Any angles give an axis, and the cylinder is the same along it either way, so
        # the angles are taken as they come. Their absolute values were taken, which
        # turned the axis of a negative polar angle by half a turn about z, and mirrored
        # that of a negative azimuth: a spread of polar angles about zero leant to one
        # side
        super().__init__(3, phase)

    @property
    def volume(self):
        """Particle volume. Only approximate if *self.delta !=0."""
        volume = (self.length + 2 * self.delta) * np.pi * (self.r_cyl + self.delta) ** 2
        return volume

    @property
    def sym_axis_unit_vec(self):
        """Get unit vector along the cylinder's symmetry axis."""
        return self.rot_mat[0]
        # The first row of the rotation, which is worked out once from the angles. It
        # was worked out from them again at each call, four times in each call of the
        # support function

    @property
    def radius(self):
        """Radius of the circumscribed sphere to the cylinder."""
        radius = np.sqrt((self.length / 2) ** 2 + self.r_cyl ** 2) + self.delta
        return radius

    @property
    def radius_insc(self):
        """Radius of the inscribed sphere to the cylinder."""
        radius_insc = np.min([self.length / 2, self.r_cyl])

        return radius_insc

    def intersection(self, other_particle: Particle, box: list) -> bool:
        """Check for the intersection between *self* and the *other_particle*."""
        intersection = self.intersection_gjk(other_particle, box)
        # Two cylinders are told apart by GJK too, as their overlap is measured by the
        # Minkowski difference. The test of their caps answered with a tuple, which is
        # always true, and took a quarter of the pairs apart for intersecting
        return intersection

    def intersection_area(self, other_particle: Particle, box: list) -> float:
        """Compute the intersection volume between *self* and *other_particle*."""
        intersection_volume = self.intersection_area_monte_carlo(other_particle, box)
        return intersection_volume

    def intersection_length(
        self, other_particle: Particle, box: list, **kwargs
    ) -> Union[float, np.array]:
        """Compute the intersection length between *self* and *other_particle* in *box*.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list(float)
            Dimensions of the simulation box.

        Returns
        -------
        intersection_length: float
            Minimum distance allowing for the removal of the intersection.

        unit_vector: np.array
            Direction of the minimum displacement allowing for the removal of the
            intersection.

        Keyword Parameters
        ------------------
        tol: float
            Tolerance for the computation of the intersection length.

        dist_met: {"dist_approx", "dist_exact"}
            Method used for the intersection length computation. Exact or approximate.
        """
        tol = kwargs.get("tol", 1e-8)
        dist_met = kwargs.get("dist_met", "dist_approx")
        intersection = self.intersection_gjk(other_particle, box)
        if intersection:
            intersection_length, unit_vector = self.intersection_length_mink_diff(
                other_particle, box, tol=tol, dist_met=dist_met
            )
        else:
            intersection_length = 0
            unit_vector = np.array([0, 0, 0])

        return intersection_length, unit_vector

    def support_function(self, direction: np.array) -> np.array:
        """Compute the point of the cylinder's support in *direction*."""
        axis = self.sym_axis_unit_vec
        along = direction.dot(axis)
        normal = direction - along * axis
        normal = normal - normal.dot(axis) * axis
        # Projected off the axis twice. For a direction along the axis the first leaves
        # rounding errors, partly along the axis, and they were scaled to the radius:
        # the point was then up to a radius off the support, above or below the cap
        normal_norm = np.linalg.norm(normal)
        axial_offset = self.length / 2 * (axis if along >= 0 else -axis)
        radial_offset = (
            self.r_cyl * normal / normal_norm if normal_norm != 0 else np.zeros(3)
        )
        dir_unit = direction / np.linalg.norm(direction)

        return (
            self.position_center + axial_offset + radial_offset + self.delta * dir_unit
        )
        # The point is on the rim of the cap that *direction* points to, or of the top
        # cap when it is normal to the axis, pushed out by the dilation

    def contract(self, distance: float):
        """Contract the particle."""
        self.delta -= distance
        # Contracting the particle size subracting the minimum distance from the semi-axis

    def dilate(self, distance: float):
        """Dilate the particle."""
        self.delta += distance
        # Dilating the particle size adding the minimum distance to the semi-axis

    def rescale(self, rescale_parameter):
        """Rescale all size parameters and the position according to *rescale_parameter*."""
        self.r_cyl *= rescale_parameter
        self.length *= rescale_parameter
        super().rescale(rescale_parameter)

    def point_inside(self, point: np.array, box: list) -> bool:
        """Check if *point* is inside *self*."""
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        to_center = self.position_center - point_nearest_pbc
        dist_on_axis = to_center.dot(self.sym_axis_unit_vec)
        dist_to_axis = np.sqrt(
            np.maximum(np.sum(to_center ** 2) - dist_on_axis ** 2, 0.0)
        )
        # The subtraction under the root is held at zero, as it is in `points_inside`. A
        # point on the axis could round it below zero, and the root of that is less than
        # no radius, so the point was left out: a third of the points on the axis of a
        # cylinder turned at random, and the voxels on the axis of one that runs through
        # their centres

        return np.abs(dist_on_axis) <= self.length / 2 and dist_to_axis < self.r_cyl
        # The test of `points_inside` for one point. The test along the axis was split
        # into two branches, and a point with a NaN coordinate took neither and raised

    def points_inside(self, points: np.array, box: list) -> np.array:
        """Say which of *points* are in the Cylinder; see `.Particle.points_inside`."""
        points_nearest_pbc = self.nearest_periodic_images(points, box)
        to_center = self.position_center - points_nearest_pbc
        dist_on_axis = to_center.dot(self.sym_axis_unit_vec)
        dist_to_axis = np.sqrt(
            np.maximum(np.sum(to_center ** 2, axis=1) - dist_on_axis ** 2, 0.0)
        )
        # The distance along the axis of symmetry and the one away from it, the second
        # of them by Pythagoras, held at zero where the subtraction falls below it

        return (np.abs(dist_on_axis) <= self.length / 2) & (dist_to_axis < self.r_cyl)

    def generate_point_inside(self):
        """Generate a random point inside the cylinder."""
        w = np.random.normal(size=2)
        # Generating 3 independent random points from the standard Gaussian distribution
        r = np.random.uniform() ** (1 / 2)
        # Sampling the "radius"
        R = np.linalg.norm(w)
        x_loc = np.array(
            [
                np.random.uniform(-self.length / 2, self.length / 2),
                r * self.r_cyl * w[0] / R,
                r * self.r_cyl * w[1] / R,
            ]
        )
        x_glob = self.rot_mat.T.dot(x_loc) + self.position_center
        # The first row of the rotation is the axis of symmetry, so the point is taken
        # to the global frame by its transpose, with its coordinate along the axis
        # first. It was taken by the rotation itself, with that coordinate last, and
        # most points fell outside the cylinder
        return x_glob


def cylinder_size(descriptors: dict, rve_dims: list) -> tuple[float, float]:
    """
    Give the radius and the length of a cylinder from its descriptors.

    Parameters
    ----------
    descriptors: dict
        Dictionary of the form *{descriptor_name: value}*.

    rve_dims: list
        List containing the dimensions of the microstructure in each direction.

    Returns
    -------
    r_cyl: float
        Radius of the cylinder.

    length: float
        Length of the cylinder.

    Raises
    ------
    ValueError:
        When the descriptors give neither the radius nor the length, or give one of
        them without what the other follows from.
    """
    if "length" in descriptors and "vf" in descriptors and "n" in descriptors:
        length = descriptors["length"]
        r_cyl = np.sqrt(
            descriptors["vf"] * np.prod(rve_dims) / (length * np.pi * descriptors["n"])
        )
        return r_cyl, length
    if "length" in descriptors and "ratio" in descriptors:
        return descriptors["length"] / descriptors["ratio"], descriptors["length"]
    if "length" in descriptors and "r_cyl" in descriptors:
        return descriptors["r_cyl"], descriptors["length"]
    if "r_cyl" in descriptors and "ratio" in descriptors:
        return descriptors["r_cyl"], descriptors["r_cyl"] * descriptors["ratio"]
    raise ValueError(
        "The descriptors {0} give neither the radius and the length of a cylinder, "
        "nor one of them and what the other follows from.".format(sorted(descriptors))
    )
    # The cases are tried in the order the constructor used to work them out in, so
    # that a length with a volume fraction and a number of particles, or with a
    # ratio, gives the radius even where a radius is given too


def rotation_from_angles(polar_angle: float, azimuth_angle: float) -> np.ndarray:
    """
    Give the rotation from global coordinates to those of a cylinder.

    The rows are the unit vectors of the cylinder's frame, in global coordinates:
    the axis of symmetry, then the directions in which the polar angle and the
    azimuth angle grow. The transpose takes coordinates in the cylinder's frame back
    to the global frame.

    Parameters
    ----------
    polar_angle: float
        Angle of the axis of symmetry from the positive z semi-axis.

    azimuth_angle: float
        Angle of the projection of the axis on the xy plane from the positive x
        semi-axis.

    Returns
    -------
    array, shape (3, 3)
        Rotation matrix.
    """
    return np.array(
        [
            [
                np.sin(polar_angle) * np.cos(azimuth_angle),
                np.sin(polar_angle) * np.sin(azimuth_angle),
                np.cos(polar_angle),
            ],
            [
                np.cos(polar_angle) * np.cos(azimuth_angle),
                np.cos(polar_angle) * np.sin(azimuth_angle),
                -np.sin(polar_angle),
            ],
            [
                -np.sin(azimuth_angle),
                np.cos(azimuth_angle),
                0,
            ],
        ]
    )
