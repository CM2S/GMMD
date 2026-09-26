"""Module containing the cylinder particle class."""
from __future__ import annotations
from typing import Union

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
import geommicgen.microstructure.particleclasses.sphere as sph_cls
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
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
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
        """Particle volume. Only approximate if *self.delta !=0."""
        volume = (self.length + 2 * self.delta) * np.pi * (self.r_cyl + self.delta) ** 2
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
        if False and isinstance(other_particle, sph_cls.Sphere):
            other_particle: sph_cls.Sphere
            _, intersection_length = other_particle.intersection_sphere_cylinder(
                self, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
        else:
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
        dist_on_axis = self.sym_axis_unit_vec.dot(
            self.position_center - point_nearest_pbc
        )
        if np.abs(dist_on_axis) > self.length / 2:
            point_inside = False

        elif np.abs(dist_on_axis) <= self.length / 2:
            L = np.sqrt(
                np.maximum(
                    np.sum((self.position_center - point_nearest_pbc) ** 2)
                    - dist_on_axis ** 2,
                    0.0,
                )
            )
            point_inside = L < self.r_cyl
            # The subtraction under the root is held at zero, as it is in
            # `points_inside`. A point on the axis could round it below zero, and the
            # root of that is less than no radius, so the point was left out: a third
            # of the points on the axis of a cylinder turned at random, and the voxels
            # on the axis of one that runs through their centres

        return point_inside

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
