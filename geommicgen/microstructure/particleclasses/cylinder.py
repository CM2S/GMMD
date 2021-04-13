"""Module containing the cylinder particle class."""
from __future__ import annotations
from typing import Union

import numpy as np

import microstructure.particleclasses.sphere as sph_cls
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
        if isinstance(other_particle, Cylinder):
            other_particle: Cylinder
            intersection = self.intersection_cylinder_cylinder(other_particle, box)
        else:
            intersection = self.intersection_gjk(other_particle, box)
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
        if False and isinstance(other_particle, Cylinder):
            other_particle: Cylinder
            _, intersection_length = self.intersection_cylinder_cylinder(
                other_particle, box
            )
            unit_vector = self.intersection_vector(other_particle, box)
        elif False and isinstance(other_particle, sph_cls.Sphere):
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
        self.position_center *= rescale_parameter

    def intersection_cylinder_cylinder(
        self: Cylinder, other_cylinder: Cylinder, box: list
    ) -> Union[bool, float]:
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
    ) -> Union[bool, float]:
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
        ) -> Union[bool, float]:
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
    ) -> Union[bool, float]:
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
            point_inside = L < self.r_cyl

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
