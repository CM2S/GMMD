"""Module containing the Sphere particle class."""
from __future__ import annotations
from typing import Union

import numpy as np

import microstructure.particleclasses.cylinder as cyl_cls
from .ellipsoid import Ellipsoid
from .particle import Particle, MINIMUM_SIZE


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
        **{
            "r": (
                "Radius",
                lambda r, rve_dims: MINIMUM_SIZE / 2 < r < min(rve_dims) / 4,
            ),
            "volume": (
                "Volume per particle",
                lambda volume, rve_dims: 4 / 3 * np.pi * (MINIMUM_SIZE / 2) ** 3
                < volume
                < 4 / 3 * np.pi * (min(rve_dims) / 4) ** 3,
            ),
        },
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
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
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

    @property
    def volume(self):
        """Volume of the sphere."""
        volume = 4 * np.pi / 3 * (self.radius + self.delta) ** 3

        return volume

    def intersection(self, other_particle: Particle, box: list, **kwargs) -> bool:
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

        Keyword Parameters
        ------------------
        inside: bool
            Consider a sphere inside if it completly inside the other.
        """
        inside = kwargs.get("inside", True)
        if isinstance(other_particle, Sphere):
            # The other particle is also a Disk
            intersection = self.intersection_sphere_sphere(
                other_particle, box, inside=inside
            )
            # Computing the intersection area
        elif isinstance(other_particle, Ellipsoid):
            other_particle: Ellipsoid
            intersection = other_particle.intersection(self, box)
        elif isinstance(other_particle, cyl_cls.Cylinder):
            other_particle: cyl_cls.Cylinder
            intersection, _ = self.intersection_sphere_cylinder(other_particle, box)
        else:
            intersection = self.intersection_gjk(other_particle, box)

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
        self: Sphere, cylinder: cyl_cls.Cylinder, box: list
    ) -> Union[bool, float]:
        """Detect the intersection between a sphere and a cylinder."""
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
        self, other_particle: Particle, box: list, **kwargs
    ) -> tuple[float, np.array]:
        """
        Compute the intersection length between the Sphere and the other particle.

        Parameters
        ----------
        other_particle: `.Particle`
            Other particle

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        overlap_length: float
            Intersection length between the particles.

        overlap_dir: 1-array, shape (3)
            Intersection direction between the particles.
        """
        dist_met = kwargs.get("dist_met", "dist_exact")
        if isinstance(other_particle, Sphere):
            # The other particle is also a Sphere
            overlap_length = self.intersection_length_sphere_sphere(other_particle, box)
            overlap_dir = self.intersection_vector(other_particle, box)
            # Computing the intersection length
        else:
            if isinstance(other_particle, cyl_cls.Cylinder):
                # The other particle is a cylinder
                other_particle: cyl_cls.Cylinder
                intersection, _ = self.intersection_sphere_cylinder(other_particle, box)
            else:
                intersection = self.intersection_gjk(other_particle, box)
            if intersection:
                overlap_length, overlap_dir = self.intersection_length_mink_diff(
                    other_particle, box, dist_met=dist_met
                )
            else:
                overlap_length = 0
                overlap_dir = np.array([0, 0, 0])
        return overlap_length, overlap_dir

    def intersection_length_sphere_sphere(
        self, other_sphere: Sphere, box: list
    ) -> float:
        """
        Compute the intersection length between two Spheres.

        Parameters
        ----------
        other_sphere: `.Sphere`
            Other sphere whose intersection length with the current sphere we want to know

        box: list
            Dimensions of the simulation box.

        Returns
        -------
        intersection_length: float
            Intersection length between the particles.
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

    def point_inside(self, point: np.array, box: list, **kwargs) -> bool:
        """Check if point is inside the particle."""
        tol = kwargs.get("tol", 1e-3)
        point_nearest_pbc = Particle.nearest_periodic_image(
            point, self.position_center, box
        )
        point_in = (
            np.linalg.norm(self.position_center - point_nearest_pbc) - self.radius
            <= tol
        )

        return point_in

    def generate_points_on_surface(
        self, n_points: int, erosion_thick: float = 0
    ) -> np.array:
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

    def compute_critical_erosion_thickness(self) -> float:
        """Compute the critical erosion thickness for a sphere."""
        erosion_thickness = 0.9 * self.radius
        # Semi-latus rectum
        return erosion_thickness

    def support_function(self, direction: np.array) -> np.array:
        """Get point of the support function in the direction *direction* ofr a sphere."""
        return (
            self.position_center + direction / np.linalg.norm(direction) * self.radius
        )
