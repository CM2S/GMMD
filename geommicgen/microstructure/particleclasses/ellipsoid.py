"""Module containing Ellipoid particle class."""
from __future__ import annotations

import numpy as np

from scipy import integrate

import microstructure.particleclasses.cylinder as cyl_cls
from .particle import Particle, MINIMUM_SIZE


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
            "axis_1": (
                "Axis 1",
                lambda axis_1, rve_dims: min(rve_dims) / 2 > axis_1 > MINIMUM_SIZE,
                "float",
            ),
            "semi_axis_1": (
                "Semi-Axis 1",
                lambda semi_axis_1, rve_dims: min(rve_dims) / 4
                > semi_axis_1
                > MINIMUM_SIZE / 2,
                "float",
            ),
            "axis_2": (
                "Axis 2",
                lambda axis_2, rve_dims: min(rve_dims) / 2 > axis_2 > MINIMUM_SIZE,
                "float",
            ),
            "axis_3": (
                "Axis 3",
                lambda axis_3, rve_dims: min(rve_dims) / 2 > axis_3 > MINIMUM_SIZE,
                "float",
            ),
            "rot_axis_comp_x": (
                "x-component rotation axis",
                lambda rot_axis_comp_x, rve_dims: True,
                "float",
            ),
            "rot_axis_comp_y": (
                "y-component rotation axis",
                lambda rot_axis_comp_y, rve_dims: True,
                "float",
            ),
            "rot_axis_comp_z": (
                "z-component rotation axis",
                lambda rot_axis_comp_z, rve_dims: True,
                "float",
            ),
            "angle": (
                "Rotation angle",
                lambda rot_axis_comp_y, rve_dims: True,
                "float",
            ),
            "ratio_12": (
                "Ratio a1/a2",
                lambda ratio_12, rve_dims: ratio_12 >= 1,
                "float",
            ),
            "ratio_21": (
                "Ratio a2/a1",
                lambda ratio_21, rve_dims: ratio_21 <= 1,
                "float",
            ),
            "ratio_13": (
                "Ratio a1/a3",
                lambda ratio_13, rve_dims: ratio_13 >= 1,
                "float",
            ),
            "ratio_32": (
                "Ratio a3/a2",
                lambda ratio_31, rve_dims: ratio_31 <= 1,
                "float",
            ),
            "ratio_321": (
                "Ratio a3/a1 and a2/a1",
                lambda ratio_321, rve_dims: ratio_321 <= 1,
                "float",
            ),
            "p_3": (
                "Angle a1 makes with XY",
                lambda rot_axis_comp_y, rve_dims: True,
                "float",
            ),
            "phi_z": (
                "Angle that the projection of a1 in XY makes with Y",
                lambda rot_axis_comp_y, rve_dims: True,
                "float",
            ),
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
        self.check_if_descriptor_values_are_valid(descriptors, rve_dims)
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
        """Volume of the ellipsoid. Only approximate if *self.delta*!=0."""
        volume = (
            4
            / 3
            * np.pi
            * (self.semi_axis_1 + self.delta)
            * (self.semi_axis_2 + self.delta)
            * (self.semi_axis_3 + self.delta)
        )

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

    def point_inside(self, point, box, **kwargs):
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
        tol = kwargs.get("tol", 1e-6)
        position = kwargs.get("position", "inside")
        # Collecting keyword arguments
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
        """Generate a random point inside the ellipsoid.

        Only approximate if the ellipsoid is dilated (self.delta != 0).
        """
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
        elif isinstance(other_particle, (cyl_cls.Cylinder, Ellipsoid)):
            intersection = self.intersection_gjk(other_particle, box)
            if intersection:
                overlap_volume = self.intersection_area_monte_carlo(other_particle, box)
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

    def intersection(self, other_particle: Particle, box: list) -> bool:
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
        if isinstance(other_particle, Ellipsoid) and False:
            other_particle: Ellipsoid
            # The other particle is also an Ellipsoid or subclass
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
        elif isinstance(other_particle, cyl_cls.Cylinder) or True:
            intersection = self.intersection_gjk(other_particle, box)
        else:
            raise ValueError("Incompatible particles.")
        return intersection
        # Returning the intersection area

    def support_function(self, direction):
        """Support function for the ellipsoid."""
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
        self, other_particle: Particle, box: list, **kwargs
    ) -> tuple[float, np.array]:
        """Intersection length between *self* and *other_particle* on *box*."""
        dist_met = kwargs.get("dist_met", "dist_exact")
        if isinstance(other_particle, Ellipsoid) and False:
            intersection = self.intersection_ellipsoid_ellipsoid(other_particle, box)
            # Saving the class name of the other particle as a string
        elif isinstance(other_particle, cyl_cls.Cylinder) or True:
            intersection = self.intersection_gjk(other_particle, box)
        if intersection:
            intersection_length, unit_vector = self.intersection_length_mink_diff(
                other_particle, box, dist_met=dist_met
            )
        else:
            intersection_length = 0
            unit_vector = np.array([0, 0, 0])
        return intersection_length, unit_vector

    def rescale(self, rescale_parameter):
        """Rescale all size parameters and the position according to *rescale_parameter*."""
        self.axis_1 *= rescale_parameter
        self.axis_2 *= rescale_parameter
        self.axis_3 *= rescale_parameter
        self.position_center *= rescale_parameter
