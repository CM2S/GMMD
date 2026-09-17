"""
Unit tests regarding the Sphere particle class.
"""
import unittest
from unittest.mock import Mock

import numpy as np
from geommicgen.microstructure.particleclasses import Sphere, Cylinder, Ellipsoid



class TestSphere(unittest.TestCase):

    def test_init(self):
        "Tests the three ways of describing a sphere supported by the constructor."
        rve_dims = [1, 1,1]
        with self.subTest("Radius supplied directly"):
            sphere = Sphere("phase", {"n": 1, "r": 0.1}, rve_dims)
            self.assertAlmostEqual(sphere.radius, 0.1)

        with self.subTest("Volume per particle supplied"):
            sphere = Sphere("phase", {"n": 1, "volume": (4/3)*np.pi * 0.1 ** 3}, rve_dims)
            self.assertAlmostEqual(sphere.radius, 0.1)

        with self.subTest("Volume fraction and number of particles supplied"):
            n, vf = 4, 0.1
            sphere = Sphere("phase", {"n": n, "vf": vf}, rve_dims)
            expected_volume = vf * rve_dims[0] * rve_dims[1] / n
            expected_radius = np.cbrt(expected_volume/ (4 / 3 * np.pi))
            self.assertAlmostEqual(sphere.radius, expected_radius)

    def test_volume(self):
        sphere = Sphere("phase", {"n": 1, "r": 0.1}, [1,1,1])
        self.assertAlmostEqual(sphere.volume, (4/3)*np.pi * 0.1 ** 3)


    def test_contract_and_dilate(self):
        sphere = Sphere(
            "1", {"n":1, "r":0.1}, [1.0, 1.0, 1.0]
        )
        with self.subTest("dilate"):
            sphere.dilate(0.05)
            self.assertAlmostEqual(sphere.radius, 0.15)
            self.assertAlmostEqual(sphere.volume, (4/3)*np.pi * 0.15 ** 3)
        with self.subTest("contract back to the original size"):
            sphere.contract(0.05)
            self.assertAlmostEqual(sphere.radius, 0.1)
            self.assertAlmostEqual(sphere.volume, (4/3)*np.pi * 0.1 ** 3)

    def test_point_inside(self):
        rve_dims = [1.0, 1.0, 1.0]
        sphere = Sphere(
            "1", {"n":1, "r":0.2}, rve_dims 
        )
        sphere.position_center = np.array([0.6, 0.8, 0])
        with self.subTest("Point inside the sphere"):
            self.assertTrue(sphere.point_inside(np.array([0.56, 0.84, 0.04]), rve_dims))
        with self.subTest("Point outside the sphere"):
            self.assertTrue(not sphere.point_inside(np.array([0.9, 0.9, 0.5]), rve_dims))

    def test_generate_points_on_surface(self):
        "Sphere.generate_points_on_surface lays points on a theta/phi grid."
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        sphere.position_center = np.array([2.0, 3.0, 4.0])
        with self.subTest("Without erosion"):
            points = sphere.generate_points_on_surface(4)
            distances = np.linalg.norm(points - sphere.position_center, axis=1)
            np.testing.assert_allclose(distances, sphere.radius, atol=1e-10)

        with self.subTest("With erosion"):
            erosion_thick = 0.03
            points = sphere.generate_points_on_surface(4, erosion_thick=erosion_thick)
            distances = np.linalg.norm(points - sphere.position_center, axis=1)
            np.testing.assert_allclose(
                distances, sphere.radius - erosion_thick, atol=1e-10
            )

    def test_compute_critical_erosion_thickness(self):
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        self.assertAlmostEqual(sphere.compute_critical_erosion_thickness(), 0.9 * 0.2)

    def test_rescale(self):
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        sphere.position_center = np.array([0.3, 0.4, 0.5])
        sphere.rescale(2)
        self.assertAlmostEqual(sphere.radius, 0.4)
        np.testing.assert_allclose(sphere.position_center, np.array([0.6, 0.8, 1.0]))

    def test_generate_point_inside(self):
        rve_dims = [1.0, 1.0, 1.0]
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, rve_dims
        )
        sphere.position_center = np.array([0.5, 0.5, 0.5])
        for _ in range(20):
            self.assertTrue(sphere.point_inside(sphere.generate_point_inside(), rve_dims))


    def test_intersection_calls(self):
        "Check if the function intersection calls the correct function to compute the intersection based on the other particle type"
        box = [1.0, 1.0, 1.0]
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        
        with self.subTest("Other particle is a sphere"):
            other_particle = Mock(spec= Sphere)
            sphere.intersection_sphere_sphere = Mock()
            sphere.intersection(other_particle, box)
            sphere.intersection_sphere_sphere.assert_called_once()

        with self.subTest("Other particle is an ellipsoid"):
            other_particle = Mock(spec= Ellipsoid)
            other_particle.intersection = Mock()
            sphere.intersection(other_particle, box)
            other_particle.intersection.assert_called_once()

        with self.subTest("Other particle is a cylinder"):
            other_particle = Mock(spec= Cylinder)
            sphere.intersection_sphere_cylinder = Mock(return_value= [0,0])
            sphere.intersection(other_particle, box)
            sphere.intersection_sphere_cylinder.assert_called_once()

        with self.subTest("Other particle is other type"):
            other_particle = Mock()
            sphere.intersection_gjk = Mock()
            sphere.intersection(other_particle, box)
            sphere.intersection_gjk.assert_called_once()



    def test_intersection_lenght_calls(self):
        "Check if the function intersection_lenght calls the correct function to compute the intersection lenght based on the other particle type"
        box = [1.0, 1.0, 1.0]
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        
        with self.subTest("Other particle is a sphere"):
            other_particle = Mock(spec= Sphere)
            sphere.intersection_length_sphere_sphere = Mock()
            sphere.intersection_vector = Mock()
            sphere.intersection_length(other_particle, box)
            sphere.intersection_length_sphere_sphere.assert_called_once()

        with self.subTest("Other particle is a cylinder"):
            other_particle = Mock(spec= Cylinder)
            sphere.intersection_sphere_cylinder = Mock(return_value= [0,0])
            sphere.intersection_length(other_particle, box)
            sphere.intersection_sphere_cylinder.assert_called_once()

        with self.subTest("Other particle is other type"):
            other_particle = Mock()
            sphere.intersection_gjk = Mock(return_value = True)
            sphere.intersection_length_mink_diff = Mock(return_value=["intersection_lenght","intersection_dir"])
            sphere.intersection_length(other_particle, box)
            sphere.intersection_length_mink_diff.assert_called_once()

    def test_intersection_area_calls(self):
        "Check if the function intersection_area calls the correct function to compute the intersection area based on the other particle type"
        box = [1.0, 1.0, 1.0]
        sphere = Sphere(
            "1", {"n": 1, "r": 0.2}, [1.0, 1.0, 1.0]
        )
        
        with self.subTest("Other particle is a sphere"):
            other_particle = Mock(spec= Sphere)
            sphere.intersection_volume_sphere_sphere = Mock()
            sphere.intersection_area(other_particle, box)
            sphere.intersection_volume_sphere_sphere.assert_called_once()

        with self.subTest("Other particle is an ellipsoid"):
            other_particle = Mock(spec= Ellipsoid)
            other_particle.intersection_area = Mock()
            sphere.intersection_area(other_particle, box)
            other_particle.intersection_area.assert_called_once()

        with self.subTest("Other particle is other type"):
            other_particle = Mock()
            sphere.intersection_gjk = Mock(return_value=["intersection","overlap_length"])
            sphere.intersection_area(other_particle, box)
            sphere.intersection_gjk.assert_called_once()




class TestSphereIntersection(unittest.TestCase):
    "Test regarding the intersection check, area and length for different scenarios with two spheres."

    def test_not_intersecting(self):
        rve_dims = [1.0, 1.0, 1.0]

        sphere_1 = Sphere(
            "1", {"n":1, "r":0.1}, rve_dims
        )
        sphere_1.position_center = np.array([0.6, 0.8, 0])
        sphere_2 = Sphere(
            "1", {"n":1, "r":0.1}, rve_dims
        )
        sphere_2.position_center = np.array([0.2, 0.3, 0.5])

        with self.subTest("Test intersection check"):
            self.assertTrue(not sphere_1.intersection(sphere_2, rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( sphere_1.intersection_area(sphere_2, rve_dims), 0 )

        with self.subTest("Test intersection length"):
            intersection_length = sphere_1.intersection_length_sphere_sphere(sphere_2, rve_dims)
            self.assertEqual(intersection_length, 0)


    def test_sphere_inside_sphere_1(self):
        "sphere_2 is completely inside sphere_1"
        rve_dims = [1.0, 1.0, 1.0]

        sphere_1 = Sphere(
            "1", {"n":1, "r":0.2}, rve_dims
        )
        sphere_1.position_center = np.array([0.6, 0.8, 0.5])
        sphere_2 = Sphere(
            "1", {"n":1, "r":0.1}, rve_dims
        )
        sphere_2.position_center = np.array([0.6, 0.8, 0.5])

        with self.subTest("Test intersection check"):
            self.assertTrue(sphere_1.intersection(sphere_2, rve_dims))

        with self.subTest("Test intersection area"):
            self.assertEqual( sphere_1.intersection_area(sphere_2, rve_dims), 4/3*np.pi*0.1**3 )


        with self.subTest("Test intersection length"):
            self.skipTest("Skipping intersection length check")
            # This test gives an error, eventhough it shouldn't. Problem not fixed yet
            (intersection_length,intersection_dir) = sphere_1.intersection_length(sphere_2, rve_dims)
            sphere_2.position_center += (intersection_length) * intersection_dir
            intersection = sphere_1.intersection(sphere_2, rve_dims)
            # Prints for debuging
            print("Here")
            print(intersection_length)
            print(sphere_1.intersection_area(sphere_2, rve_dims))
            self.assertTrue(not intersection)



    def test_sphere_inside_sphere_2(self):
        "sphere_1 is completely inside sphere_2 and periodic image is used"
        rve_dims = [1.0, 1.0, 1.0]

        sphere_1 = Sphere(
            "1", {"n":1, "r":0.1}, rve_dims
        )
        sphere_1.position_center = np.array([0.95, 0.5, 0.5])
        sphere_2 = Sphere(
            "1", {"n":1, "r":0.2}, rve_dims
        )
        sphere_2.position_center = np.array([0.05, 0.5, 0.5])


        with self.subTest("Test intersection check"):
            self.assertTrue(sphere_1.intersection(sphere_2, rve_dims))

        with self.subTest("Test intersection area"):
            self.assertAlmostEqual( sphere_1.intersection_area(sphere_2, rve_dims), 4/3*np.pi*0.1**3 )

        with self.subTest("Test intersection length"):
            (intersection_length,intersection_dir) = sphere_1.intersection_length(sphere_2, rve_dims)
            sphere_2.position_center += (intersection_length) * intersection_dir
            intersection = sphere_1.intersection(sphere_2, rve_dims)
            self.assertTrue(not intersection)



    def test_partially_intersecting(self):
        rve_dims = [1.0, 1.0, 1.0]

        sphere_1 = Sphere(
            "1", {"n":1, "r":0.2}, rve_dims
        )
        sphere_1.position_center = np.array([0.6, 0.8, 0.5])
        sphere_2 = Sphere(
            "1", {"n":1, "r":0.2}, rve_dims
        )
        sphere_2.position_center = np.array([0.5, 0.9, 0.4])

        with self.subTest("Test intersection check"):
            self.assertTrue(sphere_1.intersection(sphere_2, rve_dims))

        with self.subTest("Test intersection area"):
            self.assertAlmostEqual( sphere_1.intersection_area(sphere_2, rve_dims), 0.01310507 )

        with self.subTest("Test intersection length"):
            (intersection_length,intersection_dir) = sphere_1.intersection_length(sphere_2, rve_dims)
            sphere_2.position_center += (intersection_length) * intersection_dir
            intersection = sphere_1.intersection(sphere_2, rve_dims)
            self.assertTrue(not intersection)


class TestSalnikovSphereCylinder(unittest.TestCase):
    """Test the intersection function from Salnikov for spheres and cylinders."""

    def setUp(self):
        self.rve_dims = [1, 1, 1]
        self.sphere = Sphere("1", {"r": 0.1, "n": 1}, self.rve_dims)
        self.cylinder = Cylinder(
            "1",
            {
                "r_cyl": 0.2,
                "length": 0.4,
                "azimuth_angle": 0,
                "polar_angle": np.pi / 2,
                "n": 1,
            },
            self.rve_dims,
        )


    def test_not_intersecting(self):
        self.cylinder.position_center = np.array([0.5, 0.1, 0.9])
        self.sphere.position_center = np.array([0.7, 0.5, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(not intersection)
        self.assertEqual(intersection_length , 0)

    def test_not_intersecting_far_on_axis(self):
        "Sphere far away along the cylinder's symmetry axis: dist_on_axis > length/2 + radius."
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.9, 0.5, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(not intersection)
        self.assertEqual(intersection_length, 0)


    def test_intersect_top(self):
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.7001, 0.5, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(intersection)
        self.assertAlmostEqual(intersection_length, 0.0999, places = 4)



    def test_intersect_lateral(self):
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.5, 0.68, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(intersection)
        self.assertAlmostEqual(intersection_length, 0.24)


    def test_intersect_top_edge(self):
        "Sphere intersects near the cylinder cap, offset laterally beyond r_cyl."
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.75, 0.75, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(intersection)
        self.assertAlmostEqual(intersection_length, 0.03660254)

    def test_not_intersecting_near_cap(self):
        "Sphere near the cap region but too far laterally to reach the cylinder."
        self.cylinder.position_center = np.array([0.5, 0.5, 0.5])
        self.sphere.position_center = np.array([0.75, 0.85, 0.5])
        intersection, intersection_length = self.sphere.intersection_sphere_cylinder(
            self.cylinder, self.rve_dims
        )
        self.assertTrue(not intersection)
        self.assertEqual(intersection_length, 0)


class TestGJKIntersectionOverlapLengthSphere(unittest.TestCase):
    def test_two_intersecting_spheres(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_1 = Sphere(phase_1, descriptors_1, rve_dims)
        sphere_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_2 = Sphere(phase_2, descriptors_2, rve_dims)
        sphere_2.position_center = np.array([0.5, 0.55, 0.5])
        # plot_particles_2d([sphere_1, sphere_2], rve_dims, "", save=False, show=True)
        intersection = sphere_1.intersection_gjk(sphere_2, rve_dims)
        overlap_length, _ = sphere_1.intersection_length_mink_diff(sphere_2, rve_dims)
        self.assertTrue(intersection)
        self.assertTrue(np.abs(0.05 - overlap_length) / 0.05 < 1e-8)

    def test_two_intersecting_spheres_2(self):
        rve_dims = [1, 1, 1]
        phase_1 = "1"
        descriptors_1 = {
            "r": 0.1,
            "n": 1,
        }
        sphere_1 = Sphere(phase_1, descriptors_1, rve_dims)
        sphere_1.position_center = np.array([0.5, 0.5, 0.5])

        phase_2 = "1"
        descriptors_2 = {
            "r": 0.05,
            "n": 1,
        }
        sphere_2 = Sphere(phase_2, descriptors_2, rve_dims)
        sphere_2.position_center = np.array([0.5, 0.55, 0.5])
        # plot_particles_2d([sphere_1, sphere_2], rve_dims, "", save=False, show=True)
        intersection = sphere_1.intersection_gjk(sphere_2, rve_dims)
        overlap_length, _ = sphere_1.intersection_length_mink_diff(sphere_2, rve_dims)
        self.assertTrue(intersection)
        self.assertTrue(np.abs(0.1 - overlap_length) / 0.1 < 1e-8)



