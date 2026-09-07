import os
import tempfile
import unittest

import numpy as np

from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses import (
    Cylinder,
    CylindricalFiber,
    Disk,
    Ellipse,
    Ellipsoid,
    Sphere,
)
from geommicgen.iofuncs.microstructure_yaml import (
    FORMAT_NAME,
    read_microstructure_yaml,
    write_microstructure_yaml,
)


def build_microstructure(rve_dims, phase_type, particles):
    """Build a microstructure with a matrix phase and the supplied particles."""
    microstructure = Microstructure(rve_dims)
    microstructure.add_phase(Phase("1", {"phase_type": 1}))
    microstructure.add_phase(Phase.from_type("2", phase_type))
    for i_particle in particles:
        microstructure.phases["2"].particles.append(i_particle)

    return microstructure


class TestMicrostructureYamlRoundTrip(unittest.TestCase):
    """Test class for the round trip of every particle shape through the YAML format."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "mic.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def round_trip(self, microstructure):
        """Write and read back a microstructure, returning the one that was read."""
        write_microstructure_yaml(microstructure, self.file_path)

        return read_microstructure_yaml(self.file_path)

    def assert_particles_equal(self, original, restored):
        """Check that the geometry of every particle survived the round trip."""
        self.assertEqual(len(original.particles), len(restored.particles))
        for i_original, i_restored in zip(original.particles, restored.particles):
            self.assertIs(type(i_original), type(i_restored))
            self.assertEqual(i_original.phase, i_restored.phase)
            np.testing.assert_allclose(
                i_original.position_center, i_restored.position_center
            )
            self.assertAlmostEqual(i_original.volume, i_restored.volume)
            self.assertAlmostEqual(i_original.radius, i_restored.radius)

    def test_disk(self):
        rve_dims = [1.0, 1.0]
        particle = Disk("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.25, 0.75])
        microstructure = build_microstructure(rve_dims, Disk, [particle])
        self.assert_particles_equal(microstructure, self.round_trip(microstructure))

    def test_ellipse(self):
        rve_dims = [1.0, 1.0]
        particle = Ellipse(
            "2", {"major_axis": 0.2, "minor_axis": 0.1, "angle": 0.4}, rve_dims
        )
        particle.position_center = np.array([0.1, 0.2])
        microstructure = build_microstructure(rve_dims, Ellipse, [particle])
        restored = self.round_trip(microstructure)
        self.assert_particles_equal(microstructure, restored)
        np.testing.assert_allclose(
            microstructure.particles[0].rot_mat, restored.particles[0].rot_mat
        )

    def test_sphere(self):
        rve_dims = [1.0, 1.0, 1.0]
        particle = Sphere("2", {"r": 0.1}, rve_dims)
        particle.position_center = np.array([0.3, 0.4, 0.5])
        microstructure = build_microstructure(rve_dims, Sphere, [particle])
        self.assert_particles_equal(microstructure, self.round_trip(microstructure))

    def test_ellipsoid(self):
        rve_dims = [1.0, 1.0, 1.0]
        descriptors = {
            "axis_1": 0.2,
            "axis_2": 0.15,
            "axis_3": 0.1,
            "angle": 0.7,
            "rot_axis_comp_x": 0.0,
            "rot_axis_comp_y": 1.0,
            "rot_axis_comp_z": 1.0,
        }
        particle = Ellipsoid("2", dict(descriptors), rve_dims)
        particle.position_center = np.array([0.5, 0.5, 0.5])
        microstructure = build_microstructure(rve_dims, Ellipsoid, [particle])
        restored = self.round_trip(microstructure)
        self.assert_particles_equal(microstructure, restored)
        np.testing.assert_allclose(
            microstructure.particles[0].rotation_mat,
            restored.particles[0].rotation_mat,
        )
        # The rotation matrix is derived from the axis and the angle, so it only matches
        # if the orientation itself survived

    def test_cylinder(self):
        rve_dims = [1.0, 1.0, 1.0]
        descriptors = {
            "r_cyl": 0.05,
            "length": 0.3,
            "azimuth_angle": 0.2,
            "polar_angle": 0.5,
        }
        particle = Cylinder("2", dict(descriptors), rve_dims)
        particle.position_center = np.array([0.2, 0.3, 0.4])
        microstructure = build_microstructure(rve_dims, Cylinder, [particle])
        restored = self.round_trip(microstructure)
        self.assert_particles_equal(microstructure, restored)
        np.testing.assert_allclose(
            microstructure.particles[0].rot_mat, restored.particles[0].rot_mat
        )

    def test_cylindrical_fiber_keeps_two_dimensional_centre(self):
        rve_dims = [1.0, 1.0, 1.0]
        particle = CylindricalFiber("2", {"r": 0.1, "direction": 2}, rve_dims)
        particle.position_center = np.array([0.3, 0.6])
        microstructure = build_microstructure(rve_dims, CylindricalFiber, [particle])
        restored = self.round_trip(microstructure)
        self.assert_particles_equal(microstructure, restored)
        self.assertEqual(len(restored.particles[0].position_center), 2)
        self.assertEqual(restored.particles[0].direction_fibers, 2)
        # The fibre spans the RVE, so its centre has one coordinate fewer than the RVE


class TestMicrostructureYamlStructure(unittest.TestCase):
    """Test class for the structure of the file produced by the YAML writer."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "mic.yaml")
        self.rve_dims = [1.0, 1.0]
        particles = []
        for i_center in ([0.25, 0.75], [0.6, 0.1]):
            particle = Disk("2", {"r": 0.1}, self.rve_dims)
            particle.position_center = np.array(i_center)
            particles.append(particle)
        self.microstructure = build_microstructure(self.rve_dims, Disk, particles)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_one_line_per_particle(self):
        write_microstructure_yaml(self.microstructure, self.file_path)
        with open(self.file_path, "r") as yaml_file:
            lines = yaml_file.read().splitlines()
        particle_lines = [line for line in lines if line.startswith("- {")]
        self.assertEqual(len(particle_lines), 2)
        for line in particle_lines:
            self.assertIn("shape: Disk", line)
        # A record spread over several lines would not start with the flow style marker

    def test_matrix_phase_and_dimensions_preserved(self):
        write_microstructure_yaml(self.microstructure, self.file_path)
        restored = read_microstructure_yaml(self.file_path)
        self.assertEqual(restored.matrix_phase, "1")
        self.assertEqual(restored.dim, 2)
        np.testing.assert_allclose(restored.rve_dims, self.rve_dims)
        self.assertEqual(sorted(restored.phases.keys()), ["1", "2"])

    def test_provenance_is_written(self):
        write_microstructure_yaml(
            self.microstructure, self.file_path, provenance={"fixed_seed": 12345}
        )
        restored = read_microstructure_yaml(self.file_path)
        with open(self.file_path, "r") as yaml_file:
            contents = yaml_file.read()
        self.assertIn("provenance:", contents)
        self.assertIn("fixed_seed: 12345", contents)
        self.assertIn("geommicgen_version:", contents)
        self.assertEqual(len(restored.particles), 2)
        # The provenance is informational and must not disturb the reader

    def test_rejects_foreign_file(self):
        with open(self.file_path, "w") as yaml_file:
            yaml_file.write("format: something-else\nversion: 1\n")
        with self.assertRaises(ValueError):
            read_microstructure_yaml(self.file_path)

    def test_rejects_unsupported_version(self):
        with open(self.file_path, "w") as yaml_file:
            yaml_file.write("format: {0}\nversion: 99\n".format(FORMAT_NAME))
        with self.assertRaises(ValueError):
            read_microstructure_yaml(self.file_path)


class TestMicrostructureYamlCoated(unittest.TestCase):
    """Test class for the round trip of coated inclusions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "mic.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parent_is_preserved(self):
        rve_dims = [1.0, 1.0]
        microstructure = Microstructure(rve_dims)
        microstructure.add_phase(Phase("1", {"phase_type": 1}))
        microstructure.add_phase(Phase.from_type("2", Disk))
        microstructure.add_phase(
            Phase.from_type("3", Disk, inner_phase=True, outer_phase="2")
        )

        outer = Disk("2", {"r": 0.1}, rve_dims)
        outer.position_center = np.array([0.5, 0.5])
        microstructure.phases["2"].particles.append(outer)

        inner = Disk("3", {"r": 0.02}, rve_dims)
        inner.position_center = np.array([0.51, 0.5])
        inner.parent = outer
        microstructure.phases["3"].particles.append(inner)

        write_microstructure_yaml(microstructure, self.file_path)
        restored = read_microstructure_yaml(self.file_path)

        self.assertTrue(restored.phases["3"].inner_phase)
        self.assertEqual(restored.phases["3"].outer_phase, "2")
        restored_inner = restored.phases["3"].particles[0]
        restored_outer = restored.phases["2"].particles[0]
        self.assertIs(restored_inner.parent, restored_outer)
        self.assertIsNone(restored_outer.parent)


class TestMicrostructureYamlDescriptors(unittest.TestCase):
    """Test class for the round trip of the phase descriptors."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "mic.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_distributions_are_preserved(self):
        rve_dims = [1.0, 1.0]
        microstructure = Microstructure(rve_dims)
        microstructure.add_phase(Phase("1", {"phase_type": 1}))
        microstructure.add_phase(
            Phase(
                "2",
                {
                    "phase_type": 3,
                    "major_axis": 0.2,
                    "major_axis_distribution": "normal",
                    "major_axis_mean": 0.2,
                    "major_axis_sigma": 0.01,
                    "minor_axis": 0.1,
                    "angle": 0.0,
                    "angle_distribution": "uniform",
                    "angle_low": 0.0,
                    "angle_high": 3.14,
                    "n": 3,
                },
            )
        )

        write_microstructure_yaml(microstructure, self.file_path)
        restored = read_microstructure_yaml(self.file_path)

        descriptors = restored.phases["2"].descriptors
        self.assertEqual(descriptors["major_axis"].mean, 0.2)
        self.assertEqual(descriptors["major_axis"].sigma, 0.01)
        self.assertEqual(descriptors["angle"].low, 0.0)
        self.assertEqual(descriptors["angle"].high, 3.14)
        self.assertEqual(descriptors["n"].value, 3)


if __name__ == "__main__":
    unittest.main()
