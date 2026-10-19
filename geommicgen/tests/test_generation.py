"""
Unit tests regarding microstructure generation.
The classe tessted is GenerationMethod
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np


# pylint: disable=import-error
from geommicgen.micgenmethod.microstructure_gen_method import GenerationMethod
from geommicgen.microstructure.particleclasses.disk import Disk
from geommicgen.microstructure.particleclasses.sphere import Sphere


class TestGenerationMethod(unittest.TestCase):
    """Class for the unit test regarding the generation method"""

    def _build_disk_particle(self, position_center):
        phase = "FakePhase"
        descriptors = {"r":0.1, "n":1}
        rve_dims = [1,1]

        particle = Disk(phase, descriptors, rve_dims)
        particle.position_center = position_center
        return particle

    def _build_sphere_particle(self, position_center):
        phase = "FakePhase"
        descriptors = {"r":0.1, "n":1}
        rve_dims = [1,1,1]

        particle = Sphere(phase, descriptors, rve_dims)
        particle.position_center = position_center
        return particle


    def test_generate_microstructures_abstract(self):
        """Test if generateMicrostructure is an abstract method."""

        class MicGenTestIncomp(GenerationMethod):
            pass

        with self.assertRaises(TypeError):

            _ = MicGenTestIncomp()


    def test_compute_rve_offset_1particle_2D(self):
        " Tests if a correct value of offset is given when only one 2D particle is in the simulation box."

        # With one particle, each face is put half a box away from it, in the widest gap.
    
        rve_dims = [1,1]

        class MicGen_Test(GenerationMethod):
            def generate_microstructure(self, microstructure_sample):
                pass

        generation_method = MicGen_Test()
        
        test_cases = [
            # the particle touches all simulation box walls
            {"position_center" : [0   ,0   ], "offset" : [0.5 ,0.5 ,0]},
            # the particle touches all simulation box walls
            {"position_center" : [0.95,0.95], "offset" : [0.45,0.45,0]},
            # the particle is inside the box bounds in y
            {"position_center" : [0.05,0.5 ], "offset" : [0.55,0   ,0]},
            # the particle is inside the box bounds in x
            {"position_center" : [0.5 ,0.05], "offset" : [0   ,0.55,0]}
            ]
        for test in test_cases:
            with self.subTest(test):
                particle1 = self._build_disk_particle(test["position_center"])
                particles = [particle1]
                off_set = generation_method.compute_rve_offset(particles, rve_dims)
                np.testing.assert_almost_equal(off_set, test["offset"], decimal = 6)

    def test_compute_rve_offset_particles_2D(self):
        particle1 = self._build_disk_particle(position_center=[0.1,0])
        particle2 = self._build_disk_particle(position_center=[0.3,0])
        particle3 = self._build_disk_particle(position_center=[0.3,0.2])
        particle4 = self._build_disk_particle(position_center=[0.9,0.6])
        particle5 = self._build_disk_particle(position_center=[0.5,0.8])
        particles = [particle1,particle2,particle3,particle4,particle5]
        rve_dims = [1,1]

        class MicGen_Test(GenerationMethod):
            def generate_microstructure(self, microstructure_sample):
                pass

        generation_method = MicGen_Test()
        off_set = generation_method.compute_rve_offset(particles, rve_dims)
        np.testing.assert_almost_equal(off_set, [0.7, 0.4, 0], decimal = 6)
        # y = 0.4 is the middle of the one stretch along y no disk spans; the gap between
        # the limits at 0.1 and 0.3 is the disk at y = 0.2

    def test_compute_rve_offset_1particle_3D(self):
        " Tests if a correct value of offset is given when only one 3D particle is in the simulation box."

        # With one particle, each face is put half a box away from it, in the widest gap.

        rve_dims = [1,1,1]

        class MicGen_Test(GenerationMethod):
            def generate_microstructure(self, microstructure_sample):
                pass

        generation_method = MicGen_Test()
        
        test_cases = [
            # the particle touches all simulation box walls
            {"position_center" : [0   ,0   ,0   ], "offset" : [0.5 ,0.5 ,0.5 ]},
            # the particle touches all simulation box walls
            {"position_center" : [0.95,0.95,0.95], "offset" : [0.45,0.45,0.45]},
            # the particle is inside the box bounds in z
            {"position_center" : [0.9 ,0.05,0.5 ], "offset" : [0.4 ,0.55,0   ]},
            # the particle does not touch any wall
            {"position_center" : [0.4 ,0.6 ,0.3 ], "offset" : [0.9 ,0.1 ,0.8 ]}
            ]
        for test in test_cases:
            with self.subTest(test):
                particle1 = self._build_sphere_particle(test["position_center"])
                particles = [particle1]
                off_set = generation_method.compute_rve_offset(particles, rve_dims)
                np.testing.assert_almost_equal(off_set, test["offset"], decimal = 6)

    def test_compute_rve_offset_particles_3D(self):
        particle1 = self._build_sphere_particle(position_center=[0.1,0  ,0.3])
        particle2 = self._build_sphere_particle(position_center=[0.3,0  ,0.9])
        particle3 = self._build_sphere_particle(position_center=[0.3,0.2,0  ])
        particle4 = self._build_sphere_particle(position_center=[0.9,0.6,0.5])
        particle5 = self._build_sphere_particle(position_center=[0.5,0.8,0.6])
        particles = [particle1,particle2,particle3,particle4,particle5]
        rve_dims = [1,1,1]

        class MicGen_Test(GenerationMethod):
            def generate_microstructure(self, microstructure_sample):
                pass

        generation_method = MicGen_Test()
        off_set = generation_method.compute_rve_offset(particles, rve_dims)
        np.testing.assert_almost_equal(off_set, [0.7, 0.4, 0.75], decimal = 6)
        # Each face in the middle of a stretch no sphere spans: z = 0.3 went through the
        # sphere centred there

    def test_compute_rve_offset_2D_no_face_cuts_a_particle(self):
        " Each face is put in the widest gap along its axis, so when there is a gap no face cuts a particle."

        particle1 = self._build_disk_particle(position_center=[0.405,0.568])
        particle2 = self._build_disk_particle(position_center=[0.339,0.618])
        particle3 = self._build_disk_particle(position_center=[0.103,0.319])
        particles = [particle1,particle2,particle3]
        rve_dims = [1,1]

        class MicGen_Test(GenerationMethod):
            def generate_microstructure(self, microstructure_sample):
                pass

        generation_method = MicGen_Test()
        off_set = generation_method.compute_rve_offset(particles, rve_dims)
        np.testing.assert_almost_equal(off_set, [0.754, 0.9685, 0], decimal = 6)
        for particle in particles:
            for i_dim in range(2):
                gap = abs(off_set[i_dim] - particle.position_center[i_dim])
                self.assertGreater(min(gap, rve_dims[i_dim] - gap), particle.radius)