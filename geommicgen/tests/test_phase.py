# import unittest
# from unittest.mock import sentinel, Mock, patch, call
#
# # from microstructure.phase import Phase
#
#
# from micgenmethod.microstructure_gen_method import GenerationMethod
# from microstructure.phase import Phase
#
#
# class TestPhase(unittest.TestCase):
#     """Class for the unit test regarding the phase class"""
#
#     @patch("microstrucutre.phase.FixedValue")
#     @patch("microstructure.particleclasses.Disk")
#     def test_generate_particles_number(self, mock_fixed_value, mock_disk):
#         mock_fixed_value.value = 0.1
#         rve_dims = [1.0, 1.0]
#         descriptors = {
#             "phase_type": 2,
#             "n": 10,
#             "vf": 0.1,
#         }
#         phase = Phase("1", descriptors)
#         particles = phase.generate_particles(rve_dims)
#         for particle in particles:
#             self.assertIsInstance(particle, mock_disk)
#
#     @patch("microstructure.phase.FixedValue")
#     @patch("microstructure.particleclasses.Disk")
#     def test_generate_particles_vf(self, mock_fixed_value, mock_disk):
#         rve_dims = [1.0, 1.0]
#         descriptors = {
#             "phase_type": 2,
#             "r": 10,
#             "vf": 0.1,
#         }
#         phase = Phase("1", descriptors)
#         particles = phase.generate_particles(rve_dims)
#         self.assertEqual(len(particles), 10)


import unittest
from unittest.mock import patch

from geommicgen.microstructure import phase as phase_module
from geommicgen.microstructure.phase import Phase


class TestParticlesForAVolumeFraction(unittest.TestCase):
    """Test class for the particles of a phase given by its volume fraction."""

    def test_they_fill_the_fraction(self):
        phase = Phase("1", {"phase_type": 2, "r": 0.1, "vf": 0.3})
        phase.generate_particles([1.0, 1.0])
        filled = sum(i_particle.volume for i_particle in phase.particles)
        self.assertGreaterEqual(filled, 0.3)
        self.assertLess(filled - phase.particles[-1].volume, 0.3)

    @patch.object(phase_module, "MAX_PARTICLES", 50)
    def test_particles_too_small_for_it_are_refused(self):
        phase = Phase("1", {"phase_type": 2, "r": 0.001, "vf": 0.5})
        with self.assertRaisesRegex(ValueError, "Phase 1 has 50 particles"):
            phase.generate_particles([1.0, 1.0])
        # A hundred and sixty thousand disks would fill it; with no bound, a
        # description asking for far more built particles until the memory ran out


if __name__ == "__main__":
    unittest.main()
