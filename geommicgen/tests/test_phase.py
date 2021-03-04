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
