""" Unit test regarding the Phase class"""

import numpy as np
import unittest
from unittest.mock import Mock, patch
from geommicgen.microstructure.phase import Phase
from geommicgen.microstructure.particleclasses.particle import Particle

class PhaseInit(unittest.TestCase):
    def test_all_phase_types_are_particle_subclasses(self):
        " Check that every class in Phase.phase_types is a subclass of Particle "
        for particle_class in Phase.phase_types.values():
            self.assertTrue(
                issubclass(particle_class, Particle),
                msg="{0} is not a subclass of Particle".format(particle_class.__name__),
            )


class TestPhaseProperties(unittest.TestCase):

    def setUp(self):
        self.phase = Phase.__new__(Phase)
        self.phase.microstructure = Mock(volume=1)
               
    def test_volume_fraction(self):
        self.phase.particles = [Mock(volume=0.1), Mock(volume=0.2), Mock(volume=0.3)] 
        self.assertAlmostEqual(self.phase.volume_fraction, 0.6)

    def test_volume_fraction_circ(self):
        self.phase.particles = [Mock(volume_circ = 0.1), Mock(volume_circ = 0.15), Mock(volume_circ = 0.2) ]
        self.assertAlmostEqual(self.phase.volume_fraction_circ,0.45)

    def test_number_particles(self):
        self.phase.particles = [Mock() for _ in range(5)]
        self.assertEqual(self.phase.number_particles,5)


class GenerateParticles(unittest.TestCase):

    def setUp(self):
        self.phase = Phase.__new__(Phase)
        self.phase.name = 'FakePhase'
        self.rve_dims = [1,1]
        self.phase.microstructure = Mock(volume = 1)

    def test_generate_particles_n(self):
        " Check if generating particles by specifying their number is working"
        # add check to see if the radius of each particle is correct
        self.phase.descriptors = {'n' : Mock(value=3) , 'radius': Mock()}
        self.phase.descriptors['n'].generate_sample.return_value = [3,3,3]
        self.phase.descriptors['radius'].generate_sample.return_value = [1,2,3]
        self.phase.type = Mock()
        self.phase.generate_particles(self.rve_dims)
        self.assertIsInstance(self.phase.particles, list)
        self.assertEqual( len(self.phase.particles) , 3)

        # Edge case where n=1 and volume fraction was also specified. phase.generate_particles should ignore vf. If it does not, an error will appear because the volume of the particle is not specified.
        self.phase.descriptors = {'n' : Mock(value=1), 'vf' : Mock()}
        self.phase.descriptors['n'].generate_sample.return_value = [1]
        self.phase.type = Mock()
        self.phase.generate_particles(self.rve_dims)
        self.assertEqual( len(self.phase.particles) , 1)       


    def test_generate_particles_vf(self):
        " Check if generating particles by specifying the volume fraction is working"    
        self.phase.descriptors = {'vf' : Mock(value=0.3)}
        self.phase.descriptors['vf'].generate_sample.return_value = [0.3,0.3,0.3]
        self.phase.type = Mock()
        self.phase.type.return_value.volume = 0.2
        self.phase.generate_particles(self.rve_dims)
        self.assertIsInstance(self.phase.particles, list)
        self.assertEqual( len(self.phase.particles) , 2)
        self.assertEqual( self.phase.volume_fraction , 0.4 )

class DistributionTests(unittest.TestCase):
    from geommicgen.microstructure.phase import SpecifiedValue, FixedValue, NormalDistribution, LogNormalDistribution, UniformDistribution, VonMisesDistribution, DiscreteDistribution

    def test_specified_value_distribution(self):
        distribution = self.SpecifiedValue("name",[1,2,3,4])
        self.assertEqual(distribution.generate_sample(), 1)
        self.assertEqual(distribution.generate_sample(2), [2,3])
        self.assertEqual(distribution.generate_sample(), 4)

    def test_fixed_value_distribution(self):
        distribution = self.FixedValue("name",2)
        self.assertEqual(distribution.generate_sample(), 2)
        np.testing.assert_array_equal(distribution.generate_sample(3), np.array([2,2,2]))

    def test_normal_distribution(self):
        distribution = self.NormalDistribution("name", mean=2, sigma=0.5)
        self.assertEqual(distribution.mean, 2)
        self.assertEqual(distribution.sigma, 0.5)
        self.assertIsInstance(distribution.generate_sample(), np.floating)
        self.assertEqual(len(distribution.generate_sample(5)), 5)

    def test_log_normal_distribution(self):
        distribution = self.LogNormalDistribution("name", mean=2, sigma=0.5)
        self.assertEqual(distribution.mean, 2)
        self.assertEqual(distribution.sigma, 0.5)
        self.assertIsInstance(distribution.generate_sample(), np.floating)
        self.assertEqual(len(distribution.generate_sample(5)), 5)

    def test_uniform_distribution(self):
        distribution = self.UniformDistribution("name", 1, 2.5)
        self.assertEqual(distribution.low, 1)
        self.assertEqual(distribution.high, 2.5)
        self.assertIsInstance(distribution.generate_sample(), np.floating)
        self.assertEqual(len(distribution.generate_sample(5)), 5)
        with self.assertRaises(ValueError):    
            distribution = self.UniformDistribution("name", 2, 1) 

    def test_discrete_distribution(self):
        distribution = self.DiscreteDistribution("name", [1,2,3], [0.2,0.3,0.5])
        self.assertIn(distribution.generate_sample(), [1,2,3])
        n_samples = 10000
        sample = distribution.generate_sample(n_samples)
        self.assertEqual(len(sample), n_samples)
        # With a large sample, frequencies should approach the given
        # probabilities. Using a tolerance instead of an exact count avoids
        # flakiness that comes from checking the outcome of a small number of
        # random draws.
        counts = list(sample)
        self.assertAlmostEqual(counts.count(1) / n_samples, 0.2, delta=0.03)
        self.assertAlmostEqual(counts.count(2) / n_samples, 0.3, delta=0.03)
        self.assertAlmostEqual(counts.count(3) / n_samples, 0.5, delta=0.03)
                         
        with self.assertRaises(ValueError):
            distribution = self.DiscreteDistribution("name", [1,2,3], [0.2,0.3])  # Probabilities do not sum to 1
        with self.assertRaises(ValueError):
            distribution = self.DiscreteDistribution("name", [1,2,3], [0.2,0.9])  # Probabilities do not sum to 1



