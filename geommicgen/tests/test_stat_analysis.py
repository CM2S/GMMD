"""Test module for the statistical descriptors of a microstructure."""

import unittest

import numpy as np

from geommicgen.microstructure.particleclasses import Disk
from geommicgen.postproc.voronoimetrics.stat_analysis import (
    crosses_boundary,
    remove_particles_at_boundary,
)

RVE_DIMS = [1.0, 1.0]


def disk(center, radius=0.05):
    """Build a disk of the inclusion phase at a position."""
    particle = Disk("1", {"r": radius}, RVE_DIMS)
    particle.position_center = np.array(center)

    return particle


class TestRemoveParticlesAtBoundary(unittest.TestCase):
    """Test class for the particles left out of the statistics for touching a face."""

    def test_a_particle_is_judged_by_its_extremes(self):
        self.assertTrue(crosses_boundary(disk([0.02, 0.5]), RVE_DIMS))
        self.assertTrue(crosses_boundary(disk([0.98, 0.5]), RVE_DIMS))
        self.assertTrue(crosses_boundary(disk([0.5, 0.01]), RVE_DIMS))
        self.assertFalse(crosses_boundary(disk([0.5, 0.5]), RVE_DIMS))

    def test_the_particles_inside_are_the_ones_kept(self):
        inside = [disk([0.5, 0.5]), disk([0.3, 0.7])]
        outside = [disk([0.02, 0.5]), disk([0.98, 0.5]), disk([0.5, 0.99])]
        kept = remove_particles_at_boundary(outside + inside, RVE_DIMS)
        self.assertEqual(kept, inside)

    def test_particles_crossing_one_after_the_other_are_both_removed(self):
        kept = remove_particles_at_boundary(
            [disk([0.02, 0.5]), disk([0.98, 0.5]), disk([0.5, 0.5])], RVE_DIMS
        )
        self.assertEqual(len(kept), 1)
        np.testing.assert_allclose(kept[0].position_center, [0.5, 0.5])
        # The list was walked while it was removed from, so the particle after every
        # one taken out was never looked at: the second of these was left in

    def test_a_microstructure_of_particles_inside_is_left_alone(self):
        particles = [disk([0.3, 0.3]), disk([0.5, 0.5]), disk([0.7, 0.7])]
        self.assertEqual(remove_particles_at_boundary(particles, RVE_DIMS), particles)


if __name__ == "__main__":
    unittest.main()
