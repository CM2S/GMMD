"""
Unit tests regarding voronoi analysis.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
import time
from microstructure.particle_classes import (
    Ellipsoid,
    Ellipse,
    Cylinder,
    Particle,
    Disk,
    Sphere,
    Point,
    Line,
)
from postproc.plotfuncs.plotting_functions import plot_particles_3d, plot_particles_2d
import pickle
from postproc.voronoimetrics.voronoi_analysis import update_indices


class TestUpdateIndices(unittest.TestCase):
    def test_indices(self):
        old_indices = [1, 3, 4, 5, 7, 9, 10, 15]
        removed_ind = [2, 4, 6, 8, 12]
        new_ind = update_indices(old_indices, removed_ind)
        print(new_ind)
