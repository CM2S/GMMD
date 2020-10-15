"""
Unit tests regarding thermostats.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
import time
from microstructure.particle_classes import Ellipsoid, Ellipse, Cylinder, Particle, Disk
from postproc.plotfuncs.plotting_functions import plot_particles_3d
import pickle


# class TestEllipsoid(unittest.TestCase):
#     def test_support_function(self):
#         """Check supprt function."""
#         phase = "1"
#         descriptors = {
#             "axis_1": 0.1,
#             "axis_2": 0.15,
#             "axis_3": 0.05,
#             "rot_axis_comp_x": 0,
#             "rot_axis_comp_y": 0,
#             "rot_axis_comp_z": 1,
#             "angle": np.pi / 4,
#             "n": 1,
#         }
#         rve_dims = [1, 1, 1]
#         ellip = Ellipsoid(phase, descriptors, rve_dims)
