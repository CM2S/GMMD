"""
Module containing all the Particle abstract class and all its subclasses.

Each subclass of the Particle class is a type of particle. This module includes the Ellipse,
Disk, CylindricalFiber, Ellipsoid and Shpere classes.
"""
from __future__ import annotations

import abc
import time


from itertools import cycle

import numpy as np

from scipy import integrate

from geommicgen.microstructure.particleclasses import Disk, Particle


class CylindricalFiber(Disk):
    """
    This is the subclass of particles with the form of a circular disk.

    Attributes
    ----------
    direction_fibers: {0, 1, 2}
        Direction in which the fibers run. 'x':0, 'y':1 and 'z':2

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
            "r": ("Radius", "float"),
            "area": ("Area per particle", "float"),
            "direction": ("Fiber direction", "int"),
        },
        **Particle.possible_parameters,
    }
    # all possible_parameters
    acceptable_descriptions = [
        {"r", "n", "direction"},
        {"r", "vf", "direction"},
        {"n", "vf", "direction"},
        {"area", "vf", "direction"},
        {"area", "n", "direction"},
    ]
    dim = 3
    # List of acceptable collections of parameters

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
        self.direction_fibers = descriptors.pop("direction")
        # Integer giving the direction of the fibers
        self.length_dir_fibers = rve_dims[self.direction_fibers]
        # Setting the size of the simulation box
        box = list(rve_dims)
        del box[self.direction_fibers]
        super().__init__(phase, descriptors, box)
        # Using the constructor of the parent class

    @property
    def volume(self):
        """Volume of the cylindrical fiber."""
        volume = np.pi * self.radius ** 2 * self.length_dir_fibers

        return volume
