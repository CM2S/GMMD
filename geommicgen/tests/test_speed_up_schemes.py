"""
Unit tests regarding microstructure generation.
The classes tested are the GenerationMethod class and the MolecularDynamicsSimulation class.
"""
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np
