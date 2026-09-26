"""Module containing the helpers shared by the test modules."""

import os
import signal

import numpy as np

from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.mesh import Mesh, StructuredInfo
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from geommicgen.micgenmethod.thermostats import IsokineticThermostat
from geommicgen.microstructure.particleclasses import Disk, Ellipse, Sphere
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import Phase


class ProcessEndingMesher(GmshMesher):
    """Gmsh mesher that ends the process it meshes in, as gmsh can, giving nothing."""

    def mesh_in_this_process(self, microstructure, report=None):
        """End the process, which is the one `mesh` starts for it."""
        os.kill(os.getpid(), signal.SIGKILL)


def non_conforming_mesh(mesh):
    """
    Build a copy of a mesh whose opposite faces are no longer discretised alike.

    One node in the interior of a face is moved along the face, which leaves it without
    a partner on the opposite one.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh to be broken.

    Returns
    -------
    `.Mesh`
        The mesh with the node moved.
    """
    points = mesh.points.copy()
    points[mesh.boundary.face_interior["x+"][0], 1] += 1.0e-3

    return Mesh(
        mesh.rve_dims,
        points=points,
        cells=mesh.cells,
        phase=mesh.phase,
        phase_names=mesh.phase_names,
        matrix_phase=mesh.matrix_phase,
    )


def structured_mesh(shape, rve_dims, phase_grid=None):
    """
    Build a structured mesh with the supplied number of voxels.

    Parameters
    ----------
    shape: tuple
        Number of voxels in each spatial direction.

    rve_dims: list
        Dimensions of the microstructure in each spatial direction.

    phase_grid: array
        Array of integers with the phase of every voxel. Defaults to a single phase.

    Returns
    -------
    `.Mesh`
        The structured mesh.
    """
    if phase_grid is None:
        phase_grid = np.ones(shape, dtype=int)
    spacing = np.asarray(rve_dims, dtype=float) / np.asarray(shape, dtype=float)

    return Mesh(
        rve_dims,
        structured=StructuredInfo(phase_grid, spacing),
        phase_names={1: "1", 2: "2"},
        matrix_phase="1",
        periodic=True,
    )


def build_microstructure(rve_dims, phase_type, particles):
    """
    Build a microstructure with a matrix phase and the supplied particles.

    Parameters
    ----------
    rve_dims: list
        Dimensions of the microstructure in each spatial direction.

    phase_type: class
        Class of the particles, used to declare the inclusion phase.

    particles: list
        Particles of the inclusion phase, already positioned.

    Returns
    -------
    `.Microstructure`
        Microstructure with the matrix phase named *1* and the inclusion phase named
        *2*.
    """
    microstructure = Microstructure(rve_dims)
    microstructure.add_phase(Phase("1", {"phase_type": 1}))
    microstructure.add_phase(Phase.from_type("2", phase_type))
    for i_particle in particles:
        microstructure.phases["2"].particles.append(i_particle)

    return microstructure


def disk_microstructure():
    """Build a microstructure of two disks, one of them crossing a face of the RVE."""
    rve_dims = [1.0, 1.0]
    particles = []
    for i_center, i_radius in (([0.3, 0.3], 0.15), ([0.98, 0.8], 0.12)):
        particle = Disk("2", {"r": i_radius}, rve_dims)
        particle.position_center = np.array(i_center)
        particles.append(particle)

    return build_microstructure(rve_dims, Disk, particles)


def ellipse_microstructure():
    """Build a microstructure of two ellipses, one of them crossing a face of the RVE."""
    rve_dims = [1.0, 1.0]
    particles = []
    for i_center, i_angle in (([0.3, 0.35], 0.4), ([0.95, 0.75], -0.9)):
        particle = Ellipse(
            "2", {"major_axis": 0.15, "minor_axis": 0.1, "angle": i_angle}, rve_dims
        )
        particle.position_center = np.array(i_center)
        particles.append(particle)

    return build_microstructure(rve_dims, Ellipse, particles)


def sphere_microstructure():
    """Build a microstructure of two spheres, one of them crossing two faces."""
    rve_dims = [1.0, 1.0, 1.0]
    particles = []
    for i_center, i_radius in (([0.4, 0.4, 0.4], 0.2), ([0.9, 0.1, 0.5], 0.15)):
        particle = Sphere("2", {"r": i_radius}, rve_dims)
        particle.position_center = np.array(i_center)
        particles.append(particle)

    return build_microstructure(rve_dims, Sphere, particles)


def a_generation_run(n_particles=2, n_steps=3):
    """
    Build a generation method carrying the state a finished run would have recorded.

    Parameters
    ----------
    n_particles: int
        Number of particles whose motion was recorded.

    n_steps: int
        Number of steps that were recorded.

    Returns
    -------
    `.MolecularDynamicsSimulation`
        The generation method.
    """
    mic_generator = MolecularDynamicsSimulation(0.0, 10, 5, 1e-3, 0.0, "random", True)
    mic_generator.set_thermostat(IsokineticThermostat(1.0))
    mic_generator.step = n_steps
    mic_generator.time = 12.5
    mic_generator.status = True
    mic_generator.max_residue = 1e-6
    mic_generator.total_overlap_history = [3.0, 2.0, 1.0]
    mic_generator.kinetic_energy_history = [0.5, 0.4, 0.3]
    mic_generator.thermic_energy_history = [0.1, 0.2, 0.3]
    mic_generator.all_dt = [1e-3, 1e-3, 2e-3]
    mic_generator.position_center_history = [
        [np.array([0.1 * i_step, 0.2 * i_particle]) for i_step in range(n_steps)]
        for i_particle in range(n_particles)
    ]
    mic_generator.thermostat.temp_change_steps = [0, 2]
    mic_generator.thermostat.ratio = [1.2, 0.9]

    return mic_generator
