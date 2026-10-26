"""
Integration tests that run the files GMMD writes through the solvers that read them.

These are the tests that say whether a file works, rather than whether it is well
formed: each one hands a mesh GMMD wrote to the solver and checks that the solver ran
it to completion. A solver that is not installed skips its tests, so the module says
nothing about a format it could not run.

LINKS is found through the environment variable GEOMMICGEN_LINKS, holding the path of
the executable, or as LINKS on the PATH; Abaqus through GEOMMICGEN_ABAQUS, or as abaqus
on the PATH. FEniCS is a library rather than a program, so what is run is the example
script shipped beside the decks, with the Python that has dolfinx, whose path is given
in GEOMMICGEN_DOLFINX_PYTHON. CRATE is a Python package whose pinned requirements this
one cannot share an environment with, so it is run with the Python that has it, given
in GEOMMICGEN_CRATE_PYTHON.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import geommicgen
from geommicgen._optional import has_gmsh, has_package
from geommicgen.pipeline import MeshJob
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.tests.helpers import (
    disk_microstructure,
    sphere_microstructure,
    structured_mesh,
)
from geommicgen.translators.abaqus import REFERENCE_NODE_NAMES, AbaqusWriter
from geommicgen.translators.base import (
    PLACEHOLDER_ELASTIC,
    PLACEHOLDER_STRAIN,
    get_writer,
)
from geommicgen.translators.crate import CrateWriter
from geommicgen.translators.links import LinksWriter

FENICS_EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(geommicgen.__file__)),
    "examples",
    "fenics_elasticity.py",
)
# The script a user would run, rather than one of the tests' own, so that what is
# checked is what is shown


def links_executable():
    """Path of the LINKS executable, or None when there is none to be found."""
    return os.environ.get("GEOMMICGEN_LINKS") or shutil.which("LINKS")


@unittest.skipUnless(links_executable(), "LINKS is not available")
class TestLinksRunsTheMesh(unittest.TestCase):
    """Test class for running the LINKS files through LINKS itself."""

    def setUp(self):
        """Create a directory for the run, which LINKS fills with its outputs."""
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)

    def run_links(self, mesher, microstructure):
        """Mesh a microstructure, write the LINKS files and run LINKS on the example."""
        job = MeshJob(mesher, [LinksWriter()], "m")
        job.run(microstructure, self.run_dir.name)
        self.assertIsNone(job.error, job.trace)
        completed = subprocess.run(
            [links_executable(), "m_example.rve"],
            cwd=self.run_dir.name,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        log = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, log[-3000:])
        self.assertIn("successfully completed", log)
        self.assertIn("Homogenised first Piola-Kirchhoff", log)
        # The example is a one increment elastic analysis, so LINKS reaching the
        # homogenised stress is LINKS having read the mesh, paired the periodic
        # faces and converged

        return log

    def test_two_dimensional_grid(self):
        self.run_links(VoxelMesher([8, 8]), disk_microstructure())

    def test_three_dimensional_grid(self):
        self.run_links(VoxelMesher([4, 4, 4]), sphere_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_two_dimensional_triangles(self):
        self.run_links(GmshMesher(mesh_size=0.1, element_type="tri3"), disk_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_three_dimensional_tetrahedra(self):
        self.run_links(
            GmshMesher(mesh_size=0.2, element_type="tetra4"), sphere_microstructure()
        )



def abaqus_executable():
    """Path of the Abaqus command, or None when there is none to be found."""
    return os.environ.get("GEOMMICGEN_ABAQUS") or shutil.which("abaqus")


def reaction_forces(dat):
    """
    Read the reaction forces Abaqus printed for the node sets of the reference nodes.

    Parameters
    ----------
    dat: str
        Text of the printed output of the run, the .dat file.

    Returns
    -------
    dict
        Reaction force on the node of each set, by the name of the set.
    """
    forces = {}
    name = None
    for i_line in dat.splitlines():
        words = i_line.split()
        if "PRINTED FOR NODES BELONGING TO NODE SET" in i_line:
            name = words[-1]
        elif name and words and words[0].isdigit():
            forces[name] = np.array([float(i_word) for i_word in words[1:]])
            name = None
    # Each table is headed by the set it is printed for, and the first row that starts
    # with a node number is the one node of a reference set

    return forces


@unittest.skipUnless(abaqus_executable(), "Abaqus is not available")
class TestAbaqusRunsTheDeck(unittest.TestCase):
    """Test class for running the Abaqus input files through Abaqus itself."""

    def setUp(self):
        """Create a directory for the run, which Abaqus fills with its outputs."""
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)

    def run_deck(self, run_dir, young=None):
        """
        Run the deck of a directory through Abaqus and read the reactions it printed.

        Parameters
        ----------
        run_dir: str
            Directory holding the deck m.inp, which Abaqus fills with its outputs.

        young: dict
            Young modulus to give each phase in place of the placeholder one.

        Returns
        -------
        dict
            Reaction force on each reference node, by the name of its set.
        """
        deck_path = os.path.join(run_dir, "m.inp")
        if young:
            with open(deck_path) as deck:
                text = deck.read()
            material = "*Material, name=MATERIAL_{0}\n*Elastic\n {1}, {2}\n"
            for i_phase, i_value in young.items():
                block = material.format(i_phase, *PLACEHOLDER_ELASTIC)
                self.assertIn(block, text)
                text = text.replace(
                    block, material.format(i_phase, i_value, PLACEHOLDER_ELASTIC[1])
                )
            with open(deck_path, "w") as deck:
                deck.write(text)
        # A material is changed where a user would change it, in the deck as written

        completed = subprocess.run(
            [
                abaqus_executable(),
                "job=m",
                "input=m.inp",
                "interactive",
                "ask_delete=OFF",
            ],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        outputs = {}
        for i_extension in ("sta", "msg", "dat"):
            path = os.path.join(run_dir, "m." + i_extension)
            outputs[i_extension] = ""
            if os.path.exists(path):
                with open(path, errors="replace") as output:
                    outputs[i_extension] = output.read()
        log = completed.stdout + completed.stderr + outputs["sta"]
        self.assertIn(
            "THE ANALYSIS HAS COMPLETED SUCCESSFULLY", outputs["sta"], log[-3000:]
        )
        # The status file is where Abaqus says whether the step finished, which the
        # exit status of the command does not always carry

        counts = re.findall(
            r"^\s*(\d+)\s+(?:WARNING|ERROR) MESSAGES", outputs["msg"], re.MULTILINE
        )
        messages = "\n".join(
            i_line
            for i_line in (outputs["dat"] + outputs["msg"]).splitlines()
            if "***WARNING" in i_line or "***ERROR" in i_line
        )
        self.assertTrue(counts, outputs["msg"][-3000:])
        self.assertEqual(set(counts), {"0"}, messages[-3000:] or outputs["msg"][-3000:])
        # Abaqus finishes a step it had to repair: a degree of freedom constrained twice
        # it drops with a warning, and a singular system it solves after one about its
        # zero pivots. The step finishing says neither was absent, the counts of the
        # message file do

        return reaction_forces(outputs["dat"])

    def run_abaqus(self, mesher, microstructure):
        """Mesh a microstructure, write the Abaqus deck, run it and check the stress."""
        job = MeshJob(mesher, [AbaqusWriter()], "m")
        job.run(microstructure, self.run_dir.name)
        self.assertIsNone(job.error, job.trace)
        forces = self.run_deck(self.run_dir.name)

        along, _, lame = uniaxial_strain(PLACEHOLDER_ELASTIC[0])
        rve_dims = np.asarray(microstructure.rve_dims, dtype=float)
        dim = len(rve_dims)
        stress = np.diag([along] + [lame * PLACEHOLDER_STRAIN] * (dim - 1))
        for i_axis in range(dim):
            area = np.prod(rve_dims) / rve_dims[i_axis]
            np.testing.assert_allclose(
                forces[REFERENCE_NODE_NAMES[i_axis]] / area,
                stress[:, i_axis],
                rtol=1e-6,
                atol=1e-6 * along,
            )
        # One material stretched along x and held across it, whose stresses are known in
        # closed form. The reaction on the reference node of an axis is the force on the
        # face across that axis, the column of the stress times the area of the face, so
        # equations pairing the wrong nodes or offset by the wrong cell show here, where
        # Abaqus would have run them all the same. It prints eight significant digits

    def test_two_dimensional_grid(self):
        self.run_abaqus(VoxelMesher([8, 8]), disk_microstructure())

    def test_three_dimensional_grid(self):
        self.run_abaqus(VoxelMesher([4, 4, 4]), sphere_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_two_dimensional_triangles(self):
        self.run_abaqus(
            GmshMesher(mesh_size=0.1, element_type="tri3"), disk_microstructure()
        )

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_three_dimensional_tetrahedra(self):
        self.run_abaqus(
            GmshMesher(mesh_size=0.2, element_type="tetra4"), sphere_microstructure()
        )

    def test_the_grid_is_read_along_the_axes_it_was_written_along(self):
        young = {1: PLACEHOLDER_ELASTIC[0], 2: 10 * PLACEHOLDER_ELASTIC[0]}
        across, along = laminate_stresses(young)
        phase_grid = np.ones((16, 16), dtype=int)
        phase_grid[:8] = 2
        for i_grid, i_expected in ((phase_grid, across), (phase_grid.T.copy(), along)):
            with self.subTest(layered_along="x" if i_expected is across else "y"):
                run_dir = tempfile.mkdtemp(dir=self.run_dir.name)
                AbaqusWriter().write(
                    structured_mesh((16, 16), [1.0, 1.0], i_grid),
                    os.path.join(run_dir, "m.inp"),
                )
                forces = self.run_deck(run_dir, young)
                self.assertAlmostEqual(
                    forces[REFERENCE_NODE_NAMES[0]][0] / i_expected, 1.0, places=6
                )
        # Each phase reached a material of its own, the stiffer one making the cell
        # stiffer than the placeholder alone, and x is the first axis of the grid in the
        # deck as it was in the mesh: had the grid been transposed on its way, the two
        # laminates would have swapped answers, which differ by more than half
    # Every mesh here stays under a thousand nodes, the reference nodes included, which
    # is what the learning edition of Abaqus runs


def dolfinx_python():
    """Path of a Python that has dolfinx, or None when none was given."""
    return os.environ.get("GEOMMICGEN_DOLFINX_PYTHON")


@unittest.skipUnless(
    dolfinx_python() and os.path.isfile(FENICS_EXAMPLE) and has_package("h5py"),
    "no Python with dolfinx was given, or the example or h5py is missing",
)
class TestFenicsSolvesOnTheMesh(unittest.TestCase):
    """Test class for solving on the xdmf files with FEniCS."""

    YOUNG = 1.0e3
    POISSON = 0.3
    STRAIN = 0.1

    def setUp(self):
        """Create a directory for the mesh."""
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)

    def run_fenics(self, mesher, microstructure, young):
        """Mesh a microstructure, write it as xdmf and solve on it with the example."""
        mesh = mesher.mesh(microstructure)
        path = os.path.join(self.run_dir.name, "m.xdmf")
        get_writer("xdmf")().write(mesh, path)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [dolfinx_python(), FENICS_EXAMPLE, path, "--young"]
            + ["{0}={1}".format(i_phase, i_value) for i_phase, i_value in young.items()]
            + ["--poisson", str(self.POISSON), "--strain", str(self.STRAIN)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
        # The interpreter is another installation, and the path of this one would put
        # its modules in front of the ones that installation was built with

        result = json.loads(completed.stdout.strip().splitlines()[-1])
        phases, counts = np.unique(np.concatenate(mesh.phase), return_counts=True)
        expected = {str(int(i_phase)): int(i_n) for i_phase, i_n in zip(phases, counts)}
        self.assertEqual(result["phases"], expected)
        self.assertTrue(result["finite"])
        # dolfinx found every cell GMMD wrote, each with the phase GMMD gave it

        return result

    def homogeneous_stress(self, dim, young):
        """Average stress along x of a single material under the example's loading."""
        if dim == 2:
            return young * self.STRAIN / (1 - self.POISSON**2)

        return young * self.STRAIN
        # Uniaxial stress, in plane strain in two dimensions

    def check_one_material(self, mesher, microstructure):
        """Solve with every phase alike, which has the homogeneous solution."""
        result = self.run_fenics(
            mesher, microstructure, {1: self.YOUNG, 2: self.YOUNG}
        )
        exact = self.homogeneous_stress(result["dim"], self.YOUNG)
        self.assertAlmostEqual(result["mean_stress_xx"] / exact, 1.0, places=9)
        # Linear elements represent the homogeneous solution exactly, so the stress is
        # the one in closed form to the precision of the solver on any conforming mesh:
        # two cells not sharing the nodes of their common face show, as does a node off
        # the face it belongs to. An element numbered inside out does not, dolfinx
        # integrating over the absolute measure of every cell

    def test_two_dimensional_grid(self):
        self.check_one_material(VoxelMesher([16, 16]), disk_microstructure())

    def test_three_dimensional_grid(self):
        self.check_one_material(VoxelMesher([8, 8, 8]), sphere_microstructure())

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_two_dimensional_triangles(self):
        self.check_one_material(
            GmshMesher(mesh_size=0.1, element_type="tri3"), disk_microstructure()
        )

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_three_dimensional_tetrahedra(self):
        self.check_one_material(
            GmshMesher(mesh_size=0.2, element_type="tetra4"), sphere_microstructure()
        )

    def test_a_small_cell(self):
        scale = 2.0**-30
        self.check_one_material(
            VoxelMesher([16, 16]), disk_microstructure().scaled(scale)
        )
        # The example found the faces within the default tolerance of NumPy, a length
        # of 1e-8, and this cell is narrower than that: every node was held

    @unittest.skipUnless(has_gmsh(), "gmsh is not installed")
    def test_a_large_cell(self):
        scale = 2.0**30
        self.check_one_material(
            GmshMesher(mesh_size=0.1 * scale, element_type="tri3"),
            disk_microstructure().scaled(scale),
        )
        # A node gmsh puts on a face lies there to a few parts in 1e16 of the cell,
        # which in this one is more than a length of 1e-8, so the faces were not found

    def test_the_phases_are_the_materials(self):
        stiffer = 10 * self.YOUNG
        result = self.run_fenics(
            VoxelMesher([16, 16]), disk_microstructure(), {1: self.YOUNG, 2: stiffer}
        )
        stress = result["mean_stress_xx"]
        self.assertGreater(stress, self.homogeneous_stress(2, self.YOUNG))
        self.assertLess(stress, self.homogeneous_stress(2, stiffer))
        # Stiffer particles stiffen the cell, and it stays softer than the particles:
        # the tags reached the materials, and the right ones



def crate_python():
    """Path of a Python that has CRATE, or None when none was given."""
    return os.environ.get("GEOMMICGEN_CRATE_PYTHON")


def uniaxial_strain(young):
    """
    Stresses of one material stretched along x and held along the other axes.

    Parameters
    ----------
    young: float
        Young modulus; the Poisson ratio is the placeholder one.

    Returns
    -------
    tuple
        The stress along x, the modulus of the stretch it is, and Lame's first
        parameter, which the stress across x is the strain times.
    """
    poisson = PLACEHOLDER_ELASTIC[1]
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    modulus = lame + young / (1 + poisson)

    return modulus * PLACEHOLDER_STRAIN, modulus, lame


def laminate_stresses(young):
    """
    Stresses along x of a laminate of two layers of equal thickness, stretched along x.

    Parameters
    ----------
    young: dict
        Young modulus of each of the two phases; the Poisson ratio is the placeholder
        one.

    Returns
    -------
    tuple
        The stress along x of the cell layered along x, and of the cell layered along y.
    """
    moduli = {
        i_phase: uniaxial_strain(i_value)[1:] for i_phase, i_value in young.items()
    }
    compliance = sum(0.5 / i_modulus for i_modulus, _ in moduli.values())
    across = PLACEHOLDER_STRAIN / compliance
    normal = PLACEHOLDER_STRAIN * sum(
        0.5 * i_lame / i_modulus for i_modulus, i_lame in moduli.values()
    ) / compliance
    along = sum(
        0.5 * (i_modulus * PLACEHOLDER_STRAIN
               + i_lame * (normal - i_lame * PLACEHOLDER_STRAIN) / i_modulus)
        for i_modulus, i_lame in moduli.values()
    )
    # Layered along x, the stress along x is the same in both layers and the modulus is
    # their harmonic mean; layered along y, the strain along x is, and the layers differ
    # in the stress across them

    return across, along


@unittest.skipUnless(crate_python(), "no Python with CRATE was given")
class TestCrateRunsTheDeck(unittest.TestCase):
    """Test class for running the example CRATE input files through CRATE itself."""

    def setUp(self):
        """Create a directory for the run, which CRATE fills with its outputs."""
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)

    def run_crate(self, mesh, young=None):
        """Write a grid and its example input file, run CRATE, and read its stress."""
        run_dir = tempfile.mkdtemp(dir=self.run_dir.name)
        written = CrateWriter().write(mesh, os.path.join(run_dir, "m.rgmsh.npy"))
        # A directory for each run: CRATE finding the output of an earlier one asks
        # whether to reuse its clustering, which for another grid would be wrong
        if young:
            with open(written[1]) as deck:
                text = deck.read()
            for i_phase, i_value in young.items():
                block = "{0} elastic 1\nelastic_symmetry isotropic 2\n  E {1}\n".format(
                    i_phase, PLACEHOLDER_ELASTIC[0]
                )
                self.assertIn(block, text)
                text = text.replace(
                    block, block.replace(str(PLACEHOLDER_ELASTIC[0]), str(i_value))
                )
            with open(written[1], "w") as deck:
                deck.write(text)
        # A material is changed where a user would change it, in the file as written

        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [crate_python(), "-u", "-m", "cratepy.main", "m_example.dat",
             run_dir + os.sep],
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=environment,
        )
        log = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, log[-3000:])
        # Should CRATE ask anything, nothing answers, and the question fails the run
        # where it would otherwise wait for good

        results_path = os.path.join(run_dir, "m_example", "m_example.hres")
        with open(results_path) as results:
            header, *rows = [i_line.split() for i_line in results if i_line.strip()]

        return {i_name: float(i_value) for i_name, i_value in zip(header, rows[-1])}

    def check_placeholder(self, mesh):
        """Run the example as written, every phase being the placeholder material."""
        stress = self.run_crate(mesh)
        along, _, lame = uniaxial_strain(PLACEHOLDER_ELASTIC[0])
        self.assertAlmostEqual(stress["stress_11"] / along, 1.0, places=7)
        self.assertAlmostEqual(
            stress["stress_22"] / (lame * PLACEHOLDER_STRAIN), 1.0, places=7
        )
        # One material stretched along x and held across it, whose stresses are known in
        # closed form; CRATE prints nine significant digits of them

    def test_two_dimensional_grid(self):
        self.check_placeholder(VoxelMesher([16, 16]).mesh(disk_microstructure()))

    def test_three_dimensional_grid(self):
        self.check_placeholder(VoxelMesher([8, 8, 8]).mesh(sphere_microstructure()))

    def test_the_grid_is_read_along_the_axes_it_was_written_along(self):
        young = {1: PLACEHOLDER_ELASTIC[0], 2: 10 * PLACEHOLDER_ELASTIC[0]}
        across, along = laminate_stresses(young)
        # A laminate of two layers of equal thickness

        phase_grid = np.ones((16, 16), dtype=int)
        phase_grid[:8] = 2
        for i_grid, i_expected in ((phase_grid, across), (phase_grid.T.copy(), along)):
            with self.subTest(layered_along="x" if i_expected is across else "y"):
                stress = self.run_crate(
                    structured_mesh((16, 16), [1.0, 1.0], i_grid), young
                )
                self.assertAlmostEqual(stress["stress_11"] / i_expected, 1.0, places=7)
        # The first axis of the grid is x, and CRATE reads it so: had the grid been
        # transposed on its way, the two laminates would have swapped answers, which
        # differ by more than half. One cluster a phase is exact for a laminate, whose
        # strain is uniform in each


if __name__ == "__main__":
    unittest.main()
