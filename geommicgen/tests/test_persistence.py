"""
Unit tests regarding the persistence of a microstructure and of the run that produced it.

A generated sample is a microstructure file and, beside it, the state of the generation
run. These tests cover writing that pair, reading it back, and converting a sample
archived by an earlier version of the package.
"""

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np

from geommicgen.iofuncs.convert_mic import convert_mic_command
from geommicgen.iofuncs.file_handling import (
    MIC_FILE_NAME,
    load_previous_sample,
    save_mic,
    save_status,
)
from geommicgen.pipeline import MeshJob
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.iofuncs.md_state import (
    STATE_FILE_NAME,
    load_md_state,
    save_md_state,
)
from geommicgen.micgenmethod.molecular_dynamics_sim import MolecularDynamicsSimulation
from geommicgen.postproc.voronoimetrics.stat_analysis import adjust_rve_dims
from geommicgen.tests.helpers import a_generation_run, disk_microstructure


class MDStateTest(unittest.TestCase):
    """Tests for writing and reading the state of a generation run."""

    def setUp(self):
        """Create a directory for the state file."""
        self.state_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.state_dir)
        self.mic_generator = a_generation_run()

    def test_round_trip(self):
        """The state that is read back is the state that was written."""
        save_md_state(self.state_dir, self.mic_generator)
        state = load_md_state(os.path.join(self.state_dir, STATE_FILE_NAME))

        self.assertEqual(state.step, self.mic_generator.step)
        self.assertEqual(state.time, self.mic_generator.time)
        self.assertEqual(state.status, self.mic_generator.status)
        self.assertEqual(state.max_residue, self.mic_generator.max_residue)
        self.assertEqual(
            state.total_overlap_history, self.mic_generator.total_overlap_history
        )
        self.assertEqual(
            state.kinetic_energy_history, self.mic_generator.kinetic_energy_history
        )
        self.assertEqual(
            state.thermic_energy_history, self.mic_generator.thermic_energy_history
        )
        self.assertEqual(state.all_dt, self.mic_generator.all_dt)
        self.assertEqual(
            state.thermostat.temp_change_steps,
            self.mic_generator.thermostat.temp_change_steps,
        )
        self.assertEqual(state.thermostat.ratio, self.mic_generator.thermostat.ratio)

    def test_position_history_round_trip(self):
        """The recorded motion of every particle is read back as it was recorded."""
        save_md_state(self.state_dir, self.mic_generator)
        state = load_md_state(os.path.join(self.state_dir, STATE_FILE_NAME))

        self.assertEqual(
            len(state.position_center_history),
            len(self.mic_generator.position_center_history),
        )
        for i_read, i_written in zip(
            state.position_center_history, self.mic_generator.position_center_history
        ):
            self.assertEqual(len(i_read), len(i_written))
            for j_read, j_written in zip(i_read, i_written):
                np.testing.assert_allclose(j_read, j_written)

    def test_interrupted_history_is_cut_at_the_common_step(self):
        """A history one particle got an extra entry in is cut where they all reach."""
        self.mic_generator.position_center_history[0].append(np.array([9.0, 9.0]))
        save_md_state(self.state_dir, self.mic_generator)
        state = load_md_state(os.path.join(self.state_dir, STATE_FILE_NAME))

        self.assertEqual(
            [len(i_history) for i_history in state.position_center_history], [3, 3]
        )

    def test_a_run_that_recorded_nothing(self):
        """A run that recorded no motion writes no motion, and says so."""
        mic_generator = MolecularDynamicsSimulation(
            0.0, 10, 5, 1e-3, 0.0, "random", True
        )
        save_md_state(self.state_dir, mic_generator)
        # A generation method that has not run, and has not been given a thermostat
        state = load_md_state(os.path.join(self.state_dir, STATE_FILE_NAME))

        self.assertIsNone(state.position_center_history)
        self.assertIsNone(state.time)
        self.assertIsNone(state.max_residue)
        self.assertEqual(state.step, 0)
        self.assertFalse(state.status)

    def test_no_state_file(self):
        """A microstructure with no state beside it reads back as no state."""
        self.assertIsNone(load_md_state(os.path.join(self.state_dir, STATE_FILE_NAME)))


class SaveMicTest(unittest.TestCase):
    """Tests for writing a sample and reading it back."""

    def setUp(self):
        """Create a directory for the sample."""
        self.sample_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.sample_dir)
        self.microstructure = disk_microstructure()

    def test_writes_a_microstructure_and_a_state(self):
        """A finished sample is a microstructure file and the state beside it."""
        save_mic(
            self.sample_dir, self.microstructure, a_generation_run(), print_out=False
        )

        self.assertTrue(os.path.exists(os.path.join(self.sample_dir, MIC_FILE_NAME)))
        self.assertTrue(os.path.exists(os.path.join(self.sample_dir, STATE_FILE_NAME)))

    def test_writes_no_state_when_there_is_none(self):
        """A checkpoint of the configuration alone writes no state."""
        save_mic(self.sample_dir, self.microstructure, None, print_out=False)

        self.assertTrue(os.path.exists(os.path.join(self.sample_dir, MIC_FILE_NAME)))
        self.assertFalse(os.path.exists(os.path.join(self.sample_dir, STATE_FILE_NAME)))

    def test_load_previous_sample(self):
        """A sample reads back as the microstructure and the state that produced it."""
        save_mic(
            self.sample_dir, self.microstructure, a_generation_run(), print_out=False
        )
        restored, state = load_previous_sample(
            os.path.join(self.sample_dir, MIC_FILE_NAME)
        )

        self.assertEqual(len(restored.particles), len(self.microstructure.particles))
        for i_restored, i_original in zip(
            restored.particles, self.microstructure.particles
        ):
            np.testing.assert_allclose(
                i_restored.position_center, i_original.position_center
            )
        self.assertEqual(state.step, 3)

    def test_load_previous_sample_without_a_state(self):
        """A microstructure another tool wrote reads back with no state."""
        save_mic(self.sample_dir, self.microstructure, None, print_out=False)
        _, state = load_previous_sample(os.path.join(self.sample_dir, MIC_FILE_NAME))

        self.assertIsNone(state)

    def test_load_previous_sample_of_an_unknown_kind(self):
        """A file that is not a microstructure is refused by name."""
        file_path = os.path.join(self.sample_dir, "mic.pickle")
        with open(file_path, "w") as sample_file:
            sample_file.write("")

        with self.assertRaises(ValueError) as context:
            load_previous_sample(file_path)
        self.assertIn(".pickle", str(context.exception))


class ConvertMicTest(unittest.TestCase):
    """Tests for converting a sample archived by an earlier version of the package."""

    def setUp(self):
        """Archive a sample the way the package used to."""
        self.sample_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.sample_dir)
        self.microstructure = disk_microstructure()
        self.archive_path = os.path.join(self.sample_dir, "mic.mic")
        with open(self.archive_path, "wb") as archive:
            pickle.dump(
                {
                    "microstructure": self.microstructure,
                    "generation_method": a_generation_run(),
                },
                archive,
            )

    def test_converts_the_microstructure_and_the_state(self):
        """The archive becomes the pair of files a sample is made of now."""
        self.assertEqual(convert_mic_command([self.archive_path]), 0)

        restored, state = load_previous_sample(
            os.path.join(self.sample_dir, MIC_FILE_NAME)
        )
        self.assertEqual(len(restored.particles), len(self.microstructure.particles))
        self.assertEqual(state.step, 3)

    def test_converts_to_the_named_file(self):
        """The converted microstructure goes where it was asked to go."""
        file_path = os.path.join(self.sample_dir, "elsewhere.yaml")
        self.assertEqual(convert_mic_command([self.archive_path, "-o", file_path]), 0)

        self.assertTrue(os.path.exists(file_path))


class SaveStatusTest(unittest.TestCase):
    """Tests for what a sample records about how it turned out."""

    def setUp(self):
        """Create a directory for the sample."""
        self.sample_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.sample_dir)
        self.microstructure = disk_microstructure()
        self.microstructure.total_overlap = 0.0
        self.mic_generator = a_generation_run()

    def status_lines(self):
        """Read back the lines of the status file."""
        with open(os.path.join(self.sample_dir, "status")) as status_file:
            return status_file.read().splitlines()

    def test_the_generation_alone(self):
        """A status written before the meshing says nothing about it."""
        save_status(self.sample_dir, self.microstructure, self.mic_generator)

        self.assertEqual(
            self.status_lines(), ["Time: 12.500s", "Overlap: 0.000", "Status: True"]
        )

    def test_one_line_per_discretisation(self):
        """Every discretisation asked for says whether it was produced."""
        produced = MeshJob(VoxelMesher([8, 8]), [], "grid_8_8")
        produced.run(self.microstructure, os.path.join(self.sample_dir, "meshes"))
        refused = MeshJob(VoxelMesher([8, 8, 8]), [], "grid_8_8_8")
        refused.run(self.microstructure, os.path.join(self.sample_dir, "meshes"))
        self.assertIsNone(produced.error)
        self.assertIsNotNone(refused.error)
        # The second grid has three directions and the microstructure has two

        save_status(
            self.sample_dir,
            self.microstructure,
            self.mic_generator,
            [produced, refused],
        )
        lines = self.status_lines()

        self.assertEqual(lines[3], "Mesh grid_8_8 (Regular mesh generation): ok")
        self.assertTrue(
            lines[4].startswith("Mesh grid_8_8_8 (Regular mesh generation): failed:"),
            lines[4],
        )
        self.assertIn("ValueError", lines[4])

    def test_a_discretisation_that_never_ran_does_not_say_ok(self):
        """A job with no error has not thereby succeeded."""
        never_ran = MeshJob(VoxelMesher([8, 8]), [], "grid_8_8")

        save_status(
            self.sample_dir, self.microstructure, self.mic_generator, [never_ran]
        )

        self.assertEqual(
            self.status_lines()[3], "Mesh grid_8_8 (Regular mesh generation): not run"
        )
        # The final configuration is plotted before anything is meshed, so a run that
        # dies there reaches the status file with jobs that carry no error because they
        # never had the chance to fail


class AdjustRVEDimsTest(unittest.TestCase):
    """Tests for adjusting the RVE around the particles it holds."""

    def test_the_particles_are_left_where_they_are(self):
        """Adjusting the RVE reports the shift rather than applying it."""
        microstructure = disk_microstructure()
        original = [
            i_particle.position_center.copy() for i_particle in microstructure.particles
        ]

        _, adjusted = adjust_rve_dims(microstructure.particles)

        for i_particle, i_original in zip(microstructure.particles, original):
            np.testing.assert_allclose(i_particle.position_center, i_original)
        self.assertEqual(len(adjusted), len(original))
        # The centers the analysis works with are the adjusted ones, and the ones the
        # microstructure holds are untouched


if __name__ == "__main__":
    unittest.main()
