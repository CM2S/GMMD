""" Unit test regarding the app.py module"""

import unittest
from unittest.mock import Mock, patch

from geommicgen.app import run_program
import io
from contextlib import redirect_stdout


class RunProgramTests(unittest.TestCase):
    " Tests for geommicgen.app.run_program, with every external dependency mocked "

    def _build_all_options(self):
        " Build a fresh, complete, valid all_options dict (a new object every call) "
        return {
            "problem_type": 1,
            "n_dp_samples": 1,
            "save_min": False,
            "mic_gen_descriptors": {"matrix": {"phase_type": 1}},
            "mic_gen_parameters": {
                "rve_dimensions": [1, 1],
                "max_residue_per_particle": 10,
                "max_step": 100,
                "max_steps_to_relax": 10,
                "dt": 0.01,
                "min_distance": 0.1,
                "save_history": False,
                "type_initial_configuration" : "random",
                "speed_up_scheme" : "Cell"
            },
            "post_proc": {},
            'mesh_options' : {'femsh': {'mesh_size': 0.1, 'element_type': 'tri3'}, 'rgmsh': {'n_voxels_dims': [[1000, 1000]]}}
        }


    def _build_get_arguments_from_command_line(
            input_file_path = 'input_file_path',
            input_file_dir = 'input_file_dir',
            input_file_name = 'input_file_name',
            ext = '.mdsim',
            previous_mic_path = None
            ):
        
        return (
            input_file_path,
            input_file_dir,
            input_file_name,
            ext,
            previous_mic_path,
        )

    def setUp(self):
        # fileio patcher
        fileio_patcher = patch("geommicgen.app.fileio")
        self.mock_fileio = fileio_patcher.start()
        self.addCleanup(fileio_patcher.stop)

        self.mock_fileio.get_arguments_from_command_line.return_value = self._build_get_arguments_from_command_line()

        self.mock_fileio.create_sample_results_directory.return_value = ["sample_dir", "sample_file_path"]


        # print_funcs patcher
        print_funcs_patcher = patch("geommicgen.app.print_funcs")
        self.mock_print_funcs = print_funcs_patcher.start()
        self.addCleanup(print_funcs_patcher.stop)

        self.mock_print_funcs.print_initial_message.return_value = None

        # top_level_reader patcher
        top_level_reader_patcher = patch("geommicgen.app.top_level_reader")
        self.mock_top_level_reader = top_level_reader_patcher.start()
        self.addCleanup(top_level_reader_patcher.stop)

        self.mock_top_level_reader.all_options = self._build_all_options()

        #MolecularDunamicsSimulation patch
        MolecularDynamicsSimulation_patcher = patch("geommicgen.app.MolecularDynamicsSimulation")
        self.mock_MolecularDynamicsSimulation = MolecularDynamicsSimulation_patcher.start()
        self.addCleanup(MolecularDynamicsSimulation_patcher.stop)

        # post_proc patcher
        post_proc_patcher = patch("geommicgen.app.post_proc")
        self.mock_post_proc = post_proc_patcher.start()
        self.addCleanup(post_proc_patcher.stop)

    def test_missing_mandatory_parameters(self):
        " Missing mandatory parameter must raise KeyError "
        mandatory_top_level_keys = [
            "n_dp_samples",
            "problem_type",
            "mic_gen_descriptors",
            "mic_gen_parameters",
        ]
        for key in mandatory_top_level_keys:
            with self.subTest(key):
                all_options = self._build_all_options()
                del all_options[key]
                self.mock_top_level_reader.all_options = all_options
                with self.assertRaisesRegex(KeyError, "Mandatory parameter not supplied."):
                    run_program()
        with self.subTest():
            all_options = self._build_all_options()
            del all_options["mic_gen_parameters"]["rve_dimensions"]
            self.mock_top_level_reader.all_options = all_options
            with self.assertRaises(KeyError):
                run_program()


    def test_number_samples_error(self):
        " Invalid values for n_dp_samples (non-positive or non-integer) must raise ValueError "
        for invalid_n_dp_sample in [0, -1, 2.1]:
            with self.subTest(invalid_n_dp_sample):
                self.mock_top_level_reader.all_options['n_dp_samples'] = invalid_n_dp_sample
                with self.assertRaisesRegex(ValueError,"Number of samples must be a positve integer larger than 1."):
                    run_program()

    def test_invalid_mesh_options(self):
        " An unsupported 'mesh_options' value must raise ValueError "
        self.mock_top_level_reader.all_options['mesh_options'] = 'invalid_option'
        with self.assertRaises(ValueError):
            run_program()

    def test_invalid_file_extension(self):
        " An unsupported input file extension must raise ValueError "
        self.mock_fileio.get_arguments_from_command_line.return_value = self._build_get_arguments_from_command_line(ext = '.bogus')
        with self.assertRaisesRegex(ValueError, "Unknown input file extension: .bogus"):
            run_program()

    def test_error_missing_MDsim_mandatory_parameter(self):
        " MIssing atleast one mandatory MD simulation parameter must raise KeyError "
        MDsim_mandatory_parameters = [
            'max_residue_per_particle',
            'max_step',
            'max_steps_to_relax',
            'dt',
            'min_distance',
            'type_initial_configuration',
            'save_history'
        ]
        for key in MDsim_mandatory_parameters:
            with self.subTest(key):
                all_options = self._build_all_options()
                del all_options["mic_gen_parameters"][key]
                self.mock_top_level_reader.all_options = all_options
                with self.assertRaisesRegex(KeyError, "Missing mandatory parameter defining a MD simulation."):
                    run_program()

    def test_no_matrix_phase_specified(self):
        " Not specifying a matrix phase must raise ValueError "
        del self.mock_top_level_reader.all_options["mic_gen_descriptors"]["matrix"]
        with self.assertRaisesRegex(ValueError, "No matrix phase was specified."):
            run_program()


    def test_read_thermostats_missing_parameters(self):
        # thermostat options without their other mandatory parameters
        thermostat_options=[
            "isokinetic",
            "multi_temperature",
            "berendsen"
        ]
        for key in thermostat_options:
            with self.subTest(key):
                sample = self._build_all_options()
                sample["mic_gen_parameters"]["thermostat"] = key
                self.mock_top_level_reader.all_options = sample
                with self.assertRaises(KeyError):
                    run_program()


    def test_cell_speed_up_scheme_is_default(self):
        " When 'speed_up_scheme' is not set, CellList is used as the default "
        self.mock_top_level_reader.all_options["mic_gen_parameters"]["speed_up_scheme"] = None
        with patch("geommicgen.app.CellList") as mock_CellList, \
            patch("geommicgen.app.MolecularDynamicsSimulation") as mock_MolecularDynamicsSimulation ,\
            patch("geommicgen.app.post_proc"):
            mock_MolecularDynamicsSimulation.generate_microstructure.return_value=None
            run_program()
            mock_CellList.assert_called_once()

    def test_speed_up_schemes_are_read(self):
        " Check that each 'speed_up_scheme' option builds and wires up the matching class "
        speed_up_schemes = {
            "Cell": {"class_path": "geommicgen.app.CellList", "extra_parameter": None},
            "Verlet": {"class_path": "geommicgen.app.VerletList", "extra_parameter": "verlet_factor"},
            "Verlet2": {"class_path": "geommicgen.app.VerletPartialUpdate", "extra_parameter": "verlet_factor"},
            "Naive": {"class_path": "geommicgen.app.Naive", "extra_parameter": None},
        }

        for speed_up_scheme, scheme_info in speed_up_schemes.items():
            with self.subTest(speed_up_scheme):
                # Start from a fresh, complete set of options for every scheme
                all_options = self._build_all_options()
                all_options["mic_gen_parameters"]["speed_up_scheme"] = speed_up_scheme
                if scheme_info["extra_parameter"] is not None:
                    all_options["mic_gen_parameters"][scheme_info["extra_parameter"]] = 1.0
                self.mock_top_level_reader.all_options = all_options

                # Reset call history left over from the previous scheme's iteration
                self.mock_MolecularDynamicsSimulation.reset_mock()

                with patch(scheme_info["class_path"]) as mock_speed_up_scheme_class:
                    run_program()

                    mock_speed_up_scheme_class.assert_called_once()
                    self.mock_MolecularDynamicsSimulation.return_value.set_speed_up_scheme.assert_called_once_with(
                        mock_speed_up_scheme_class.return_value
                    )


    def test_previous_mic_path_is_not_none(self):
        " When previous_mic_path is given, run_program loads that sample and post-processes it directly "
        self.mock_fileio.get_arguments_from_command_line.return_value = self._build_get_arguments_from_command_line(
            previous_mic_path="previous_mic_path"
        )
        mock_current_sample = Mock(rve_dims=[1, 1])
        mock_current_mic_generator = Mock()
        self.mock_fileio.load_previous_sample.return_value = (mock_current_sample, mock_current_mic_generator)

        run_program()

        # The previous sample is loaded and passed straight through to post-processing
        self.mock_fileio.load_previous_sample.assert_called_once_with("previous_mic_path")
        self.mock_post_proc.assert_called_once()
        _, called_current_sample, called_current_mic_generator, _, called_post_proc_options = (
            self.mock_post_proc.call_args.args
        )
        self.assertIs(called_current_sample, mock_current_sample)
        self.assertIs(called_current_mic_generator, mock_current_mic_generator)
        self.assertEqual(called_post_proc_options, self.mock_top_level_reader.all_options["post_proc"])

        # This branch skips microstructure generation entirely
        self.mock_fileio.create_sample_results_directory.assert_not_called()
        self.mock_MolecularDynamicsSimulation.assert_not_called()


    def test_save_min(self):
        self.mock_top_level_reader.all_options["save_min"] = True
        run_program()
        self.mock_fileio.save_mic.assert_called_once()
        self.mock_fileio.save_status.assert_called_once()
        self.mock_fileio.delete_screen.assert_called_once()