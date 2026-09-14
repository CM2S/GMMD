"""
Unit tests regarding the module iofuncs.file_handling.py
"""
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

# pylint: disable=import-error
from geommicgen.iofuncs.file_handling import (
    create_sample_results_directory,
    copy_input_file,
    create_design_point_results_directory,
    get_arguments_from_command_line,
    load_previous_sample,
    save_mic,
    save_status,
    delete_screen,
)


class TestFileHandling(unittest.TestCase):
    """Class for the unit tests regarding `.create_sample_results_directory`."""

    def test_sample_results_directory_suffix(self):
        "Check that the results directories have the correct suffixed"
        with tempfile.TemporaryDirectory() as dp_dir:
            for i in range(4):
                results_folder,file_path = create_sample_results_directory(dp_dir)
                self.assertEqual(results_folder, os.path.join(dp_dir, "mic_"+str(i)))


    def test_copy_input_file(self):
        "Check that the input file is copied to the results folder"
        with tempfile.NamedTemporaryFile() as input_file, tempfile.TemporaryDirectory() as results_folder:
            copy_input_file(input_file.name, results_folder)
            copied_file = os.listdir(results_folder)[0]
            expected_file_name = os.path.basename(input_file.name)
            self.assertEqual(copied_file, expected_file_name)
            self.assertTrue(os.path.isfile(os.path.join(results_folder, expected_file_name)))



    def test_create_design_point_results_directory(self):
        with tempfile.TemporaryDirectory() as input_file_dir:
            input_file_name = "input_file_name"
            create_design_point_results_directory(input_file_dir, input_file_name)
            results_folder = os.listdir(input_file_dir)[0]
            self.assertEqual(results_folder, input_file_name)

            # Lets now test when there is already a folder with the results folder name
            # Each new folder must be named input_file_name_1, then input_file_name_2, then input_file_name_3 and so on
            for i in range(1,3):
                with self.subTest(i):
                    create_design_point_results_directory(input_file_dir, input_file_name)
                    results_folder_list = os.listdir(input_file_dir)
                    self.assertTrue((input_file_name+"_"+str(i)) in results_folder_list)


                
    def test_get_arguments_from_command_line(self):
        "Check that the command line arguments are correctly parsed"
        with self.subTest("No input file supplied"):
            with patch.object(sys, "argv", ["script_name"]):
                with self.assertRaises(ValueError):
                    get_arguments_from_command_line()

        with self.subTest("Too many arguments supplied"):
            with patch.object(sys, "argv", ["script_name", "input.txt", "previous.mic", "extra"]):
                with self.assertRaises(ValueError):
                    get_arguments_from_command_line()

        with self.subTest("Only the input file is supplied"):
            input_file_path = os.path.join("some", "dir", "input_file.txt")
            with patch.object(sys, "argv", ["script_name", input_file_path]):
                result = get_arguments_from_command_line()
                self.assertEqual(
                    result,
                    (input_file_path, os.path.join("some", "dir"), "input_file", ".txt", None),
                )

        with self.subTest("Input file and a valid previous microstructure file are supplied"):
            input_file_path = os.path.join("some", "dir", "input_file.txt")
            for previous_ext in (".mic", ".csv", ".txt"):
                with self.subTest(previous_ext=previous_ext):
                    previous_mic_path = os.path.join("some", "dir", "previous" + previous_ext)
                    with patch.object(sys, "argv", ["script_name", input_file_path, previous_mic_path]):
                        result = get_arguments_from_command_line()
                        # Note: `ext` in the returned tuple ends up being the extension of the
                        # previous microstructure file, not of the input file, per the current
                        # implementation of `get_arguments_from_command_line`.
                        self.assertEqual(
                            result,
                            (input_file_path, os.path.join("some", "dir"), "input_file", previous_ext, previous_mic_path),
                        )

        with self.subTest("Input file and a previous file with an invalid extension are supplied"):
            input_file_path = os.path.join("some", "dir", "input_file.txt")
            previous_mic_path = os.path.join("some", "dir", "previous.bad")
            with patch.object(sys, "argv", ["script_name", input_file_path, previous_mic_path]):
                with self.assertRaises(ValueError):
                    get_arguments_from_command_line()


    def test_load_previous_sample(self):
        with patch('geommicgen.iofuncs.file_handling.generate_microstructure_from_csv') as generate_microstructure_from_csv, \
            patch('geommicgen.iofuncs.file_handling.generate_microstructure_from_txt') as generate_microstructure_from_txt:

            temp_previous_mick_folder = tempfile.TemporaryDirectory()

            # test .mic extension
            temp_previous_mick_path = os.path.join(temp_previous_mick_folder.name, "mic.mic")
            with open(temp_previous_mick_path, "wb") as mic_file:
                pickle.dump(
                    {
                        "microstructure": "mic_place_holder",
                        "generation_method": "mic_generation_method_place_holder",
                    },
                    mic_file,
                )
            result1 = load_previous_sample(temp_previous_mick_path)
            self.assertEqual(result1, ("mic_place_holder", "mic_generation_method_place_holder"))

            # test .csv extension
            generate_microstructure_from_csv.return_value = "csv_place_holder"
            temp_previous_mick_path = os.path.join(temp_previous_mick_folder.name, "mic.csv")
            result2 = load_previous_sample(temp_previous_mick_path)
            self.assertEqual(result2, ("csv_place_holder",None))

            # test .txt extension
            generate_microstructure_from_txt.return_value = "txt_place_holder"
            temp_previous_mick_path = os.path.join(temp_previous_mick_folder.name, "mic.txt")
            result3 = load_previous_sample(temp_previous_mick_path)
            self.assertEqual(result3, ("txt_place_holder",None))


    def test_save_mic(self):
        "Check that the microstructure sample is correctly pickled to disk"
        with tempfile.TemporaryDirectory() as sample_dir:
            mic_file_path = os.path.join(sample_dir, "mic.mic")

            with self.subTest("Saves the sample and generation method"):
                save_mic(sample_dir, sentinel.current_sample, sentinel.current_mic_generator, print_out=False)
                self.assertTrue(os.path.isfile(mic_file_path))
                with open(mic_file_path, "rb") as mic_file:
                    saved_info = pickle.load(mic_file)
                self.assertEqual(
                    saved_info,
                    {
                        "microstructure": sentinel.current_sample,
                        "generation_method": sentinel.current_mic_generator,
                    },
                )

            with self.subTest("Overwrites a previously saved mic.mic file"):
                save_mic(sample_dir, sentinel.new_sample, sentinel.new_mic_generator, print_out=False)
                with open(mic_file_path, "rb") as mic_file:
                    saved_info = pickle.load(mic_file)
                self.assertEqual(
                    saved_info,
                    {
                        "microstructure": sentinel.new_sample,
                        "generation_method": sentinel.new_mic_generator,
                    },
                )

    def test_save_status(self):
        """Chack if the status file is saved and if its contents are in the correct formatting."""
        mock_current_mic_generator = Mock(time = 1, status = True)
        mock_current_sample = Mock(total_overlap = 0)
        temp_sample_dir = tempfile.TemporaryDirectory()
        save_status(temp_sample_dir.name, mock_current_sample, mock_current_mic_generator)
        self.assertEqual(os.listdir(temp_sample_dir.name)[0], "status")
        status_file = os.path.join(temp_sample_dir.name, "status")
        with open( status_file , mode = "r") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            third_line = f.readline().strip()
            self.assertEqual(first_line, "Time: 1.000s")
            self.assertEqual(second_line, "Overlap: 0.000")
            self.assertEqual(third_line, "Status: True")