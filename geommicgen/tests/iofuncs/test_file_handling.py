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
    save_status,
)
# The reading of the command line, and the reading and writing of a sample, are tested
# where they are now done: test__main__.py and test_persistence.py


class TestFileHandling(unittest.TestCase):
    """Class for the unit tests regarding `.create_sample_results_directory`."""

    def test_sample_results_directory_suffix(self):
        "Check that the results directories have the correct suffixed"
        with tempfile.TemporaryDirectory() as dp_dir:
            for i in range(4):
                results_folder = create_sample_results_directory(dp_dir)
                self.assertEqual(results_folder, os.path.join(dp_dir, "mic_"+str(i)))


    def test_copy_input_file(self):
        "Check that the input file is copied to the results folder"
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as results_folder:
            # The input file must be closed before copying: Windows does not allow
            # opening a file that is still held open by NamedTemporaryFile
            input_file_path = os.path.join(input_dir, "input_file.mdsim")
            with open(input_file_path, "w") as input_file:
                input_file.write("input file contents")
            copy_input_file(input_file_path, results_folder)
            copied_file = os.listdir(results_folder)[0]
            expected_file_name = os.path.basename(input_file_path)
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
            self.assertEqual(second_line, "Overlap: 0.000e+00")
            self.assertEqual(third_line, "Status: True")