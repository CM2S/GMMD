"""
Unit tests regarding the module iofuncs.printing.py
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import sentinel, Mock, patch, call

import numpy as np

# pylint: disable=import-error
from geommicgen.iofuncs.printing import print_to_file


class TestPrinting(unittest.TestCase):
    "Class for the unit tests regarding `.print_to_file`."

    def test_print_to_file(self):
        "Check that print_to_file writes to the screen file and/or the terminal as requested"
        with tempfile.TemporaryDirectory() as screen_dir, \
                patch("geommicgen.iofuncs.printing.SCREEN_DIR", screen_dir):
            screen_path = os.path.join(screen_dir, "mic.screen")

            with self.subTest("Creates the screen file on the first write"):
                out = io.StringIO()
                with redirect_stdout(out):
                    print_to_file("first message")
                self.assertTrue(os.path.isfile(screen_path))
                with open(screen_path) as screen_file:
                    self.assertEqual(screen_file.read(), "first message\n")
                self.assertEqual(out.getvalue(), "first message\n")

            with self.subTest("Appends to the screen file on subsequent writes"):
                out = io.StringIO()
                with redirect_stdout(out):
                    print_to_file("second message")
                with open(screen_path) as screen_file:
                    self.assertEqual(screen_file.read(), "first message\nsecond message\n")
                self.assertEqual(out.getvalue(), "second message\n")

            with self.subTest("to_screen=False skips writing the screen file"):
                os.remove(screen_path)
                out = io.StringIO()
                with redirect_stdout(out):
                    print_to_file("terminal only", to_screen=False)
                self.assertFalse(os.path.exists(screen_path))
                self.assertEqual(out.getvalue(), "terminal only\n")

            with self.subTest("to_terminal=False skips printing to the terminal"):
                out = io.StringIO()
                with redirect_stdout(out):
                    print_to_file("file only", to_terminal=False)
                self.assertEqual(out.getvalue(), "")
                with open(screen_path) as screen_file:
                    self.assertEqual(screen_file.read(), "file only\n")

            with self.subTest("Respects a custom `end`"):
                os.remove(screen_path)
                out = io.StringIO()
                with redirect_stdout(out):
                    print_to_file("no newline", end="")
                self.assertEqual(out.getvalue(), "no newline")
                with open(screen_path) as screen_file:
                    self.assertEqual(screen_file.read(), "no newline")
