"""
Module containing the command line entry point that converts an archived microstructure.

A microstructure used to be stored as a pickle of the objects that held it, which only
the version of the package that wrote it can be relied on to read, and which carries
executable content. It is stored as a microstructure file now. This converts the one
into the other, so that a microstructure generated before the change stays readable.
"""

import argparse
import os
import pickle

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.iofuncs.file_handling import MIC_FILE_NAME
from geommicgen.iofuncs.md_state import save_md_state
from geommicgen.iofuncs.microstructure_yaml import write_microstructure_yaml


def convert_mic_command(argv=None):
    """
    Convert an archived microstructure into a microstructure file.

    Parameters
    ----------
    argv: list
        Arguments, taken from the command line when they are not given.

    Returns
    -------
    int
        Status to exit with.
    """
    parser = argparse.ArgumentParser(
        prog="geommicgen-convert-mic",
        description="Convert a microstructure archived as a .mic file into a "
        "microstructure file. Only .mic files this package wrote should be given to it: "
        "reading one runs the code it carries.",
    )
    parser.add_argument("mic", help="archived .mic file to be converted")
    parser.add_argument(
        "-o",
        "--output",
        help="microstructure file to be written, mic.yaml beside the input by default",
    )
    arguments = parser.parse_args(argv)

    with open(arguments.mic, "rb") as mic_file:
        archive = pickle.load(mic_file)
    # The archive is the dictionary the pickle-based save wrote, holding the
    # microstructure and the generation method that produced it

    file_path = arguments.output
    if file_path is None:
        file_path = os.path.join(os.path.dirname(arguments.mic), MIC_FILE_NAME)

    write_microstructure_yaml(
        archive["microstructure"],
        file_path,
        provenance={"converted_from": os.path.basename(arguments.mic)},
    )
    print(file_path)

    if archive.get("generation_method") is not None:
        print(
            save_md_state(
                os.path.dirname(file_path) or ".", archive["generation_method"]
            )
        )
        # TODO: a run archived before the records were brought back to the user's
        # units kept its overlaps and paths in the normalised box's, and the state
        # written from it cannot be told apart from one written now
    # The histories the motion analysis plots were part of the archive, and are written
    # beside the microstructure rather than inside it

    return 0
