"""Module containing the printing functions."""

import datetime
import logging
import os
import sys

from tabulate import tabulate

LOGGER = logging.getLogger("geommicgen")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.NullHandler())
# Everything the program reports passes through this logger. The package attaches no
# handler of its own beyond the null one, as a library should: an entry point attaches
# the terminal, a run attaches its screen file, and a script that imports the package
# hears nothing unless it asks to

FORMATTER = logging.Formatter("%(message)s")
# The lines are written as they are given, so that the terminal and the screen file
# read exactly as the print statements they replace did

SCREEN_FILE_NAME = "mic.screen"


class TerminalHandler(logging.StreamHandler):
    """Handler writing to whatever *sys.stdout* is when a record is emitted."""

    def __init__(self):
        """Initializer for the TerminalHandler class."""
        super().__init__(sys.stdout)

    @property
    def stream(self):
        """The current standard output."""
        return sys.stdout

    @stream.setter
    def stream(self, value):
        """Ignore the stream given, since it is looked up at each record instead."""

    # print looks the standard output up at every call, and so must this, so that
    # output redirected by the caller is still seen -- which is how the tests read it


_TERMINAL = None
_SCREEN = None
# The handlers attached at the moment, if any, so that each can be detached again


def log_to_terminal():
    """
    Send what the program reports to the terminal, once.

    Returns
    -------
    `.TerminalHandler`
        The handler attached, whether by this call or an earlier one.
    """
    global _TERMINAL
    if _TERMINAL is None:
        _TERMINAL = TerminalHandler()
        _TERMINAL.setLevel(logging.INFO)
        _TERMINAL.setFormatter(FORMATTER)
        LOGGER.addHandler(_TERMINAL)
    # Called by every entry point, and idempotent so that the tests can call it too

    return _TERMINAL


def screen_to(directory):
    """
    Send what the program reports to the screen file of a directory, or stop doing so.

    Parameters
    ----------
    directory: str
        Directory the screen file is written in, appended to when it exists. None
        detaches the current screen file and attaches nothing.

    Returns
    -------
    str
        Path of the screen file, or None when there is none.
    """
    global _SCREEN
    if _SCREEN is not None:
        LOGGER.removeHandler(_SCREEN)
        _SCREEN.close()
        _SCREEN = None
    if directory is None:
        return None

    path = os.path.join(directory, SCREEN_FILE_NAME)
    _SCREEN = logging.FileHandler(path, mode="a")
    _SCREEN.setLevel(logging.DEBUG)
    _SCREEN.setFormatter(FORMATTER)
    LOGGER.addHandler(_SCREEN)
    # The file takes everything, the traceback of a failed discretisation included,
    # where the terminal is kept to what a user needs

    return path


def print_initial_message(input_file_path):
    """Print initial message."""
    print_to_file("\n")
    print_to_file("Geometrical microstructure generation")
    print_to_file("=" * 80)
    print_to_file("Computational Multi-Scale Modelling of".rjust(80))
    print_to_file("Solids and Structures Research Group".rjust(80))
    print_to_file("\n\n")
    print_to_file("Input file: {0}".format(input_file_path))
    print_to_file("\n")
    print_to_file(
        "Starting program execution at : {0}\n".format(datetime.datetime.now())
    )
    print_to_file("\n")


def print_output(filepath):
    """Print output."""
    print_to_file("Output")
    print_to_file("=" * 80 + "\n")
    print_to_file("Microstructure output file: {0}\n".format(filepath))


def print_femsh_output(filepath):
    """Print finite element mesh output."""
    print_to_file("\t Output file: {0}".format(filepath))


def print_rgmsh_output(filepath):
    """Print regular mesh output."""
    print_to_file("Regular grid mesh:")
    print_to_file("\t Output file: {0}".format(filepath))


def print_final_message_md(time, total_overlap, number_iterations, max_overlap):
    """Print final message for molecular dynamics simulation."""
    print_to_file("")
    print_to_file("MD simulation results")
    print_to_file("=" * 80 + "\n")
    print_to_file("Total iterations: {0}".format(number_iterations))
    print_to_file("Simulation time: {:.3f} s".format(time))
    print_to_file("Total overlap: {:.2e}".format(total_overlap))
    print_to_file("Maximum overlap: {:.2e}".format(max_overlap))
    print_to_file("\n")


def print_to_file(message):
    """Report a line, to the terminal and to the screen file of the current run."""
    LOGGER.info(message)


def print_to_terminal_refresh(step, total_overlap, **kwargs):
    """Print info about the current iteration."""
    if kwargs.get("first"):
        # First meassage containing information about the iteration
        print("MD simulation info")
        print("=" * 80 + "\n")
        print("Step: {0}".format(step))
        print("Total Overlap: {:.2e}".format(total_overlap))
        # print("Relative Energy: {:.2e}".format(relative_energy))
        # print("Kinetic Energy: {:.2e}".format(kin_energy))
    else:
        for _ in range(2):
            print("\033[F\033[K", end="")
        print("Step: {0}".format(step))
        print("Total Overlap: {:.2e}".format(total_overlap))
        # print("Relative Energy: {:.2e}".format(relative_energy))
        # print("Kinetic Energy: {:.2e}".format(kin_energy))


def print_microstructure_info(microstructure):
    """Print microstructure info."""
    print_to_file("Microstructure descriptors")
    print_to_file("=" * 80 + "\n")

    for i_phase_name, i_phase in microstructure.phases.items():
        print_to_file("Phase {0}: ({1})".format(i_phase_name, i_phase.type.__name__))
        if i_phase.type.__name__ == "Matrix":
            print_to_file("")
            continue
        volume_fraction = "\t\t- {:.6f}%".format(i_phase.volume_fraction * 100)
        if "vf" in i_phase.descriptors:
            print_to_file(
                "{0} (Specified: {1:.6f}%)".format(
                    volume_fraction, i_phase.descriptors["vf"].value * 100
                )
            )
            print_to_file("")
        else:
            print_to_file(volume_fraction + "\n")
        print_to_file("\t- Number of particles:")
        number = "\t\t- {0}".format(i_phase.number_particles)
        if "n" in i_phase.descriptors:
            print_to_file(
                "{0} (Specified: {1})".format(number, i_phase.descriptors["n"].value)
            )
            print_to_file("")
        else:
            print_to_file(number + "\n")
        # Each line is built whole: a report is a line, not a stream

        for j_descriptor_name, j_descriptor in i_phase.descriptors.items():
            if j_descriptor_name in ("vf", "n"):
                continue
            print_to_file(
                "\t- {0}: ({1})".format(
                    i_phase.type.possible_parameters[j_descriptor_name][0],
                    j_descriptor.__class__.__name__,
                )
            )

            for k_parameter in j_descriptor.__class__.parameters:
                print_to_file(
                    "\t\t- {0}: {1}:".format(
                        k_parameter.capitalize(), getattr(j_descriptor, k_parameter)
                    )
                )
            print_to_file("")


def print_virtual_total_volume_fraction(real_vf, virtual_vf, min_distance):
    """Print real and virtual total particle volume fraction and minimum distance."""
    print_to_file("Real and virtual volume fraction")
    print_to_file("=" * 80 + "\n")
    print_to_file("Total real volume fraction: {0:.3f}%".format(real_vf * 100))
    print_to_file(
        "Total vitual volume fraction: {0:.3f}% (minimum distance: {1:.5f})\n".format(
            virtual_vf * 100, min_distance
        )
    )


def print_particle_progress(index, total):
    """
    Report that one more particle has been dealt with, overwriting the line before.

    Parameters
    ----------
    index: int
        Index of the particle that has been dealt with.

    total: int
        Total number of particles.
    """
    print("\t\t- Particle {0} of {1}".format(index + 1, total))
    if index + 1 != total:
        print("\033[F\033[K", end="")
    # The line is overwritten by the next one, so a long run reports its progress
    # without filling the screen file with a line per particle. This is the terminal
    # being driven, not a report, so it does not go through the logger


def print_final_message(mic_generator, mesh_generators, times_dict):
    """Print final message."""
    print_to_file(80 * "-")

    print_to_file("Ending program execution at : {0}\n".format(datetime.datetime.now()))

    total_time = 0
    if mic_generator.time is not None:
        total_time += mic_generator.time
    for generator in mesh_generators:
        if generator.time is not None:
            total_time += generator.time

    for post_proc_time in times_dict.values():
        total_time += post_proc_time
    # A step that did not finish has no time to report. This runs in a finally block, so
    # an error raised over a missing one would replace the error that stopped the run,
    # and the run would end reporting the wrong thing entirely.

    hours = int(total_time // 3600)
    minutes_rem = int(total_time // 60 - hours * 60)
    print_to_file(
        "Total execution time: {0:.2e}s (~{1}h{2}m)\n".format(
            total_time, hours, minutes_rem
        )
    )

    print_to_file("Execution times:\n")

    def share(duration):
        """Give the percentage of the total a duration is."""
        return round(duration / total_time * 100, ndigits=2) if total_time else 0.0

    data_to_print = []
    if mic_generator.time is not None:
        data_to_print.append(
            [
                "Molecular Dynamics Simulation",
                "{0:.2e}".format(mic_generator.time),
                share(mic_generator.time),
            ]
        )
    for generator in mesh_generators:
        if generator.time is None:
            continue
        name = getattr(generator, "description", None) or type(generator).__name__
        data_to_print.append(
            [name, "{0:.2e}".format(generator.time), share(generator.time)]
        )
        # A step that does not say what it is called is reported under its class name,
        # rather than under whichever name the loop happened to leave behind

    for post_proc_op_name, post_proc_op_time in times_dict.items():
        data_to_print.append(
            [
                post_proc_op_name,
                "{0:.2e}".format(post_proc_op_time),
                share(post_proc_op_time),
            ]
        )
    formated_data = tabulate(data_to_print, headers=["Phase", "Duration(s)", "%"])
    for row in formated_data.split("\n"):
        print_to_file("\t" + row)

    print_to_file("\n")
    print_to_file("{0: ^80}\n".format("Program Completed"))


def print_failed_jobs(jobs):
    """
    Report the discretisations that failed, once everything else has been attempted.

    Parameters
    ----------
    jobs: list
        The `.MeshJob` objects that were run, failed or not.

    Returns
    -------
    list
        The jobs that failed.
    """
    failed = [i_job for i_job in jobs if i_job.error is not None]
    if not failed:
        return failed

    print_to_file("\n" + "=" * 80)
    print_to_file(
        "{0} of the discretisations asked for could not be produced:\n".format(
            len(failed)
        )
    )
    for i_job in failed:
        print_to_file(
            "\t- {0} ({1}): {2}: {3}".format(
                i_job.description,
                i_job.base_name,
                type(i_job.error).__name__,
                i_job.error,
            )
        )
        if i_job.trace is not None:
            LOGGER.debug("\t\t" + i_job.trace.rstrip("\n").replace("\n", "\n\t\t"))
        # The traceback goes to the screen file only: a message is what a user needs,
        # and the frames are what whoever has to fix it needs. The file listens at the
        # debug level and the terminal does not, which is all the routing there is

    print_to_file("")

    return failed
