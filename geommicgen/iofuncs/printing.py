"""Module containing the printing functions."""

import os

# Screen directory
SCREEN_DIR = ""


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


def print_analysis_previous(previous_mic_path):
    """Print previous mic path."""
    print_to_file("Previous microstructure: {0}".format(previous_mic_path))
    print_to_file("\n")


def print_output_header():
    print_to_file("Output")
    print_to_file("=" * 80 + "\n")


def print_output(filepath):
    print_to_file("Output")
    print_to_file("=" * 80 + "\n")
    print_to_file("Microstructure output file: {0}".format(filepath))


def print_femsh_output(filepath):
    print_to_file("Finite element method mesh:")
    print_to_file("\t Output file: {0}".format(filepath))


def print_rgmsh_output(filepath):
    print_to_file("Regular grid mesh:")
    print_to_file("\t Output file: {0}".format(filepath))


def print_final_message(time, total_overlap, number_iterations, max_overlap):
    print_to_file("")
    print_to_file("MD simulation results")
    print_to_file("=" * 80 + "\n")
    print_to_file("Total iterations: {0}".format(number_iterations))
    print_to_file("Simulation time: {:.3f} s".format(time))
    print_to_file("Total overlap: {:.2e}".format(total_overlap))
    print_to_file("Maximum overlap: {:.2e}".format(max_overlap))
    print_to_file("\n")


def print_to_file(message, end="\n"):
    """Print to the screen file of corresponding to the current microstructure sample."""
    screen_path = os.path.join(SCREEN_DIR, "mic.screen")
    if os.path.exists(screen_path):
        action = "a"
    else:
        action = "w"
    with open(screen_path, action) as screen:
        print(message, file=screen, end=end)
    print(message, end=end)


def print_to_terminal_refresh(
    step, total_overlap, relative_energy, kin_energy, **kwargs
):
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
        print_to_file("\t\t- {:.6f}%".format(i_phase.volume_fraction * 100), end=" ")
        if "vf" in i_phase.descriptors:
            print_to_file(
                "(Specified: {:.6f}%)".format(i_phase.descriptors["vf"].value * 100)
            )
            print_to_file("")
        else:
            print_to_file("\n")
        print_to_file("\t- Number of particles:")
        print_to_file("\t\t- {0}".format(i_phase.number_particles), end=" ")
        if "n" in i_phase.descriptors:
            print_to_file("(Specified: {0})".format(i_phase.descriptors["n"].value))
            print_to_file("")
        else:
            print_to_file("\n")

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
