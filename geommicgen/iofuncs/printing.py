def print_initial_message():
    printToFile("name program")
    printToFile("stuff, my name, CM2S")
    printToFile("=" * 80)
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


def print_final_message(time, total_overlap, number_iterations, sample, max_overlap):
    printToFile("=" * 80)
    printToFile("**RESULTS**\n")
    printToFile("Total iterations: {0}".format(number_iterations))
    printToFile("Simulation time: {:.3f} s".format(time))
    printToFile("Total overlap: {:.2e}".format(total_overlap))
    printToFile("Maximum overlap: {:.2e}".format(max_overlap))
def print_rgmsh_output(filepath):
    print_to_file("Regular grid mesh:")
    print_to_file("\t Output file: {0}".format(filepath))


def printToFile(message, end="\n"):
    """Print to the screen file of corresponding to the current `.Particle` object"""
    with open("temp.screen", "a") as screen:
        print(message, file=screen, end=end)
    print(message, end=end)


def print_to_terminal_refresh(
    step, total_overlap, relative_energy, kin_energy, **kwargs
):
    """Print info about the current iteration."""
    if kwargs.get("first"):
        # First meassage containing information about the iteration
        printToFile("**SIMULATION INFO**\n")
        print("Step: {0}".format(step))
        print("Total Overlap: {:.2e}".format(total_overlap))
    else:
        for i in range(2):
            print("\033[F\033[K", end="")
        print("Step: {0}".format(step))
        print("Total Overlap: {:.2e}".format(total_overlap))


def print_microstructure_info(microstucutre):

    print_to_file("Microstructure descriptors")
    print_to_file("=" * 80 + "\n")

    for i_phase_name, i_phase in microstucutre.phases.items():
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
