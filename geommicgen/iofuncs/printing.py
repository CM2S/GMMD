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
