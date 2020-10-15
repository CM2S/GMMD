"""Module for comparing samples."""
import sys
import os

import pickle
from iofuncs.file_handling import load_previous_sample
import plotting_functions as my_plt
import matplotlib.pyplot as plt


def load_samples(*list_file_paths):
    samples_info = []
    for sample_file_path in list_file_paths:
        samples_info.append(load_previous_sample(sample_file_path))

    return samples_info


def load_all_samples(results_dir):
    ind = 0
    samples_info = []
    while True:
        current_sample_dir = os.path.join(results_dir, "mic_{0}".format(ind))
        if os.path.exists(current_sample_dir):
            sample_file_path = os.path.join(current_sample_dir, "mic.mic")
            samples_info.append(load_previous_sample(sample_file_path))
        else:
            break
        ind += 1
    return samples_info


if __name__ == "__main__":
    results_dir = sys.argv[1]
    samples_info_original = load_all_samples(sys.argv[2])
    samples_info_ave = load_all_samples(sys.argv[3])
    fig, axs, _ = my_plt.create_figure()
    colors = my_plt.generate_colors(2)
    for sample in samples_info_original:
        sample_mic_generator = sample[1]
        my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[0],
        )
    for sample in samples_info_ave:
        sample_mic_generator = sample[1]
        my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[1],
        )
    plt.savefig(os.path.join(results_dir, "relative_energy_disp.pdf"))
