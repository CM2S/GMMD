"""Module for comparing samples."""
import sys
import os

import pickle
from iofuncs.file_handling import load_previous_sample
import postproc.plotfuncs.plotting_functions as my_plt
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


def compare_samples(file_path_mic_1, file_path_mic_2, axs, **kwargs):
    results_dir = ""
    samples_info_original = load_all_samples(file_path_mic_1)
    samples_info_ave = load_all_samples(file_path_mic_2)
    if "colors" in kwargs:
        colors = kwargs["colors"]
    else:
        colors = my_plt.generate_colors(2)
    artists_1 = []
    for sample in samples_info_original:
        sample_mic_generator = sample[1]
        artists_1 += my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[0],
        )
    artists_2 = []
    for sample in samples_info_ave:
        sample_mic_generator = sample[1]
        artists_2 += my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[1],
        )
    return artists_1, artists_2


if __name__ == "__main__":
    results_dir = sys.argv[1]
    samples_info_original = load_all_samples(sys.argv[2])
    samples_info_ave = load_all_samples(sys.argv[3])
    fig, axs, _ = my_plt.create_figure()
    colors = my_plt.generate_colors(2)
    artists = []
    labels_original = []
    print("here", sys.argv)
    print("samples", samples_info_original)
    for sample in samples_info_original:
        labels_original.append("_nolegend_")
        sample_mic_generator = sample[1]
        artists += my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[0],
        )
        print("here", labels_original)
    labels_original[0] = "original"
    labels_roll = []
    for sample in samples_info_ave:
        sample_mic_generator = sample[1]
        labels_roll.append("_nolegend_")
        artists += my_plt.plot_overlap_history(
            sample_mic_generator.total_overlap_history,
            sample_mic_generator.max_residue,
            results_dir,
            # temp_change=True,
            # temp_change_steps=sample_mic_generator.thermostat.temp_change_steps,
            axes=axs,
            color=colors[1],
        )

    labels_roll[0] = "rolling average"
    print("labels", labels_original + labels_roll)
    my_plt.create_legend(artists, labels_original + labels_roll, axs)
    plt.savefig(os.path.join(results_dir, "{0}.pdf".format(sys.argv[4])))
