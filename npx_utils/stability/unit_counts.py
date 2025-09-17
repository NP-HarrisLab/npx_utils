import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import npx_utils as npx


def plot_subject_unit_counts(ks_folders, day_list):
    good_counts = np.zeros(len(ks_folders))
    non_noise_counts = np.zeros(len(ks_folders))

    for i, ks_folder in enumerate(ks_folders):
        label_path = os.path.join(ks_folder, "cluster_group.tsv")
        labels = pd.read_csv(label_path, sep="\t", index_col="cluster_id")
        num_good = (labels["label"] == "good").sum()
        num_non_noise = (labels["label"] != "noise").sum()
        good_counts[i] = num_good
        non_noise_counts[i] = num_non_noise

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(day_list, good_counts, label=f"good")
    ax.plot(day_list, non_noise_counts, "--", label=f"non-noise")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of units")
    ax.legend()
    return fig


def main():
    # samples for calling the above functions
    # note that the file paths, etc are specific, alter to match your data
    subject_folder = r"D:\Psilocybin\Cohort_3\T22"
    implant_day = "20250708"  # if None date will go based on first recording day
    save_dir = os.path.join(subject_folder, "stability")  # None if show
    save_type = "png"  # svg
    ks_version = "4"
    probe_ids = None  # get all

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    # only affects plotting in plotMult raster plots
    run_folders = npx.get_run_folders(subject_folder)
    days = [
        datetime.strptime(
            os.path.basename(os.path.dirname(run_folder)).split("_")[0], "%Y%m%d"
        )
        for run_folder in run_folders
    ]
    if implant_day is None:
        start_day = days[0]
    else:
        start_day = datetime.strptime(implant_day, "%Y%m%d")
    day_list = [(day - start_day).days for day in days]

    # group folders by probes
    ks_folders = []
    for folder in run_folders:
        ks_folders.extend(npx.get_ks_folders(folder, ks_version))

    all_probe_folders = npx.get_probe_folders(ks_folders)
    if probe_ids is not None:
        all_probe_folders = {
            probe_id: all_probe_folders[probe_id]
            for probe_id in probe_ids
            if probe_id in all_probe_folders
        }

    pbar1 = tqdm(all_probe_folders, "Processing probes...", position=0)
    for probe_num in pbar1:
        pbar1.set_description(f"Processing probe {probe_num}")
        all_probe_ks_folders = all_probe_folders[probe_num]
        probe_ks_folders = npx.sglx.sglx_helpers.get_same_channel_positions(
            all_probe_ks_folders
        )
        # get indices of removed ks_folders
        removed_indices = [
            i
            for i, folder in enumerate(all_probe_ks_folders)
            if folder not in probe_ks_folders
        ]
        # remove from day_list
        updated_day_list = [
            day for i, day in enumerate(day_list) if i not in removed_indices
        ]

        fig = plot_subject_unit_counts(probe_ks_folders, updated_day_list)

        if save_dir is not None:
            fname = os.path.join(save_dir, f"unit_counts_imec{probe_num}.{save_type}")
            fig.savefig(fname, dpi=300, format=save_type)
        else:
            plt.show()


if __name__ == "__main__":
    main()
