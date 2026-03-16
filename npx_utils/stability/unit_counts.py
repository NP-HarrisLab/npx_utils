import os
from datetime import datetime

import matplotlib.pyplot as plt
import npx_utils as npx
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


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
    return fig, ax


def plot_unit_counts(
    subject_folder,
    implant_day,
    probe_ids=None,
    save_dir=None,
    overwrite=False,
    save_type="png",
    ks_version="4",
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    # only affects plotting in plotMult raster plots
    run_folders = npx.get_run_folders(subject_folder)

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

    for probe_num in tqdm(
        all_probe_folders, "Plotting unit counts...", position=0, unit="probe"
    ):
        # check if file exists
        if save_dir is not None:
            fname = os.path.join(save_dir, f"unit_counts_imec{probe_num}.{save_type}")
            if os.path.exists(fname) and not overwrite:
                # load image and skip
                fig = plt.figure(figsize=(6, 4))
                img = plt.imread(fname)
                plt.imshow(img)
                plt.axis("off")
                continue

        all_probe_ks_folders = all_probe_folders[probe_num]
        probe_ks_folders = npx.sglx_helpers.get_same_channel_positions(
            all_probe_ks_folders
        )
        days = [
            datetime.strptime(
                os.path.basename(os.path.dirname(folder)).split("_")[0], "%Y%m%d"
            )
            for folder in probe_ks_folders
        ]
        if implant_day is None:
            start_day = days[0]
        else:
            start_day = datetime.strptime(implant_day, "%Y%m%d")
        day_list = [(day - start_day).days for day in days]

        fig, ax = plot_subject_unit_counts(probe_ks_folders, day_list)
        ax.set_title(f"{os.path.basename(subject_folder)} imec {probe_num} unit counts")

        if save_dir is not None:
            fname = os.path.join(save_dir, f"unit_counts_imec{probe_num}.{save_type}")
            fig.savefig(fname, dpi=300, format=save_type)
