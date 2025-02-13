import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt
from tqdm import tqdm


def calc_sliding_RP_viol(
    times_multi: dict[NDArray[np.float_]],
    clust_ids: NDArray[np.int_],
    n_clust: int,
    bin_size: float = 0.25,
    acceptThresh: float = 0.25,
    window_size: float = 2,
    overlap_tol: int = 5,
) -> NDArray[np.float32]:
    """
    Calculate the sliding refractory period violation confidence for each cluster.
    Args:
        times_multi (list[NDArray[np.float_]]): A list of arrays containing spike times for each cluster.
        clust_ids (NDArray[np.int_]): An array indicating cluster_ids to process. Should be "good" clusters.
        n_clust (int): The total number of clusters (shape of mean_wf or max_clust_id + 1).
        bin_size (float, optional): The size of each bin in milliseconds. Defaults to 0.25.
        acceptThresh (float, optional): The threshold for accepting refractory period violations. Defaults to 0.25.
        window_size (float, optional): The size of the window to calculate refractory period violations in s. Defaults to 2.
        overlap_tol (int, optional): The tolerance for overlap in samples. Defaults to 5.
    Returns:
        NDArray[np.float32]: An array containing the refractory period violation confidence for each cluster.
    """
    # create various refractory periods sizes to test (between 0 and 10.25 ms)
    b = np.arange(0, 10.25, bin_size) / 1000
    bTestIdx = np.array([1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40], dtype="int8")
    bTest = [b[i] for i in bTestIdx]

    RP_conf = np.zeros(n_clust, dtype=np.float32)

    for i in tqdm(clust_ids, desc="Calculating RP viol confs"):
        times = times_multi[i] / 30000  # convert times to seconds
        if times.shape[0] > 1:
            # calculate and avg halves of acg to ensure symmetry
            # keep only second half of acg, refractory period violations are compared from the center of acg
            acg = auto_correlogram(
                times, window_size, bin_size / 1000, overlap_tol / 30000
            )
            half_len = int(acg.shape[0] / 2)
            acg = (acg[half_len:] + acg[:half_len][::-1]) / 2

            acg_cumsum = np.cumsum(acg)
            sum_res = acg_cumsum[bTestIdx - 1]  # -1 bc 0th bin corresponds to 0-0.5 ms

            # create two methods use as reference for acceptable contamination
            # 1. Standard: 0.25 of steady state firing rate (look at 1-2s)
            num_bins_end = acg.shape[0]  # default will be 2s
            num_bins_half = int(num_bins_end / 2)  # default will be 1s
            bin_rate_ss = np.mean(acg[num_bins_half:num_bins_end])
            max_conts_ss = (
                np.array(bTest) / bin_size * 1000 * (bin_rate_ss * acceptThresh)
            )
            confs_ss = 1 - stats.poisson.cdf(sum_res, max_conts_ss)

            # 2. New Approach: If low firing rate or bursty, use 0.1 of max rate, use smoothed acg
            order = 4  # Hz
            cutoff_freq = 500  # Hz
            fs = 1 / bin_size * 1000
            nyqist = fs / 2
            cutoff = cutoff_freq / nyqist
            sos = butter(order, cutoff, btype="low", output="sos")
            smoothed_acg = sosfiltfilt(sos, acg)
            # smoothed_acg = gaussian_filter1d(acg, sigma=1)
            bin_rate_max = np.max(smoothed_acg)
            max_conts_max = (
                np.array(bTest) / bin_size * 1000 * (bin_rate_max * acceptThresh)
            )

            # compute confidence of less than acceptThresh contamination at each refractory period
            confs_max = 1 - stats.poisson.cdf(sum_res, max_conts_max)

            # min confidence of the two methods
            conf = max(np.max(confs_ss), np.max(confs_max))
            RP_conf[i] = 1 - conf

            # Plotting the results
            plt.figure(figsize=(10, 10))

            # Original ACG
            acg = np.concatenate((acg[::-1], acg))
            plt.subplot(2, 1, 1)
            plt.plot(acg, label="Original Half ACG", color="blue")
            plt.title("Original Half ACG")
            plt.xlabel("Time (bins)")
            plt.ylabel("ACG Value")
            plt.legend()

            # Smoothed ACG
            smoothed_acg = np.concatenate((smoothed_acg[::-1], smoothed_acg))
            plt.subplot(2, 1, 2)
            plt.plot(smoothed_acg, label="Smoothed ACG", color="red")
            plt.title("Smoothed ACG")
            plt.xlabel("Time (bins)")
            plt.ylabel("Smoothed ACG Value")
            plt.legend()

            # Display the plots
            plt.tight_layout()
            plt.show()

    return RP_conf


def auto_correlogram(
    c1_times: NDArray[np.float_],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> NDArray[np.float_]:
    """
    Calculates the auto correlogram for a spike train.

    Args:
        c1_times (NDArray): Spike times (sorted least to greatest)
            in seconds.
        window_size (float): Width of cross correlogram window in seconds.
        bin_width (float): Width of cross correlogram bins in seconds.
        overlap_tol (float): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        corrgram (NDArray): The calculated cross-correlogram.
    """
    corrgram = np.zeros((math.ceil(window_size / bin_width)))
    start = 0

    # To calculate the auto-correlogram, we iterate over spikes as reference spikes
    # and count the number of other spikes that fall within window_size of the
    # reference spike.
    for ref_spk in range(c1_times.shape[0]):
        while (start < c1_times.shape[0]) and (
            c1_times[start] < (c1_times[ref_spk] - window_size / 2)
        ):
            start += 1  # start tracks the first in-window spike.

        spk_idx = start  # spk_idx iterates over in-window spikes.
        if spk_idx >= c1_times.shape[0]:
            continue

        while (spk_idx < c1_times.shape[0]) and (
            c1_times[spk_idx] < (c1_times[ref_spk] + window_size / 2)
        ):
            if (ref_spk != spk_idx) and (
                abs(c1_times[ref_spk] - c1_times[spk_idx]) > overlap_tol
            ):
                gram_ind = min(
                    math.floor(
                        (c1_times[ref_spk] - c1_times[spk_idx] + window_size / 2)
                        / bin_width
                    ),
                    corrgram.shape[0] - 1,
                )
                corrgram[gram_ind] += 1
            spk_idx += 1

    return corrgram
