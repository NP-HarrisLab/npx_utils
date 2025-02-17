import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


def calc_sliding_RP_viol(
    times_multi: dict[NDArray[np.float_]],
    clust_ids: NDArray[np.int_],
    n_clust: int,
    bin_size=0.25,
    acceptThresh: float = 0.25,
    window_size: float = 2,
    overlap_tol: int = 5,
):
    RP_viol = {}

    for i in tqdm(clust_ids, desc="Calculating RP viol confs"):
        times = times_multi[i] / 30000  # convert times to seconds

        if times.shape[0] <= 1:
            RP_viol[i] = 0
        else:
            acg = auto_correlogram(
                times_multi[i], window_size, bin_size / 1000, overlap_tol / 30000
            )
            RP_viol[i] = _sliding_RP_viol(
                acg,
                bin_size,
                acceptThresh,
            )

    return RP_viol


def _sliding_RP_viol(
    correlogram,
    bin_size: float = 0.25,
    acceptThresh: float = 0.1,
) -> float:
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

    # calculate and avg halves of acg to ensure symmetry
    # keep only second half of acg, refractory period violations are compared from the center of acg
    half_len = int(correlogram.shape[0] / 2)
    correlogram = (correlogram[half_len:] + correlogram[:half_len][::-1]) / 2

    acg_cumsum = np.cumsum(correlogram)
    sum_res = acg_cumsum[bTestIdx - 1]  # -1 bc 0th bin corresponds to 0-0.5 ms

    # low-pass filter acg and use max as baseline event rate
    order = 4  # Hz
    cutoff_freq = 100  # Hz
    fs = 1 / bin_size * 1000
    nyqist = fs / 2
    cutoff = cutoff_freq / nyqist
    sos = butter(order, cutoff, btype="low", output="sos")
    smoothed_acg = sosfiltfilt(sos, correlogram)

    bin_rate_max = np.max(smoothed_acg)
    max_conts_max = np.array(bTest) / bin_size * 1000 * (bin_rate_max * acceptThresh)
    # compute confidence of less than acceptThresh contamination at each refractory period
    confs = 1 - stats.poisson.cdf(sum_res, max_conts_max)
    rp_viol = 1 - confs.max()

    return rp_viol


def auto_correlogram(
    c1_times: NDArray[np.float_],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> NDArray[np.float_]:
    """
    Calculates the auto-correlogram for a spike train.

    Args:
        c1_times (NDArray): Spike times (sorted least to greatest)
            in seconds.
        window_size (float): Width of cross correlogram window in seconds.
        bin_width (float): Width of cross correlogram bins in seconds.
        overlap_tol (float): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        corrgram (NDArray): The calculated auto-correlogram.
    """
    return _correlogram(c1_times, c1_times, window_size, bin_width, overlap_tol)


def x_correlogram(
    c1_times: NDArray[np.float_],
    c2_times: NDArray[np.float_],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> NDArray[np.float_]:
    """
    Calculates the cross-correlogram for two spike trains.

    Args:
        c1_times (NDArray[np.float_]): Cluster 1 spike times (sorted least to greatest)
            in seconds.
        c1_times (NDArray[np.float_]): Cluster 2 spike times (sorted least to greatest) '
            in seconds.
        window_size (float): Width of cross correlogram window in seconds.
            Defaults to 1 ms.
        bin_width (float): Width of cross correlogram bins in seconds.
        overlap_tol (float): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        NDArray[np.float_]: The calculated cross-correlogram.
    """

    return _correlogram(c1_times, c2_times, window_size, bin_width, overlap_tol)


def _correlogram(
    c1_times: NDArray[np.float64],
    c2_times: NDArray[np.float64],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> NDArray[np.float64]:
    """
    Calculates the correlogram between two spike trains.

    Args:
        c1_times (NDArray): Spike times in seconds.
        c2_times (NDArray): Spike times in seconds.
        window_size (float, optional): Width of cross correlogram window in seconds.
            Defaults to 100 ms.
        bin_width (float, optional): Width of cross correlogram bins in seconds.
            Defaults to 1 ms.
        overlap_tol (float, optional): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        corrgram (NDArray): The calculated cross-correlogram.
    """
    # Call the cluster with more spikes c1.
    corrgram = np.zeros((math.ceil(window_size / bin_width)))

    c2_start = 0
    if c1_times.shape[0] < c2_times.shape[0]:
        c1_times, c2_times = c2_times, c1_times

    # To calculate the cross-correlogram, we iterate over c1 spikes as reference spikes
    # and count the number of c2 spikes that fall within window_size of the
    # reference spike.
    for ref_spk in range(c1_times.shape[0]):
        while (c2_start < c2_times.shape[0]) and (
            c2_times[c2_start] < (c1_times[ref_spk] - window_size / 2)
        ):
            c2_start += 1  # c2_start tracks the first in-window spike.

        spk_idx = c2_start  # spk_idx iterates over in-window c2 spikes.
        if spk_idx >= c2_times.shape[0]:
            continue

        while (spk_idx < c2_times.shape[0]) and (
            c2_times[spk_idx] < (c1_times[ref_spk] + window_size / 2)
        ):
            if abs(c1_times[ref_spk] - c2_times[spk_idx]) > overlap_tol:
                bin_idx = min(
                    math.floor(
                        (c1_times[ref_spk] - c2_times[spk_idx] + window_size / 2)
                        / bin_width
                    ),
                    corrgram.shape[0] - 1,
                )
                corrgram[bin_idx] += 1
            spk_idx += 1

    return corrgram
