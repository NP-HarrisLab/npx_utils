import os
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .sglx_helpers import get_bits_to_uV, get_data_memmap


def extract_spikes(
    data: NDArray[np.int_],
    times_multi: list[NDArray[np.float_]],
    clust_id: int,
    pre_samples: int,
    post_samples: int,
    max_spikes: int,
) -> NDArray[np.int_]:
    """
    Extracts spike waveforms for the specified cluster.

    If the cluster contains more than `max_spikes` spikes, `max_spikes` random
    spikes are extracted instead.

    Args:
        data (NDArray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        times_multi (list): Spike times indexed by cluster id.
        clust_id (list): The cluster to extract spikes from
        pre_samples (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        post_samples (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.
        max_spikes (int): The maximum number of spikes to extract. If -1, all
            spikes are extracted. Defaults to -1.

    Returns:
        spikes (NDArray): Array of extracted spike waveforms with shape
            (# of spikes, # of channels, # of timepoints).
    """
    times = times_multi[clust_id].astype("int64")
    # spikes cut off by the ends of the recording is handled in times_multi
    # times = times[(times >= pre_samples) & (times < data.shape[0] - post_samples)]

    # Randomly pick spikes if the cluster has too many
    if (max_spikes != -1) and (times.shape[0] > max_spikes):
        np.random.shuffle(times)
        times = times[:max_spikes]

    # Create an array to store the spikes
    # Extract spike data around each spike time and avoid for loops for speed
    start_times = times - pre_samples
    n_spikes = len(start_times)
    n_channels = data.shape[1]
    n_samples = post_samples + pre_samples

    # Create an array to store the spikes
    spikes = np.empty((n_spikes, n_channels, n_samples), dtype=data.dtype)

    # Use broadcasting to create index arrays for slicing
    row_indices = np.arange(n_samples).reshape(-1, 1) + start_times

    # Extract the spikes using advanced indexing
    spikes = data[row_indices, :].transpose(
        1, 2, 0
    )  # Shape (n_spikes, n_channels, n_samples)

    return spikes


def calc_mean_and_std_wf(
    params: dict[str, Any],
    n_clusters: int,
    cluster_ids: list[int],
    spike_times: list[NDArray[np.int_]],
    data: NDArray[np.int_],
    return_std: bool = True,
    return_spikes: bool = False,
) -> tuple[NDArray, NDArray, dict[int, NDArray]]:
    """
    Calculate mean waveform and std waveform for each cluster. Need to have loaded some metrics. If return_spikes is True, also returns the spike waveforms.
    Use GPU acceleration with cupy. If the mean waveform is the incorrect shape, it will be recalculated.

    Args:
        params (dict): Parameters for the recording.
        n_clusters (int): Number of clusters in the recording. Equal to the maximum cluster id + 1.
        cluster_ids (list): List of cluster ids to calculate waveforms for.
        spike_times (list): List of spike times indexed by cluster id.
        data (NDArray): Ephys data with shape (n_timepoints, n_channels).
        return_std (bool): Whether to return the standard deviation of the waveforms. Defaults to True.
        return_spikes (bool): Whether to return the spike waveforms. Defaults to False.

    Returns:
        NDArray: Mean waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        NDArray: Std waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        dict[int, NDArray]: Spike waveforms for each cluster (bits). NDArray shape (n_spikes, n_channels, pre_samples + post_samples) dtype int16
    """
    mean_wf_path = os.path.join(params["KS_folder"], "mean_waveforms.npy")
    std_wf_path = os.path.join(params["KS_folder"], "std_waveforms.npy")

    spikes = {}
    if os.path.exists(mean_wf_path) and (not return_std or os.path.exists(std_wf_path)):
        mean_wf = np.load(mean_wf_path)
        # recalculate mean_wf if it is not the right shape
        if mean_wf.shape[0] == n_clusters:
            try:
                std_wf = np.load(std_wf_path)
            except FileNotFoundError:
                std_wf = np.array([])
            if np.any(np.isnan(mean_wf)):
                mean_wf = np.nan_to_num(mean_wf, nan=0)
                # save the fixed mean waveform
                np.save(mean_wf_path, mean_wf)

            if return_spikes:
                # Extracting spikes is faster than saving and loading them from file
                for i in tqdm(cluster_ids, desc="Loading spikes"):
                    spikes_i = extract_spikes(
                        data,
                        spike_times,
                        i,
                        params["pre_samples"],
                        params["post_samples"],
                        params["max_spikes"],
                    )
                    spikes[i] = spikes_i
            return mean_wf, std_wf, spikes

    bits_to_uV = get_bits_to_uV(params)  # convert from bits to uV
    bits_to_uV = cp.float32(bits_to_uV)  # convert to cupy float32
    mean_wf = cp.zeros(
        (
            n_clusters,
            params["n_chan"],
            params["pre_samples"] + params["post_samples"],
        )
    )
    std_wf = cp.zeros_like(mean_wf)
    for i in tqdm(cluster_ids, desc="Calculating mean and std waveforms"):
        spikes[i] = extract_spikes(
            data,
            spike_times,
            i,
            params["pre_samples"],
            params["post_samples"],
            params["max_spikes"],
        )
        if len(spikes[i]) > 0:  # edge case
            spikes_cp = cp.array(spikes[i], dtype=cp.float32)
            mean_wf[i, :, :] = cp.mean(spikes_cp, axis=0)
            std_wf[i, :, :] = cp.std(spikes_cp, axis=0)

    # convert mean_wf and std_wf to uV
    mean_wf *= bits_to_uV
    std_wf *= bits_to_uV

    tqdm.write("Saving mean and std waveforms...")
    cp.save(mean_wf_path, mean_wf)
    cp.save(std_wf_path, std_wf)

    # Convert back to numpy arrays for compatibility
    mean_wf = cp.asnumpy(mean_wf)
    std_wf = cp.asnumpy(std_wf)

    return mean_wf, std_wf, spikes


def find_times_multi_ks(
    ks_folder: str,
    clust_ids: list[int],
    pre_samples=20,
    post_samples=62,
):
    sp_times = np.load(os.path.join(ks_folder, "spike_times.npy"))
    sp_clust = np.load(os.path.join(ks_folder, "spike_clusters.npy"))
    data = get_data_memmap(ks_folder)
    return find_times_multi(
        sp_times, sp_clust, clust_ids, data, pre_samples, post_samples
    )


def find_times_multi(
    sp_times: NDArray[np.float_],
    sp_clust: NDArray[np.int_],
    clust_ids: list[int],
    data: NDArray[np.int_],
    pre_samples: int,
    post_samples: int,
) -> list[NDArray[np.float_]]:
    """
    Finds all the spike times for each of the specified clusters.

    Args:
        sp_times (NDArray): Spike times (in any unit of time).
        sp_clust (NDArray): Spike cluster assignments.
        clust_ids (NDArray): Clusters for which spike times should be returned.
        data (NDArray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        pre_samples (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        post_samples (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.

    Returns:
        cl_times (list): found cluster spike times.
    """
    times_multi = []

    for i in clust_ids:
        cl_spike_times = sp_times[sp_clust == i]
        cl_spike_times = cl_spike_times[
            (cl_spike_times >= pre_samples)
            & (cl_spike_times < data.shape[0] - post_samples)
        ]
        times_multi.append(cl_spike_times)

    return times_multi
