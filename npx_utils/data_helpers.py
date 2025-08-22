import os
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from npx_utils.sglx_helpers import get_bits_to_uV, get_data_memmap, read_meta


def extract_spikes(
    data: NDArray[np.int_],
    times_multi: list[NDArray[np.float64]],
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


def extract_all_spikes(
    data: NDArray[np.int_],
    times_multi: list[NDArray[np.float64]],
    clust_ids: int,
    pre_samples: int,
    post_samples: int,
    max_spikes: int,
):
    spikes = {}
    for clust_id in tqdm(clust_ids, "Extracting spikes..."):
        spikes[clust_id] = extract_spikes(
            data, times_multi, clust_id, pre_samples, post_samples, max_spikes
        )
    return spikes


def calc_mean_wf(
    params: dict[str, Any],
    n_clusters: int,
    cluster_ids: list[int],
    times_multi: dict[NDArray[np.int_]],
    data: NDArray[np.int_],
) -> NDArray:
    """
    Calculate mean waveform and std waveform for each cluster. Need to have loaded some metrics. If return_spikes is True, also returns the spike waveforms.
    Use GPU acceleration with cupy. If the mean waveform is the incorrect shape, it will be recalculated.

    Args:
        params (dict): Parameters for the recording.
        n_clusters (int): Number of clusters in the recording. Equal to the maximum cluster id + 1.
        cluster_ids (list): List of cluster ids to calculate waveforms for.
        times_multi (dict): Dictionary of spike times indexed by cluster id.
        data (NDArray): Ephys data with shape (n_timepoints, n_channels).

    Returns:
        NDArray: Mean waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        NDArray: Std waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        dict[int, NDArray]: Spike waveforms for each cluster (bits). NDArray shape (n_spikes, n_channels, pre_samples + post_samples) dtype int16
    """
    mean_wf_path = os.path.join(params["KS_folder"], "mean_waveforms.npy")

    if os.path.exists(mean_wf_path):
        mean_wf = np.load(mean_wf_path)
        # recalculate mean_wf if it is not the right shape
        if mean_wf.shape[0] == n_clusters:
            if np.any(np.isnan(mean_wf)):
                mean_wf = np.nan_to_num(mean_wf, nan=0)
                # save the fixed mean waveform
                np.save(mean_wf_path, mean_wf)

            return mean_wf

    mean_wf = cp.zeros(
        (
            n_clusters,
            params["n_chan"],
            params["pre_samples"] + params["post_samples"],
        )
    )
    for i in tqdm(cluster_ids, desc="Calculating mean waveforms"):
        spikes = extract_spikes(
            data,
            times_multi,
            i,
            params["pre_samples"],
            params["post_samples"],
            params["max_spikes"],
        )
        if len(spikes) > 0:  # edge case
            spikes_cp = cp.array(spikes, dtype=cp.float32)
            mean_wf[i, :, :] = cp.mean(spikes_cp, axis=0)

    # convert mean_wf uV
    meta = read_meta(params["meta_path"])
    bits_to_uV = get_bits_to_uV(meta)  # convert from bits to uV
    bits_to_uV = cp.float32(bits_to_uV)  # convert to cupy float32
    mean_wf *= bits_to_uV

    tqdm.write("Saving mean waveforms...")
    cp.save(mean_wf_path, mean_wf)

    # Convert back to numpy arrays for compatibility
    mean_wf = cp.asnumpy(mean_wf)

    return mean_wf


def calc_mean_wf_split(
    params: dict[str, Any],
    n_clusters: int,
    cluster_ids: list[int],
    times_multi: dict[NDArray[np.int_]],
    data: NDArray[np.int_],
    n_splits: int = 2,
):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2. Otherwise use calc_mean_wf.")

    mean_wf_path = os.path.join(
        params["KS_folder"], f"mean_waveforms_{n_splits}split.npy"
    )

    if os.path.exists(mean_wf_path):
        try:
            mean_wf = np.load(mean_wf_path)
            # recalculate mean_wf if it is not the right shape
            if mean_wf.shape[0] == n_clusters:
                if np.any(np.isnan(mean_wf)):
                    mean_wf = np.nan_to_num(mean_wf, nan=0)
                    # save the fixed mean waveform
                    np.save(mean_wf_path, mean_wf)
                return mean_wf
        except ValueError:
            pass

    mean_wf = cp.zeros(
        (
            n_clusters,
            params["n_chan"],
            params["pre_samples"] + params["post_samples"],
            n_splits,
        )
    )
    for i in tqdm(cluster_ids, desc="Calculating mean waveforms"):
        spikes = extract_spikes(
            data,
            times_multi,
            i,
            params["pre_samples"],
            params["post_samples"],
            params["max_spikes"],
        )
        if len(spikes) > 0:  # edge case
            spikes_cp = cp.array(spikes, dtype=cp.float32)
            for split in range(n_splits):
                # Split the spikes into n_splits parts
                split_size = spikes_cp.shape[0] // n_splits
                start = split * split_size
                end = (split + 1) * split_size if split != n_splits - 1 else None
                mean_wf[i, :, :, split] = cp.mean(spikes_cp[start:end], axis=0)

    # convert mean_wf uV
    meta = read_meta(params["meta_path"])
    bits_to_uV = get_bits_to_uV(meta)  # convert from bits to uV
    bits_to_uV = cp.float32(bits_to_uV)  # convert to cupy float32
    mean_wf *= bits_to_uV

    tqdm.write("Saving mean waveforms...")
    cp.save(mean_wf_path, mean_wf)

    # Convert back to numpy arrays for compatibility
    mean_wf = cp.asnumpy(mean_wf)

    return mean_wf


def find_times_multi_ks(
    ks_folder: str,
    clust_ids: list[int] = None,
    pre_samples=20,
    post_samples=62,
):
    sp_times = np.load(os.path.join(ks_folder, "spike_times.npy"))
    sp_clust = np.load(os.path.join(ks_folder, "spike_clusters.npy"))
    data = get_data_memmap(ks_folder)
    if clust_ids is None:
        clust_ids = np.arange(np.max(sp_clust) + 1)

    return find_times_multi(
        sp_times, sp_clust, clust_ids, data, pre_samples, post_samples
    )


def find_times_multi(
    sp_times: NDArray[np.float64],
    sp_clust: NDArray[np.int_],
    clust_ids: list[int],
    data: NDArray[np.int_],
    pre_samples: int,
    post_samples: int,
) -> dict[NDArray[np.float64]]:
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
        cl_times (dict): found cluster spike times in samples.
    """
    times_multi = {}

    for i in clust_ids:
        cl_spike_times = sp_times[sp_clust == i]
        cl_spike_times = cl_spike_times[
            (cl_spike_times >= pre_samples)
            & (cl_spike_times < data.shape[0] - post_samples)
        ]
        times_multi[i] = cl_spike_times

    return times_multi
