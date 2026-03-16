import npx_utils as npx
import numpy as np
import scipy.signal


def detect_noise_channels(
    data,
    meta,
    method="rms",
    time_s=20,
    # parameters for rms method
    rms_threshold_uv=40,
    # parameters for rms no spikes method
    rms_no_spikes_threshold_uv=30,
    spike_times=None,
    # parameters for psd coherence method
    psd_hf_threshold=0.2,
    channel_positions=None,
):
    """
    Detects noisy channels in the given data based.

    Returns:
        Channels labels: 0: good,  1: dead low coherence / amplitude, 2: noisy, 3: outside of the brain
    """
    method_list = ["rms", "rms_no_spikes", "psd_coherence"]
    assert (
        method in method_list
    ), f"Method {method} not recognized. Choose from {method_list}"

    # get random subset of data for estimation
    n_samples, n_channels = data.shape
    fs = npx.sglx_helpers.get_sample_rate(meta)
    assert time_s * fs < n_samples, "Data too short for noise estimation"
    # random start in middle 3/4 of data
    start = np.random.randint(n_samples // 4, n_samples // 4 * 3 - time_s * fs)
    data_sub = data[start : int(start + time_s * fs), :]
    data_uv = npx.sglx_helpers.convert_data_to_uV(
        data_sub, np.arange(n_channels - 1), meta
    )

    channel_labels = np.zeros(n_channels, dtype=int)
    if method == "rms":
        rms = np.sqrt(np.mean(data_uv**2, axis=0))
        noisy_channels = np.where(rms > rms_threshold_uv)[0]
        channel_labels[noisy_channels] = 2

    elif method == "rms_no_spikes":
        assert (
            spike_times is not None
        ), "spike_times must be provided for rms_no_spikes method"
        # adjust spike times to subset
        spike_times = (
            spike_times[(spike_times >= start) & (spike_times < start + time_s * fs)]
            - start
        )
        noise_data = extract_noise(
            data_uv, spike_times, pre_samples=20, post_samples=62
        )
        rms = np.sqrt(np.mean(noise_data**2, axis=0))
        noisy_channels = np.where(rms > rms_no_spikes_threshold_uv)[0]
        channel_labels[noisy_channels] = 2
    elif method == "psd_coherence":
        assert (
            channel_positions is not None
        ), "channel_positions must be provided for psd_coherence method"
        # sort channels by x position then y position
        sort_idx = np.lexsort((channel_positions[:, 1], channel_positions[:, 0]))
        data_sorted = data_uv[:, sort_idx]

        channel_labels = psd_coherence_detection(
            data_sorted,
            fs,
            psd_hf_threshold,
        )
        # unsort channel labels
        unsort_idx = np.argsort(sort_idx)
        channel_labels = channel_labels[unsort_idx]
    return channel_labels


def extract_noise(data, times, pre_samples=20, post_samples=62):
    """
    Extract snippets of noise from the data.
    Args:
        data (NDArray): The data to extract noise from.
        times (NDArray): The spike times.
        post_samples (int): The number of samples after the spike time to include.
        pre_samples (int): The number of samples before the spike time to include.
    Returns:
        NDArray: The extracted noise snippets. Shape (max_snippets, n_channels).
    """
    total_samples = len(data)
    signal_mask = np.zeros(total_samples, dtype=bool)
    for time in times:
        start_idx = max(0, time - pre_samples)
        end_idx = min(total_samples, time + post_samples + 1)
        signal_mask[start_idx:end_idx] = True
    noise_indices = np.where(~signal_mask)[0]
    noise_samples = data[noise_indices, :]
    return noise_samples


# taken from spikeinterface
def psd_coherence_detection(
    raw,
    fs,
    psd_hf_threshold,
    dead_channel_thr=-0.5,
    noisy_channel_thr=1.0,
    outside_channel_thr=-0.75,
    n_neighbors=11,
    nyquist_threshold=0.8,
    welch_window_ms=0.3,
    outside_channels_location="top",
):
    """
    Bad channels detection for Neuropixel probes developed by IBL

    Parameters
    ----------
    raw : traces
        (num_samples, n_channels) raw traces
    fs : float
        sampling frequency
    psd_hf_threshold : float
        Threshold for high frequency PSD. If mean PSD above `nyquist_threshold` * fn is greater than this
        value, channels are flagged as noisy (together with channel coherence condition).
    dead_channel_thr : float, default: -0.5
        Threshold for channel coherence below which channels are labeled as dead
    noisy_channel_thr : float, default: 1
        Threshold for channel coherence above which channels are labeled as noisy (together with psd condition)
    outside_channel_thr : float, default: -0.75
        Threshold for channel coherence above which channels
    n_neighbors : int, default: 11
        Number of neighbors to compute median fitler
    nyquist_threshold : float, default: 0.8
        Threshold on Nyquist frequency to calculate HF noise band
    welch_window_ms : float, default: 0.3
        Window size for the scipy.signal.welch that will be converted to nperseg
    outside_channels_location : "top" | "bottom" | "both", default: "top"
        Location of the outside channels. If "top", only the channels at the top of the probe can be
        marked as outside channels. If "bottom", only the channels at the bottom of the probe can be
        marked as outside channels. If "both", both the channels at the top and bottom of the probe can be
        marked as outside channels

    Returns
    -------
    1d array
        Channels labels: 0: good,  1: dead low coherence / amplitude, 2: noisy, 3: outside of the brain
    """
    _, nc = raw.shape
    raw = raw - np.mean(raw, axis=0)[np.newaxis, :]
    nperseg = int(welch_window_ms * fs / 1000)
    fscale, psd = scipy.signal.welch(raw, fs=fs, axis=0, window="hann", nperseg=nperseg)

    # compute similarities
    ref = np.median(raw, axis=1)
    xcorr = np.sum(raw * ref[:, np.newaxis], axis=0) / np.sum(ref**2)

    # compute coherence
    xcorr_neighbors = detrend(xcorr, n_neighbors)
    xcorr_distant = xcorr - detrend(xcorr, n_neighbors) - 1

    # make recommendation
    psd_hf = np.mean(psd[fscale > (fs / 2 * nyquist_threshold), :], axis=0)

    ichannels = np.zeros(nc, dtype=int)
    idead = np.where(xcorr_neighbors < dead_channel_thr)[0]
    inoisy = np.where(
        np.logical_or(psd_hf > psd_hf_threshold, xcorr_neighbors > noisy_channel_thr)
    )[0]

    ichannels[idead] = 1
    ichannels[inoisy] = 2

    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    # the chanels outside need to be at the extreme of the probe
    (ioutside,) = np.where(xcorr_distant < outside_channel_thr)
    a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
    if ioutside.size > 0:
        if outside_channels_location == "top":
            # channels are sorted bottom to top, so the last channel needs to be (nc - 1)
            if ioutside[-1] == (nc - 1):
                ioutside = ioutside[(a == np.max(a)) & (a > 0)]
                ichannels[ioutside] = 3
        elif outside_channels_location == "bottom":
            # outside channels are at the bottom of the probe, so the first channel needs to be 0
            if ioutside[0] == 0:
                ioutside = ioutside[(a == np.min(a)) & (a < np.max(a))]
                ichannels[ioutside] = 3
        else:  # both extremes are considered
            if ioutside[-1] == (nc - 1) or ioutside[0] == 0:
                ioutside = ioutside[(a == np.max(a)) | (a == np.min(a))]
                ichannels[ioutside] = 3

    return ichannels


def detrend(x, nmed):
    """
    Subtract the trend from a vector
    The trend is a median filtered version of the said vector with tapering

    x: input vector
    nmed : number of points of the median filter
    """
    ntap = int(np.ceil(nmed / 2))
    xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]

    xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
    return x - xf
