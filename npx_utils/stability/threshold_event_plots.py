# -*- coding: utf-8 -*-
# taken from Jennifer Collonel
"""
Created on Sat Mar  1 18:07:21 2025

Should only be run on filtered, background subtracted data (e.g. from CatGT).
Threshold the binary, merge spikes that are sufficiently close in time and space
to count as events. Algorithm adapated from JRClust.

Requires SGLX utilties to read metadata and binary.

readData creates an npy file of spike properties for a selected shank, named
'{binary_name}_dd_sh{shank index}.npy'. The file is saved in the directory with the
Binary. The plotting routines read these files to make the plots.

The columns in the drfit data array are:
    spike times(sec)
    spike z position (from center of mass), in um
    amplitude of negative going peak, uV
    spike x position (from center of mass), in um
    peak channel (of the channels on the shank -- not remapped to original channel in binary)


Plot types:
    plotOne: drift raster plot for a single shank, single recording
    plotMult: plot drift rasters for single shank, multiple recordings (typically, multiple days)
              useful for by eye drift assessment
    plotMultPDF: plot amplitude prob density function across all shanks, multiple recordings
                 (figure 2A of Steinmetz, et al. NP 2.0 paper)
    plotSpikeRate: plot total spike rate across all shanks
                 (figure 2C of Steinmetz, et al. NP 2.0 paper)
    plotRateVsZ: plot relative spike rate vs. z position for a single shank, multiple recordings
                 (figure 2B of Steinmetz, et al. NP 2.0 paper)


"""
import os
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from npx_utils.ks_helpers import get_binary_path, get_ks_folders
from npx_utils.other_helpers import get_probe_folders, get_run_folders
from npx_utils.sglx._SGLXMetaToCoords import MetaToCoords
from npx_utils.sglx.sglx_helpers import (
    ChanGainsIM,
    ChannelCountsIM,
    get_data_memmap,
    get_same_channel_positions,
    get_sample_rate,
    read_meta,
)
from tqdm.autonotebook import tqdm


def getDriftDataPath(binFullPath, sh_index):
    parent_path = binFullPath.parent
    bin_name = binFullPath.stem
    dd_name = f"{bin_name}_dd_sh{sh_index}.npy"
    dd_path = parent_path.joinpath(dd_name)
    return dd_path


def findPeaks(samplesIn, thresh, xc, zc, excl_chan, fs=30000):
    # find peaks with negative amp > threshold, in a batch of
    # samples from all channels
    # range of points for calculateing a full waveform
    n_chan, n_samp = samplesIn.shape
    start_time = np.floor(fs * 1.0 / 1000).astype(int)
    end_time = n_samp - np.floor(fs * 1.0 / 1000).astype(int)

    # threshold should be in bits
    exceeds_thresh = samplesIn < -np.abs(thresh)  # boolean (n_chan, n_samp)

    # find local minima by taking only points whose neighbor points are
    # more positive.
    (peak_chan, peak_ind) = np.where(exceeds_thresh)

    # remove spikes that occur on excluded channels
    for ch in excl_chan:
        rem_ind = np.where(peak_chan == ch)[0]
        if rem_ind.size > 0:
            peak_ind = np.delete(peak_ind, rem_ind)
            peak_chan = np.delete(peak_chan, rem_ind)

    too_early_ind = np.where(peak_ind < start_time)[0]
    if too_early_ind.size > 0:
        peak_ind = np.delete(peak_ind, too_early_ind)
        peak_chan = np.delete(peak_chan, too_early_ind)

    too_late_ind = np.where(peak_ind > end_time)[0]
    if too_late_ind.size > 0:
        peak_ind = np.delete(peak_ind, too_late_ind)
        peak_chan = np.delete(peak_chan, too_late_ind)

    peak_center = samplesIn[peak_chan, peak_ind]  # amplitudes of the peaks
    # compare each peak to neighbors in time, earlier and later, keep only
    # those that are local minima
    loc_min = (samplesIn[peak_chan, peak_ind - 1] > peak_center) & (
        samplesIn[peak_chan, peak_ind + 1] > peak_center
    )
    peak_ind = peak_ind[loc_min]
    peak_chan = peak_chan[loc_min]

    # valid peaks have 3 points in a row below threshold
    pass_inarow = (samplesIn[peak_chan, peak_ind - 1] < thresh) & (
        samplesIn[peak_chan, peak_ind + 1] < thresh
    )
    peak_ind = peak_ind[pass_inarow]
    peak_chan = peak_chan[pass_inarow]
    peak_sig = samplesIn[peak_chan, peak_ind]
    # print(f'before merging: {peak_ind.size}')
    peak_ind, peak_chan, peak_sig = mergePeaks(
        peak_ind, peak_chan, peak_sig, xc, zc, fs
    )
    # print(f'after merging: {peak_ind.size}')
    xz = spike_pos(samplesIn, peak_ind, peak_chan, xc, zc)
    return peak_ind, peak_chan, peak_sig, xz


def mergePeaks(peak_ind, peak_chan, peak_sig, xc, zc, fs=30000):
    # spikes are detected on multiple sites
    # assume that spikes detected within a time threshold
    # and physical radius belong to the same spikeing 'event.'
    # Merge these into one event, calculate peak channel,
    # x and z center of mass based on the negative-going signal

    nLim = np.floor((1 / 1000) * fs).astype(int)  # merge spikes within +/- 1 ms
    neigh_radius_um = 60
    near_sites = calc_neighbor_sites(xc, zc, neigh_radius_um)

    # sort all spikes in time
    sort_order = np.argsort(peak_ind)
    peak_ind = peak_ind[sort_order]
    peak_chan = peak_chan[sort_order]
    peak_sig = peak_sig[sort_order]

    chan_set = np.unique(peak_chan)
    num_chan = chan_set.size

    # remove spikes that are within 1 ms in each channels spike train
    for i_chan in chan_set:
        curr_ind = np.where(peak_chan == i_chan)[0]
        curr_times = peak_ind[curr_ind]
        curr_amp = peak_sig[curr_ind]
        spikes_to_check = np.where(np.diff(curr_times) < nLim)[0]
        amp_early = curr_amp[spikes_to_check]
        amp_late = curr_amp[spikes_to_check + 1]
        keep_early = (amp_early < amp_late).astype(
            int
        )  # looking for the more negative spike
        ind_to_remove = curr_ind[spikes_to_check + keep_early]
        peak_ind = np.delete(peak_ind, ind_to_remove)
        peak_chan = np.delete(peak_chan, ind_to_remove)
        peak_sig = np.delete(peak_sig, ind_to_remove)

    for i_chan in chan_set:
        neigh_chan = near_sites[i_chan]
        # remove current channel
        neigh_chan = neigh_chan[neigh_chan != i_chan]

        for j_chan in neigh_chan:

            i_ind = np.where(peak_chan == i_chan)[0]
            j_ind = np.where(peak_chan == j_chan)[0]

            orig_ind = np.concatenate((i_ind, j_ind))
            ij_labels = np.concatenate(
                (np.zeros((i_ind.size,), dtype=int), np.ones((j_ind.size,), dtype=int))
            )
            ij_times = np.concatenate((peak_ind[i_ind], peak_ind[j_ind]))
            ij_amps = np.concatenate((peak_sig[i_ind], peak_sig[j_ind]))

            order = np.argsort(ij_times)
            # reorder everything
            orig_ind = orig_ind[order]
            ij_labels = ij_labels[order]
            ij_times = ij_times[order]
            ij_amps = ij_amps[order]

            spikes_to_check = np.where(np.diff(ij_times) < nLim)[0]
            amp_early = ij_amps[spikes_to_check]
            amp_late = ij_amps[spikes_to_check + 1]
            keep_early = (amp_early < amp_late).astype(
                int
            )  # looking for the more negative spike
            relative_ind_to_remove = spikes_to_check + keep_early
            ind_to_remove = orig_ind[relative_ind_to_remove]
            peak_ind = np.delete(peak_ind, ind_to_remove)
            peak_chan = np.delete(peak_chan, ind_to_remove)
            peak_sig = np.delete(peak_sig, ind_to_remove)

    return peak_ind, peak_chan, peak_sig


def spike_pos(samplesIn, peak_ind, peak_chan, xc, zc):
    xz = np.zeros((peak_ind.size, 2))
    near_sites = calc_neighbor_sites(xc, zc, neigh_radius_um=60)
    chan_set = np.unique(peak_chan)
    nSamp = 15  # before and after peak, 31 total ~  1 msec

    # calculate these channel-wise because those use the same set of
    # neighbor channels
    for i_chan in chan_set:
        i_ind = np.where(peak_chan == i_chan)[0]
        cn = near_sites[i_chan]
        for ci in i_ind:
            # get section of data
            ct = peak_ind[ci]
            curr_dat = samplesIn[cn, ct - nSamp : ct + nSamp]
            amps = np.abs(np.squeeze(np.min(curr_dat, axis=1)))
            norm = np.sum(amps)
            cm_x = np.sum(np.multiply(xc[cn], amps)) / norm
            cm_z = np.sum(np.multiply(zc[cn], amps)) / norm
            xz[ci] = [cm_x, cm_z]

    return xz


def calc_neighbor_sites(xc, zc, neigh_radius_um):
    # return a list of arrays of site indicies within site_radius_um
    n_site = xc.size
    near_sites = list()
    rad_sq = neigh_radius_um * neigh_radius_um
    for i in range(n_site):
        dist = np.square(xc - xc[i]) + np.square(zc - zc[i])
        neigh = np.where(dist < rad_sq)[0]
        near_sites.append(neigh)
    return near_sites


def plotOne(drift_data):
    fig, ax = plt.subplots(figsize=(6, 2))
    n_spike = drift_data.shape[0]
    skip_step = np.floor(n_spike / 50000) + 1
    plot_spikes = np.arange(0, n_spike, skip_step).astype(int)
    pd = drift_data[plot_spikes]

    c_lim = np.asarray([np.quantile(pd[:, 2], 0.1), np.quantile(pd[:, 2], 0.9)])
    even_divisor = 50
    c_lim = even_divisor * np.floor(c_lim / even_divisor)
    plt.scatter(
        pd[:, 0],
        pd[:, 1],
        s=0.1,
        c=pd[:, 2],
        cmap="plasma",
        vmin=c_lim[0],
        vmax=c_lim[1],
    )
    c = plt.colorbar()
    return fig


def plotMult(bin_list, drift_list, day_list, sh_list):
    """
    Plots raster plot of spikes for each shank across recording sessions to give estimate of drift.
    """
    # build a large sampled array from the n_sets, to look for 'obvious' drift
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(0.25 + 1 * len(bin_list), 8),
        sharex=True,
        sharey=False,
    )
    fig.suptitle("Spike Drift Across Shanks", fontsize=16)
    max_spike = 50000  # total spikes in the output plot
    even_divisor = 10  # color and z limits will be integer multiples of this value

    # get global limits for color bar
    # get duration of a recording
    all_amplitudes = []
    session_durations = [[] for _ in bin_list]
    for sh_ind in sh_list:
        for n, curr_bin in enumerate(bin_list):
            curr_path = getDriftDataPath(curr_bin, sh_ind)
            try:
                data = np.load(curr_path)
                if data.size > 0:
                    session_durations[n].append(np.max(data[:, 0]))
                    all_amplitudes.append(
                        data[:, 2]
                    )  # Append only the amplitude column
            except FileNotFoundError:
                print(f"Warning: File not found for shank {sh_ind}, bin {curr_bin}")
    global_set_dur = np.array(
        [np.max(durs) if durs else 0 for durs in session_durations]
    )
    global_segment_endpoints = np.cumsum(global_set_dur)

    if all_amplitudes:
        all_amplitudes = np.concatenate(all_amplitudes)
        global_c_lim = np.asarray(
            [np.quantile(all_amplitudes, 0.05), np.quantile(all_amplitudes, 0.95)]
        )
        global_c_lim = even_divisor * np.floor(global_c_lim / even_divisor)
    else:
        global_c_lim = [0, 100]

    scatter = None
    sh_inds = [0, 1, 2, 3]
    for sh_ind in sh_inds:
        row = sh_ind // 2
        col = sh_ind % 2
        current_ax = ax[row, col]
        current_ax.set_title(f"Shank {sh_ind}", fontsize=14)
        if sh_ind not in sh_list:
            continue

        # load data
        n_set = len(bin_list)
        total_spike = 0
        set_dur = np.zeros((n_set,))
        dd_list = list()
        for n, curr_bin in enumerate(bin_list):
            curr_path = getDriftDataPath(curr_bin, sh_ind)
            dd_list.append(np.load(curr_path))
            n_spike, n_meas = dd_list[n].shape
            total_spike = total_spike + n_spike
            if total_spike == 0:
                continue
            set_dur[n] = np.max(dd_list[n][:, 0])
        skip_step = np.floor(total_spike / max_spike) + 1

        if total_spike == 0:
            continue

        # loop over the arrays, buid sampled array that covers all the datasets
        samp_spike = np.zeros((max_spike, n_meas))
        n_samp = 0
        boundaries = np.concatenate(([0], global_segment_endpoints[:-1]))
        for n, dd in enumerate(dd_list):
            curr_nspike = dd.shape[0]
            curr_ind = np.arange(0, curr_nspike, skip_step).astype(int)
            curr_samp = dd[curr_ind, :]
            curr_samp[:, 0] = curr_samp[:, 0] + boundaries[n]
            curr_samp[:, 1] = curr_samp[:, 1] + np.sum(drift_list[0 : n + 1])
            n_curr = curr_ind.size
            samp_spike[n_samp : n_samp + n_curr, :] = curr_samp
            n_samp = n_samp + n_curr

        scatter = current_ax.scatter(
            samp_spike[:, 0],
            samp_spike[:, 1],
            s=0.1,
            c=samp_spike[:, 2],
            cmap="plasma",
            vmin=global_c_lim[0],
            vmax=global_c_lim[1],
        )

    for single_ax in ax.flat:
        single_ax.tick_params(axis="both", labelsize=12)
        ymin, ymax = single_ax.get_ylim()
        single_ax.vlines(
            global_segment_endpoints[:-1],
            ymin=ymin,
            ymax=ymax,
            color="black",
            linestyle="solid",
            alpha=0.6,
            zorder=5,  # Draw lines behind data points but in front of the grid
        )
        if global_segment_endpoints.size > 0:
            single_ax.set_xlim(0, global_segment_endpoints[-1])
    bottom_ax = ax[1, 0]
    all_boundaries = np.concatenate(([0], global_segment_endpoints))
    label_positions = (all_boundaries[:-1] + all_boundaries[1:]) / 2
    labels = [f"{d}" for d in day_list]
    bottom_ax.set_xticks(label_positions)
    bottom_ax.set_xticklabels(labels)
    # Add a single X-axis label for the whole figure
    fig.supxlabel("Recording Day", y=0.02, fontsize=14)
    fig.supylabel("Distance from tip (µm)", x=0.08, fontsize=14)

    cbar = fig.colorbar(scatter, ax=ax.ravel().tolist(), pad=0.01, aspect=40)
    cbar.set_label("Amplitude (µV)", rotation=270, labelpad=15, fontsize=14)

    return fig


def readData(
    binFullPath, selected_sh, time_sec=300, excl_chan=[127], thresh=-80, overwrite=False
):
    parent_path = binFullPath.parent
    dd_path = getDriftDataPath(binFullPath, selected_sh)
    save_path = parent_path.joinpath(dd_path)
    if save_path.exists() and not overwrite:
        return
    # Read in metadata; returns a dictionary with string for values
    meta_path = str(binFullPath).replace(".bin", ".meta")
    meta = read_meta(meta_path)

    # plan to detect peaks in the last 5 minutes of recording
    sRate = get_sample_rate(meta)
    n_ap, n_lf, n_sync = ChannelCountsIM(meta)
    APChan0_to_uV = ChanGainsIM(meta)[2]
    thresh_bits = thresh / APChan0_to_uV
    x_coord, z_coord, sh_ind, connected, n_chan_tot = MetaToCoords(
        binFullPath.with_suffix(".meta"), -1
    )
    rawData = get_data_memmap(binFullPath, meta)
    # transpose
    rawData = rawData.T
    nChan, nFileSamp = rawData.shape

    start_samp = np.floor(nFileSamp - time_sec * sRate).astype(int)
    if start_samp < 0:
        start_samp = 0
    batch_samp = np.floor(2 * sRate).astype(int)
    n_batch = np.floor((nFileSamp - start_samp) / batch_samp).astype(int)
    sel_sh_ind = np.where(sh_ind == selected_sh)[0]
    # translate excluded channels for this shank
    ex_sh = list()
    for ch in excl_chan:
        if np.sum(sel_sh_ind == ch) > 0:
            ex_sh.append(np.where(sel_sh_ind == ch)[0][0])

    x_sh = x_coord[sel_sh_ind]
    z_sh = z_coord[sel_sh_ind]

    if n_batch == 0:
        print("Warning: not enough data to process")
    for j in tqdm(
        range(n_batch),
        desc=f"\tExtracting threshold events: sh {selected_sh}",
        leave=False,
    ):
        st = start_samp + j * batch_samp
        cb = rawData[sel_sh_ind, st : st + batch_samp]
        peak_ind, peak_chan, peak_sig, xz = findPeaks(
            cb, thresh_bits, x_sh, z_sh, ex_sh, fs=sRate
        )
        # add offset to peak_ind, concatenate onto set
        offset = st - start_samp
        peak_ind = (peak_ind + offset) / sRate  # convert the times to sec
        peak_sig = abs(peak_sig * APChan0_to_uV)
        if j == 0:
            all_spikes = np.vstack(
                (peak_ind, xz[:, 1].T, peak_sig, xz[:, 0].T, peak_chan)
            ).T
        else:
            curr_spikes = np.vstack(
                (peak_ind, xz[:, 1].T, peak_sig, xz[:, 0].T, peak_chan)
            ).T
            all_spikes = np.concatenate((all_spikes, curr_spikes))

    np.save(save_path, all_spikes)


def calc_pdf(bin_path, sh_list):
    # calculate prob density function across all shanks
    for j, sh_ind in enumerate(sh_list):
        curr_path = getDriftDataPath(bin_path, sh_ind)
        curr_dat = np.load(curr_path)
        if len(curr_dat) == 0:
            sum_hist = None
            continue
        if j == 0:
            # calculate bin width and bin edgeds
            amp_sort = np.sort(curr_dat[:, 2])
            # if no data just do zeros
            bin_width = np.unique(np.diff(amp_sort))[1]
            bin_edges = (0.5 * bin_width) + np.arange(0, 1000, bin_width)
            sum_hist = np.histogram(curr_dat[:, 2], bin_edges)[0]
        else:
            if sum_hist is None:
                amp_sort = np.sort(curr_dat[:, 2])
                # if no data just do zeros
                bin_width = np.unique(np.diff(amp_sort))[1]
                bin_edges = (0.5 * bin_width) + np.arange(0, 1000, bin_width)
                sum_hist = np.histogram(curr_dat[:, 2], bin_edges)[0]
            else:
                sum_hist = sum_hist + np.histogram(curr_dat[:, 2], bin_edges)[0]
    # convert to pdf by normalizing
    npts = len(sum_hist)
    pdf = np.zeros((npts, 2))
    pdf[:, 0] = bin_edges[0:npts]
    pdf[:, 1] = sum_hist / (np.sum(sum_hist) * bin_width)
    # plt.plot(pdf[:,0],pdf[:,1])
    return pdf


def calcRate(bin_path, sh_list):
    # calculate total spike rate for the probe
    for j, sh_ind in enumerate(sh_list):
        curr_path = getDriftDataPath(bin_path, sh_ind)
        curr_dat = np.load(curr_path)
        if len(curr_dat) == 0:
            min_time = None
            max_time = None
            total_count = 0
            continue
        if j == 0:
            # calculate bin width and bin edge
            min_time = np.min(curr_dat[:, 0])
            max_time = np.max(curr_dat[:, 0])
            total_count = curr_dat.shape[0]
        else:
            if min_time is None:
                min_time = np.min(curr_dat[:, 0])
            else:
                min_time = np.min([min_time, np.min(curr_dat[:, 0])])
            if max_time is None:
                max_time = np.max(curr_dat[:, 0])
            else:
                max_time = np.max([max_time, np.max(curr_dat[:, 0])])
            total_count = total_count + curr_dat.shape[0]
    # convert rate
    spike_rate = total_count / (max_time - min_time)

    return spike_rate


def adjust_figure_for_legend(figure, legend, bottom_margin=0.05):
    """
    Adjust the figure height to make sure the legend fits.

    Args:
        figure (matplotlib.figure.Figure): The figure object.
        legend (matplotlib.legend.Legend): The legend object.
        bottom_margin (float): The desired margin below the legend in inches.
    """
    # Draw the canvas to get the final rendered size of the legend
    figure.canvas.draw()

    # Get the bounding box of the legend in pixels
    legend_bbox = legend.get_window_extent()

    # If the bottom of the legend is below the figure (y=0)
    if legend_bbox.y0 < 0:
        # Calculate the overflow in pixels
        overflow_pixels = -legend_bbox.y0

        # Get the figure's DPI (dots per inch)
        dpi = figure.get_dpi()

        # Calculate the required additional height in inches
        margin_pixels = bottom_margin * dpi
        required_height_increase = (overflow_pixels + margin_pixels) / dpi

        # Get the current figure size in inches
        current_width, current_height = figure.get_size_inches()

        # Set the new figure size
        figure.set_size_inches(current_width, current_height + required_height_increase)

        # Optional: Redraw the figure to apply changes
        figure.canvas.draw()


def plotMultPDF(bin_list, sh_list, day_list):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_pdf = len(bin_list)

    cmap = mpl.colormaps["winter"]
    colors = cmap(np.linspace(0, 1, n_pdf))

    for j in range(n_pdf):
        curr_pdf = calc_pdf(bin_list[j], sh_list)
        ax.plot(curr_pdf[:, 0], curr_pdf[:, 1], color=colors[j], label=f"{day_list[j]}")

    ax.set_xlim(0, 500)
    ax.tick_params(axis="both", labelsize=12)

    plt.xlabel("Spike Amplitude (µV)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.set_title("Spike Amplitude Distribution Over Time", fontsize=16, pad=10)

    # Add a legend to identify the lines
    legend = ax.legend(title="Recording Day", fontsize=10)

    # Remove top and right plot borders for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    adjust_figure_for_legend(fig, legend)
    fig.tight_layout()
    return fig


def plotSpikeRate(bin_list, day_list, sh_list):
    n_meas = len(bin_list)
    spike_rates = np.zeros((n_meas,))

    for j in range(n_meas):
        spike_rates[j] = calcRate(bin_list[j], sh_list)
    trend_x = np.asarray([min(day_list), max(day_list)])
    trend_fit = np.poly1d(np.polyfit(np.asarray(day_list), spike_rates, 1))
    trend_y = trend_fit(trend_x)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.scatter(day_list, spike_rates, s=3)
    plt.plot(trend_x, trend_y, marker=None, linewidth=1, linestyle="dashed")
    ax.tick_params(axis="both", labelsize=12)
    plt.xlabel("Recording Day", fontsize=14)
    plt.ylabel("Spike Rate (Hz)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def calcFiringVsZ(binFullPath, selected_sh, bin_width, z_edges):
    # plot the relative firing rate vs. channel
    curr_path = getDriftDataPath(binFullPath, selected_sh)
    curr_dat = np.load(curr_path)

    if len(curr_dat) == 0:
        return None, None

    if z_edges is None:
        min_z = bin_width * np.floor(np.min(curr_dat[:, 1]) / bin_width)
        max_z = bin_width * np.ceil(np.max(curr_dat[:, 1]) / bin_width)
        z_edges = np.arange(min_z, max_z, bin_width)

    z_hist = (np.histogram(curr_dat[:, 1], z_edges)[0]).astype("float64")
    z_rel = z_hist / np.sum(z_hist)

    return z_rel, z_edges


def plotRateVsZ(bin_list, day_list, sh_list):
    bin_width = 15
    n_meas = len(bin_list)

    # Pre-calc for scaling
    global_max_rate = 0
    z_edges = None
    for sh_ind in sh_list:
        for b in bin_list:
            rates, z_edges = calcFiringVsZ(b, sh_ind, bin_width, z_edges)
            global_max_rate = max(global_max_rate, np.max(rates))
    n_bin = len(z_edges) - 1
    bin_center = z_edges[0:n_bin] + bin_width / 2

    c_lim = [0, global_max_rate]
    c_range = c_lim[1] - c_lim[0]

    # Setup plotting
    original_cmap = mpl.colormaps["plasma"]
    cmap_colors = original_cmap(np.linspace(0, 1, 256))
    cmap_colors[0] = (1, 1, 1, 1)  # RGBA for white
    custom_cmap = ListedColormap(cmap_colors)
    even_divisor = 10  # z limits will be integer multiples of this value

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(0.5 + 1 * len(bin_list), 8),
        sharex=True,
        sharey=False,
    )
    fig.suptitle("Firing Rate vs Depth Across Days", fontsize=18)

    for sh_ind in [0, 1, 2, 3]:
        row = sh_ind // 2
        col = sh_ind % 2
        ax = axes[row, col]
        ax.set_title(f"Shank {sh_ind}", fontsize=14)
        if sh_ind not in sh_list:
            continue

        rel0, z_edges = calcFiringVsZ(bin_list[0], sh_ind, bin_width, z_edges)

        rel_rates = np.zeros((n_meas * n_bin,))
        x_vals = np.zeros((n_meas * n_bin))
        z_vals = np.zeros((n_meas * n_bin))

        rel_rates[0:n_bin] = rel0
        z_vals[0:n_bin] = bin_center

        for i in range(1, n_meas):
            x_vals[i * n_bin : (i + 1) * n_bin] = i
            z_vals[i * n_bin : (i + 1) * n_bin] = bin_center
            rel_rates[i * n_bin : (i + 1) * n_bin] = calcFiringVsZ(
                bin_list[i], sh_ind, bin_width, z_edges
            )[0]

        min_z = even_divisor * np.floor(np.min(z_vals) / even_divisor)
        max_z = even_divisor * np.floor(np.max(z_vals) / even_divisor)
        ax.set_ylim([min_z - 15, max_z + 15])

        # these points get covered by the patches; coloring with rel_rates
        # creates teh correct colorbar
        scatter = ax.scatter(
            x_vals,
            z_vals,
            c=rel_rates,
            s=2,
            marker="s",
            cmap=custom_cmap,
            vmin=c_lim[0],
            vmax=c_lim[1],
        )

        # Add rectangles
        width = 0.75  # in 'recording day' units
        height = 15  # in 'um'

        for i in range(len(x_vals)):
            color_val = (rel_rates[i] - c_lim[0]) / c_range
            ax.add_patch(
                Rectangle(
                    xy=(x_vals[i] - width / 2, z_vals[i] - height / 2),
                    width=width,
                    height=height,
                    edgecolor="None",
                    facecolor=custom_cmap(color_val),
                )
            )

    xt_range = np.arange(n_meas)
    xt_labels = np.asarray(day_list).astype("str")
    bottom_ax = axes[1, 0]
    bottom_ax.set_xlim([-0.5, n_meas - 0.5])
    bottom_ax.set_xticks(xt_range)
    bottom_ax.set_xticklabels(xt_labels)
    fig.supxlabel("Recording Day", y=0.02, fontsize=14)
    fig.supylabel("Distance from tip (µm)", x=0.08, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), pad=0.01, aspect=40)
    cbar.set_label("Spiking Rate (Hz)", rotation=270, labelpad=15, fontsize=14)
    return fig


def get_available_shanks(binFullPath):
    binFullPath = Path(binFullPath)
    meta_path = binFullPath.with_suffix(".meta")
    if not meta_path.exists():
        raise Warning("Metadata file not found")
    _, _, sh_ind, connected, _ = MetaToCoords(meta_path, -1)
    all_shanks = np.unique(sh_ind)
    active_shanks = []
    for shank_idx in all_shanks:
        channels_on_shank = sh_ind == shank_idx
        if np.any(connected[channels_on_shank]):
            active_shanks.append(int(shank_idx))
    return sorted(active_shanks)


def plot_threshold_events(
    subject_folder,
    implant_day=None,
    recordings=None,
    probe_ids=None,
    overwrite=False,
    save_dir=None,
    save_type="png",
    ks_version="4",
    analysis_time_sec=300,
    excl_chan=[127],
    threshold=-80,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    # only affects plotting in plotMult raster plots
    run_folders = get_run_folders(subject_folder, day_folders=recordings)
    drift_list = np.zeros(len(run_folders) - 1)
    # group folders by probes
    ks_folders = []
    for folder in run_folders:
        ks_folders.extend(get_ks_folders(folder, ks_version))

    all_probe_folders = get_probe_folders(ks_folders)
    if probe_ids is not None:
        all_probe_folders = {
            probe_id: all_probe_folders[probe_id]
            for probe_id in probe_ids
            if probe_id in all_probe_folders
        }

    for probe_num in tqdm(
        all_probe_folders, "Processing threshold_events...", position=0, unit="probe"
    ):
        all_probe_ks_folders = all_probe_folders[probe_num]
        probe_ks_folders = get_same_channel_positions(all_probe_ks_folders)
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

        probe_bins = [
            Path(get_binary_path(ks_folder)) for ks_folder in probe_ks_folders
        ]
        sh_list = get_available_shanks(probe_bins[0])

        for binFullPath in tqdm(
            probe_bins, desc="\tProcessing recordings", leave=True, position=1
        ):
            for sh_ind in sh_list:
                readData(
                    binFullPath,
                    sh_ind,
                    analysis_time_sec,
                    excl_chan,
                    threshold,
                    overwrite,
                )
        plt_fig1 = True
        plt_fig2 = True
        plt_fig3 = True
        plt_fig4 = True

        if save_dir is not None and not overwrite:
            fname1 = os.path.join(
                save_dir, f"multi_shank_drift_imec{probe_num}.{save_type}"
            )
            if os.path.exists(fname1):
                # load saved drift data and plot
                fig1 = plt.figure(figsize=(8, 5))
                img = plt.imread(fname1)
                plt.imshow(img)
                plt.axis("off")
                plt_fig1 = False
            fname2 = os.path.join(
                save_dir, f"multi_spike_rate_depth_imec{probe_num}.{save_type}"
            )
            if os.path.exists(fname2):
                fig2 = plt.figure(figsize=(7, 5))
                img = plt.imread(fname2)
                plt.imshow(img)
                plt.axis("off")
                plt_fig2 = False
            fname3 = os.path.join(
                save_dir, f"multi_spike_amplitude_imec{probe_num}.{save_type}"
            )
            if os.path.exists(fname3):
                fig3 = plt.figure(figsize=(8, 5))
                img = plt.imread(fname3)
                plt.imshow(img)
                plt.axis("off")
                plt_fig3 = False
            fname4 = os.path.join(
                save_dir, f"multi_spike_rate_imec{probe_num}.{save_type}"
            )
            if os.path.exists(fname4):
                fig4 = plt.figure(figsize=(7, 5))
                img = plt.imread(fname4)
                plt.imshow(img)
                plt.axis("off")
                plt_fig4 = False

        if plt_fig1:
            if len(probe_bins) == 1:
                drift_path = getDriftDataPath(
                    probe_bins[0], sh_list[0]
                )  # TODO update function
                drift_data = np.load(probe_bins[0].parent.joinpath(drift_path))
                fig1 = plotOne(drift_data)
            else:
                fig1 = plotMult(probe_bins, drift_list, day_list, sh_list)
            if save_dir is not None:
                fname1 = os.path.join(
                    save_dir, f"multi_shank_drift_imec{probe_num}.{save_type}"
                )
                fig1.savefig(fname1, dpi=300, format=save_type, bbox_inches="tight")

        if plt_fig2:
            fig2 = plotRateVsZ(probe_bins, day_list, sh_list)
            if save_dir is not None:
                fname2 = os.path.join(
                    save_dir, f"multi_spike_rate_depth_imec{probe_num}.{save_type}"
                )
                fig2.savefig(fname2, dpi=300, format=save_type, bbox_inches="tight")

        if plt_fig3:
            fig3 = plotMultPDF(probe_bins, sh_list, day_list)
            fname3 = os.path.join(
                save_dir, f"multi_spike_amplitude_imec{probe_num}.{save_type}"
            )
            if save_dir is not None:
                fig3.savefig(fname3, dpi=300, format=save_type, bbox_inches="tight")

        if plt_fig4:
            fig4 = plotSpikeRate(probe_bins, day_list, sh_list)
            if save_dir is not None:
                fname4 = os.path.join(
                    save_dir, f"multi_spike_rate_imec{probe_num}.{save_type}"
                )
                fig4.savefig(fname4, dpi=300, format=save_type, bbox_inches="tight")
