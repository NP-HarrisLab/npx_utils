import os
import re
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import npx_utils as npx
import numpy as np
from scipy.signal import welch


def _my_time_power_spectrum_mat(x, fs):
    """
        Helper function to calculate the power spectrum using Welch's method.
    \ """
    L = x.shape[0]
    # get the next power of 2
    nfft = 1 << (L - 1).bit_length()
    # use 50% overlap - default in MATLAB welch
    nperseg = int(np.trunc(L / 4.5))
    noverlap = int(np.trunc(nperseg / 2))
    f, Pxx = welch(x, nperseg=nperseg, noverlap=noverlap, fs=fs, nfft=nfft, axis=0)
    return Pxx, f, nfft


def _analyze_channel_spectra(
    all_power_est,
    i,
    int_wind,
    meas_chan,
    skip_chan,
    lfp_fs,
    nfft,
    noise_range,
    hz_span,
    back_skip_peak=1,
):
    """
    Analyzes power spectra for multiple channels to estimate signal and noise.

    This is the Python equivalent of the 'if measChan > 3' block.

    Args:
        all_power_est (np.ndarray): 2D array of power spectra (channels x frequency bins).
        i (np.ndarray): 1D array with the index of the power peak for each channel.
        int_wind (int): Integration window size in bins.
        meas_chan (int): Total number of measured channels.
        skip_chan (list): List of channels to exclude from analysis.
        lfp_fs (float): Sampling frequency.
        NFFT (int): NFFT value from the Welch calculation.
        noise_range (tuple): (start_hz, end_hz) for noise calculation.
        hz_span (float): Frequency resolution (Hz per bin).
        back_skip_peak (int): Flag to skip the peak when calculating noise (1=skip).

    Returns:
        tuple: (peak_to_peak_est, back_integrated, chan_label)
    """
    # Get list of channels to analyze
    chan_label = np.delete(np.arange(meas_chan), skip_chan)
    n_valid_chans = len(chan_label)

    peak_to_peak_est = np.zeros(n_valid_chans, dtype=np.float32)
    back_integrated = np.zeros(n_valid_chans, dtype=np.float32)

    n_psd_bin = all_power_est.shape[1]

    # --- Loop over each channel to perform the analysis ---
    for j, curr_chan in enumerate(chan_label):
        peak_idx = i[curr_chan]  # Peak index for the current channel

        # 1. ESTIMATE BACKGROUND POWER AROUND THE PEAK
        # Window before the peak
        b_start1 = max(0, peak_idx - 2 * int_wind)
        b_end1 = min(peak_idx - int_wind, n_psd_bin)
        min_peak_start = b_start1
        back1 = all_power_est[curr_chan, b_start1:b_end1]

        # Window after the peak
        b_start2 = max(0, peak_idx + int_wind)
        b_end2 = min(n_psd_bin, peak_idx + 2 * int_wind)
        max_peak_end = b_end2
        back2 = all_power_est[curr_chan, b_start2:b_end2]

        # Combine and average to get the background estimate
        back_both = np.concatenate((back1, back2))
        back_est = np.mean(back_both)

        # 2. INTEGRATE THE PEAK POWER (after subtracting background)
        b_start_peak = max(0, peak_idx - int_wind)
        b_end_peak = min(n_psd_bin, peak_idx + int_wind)

        # Sum the power at the peak, subtracting the background from each bin
        peak_int = np.sum(all_power_est[curr_chan, b_start_peak:b_end_peak] - back_est)

        # 3. CONVERT PEAK POWER TO PEAK-TO-PEAK VOLTAGE
        if peak_int > 0:
            peak_to_peak_est[j] = 2 * np.sqrt(2 * peak_int / ((1 / lfp_fs) * nfft))

        # 4. INTEGRATE NOISE POWER over the specified noise range
        b_start_noise = int(round(noise_range[0] / hz_span))
        b_end_noise = int(round(noise_range[1] / hz_span))
        back_sum = 0

        # Sum power in the noise band but excludes the signal peak
        if back_skip_peak == 1:
            if b_start_noise < min_peak_start and b_end_noise > max_peak_end:
                sum1 = np.sum(all_power_est[curr_chan, b_start_noise:min_peak_start])
                sum2 = np.sum(all_power_est[curr_chan, max_peak_end:b_end_noise])
                back_sum = sum1 + sum2
            elif (b_start_noise < min_peak_start and b_end_noise <= max_peak_end) or (
                b_start_noise >= min_peak_start and b_end_noise > max_peak_end
            ):
                back_sum = np.sum(all_power_est[curr_chan, b_start_noise:b_end_noise])
            elif b_start_noise < min_peak_start and b_end_noise <= max_peak_end:
                back_sum = np.sum(
                    all_power_est[curr_chan, b_start_noise:min_peak_start]
                )
            elif b_start_noise >= min_peak_start and b_end_noise > max_peak_end:
                back_sum = np.sum(all_power_est[curr_chan, max_peak_end])
            else:
                back_sum = 0
        else:
            back_sum = np.sum(all_power_est[curr_chan, b_start_noise:b_end_noise])

        # 5. CONVERT NOISE POWER TO RMS VOLTAGE
        if back_sum > 0:
            back_integrated[j] = np.sqrt(back_sum / ((1 / lfp_fs) * nfft))

    return peak_to_peak_est, back_integrated, chan_label


def _lfp_band_power(
    lfp_filename,
    lfp_fs,
    lfp_tomv,
    peak_pos_hz,
    n_chans_in_file,
    meas_chan,
    freq_band,
    n_clips,
    clip_dur,
    skip_chan,
    noise_range,
    t_start,
):
    """
    Computes the power in particular bands and across all frequencies.
    """
    if freq_band and not isinstance(freq_band, list):
        freq_band = [freq_band]
    nf = len(freq_band) if freq_band else 0

    # Get the number of samples from the file size
    file_bytes = os.path.getsize(lfp_filename)
    n_samps = file_bytes // (2 * n_chans_in_file)

    # For survey data, calculate start times for clips
    sec_starts = t_start + np.arange(n_clips) * (clip_dur + 1)
    samp_starts = np.round(sec_starts * lfp_fs).astype(int)
    n_clip_samps = int(round(lfp_fs * clip_dur))

    mmf = np.memmap(
        lfp_filename,
        dtype="int16",
        mode="r",
        shape=(n_chans_in_file, n_samps),
        order="F",
    )

    all_power_est_by_band = np.zeros((n_clips, n_chans_in_file, nf))

    # Process each clip
    for n in range(n_clips):
        start_samp = samp_starts[n]
        end_samp = start_samp + n_clip_samps

        # Pull out the data, convert to double, and apply scaling factor
        this_dat = lfp_tomv * mmf[:, start_samp:end_samp].astype(np.float64)

        # Subtract the mean of each channel (row)
        this_dat = this_dat - np.mean(this_dat, axis=1, keepdims=True)

        Pxx, f, nfft = _my_time_power_spectrum_mat(this_dat.T, lfp_fs)

        if n == 0:
            all_power_est = np.zeros((n_clips, Pxx.shape[0], Pxx.shape[1]))

        all_power_est[n, :, :] = Pxx

        for f_idx, band in enumerate(freq_band):
            incl_f = (f > band[0]) & (f <= band[1])
            all_power_est_by_band[n, :, f_idx] = np.mean(Pxx[incl_f, :], axis=0)

    if nf > 0:
        lfp_by_channel = np.mean(all_power_est_by_band, axis=0)
    else:
        lfp_by_channel = np.array([])

    all_power_var = np.var(all_power_est, axis=0).T
    all_power_est = np.mean(all_power_est, axis=0).T

    hz_span = f[1] - f[0]
    # integration window ~1 for lfp, ~16 for ap
    n_hz = 16
    int_wind = int(np.ceil(n_hz / hz_span))

    # Find peaks for each channel in a window around the expected peak
    pk_bin_ind = np.argmin(np.abs(f - peak_pos_hz))

    b_start = max(0, pk_bin_ind - 10 * int_wind)
    b_end = pk_bin_ind + 10 * int_wind

    # Find the index of the max power within the window for each channel
    i = np.argmax(all_power_est[:, b_start:b_end], axis=1) + b_start

    ptp_mean, ptp_std, noise_mean, noise_std = (0, 0, 0, 0)
    outliers = np.array([])

    if meas_chan > 3:
        peak_to_peak_est, back_integrated, chan_label = _analyze_channel_spectra(
            all_power_est,
            i,
            int_wind,
            meas_chan,
            skip_chan,
            lfp_fs,
            nfft,
            noise_range,
            hz_span,
            back_skip_peak=1,
        )
        ptp_mean = np.mean(1000 * peak_to_peak_est)
        ptp_std = np.std(1000 * peak_to_peak_est)
        noise_mean = np.mean(1000 * back_integrated)
        noise_std = np.std(1000 * back_integrated)
    else:
        back1 = all_power_est[i - 2 * int_wind : i - int_wind]
        back2 = all_power_est[i + int_wind : i + 2 * int_wind]
        back_both = np.concatenate((back1, back2))
        back_est = np.mean(back_both)
        peak_int = np.sum(all_power_est[i - int_wind : i + int_wind] - back_est)
        ptp_mean = 2 * np.sqrt(2 * peak_int / ((1 / lfp_fs) * nfft))
        ptp_std = 0
        noise_mean = 0
        noise_std = 0
        outliers = 1

    return (
        lfp_by_channel,
        all_power_est,
        f,
        all_power_var,
        outliers,
        ptp_mean,
        ptp_std,
        noise_mean,
        noise_std,
    )


def lfp_band_power(run_dir=None, stream="ap", prb_ind=0, peak_pos_hz=1000, plot=True):
    if run_dir is None:
        print("Please select a SpikeGLX .bin file...")
        root = Tk()
        root.withdraw()
        bin_file = filedialog.askopenfilename(
            title="Select SpikeGLX .bin file",
            filetypes=[("SpikeGLX .bin files", "*.bin")],
        )
        if not bin_file:
            print("No file selected. Exiting.")
            return
        file_dir, file_name = os.path.split(bin_file)
    else:
        run_name = os.path.basename(run_dir)
        if stream == "ap" or stream == "lf":
            probe_dir = f"{run_name}_imec{prb_ind}"
            file_dir = os.path.join(run_dir, probe_dir)
            file_name = f"{run_name}_t0.imec{prb_ind}.{stream}.bin"
        elif stream == "nidq":
            file_dir = run_dir
            file_name = f"{run_name}_t0.nidq.bin"
        else:
            raise ValueError("Stream must be 'ap', 'lf', or 'nidq'.")
    bin_file = os.path.join(file_dir, file_name)
    if not os.path.exists(bin_file):
        print(f"File {bin_file} does not exist. Exiting.")
        return
    print(f"\t\tAnalyzing: {bin_file}")

    meta_file = bin_file.replace(".bin", ".meta")
    if not os.path.exists(meta_file):
        print(f"Meta file {meta_file} does not exist. Exiting.")
        return
    meta = npx.read_meta(meta_file)
    bank_times = npx.get_svy_bank_times(meta)

    fs = npx.get_sample_rate(meta)
    n_chans_in_file = int(meta["nSavedChans"])
    if meta["typeThis"] == "imec":
        [ap_gain, lf_gain, _, _] = npx.get_chan_gains_imec(meta)
        [ap, _, _] = npx.get_chan_counts_imec(meta)
        n_chans = ap
        if stream == "ap":
            gain = ap_gain[0]
            disp_range = (0.5, 15000)
            chan_step = n_chans_in_file // 3
            marginal_chans = np.arange(10, n_chans_in_file, chan_step)
            freq_bands = [[0.1, 200]]
            noise_range = (300, 10000)
        elif stream == "lf":
            gain = lf_gain[0]
            disp_range = (0.5, 100)
            marginal_chans = np.arange(10, n_chans_in_file, 100)
            freq_bands = [[0.5, 1000]]
            noise_range = (0.5, 1000)
    elif meta["typeThis"] == "nidq":
        gain = npx.get_chan_gains_ni(meta)
        [_, _, n_chans, _] = npx.get_chan_counts_ni(meta)
        skip_chan = []
        disp_range = (0.5, 15000)
        marginal_chans = [0]
        freq_bands = [[0.001, 200]]
        noise_range = (0.001, 500)
    elif meta["typeThis"] == "obx":
        gain = 1
        [_, n_chans, _] = npx.get_chan_counts_obx(meta)
        skip_chan = []
        disp_range = (0.5, 15000)
        marginal_chans = [0]
        freq_bands = [[0.001, 200]]
        noise_range = (0.001, 500)
    else:
        raise ValueError("Unknown stream type in metadata.")

    tomV = 1e3 * (npx.int2volts(meta) / gain)

    # Get disabled channels from the metadata
    skip_chan = []
    if meta["typeThis"] == "imec":
        # Use regex to parse the shank map string, similar to MATLAB's textscan
        shank_map_str = meta.get("snsShankMap", meta.get("snsGeomMap"))
        # Matches tuples like (c:h:a:n) and captures the 4th number (enabled flag)
        matches = re.findall(r"\(\d+:\d+:\d+:(\d)\)", shank_map_str)
        if matches:
            enabled = np.array([int(m) for m in matches])
            skip_chan = np.where(enabled == 0)[0].tolist()

    clip_dur = 5
    nclips = 3
    n_banks = 1 if type(bank_times) == int else len(bank_times)

    # --- 4. Run Analysis ---
    if n_banks > 1:
        for k in range(n_banks):
            t_start = bank_times[k, 3] + 200
            # shank_index = bank_times[k, 0]
            # bank_index = bank_times[k, 1]
            [
                _,
                all_power_est,
                f,
                _,
                _,
                ptp_mean,
                ptp_std,
                noise_mean,
                noise_std,
            ] = _lfp_band_power(
                bin_file,
                fs,
                tomV,
                peak_pos_hz,
                n_chans_in_file,
                n_chans,
                freq_bands,
                n_clips=nclips,
                clip_dur=clip_dur,
                skip_chan=skip_chan,
                noise_range=noise_range,
                t_start=t_start,
            )
    else:
        t_start = 11
        [
            _,
            all_power_est,
            f,
            _,
            _,
            ptp_mean,
            ptp_std,
            noise_mean,
            noise_std,
        ] = _lfp_band_power(
            bin_file,
            fs,
            tomV,
            peak_pos_hz,
            n_chans_in_file,
            n_chans,
            freq_bands,
            n_clips=nclips,
            clip_dur=clip_dur,
            skip_chan=skip_chan,
            noise_range=noise_range,
            t_start=t_start,
        )

    if n_chans > 1:
        all_power_est = all_power_est[:n_chans, :]

    # --- 5. Plot Results ---
    if plot:
        disp_f = (f > disp_range[0]) & (f <= disp_range[1])

        # Plot 1: Power Spectrum of Marginal Channels
        fig, ax = plt.subplots(figsize=(6, 2.5))

        if n_chans > 1:
            # Power Est is (Freq x Chan), select marginal channels
            yp = all_power_est[marginal_chans, :]
            yp = yp[:, disp_f]
            yp = 10 * np.log10(yp)
            ax.semilogx(f[disp_f], yp.T, linewidth=0.75)
        else:
            yp = 10 * np.log10(all_power_est[disp_f])
            ax.semilogx(f[disp_f], yp, linewidth=0.75)

        ax.set_ylabel("Power (dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend([str(c) for c in marginal_chans])
        ax.set_xlim(disp_range)
        ax.set_ylim(-90, -40)
        fig.tight_layout()

        # Plot 2: Power Spectrum of Outlier Channels
        # if n_chans > 1 and outliers.size > 0:
        #     fig2, ax2 = plt.subplots()
        #     yp_outlier = 10 * np.log10(all_power_est[:, outliers])
        #     ax2.plot(fig2, yp_outlier)
        #     ax2.set_title("Power Spectra for Outlier Channels")
        #     ax2.set_ylabel("Power (dB)")
        #     ax2.set_xlabel("Frequency (Hz)")
        #     ax2.legend([f"Chan {c}" for c in outliers])
        #     ax2.set_xlim(disp_range)

        # plt.show()

    return ptp_mean, ptp_std, noise_mean, noise_std, fig

import os
import re

from lfp_power import lfp_band_power


def find_match_dir(search_path, pattern):
    """
    Finds subdirectories within a given path that match a regex pattern.
    """
    pattern = re.compile(pattern)
    subdirs = []
    for dirpath, dirnames, filenames in os.walk(search_path):
        for dirname in dirnames:
            if pattern.search(dirname):
                subdirs.append(os.path.join(dirpath, dirname))
    return subdirs


def run_noise(base_path, stream="ap", peak_pos_hz=1000, overwrite=False, plot=True):
    """
    Main function to run the batch analysis.
    """
    # --- Start Processing ---
    # Find all run folders (e.g., matching '_g' followed by digits)
    run_folders = find_match_dir(base_path, r"_g\d+$")

    for run_name in run_folders:
        run_dir = os.path.join(base_path, run_name)

        print(f"Processing: {run_dir}")
        # Find all probe folders within the run folder (e.g., matching '_imec' followed by digits)
        probe_folders = find_match_dir(run_dir, r"_imec\d+$")

        for probe_dir_name in probe_folders:
            # Extract the probe number using a regex group
            match = re.search(r"_imec(\d+)$", probe_dir_name)
            if not match:
                continue  # Skip if the folder name is malformed

            probe_num = int(match.group(1))
            print(f"\tProbe {probe_num}")

            # Check if output already exists
            output_name = f"{probe_dir_name}_spectra.txt"
            output_path = os.path.join(run_dir, output_name)

            if not overwrite and os.path.exists(output_path):
                print(f"\t\tOutput exists. Skipping.")
                continue

            # Call the main analysis function
            # try:
            (_, _, noise_mean, noise_std, fig1) = lfp_band_power(
                run_dir, stream, probe_num, peak_pos_hz, True
            )

            # Save the numerical results to a text file
            # The 'with' statement automatically handles closing the file
            with open(output_path, "w") as f:
                result_str = f"noise mean & std: {noise_mean:.2f}, {noise_std:.2f}\n"
                print(f"\t\t{result_str.strip()}")
                f.write(result_str)

            # Save the figure generated by the analysis function
            fig_path = os.path.join(run_dir, f"{probe_dir_name}_spectra.png")

            fig1.savefig(fig_path, dpi=300)
            print(f"\t\tSaved results to {output_name} and spectra PNG.")

            # except Exception as e:
            #     print(f"\t\tERROR processing {probe_dir_name}: {e}")
