import math
from pathlib import Path

import numpy as np
from scipy import interpolate
from tqdm import tqdm


def get_sign(mean_wf):
    troughs = np.min(mean_wf, axis=1)
    peaks = np.max(mean_wf, axis=1)
    sign = abs(np.min(troughs)) > abs(np.max(peaks))
    return sign


def get_amp_map(shank_pos, shank_amplitudes):
    max_amp_idx = np.argmax(shank_amplitudes)
    max_y = shank_pos[max_amp_idx, 1]
    # find channels within 500 um
    nearby_mask = np.abs(shank_pos[:, 1] - max_y) <= 500
    xcoords = shank_pos[nearby_mask, 0]
    ycoords = shank_pos[nearby_mask, 1]
    xmin, xmax = xcoords.min(), xcoords.max()
    ymin, ymax = ycoords.min(), ycoords.max()
    width = np.ceil(xmax - xmin + 1).astype(int)
    height = np.ceil(ymax - ymin + 1).astype(int)
    xcoords2 = np.round(xcoords - xmin).astype(int)
    ycoords2 = np.round(ycoords - ymin).astype(int)
    width1 = np.arange(0, width)
    height1 = np.arange(0, height)
    xx, yy = np.meshgrid(width1, height1)
    if len(np.unique(xcoords)) > 1:
        gd1 = interpolate.griddata(
            (ycoords2, xcoords2), shank_amplitudes, (yy, xx), method="cubic"
        )
    else:
        spl = interpolate.interp1d(
            ycoords2, shank_amplitudes, kind="cubic", fill_value="extrapolate"
        )
        gd1 = spl(yy)
    return gd1


def polar_to_cartesian(theta, r):
    xall = []
    yall = []
    for i, r1 in enumerate(r):
        x = r1 * math.cos(theta[i])
        y = r1 * math.sin(theta[i])
        xall.append(x)
        yall.append(y)

    return xall, yall


def create_sampling_matrix(pointN):
    # sample amp at different radius from the peak site
    # first design the sampling position matrix
    rho = np.arange(0, pointN)
    # Initialize theta array and set up parameters
    theta = np.zeros_like(rho)
    count1 = 0
    # Define beta range from -pi to pi with steps of pi/6
    beta_values = np.arange(-np.pi, np.pi, np.pi / 6)
    # Initialize xq and yq as empty lists
    xq = np.empty((0, pointN))
    yq = np.empty((0, pointN))
    for beta in beta_values:
        x, y = polar_to_cartesian(theta + beta, rho)  # Use numpy's pol2cart equivalent
        xq = np.append(xq, np.array(x).reshape(1, -1), axis=0)
        yq = np.append(yq, np.array(y).reshape(1, -1), axis=0)

    return xq, yq


def get_spatial_footprint(ks_folder, threshold=30):
    "https://github.com/zhiwen10/Neuropixels-footprint/blob/main/python/get_footprint_new3.py#L74"
    # make ks_folder a Path object if not already
    ks_folder = Path(ks_folder)
    mean_wfs = np.load(ks_folder / "mean_waveforms.npy")
    channel_positions = np.load(ks_folder / "channel_positions.npy")
    footprint = []
    for mean_wf in tqdm(
        mean_wfs, total=len(mean_wfs), desc="Calculating spatial footprints"
    ):
        sign = get_sign(mean_wf)
        if sign == 0:
            mean_wf = -mean_wf

        trough_idx = np.argmin(mean_wf, axis=1)
        peak_idx = np.argmax(mean_wf[:, trough_idx:], axis=1) + trough_idx
        amplitudes = mean_wf[:, peak_idx] - mean_wf[:, trough_idx]
        # loop through each shank
        for shank in range(4):
            shank_mask = (channel_positions[:, 0] >= shank * 250) & (
                channel_positions[:, 0] < (shank + 1) * 250
            )
            shank_pos = channel_positions[shank_mask]
            shank_amplitudes = amplitudes[shank_mask]
            if len(shank_amplitudes) == 0:
                continue
            gd1 = get_amp_map(shank_pos, shank_amplitudes)
            flat_index = np.nanargmax(np.abs(gd1))
            row, col = np.unravel_index(flat_index, gd1.shape)
            if len(np.unique(shank_pos[:, 0])) > 1:
                pointN = 401
                xq, yq = create_sampling_matrix(pointN)
                xq2 = np.round(xq + col).astype(int)
                yq2 = np.round(yq + row).astype(int)
                vq = np.full(xq2.shape, np.nan)
                for i in range(xq2.shape[0]):
                    for j in range(xq2.shape[1]):
                        if (1 <= yq2[i, j] < gd1.shape[0]) and (
                            1 <= xq2[i, j] < gd1.shape[1]
                        ):
                            vq[i, j] = gd1[yq2[i, j], xq2[i, j]]
                else:
                    rho = np.arange(0, 401)
                    yq2 = np.empty((0, 401))
                    a1 = np.round(rho + row).astype(int)
                    yq2 = np.append(yq2, np.array(a1).reshape(1, -1), axis=0)
                    a2 = np.round(-rho + row).astype(int)
                    yq2 = np.append(yq2, np.array(a2).reshape(1, -1), axis=0)
                    yq2 = yq2.astype(int)
                    # Initialize vq with NaN values
                    vq = np.full(yq2.shape, np.nan)
                    # Iterate through each element in xq2 and yq2
                    for i in range(0, 2):
                        for j in range(yq2.shape[1]):
                            if 1 <= yq2[i, j] < gd1.shape[0]:
                                vq[i, j] = gd1[yq2[i, j]]
                vq_mean = np.nanmean(vq, axis=0)
                indices = np.where(vq_mean <= threshold)[0]
                ft = indices[0] if indices.size > 0 else 100
                footprint.append(ft)
        return np.array(footprint)