import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import mode

from .ks_helpers import load_params


def read_meta(meta_path):
    meta_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            mdatList = f.read()
        mdatList = mdatList.splitlines()
        # convert the list entries into key value pairs
        for m in mdatList:
            csList = m.split(sep="=")
            if csList[0][0] == "~":
                currKey = csList[0][1 : len(csList[0])]
            else:
                currKey = csList[0]
            meta_dict.update({currKey: csList[1]})
    else:
        print("no meta file")

    return meta_dict


def get_all_channel_counts(meta):
    chanCountList = meta["snsApLfSy"].split(sep=",")
    n_channel_ap = int(chanCountList[0])
    n_channel_lf = int(chanCountList[1])
    n_channel_sync = int(chanCountList[2])

    return n_channel_ap, n_channel_lf, n_channel_sync


def get_ap_data_channel_count(meta):
    return get_all_channel_counts(meta)[0]


def get_bits_to_uV(meta):
    if "imDatPrb_type" in meta:
        pType = meta["imDatPrb_type"]
        if pType == "0":
            probe_type = "NP1"
        else:
            probe_type = "NP" + pType
    else:
        probe_type = "3A"  # 3A probe is default

    # first check if metadata includes the imChan0apGain key
    if "uVPerBit" in meta:
        return float(meta["uVPerBit"])

    if "imChan0apGain" in meta:
        APgain = float(meta["imChan0apGain"])
        voltage_range = float(meta["imAiRangeMax"]) - float(meta["imAiRangeMin"])
        maxInt = float(meta["imMaxInt"])
        uVPerBit = (1e6) * (voltage_range / APgain) / (2 * maxInt)

    else:
        imroList = meta["imroTbl"].split(sep=")")
        # One entry for each channel plus header entry,
        # plus a final empty entry following the last ')'
        # channel zero is the 2nd element in the list

        if probe_type == "NP21" or probe_type == "NP24":
            # NP 2.0; APGain = 80 for all channels
            # voltage range = 1V
            # 14 bit ADC
            uVPerBit = (1e6) * (1.0 / 80) / pow(2, 14)
        elif probe_type == "NP1110":
            # UHD2 with switches, special imro table with gain in header
            currList = imroList[0].split(sep=",")
            APgain = float(currList[3])
            uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)
        else:
            # 3A, 3B1, 3B2 (NP 1.0), or other NP 1.0-like probes
            # voltage range = 1.2V
            # 10 bit ADC
            currList = imroList[1].split(
                sep=" "
            )  # 2nd element in list, skipping header
            APgain = float(currList[3])
            uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)

    # # save this value in meta
    # with open(meta_path, "ab") as f:
    #     f.write(f"uVPerBit={uVPerBit}\n".encode("utf-8"))

    return uVPerBit


def get_same_channel_positions(ks_folders):
    """
    Get the kilosort folders with the same channel positions.
    """
    channel_positions = []
    channel_positions_tuples = []
    for ks_folder in ks_folders:
        channel_position = np.load(os.path.join(ks_folder, "channel_positions.npy"))
        # convert to immutable tuple
        channel_position_tuple = tuple(map(tuple, channel_position))
        channel_positions.append(channel_position)
        channel_positions_tuples.append(channel_position_tuple)

    most_common = mode(channel_positions_tuples, axis=0)
    indices = [
        i
        for i in range(len(channel_positions))
        if np.array_equal(channel_positions[i], most_common.mode)
    ]
    return [ks_folders[i] for i in indices]


def get_data_memmap(ks_folder):
    """
    Load the data from the binary file as a memory-mapped array.
    """
    params = load_params(ks_folder)
    data = np.memmap(params["dat_path"], dtype="int16", mode="r")
    data = np.reshape(data, (-1, params["n_channels_dat"]))
    return data
