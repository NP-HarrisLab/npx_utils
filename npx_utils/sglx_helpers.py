import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import mode

from .ks_helpers import get_lfp_meta_path, load_params


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


def get_lfp_memmap(ks_folder):
    """
    Load the data from the binary file as a memory-mapped array.
    """
    params = load_params(ks_folder)
    lfp_path = params["dat_path"].replace(".ap.bin", ".lf.bin")
    data = np.memmap(lfp_path, dtype="int16", mode="r")
    data = np.reshape(data, (-1, params["n_channels_dat"]))
    return data


def get_lfp_sample_rate(ks_folder):
    meta_path = get_lfp_meta_path(ks_folder)
    meta = read_meta(meta_path)
    return float(meta["imSampRate"])


def get_sample_rate(meta):
    """
    Get the sample rate from the metadata.
    """
    if meta["typeThis"] == "imec":
        sample_rate = float(meta["imSampRate"])
    elif meta["typeThis"] == "nidq":
        sample_rate = float(meta["niSampRate"])
    elif meta["typeThis"] == "obx":
        sample_rate = float(meta["obSampRate"])
    else:
        print("Error: unknown stream type")
        sample_rate = 1
    return sample_rate


def get_svy_bank_times(meta):
    if meta.get("svySBTT") is None:
        return 0
    pattern = r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    parsed_data = re.findall(pattern, meta["svySBTT"])
    if not parsed_data:
        return ValueError("No valid bank times found in svySBTT")
    parsed_data = np.array(parsed_data, dtype=float)
    n_bank = len(parsed_data) + 1
    bank_times = np.zeros((n_bank, 4), dtype=float)
    sample_rate = get_sample_rate(meta)
    bank_times[1:, 0:2] = parsed_data[:, 0:2]  # Shank and Bank IDs
    bank_times[1:, 2:4] = parsed_data[:, 2:4] / sample_rate
    return bank_times


def get_chan_gains_imec(meta):
    # list of probe types with NP 1.0 imro format
    np1_imro = [0, 1020, 1030, 1200, 1100, 1120, 1121, 1122, 1123, 1300]
    # number of channels acquired
    acqCountList = meta["acqApLfSy"].split(sep=",")
    APgain = np.zeros(int(acqCountList[0]))  # default type = float64
    LFgain = np.zeros(int(acqCountList[1]))  # empty array for 2.0

    if "imDatPrb_type" in meta:
        probeType = int(meta["imDatPrb_type"])
    else:
        probeType = 0

    if sum(np.isin(np1_imro, probeType)):
        # imro + probe allows setting gain independently for each channel
        imroList = meta["imroTbl"].split(sep=")")
        # One entry for each channel plus header entry,
        # plus a final empty entry following the last ')'
        for i in range(0, int(acqCountList[0])):
            currList = imroList[i + 1].split(sep=" ")
            APgain[i] = float(currList[3])
            LFgain[i] = float(currList[4])
    else:
        # get gain from imChan0apGain
        if "imChan0apGain" in meta:
            APgain = APgain + float(meta["imChan0apGain"])
            if int(acqCountList[1]) > 0:
                LFgain = LFgain + float(meta["imChan0lfGain"])
        elif probeType == 1110:
            # active UHD, for metadata lacking imChan0apGain, get gain from
            # imro table header
            imroList = meta["imroTbl"].split(sep=")")
            currList = imroList[0].split(sep=",")
            APgain = APgain + float(currList[3])
            LFgain = LFgain + float(currList[4])
        elif (probeType == 21) or (probeType == 24):
            # development NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            APgain = APgain + 80
        elif probeType == 2013:
            # commercial NP 2.0; APGain = 100 for all AP
            APgain = APgain + 100
        else:
            print("unknown gain, setting APgain to 1")
            APgain = APgain + 1
    fI2V = int2volts(meta)
    APChan0_to_uV = 1e6 * fI2V / APgain[0]
    if LFgain.size > 0:
        LFChan0_to_uV = 1e6 * fI2V / LFgain[0]
    else:
        LFChan0_to_uV = 0
    return (APgain, LFgain, APChan0_to_uV, LFChan0_to_uV)


def int2volts(meta):
    if meta["typeThis"] == "imec":
        if "imMaxInt" in meta:
            maxInt = int(meta["imMaxInt"])
        else:
            maxInt = 512
        fI2V = float(meta["imAiRangeMax"]) / maxInt
    elif meta["typeThis"] == "nidq":
        maxInt = int(meta["niMaxInt"])
        fI2V = float(meta["niAiRangeMax"]) / maxInt
    elif meta["typeThis"] == "obx":
        maxInt = int(meta["obMaxInt"])
        fI2V = float(meta["obAiRangeMax"]) / maxInt
    else:
        print("Error: unknown stream type")
        fI2V = 1

    return fI2V


def get_chan_gains_ni(ichan, savedMN, savedMA, meta):
    if ichan < savedMN:
        gain = float(meta["niMNGain"])
    elif ichan < (savedMN + savedMA):
        gain = float(meta["niMAGain"])
    else:
        gain = 1  # non multiplexed channels have no extra gain
    return gain


def get_chan_counts_imec(meta):
    chanCountList = meta["snsApLfSy"].split(sep=",")
    AP = int(chanCountList[0])
    LF = int(chanCountList[1])
    SY = int(chanCountList[2])
    return (AP, LF, SY)


def get_chan_counts_ni(meta):
    chanCountList = meta["snsMnMaXaDw"].split(sep=",")
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return (MN, MA, XA, DW)


def get_chan_counts_obx(meta):
    chanCountList = meta["snsXaDwSy"].split(sep=",")
    XA = int(chanCountList[0])
    DW = int(chanCountList[1])
    SY = int(chanCountList[2])
    return (XA, DW, SY)
