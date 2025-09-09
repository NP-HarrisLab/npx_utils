import os
import re

import cupy as cp
import numpy as np
from scipy.stats import mode

from . import _SGLXMetaToCoords


def read_meta(meta_path):
    return _SGLXMetaToCoords.readMeta(meta_path)


def get_channel_counts(meta):
    return _SGLXMetaToCoords.ChannelCountsIM(meta)


def get_sample_rate(meta):
    if meta["typeThis"] == "imec":
        srate = float(meta["imSampRate"])
    elif meta["typeThis"] == "nidq":
        srate = float(meta["niSampRate"])
    elif meta["typeThis"] == "obx":
        srate = float(meta["obSampRate"])
    else:
        raise ValueError("Unknown stream type")
    return srate


def convert_data_to_uV(data, chan_list, meta):
    if meta["typeThis"] == "imec":
        data_V = GainCorrectIM(data, chan_list, meta)
    elif meta["typeThis"] == "nidq":
        data_V = GainCorrectNI(data, chan_list, meta)
    elif meta["typeThis"] == "obx":
        data_V = GainCorrectOBX(data, chan_list, meta)
    else:
        raise ValueError("Unknown stream type")
    data_uV = data_V * 1e6
    return data_uV


def get_bits_to_uV(chan_list, meta):
    if meta["typeThis"] == "imec":
        bits_to_V = get_gain_correction_im(chan_list, meta)
    elif meta["typeThis"] == "nidq":
        # bits_to_V = GainCorrectNI(data, chan_list, meta)
        pass
    elif meta["typeThis"] == "obx":
        # bits_to_V = GainCorrectOBX(data, chan_list, meta)
        pass
    else:
        raise ValueError("Unknown stream type")
    bits_to_uV = bits_to_V * 1e6
    return bits_to_uV


def get_data_memmap(bin_path, meta):
    nChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nChan))
    rawData = np.memmap(
        bin_path, dtype="int16", mode="r", shape=(nChan, nFileSamp), offset=0, order="F"
    )
    return rawData


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


### Helpers
def OriginalChans(meta):
    """
    Return array of original channel IDs. As an example, suppose we want the
    imec gain for the ith channel stored in the binary data. A gain array
    can be obtained using ChanGainsIM(), but we need an original channel
    index to do the lookup. Because you can selectively save channels, the
    ith channel in the file isn't necessarily the ith acquired channel.
    Use this function to convert from ith stored to original index.
    Note that the SpikeGLX channels are 0 based.
    """
    if meta["snsSaveChanSubset"] == "all":
        # output = int32, 0 to nSavedChans - 1
        chans = np.arange(0, int(meta["nSavedChans"]))
    else:
        # parse the snsSaveChanSubset string
        # split at commas
        chStrList = meta["snsSaveChanSubset"].split(sep=",")
        chans = np.arange(0, 0)  # creates an empty array of int32
        for sL in chStrList:
            currList = sL.split(sep=":")
            if len(currList) > 1:
                # each set of contiguous channels specified by
                # chan1:chan2 inclusive
                newChans = np.arange(int(currList[0]), int(currList[1]) + 1)
            else:
                newChans = np.arange(int(currList[0]), int(currList[0]) + 1)
            chans = np.append(chans, newChans)
    return chans


def ChannelCountsNI(meta):
    """
    Return counts of each nidq channel type that composes the timepoints stored in the binary file.
    """
    chanCountList = meta["snsMnMaXaDw"].split(sep=",")
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return (MN, MA, XA, DW)


def ChannelCountsIM(meta):
    """
    Return counts of each imec channel type that composes the timepoints stored in the binary files.
    """
    chanCountList = meta["snsApLfSy"].split(sep=",")
    AP = int(chanCountList[0])
    LF = int(chanCountList[1])
    SY = int(chanCountList[2])
    return (AP, LF, SY)


def ChannelCountsOBX(meta):
    """
    Return counts of each obx channel type that composes the timepoints stored in the binary files.
    """
    chanCountList = meta["snsXaDwSy"].split(sep=",")
    XA = int(chanCountList[0])
    DW = int(chanCountList[1])
    SY = int(chanCountList[2])
    return (XA, DW, SY)


def ChanGainNI(ichan, savedMN, savedMA, meta):
    """
    Return gain for ith channel stored in nidq file.
    ichan is a saved channel index, rather than the original (acquired) index.
    """
    if ichan < savedMN:
        gain = float(meta["niMNGain"])
    elif ichan < (savedMN + savedMA):
        gain = float(meta["niMAGain"])
    else:
        gain = 1  # non multiplexed channels have no extra gain
    return gain


def ChanGainsIM(meta):
    """
    Return gain for imec channels.
    Index into these with the original (acquired) channel IDs.
    """
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
    fI2V = Int2Volts(meta)
    APChan0_to_uV = 1e6 * fI2V / APgain[0]
    if LFgain.size > 0:
        LFChan0_to_uV = 1e6 * fI2V / LFgain[0]
    else:
        LFChan0_to_uV = 0
    return (APgain, LFgain, APChan0_to_uV, LFChan0_to_uV)


def Int2Volts(meta):
    """
    Return a multiplicative factor for converting 16-bit file data
    to voltage. This does not take gain into account. The full
    conversion with gain is:
        dataVolts = dataInt * fI2V / gain
    Note that each channel may have its own gain.
    """
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


def GainCorrectNI(dataArray, chanList, meta):
    """
    Having accessed a block of raw nidq data using makeMemMapRaw, convert
    values to gain-corrected voltage. The conversion is only applied to the
    saved-channel indices in chanList
    """
    MN, MA, XA, DW = ChannelCountsNI(meta)
    fI2V = Int2Volts(meta)
    # print statements used for testing...
    # print("NI fI2V: %.3e" % (fI2V))
    # print("NI ChanGainNI: %.3f" % (ChanGainNI(0, MN, MA, meta)))

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        j = chanList[i]  # index in saved data
        conv = fI2V / ChanGainNI(j, MN, MA, meta)
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * conv
    return convArray


def GainCorrectOBX(dataArray, chanList, meta):
    """
    Having accessed a block of raw obx data using makeMemMapRaw, convert
    values to volts. The conversion is only applied to the
    saved-channel
    """
    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * fI2V
    return convArray


def get_gain_correction_im(chanList, meta):
    chans = OriginalChans(meta)
    APgain, LFgain, _, _ = ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2
    fI2V = Int2Volts(meta)
    conversion = np.zeros(len(chanList))
    for i in range(0, len(chanList)):
        j = chanList[i]  # index into timepoint
        k = chans[j]  # acquisition index
        if k < nAP:
            conv = fI2V / APgain[k]
        elif k < nNu:
            conv = fI2V / LFgain[k - nAP]
        else:
            conv = 1
        # The dataArray contains only the channels in chanList
        conversion[i] = conv
    return conversion


def GainCorrectIM(dataArray, chanList, meta):
    """
    Having accessed a block of raw imec data using makeMemMapRaw, convert
    values to gain corrected voltages. The conversion is only applied to
    the saved-channel indices in chanList.
    """
    # Look up gain with acquired channel ID
    chans = OriginalChans(meta)
    APgain, LFgain, _, _ = ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2

    # Common conversion factor
    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype="float")
    for i in range(0, len(chanList)):
        j = chanList[i]  # index into timepoint
        k = chans[j]  # acquisition index
        if k < nAP:
            conv = fI2V / APgain[k]
        elif k < nNu:
            conv = fI2V / LFgain[k - nAP]
        else:
            conv = 1
        # The dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * conv
    return convArray
