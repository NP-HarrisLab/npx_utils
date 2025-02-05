import os
import re

import numpy as np

def is_run_folder(folder):
    pattern = re.compile(r'.*_g\d+$')
    return pattern.match(folder)

def get_ks_folders(root_dir, ks_version="4", catgt=True):
    """
    Find all kilosort folders in the root_dir for given ks_version. Will aslo only find those in catgt folders or non-catgt.
    """
    if is_run_folder(root_dir) and catgt:
        root_dir = os.path.join(os.path.dirname(root_dir), f"catgt_{os.path.basename(root_dir)}")
    if ks_version == "2.5":
        ks_version = "25"
    pattern = re.compile(r"imec\d_ks\d+$")
    matching_folders = []
    for root, dirs, _ in os.walk(root_dir):
        if "$RECYCLE.BIN" in root:
            continue
        for dir in dirs:
            if pattern.match(dir) and (not catgt or "catgt" in root):
                if dir.split("_")[1] == f"ks{ks_version}":
                    matching_folders.append(os.path.join(root, dir))
    return matching_folders


def load_params(ks_folder):
    """
    Load the parameters from the params.py file. Update the data_path to be an absolute path.
    """
    params = {}
    with open(os.path.join(ks_folder, "params.py")) as f:
        code = f.read()
        exec(code, {}, params)
    if not os.path.isabs(params["dat_path"]):
        params["dat_path"] = os.path.abspath(
            os.path.join(ks_folder, params["dat_path"])
        )
    return params


def load_data(params):
    data = np.memmap(params["dat_path"], dtype="int16", mode="r")
    data = np.reshape(data, (-1, params["n_channels_dat"]))
    return data


def get_binary_path(ks_folder, params=None):
    """
    Get the path to the binary file.
    """
    if params is None:
        params = load_params(ks_folder)
    return params["dat_path"]


def get_meta_path(ks_folder, params=None):
    """
    Get the path to the meta file.
    """
    binary_path = get_binary_path(ks_folder, params)
    meta_path = binary_path.replace(".bin", ".meta")
    return meta_path


def get_n_channels(ks_folder, params=None):
    """
    Get the number of channels from the params file.
    """
    if params is None:
        params = load_params(ks_folder)
    return params["n_channels_dat"]


def get_probe_id(ks_folder):
    """
    Extract the probe id from the kilosort folder name.
    """
    folders = ks_folder.split(os.sep)
    probe_id = re.search(r"(?<=imec)\d+(?=_)", folders[-1]).group(0)
    return probe_id


def get_sample_rate(ks_folder, params=None):
    """
    Get the sample rate from the params file.
    """
    if params is None:
        params = load_params(ks_folder)
    return params["sample_rate"]
