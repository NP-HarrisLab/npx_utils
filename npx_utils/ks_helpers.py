import os
import re

import numpy as np


def is_run_folder(folder):
    pattern = re.compile(r".*_g\d+$")
    # ignore those that have SvyPrb in the name
    if "SvyPrb" in folder:
        return False
    if "old" in folder:
        return False
    return pattern.match(folder) != None


def get_ks_folders(root_dir, ks_version="4", subfolders=True):
    """
    Find all kilosort folders in the root_dir for given ks_version. if subfolders is true, only one recording or supercat recording session in folder.
    """
    if ks_version == "2.5":
        ks_version = "25"
    pattern = re.compile(rf"imec\d+_ks{ks_version}$")
    matching_folders = []
    for root, dirs, _ in os.walk(root_dir):
        if "$RECYCLE.BIN" in root:
            continue
        if subfolders:
            root_base = os.path.basename(root)
            supercat_dir = (
                root if root_base.startswith("supercat_") else None
            ) or next(
                (os.path.join(root, d) for d in dirs if d.startswith("supercat_")), None
            )
            catgt_dir = (root if root_base.startswith("catgt_") else None) or next(
                (os.path.join(root, d) for d in dirs if d.startswith("catgt_")), None
            )

            selected_dir = supercat_dir or catgt_dir
            if selected_dir:
                for sub_root, sub_dirs, _ in os.walk(selected_dir):
                    for sub_dir in sub_dirs:
                        if pattern.match(sub_dir):
                            matching_folders.append(os.path.join(sub_root, sub_dir))
                dirs[:] = []  # Clear dirs to prevent os.walk from going deeper
        else:
            for dir in dirs:
                if pattern.match(dir):
                    matching_folders.append(os.path.join(root, dir))
    # TO DO make better Remove duplicates
    matching_folders = list(set(matching_folders))
    # remove any folders with "old" or "SvyPrb" in the name
    matching_folders = [
        folder
        for folder in matching_folders
        if "old" not in folder and "SvyPrb" not in folder
    ]
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


def get_lfp_binary_path(ks_folder, params=None):
    """
    Get the path to the lfp binary file.
    """
    binary_path = get_binary_path(ks_folder, params)
    lfp_binary_path = binary_path.replace(".ap.bin", ".lf.bin")
    return lfp_binary_path


def get_lfp_meta_path(ks_folder, params=None):
    """
    Get the path to the lfp meta file.
    """
    binary_path = get_lfp_binary_path(ks_folder, params)
    lfp_meta_path = binary_path.replace(".bin", ".meta")
    return lfp_meta_path


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
    return int(probe_id)


def get_sample_rate(ks_folder, params=None):
    """
    Get the sample rate from the params file.
    """
    if params is None:
        params = load_params(ks_folder)
    return params["sample_rate"]
