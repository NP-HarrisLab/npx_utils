import os
import shutil
from datetime import datetime

from tqdm import tqdm

import npx_utils as npx

from .ks_helpers import get_meta_path, get_probe_id
from .sglx.sglx_helpers import read_meta


def copy_folder_with_progress(src, dest, overwrite=False):
    """
    Copies a folder from src to dest with a progress bar.
    """
    # Get the list of all files and directories
    # check if src and dest are the same
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    tqdm.write(f"Copying from {src} to {dest}")
    files_and_dirs = []
    for root, _, files in os.walk(src):
        for file in files:
            files_and_dirs.append(os.path.join(root, file))

    # Initialize the progress bar
    for item in tqdm(files_and_dirs, desc=f"Copying files...", unit=" file"):
        # Determine destination path
        relative_path = os.path.relpath(item, src)
        dest_path = os.path.join(dest, relative_path)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy the file
        try:
            if os.path.exists(dest_path):
                # Check if the file is the same size
                if overwrite:
                    os.remove(dest_path)
                elif os.path.getsize(item) == os.path.getsize(dest_path):
                    continue
            shutil.copy2(item, dest_path)
        except Exception as e:
            print(f"Error copying {item} to {dest_path}: {e}")


def get_probe_folders(ks_folders):
    probe_folders = {}
    for ks_folder in ks_folders:
        probe_num = get_probe_id(ks_folder)
        if probe_num not in probe_folders:
            probe_folders[probe_num] = []
        probe_folders[probe_num].append(ks_folder)
    return probe_folders


def get_details(ks_folder, drug_dict=None):
    """
    Get the details of the kilosort folder.
    Assumes imro file is named in the form of subject_region.imro
    """
    meta_path = get_meta_path(ks_folder)
    meta = read_meta(meta_path)
    imroFile = os.path.basename(meta["imroFile"])
    subject, region = imroFile.split("_")[:2]
    region = region.split(".")[0]
    if drug_dict is not None:
        drug = drug_dict.get(subject, "unknown")
    else:
        drug = "unknown"
    date_obj = datetime.fromisoformat(meta["fileCreateTime"])
    date = date_obj.strftime("%Y%m%d")
    probe_num = get_probe_id(ks_folder)

    details = {
        "subject": subject,
        "region": region,
        "date": date,
        "probe_num": probe_num,
        "drug": drug,
    }
    return details


def get_run_folders(subject_folder):
    day_folders = [
        os.path.join(subject_folder, folder)
        for folder in os.listdir(subject_folder)
        if os.path.isdir(os.path.join(subject_folder, folder))
        and ("SvyPrb" not in folder)
        and ("old" not in folder)
    ]

    run_folders = []
    for day_folder in day_folders:
        possible_run_folders = [
            os.path.join(day_folder, folder)
            for folder in os.listdir(day_folder)
            if npx.is_run_folder(folder)
        ]
        # find one with supercat
        supercat_folders = [
            folder for folder in possible_run_folders if "supercat" in folder
        ]
        if len(supercat_folders) > 1:
            raise ValueError(
                f"Multiple supercat folders found in {day_folder}: {supercat_folders}"
            )
        if len(supercat_folders) == 1:
            run_folders.append(supercat_folders[0])
        else:
            # find catgt folders
            catgt_folders = [
                folder for folder in possible_run_folders if "catgt" in folder
            ]
            if len(catgt_folders) > 1:
                raise ValueError(
                    f"Multiple catgt folders found in {day_folder}: {catgt_folders}"
                )
            if len(catgt_folders) == 1:
                run_folders.append(catgt_folders[0])
            else:
                if len(possible_run_folders) == 1:
                    run_folders.append(possible_run_folders[0])
                elif len(possible_run_folders) == 0:
                    continue
                else:
                    raise ValueError(
                        f"Multiple run folders found in {day_folder}: {possible_run_folders}"
                    )
    run_folders.sort()
    return run_folders
