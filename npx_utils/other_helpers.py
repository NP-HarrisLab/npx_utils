import os
import shutil

from tqdm import tqdm

from .ks_helpers import get_probe_id


def copy_folder_with_progress(src, dest):
    """
    Copies a folder from src to dest with a progress bar.
    """
    # Get the list of all files and directories
    files_and_dirs = []
    for root, _, files in os.walk(src):
        for file in files:
            files_and_dirs.append(os.path.join(root, file))

    # Initialize the progress bar
    for item in tqdm(files_and_dirs, desc="Copying files", unit=" file"):
        # Determine destination path
        relative_path = os.path.relpath(item, src)
        dest_path = os.path.join(dest, relative_path)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy the file
        try:
            if os.path.exists(dest_path):
                # Check if the file is the same size
                if os.path.getsize(item) == os.path.getsize(dest_path):
                    continue
            shutil.copy2(item, dest_path)
        except Exception as e:
            print(f"Error copying {item} to {dest_path}: {e}")


def get_probe_folders(ks_folders, catgt_only=True):
    probe_folders = {}
    for ks_folder in ks_folders:
        probe_num = get_probe_id(ks_folder)
        if probe_num not in probe_folders:
            probe_folders[probe_num] = []
        catgt_folder = ks_folder.split(os.sep)[-3]
        if not catgt_only or catgt_folder.startswith("catgt_"):
            probe_folders[probe_num].append(ks_folder)
    return probe_folders
