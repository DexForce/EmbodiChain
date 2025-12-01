# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
import sys
import shutil
import hashlib
import open3d as o3d

from embodichain.utils import logger


class EmbodiChainDataset(o3d.data.DownloadDataset):
    def __init__(self, prefix, data_descriptor, path):
        # Perform the zip file and extracted contents check
        # If the zip was not valid, the zip file would have been removed
        # and the parent class would download and extract it again
        self.check_zip(prefix, data_descriptor, path)
        # Call the parent class constructor
        super().__init__(prefix, data_descriptor, path)

    def check_zip(self, prefix, data_descriptor, path):
        """Check the integrity of the zip file and its extracted contents."""
        # Path to the downloaded zip file
        zip_file_name = os.path.split(data_descriptor.urls[0])[1]
        zip_dir_path = os.path.join(path, "download", f"{prefix}")
        zip_path = os.path.join(path, "download", f"{prefix}", f"{zip_file_name}")
        # Path to the extracted directory
        extracted_path = os.path.join(path, "extract", prefix)

        def is_safe_path(path_to_check):
            """Verify if the path is within safe directory boundaries"""
            return (
                "embodichain_data/download" in path_to_check
                or "embodichain_data/extract" in path_to_check
            )

        def safe_remove_directory(dir_path):
            """Safely remove a directory after path validation"""
            if not is_safe_path(dir_path):
                logger.log_warning(
                    f"Safety check failed, refusing to delete directory: {dir_path}"
                )
                return False

            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logger.log_info(f"Successfully removed directory: {dir_path}")
                    return True
                except OSError as e:
                    logger.log_warning(f"Error while removing directory: {e}")
                    return False
            return True

        # Check if the file already exists
        if os.path.exists(zip_path):
            # Calculate MD5 checksum of the existing file
            md5_existing = self.calculate_md5(zip_path)
            # Compare with the expected MD5 checksum
            if md5_existing != data_descriptor.md5:
                # If checksums do not match, delete the existing file
                os.remove(zip_path)
                # Ensure the extracted directory is removed if it exists
                safe_remove_directory(extracted_path)
                logger.log_warning(
                    f"Invalid MD5 checksum detected:\n"
                    f"  - File: {zip_path}\n"
                    f"  - Expected MD5: {data_descriptor.md5}\n"
                    f"  - Actual MD5: {md5_existing}\n"
                    f"Cleaned up invalid files and directories for fresh download."
                )
                return
        else:
            safe_remove_directory(zip_dir_path)
            safe_remove_directory(extracted_path)
            logger.log_info(
                f"ZIP file not found at {zip_path}."
                f"Cleaning up related directories for fresh download."
            )
            return

        # Check if the extracted directory exists and is not empty
        if not os.path.exists(extracted_path) or not os.listdir(extracted_path):
            # Remove the zip file to trigger Open3D's automatic download mechanism
            # Open3D will re-download and extract when the zip file is missing
            if os.path.exists(zip_path):
                os.remove(zip_path)

            # Clean up any existing empty extraction directory
            # This ensures a clean state for the upcoming extraction process
            safe_remove_directory(extracted_path)
            logger.log_info(
                f"Removed zip file {zip_path} and extracted path {extracted_path} to trigger Open3D download and extract. "
                f"Reason: {'Missing extraction directory.' if not os.path.exists(extracted_path) else 'Empty extraction directory.'}"
            )
            return

    def calculate_md5(self, file_path, chunk_size=8192):
        """Calculate the MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def get_data_class(dataset_name: str):
    """Retrieve the dataset class from the available modules.

    Args:
        dataset_name (str): The name of the dataset class.

    Returns:
        type: The dataset class.

    Raises:
        AttributeError: If the dataset class is not found in any module.
    """
    module_names = [
        "embodichain.data",
        "embodichain.data.assets",
        __name__,
    ]

    for module_name in module_names:
        try:
            return getattr(sys.modules[module_name], dataset_name)
        except AttributeError:
            continue

    raise AttributeError(f"Dataset class '{dataset_name}' not found in any module.")


def get_data_path(data_path_in_config: str) -> str:
    """Get the absolute path of the data file.

    Args:
        data_path_in_config (str): The dataset path in the format "${dataset_name}/subpath".

    Returns:
        str: The absolute path of the data file.
    """
    if os.path.isabs(data_path_in_config):
        return data_path_in_config

    split_str = data_path_in_config.split("/")
    dataset_name = split_str[0]
    sub_path = os.path.join(*split_str[1:])

    # Use the optimized get_data_class function
    data_class = get_data_class(dataset_name)
    data_obj = data_class()
    data_dir = data_obj.extract_dir
    data_path = os.path.join(data_dir, sub_path)
    return data_path
