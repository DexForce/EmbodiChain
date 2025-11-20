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
from typing import List
from pathlib import Path


from embodichain.utils import logger
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


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


class SimResources(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "embodysim_resources.zip",
            "53c054b3ae0857416dc52632eb562c12",
        )
        prefix = "SimResources"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)

    def get_ibl_path(self, name: str) -> str:
        """Get the path of the IBL resource.

        Args:
            name (str): The name of the IBL resource.

        Returns:
            str: The path to the IBL resource.
        """
        ibl_names = self.get_ibl_list()
        if name not in ibl_names:
            logger.log_error(
                f"Invalid IBL name: {name}. Available names are: {ibl_names}"
            )
        return str(Path(self.extract_dir) / "embodysim_resources" / "IBL" / name)

    def get_ibl_list(self) -> List[str]:
        """Get the names of all IBL resources.

        Returns:
            List[str]: The names of all IBL resources.
        """
        return [
            f.name
            for f in Path(self.extract_dir).glob("embodysim_resources/IBL/*")
            if f.is_dir()
        ]

    def get_material_path(self, name: str) -> str:
        """Get the path of the material resource.

        Args:
            name (str): The name of the material resource.

        Returns:
            str: The path to the material resource.
        """
        material_names = self.get_material_list()
        if name not in material_names:
            logger.log_error(
                f"Invalid material name: {name}. Available names are: {material_names}"
            )
        return str(Path(self.extract_dir) / "embodysim_resources" / "materials" / name)

    def get_material_list(self) -> List[str]:
        """Get the names of all material resources.

        Returns:
            List[str]: The names of all material resources.
        """
        return [
            f.name
            for f in Path(self.extract_dir).glob("embodysim_resources/materials/*")
            if f.is_dir()
        ]


class Dinov2WRegister(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "dinov2_register.zip",
            "3adc8ea99fc0ba09608b7a7615553d31",
        )
        prefix = "Dinov2WRegister"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class SmolVLMRegister(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "SmolVLM-500M-Instruct.zip",
            "647f5466b60f91aa4e37f47f617f4652",
        )
        prefix = "SmolVLMRegister"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PourWaterDemo(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PourWaterDemo.zip",
            "72119a9b5e39f2a22951f0a9564c2dd2",
        )
        prefix = "PourWaterDemo"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class BiPourWaterDemo(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "BiPourWaterDemo.zip",
            "651402fd80c0481a7a746f8198a71d9a",
        )
        prefix = "BiPourWaterDemo"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class RearrangementTrajGoldStandard(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "RearrangementTrajGoldStandard_0814.zip",
            "251c1a26bbffd3a65dd7d1ee2b80bf52",
        )
        prefix = "RearrangementTrajGoldStandard"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PourWaterHDF5(EmbodiChainDataset):
    # PourWater is for single arm, PourWaterDual is for dual arm
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PourWaterHDF5.zip",
            "5a5d2f30dbf8b3cddd3175a25b5c5224",
        )
        prefix = "PourWaterHDF5"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PourWaterTrajGoldStandard(EmbodiChainDataset):
    # PourWater is for single arm, PourWaterDual is for dual arm
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PourWaterTrajGoldStandard_0812.zip",
            "dc472041e41bc38d4c55a1e4a3a87ba7",
        )
        prefix = "PourWaterTrajGoldStandard"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DexsimMaterials(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "dexsim_materials.zip",
            "d74dd06f7121efc793d6e71634c95b16",
        )
        prefix = "DexsimMaterials"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DefaultPlasticMaterials(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "DefaultPlasticMaterials.zip",
            "eaed2f749cdaa621a86986ed88db69fb",
        )
        prefix = "DefaultPlasticMaterials"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CocoBackground(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "CocoBackground.zip",
            "fda82404a317281263bd5849e9eb31a1",
        )
        prefix = "CocoBackground"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Graspnet(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "graspnet.zip",
            "b69b6e49b4b71520cdc2346430ea524d",
        )
        prefix = "Graspnet"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Items(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "items.zip",
            "2050c620665b5d69cce68ffd18ad52ec",
        )
        prefix = "Items"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class VlmPhoto(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "test_photo.zip",
            "6a6572fbfedbfa9efb23ecbdf9d6cf79",
        )
        prefix = "VlmPhoto"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DeerMarked(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "deer_marked.zip",
            "e19fb95e9e87071dcec64cec16471b17",
        )
        prefix = "DeerMarked"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class LegoBig(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "legobig.zip",
            "855ed7ee4491c429d5a4013a99955850",
        )
        prefix = "LegoBig"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Lego(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "lego.zip", "2b373ed81ae5f7997a66f5fc99695bf1"
        )
        prefix = "Lego"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Lego2(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "lego2.zip",
            "4b03f393a7b85683a6925238ea94a790",
        )
        prefix = "Lego2"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class LegoHippo(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "lego_hippo.zip",
            "0d675a5b5ee74cfa97645b50a1ca3514",
        )
        prefix = "LegoHippo"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class GraspnetObjectPartial(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "graspnet_object_partial.zip",
            "72b65f21ead6aefd20c9d66fcb5a7f58",
        )
        prefix = "GraspnetObjectPartial"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class GraspnetDemoImage(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "graspnet_demo_image.zip",
            "f43ed2a9b76edd9c0d1279da70b69622",
        )
        prefix = "GraspnetDemoImage"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class SamModel(o3d.data.DownloadDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "sam_vit_b_01ec64.pth",
            "01ec64d29a2fca3f0661936605ae66f8",
        )
        prefix = "SamModel"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CardboardBox(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "CardboardBox.zip",
            "76091effabd79bb88a46669b905b1503",
        )
        prefix = "CardboardBox"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CardboardCrate(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "CardboardCrate.zip",
            "0592e38d68be84e1146727a45fb2537e",
        )
        prefix = "CardboardCrate"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class TableWare(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "tableware.zip",
            "403e340fc0e4996c002ee774f89cd236",
        )
        prefix = "TableWare"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ScannedBottle(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "ScannedBottle.zip",
            "d2b2d4deb7b463a734af099f7624b4af",
        )
        prefix = "ScannedBottle"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class MultiW1Data(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "multi_w1_demo.zip",
            "984e8fa3aa05cb36a1fd973a475183ed",
        )
        prefix = "MultiW1Data"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class GraspGeneration(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "GraspGeneration.zip",
            "52314e1043a0dbc47c48f1ffdb46d179",
        )
        prefix = "GraspGeneration"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class DecomposeInstruct(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "images_zzy_0708.zip",
            "d4f862438ba17f48e3a8f39f378905fa",
        )
        prefix = "DecomposeInstruct"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CheckCap(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "images_zzy_0723.zip",
            "85268266fa100decd63edbeb94bc9d0d",
        )
        prefix = "CheckCap"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ChainRainSec(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "lianguijie.zip",
            "2387589040a4d3f2676b622362452242",
        )
        prefix = "ChainRainSec"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ScoopIceNewEnv(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "ScoopIceNewEnv.zip",
            "e92734a9de0f64be33a11fbda0fbd3b6",
        )
        prefix = "ScoopIceNewEnv"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PourWaterTeleW1PGC(EmbodiChainDataset):
    # PourWater telecontrol hdf5 data for W1withPGC
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PourWaterTeleW1PGC.zip",
            "6b9d58c50b18b0077666ece0ba8f617c",
        )
        prefix = "PourWaterTeleW1PGC"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class PourWaterTeleW1(EmbodiChainDataset):
    # PourWater telecontrol hdf5 data for W1
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "PourWaterTeleW1.zip",
            "e4dbebf2a57a4c5714c15dea643d558d",
        )
        prefix = "PourWaterTeleW1"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class TmpACTTraining(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "TmpACTTraining.zip",
            "7b1eb7ef64099ac6dbfb69bea62363ce",
        )
        prefix = "TmpACTTraining"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Real2SimWareHouse(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "Real2SimWareHouse.zip",
            "e8b8047abc049671f24a810341c3b30b",
        )
        prefix = "Real2SimWareHouse"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Real2SimTestData(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "Real2SimTestData_2.zip",
            "97cdc1af7d17af8646bcfe6b86c74bfd",
        )
        prefix = "Real2SimTestData"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class CobotMaticTestTrajectory(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            EMBODICHAIN_DOWNLOAD_PREFIX + "CobotMaticTestTrajectory.zip",
            "763c8f744c83ed192f5a15d778704f06",
        )
        prefix = "CobotMaticTestTrajectory"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


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
