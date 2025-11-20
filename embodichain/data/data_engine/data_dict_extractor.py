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

from embodichain.utils.logger import log_warning, log_error

try:
    import h5ffmpeg as hf

    has_h5ffmpeg = True
except Exception as e:
    has_h5ffmpeg = False
    log_warning("Fail to import h5ffmpeg.")

import h5py
import os
import random
import torch
import numpy as np

from functools import cached_property
from typing import Dict, Any, List, Union, Optional
from embodichain.data.enum import (
    Modality,
    PrivilegeType,
    JointType,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import StereoCamera
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import map_qpos_to_eef_pose
from embodichain.utils.utility import get_right_name
from embodichain.lab.gym.utils.misc import _data_key_to_control_part
from embodichain.utils import logger
from embodichain.data.data_engine.unified_state import (
    StateUnifier,
)
from embodichain.data.enum import (
    SUPPORTED_PROPRIO_TYPES,
    SUPPORTED_ACTION_TYPES,
    SUPPORTED_EXTRA_VISION_TYPES,
)
from tqdm import tqdm
from copy import deepcopy

SCALE_FACTOR = 4e3  # Scale factor for depth data
FAR_CLIP = 4.0  # m

DATA_FORMATS = {
    "observations": {
        Modality.IMAGES.value: {},
        Modality.GEOMAP.value: {},
        PrivilegeType.MASK.value: {},
        PrivilegeType.EXTEROCEPTION.value: {},
        Modality.STATES.value: {},
    },
    Modality.ACTIONS.value: {},
}


class CompressedVideoHDF5:
    def __init__(self, save_path: str, chunks: int = 20) -> None:
        """
        Initializes the data dictionary extractor with the specified save path and number of chunks.
        Attempts to configure video encoding settings based on the detected GPU model using the h5ffmpeg library.
        Supported GPUs include NVIDIA A800 and NVIDIA GeForce RTX 3060, with specific encoding configurations for each.
        If the GPU is unsupported or an error occurs during initialization, a warning is logged and default configuration is used.

        Args:
            save_path (str): Path where extracted data will be saved.
            chunks (int, optional): Number of chunks to split the data into. Defaults to 20.

        Raises:
            ValueError: If the detected GPU is not supported.
        """
        self.save_path = save_path
        self.chunks = chunks

        try:
            import h5ffmpeg as hf
            import torch

            name = torch.cuda.get_device_name()

            if "A800" in name or name == "NVIDIA A800-SXM4-80GB":
                self.conf = {
                    Modality.GEOMAP.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    Modality.IMAGES.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    PrivilegeType.MASK.value: hf.x264(
                        preset="veryslow", tune="ssim", crf=0
                    ),
                }
            elif "3060" in name or name == "NVIDIA GeForce RTX 3060":
                self.conf = {
                    Modality.GEOMAP.value: hf.h264_nvenc(),
                    Modality.IMAGES.value: hf.h264_nvenc(),
                    PrivilegeType.MASK.value: hf.h264_nvenc(),
                }
            elif "3090" in name or name == "NVIDIA GeForce RTX 3090":
                self.conf = {
                    Modality.GEOMAP.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    Modality.IMAGES.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    PrivilegeType.MASK.value: hf.x264(
                        preset="veryslow", tune="ssim", crf=0
                    ),
                }
            elif "4090" in name or name == "NVIDIA GeForce RTX 4090":
                self.conf = {
                    Modality.GEOMAP.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    Modality.IMAGES.value: hf.x264(
                        preset="veryfast", tune="fastdecode"
                    ),
                    PrivilegeType.MASK.value: hf.x264(
                        preset="veryslow", tune="ssim", crf=0
                    ),
                }
            else:
                raise ValueError("Unsupported GPU: {}".format(name))

        except Exception as e:
            log_warning(
                "{}. Please make sure h5ffmpeg is successfully installed.".format(e)
            )
            self.conf = {}

    @staticmethod
    def is_compressed_hdf5(data: Dict) -> bool:
        images_group = data.get("observations", {}).get(Modality.IMAGES.value, {})
        has_compressed_keys = any(
            (isinstance(k, str) and ("index" in k or "start" in k))
            for k in images_group.keys()
        )
        return has_compressed_keys

    @staticmethod
    def get_chunk_name(name: str, id: Union[int, str]) -> str:
        """
        Generates a chunk name by concatenating the given name with the provided id, separated by an underscore.
        Args:
            name (str): The base name for the chunk.
            id (Union[int, str]): The identifier to append to the name.
        Returns:
            str: The resulting chunk name in the format 'name_id'.
        """

        return name + "_{}".format(id)

    @staticmethod
    def video_save(
        f,
        chunks: int,
        data: Dict[str, np.ndarray],
        key: str,
        dtype=np.uint8,
        conf: Dict = None,
    ):
        """
        Saves video data from multiple cameras into an HDF5 file, splitting the data into chunks for efficient storage.
        Args:
            f: An open HDF5 file handle where the video data will be saved.
            data (Dict[str, np.ndarray]): Dictionary mapping camera names to their corresponding video data arrays.
            key (str): Key under "observations" group in the HDF5 file to store the video data.
            dtype (type, optional): Data type to convert the video frames to before saving (default: np.uint8).
            conf (Dict, optional): Additional configuration parameters for HDF5 dataset creation.
        Notes:
            - Video data for each camera is processed and split into the specified number of chunks.
            - Index and start datasets are created for each camera to map frame indices to chunk IDs and chunk start indices.
            - Uses CompressedVideoHDF5 utility functions for data formatting and conversion.
            - Progress is displayed using tqdm for each chunk being saved.
        """
        import h5ffmpeg as hf

        f_images = f["observations"].create_group(key)

        for cam_name in data.keys():
            data_ = data[cam_name]
            if len(data_) != 0:
                data_ = CompressedVideoHDF5.to_bhw(data_)

                if dtype == np.uint16:
                    data_ = CompressedVideoHDF5.uint16_depth(data_)
                else:
                    data_ = data_.astype(dtype)

                data_chunks = np.array_split(data_, chunks, axis=0)
                data_chunk_ids = np.arange(data_.shape[0])
                data_chunk_ids_ = np.array_split(data_chunk_ids, chunks)
                idtochunkid = np.zeros((data_.shape[0]))
                chunkid2startid = np.zeros((chunks,))
                for chunkid, temp in enumerate(data_chunk_ids_):
                    chunkid2startid[chunkid] = min(temp)
                    for tempi in temp:
                        idtochunkid[tempi] = chunkid
                _ = f_images.create_dataset(
                    CompressedVideoHDF5.get_chunk_name(cam_name, "index"),
                    data=idtochunkid,
                )
                _ = f_images.create_dataset(
                    CompressedVideoHDF5.get_chunk_name(cam_name, "start"),
                    data=chunkid2startid,
                )

                for t, data_chunk in enumerate(tqdm(data_chunks)):
                    _ = f_images.create_dataset(
                        "{}/{}".format(cam_name, t),
                        data=data_chunk,
                        chunks=data_chunk.shape,
                        **conf,
                    )

    @staticmethod
    def uint16_depth(
        data: np.ndarray, scale_factor: float = SCALE_FACTOR, far_clip: float = FAR_CLIP
    ) -> np.ndarray:
        """
        Converts a depth data array to a uint16 format after applying scaling and clipping.
        Args:
            data (np.ndarray): The input depth data as a NumPy array.
            scale_factor (float, optional): The factor by which to scale the depth data.
                Defaults to SCALE_FACTOR.
            far_clip (float, optional): The maximum depth value (far clipping plane)
                before scaling. Defaults to FAR_CLIP.
        Returns:
            np.ndarray: The scaled and clipped depth data as a NumPy array of type uint16.
        """
        return (np.clip(data * scale_factor, 0, far_clip * scale_factor)).astype(
            np.uint16
        )

    @staticmethod
    def float32_depth(
        data: np.ndarray, scale_factor: float = SCALE_FACTOR, far_clip: float = FAR_CLIP
    ) -> np.ndarray:
        """
        Converts depth data to float32 and scales it by the given scale factor.
        Args:
            data (np.ndarray): The input depth data array.
            scale_factor (float, optional): The factor by which to scale the depth values. Defaults to SCALE_FACTOR.
            far_clip (float, optional): The far clipping distance (unused in this function). Defaults to FAR_CLIP.
        Returns:
            np.ndarray: The scaled depth data as a float32 numpy array.
        """

        return data.astype(np.float32) / scale_factor

    @staticmethod
    def to_bhw(data: np.ndarray) -> np.ndarray:
        """
        Reshapes a 4D numpy array from (vdepth, height, width, channels) to (vdepth, height, width * channels).
        If the input is already a 3D array, returns it unchanged.
        Args:
            data (np.ndarray): Input array of shape (vdepth, height, width, channels) or (vdepth, height, width).
        Returns:
            np.ndarray: Reshaped array of shape (vdepth, height, width * channels) or the original array if 3D.
        Raises:
            Logs an error if the input array does not have 3 or 4 dimensions.
        """

        if len(data.shape) == 4:
            vdepth, h, w, channels = (
                data.shape[0],
                data.shape[1],
                data.shape[2],
                data.shape[3],
            )
            return data.reshape(vdepth, h, w * channels)
        elif len(data.shape) == 3:
            return data
        else:
            log_error("Unsupported data shape: {}".format(data.shape))

    @staticmethod
    def to_bhwc(data: np.ndarray):
        """
        Converts a numpy array to BHWC (Batch, Height, Width, Channels) format.
        If the input array has 3 dimensions, it reshapes the array to have a channel dimension of size 3.
        If the input array already has 4 dimensions, it returns the array unchanged.
        Otherwise, logs an error for unsupported shapes.
        Args:
            data (np.ndarray): Input numpy array to be converted.
        Returns:
            np.ndarray: Array in BHWC format.
        Raises:
            Logs an error if the input array shape is not supported.
        """

        if len(data.shape) == 3:
            vdepth, h, w = data.shape
            return data.reshape(vdepth, h, -1, 3)
        elif len(data.shape) == 4:
            return data
        else:
            log_error("Unsupported data shape: {}".format(data.shape))

    def dump(
        self,
        ret: Dict,
        video_names: List[str] = [
            Modality.IMAGES.value,
            PrivilegeType.MASK.value,
            Modality.GEOMAP.value,
        ],
        dtypes: List = [np.uint8, np.uint8, np.uint16],
    ):
        """
        Dumps the provided data into an HDF5 file, saving specific video data with
        compression and specified data types.
        Args:
            ret (Dict): The data dictionary containing observations and other metadata.
            video_names (List[str], optional): A list of video names to extract from
                the observations. Defaults to [Modality.IMAGES.value, PrivilegeType.MASK.value, Modality.GEOMAP.value].
            dtypes (List, optional): A list of data types corresponding to each video
                name. Defaults to [np.uint8, np.uint8, np.uint16].
        Raises:
            AssertionError: If the lengths of `video_names` and `dtypes` are not equal.
            RuntimeError: If the configuration (`self.conf`) is empty, indicating that
                `h5ffmpeg` is not installed or configured properly.
        Notes:
            - The method modifies the `ret` dictionary by temporarily removing the
              specified video data during the HDF5 file creation process and then
              restoring it afterward.
            - The `hdfdict.dump` function is used to save the remaining data in the
              dictionary, while the `CompressedVideoHDF5.video_save` function handles
              the saving of video data with compression.
        """

        assert len(video_names) == len(
            dtypes
        ), "Inequal length of video names {} and dtypes {}.".format(video_names, dtypes)
        import hdfdict

        if self.conf == {}:
            raise RuntimeError(
                "Please make sure h5ffmpeg is successfully installed before using `dump`."
            )

        pop_ret = {}
        for video_name, dtype in zip(video_names, dtypes):
            video_data = ret["observations"].pop(video_name)
            pop_ret[video_name] = video_data

        # Open the file once and pass the open file object to hdfdict.dump so
        # h5py doesn't try to truncate the same path while it is already open.
        with h5py.File(self.save_path, "w") as f:
            hdfdict.dump(ret, f)
            for video_name, dtype in zip(video_names, dtypes):
                CompressedVideoHDF5.video_save(
                    f,
                    self.chunks,
                    pop_ret[video_name],
                    video_name,
                    dtype=dtype,
                    conf=self.conf[video_name],
                )

        ret["observations"].update(pop_ret)

    @staticmethod
    def decode_resources(
        f: Dict,
        ret: Dict,
        name: str,
        slice_id: int,
        condition: callable,
        function: callable,
        padding: bool = True,
        chunk_id: int = None,
    ):
        """
        Decodes and processes resources from a hierarchical data structure, applying
        a condition and transformation function to the data, and optionally adding
        zero-padding.
        Args:
            f (Dict): The input data dictionary containing observations and metadata.
            ret (Dict): The output data dictionary where processed data will be stored.
            name (str): The key name under "observations" to access specific data.
            slice_id (int): The slice index used to retrieve the corresponding chunk ID.
            condition (callable): A function that takes the data as input and returns
                a boolean indicating whether the transformation function should be applied.
            function (callable): A function to transform the data if the condition is met.
            padding (bool, optional): Whether to add zero-padding to the data. Defaults to True.
            chunk_id (int, optional): The chunk ID to use instead of deriving it from the slice ID.
                Defaults to None.
        Returns:
            None: The function modifies the `ret` dictionary in place.
        """

        import time

        images = f["observations"][name]

        for cam_name in images.keys():
            if "index" in cam_name:
                continue
            if "start" in cam_name:
                continue

            start_time = time.time()
            sliceid2chunkid = images[
                CompressedVideoHDF5.get_chunk_name(cam_name, "index")
            ][:]
            chunkid = int(sliceid2chunkid[slice_id]) if chunk_id is None else chunk_id
            data_ = images[cam_name][str(chunkid)][:]
            # log_warning("".format(time.time() - start_time)
            if condition(data_):
                data_ = function(data_)

            if padding:
                chunkid2startid = images[
                    CompressedVideoHDF5.get_chunk_name(cam_name, "start")
                ][:]
                start_idx = chunkid2startid[chunkid]
                zero_padding = np.zeros_like(data_)[0:1]
                zero_padding = np.repeat(zero_padding, repeats=start_idx, axis=0)
                ret["observations"][name][cam_name] = np.concatenate(
                    [zero_padding, data_], 0
                )
            else:
                if ret["observations"][name][cam_name] is None:
                    ret["observations"][name][cam_name] = data_
                else:
                    ret["observations"][name][cam_name] = np.concatenate(
                        [ret["observations"][name][cam_name], data_], 0
                    )

    def safe_filter(self, f: Dict, slice_id: int = None) -> Dict:
        """
        Filters and processes the input data dictionary based on the configuration
        and specified slice ID.
        Args:
            f (Dict): The input data dictionary containing observations, including
                images, masks, and geomap.
            slice_id (int, optional): The specific slice ID to process. If None,
                processes all chunks. Defaults to None.
        Returns:
            Dict: The filtered and processed data dictionary with updated
            observations for images, masks, and geomap.
        Notes:
            - The method filters out camera names containing "index" or "start".
            - It initializes the return dictionary with None values for images,
              masks, and geomap for the filtered camera names.
            - Depending on the `slice_id`, it either processes all chunks or a
              specific slice using the `CompressedVideoHDF5.decode_resources`
              method.
            - The processed observations are updated in the input dictionary `f`.
        """

        if self.conf is {}:
            return f

        cam_names = []
        for cam_name in f["observations"][Modality.IMAGES.value].keys():
            if "index" in cam_name:
                continue
            if "start" in cam_name:
                continue
            cam_names.append(cam_name)

        # Only build return structure for actually present modalities, avoid errors when real data lacks mask/geomap
        present_modalities = []
        if Modality.IMAGES.value in f["observations"]:
            present_modalities.append(Modality.IMAGES.value)
        if PrivilegeType.MASK.value in f["observations"]:
            present_modalities.append(PrivilegeType.MASK.value)
        if Modality.GEOMAP.value in f["observations"]:
            present_modalities.append(Modality.GEOMAP.value)

        ret = {"observations": {}}
        for modality_key in present_modalities:
            ret["observations"][modality_key] = {
                cam_name: None for cam_name in cam_names
            }

        if slice_id == None:
            # For all chunks
            for chunk_id_ in range(self.chunks):
                if Modality.IMAGES.value in present_modalities:
                    CompressedVideoHDF5.decode_resources(
                        f,
                        ret,
                        Modality.IMAGES.value,
                        None,
                        lambda x: len(x.shape) == 3,
                        self.to_bhwc,
                        chunk_id=chunk_id_,
                        padding=False,
                    )
                if PrivilegeType.MASK.value in present_modalities:
                    CompressedVideoHDF5.decode_resources(
                        f,
                        ret,
                        PrivilegeType.MASK.value,
                        None,
                        lambda x: len(x.shape) == 3,
                        self.to_bhwc,
                        chunk_id=chunk_id_,
                        padding=False,
                    )
                if Modality.GEOMAP.value in present_modalities:
                    CompressedVideoHDF5.decode_resources(
                        f,
                        ret,
                        Modality.GEOMAP.value,
                        None,
                        lambda x: x.dtype == np.uint16 and len(x) != 0,
                        self.float32_depth,
                        chunk_id=chunk_id_,
                        padding=False,
                    )

        else:
            if Modality.IMAGES.value in present_modalities:
                CompressedVideoHDF5.decode_resources(
                    f,
                    ret,
                    Modality.IMAGES.value,
                    slice_id,
                    lambda x: len(x.shape) == 3,
                    self.to_bhwc,
                )
            if PrivilegeType.MASK.value in present_modalities:
                CompressedVideoHDF5.decode_resources(
                    f,
                    ret,
                    PrivilegeType.MASK.value,
                    slice_id,
                    lambda x: len(x.shape) == 3,
                    self.to_bhwc,
                )
            if Modality.GEOMAP.value in present_modalities:
                CompressedVideoHDF5.decode_resources(
                    f,
                    ret,
                    Modality.GEOMAP.value,
                    slice_id,
                    lambda x: x.dtype == np.uint16 and len(x) != 0,
                    self.float32_depth,
                )
        if Modality.IMAGES.value in present_modalities:
            f["observations"][Modality.IMAGES.value] = ret["observations"][
                Modality.IMAGES.value
            ]
        if PrivilegeType.MASK.value in present_modalities:
            f["observations"][PrivilegeType.MASK.value] = ret["observations"][
                PrivilegeType.MASK.value
            ]
        if Modality.GEOMAP.value in present_modalities:
            f["observations"][Modality.GEOMAP.value] = ret["observations"][
                Modality.GEOMAP.value
            ]

        return f


class ActStateStatistic:
    def __init__(self, data_dict: Dict, min_len_steps: int) -> None:
        self.data_dict = data_dict
        self.min_len_steps = min_len_steps

    def prepare_state_and_action(
        self,
    ):
        proprio = self.data_dict["observations"][Modality.STATES.value][:]
        num_steps = proprio.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < self.min_len_steps:
            return False, None
        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        proprio_delta = np.abs(proprio - proprio[0:1])
        indices = np.where(np.any(proprio_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")
        target_actions = self.data_dict[Modality.ACTIONS.value][:]
        # Parse the state and action
        state = proprio[first_idx - 1 :]
        action = target_actions[first_idx - 1 :]
        # Return the resulting sample

        return True, {Modality.STATES.value: state, Modality.ACTIONS.value: action}

    def statistic(
        self,
    ) -> Dict:
        EPS = 1e-8
        episode_cnt = 0
        state_sum = 0
        state_sum_sq = 0
        z_state_sum = 0
        z_state_sum_sq = 0
        state_cnt = 0
        nz_state_cnt = None
        state_max = None
        state_min = None
        _, episode = self.prepare_state_and_action()
        episode_cnt += 1

        states = episode[Modality.STATES.value]

        # Zero the values that are close to zero
        z_states = states.copy()
        z_states[np.abs(states) <= EPS] = 0
        # Compute the non-zero count
        if nz_state_cnt is None:
            nz_state_cnt = np.zeros(states.shape[1])
        nz_state_cnt += np.sum(np.abs(states) > EPS, axis=0)

        # Update statistics
        state_sum += np.sum(states, axis=0)
        state_sum_sq += np.sum(states**2, axis=0)
        z_state_sum += np.sum(z_states, axis=0)
        z_state_sum_sq += np.sum(z_states**2, axis=0)
        state_cnt += states.shape[0]
        if state_max is None:
            state_max = np.max(states, axis=0)
            state_min = np.min(states, axis=0)
        else:
            state_max = np.maximum(state_max, np.max(states, axis=0))
            state_min = np.minimum(state_min, np.min(states, axis=0))

        # Add one to avoid division by zero
        nz_state_cnt = np.maximum(nz_state_cnt, np.ones_like(nz_state_cnt))

        result = {
            "state_mean": (state_sum / state_cnt).tolist(),
            "state_std": np.sqrt(
                np.maximum(
                    (z_state_sum_sq / nz_state_cnt)
                    - (z_state_sum / state_cnt) ** 2 * (state_cnt / nz_state_cnt),
                    np.zeros_like(state_sum_sq),
                )
            ).tolist(),
            "state_min": state_min.tolist(),
            "state_max": state_max.tolist(),
        }

        return result


class DataDictExtractor:
    def __init__(
        self,
        env: Union[BaseEnv, EmbodiedEnv],
        save_path: str = None,
        compression_opts: int = 9,
    ):
        self.env = env
        self.save_path = save_path
        self.data = {}

        # save all supported proprio and action types.
        robot_meta_config = deepcopy(self.env.metadata["dataset"]["robot_meta"])
        robot_meta_config["observation"][
            Modality.STATES.value
        ] = SUPPORTED_PROPRIO_TYPES
        robot_meta_config[Modality.ACTIONS.value] = SUPPORTED_ACTION_TYPES

        self.state_unifier = StateUnifier(robot_meta=robot_meta_config)
        self.compression_opts = compression_opts

    @cached_property
    def robot_control_parts(self) -> List[str]:
        """Get the robot's control parts.

        Note:
            If control_parts is specified in the robot metadata, return those parts.
            Otherwise, return all control parts.

        Returns:
            List[str]: The robot's control parts.
        """
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        control_parts = robot_meta_config.get("control_parts", None)
        if control_parts is None:
            return []
        else:
            return control_parts

    def _get_arm_control_parts(self) -> List[str]:
        control_parts = self.robot_control_parts
        arm_control_parts = []
        for part in control_parts:
            if "arm" in part:
                arm_control_parts.append(part)
        return arm_control_parts

    def _has_exteroception(self) -> bool:
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        return PrivilegeType.EXTEROCEPTION.value in robot_meta_config["observation"]

    def extract(
        self,
        obs_list: List[Dict[str, Any]],
        action_list: List[Dict[str, Any]],
        data_dict: Dict = DATA_FORMATS,
        save: bool = True,
    ):
        if save:
            assert (
                self.save_path is not None
            ), "Please provide a save path for the dataset."
        data_dict = deepcopy(data_dict)

        self._init_data(data_dict)

        ret = {}
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]

        for i, (obs, action) in enumerate(zip(obs_list, action_list)):
            self._extract_vision_obs(obs, data_dict)
            self._extract_proprioception(obs, data_dict)
            self._extract_action(action, data_dict)

        action = self._collate_action(data_dict)
        proprio = self._collate_proprio(data_dict)
        robot_meta = self._collate_metainfo()

        extra_vision_config = robot_meta_config["observation"]["vision"]
        obs = {"observations": {}}
        images = self.collate_sub_anns(
            data_dict, extra_vision_config, Modality.IMAGES.value
        )
        obs["observations"].update(proprio)
        obs["observations"].update(images)

        extra_vision_names = list(
            set([name for list in extra_vision_config.values() for name in list])
        )
        for extra_vision_name in extra_vision_names:
            extra_vision_obs = self.collate_sub_anns(
                data_dict, extra_vision_config, extra_vision_name
            )
            obs["observations"].update(extra_vision_obs)

        ret.update(robot_meta)
        ret.update(obs)
        ret.update(action)

        statistics = ActStateStatistic(
            ret, self.env.metadata["dataset"]["robot_meta"]["min_len_steps"]
        ).statistic()
        ret.update(statistics)

        if save:
            if has_h5ffmpeg:
                cvhdf5 = CompressedVideoHDF5(self.save_path)
                all_video_names = [Modality.IMAGES.value] + [
                    name
                    for name in extra_vision_names
                    if name != PrivilegeType.EXTEROCEPTION.value
                ]
                all_dtypes = [
                    np.uint16 if name == Modality.GEOMAP.value else np.uint8
                    for name in all_video_names
                ]
                cvhdf5.dump(ret, video_names=all_video_names, dtypes=all_dtypes)
            else:
                logger.log_info(
                    "h5ffmpeg is not installed, saving dataset without compression."
                )
                import hdfdict

                # Open the file once and pass the file object to hdfdict.dump to
                # avoid opening/truncating the same file path twice which causes
                # "unable to truncate a file which is already open" errors on
                # some platforms and HDF5 builds.
                with h5py.File(self.save_path, "w") as f:
                    hdfdict.dump(ret, f)

        return ret

    def _init_data(self, data_dict: Dict):
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]

        for proprio_name in SUPPORTED_PROPRIO_TYPES:
            data_dict["observations"][Modality.STATES.value][proprio_name] = []
        for action_name in SUPPORTED_ACTION_TYPES:
            data_dict[Modality.ACTIONS.value][action_name] = []

        for camera_name, extra_vision_list in extra_vision_config.items():
            is_stereo = isinstance(self.env.get_sensor(camera_name), StereoCamera)

            data_dict["observations"][Modality.IMAGES.value][camera_name] = []
            if is_stereo:
                data_dict["observations"][Modality.IMAGES.value][
                    get_right_name(camera_name)
                ] = []

            for extra_vision_name in extra_vision_list:
                if extra_vision_name in SUPPORTED_EXTRA_VISION_TYPES:
                    data_dict["observations"][extra_vision_name][camera_name] = []
                else:
                    log_error(
                        f"Extra vision observation name {extra_vision_name} is not in SUPPORTED_EXTRA_VISION_TYPES {SUPPORTED_EXTRA_VISION_TYPES}, please check again."
                    )
                if is_stereo:
                    data_dict["observations"][extra_vision_name][
                        get_right_name(camera_name)
                    ] = []

    def _extract_vision_obs(self, obs: Dict[str, Any], data_dict: Dict):
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]

        for camera_name, extra_vision_list in extra_vision_config.items():
            if camera_name in obs["sensor"]:
                is_stereo = isinstance(self.env.get_sensor(camera_name), StereoCamera)

                data_dict["observations"][Modality.IMAGES.value][camera_name].append(
                    obs["sensor"][camera_name]["color"]
                    .squeeze(0)[:, :, :3]
                    .cpu()
                    .numpy()
                )
                if is_stereo:
                    # save rgb right
                    data_dict["observations"][Modality.IMAGES.value][
                        get_right_name(camera_name)
                    ].append(
                        obs["sensor"][camera_name]["color_right"]
                        .squeeze_(0)[:, :, :3]
                        .cpu()
                        .numpy()
                    )

                for extra_vision_name in extra_vision_list:
                    if extra_vision_name in SUPPORTED_EXTRA_VISION_TYPES:
                        if extra_vision_name == PrivilegeType.EXTEROCEPTION.value:
                            if is_stereo:
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(
                                    obs[extra_vision_name][camera_name]["l"]
                                    .cpu()
                                    .numpy()
                                )
                                data_dict["observations"][extra_vision_name][
                                    get_right_name(camera_name)
                                ].append(
                                    obs[extra_vision_name][camera_name]["r"]
                                    .cpu()
                                    .numpy()
                                )
                            elif camera_name in obs.get(extra_vision_name, {}):
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(
                                    obs[extra_vision_name][camera_name].cpu().numpy()
                                )
                        elif extra_vision_name == PrivilegeType.MASK.value:
                            # save semantic mask for monocular cameras
                            data_dict["observations"][extra_vision_name][
                                camera_name
                            ].append(
                                obs["sensor"][camera_name]["semantic_mask_l"]
                                .squeeze_(0)
                                .numpy()
                                .astype(np.uint8)
                            )
                            if is_stereo:
                                data_dict["observations"][extra_vision_name][
                                    get_right_name(camera_name)
                                ].append(
                                    obs["sensor"][camera_name]["semantic_mask_r"]
                                    .squeeze_(0)
                                    .numpy()
                                    .astype(np.uint8)
                                )
                        elif extra_vision_name == Modality.GEOMAP.value:
                            if not is_stereo:
                                log_error(
                                    f"Camera {camera_name} is not stereo, while '{extra_vision_name}' is in gym_config.dataset.robot_meta.vision, please check again."
                                )
                            if "depth" in obs["sensor"][camera_name]:
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(
                                    obs["sensor"][camera_name]["depth"]
                                    .squeeze_()
                                    .numpy()
                                )
                            else:
                                log_error(
                                    f"obs['sensor'][{camera_name}] has no key named 'depth' while it's required in gym_config.dataset.robot_meta.vision, please check again."
                                )
                    else:
                        log_error(
                            f"Extra vision observation name {extra_vision_name} is not in SUPPORTED_EXTRA_VISION_TYPES {SUPPORTED_EXTRA_VISION_TYPES}, please check again."
                        )
            else:
                logger.log_error(
                    f"Camera {camera_name} not found in observations, please check your sensor configuration in gym_config.json"
                )

    def _extract_action(
        self,
        action: torch.Tensor,
        data_dict: Dict,
    ):
        robot: Robot = self.env.robot

        for key in data_dict[Modality.ACTIONS.value].keys():
            part = _data_key_to_control_part(
                robot=robot,
                control_parts=self.env.metadata["dataset"]["robot_meta"].get(
                    "control_parts", []
                ),
                data_key=key,
            )
            if part is None:
                continue
            indices = robot.get_joint_ids(part, remove_mimic=True)
            data_dict[Modality.ACTIONS.value][key].append(
                action[0, indices].cpu().numpy()
                if isinstance(action, torch.Tensor)
                else action[0, indices]
            )

        eef_pose_dict = map_qpos_to_eef_pose(
            robot, action, control_parts=self._get_arm_control_parts()
        )
        for key, val in eef_pose_dict.items():
            data_dict[Modality.ACTIONS.value][key].append(
                val.squeeze_(0).cpu().numpy()
                if isinstance(val, torch.Tensor)
                else val.squeeze_(0)
            )

    def _extract_proprioception(
        self,
        obs: Dict[str, Any],
        data_dict: Dict,
    ):
        robot: Robot = self.env.robot

        qpos = obs["robot"][JointType.QPOS.value]
        for key in data_dict["observations"][Modality.STATES.value].keys():
            part = _data_key_to_control_part(
                robot=robot,
                control_parts=self.env.metadata["dataset"]["robot_meta"].get(
                    "control_parts", []
                ),
                data_key=key,
            )
            if part is None:
                continue
            indices = robot.get_joint_ids(part, remove_mimic=True)
            data_dict["observations"][Modality.STATES.value][key].append(
                qpos[0][indices].cpu().numpy()
            )

        eef_pose_dict = map_qpos_to_eef_pose(
            robot, qpos, control_parts=self._get_arm_control_parts()
        )
        for key, val in eef_pose_dict.items():
            data_dict["observations"][Modality.STATES.value][key].append(
                val.squeeze_(0).cpu().numpy()
            )

    def _collate_proprio(self, data_dict: Dict) -> Dict:
        proprio_dict = {}
        for proprio_name in self.state_unifier.proprio_meta:
            proprio = np.array(
                data_dict["observations"][Modality.STATES.value][proprio_name]
            )
            proprio_dict[proprio_name] = proprio
        proprios = self.state_unifier.fill_in_state(proprio_dict)
        return {Modality.STATES.value: proprios}

    def _collate_metainfo(
        self,
    ) -> Dict:
        meta_info = {
            "arm_dofs": self.env.metadata["dataset"]["robot_meta"].get("arm_dofs", 12),
            "observation": self.env.metadata["dataset"]["robot_meta"].get(
                "observation", {}
            ),
            "min_len_steps": self.env.metadata["dataset"]["robot_meta"].get(
                "min_len_steps", 125
            ),
        }
        return {
            "robot_meta": meta_info,
            "instruction": {
                "lang": self.env.metadata["dataset"]["instruction"].get("lang", "")
            },
        }

    def _collate_action(self, data_dict: Dict) -> Dict:
        action_data_dict = data_dict[Modality.ACTIONS.value]
        for k, v in action_data_dict.items():
            action_data_dict[k] = np.array(v)

        action_dict = {}
        action_dict.update(action_data_dict)
        action = self.state_unifier.fill_in_action(action_dict)
        return {Modality.ACTIONS.value: action}

    @staticmethod
    def collate_sub_anns(
        data_dict: Dict,
        extra_vision_config: Dict,
        key: str = Modality.IMAGES.value,
    ) -> Dict:
        ret = {key: {}}
        for camera_name in extra_vision_config:
            images_list = data_dict["observations"][key].pop(camera_name, None)
            if images_list is None:
                continue
            if len(images_list) > 0:
                ret[key][camera_name] = np.empty(
                    (len(images_list),) + images_list[0].shape,
                    dtype=images_list[0].dtype,
                )
                for idx, image in enumerate(images_list):
                    ret[key][camera_name][idx] = image
            else:
                ret[key][camera_name] = np.array([])

            del images_list
            if get_right_name(camera_name) in data_dict["observations"][key]:
                images_right_list = data_dict["observations"][key].pop(
                    get_right_name(camera_name), None
                )
                if images_right_list is None:
                    continue
                if len(images_right_list) > 0:
                    ret[key][get_right_name(camera_name)] = np.empty(
                        (len(images_right_list),) + images_right_list[0].shape,
                        dtype=images_right_list[0].dtype,
                    )
                    for idx, image in enumerate(images_right_list):
                        ret[key][get_right_name(camera_name)][idx] = image
                else:
                    ret[key][get_right_name(camera_name)] = np.array([])
                del images_right_list

        return ret


def fetch_imitation_dataset(
    env: BaseEnv,
    obs_list: List[Dict[str, Any]],
    action_list: List[Dict[str, Any]],
    id: str,
    folder_name: str,
) -> Dict:
    """
    Save imitation dataset for a single episode.

    Args:
        env (BaseEnv): Environment instance.
        obs_list (List[Dict]): List of observation dicts.
        action_list (List[Dict]): List of action dicts.
        id (str): Unique identifier for the episode.
        folder_name (str): Folder name for saving the dataset.

    Returns:
        dict: Contains data_path, id, current_episode, and extracted data.
    """
    # Get dataset save path
    dataset_path = env.metadata["dataset"].get("save_path", None)
    if dataset_path is None:
        from embodichain.data import database_demo_dir

        dataset_path = database_demo_dir

    # Create folder if first episode
    dataset_save_path = os.path.join(dataset_path, folder_name)
    if env.curr_episode == 0 and id:
        os.makedirs(dataset_save_path, exist_ok=True)

    # Check robot dof validity
    try:
        robot: Robot = env.robot
        assert (
            env.metadata["dataset"]["robot_meta"]["arm_dofs"] <= robot.dof
        ), f"Control dof {env.metadata['dataset']['robot_meta']['arm_dofs']} must be less than {robot.dof}."
    except Exception as e:
        logger.log_error(f"Robot DOF check failed: {e}")
        return None

    # Select data format
    data_format = DATA_FORMATS

    # Extract and save data
    if id is None:
        ret = DataDictExtractor(env).extract(
            obs_list, action_list, save=False, data_dict=data_format
        )
        save_path = None
    else:
        save_path = os.path.join(dataset_save_path, id + ".hdf5")
        logger.log_info(f"Save episode {env.curr_episode} to '{save_path}'")
        ret = DataDictExtractor(env, save_path).extract(
            obs_list, action_list, save=True, data_dict=data_format
        )

    # Update episode count
    env.curr_episode += 1

    # Return result dict
    return {
        "data_path": dataset_save_path,
        "id": id,
        "current_episode": env.curr_episode,
        "data": ret,
        "save_path": save_path,
    }
