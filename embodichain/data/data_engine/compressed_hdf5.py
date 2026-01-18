# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from embodichain.utils.logger import log_warning, log_error

try:
    import h5ffmpeg as hf

    has_h5ffmpeg = True
except Exception as e:
    has_h5ffmpeg = False
    log_warning("Fail to import h5ffmpeg.")

import h5py
import numpy as np

from typing import Dict, Any, List, Union, Optional
from embodichain.data.enum import (
    Modality,
    PrivilegeType,
)
from tqdm import tqdm

SCALE_FACTOR = 4e3  # Scale factor for depth data
FAR_CLIP = 4.0  # m


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
            elif "Orin" in name:
                # FIXME: temporary solution for Orin GPU. Need to test and adjust parameters later for nvenc encoder.
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
