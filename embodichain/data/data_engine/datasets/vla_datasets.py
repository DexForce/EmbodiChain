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
import fnmatch
from embodichain.utils.logger import log_warning, log_info

try:
    import h5ffmpeg as hf
except Exception as e:
    log_warning("Fail to import h5ffmpeg.")
import h5py
import numpy as np
from typing import Dict, Callable, List, Tuple
from embodichain.utils.utility import get_right_name, pad_to_chunk, convert_bytes
from embodichain.utils.logger import log_warning, log_info
from embodichain.data.enum import Proprioception, Image, Exteroception, ModalInput
from copy import deepcopy
from typing import Dict
from embodichain.data.enum import (
    Modality,
    PrivilegeType,
    ActionMode,
    JointType,
    EefType,
    CameraName,
    TeleoperationData,
)
from embodichain.data.global_mapping import GlobalMapping
from scipy.ndimage import gaussian_filter1d
from embodichain.data.data_engine.unified_state import ActionIndicesGenerator


class VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        chunk_size: int,
        state: List,
        output: List,
        img_history_size: int,
        state_history_len: int,
        precomp_lang_embed: bool = True,
        online_config: Dict = None,
        camera_used: List[str] = None,
        indices_generator=None,
    ) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode

        self.precomp_lang_embed = precomp_lang_embed
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.state = state
        self.output = output
        self.img_history_size = img_history_size
        self.state_history_len = state_history_len
        self.online_config = online_config
        self.camera_used = camera_used
        self.indices_generator = indices_generator
        if self.camera_used is not None:
            for cam in CameraName:
                if cam.value not in camera_used:
                    log_warning(
                        "{} does not exist in {}".format(cam.value, camera_used)
                    )

        if self.online_config is not None:
            from embodichain.data.data_engine.online.engine import OnlineEngine

            log_info("Init online vla dataset.", color="purple")
            self.engine = OnlineEngine(**self.online_config)
            self.DATASET_NAME = "online_whatever"
        else:
            log_info("Init offline vla dataset.", color="purple")
            self.engine = None
            self.data_path = data_path
            assert os.path.exists(self.data_path), "{} does not exist.".format(
                self.data_path
            )
            if os.path.isabs(self.data_path) is False:
                self.data_path = os.path.join(os.getcwd(), self.data_path)
            self.DATASET_NAME = os.path.basename(self.data_path)
            self.file_paths = []
            for root, _, files in os.walk(self.data_path):
                for filename in fnmatch.filter(files, "*.hdf5"):
                    file_path = os.path.join(root, filename)
                    self.file_paths.append(file_path)
            log_info(
                f"Init dataset with size of: {len(self.file_paths)}", color="purple"
            )

    def update_data_size(self):
        """Interface for update validation dataset size generated on the fly."""
        self.file_paths = []
        for root, _, files in os.walk(self.data_path):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        log_info(f"Update dataset with size of: {len(self.file_paths)}", color="purple")

    def __len__(self):
        return (
            len(self.file_paths)
            if self.online_config is None
            else np.maximum(self.engine.episode_limit, self.batch_size)
        )

    def get_item(self, index: int = None, chunk_size: int = None):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        while True:
            if self.online_config is None:
                # offline
                if index is None:
                    file_path = np.random.choice(self.file_paths)
                else:
                    file_path = self.file_paths[index]
                valid, sample = self.parse_hdf5_file(file_path, chunk_size)
            else:
                data_dict = self.engine.sample_data()
                valid, sample = self.parse_dict(data_dict, chunk_size)

            if valid:
                return sample
            else:
                if self.online_config is None:
                    index = np.random.randint(0, len(self.file_paths))

    @staticmethod
    def parse_exteroception(
        file: Dict,
        step_id: int,
        chunk_size: int,
        camera_used: List[str] = [],
    ) -> Exteroception:
        exteroception = []
        for cam in camera_used:
            exteroception_full = file["observations"][
                PrivilegeType.EXTEROCEPTION.value
            ][cam]
            exteroception.append(exteroception_full[step_id : step_id + chunk_size])

        exteroception = np.concatenate(exteroception, 1)
        _, cs, kn, _ = exteroception.shape
        exteroception = pad_to_chunk(exteroception, chunk_size)
        return Exteroception(
            data=exteroception.reshape(chunk_size, cs, kn, 2).transpose(
                1, 0, 2, 3
            )  # cs, chunk_size, kn, 2
        )

    @staticmethod
    # Parse the images
    def parse_img(
        file: Dict,
        step_id: int,
        first_idx: int,
        cam: str,
        chunk_size: int,
        key: str = Modality.IMAGES.value,
        camera_used: List[str] = [],
        np_ops: Callable = lambda x: x,
    ) -> Image:
        valid_len = min(step_id - (first_idx - 1) + 1, chunk_size)
        cam_mask = np.array([False] * (chunk_size - valid_len) + [True] * valid_len)
        if cam in camera_used:
            temp = file["observations"][key][cam][0]
            imgs = np.zeros((valid_len,) + temp.shape, dtype=temp.dtype)
            for t, i in enumerate(range(max(step_id - chunk_size + 1, 0), step_id + 1)):
                img = file["observations"][key][cam][i]
                imgs[t] = img
            imgs = np_ops(imgs)
            imgs = pad_to_chunk(imgs, chunk_size=chunk_size)
            mask = cam_mask.copy()
        else:
            imgs = np.zeros((chunk_size, 0, 0, 0))
            mask = np.zeros((chunk_size,), dtype=bool)
        return Image(data=imgs, mask=mask, name=cam)

    def parse_hdf5_file(self, file_path, chunk_size: int) -> Dict[str, ModalInput]:
        import hdfdict
        from embodichain.data.data_engine.data_dict_extractor import (
            CompressedVideoHDF5,
        )

        with h5py.File(file_path, "r") as f:
            data = hdfdict.load(f)
            keyname = (
                JointType.QPOS.value
                if VLADataset.is_real_datasets(data)
                else Modality.STATES.value
            )
            step_id = VLADataset.random_step_id(data, chunk_size, keyname)
            if not VLADataset.is_real_datasets(data):
                data = CompressedVideoHDF5(file_path, chunks=None).safe_filter(
                    data, step_id
                )
            else:
                # Real data: if compressed structure is detected (containing *_index/*_start), also perform decoding filtering
                try:
                    if CompressedVideoHDF5.is_compressed_hdf5(data):
                        data = CompressedVideoHDF5(file_path, chunks=None).safe_filter(
                            data, step_id
                        )
                except Exception:
                    pass

            ret = self.parse_dict(data, chunk_size, step_id)

        return ret

    @staticmethod
    def random_step_id(
        f: Dict, chunk_size: int, key: str = Modality.STATES.value
    ) -> int:
        obs = f["observations"]
        proprio = obs[key][:]
        num_steps = proprio.shape[0]
        # We randomly sample a timestep
        first_idx = 1
        step_id = np.random.randint(
            first_idx, np.maximum(first_idx + 1, num_steps - 1 - chunk_size)
        )
        return step_id

    @staticmethod
    def is_real_datasets(f: Dict):
        return "robot_meta" not in f.keys()

    def parse_dict(
        self, f: Dict, chunk_size: int, step_id: int = None
    ) -> Dict[str, ModalInput]:
        if not VLADataset.is_real_datasets(f):
            log_warning("Using simulation hdf5 datasets.")
            return self.parse_sim_dict(f, chunk_size, step_id)
        else:
            log_warning("Using real world offline hdf5 datasets.")
            return self.parse_real_dict(f, chunk_size, step_id)

    def parse_real_dict(
        self, f: Dict, chunk_size: int, step_id: int = None
    ) -> Dict[str, ModalInput]:

        from embodichain.data.data_engine.unified_state import (
            StateUnifier,
        )
        from embodichain.data.enum import (
            ControlParts,
            EndEffector,
            JointType,
        )

        if step_id is None:
            step_id = VLADataset.random_step_id(f, chunk_size, "qpos")
        obs = f["observations"]
        first_idx = 1
        proprio = obs["qpos"][:]
        num_steps = proprio.shape[0]
        camera_used_in_real = list(obs[Modality.IMAGES.value].keys())
        camera_used_from_real_to_dualsys = {
            "cam_hand_left": CameraName.LEFT_WRIST.value,
            "cam_hand_right": CameraName.RIGHT_WRIST.value,
            "cam_high_left": CameraName.HEAD.value,
        }
        camera_used_from_dualsys_to_real = {
            val: key for key, val in camera_used_from_real_to_dualsys.items()
        }
        # Now assume it is from W1.
        camera_used = [
            camera_used_from_real_to_dualsys[cam]
            for cam in camera_used_in_real
            if cam in camera_used_from_real_to_dualsys
        ]

        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "camera_used": camera_used,
            "instruction": "",
        }
        # save all supported proprio and action types.
        robot_meta_config = {"arm_dofs": 14, "observation": {}}

        REAL_SUPPORTED_PROPRIO_TYPES = [
            ControlParts.LEFT_ARM.value + JointType.QPOS.value,
            ControlParts.RIGHT_ARM.value + JointType.QPOS.value,
            ControlParts.HEAD.value + JointType.QPOS.value,
            ControlParts.WAIST.value + JointType.QPOS.value,
            ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
            ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
        ]
        REAL_SUPPORTED_ACTION_TYPES = REAL_SUPPORTED_PROPRIO_TYPES

        robot_meta_config["observation"][
            Modality.STATES.value
        ] = REAL_SUPPORTED_PROPRIO_TYPES
        robot_meta_config[Modality.ACTIONS.value] = REAL_SUPPORTED_ACTION_TYPES
        state_unifier = StateUnifier(robot_meta=robot_meta_config)

        qpos_index_dict = {
            ControlParts.LEFT_ARM.value
            + JointType.QPOS.value: TeleoperationData.LEFT_ARM_QPOS_INDICES.value,
            ControlParts.RIGHT_ARM.value
            + JointType.QPOS.value: TeleoperationData.RIGHT_ARM_QPOS_INDICES.value,
            ControlParts.LEFT_EEF.value
            + EndEffector.DEXTROUSHAND.value: TeleoperationData.LEFT_EEF_DEXTROUSHAND_INDICES.value,
            ControlParts.RIGHT_EEF.value
            + EndEffector.DEXTROUSHAND.value: TeleoperationData.RIGHT_EEF_DEXTROUSHAND_INDICES.value,
            ControlParts.HEAD.value
            + JointType.QPOS.value: TeleoperationData.HEAD_QPOS_INDICES.value,
            ControlParts.WAIST.value
            + JointType.QPOS.value: TeleoperationData.WAIST_QPOS_INDICES.value,
        }
        qpos_dict = {}
        for key, indices in qpos_index_dict.items():
            qpos_dict[key] = proprio[:, indices]

        actions = state_unifier.fill_in_action(qpos_dict)
        proprio = state_unifier.fill_in_state(qpos_dict)
        parse_dict = self.parse_core(proprio, actions, step_id, chunk_size)
        parse_dict.update({"meta": meta})
        for cam in camera_used:
            parse_dict[cam] = VLADataset.parse_img(
                f,
                step_id,
                first_idx,
                camera_used_from_dualsys_to_real[cam],
                self.img_history_size,
                Modality.IMAGES.value,
                camera_used=camera_used_from_dualsys_to_real[cam],
            )
        return True, parse_dict

    def parse_sim_dict(
        self, f: Dict, chunk_size: int, step_id: int = None
    ) -> Dict[str, ModalInput]:

        if step_id is None:
            step_id = VLADataset.random_step_id(f, chunk_size)

        obs = f["observations"]
        metadata = dict(f["robot_meta"])
        first_idx = 1

        proprio = obs[Modality.STATES.value][:]
        num_steps = proprio.shape[0]
        min_len_step = metadata["min_len_steps"]
        # [Optional] We drop too-short episode
        if num_steps < min_len_step:
            return False, None

        # We randomly sample a timestep

        camera_used = (
            convert_bytes(list(metadata["observation"]["vision"].keys()))
            if self.camera_used is None
            else self.camera_used
        )

        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": "",
            "camera_used": camera_used,
        }

        assert (
            self.indices_generator.dof == metadata["arm_dofs"]
        ), "Train dof {} but dataset dof {}.".format(
            self.indices_generator.dof, metadata["arm_dofs"]
        )
        parse_dict = self.parse_core(
            proprio, f[Modality.ACTIONS.value], step_id, chunk_size
        )
        parse_dict.update({"meta": meta})

        for cam in camera_used:
            cam_r = get_right_name(cam)
            if cam_r in obs[Modality.IMAGES.value] and cam_r not in camera_used:
                # insert camera name after cam
                camera_used.insert(camera_used.index(cam) + 1, cam_r)

        for cam in camera_used:
            parse_dict[cam] = VLADataset.parse_img(
                f,
                step_id,
                first_idx,
                cam,
                self.img_history_size,
                Modality.IMAGES.value,
                camera_used=camera_used,
            )

        if PrivilegeType.MASK.value in obs:
            parse_dict[
                cam + "_{}".format(PrivilegeType.MASK.value)
            ] = VLADataset.parse_img(
                f,
                step_id,
                first_idx,
                cam,
                self.img_history_size,
                PrivilegeType.MASK.value,
                camera_used=camera_used,
            )
        if PrivilegeType.EXTEROCEPTION.value in obs:
            if obs[PrivilegeType.EXTEROCEPTION.value][camera_used[0]].shape[0] != 0:
                parse_dict[
                    PrivilegeType.EXTEROCEPTION.value
                ] = VLADataset.parse_exteroception(
                    f,
                    step_id,
                    chunk_size,
                    camera_used=camera_used,
                )

        if Modality.GEOMAP.value in obs:
            if (
                hasattr(obs[Modality.GEOMAP.value][camera_used[0]], "shape")
                and obs[Modality.GEOMAP.value][camera_used[0]].shape[0] != 0
            ):
                parse_dict[Modality.GEOMAP.value] = VLADataset.parse_img(
                    f,
                    step_id,
                    first_idx,
                    CameraName.HEAD.value,
                    self.img_history_size,
                    Modality.GEOMAP.value,
                    camera_used=camera_used,
                    np_ops=lambda x: np.tile(np.expand_dims(x, -1), [1, 1, 1, 3]),
                )

        # Return the resulting sample
        # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
        # E.g., return np.zeros((self.img_history_size, 0, 0, 0)) for the key "cam_left_wrist",
        # if the left-wrist camera is unavailable on your robot
        return True, parse_dict

    def parse_core(
        self, proprio: np.ndarray, actions: np.ndarray, step_id: int, chunk_size: int
    ):
        # Parse the state and action
        state = proprio[np.maximum(step_id - self.state_history_len, 0) : step_id]
        state = np.concatenate(
            [np.tile(state[0:1], [self.state_history_len - state.shape[0], 1]), state],
            0,
        )
        self.indices_generator: ActionIndicesGenerator
        global_mapping = self.indices_generator.global_mapping
        state_indices = global_mapping.get_indices(
            convert_bytes(self.state),
        )
        state_indicator = np.zeros_like(state, dtype=np.int8)
        state_indicator[:, state_indices] = 1
        state *= state_indicator
        proprio *= state_indicator[0:1]
        state_std = np.std(proprio, axis=0)
        state_mean = np.mean(proprio, axis=0)
        state_norm = np.sqrt(np.mean(proprio**2, axis=0))
        action_indices = self.indices_generator.get(
            self.output,
        )
        actions = deepcopy(actions[step_id : step_id + chunk_size])
        # FIXME: handness injection
        delta_qpos_indices = self.indices_generator.get_all_delta_qpos()
        qpos_indices = self.indices_generator.get_all_qpos()
        # NOTE: Ops `cumsum` equal to action[:horizon]-action[0:1].
        # TODO: action = action_chunk - current_obs.
        actions[:, delta_qpos_indices] = (
            actions[:, qpos_indices] - state[-1:, qpos_indices]
        )

        actions = pad_to_chunk(actions, chunk_size=chunk_size)
        action_indicator = np.zeros_like(actions, dtype=np.int8)
        action_indicator[:, action_indices] = 1
        actions *= action_indicator[0:1]

        parse_dict = {
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            Modality.STATES.value: Proprioception(data=state, mask=state_indicator),
            Modality.ACTIONS.value: Proprioception(data=actions, mask=action_indicator),
            PrivilegeType.PROGRESS.value: step_id / proprio.shape[0],
        }
        return parse_dict
