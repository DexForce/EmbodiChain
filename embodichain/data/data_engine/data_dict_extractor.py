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
import os
import random
import torch
import numpy as np

from functools import cached_property
from typing import Dict, Any, List, Union, Optional
from embodichain.data.enum import (
    HandQposNormalizer,
    Modality,
    PrivilegeType,
    JointType,
    ActionMode,
    EefType,
    EndEffector,
    ControlParts,
    ArmName,
)
from embodichain.data.global_mapping import GlobalMapping
from embodichain.lab.sim.sensors import StereoCamera
from embodichain.lab.sim.objects import Robot
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import map_qpos_to_eef_pose
from embodichain.utils.utility import get_right_name
from embodichain.lab.gym.robots.interface import LearnableRobot
from embodichain.lab.gym.utils.misc import is_binocularcam, _data_key_to_control_part
from embodichain.utils import logger
from embodichain.data.data_engine.indices_unifier import (
    StateUnifier,
)
from embodichain.data.data_engine.compressed_hdf5 import CompressedVideoHDF5
from embodichain.data.enum import (
    SUPPORTED_PROPRIO_TYPES,
    SUPPORTED_ACTION_TYPES,
    SUPPORTED_EXTRA_VISION_TYPES,
)
from copy import deepcopy
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    get_control_part_joint_ids,
)

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
        self.filtered_action_types = []
        self.filtered_proprio_types = []

        # First check if control_parts exists and is non-empty. Only filter if valid, else use the original types.
        control_parts = self.env.metadata["dataset"]["robot_meta"].get(
            "control_parts", None
        )
        if control_parts and len(control_parts) > 0:
            control_parts_set = set(control_parts)
            self.filtered_proprio_types = [
                proprio_name
                for proprio_name in SUPPORTED_PROPRIO_TYPES
                if any(part in proprio_name for part in control_parts_set)
            ]
            self.filtered_action_types = [
                action_name
                for action_name in SUPPORTED_ACTION_TYPES
                if any(part in action_name for part in control_parts_set)
            ]

        if (
            len(self.filtered_proprio_types) == 0
            or len(self.filtered_action_types) == 0
        ):
            log_warning(
                "No control parts found in the robot metadata. Using all supported proprio and action types."
            )
            self.filtered_proprio_types = list(SUPPORTED_PROPRIO_TYPES)
            self.filtered_action_types = list(SUPPORTED_ACTION_TYPES)

        # save all supported proprio and action types.
        robot_meta_config = deepcopy(self.env.metadata["dataset"]["robot_meta"])
        robot_meta_config["observation"][
            Modality.STATES.value
        ] = self.filtered_proprio_types
        robot_meta_config[Modality.ACTIONS.value] = self.filtered_action_types

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
            log_warning(
                "Please make sure you have configurated the control parts. This branch may cause underlying error for training data."
            )
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

        if isinstance(self.env, BaseEnv):
            for i, (obs, action) in enumerate(zip(obs_list, action_list)):
                self._extract_vision_obs(obs, data_dict)
                self._extract_proprioception(obs, data_dict)
                self._extract_action(action, data_dict)
            action = self._collate_action(data_dict)
            proprio = self._collate_proprio(data_dict)
        else:
            for i, (obs, action) in enumerate(zip(obs_list, action_list)):
                self._extract_vision_obs_v2(obs, data_dict)
                self._extract_proprioception_v2(obs, data_dict)
                self._extract_action_v2(action, data_dict)
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

        for proprio_name in self.filtered_proprio_types:
            data_dict["observations"][Modality.STATES.value][proprio_name] = []
        for action_name in self.filtered_action_types:
            data_dict[Modality.ACTIONS.value][action_name] = []

        for camera_name, extra_vision_list in extra_vision_config.items():
            is_stereo = is_binocularcam(self.env.get_sensor(camera_name))

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
                is_stereo = is_binocularcam(self.env.get_sensor(camera_name))

                data_dict["observations"][Modality.IMAGES.value][camera_name].append(
                    obs["sensor"][camera_name]["rgb"]
                )
                if is_stereo:
                    # save rgb right
                    data_dict["observations"][Modality.IMAGES.value][
                        get_right_name(camera_name)
                    ].append(obs["sensor"][camera_name]["rgb_right"])

                for extra_vision_name in extra_vision_list:
                    if extra_vision_name in SUPPORTED_EXTRA_VISION_TYPES:
                        if extra_vision_name == PrivilegeType.EXTEROCEPTION.value:
                            if is_stereo:
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(obs[extra_vision_name][camera_name]["l"])
                                data_dict["observations"][extra_vision_name][
                                    get_right_name(camera_name)
                                ].append(obs[extra_vision_name][camera_name]["r"])
                            elif camera_name in obs.get(extra_vision_name, {}):
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(obs[extra_vision_name][camera_name])
                        elif extra_vision_name == PrivilegeType.MASK.value:
                            # save semantic mask for monocular cameras
                            data_dict["observations"][extra_vision_name][
                                camera_name
                            ].append(
                                obs["sensor"][camera_name]["semantic_mask_l"].astype(
                                    np.uint8
                                )
                            )
                            if is_stereo:
                                data_dict["observations"][extra_vision_name][
                                    get_right_name(camera_name)
                                ].append(
                                    obs["sensor"][camera_name][
                                        "semantic_mask_r"
                                    ].astype(np.uint8),
                                )
                        elif extra_vision_name == Modality.GEOMAP.value:
                            if not is_stereo:
                                log_error(
                                    f"Camera {camera_name} is not stereo, while '{extra_vision_name}' is in gym_config.dataset.robot_meta.vision, please check again."
                                )
                            if "depth" in obs["sensor"][camera_name]:
                                data_dict["observations"][extra_vision_name][
                                    camera_name
                                ].append(obs["sensor"][camera_name]["depth"])
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

    def _extract_vision_obs_v2(self, obs: Dict[str, Any], data_dict: Dict):
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]

        for camera_name, extra_vision_list in extra_vision_config.items():
            if camera_name in obs["sensor"]:
                is_stereo = is_binocularcam(self.env.get_sensor(camera_name))

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
        action: Dict[str, Any],
        data_dict: Dict,
    ):

        agent: LearnableRobot = self.env.get_agent()
        # extract qpos.
        for key in data_dict[Modality.ACTIONS.value].keys():
            indices = agent.get_data_index(key, warning=False)
            if len(indices) > 0:
                action_data = action[JointType.QPOS.value][indices].copy()
                action_data = HandQposNormalizer.normalize_hand_qpos(
                    action_data, key, agent=agent
                )
                data_dict[Modality.ACTIONS.value][key].append(action_data)
        qpos = action[JointType.QPOS.value]
        action_eef_pose_dict = agent.map_env_qpos_to_eef_pose(
            np.array([qpos]), to_dict=True
        )
        for key, val in action_eef_pose_dict.items():
            data_dict[Modality.ACTIONS.value][key].append(val[0])

    def _extract_action_v2(
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
            indices = get_control_part_joint_ids(self.env, key)
            qpos_data = (
                action[0, indices].cpu().numpy()
                if isinstance(action, torch.Tensor)
                else action[0, indices]
            )
            qpos_data = HandQposNormalizer.normalize_hand_qpos(
                qpos_data, part, robot=robot
            )
            data_dict[Modality.ACTIONS.value][key].append(qpos_data)

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
        agent: LearnableRobot = self.env.get_agent()
        # extract qpos.
        qpos = obs["agent"][agent.uid][JointType.QPOS.value]
        for key in data_dict["observations"][Modality.STATES.value].keys():
            indices = agent.get_data_index(key, warning=False)
            if len(indices) > 0:
                qpos_data = qpos[
                    indices
                ].copy()  # Deep copy to avoid modifying original data
                qpos_data = HandQposNormalizer.normalize_hand_qpos(
                    qpos_data, key, agent=agent
                )
                data_dict["observations"][Modality.STATES.value][key].append(qpos_data)

        eef_pose_dict: Dict = agent.map_env_qpos_to_eef_pose(
            np.array([qpos]), to_dict=True
        )
        for key, val in eef_pose_dict.items():
            data_dict["observations"][Modality.STATES.value][key].append(val[0])

    def _extract_proprioception_v2(
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
            indices = get_control_part_joint_ids(self.env, key)
            qpos_data = qpos[0][indices].cpu().numpy()
            qpos_data = HandQposNormalizer.normalize_hand_qpos(
                qpos_data, part, robot=robot
            )
            data_dict["observations"][Modality.STATES.value][key].append(qpos_data)

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
    save: bool = True,
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
        if isinstance(env, BaseEnv):
            agent: LearnableRobot = env.get_agent()
            max_dofs = len(agent.get_data_index(agent.uid))
            assert (
                env.metadata["dataset"]["robot_meta"]["arm_dofs"] <= max_dofs
            ), f"Control dof {env.metadata['dataset']['robot_meta']['arm_dofs']} must be less than {max_dofs}."
        else:
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
            obs_list, action_list, save=save, data_dict=data_format
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
