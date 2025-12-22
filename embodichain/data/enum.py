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

from enum import Enum, IntEnum
import torch
import numpy as np


class SemanticMask(IntEnum):
    """
    SemanticMask is an enumeration representing different semantic regions in an image or scene.

    Attributes:
        BACKGROUND (int): Represents the background region (value: 0).
        FOREGROUND (int): Represents the foreground objects (value: 1).
        ROBOT (int): Represents the robot region (value: 2).
    """

    BACKGROUND = 0
    FOREGROUND = 1
    ROBOT = 2


class EndEffector(Enum):
    GRIPPER = "gripper"
    DEXTROUSHAND = "hand"


class EefExecute(Enum):
    OPEN = "execute_open"
    CLOSE = "execute_close"


class ControlParts(Enum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_EEF = "left_eef"
    RIGHT_EEF = "right_eef"
    HEAD = "head"
    WAIST = "waist"


class Hints(Enum):
    EEF = (
        ControlParts.LEFT_EEF.value,
        ControlParts.RIGHT_EEF.value,
        EndEffector.GRIPPER.value,
        EndEffector.DEXTROUSHAND.value,
    )
    ARM = (ControlParts.LEFT_ARM.value, ControlParts.RIGHT_ARM.value)


class Modality(Enum):
    STATES = "states"
    STATE_INDICATOR = "state_indicator"
    ACTIONS = "actions"
    ACTION_INDICATOR = "action_indicator"
    IMAGES = "images"
    LANG = "lang"
    LANG_INDICATOR = "lang_indicator"
    GEOMAP = "geomap"  # e.g., depth, point cloud, etc.
    VISION_LANGUAGE = "vision_language"  # e.g., image + lang


class JointType(Enum):
    QPOS = "qpos"


class EefType(Enum):
    POSE = "eef_pose"


class ActionMode(Enum):
    ABSOLUTE = ""
    RELATIVE = "delta_"  # This indicates the action is relative change with respect to last state.


SUPPORTED_PROPRIO_TYPES = [
    ControlParts.LEFT_ARM.value + EefType.POSE.value,
    ControlParts.RIGHT_ARM.value + EefType.POSE.value,
    ControlParts.LEFT_ARM.value + JointType.QPOS.value,
    ControlParts.RIGHT_ARM.value + JointType.QPOS.value,
    ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
    ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
    ControlParts.LEFT_EEF.value + EndEffector.GRIPPER.value,
    ControlParts.RIGHT_EEF.value + EndEffector.GRIPPER.value,
]
SUPPORTED_ACTION_TYPES = SUPPORTED_PROPRIO_TYPES + [
    ControlParts.LEFT_ARM.value + ActionMode.RELATIVE.value + JointType.QPOS.value,
    ControlParts.RIGHT_ARM.value + ActionMode.RELATIVE.value + JointType.QPOS.value,
]


class HandQposNormalizer:
    """
    A class for normalizing and denormalizing dexterous hand qpos data.
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize_hand_qpos(
        qpos_data: np.ndarray,
        key: str,
        agent=None,
        robot=None,
    ) -> np.ndarray:
        """
        Clip and normalize dexterous hand qpos data.

        Args:
            qpos_data: Raw qpos data
            key: Control part key
            agent: LearnableRobot instance (for V2 API)
            robot: Robot instance (for V3 API)

        Returns:
            Normalized qpos data in range [0, 1]
        """
        if isinstance(qpos_data, torch.Tensor):
            qpos_data = qpos_data.cpu().numpy()

        if agent is not None:
            if key not in [
                ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
                ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
            ]:
                return qpos_data
            indices = agent.get_data_index(key, warning=False)
            full_limits = agent.get_joint_limits(agent.uid)
            limits = full_limits[indices]  # shape: [num_joints, 2]
        elif robot is not None:
            if key not in [
                ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
                ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
            ]:
                if key in [ControlParts.LEFT_EEF.value, ControlParts.RIGHT_EEF.value]:
                    # Note: In V3, robot does not distinguish between GRIPPER EEF and HAND EEF in uid,
                    # _data_key_to_control_part maps both to EEF. Under current conditions, normalization
                    # will not be performed. Please confirm if this is intended.
                    pass
                return qpos_data
            indices = robot.get_joint_ids(key, remove_mimic=True)
            limits = robot.body_data.qpos_limits[0][indices]  # shape: [num_joints, 2]
        else:
            raise ValueError("Either agent or robot must be provided")

        if isinstance(limits, torch.Tensor):
            limits = limits.cpu().numpy()

        qpos_min = limits[:, 0]  # Lower limits
        qpos_max = limits[:, 1]  # Upper limits

        # Step 1: Clip to valid range
        qpos_clipped = np.clip(qpos_data, qpos_min, qpos_max)

        # Step 2: Normalize to [0, 1]
        qpos_normalized = (qpos_clipped - qpos_min) / (qpos_max - qpos_min + 1e-8)

        return qpos_normalized

    @staticmethod
    def denormalize_hand_qpos(
        normalized_qpos: torch.Tensor,
        key: str,  # "left" or "right"
        agent=None,
        robot=None,
    ) -> torch.Tensor:
        """
        Denormalize normalized dexterous hand qpos back to actual angle values

        Args:
            normalized_qpos: Normalized qpos in range [0, 1]
            key: Control part key
            robot: Robot instance

        Returns:
            Denormalized actual qpos values
        """

        if agent is not None:
            if key not in [
                ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
                ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
            ]:
                return normalized_qpos
            indices = agent.get_data_index(key, warning=False)
            full_limits = agent.get_joint_limits(agent.uid)
            limits = full_limits[indices]  # shape: [num_joints, 2]
        elif robot is not None:
            if key not in [
                ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
                ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
            ]:
                if key in [ControlParts.LEFT_EEF.value, ControlParts.RIGHT_EEF.value]:
                    # Note: In V3, robot does not distinguish between GRIPPER EEF and HAND EEF in uid,
                    # _data_key_to_control_part maps both to EEF. Under current conditions, denormalization
                    # will not be performed. Please confirm if this is intended.
                    pass
                return normalized_qpos
            indices = robot.get_joint_ids(key, remove_mimic=True)
            limits = robot.body_data.qpos_limits[0][indices]  # shape: [num_joints, 2]
        else:
            raise ValueError("Either agent or robot must be provided")

        qpos_min = limits[:, 0].cpu().numpy()  # Lower limits
        qpos_max = limits[:, 1].cpu().numpy()  # Upper limits

        if isinstance(normalized_qpos, torch.Tensor):
            normalized_qpos = normalized_qpos.cpu().numpy()

        denormalized_qpos = normalized_qpos * (qpos_max - qpos_min) + qpos_min

        return denormalized_qpos
