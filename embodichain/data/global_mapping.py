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

from embodichain.data.enum import (
    ControlParts,
    ActionMode,
    EndEffector,
    JointType,
    EefType,
    is_dual_arms,
)
from embodichain.data.global_indices import GLOBAL_INDICES
import numpy as np
from typing import List


class GlobalMapping:
    def __init__(self, dof: int):
        self_attrs = GlobalMapping.__dict__
        num_arm = 2 if is_dual_arms(dofs=dof) else 1
        single_dof = dof // num_arm
        function_dict = {}
        for k, v in self_attrs.items():
            if isinstance(v, staticmethod) and "__" not in k:
                function_dict.update(v.__func__(dof=single_dof, num_arm=num_arm))
        self.mapping_from_name_to_indices = function_dict

    @staticmethod
    def get_qpos_indices(dof: int, num_arm, **kwrags):

        return {
            ControlParts.LEFT_ARM.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES[f"left_arm_joint_{i}_pos"] for i in range(dof)
            ],
            ControlParts.RIGHT_ARM.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES[f"right_arm_joint_{i}_pos"] for i in range(dof)
            ],
            ControlParts.HEAD.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES["head_joint_{}_pos".format(i)] for i in range(2)
            ],
            ControlParts.WAIST.value + JointType.QPOS.value: [GLOBAL_INDICES["waist"]],
        }

    @staticmethod
    def get_gripper_open_state_indices(num_arm, **kwrags):
        return {
            ControlParts.LEFT_EEF.value
            + EndEffector.GRIPPER.value: [GLOBAL_INDICES["left_gripper_open"]],
            ControlParts.RIGHT_EEF.value
            + EndEffector.GRIPPER.value: [GLOBAL_INDICES["right_gripper_open"]],
        }

    @staticmethod
    def get_hand_qpos_indices(num_arm: int, hand_dof: int = 6, **kwrags):
        return {
            ControlParts.LEFT_EEF.value
            + EndEffector.DEXTROUSHAND.value: [
                GLOBAL_INDICES[f"left_hand_joint_{i}_pos"] for i in range(hand_dof)
            ],
            ControlParts.RIGHT_EEF.value
            + EndEffector.DEXTROUSHAND.value: [
                GLOBAL_INDICES[f"right_hand_joint_{i}_pos"] for i in range(hand_dof)
            ],
        }

    @staticmethod
    def get_gripper_open_vel_indices(num_arm, **kwrags):
        return {
            ControlParts.LEFT_EEF.value
            + ActionMode.RELATIVE.value
            + EndEffector.GRIPPER.value: [GLOBAL_INDICES["left_gripper_open_vel"]],
            ControlParts.RIGHT_EEF.value
            + ActionMode.RELATIVE.value
            + EndEffector.GRIPPER.value: [GLOBAL_INDICES["right_gripper_open_vel"]],
        }

    @staticmethod
    def get_delta_qpos_indices(dof: int, num_arm, **kwrags):
        return {
            ControlParts.LEFT_ARM.value
            + ActionMode.RELATIVE.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES[f"left_arm_joint_{i}_vel"] for i in range(dof)
            ],
            ControlParts.RIGHT_ARM.value
            + ActionMode.RELATIVE.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES[f"right_arm_joint_{i}_vel"] for i in range(dof)
            ],
            ControlParts.HEAD.value
            + ActionMode.RELATIVE.value
            + JointType.QPOS.value: [
                GLOBAL_INDICES["head_joint_{}_vel".format(i)] for i in range(2)
            ],
            ControlParts.WAIST.value
            + ActionMode.RELATIVE.value
            + JointType.QPOS.value: [GLOBAL_INDICES["waist_vel"]],
        }

    @staticmethod
    def get_eef_pose_indices(num_arm, **kwrags):
        return {
            ControlParts.LEFT_ARM.value
            + EefType.POSE.value: [
                GLOBAL_INDICES["left_eef_pos_x"],
                GLOBAL_INDICES["left_eef_pos_y"],
                GLOBAL_INDICES["left_eef_pos_z"],
                GLOBAL_INDICES["left_eef_angle_0"],
                GLOBAL_INDICES["left_eef_angle_1"],
                GLOBAL_INDICES["left_eef_angle_2"],
                GLOBAL_INDICES["left_eef_angle_3"],
                GLOBAL_INDICES["left_eef_angle_4"],
                GLOBAL_INDICES["left_eef_angle_5"],
            ],
            ControlParts.RIGHT_ARM.value
            + EefType.POSE.value: [
                GLOBAL_INDICES["right_eef_pos_x"],
                GLOBAL_INDICES["right_eef_pos_y"],
                GLOBAL_INDICES["right_eef_pos_z"],
                GLOBAL_INDICES["right_eef_angle_0"],
                GLOBAL_INDICES["right_eef_angle_1"],
                GLOBAL_INDICES["right_eef_angle_2"],
                GLOBAL_INDICES["right_eef_angle_3"],
                GLOBAL_INDICES["right_eef_angle_4"],
                GLOBAL_INDICES["right_eef_angle_5"],
            ],
        }

    def get_indices(self, state_meta: List[str]):
        state_indices = []

        for proprio_name in state_meta:
            state_indices += self.mapping_from_name_to_indices[proprio_name]

        return state_indices

    def ret_all_state(
        self,
    ):
        state_indices = []

        for val in self.mapping_from_name_to_indices.values():
            state_indices += val

        return state_indices
