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

import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Union, List, Any, Optional, Callable
from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_node,
    tag_edge,
)

from embodichain.lab.gym.utils.misc import (
    resolve_env_params,
    mul_linear_expand,
    get_offset_pose_list,
    get_changed_pose,
)

from embodichain.lab.sim.planners.motion_generator import MotionGenerator
from embodichain.utils import logger


__all__ = ["PourWaterActionBank"]


class PourWaterActionBank(ActionBank):
    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: Optional[List] = None,
    ):
        # FIXME FIXME FIXME FIXME
        logger.log_warning(
            f"CAUTION=============================THIS FUNC generate_left_arm_aim_qpos IS WRONG!!!! PLEASE FIX IT!!!!"
        )
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["cup_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_qpos"] = left_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: valid & process qpos & fk
    def generate_right_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: Optional[List] = None,
    ):
        # FIXME FIXME FIXME FIXME
        logger.log_warning(
            f"CAUTION=============================THIS FUNC generate_right_arm_aim_qpos IS WRONG!!!! PLEASE FIX IT!!!!"
        )
        right_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["bottle_pose"][:2, 3]
                    - env.affordance_datas["right_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )
        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def compute_unoffset_for_exp(env, pose_input_output_names_changes: Dict = {}):
        env.affordance_datas["bottle_grasp_unoffset_matrix_object"] = np.eye(
            4
        )  # For the overall transform matrix calculation
        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    @staticmethod
    @tag_edge
    @tag_node
    # TODO: Got the dimension from the scope
    def execute_open(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[1.0]])]).transpose()
            else:
                action = np.ones((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(env, return_action: bool = False, **kwargs):

        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[0.0]])]).transpose()
            else:
                action = np.zeros((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        keyposes = [
            kp.cpu().numpy() if hasattr(kp, "cpu") and hasattr(kp, "numpy") else kp
            for kp in keyposes
        ]

        if all(
            np.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                f"Applying plan_trajectory to two very close qpos! Using stand_still."
            )
            keyposes = [keyposes[0]] * 2
            ret_transposed = PourWaterActionBank.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
            )

            return ret_transposed

        else:
            mo_gen = MotionGenerator(robot=env.robot, uid=agent_uid)
            ret, _ = mo_gen.create_discrete_trajectory(
                qpos_list=keyposes,
                sample_num=duration,
                qpos_seed=keyposes[0],
                is_use_current_qpos=False,
            )

            return ret.T

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        stand_still_qpos = keyposes[0]

        if (
            stand_still_qpos.shape
            != np.asarray(env.robot.get_joint_ids("left_arm")).shape
        ):
            logger.log_error(
                f"The shape of stand_still qpos is different from {agent_uid}'s setting."
            )

        if any(
            np.linalg.norm(former - latter).sum() > 1e-6
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                f"Applying stand still to two different qpos! Using the first qpos {stand_still_qpos}"
            )
            keyposes = [stand_still_qpos] * 2

        ret = np.asarray([stand_still_qpos] * duration)

        return ret.T

    @staticmethod
    @tag_edge
    def left_arm_go_back(env, duration: int):
        left_arm_monitor_qpos, left_arm_init_qpos = (
            env.affordance_datas["left_arm_monitor_qpos"],
            env.affordance_datas["left_arm_init_qpos"],
        )
        left_home_sample_num = duration
        qpos_expand_left = np.array([left_arm_monitor_qpos, left_arm_init_qpos])
        qpos_expand_left = mul_linear_expand(qpos_expand_left, [left_home_sample_num])
        ret = np.array(qpos_expand_left).T
        return ret
