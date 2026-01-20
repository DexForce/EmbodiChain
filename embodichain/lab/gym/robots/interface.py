# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
import numpy as np
import pytorch_kinematics as pk

from abc import abstractmethod
from typing import List, Dict, Tuple, Union

from gymnasium import spaces

from embodichain.data.enum import ControlParts, EndEffector, JointType
from embodichain.lab.sim.objects import Robot
from embodichain.utils import logger
from embodichain.data.enum import JointType, EefType, ActionMode


class LearnableRobot(Robot):
    """The interface class for the learnable robot agent.

    There are three types of actions should be explained:
    - Real robot actions: The actions that the real robot can execute.
    - Control actions: The actions that the robot interacts with the policy.
    - Environment actions: The actions that the robot executes in the simulation environment.
    """

    def get_single_action_space(self) -> spaces.Dict:
        limits = self.get_joint_limits(self.uid)
        low, high = limits[:, 0], limits[:, 1]
        single_action_space = spaces.Dict(
            {
                JointType.QPOS.value: spaces.Box(low=low, high=high, dtype=np.float32),
            }
        )

        return single_action_space

    def step_env_action(self, action: Dict):
        qpos = action[JointType.QPOS.value]

        self.set_current_qpos(self.uid, qpos)
        return action

    def get_debug_xpos_dict(
        self,
    ) -> Dict[str, np.ndarray]:
        """Get the debug xpos list."""
        return {}

    def get_data_index(self, name: str, warning: bool = True) -> List[int]:
        """
        Get the data index for the control part. Subclasses must implement the index_map attribute.

        Args:
            name (str): The name of the control part.
            warning (bool, optional): Whether to log a warning if the control part is not supported. Defaults to True.

        Returns:
            List[int]: The list of indices for the control part. Returns an empty list if not found.

        Raises:
            NotImplementedError: If the subclass does not define the index_map attribute.
        """
        if not hasattr(self, "index_map"):
            raise NotImplementedError("Subclasses must define the index_map attribute.")
        if name in self.index_map:
            return self.index_map[name]
        else:
            if warning:
                logger.log_warning(f"Control part {name} is not supported.")
            return []

    def map_ee_state_to_env_actions(
        self, ee_state: np.ndarray, env_actions: np.ndarray
    ) -> np.ndarray:
        """Map the end-effector state to the environment actions of robot agent.

        Args:
            ee_state (np.ndarray): The end-effector state.
            env_actions (np.ndarray): The environment actions of the robot agent.

        Returns:
            np.ndarray: The environment actions of the robot agent.
        """
        return env_actions

    def map_real_actions_to_control_actions(self, actions: np.ndarray) -> np.ndarray:
        """Map the real robot actions to the control actions of robot agent, which
            should has the same dimension.

            The control actions should be the actions that match the articulation joint limits.

        Note:
            Real robot may have gap in the action compared to the simulation robot agent. The
            method provides a place the process the gap.

        Args:
            actions (np.ndarray): The real robot actions collected from the robot.

        Returns:
            np.ndarray: The environment actions of the robot agent.
        """
        return actions

    def map_control_actions_to_env_actions(
        self,
        actions: np.ndarray,
        env_action_dim: int,
        action_type: str = JointType.QPOS.value,
    ) -> np.ndarray:
        """Map the control actions to the environment actions of robot agent.

        Args:
            actions (np.ndarray): The control actions.
            env_action_dim (int): The dimension of the environment action space.
            action_type (str, optional): The type of action. Defaults to JointType.QPOS.value.

        Returns:
            np.ndarray: The environment actions of the robot agent.
        """
        control_index = self.get_data_index(self.uid)
        if action_type != EefType.POSE.value and actions.shape[1] != len(control_index):
            logger.log_error(
                f"The policy action dimension {actions.shape[1]} does not match the control index dimension {len(control_index)}."
            )

        length = len(actions)
        env_actions = np.zeros((length, env_action_dim))
        if action_type == JointType.QPOS.value:
            env_actions[:, control_index] = actions
        elif action_type == EefType.POSE.value:
            # TODO: the eef state is also mapped in this function, which should be separated.
            env_actions = self.map_eef_pose_to_env_qpos(actions, env_actions)
        else:
            logger.log_error(f"Invalid action type: {action_type}")

        return env_actions

    def map_env_qpos_to_eef_pose(
        self, env_qpos: np.ndarray, to_dict: bool = False, ret_mat: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Map environment joint positions to end-effector pose representation.

        Args:
            env_qpos (np.ndarray): Joint positions from the environment, shape [batch, total_dof].

        Returns:
            np.ndarray: End-effector pose, shape [batch, 18].
                [left_pos(3), left_x(3), left_y(3), right_pos(3), right_x(3), right_y(3)]
        """
        num_pose = env_qpos.shape[0]
        left_indices = self.get_joint_ids(
            ControlParts.LEFT_ARM.value + JointType.QPOS.value
        )
        right_indices = self.get_joint_ids(
            ControlParts.RIGHT_ARM.value + JointType.QPOS.value
        )

        left_env_qpos = torch.as_tensor(env_qpos[:, left_indices], dtype=torch.float32)
        right_env_qpos = torch.as_tensor(
            env_qpos[:, right_indices], dtype=torch.float32
        )

        left_ret = (
            self.pk_serial_chain[ControlParts.LEFT_ARM.value + JointType.QPOS.value]
            .forward_kinematics(left_env_qpos, end_only=True)
            .get_matrix()
        )
        right_ret = (
            self.pk_serial_chain[ControlParts.RIGHT_ARM.value + JointType.QPOS.value]
            .forward_kinematics(right_env_qpos, end_only=True)
            .get_matrix()
        )

        eef_pose = np.zeros((num_pose, 18))
        eef_pose[..., :3] = left_ret[..., :3, 3]
        eef_pose[..., 3:6] = left_ret[..., :3, 0]
        eef_pose[..., 6:9] = left_ret[..., :3, 1]
        eef_pose[..., 9:12] = right_ret[..., :3, 3]
        eef_pose[..., 12:15] = right_ret[..., :3, 0]
        eef_pose[..., 15:18] = right_ret[..., :3, 1]

        if to_dict:
            from embodichain.data.enum import EefType

            if not ret_mat:
                return {
                    ControlParts.LEFT_ARM.value + EefType.POSE.value: eef_pose[..., :9],
                    ControlParts.RIGHT_ARM.value
                    + EefType.POSE.value: eef_pose[..., 9:],
                }
            else:
                return {
                    ControlParts.LEFT_ARM.value + EefType.POSE.value: left_ret,
                    ControlParts.RIGHT_ARM.value + EefType.POSE.value: right_ret,
                }
        else:
            return eef_pose

    def map_eef_pose_to_env_qpos(
        self, eef_pose: np.ndarray, env_qpos: np.ndarray
    ) -> np.ndarray:
        """Map the end-effector pose to the environment actions.

        Args:
            eef_pose (np.ndarray): The end-effector pose.
            env_qpos (np.ndarray): The env qpos to be mapped.

        Returns:
            np.ndarray: The environment actions.
        """
        return env_qpos

    def clip_env_qpos(self, env_qpos: np.ndarray) -> np.ndarray:
        """Clip the environment qpos based on the robot joint limits.

        Args:
            env_qpos (np.ndarray): The environment qpos to be clipped.

        Returns:
            np.ndarray: The clipped environment qpos.
        """
        limits = self.get_joint_limits(self.uid)
        low, high = limits[:, 0], limits[:, 1]
        env_qpos = np.clip(env_qpos, low, high)
        return env_qpos

    @staticmethod
    def build_pk_serial_chain(**kwargs) -> Dict[str, pk.SerialChain]:
        """Build the serial chain from the URDF file.

        Args:
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            Dict[str, pk.SerialChain]: The serial chain of the robot.
        """
        return {}
