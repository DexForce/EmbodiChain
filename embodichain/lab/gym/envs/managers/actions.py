# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Action terms for processing policy actions into robot control commands.

This module provides concrete implementations of :class:`ActionTerm` that convert
raw policy actions into different control formats (e.g., joint positions, velocities,
forces, or end-effector poses).

The action terms are typically used in conjunction with :class:`ActionManager` which
handles calling the appropriate term based on configuration.

Available action terms:

- :class:`DeltaQposTerm`: Delta joint position (current + scale * action)
- :class:`QposTerm`: Absolute joint position (scale * action)
- :class:`QposDenormalizedTerm`: Normalized action [-1,1] -> joint limits
- :class:`EefPoseTerm`: End-effector pose -> IK -> joint position
- :class:`QvelTerm`: Joint velocity (scale * action)
- :class:`QfTerm`: Joint force/torque (scale * action)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction
from embodichain.utils.math import matrix_from_euler, matrix_from_quat

from .action_manager import ActionTerm
from .cfg import ActionTermCfg

# Import ActionTerm from action_manager after it's defined
# This is a late import to avoid circular dependency
if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


__all__ = [
    "DefaultJointPositionTerm",
    "DeltaQposTerm",
    "EefPoseGripperTerm",
    "EefPoseTerm",
    "JointPositionGripperTerm",
    "QfTerm",
    "QposDenormalizedTerm",
    "QposNormalizedTerm",
    "QposTerm",
    "QvelTerm",
]


# ----------------------------------------------------------------------------
# Concrete ActionTerm implementations
# ----------------------------------------------------------------------------


class DeltaQposTerm(ActionTerm):
    """Delta joint position action: current_qpos + scale * action -> qpos.

    This action term adds a scaled delta to the current joint positions.
    Useful for relative position control where the policy outputs position offsets.

    Args:
        scale: Scaling factor for the action. Defaults to 1.0.

    Example:
        >>> cfg = ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1})
        >>> term = DeltaQposTerm(cfg, env)
        >>> action = torch.ones(num_envs, dof) * 2.0
        >>> result = term.process_action(action)
        >>> # result["qpos"] = current_qpos + 0.1 * action
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._scale + self._env.robot.get_qpos()


class QposTerm(ActionTerm):
    """Absolute joint position action: scale * action -> qpos.

    This action term directly uses the scaled action as target joint positions.
    Useful for absolute position control.

    Args:
        scale: Scaling factor for the action. Defaults to 1.0.

    Example:
        >>> cfg = ActionTermCfg(func=QposTerm, params={"scale": 1.0})
        >>> term = QposTerm(cfg, env)
        >>> action = torch.ones(num_envs, dof) * 0.5
        >>> result = term.process_action(action)
        >>> # result["qpos"] = 0.5 * action
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        qpos = action * self._scale
        return qpos


class QposDenormalizedTerm(ActionTerm):
    """Normalized action in [range[0], range[1]] -> denormalize to joint limits -> qpos.

    The policy outputs normalized actions in the range [range[0], range[1]] which are then
    mapped to the joint's position limits.

    The policy output is scaled by ``params.scale`` before denormalization.
    With scale=1.0 (default), action in [range[0], range[1]] maps to [low, high].
    With scale<1.0, the effective range shrinks toward the center (e.g. scale=0.5
    maps to 25%-75% of joint range). Use scale=1.0 for standard normalized control.

    Args:
        scale: Scaling factor applied before denormalization. Defaults to 1.0.
        joint_ids: List of joint IDs to apply the action to. Defaults to all active joints.
        range: The range of the normalized action. Defaults to [-1.0, 1.0].

    Example:
        >>> cfg = ActionTermCfg(func=QposDenormalizedTerm, params={"scale": 1.0})
        >>> term = QposDenormalizedTerm(cfg, env)
        >>> action = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])  # min/max per joint
        >>> result = term.process_action(action)
        >>> # Maps [-1, 1] to [qpos_limits_low, qpos_limits_high]
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)
        self._joint_ids = cfg.params.get("joint_ids", self._env.active_joint_ids)
        self._range = cfg.params.get("range", [-1.0, 1.0])

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        scaled = action * self._scale
        qpos_limits = self._env.robot.body_data.qpos_limits[0, self._joint_ids]
        low = qpos_limits[:, 0]
        high = qpos_limits[:, 1]
        scaled[:, self._joint_ids] = low + (
            scaled[:, self._joint_ids] - self._range[0]
        ) / (self._range[1] - self._range[0]) * (high - low)
        return scaled


class QposNormalizedTerm(ActionTerm):
    """Normalize action from qpos limits -> [range[0], range[1]].

    Map joint positions to a normalized range [range[0], range[1]] based on the joint limits.
    This is the usually used for post processing the output of action.

    Args:
        joint_ids: List of joint IDs to apply the action to. Defaults to all active joints.
        range: The range of the normalized action. Defaults to [0.0, 1.0].

    Example:
        >>> cfg = ActionTermCfg(func=QposNormalizedTerm)
        >>> term = QposNormalizedTerm(cfg, env)
        >>> action = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])  # min/max per joint
        >>> result = term.process_action(action)
        >>> # Maps [-1, 1] to [0, 1] based on joint limits
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._joint_ids = cfg.params.get("joint_ids", self._env.active_joint_ids)
        self._range = cfg.params.get("range", [0.0, 1.0])

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        qpos_limits = self._env.robot.body_data.qpos_limits[0, self._joint_ids]
        low = qpos_limits[:, 0]
        high = qpos_limits[:, 1]
        action[:, self._joint_ids] = (action[:, self._joint_ids] - low) / (
            high - low
        ) * (self._range[1] - self._range[0]) + self._range[0]
        return action


class EefPoseTerm(ActionTerm):
    """End-effector pose (6D or 7D) -> IK -> qpos.

    The policy outputs a target end-effector pose which is converted to joint
    positions using inverse kinematics.

    Supports two pose representations:
    - 6D: position (3) + Euler angles (3)
    - 7D: position (3) + quaternion in ``xyzw`` order (4)

    On IK failure, falls back to current_qpos for that env.
    Returns ``ik_success`` in the TensorDict so reward/observation
    can penalize or condition on IK failures.

    Args:
        scale: Scaling factor for the pose. Defaults to 1.0.
        pose_dim: Dimension of the pose (6 for Euler, 7 for quaternion). Defaults to 7.

    Example:
        >>> cfg = ActionTermCfg(func=EefPoseTerm, params={"scale": 1.0, "pose_dim": 7})
        >>> term = EefPoseTerm(cfg, env)
        >>> # 7D: position (3) + quaternion (4)
        >>> action = torch.zeros(num_envs, 7)
        >>> action[:, :3] = 0.1  # target position
        >>> action[:, 6] = 1.0   # quaternion w (xyzw identity)
        >>> result = term.process_action(action)
        >>> # result["qpos"] = IK solution
        >>> # result["ik_success"] = bool tensor indicating IK success
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)
        self._pose_dim = cfg.params.get("pose_dim", 7)  # 6 for euler, 7 for quat
        self._part_name = cfg.params.get("part_name")

    @property
    def input_key(self) -> str:
        return "eef_pose"

    @property
    def action_dim(self) -> int:
        return self._pose_dim

    def process_action(self, action: torch.Tensor) -> EnvAction:
        scaled = action * self._scale
        current_qpos = self._env.robot.get_qpos()
        batch_size = scaled.shape[0]
        target_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        if scaled.shape[-1] == 6:
            target_pose[:, :3, 3] = scaled[:, :3]
            target_pose[:, :3, :3] = matrix_from_euler(scaled[:, 3:6])
        elif scaled.shape[-1] == 7:
            target_pose[:, :3, 3] = scaled[:, :3]
            target_pose[:, :3, :3] = matrix_from_quat(scaled[:, 3:7])
        else:
            raise ValueError(
                f"EEF pose action must be 6D or 7D, got {scaled.shape[-1]}D"
            )
        # Batch IK: robot.compute_ik supports (num_envs, 4, 4) pose and (num_envs, dof) seed
        joint_seed = current_qpos
        part_joint_ids = None
        if self._part_name is not None:
            part_joint_ids = self._env.robot.get_joint_ids(
                self._part_name, remove_mimic=True
            )
            joint_seed = current_qpos[:, part_joint_ids]
        ik_kwargs = {"pose": target_pose, "joint_seed": joint_seed}
        if self._part_name is not None:
            ik_kwargs["name"] = self._part_name
        ret, qpos_ik = self._env.robot.compute_ik(**ik_kwargs)
        if part_joint_ids is None or qpos_ik.shape[-1] == current_qpos.shape[-1]:
            result_qpos = torch.where(
                ret.unsqueeze(-1).expand_as(qpos_ik), qpos_ik, current_qpos
            )
        else:
            result_qpos = current_qpos.clone()
            selected_qpos = torch.where(
                ret.unsqueeze(-1).expand_as(qpos_ik),
                qpos_ik,
                current_qpos[:, part_joint_ids],
            )
            result_qpos[:, part_joint_ids] = selected_qpos
        return TensorDict(
            {
                "qpos": result_qpos,
                "ik_success": ret,
                # Preserve the requested Cartesian command through the action
                # manager. The controller consumes only ``qpos``; dataset
                # recorders may use this field as the explicit EEF target.
                "eef_pose": scaled.clone(),
            },
            batch_size=[batch_size],
            device=self.device,
        )


class EefPoseGripperTerm(EefPoseTerm):
    """Convert a 7D Cartesian action into arm IK targets and gripper joints.

    The action is ``[x, y, z, roll, pitch, yaw, gripper]``. The first six
    values use the same absolute-pose convention as :class:`EefPoseTerm`; the
    final value is normalized from ``[-1, 1]`` to the active hand joint limits.
    """

    action_contract_representation = "eef_pose_gripper"
    """Dataset action representation emitted by this term."""

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return the flat pose-plus-gripper policy space.

        Pose coordinates are intentionally unbounded here; task-specific
        workspace limits belong to the controller/IK boundary. The gripper
        coordinate is normalized and therefore has an explicit ``[-1, 1]``
        contract.
        """
        low = np.array([-np.inf] * 6 + [-1.0], dtype=np.float32)
        high = np.array([np.inf] * 6 + [1.0], dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        if action.shape[-1] != 7:
            raise ValueError(
                f"EefPoseGripperTerm expects 7D actions, got {action.shape[-1]}D."
            )

        arm_action = action[..., :6]
        result = super().process_action(arm_action)
        hand_joint_ids = self._env.robot.get_joint_ids("hand", remove_mimic=True)
        if not hand_joint_ids:
            raise ValueError("EefPoseGripperTerm requires a robot hand control part.")

        limits = self._env.robot.body_data.qpos_limits[0, hand_joint_ids]
        gripper = action[..., 6:7].clamp(-1.0, 1.0)
        hand_qpos = limits[:, 0] + (gripper + 1.0) * 0.5 * (limits[:, 1] - limits[:, 0])
        result["qpos"][:, hand_joint_ids] = hand_qpos.expand(-1, len(hand_joint_ids))
        # The full 7D command is the requested target, including the gripper.
        result["eef_pose"] = action.clone()
        return result


class JointPositionGripperTerm(ActionTerm):
    """Map 8D joint actions to 7 arm joints and one shared gripper value."""

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return arm joint limits plus normalized shared-gripper bounds."""
        arm_ids = self._env.robot.get_joint_ids("arm", remove_mimic=True)
        limits = self._env.robot.body_data.qpos_limits[0, arm_ids]
        low = torch.cat((limits[:, 0], limits.new_tensor([-1.0]))).cpu().numpy()
        high = torch.cat((limits[:, 1], limits.new_tensor([1.0]))).cpu().numpy()
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != 8:
            raise ValueError(
                f"JointPositionGripperTerm expects 8D actions, got {action.shape[-1]}D."
            )
        qpos = self._env.robot.get_qpos().clone()
        arm_ids = self._env.robot.get_joint_ids("arm", remove_mimic=True)
        hand_ids = self._env.robot.get_joint_ids("hand", remove_mimic=True)
        qpos[:, arm_ids] = action[:, : len(arm_ids)]
        limits = self._env.robot.body_data.qpos_limits[0, hand_ids]
        gripper = action[:, 7:8].clamp(-1.0, 1.0)
        qpos[:, hand_ids] = limits[:, 0] + (gripper + 1.0) * 0.5 * (
            limits[:, 1] - limits[:, 0]
        )
        return qpos


class QvelTerm(ActionTerm):
    """Joint velocity action: scale * action -> qvel.

    This action term outputs target joint velocities.
    Useful for velocity control tasks.

    Args:
        scale: Scaling factor for the action. Defaults to 1.0.

    Example:
        >>> cfg = ActionTermCfg(func=QvelTerm, params={"scale": 0.2})
        >>> term = QvelTerm(cfg, env)
        >>> action = torch.ones(num_envs, dof)
        >>> result = term.process_action(action)
        >>> # result["qvel"] = 0.2 * action
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def input_key(self) -> str:
        return "qvel"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._scale


class QfTerm(ActionTerm):
    """Joint force/torque action: scale * action -> qf.

    This action term outputs target joint forces/torques.
    Useful for impedance control or force-based tasks.

    Args:
        scale: Scaling factor for the action. Defaults to 1.0.

    Example:
        >>> cfg = ActionTermCfg(func=QfTerm, params={"scale": 10.0})
        >>> term = QfTerm(cfg, env)
        >>> action = torch.ones(num_envs, dof)
        >>> result = term.process_action(action)
        >>> # result["qf"] = 10.0 * action
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def input_key(self) -> str:
        return "qf"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._scale


class DefaultJointPositionTerm(ActionTerm):
    """Convert policy actions to joint positions around a configured default pose.

    Parameters are explicit: ``joint_names`` defines policy order, ``offset``
    supplies the default pose (otherwise robot init_qpos is used), ``scale``
    accepts a scalar or per-joint values, and ``clip`` bounds policy actions.
    Public ``action``, ``previous_action`` and ``position_bias`` buffers let
    observations and randomization consume this term's state directly.
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        """Create the mapping and per-environment action state.

        Args:
            cfg: Term configuration with explicit mapping parameters.
            env: Environment providing the robot and active joint order.
        """
        super().__init__(cfg, env)
        names = tuple(cfg.params["joint_names"])
        self._joint_ids = torch.tensor(
            [env.robot.joint_names.index(name) for name in names],
            dtype=torch.long,
            device=env.device,
        )
        if not torch.equal(
            self._joint_ids, torch.as_tensor(env.active_joint_ids, device=env.device)
        ):
            raise ValueError("joint_names must match the active-joint order.")
        robot_default = torch.as_tensor(env.robot.cfg.init_qpos, device=env.device)[
            self._joint_ids
        ]
        self._offset = torch.as_tensor(
            cfg.params.get("offset", robot_default),
            dtype=torch.float32,
            device=env.device,
        )
        self._scale = torch.as_tensor(
            cfg.params.get("scale", 1.0), dtype=torch.float32, device=env.device
        )
        for label, value in (("offset", self._offset), ("scale", self._scale)):
            if value.ndim > 0 and value.shape != (self.action_dim,):
                raise ValueError(
                    f"{label} must be scalar or contain one value per joint."
                )
        self._clip = cfg.params.get("clip")
        self.action = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self.previous_action = torch.zeros_like(self.action)
        self.position_bias = torch.zeros_like(self.action)

    @property
    def input_key(self) -> str:
        """Robot command field produced by the term."""
        return "qpos"

    @property
    def action_dim(self) -> int:
        """Number of controlled joints."""
        return self._joint_ids.numel()

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Store policy actions and return mapped joint targets.

        Args:
            action: Batched policy actions in the configured joint order.

        Returns:
            Default-offset, scaled and bias-corrected joint positions.
        """
        if action.shape != self.action.shape:
            raise ValueError(
                f"Expected action shape {self.action.shape}, got {action.shape}."
            )
        self.previous_action.copy_(self.action)
        self.action.copy_(
            action if self._clip is None else action.clamp(-self._clip, self._clip)
        )
        return self._offset + self._scale * self.action - self.position_bias

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Clear action history for selected rows, preserving calibrated bias.

        Args:
            env_ids: Rows to reset. None selects all rows.
        """
        ids = slice(None) if env_ids is None else env_ids
        self.action[ids] = 0
        self.previous_action[ids] = 0
