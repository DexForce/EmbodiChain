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
    "DeltaQposTerm",
    "QposTerm",
    "QposDenormalizedTerm",
    "QposNormalizedTerm",
    "EefPoseTerm",
    "QvelTerm",
    "QfTerm",
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
    - 7D: position (3) + quaternion (4)

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
        >>> action[:, 3] = 1.0   # quaternion w
        >>> result = term.process_action(action)
        >>> # result["qpos"] = IK solution
        >>> # result["ik_success"] = bool tensor indicating IK success
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)
        self._pose_dim = cfg.params.get("pose_dim", 7)  # 6 for euler, 7 for quat

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
        # Batch IK: robot.compute_ik supports (n_envs, 4, 4) pose and (n_envs, dof) seed
        ret, qpos_ik = self._env.robot.compute_ik(
            pose=target_pose,
            joint_seed=current_qpos,
        )
        # Fallback to current_qpos where IK failed
        result_qpos = torch.where(
            ret.unsqueeze(-1).expand_as(qpos_ik), qpos_ik, current_qpos
        )
        return TensorDict(
            {"qpos": result_qpos, "ik_success": ret},
            batch_size=[batch_size],
            device=self.device,
        )


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
