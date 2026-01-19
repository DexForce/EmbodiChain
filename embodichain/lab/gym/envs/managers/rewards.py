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

"""Common reward functors for reinforcement learning tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def reward_from_obs(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    obs_key: str = "robot/qpos",
    target_value: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward based on observation values."""
    # Parse nested keys (e.g., "robot/qpos")
    keys = obs_key.split("/")
    value = obs
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return torch.zeros(env.num_envs, device=env.device)

    # Compute distance to target
    if isinstance(value, torch.Tensor):
        if value.dim() > 1:
            # Multiple values, compute norm
            distance = torch.norm(value - target_value, dim=-1)
        else:
            distance = torch.abs(value - target_value)
        reward = -scale * distance
    else:
        reward = torch.zeros(env.num_envs, device=env.device)

    return reward


def distance_between_objects(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_entity_cfg: SceneEntityCfg = None,
    exponential: bool = False,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Reward based on distance between two entities."""
    # get source entity position
    source_obj = env.sim[source_entity_cfg.uid]
    if hasattr(source_obj, "get_body_pose"):
        source_pos = source_obj.get_body_pose(body_ids=source_entity_cfg.body_ids)[
            :, :3, 3
        ]
    elif hasattr(source_obj, "get_local_pose"):
        source_pos = source_obj.get_local_pose(to_matrix=True)[:, :3, 3]
    else:
        raise ValueError(
            f"Entity '{source_entity_cfg.uid}' does not support position query."
        )

    # get target entity position
    target_obj = env.sim[target_entity_cfg.uid]
    if hasattr(target_obj, "get_body_pose"):
        target_pos = target_obj.get_body_pose(body_ids=target_entity_cfg.body_ids)[
            :, :3, 3
        ]
    elif hasattr(target_obj, "get_local_pose"):
        target_pos = target_obj.get_local_pose(to_matrix=True)[:, :3, 3]
    else:
        raise ValueError(
            f"Entity '{target_entity_cfg.uid}' does not support position query."
        )

    # compute distance
    distance = torch.norm(source_pos - target_pos, dim=-1)

    # compute reward
    if exponential:
        # exponential reward: exp(-distance^2 / (2 * sigma^2))
        reward = torch.exp(-(distance**2) / (2 * sigma**2))
    else:
        # negative distance reward
        reward = -distance

    return reward


def joint_velocity_penalty(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    robot_uid: str = "robot",
    joint_ids: slice | list[int] = slice(None),
) -> torch.Tensor:
    """Penalize large joint velocities."""
    robot = env.sim[robot_uid]

    # get joint velocities
    qvel = robot.body_data.qvel[:, joint_ids]

    # compute L2 norm of joint velocities
    velocity_norm = torch.norm(qvel, dim=-1)

    # negative penalty (higher velocity = more negative reward)
    return -velocity_norm


def action_smoothness_penalty(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
) -> torch.Tensor:
    """Penalize large changes in action between steps."""
    # compute difference between current and previous action
    if hasattr(env, "_prev_actions"):
        action_diff = action - env._prev_actions
        penalty = -torch.norm(action_diff, dim=-1)
    else:
        # no previous action, no penalty
        penalty = torch.zeros(env.num_envs, device=env.device)

    # store current action for next step
    env._prev_actions = action.clone()

    return penalty


def joint_limit_penalty(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    robot_uid: str = "robot",
    joint_ids: slice | list[int] = slice(None),
    margin: float = 0.1,
) -> torch.Tensor:
    """Penalize joints approaching their limits."""
    robot = env.sim[robot_uid]

    # get joint positions and limits
    qpos = robot.body_data.qpos[:, joint_ids]
    qpos_limits = robot.body_data.qpos_limits[:, joint_ids, :]

    # compute normalized position in range [0, 1]
    qpos_normalized = (qpos - qpos_limits[:, :, 0]) / (
        qpos_limits[:, :, 1] - qpos_limits[:, :, 0]
    )

    # compute distance to limits (minimum of distance to lower and upper limit)
    dist_to_lower = qpos_normalized
    dist_to_upper = 1.0 - qpos_normalized
    dist_to_limit = torch.min(dist_to_lower, dist_to_upper)

    # penalize joints within margin of limits
    penalty_mask = dist_to_limit < margin
    penalty = torch.where(
        penalty_mask,
        -(margin - dist_to_limit),  # negative penalty
        torch.zeros_like(dist_to_limit),
    )

    # sum over all joints
    return penalty.sum(dim=-1)


def collision_penalty(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    robot_uid: str = "robot",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize collisions based on contact forces."""
    robot = env.sim[robot_uid]

    # get joint forces (torques)
    qf = robot.body_data.qf

    # check if any joint force exceeds threshold
    collision_detected = (torch.abs(qf) > force_threshold).any(dim=-1)

    # return penalty for collisions
    penalty = torch.where(
        collision_detected,
        torch.full((env.num_envs,), -1.0, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )

    return penalty


def orientation_alignment_reward(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_entity_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward alignment of orientations between two entities."""
    # get source entity rotation matrix
    source_obj = env.sim[source_entity_cfg.uid]
    if hasattr(source_obj, "get_body_pose"):
        source_rot = source_obj.get_body_pose(body_ids=source_entity_cfg.body_ids)[
            :, :3, :3
        ]
    elif hasattr(source_obj, "get_local_pose"):
        source_rot = source_obj.get_local_pose(to_matrix=True)[:, :3, :3]
    else:
        raise ValueError(
            f"Entity '{source_entity_cfg.uid}' does not support orientation query."
        )

    # get target entity rotation matrix
    target_obj = env.sim[target_entity_cfg.uid]
    if hasattr(target_obj, "get_body_pose"):
        target_rot = target_obj.get_body_pose(body_ids=target_entity_cfg.body_ids)[
            :, :3, :3
        ]
    elif hasattr(target_obj, "get_local_pose"):
        target_rot = target_obj.get_local_pose(to_matrix=True)[:, :3, :3]
    else:
        raise ValueError(
            f"Entity '{target_entity_cfg.uid}' does not support orientation query."
        )

    # compute rotation difference
    rot_diff = torch.bmm(source_rot, target_rot.transpose(-1, -2))

    # trace of rotation matrix difference (measure of alignment)
    # trace = 1 + 2*cos(theta) for rotation by angle theta
    # normalized to range [0, 1] where 1 is perfect alignment
    trace = rot_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    alignment = (trace - 1.0) / 2.0  # normalize to [-1, 1]

    return alignment


def success_reward(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    reward_value: float = 1.0,
) -> torch.Tensor:
    """Sparse reward for task success."""
    # Check if success info is available in info dict
    if "success" in info:
        success = info["success"]
        if isinstance(success, bool):
            success = torch.tensor([success], device=env.device, dtype=torch.bool)
        elif not isinstance(success, torch.Tensor):
            success = torch.tensor(success, device=env.device, dtype=torch.bool)
    else:
        # No success info available
        return torch.zeros(env.num_envs, device=env.device)

    # return reward
    reward = torch.where(
        success,
        torch.full((env.num_envs,), reward_value, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )

    return reward


def reaching_behind_object_reward(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    object_cfg: SceneEntityCfg = None,
    target_pose_key: str = "goal_pose",
    behind_offset: float = 0.015,
    height_offset: float = 0.015,
    distance_scale: float = 5.0,
    part_name: str = None,
) -> torch.Tensor:
    """Reward for reaching behind an object along object-to-goal direction."""
    # get end effector position from robot FK
    robot = env.robot
    joint_ids = robot.get_joint_ids(part_name)
    qpos = robot.get_qpos()[:, joint_ids]
    ee_pose = robot.compute_fk(name=part_name, qpos=qpos, to_matrix=True)
    ee_pos = ee_pose[:, :3, 3]

    # get object position
    obj = env.sim.get_rigid_object(object_cfg.uid)
    obj_pos = obj.get_local_pose(to_matrix=True)[:, :3, 3]

    # get goal position from info
    if target_pose_key not in info:
        raise ValueError(
            f"Target pose '{target_pose_key}' not found in info dict. "
            f"Make sure to provide it in get_info()."
        )

    target_poses = info[target_pose_key]
    if target_poses.dim() == 2:  # (num_envs, 3)
        goal_pos = target_poses
    else:  # (num_envs, 4, 4)
        goal_pos = target_poses[:, :3, 3]

    # compute push direction (from object to goal)
    push_direction = goal_pos - obj_pos
    push_dir_norm = torch.norm(push_direction, dim=-1, keepdim=True) + 1e-6
    push_dir_normalized = push_direction / push_dir_norm

    # compute target "behind" position
    height_vec = torch.tensor(
        [0, 0, height_offset], device=env.device, dtype=torch.float32
    )
    target_pos = obj_pos - behind_offset * push_dir_normalized + height_vec

    # distance to target position
    ee_to_target_dist = torch.norm(ee_pos - target_pos, dim=-1)

    # tanh-shaped reward (1.0 when at target, 0.0 when far)
    reward = 1.0 - torch.tanh(distance_scale * ee_to_target_dist)

    return reward


def distance_to_target(
    env: "EmbodiedEnv",
    obs: dict,
    action: torch.Tensor,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_pose_key: str = "target_pose",
    exponential: bool = False,
    sigma: float = 1.0,
    use_xy_only: bool = False,
) -> torch.Tensor:
    """Reward based on distance to a virtual target pose from info."""
    # get source entity position
    source_obj = env.sim[source_entity_cfg.uid]
    if hasattr(source_obj, "get_body_pose"):
        source_pos = source_obj.get_body_pose(body_ids=source_entity_cfg.body_ids)[
            :, :3, 3
        ]
    elif hasattr(source_obj, "get_local_pose"):
        source_pos = source_obj.get_local_pose(to_matrix=True)[:, :3, 3]
    else:
        raise ValueError(
            f"Entity '{source_entity_cfg.uid}' does not support position query."
        )

    # get target position from info
    if target_pose_key not in info:
        raise ValueError(
            f"Target pose '{target_pose_key}' not found in info dict. "
            f"Make sure to provide it in get_info()."
        )

    target_poses = info[target_pose_key]
    if target_poses.dim() == 2:  # (num_envs, 3)
        target_pos = target_poses
    else:  # (num_envs, 4, 4)
        target_pos = target_poses[:, :3, 3]

    # compute distance
    if use_xy_only:
        distance = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)
    else:
        distance = torch.norm(source_pos - target_pos, dim=-1)

    # compute reward
    if exponential:
        # exponential reward: exp(-distance^2 / (2 * sigma^2))
        reward = torch.exp(-(distance**2) / (2 * sigma**2))
    else:
        # negative distance reward
        reward = -distance

    return reward


def incremental_distance_to_target(
    env: "EmbodiedEnv",
    obs: dict,
    action: torch.Tensor,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_pose_key: str = "target_pose",
    tanh_scale: float = 10.0,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    use_xy_only: bool = False,
) -> torch.Tensor:
    """Incremental reward for progress toward a virtual target pose from info."""
    # get source entity position
    source_obj = env.sim.get_rigid_object(source_entity_cfg.uid)
    source_pos = source_obj.get_local_pose(to_matrix=True)[:, :3, 3]

    # get target position from info
    if target_pose_key not in info:
        raise ValueError(
            f"Target pose '{target_pose_key}' not found in info dict. "
            f"Make sure to provide it in get_info()."
        )

    target_poses = info[target_pose_key]
    if target_poses.dim() == 2:  # (num_envs, 3)
        target_pos = target_poses
    else:  # (num_envs, 4, 4)
        target_pos = target_poses[:, :3, 3]

    # compute current distance
    if use_xy_only:
        current_dist = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)
    else:
        current_dist = torch.norm(source_pos - target_pos, dim=-1)

    # initialize previous distance on first call
    prev_dist_key = f"_prev_dist_{source_entity_cfg.uid}_{target_pose_key}"
    if not hasattr(env, prev_dist_key):
        setattr(env, prev_dist_key, current_dist.clone())
        return torch.zeros(env.num_envs, device=env.device)

    # compute distance delta (positive = getting closer)
    prev_dist = getattr(env, prev_dist_key)
    distance_delta = prev_dist - current_dist

    # apply tanh shaping
    distance_delta_normalized = torch.tanh(tanh_scale * distance_delta)

    # asymmetric weighting
    reward = torch.where(
        distance_delta_normalized >= 0,
        positive_weight * distance_delta_normalized,
        negative_weight * distance_delta_normalized,
    )

    # update previous distance
    setattr(env, prev_dist_key, current_dist.clone())

    return reward
