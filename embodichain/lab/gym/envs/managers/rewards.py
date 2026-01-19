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
    """Reward based on distance between two rigid objects.

    Encourages the source object to get closer to the target object. Can use either
    linear negative distance or exponential Gaussian-shaped reward.

    Args:
        source_entity_cfg: Configuration for the source object (e.g., {"uid": "cube"})
        target_entity_cfg: Configuration for the target object (e.g., {"uid": "goal_sphere"})
        exponential: If True, use exponential reward exp(-d²/2σ²), else use -distance
        sigma: Standard deviation for exponential reward (controls reward spread)

    Returns:
        Reward tensor of shape (num_envs,). Higher when objects are closer.
        - Linear mode: ranges from -inf to 0 (0 when objects touch)
        - Exponential mode: ranges from 0 to 1 (1 when objects touch)

    Example:
        ```json
        {
            "func": "distance_between_objects",
            "weight": 0.5,
            "params": {
                "source_entity_cfg": {"uid": "cube"},
                "target_entity_cfg": {"uid": "target"},
                "exponential": true,
                "sigma": 0.2
            }
        }
        ```
    """
    # get source entity position
    source_obj = env.sim.get_rigid_object(source_entity_cfg.uid)
    source_pos = source_obj.get_local_pose(to_matrix=True)[:, :3, 3]

    # get target entity position
    target_obj = env.sim.get_rigid_object(target_entity_cfg.uid)
    target_pos = target_obj.get_local_pose(to_matrix=True)[:, :3, 3]

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
    joint_ids: slice | list[int] | None = None,
    part_name: str | None = None,
) -> torch.Tensor:
    """Penalize high joint velocities to encourage smooth motion.

    Computes the L2 norm of joint velocities and returns negative value as penalty.
    Useful for preventing jerky or unstable robot movements.

    Args:
        robot_uid: Robot entity UID in simulation (default: "robot")
        joint_ids: Specific joint indices to penalize. Takes priority over part_name.
                   Example: [0, 1, 2] or slice(0, 6)
        part_name: Control part name (e.g., "arm"). Used only if joint_ids is None.
                   Will penalize all joints in the specified part.

    Returns:
        Penalty tensor of shape (num_envs,). Always negative or zero.
        Magnitude increases with joint velocity (larger velocity = more negative).

    Example:
        ```json
        {
            "func": "joint_velocity_penalty",
            "weight": 0.001,
            "params": {
                "robot_uid": "robot",
                "part_name": "arm"
            }
        }
        ```
    """
    robot = env.sim.get_robot(robot_uid)

    # get joint velocities
    if joint_ids is not None:
        qvel = robot.get_qvel()[:, joint_ids]
    elif part_name is not None:
        qvel = robot.get_qvel(name=part_name)
    else:
        qvel = robot.get_qvel()

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
    """Penalize large action changes between consecutive timesteps.

    Encourages smooth control commands by penalizing sudden changes in actions.
    Stores previous action in env._reward_states for comparison.

    Returns:
        Penalty tensor of shape (num_envs,). Zero on first call (no previous action),
        negative on subsequent calls (larger change = more negative).

    Note:
        This function maintains state across calls using env._reward_states['prev_actions'].
        State is automatically reset when the environment resets.

    Example:
        ```json
        {
            "func": "action_smoothness_penalty",
            "weight": 0.01,
            "params": {}
        }
        ```
    """
    # Use dictionary-based state management
    if not hasattr(env, "_reward_states"):
        env._reward_states = {}

    # compute difference between current and previous action
    if "prev_actions" in env._reward_states:
        action_diff = action - env._reward_states["prev_actions"]
        penalty = -torch.norm(action_diff, dim=-1)
    else:
        # no previous action, no penalty
        penalty = torch.zeros(env.num_envs, device=env.device)

    # store current action for next step
    env._reward_states["prev_actions"] = action.clone()

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
    """Penalize robot joints that are close to their position limits.

    Prevents joints from reaching their physical limits, which can cause instability
    or singularities. Penalty increases as joints approach limits within the margin.

    Args:
        robot_uid: Robot entity UID in simulation (default: "robot")
        joint_ids: Joint indices to monitor (default: all joints)
        margin: Normalized distance threshold (0 to 1). Penalty applied when joint
                is within this fraction of its range from either limit.
                Example: 0.1 means penalty when within 10% of limits.

    Returns:
        Penalty tensor of shape (num_envs,). Always negative or zero.
        Sum of penalties across all monitored joints.

    Example:
        ```json
        {
            "func": "joint_limit_penalty",
            "weight": 0.01,
            "params": {
                "robot_uid": "robot",
                "joint_ids": [0, 1, 2, 3, 4, 5],
                "margin": 0.1
            }
        }
        ```
    """
    robot = env.sim.get_robot(robot_uid)

    # get joint positions and limits
    qpos = robot.get_qpos()[:, joint_ids]
    qpos_limits = robot.get_qpos_limits()[:, joint_ids, :]

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


def orientation_alignment(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_entity_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward rotational alignment between two rigid objects.

    Encourages the source object's orientation to match the target object's orientation.
    Uses rotation matrix trace to measure alignment.

    Args:
        source_entity_cfg: Configuration for the source object (e.g., {"uid": "cube"})
        target_entity_cfg: Configuration for the target object (e.g., {"uid": "reference"})

    Returns:
        Reward tensor of shape (num_envs,). Ranges from -1 to 1.
        - 1.0: Perfect alignment (same orientation)
        - 0.0: 90° rotation difference
        - -1.0: 180° rotation difference (opposite orientation)

    Example:
        ```json
        {
            "func": "orientation_alignment",
            "weight": 0.5,
            "params": {
                "source_entity_cfg": {"uid": "object"},
                "target_entity_cfg": {"uid": "goal_object"}
            }
        }
        ```
    """
    # get source entity rotation matrix
    source_obj = env.sim.get_rigid_object(source_entity_cfg.uid)
    source_rot = source_obj.get_local_pose(to_matrix=True)[:, :3, :3]

    # get target entity rotation matrix
    target_obj = env.sim.get_rigid_object(target_entity_cfg.uid)
    target_rot = target_obj.get_local_pose(to_matrix=True)[:, :3, :3]

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
) -> torch.Tensor:
    """Sparse bonus reward when task succeeds.

    Provides a fixed reward when the task success condition is met.
    Reads success status from info['success'] which should be set by the environment.

    Returns:
        Reward tensor of shape (num_envs,).
        - 1.0 when successful
        - 0.0 when not successful or if 'success' key missing

    Note:
        The environment's get_info() must populate info['success'] with a boolean
        tensor indicating success status for each environment.

    Example:
        ```json
        {
            "func": "success_reward",
            "weight": 10.0,
            "params": {}
        }
        ```
    """
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
    return torch.where(
        success,
        torch.ones(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )


def reaching_behind_object(
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
    """Reward for positioning end-effector behind object for pushing.

    Encourages the robot's end-effector to reach a position behind the object along
    the object-to-goal direction. Useful for push manipulation tasks.

    Args:
        object_cfg: Configuration for the object to push (e.g., {"uid": "cube"})
        target_pose_key: Key in info dict for goal pose (default: "goal_pose")
                        Can be (num_envs, 3) position or (num_envs, 4, 4) transform
        behind_offset: Distance behind object to reach (in meters, default: 0.015)
        height_offset: Additional height above object (in meters, default: 0.015)
        distance_scale: Scaling factor for tanh function (higher = steeper, default: 5.0)
        part_name: Robot part name for FK computation (e.g., "arm")

    Returns:
        Reward tensor of shape (num_envs,). Ranges from 0 to 1.
        - 1.0: End-effector at ideal pushing position
        - 0.0: End-effector far from ideal position

    Example:
        ```json
        {
            "func": "reaching_behind_object",
            "weight": 0.1,
            "params": {
                "object_cfg": {"uid": "cube"},
                "target_pose_key": "goal_pose",
                "behind_offset": 0.015,
                "height_offset": 0.015,
                "distance_scale": 5.0,
                "part_name": "arm"
            }
        }
        ```
    """
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
    """Reward based on absolute distance to a virtual target pose.

    Encourages an object to get closer to a target pose specified in the info dict.
    Unlike incremental_distance_to_target, this provides direct distance-based reward.

    Args:
        source_entity_cfg: Configuration for the object (e.g., {"uid": "cube"})
        target_pose_key: Key in info dict for target pose (default: "target_pose")
                        Can be (num_envs, 3) position or (num_envs, 4, 4) transform
        exponential: If True, use exponential reward exp(-d²/2σ²), else use -distance
        sigma: Standard deviation for exponential reward (default: 1.0)
        use_xy_only: If True, ignore z-axis and only consider horizontal distance

    Returns:
        Reward tensor of shape (num_envs,).
        - Linear mode: -distance (negative, approaches 0 when close)
        - Exponential mode: exp(-d²/2σ²) (0 to 1, approaches 1 when close)

    Example:
        ```json
        {
            "func": "distance_to_target",
            "weight": 0.5,
            "params": {
                "source_entity_cfg": {"uid": "cube"},
                "target_pose_key": "goal_pose",
                "exponential": false,
                "use_xy_only": true
            }
        }
        ```
    """
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
    """Incremental reward for progress toward a virtual target pose.

    Rewards the robot for getting closer to the target compared to previous timestep.
    Stores previous distance in env._reward_states for comparison. Uses tanh shaping
    to normalize rewards and supports asymmetric weighting for approach vs. retreat.

    Args:
        source_entity_cfg: Configuration for the object (e.g., {"uid": "cube"})
        target_pose_key: Key in info dict for target pose (default: "target_pose")
                        Can be (num_envs, 3) position or (num_envs, 4, 4) transform
        tanh_scale: Scaling for tanh normalization (higher = more sensitive, default: 10.0)
        positive_weight: Multiplier for reward when getting closer (default: 1.0)
        negative_weight: Multiplier for penalty when moving away (default: 1.0)
        use_xy_only: If True, ignore z-axis and only consider horizontal distance

    Returns:
        Reward tensor of shape (num_envs,). Zero on first call, then:
        - Positive when getting closer (scaled by positive_weight)
        - Negative when moving away (scaled by negative_weight)
        - Magnitude bounded by tanh function

    Note:
        This function maintains state using env._reward_states[f"prev_dist_{uid}_{key}"].
        State is automatically reset when the environment resets.

    Example:
        ```json
        {
            "func": "incremental_distance_to_target",
            "weight": 1.0,
            "params": {
                "source_entity_cfg": {"uid": "cube"},
                "target_pose_key": "goal_pose",
                "tanh_scale": 10.0,
                "positive_weight": 2.0,
                "negative_weight": 0.5,
                "use_xy_only": true
            }
        }
        ```
    """
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
    # Use dictionary-based state management for better organization
    if not hasattr(env, "_reward_states"):
        env._reward_states = {}

    state_key = f"prev_dist_{source_entity_cfg.uid}_{target_pose_key}"
    if state_key not in env._reward_states:
        env._reward_states[state_key] = current_dist.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # compute distance delta (positive = getting closer)
    prev_dist = env._reward_states[state_key]
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
    env._reward_states[state_key] = current_dist.clone()

    return reward
