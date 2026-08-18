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

"""Shared internal helpers for concrete atomic actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..state import PlanningContext
from ..trajectory_ops import build_pose_plan_states

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator

    from ..policies import MotionPolicy


def resolve_batched_pose(
    pose: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Copy and broadcast one homogeneous pose to ``(num_envs, 4, 4)``."""
    pose = pose.to(device=device, dtype=torch.float32).clone()
    if pose.shape == (4, 4):
        pose = pose.unsqueeze(0).repeat(num_envs, 1, 1)
    if pose.shape != (num_envs, 4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4) or ({num_envs}, 4, 4), "
            f"but got {pose.shape}"
        )
    return pose


def resolve_object_target(
    target: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
    name: str = "object_target_pose",
) -> torch.Tensor:
    """Broadcast an object target pose to ``(num_envs, 4, 4)`` or validate it."""
    return resolve_batched_pose(
        target,
        num_envs=num_envs,
        device=device,
        name=name,
    )


def repeat_qpos(qpos: torch.Tensor, n_waypoints: int) -> torch.Tensor:
    """Repeat batched joint positions along a waypoint dimension."""
    return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)


def assemble_full_robot_trajectory(
    base_qpos: torch.Tensor,
    part_trajectories: Sequence[tuple[Sequence[int], torch.Tensor]],
) -> torch.Tensor:
    """Overlay control-part trajectories on repeated full-robot positions."""
    if not part_trajectories:
        raise ValueError("part_trajectories must not be empty.")
    n_waypoints = part_trajectories[0][1].shape[1]
    full = repeat_qpos(
        base_qpos.to(
            device=part_trajectories[0][1].device,
            dtype=torch.float32,
        ),
        n_waypoints,
    ).clone()
    for joint_ids, trajectory in part_trajectories:
        full[:, :, list(joint_ids)] = trajectory
    return full


def plan_named_arm_trajectory(
    motion_generator: MotionGenerator,
    control_part: str,
    start_qpos: torch.Tensor,
    target_poses: torch.Tensor,
    n_waypoints: int,
    motion_policy: MotionPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan a fixed-size pose trajectory for one named manipulator."""
    result = motion_generator.generate(
        build_pose_plan_states(target_poses),
        options=motion_policy.to_motion_gen_options(
            start_qpos=start_qpos,
            control_part=control_part,
            sample_count=n_waypoints,
        ),
    )
    if not isinstance(result.success, torch.Tensor):
        raise TypeError("Motion planning success must be a torch.Tensor.")
    if result.positions is None:
        raise ValueError("Motion planning result must contain joint positions.")
    return result.success, result.positions


def arm_qpos_from_state(
    context: PlanningContext,
    arm_joint_ids: list[int],
) -> torch.Tensor:
    """Extract the arm slice of the measured planning-start joint positions."""
    return context.robot.qpos[:, arm_joint_ids]


__all__ = [
    "arm_qpos_from_state",
    "assemble_full_robot_trajectory",
    "plan_named_arm_trajectory",
    "repeat_qpos",
    "resolve_batched_pose",
    "resolve_object_target",
]
