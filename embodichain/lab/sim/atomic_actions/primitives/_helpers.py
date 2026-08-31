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

from embodichain.lab.sim.utility.action_utils import resample_with_distance

from ..bindings import EndpointBinding
from ..state import PlanningContext
from ..trajectory_ops import build_pose_plan_states

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
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


def require_shared_task_state_key(
    motion: EndpointBinding,
    grasp: EndpointBinding,
    *,
    participant: str,
) -> str:
    """Return the logical held-object key shared by one participant's endpoints."""
    motion_key = motion.task_state_key
    grasp_key = grasp.task_state_key
    if motion_key != grasp_key:
        raise ValueError(
            f"{participant} motion and grasp endpoints must share one "
            f"task_state_key, but got {motion_key!r} and {grasp_key!r}."
        )
    if not isinstance(motion_key, str) or not motion_key:
        raise ValueError(f"{participant} task_state_key must be a non-empty string.")
    return motion_key


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
    interpolation_dt: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan a fixed-size pose trajectory for one named manipulator."""
    result = motion_generator.generate(
        build_pose_plan_states(target_poses),
        options=motion_policy.to_motion_gen_options(
            start_qpos=start_qpos,
            control_part=control_part,
            sample_count=n_waypoints,
            interpolation_dt=interpolation_dt,
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


def split_joint_trajectory_at_pose(
    trajectory: torch.Tensor,
    split_pose: torch.Tensor,
    *,
    robot: Robot,
    control_part: str,
    first_sample_count: int,
    second_sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split and resample a joint path at its closest target-pose sample.

    The split sample is included at both segment boundaries so a hand-command
    gap can hold the arm at that pose.

    Args:
        trajectory: Planned controlled-joint path with shape ``(B, N, D)``.
        split_pose: Target end-effector pose with shape ``(B, 4, 4)``.
        robot: Robot providing batched forward kinematics.
        control_part: Controlled robot part used for forward kinematics.
        first_sample_count: Output samples before the hand-command gap.
        second_sample_count: Output samples after the hand-command gap.

    Returns:
        The resampled trajectory segments before and after the split pose.
    """
    if trajectory.dim() != 3 or trajectory.shape[1] == 0:
        raise ValueError("trajectory must have shape (B, N, D) with N > 0.")
    if split_pose.shape != (trajectory.shape[0], 4, 4):
        raise ValueError("split_pose must have shape (B, 4, 4).")
    if first_sample_count < 2 or second_sample_count < 2:
        raise ValueError("Both split trajectory segments require at least two samples.")

    trajectory_xpos = robot.compute_batch_fk(
        qpos=trajectory,
        name=control_part,
        to_matrix=True,
    )
    position_error = torch.linalg.vector_norm(
        trajectory_xpos[..., :3, 3] - split_pose[:, None, :3, 3],
        dim=-1,
    )
    relative_rotation = torch.matmul(
        trajectory_xpos[..., :3, :3].transpose(-1, -2),
        split_pose[:, None, :3, :3],
    )
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_error = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    pose_error = position_error + rotation_error
    minimum_error = pose_error.min(dim=1, keepdim=True).values
    candidates = torch.isclose(pose_error, minimum_error, rtol=1.0e-5, atol=1.0e-6)
    expected_index = round(
        (trajectory.shape[1] - 1)
        * first_sample_count
        / (first_sample_count + second_sample_count)
    )
    progress_error = torch.abs(
        torch.arange(trajectory.shape[1], device=trajectory.device) - expected_index
    )
    split_indices = torch.where(
        candidates,
        progress_error[None],
        torch.full_like(progress_error[None], trajectory.shape[1]),
    ).argmin(dim=1)

    sample_indices = torch.arange(trajectory.shape[1], device=trajectory.device)
    first_indices = torch.minimum(sample_indices[None], split_indices[:, None])
    second_indices = torch.minimum(
        split_indices[:, None] + sample_indices[None],
        torch.full_like(sample_indices[None], trajectory.shape[1] - 1),
    )
    gather_shape = (-1, -1, trajectory.shape[2])
    first = torch.gather(
        trajectory,
        1,
        first_indices.unsqueeze(-1).expand(*gather_shape),
    )
    second = torch.gather(
        trajectory,
        1,
        second_indices.unsqueeze(-1).expand(*gather_shape),
    )
    return (
        resample_with_distance(first, first_sample_count, device=trajectory.device),
        resample_with_distance(second, second_sample_count, device=trajectory.device),
    )


__all__ = [
    "arm_qpos_from_state",
    "assemble_full_robot_trajectory",
    "plan_named_arm_trajectory",
    "require_shared_task_state_key",
    "repeat_qpos",
    "resolve_batched_pose",
    "resolve_object_target",
    "split_joint_trajectory_at_pose",
]
