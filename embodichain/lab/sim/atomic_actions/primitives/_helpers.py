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

from embodichain.compute.trajectory import (
    resample_with_distance,
    retime_to_control_grid,
)

from embodichain.utils.math import get_relative_rotation

from ..bindings import EndpointBinding
from ..state import PlanningContext
from ..trajectory_ops import build_pose_plan_states

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator

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


def hold_inactive_cartesian_rows(
    path: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    hold_pose: torch.Tensor,
) -> torch.Tensor:
    """Replace inactive Cartesian path rows with a consistent pose hold.

    Exact-path interpolation validates its Cartesian start against FK of the
    supplied joint seed.  A row that failed an earlier segment therefore must
    not retain the successful row's requested contact geometry.
    """
    if path.ndim != 4 or path.shape[2:] != (4, 4):
        raise ValueError("path must have shape (B, T, 4, 4).")
    if active_mask.dtype != torch.bool or active_mask.shape != (path.shape[0],):
        raise ValueError("active_mask must have bool shape (B,).")
    if hold_pose.shape != (path.shape[0], 4, 4):
        raise ValueError("hold_pose must have shape (B, 4, 4).")
    if (
        active_mask.device != path.device
        or hold_pose.device != path.device
        or hold_pose.dtype != path.dtype
    ):
        raise ValueError(
            "path, active_mask, and hold_pose must share a device, and path and "
            "hold_pose must share a dtype."
        )
    return torch.where(
        active_mask[:, None, None, None],
        path,
        hold_pose[:, None].expand_as(path),
    )


def resample_planned_trajectory(
    trajectory: torch.Tensor,
    sample_count: int,
    *,
    dt: torch.Tensor | None = None,
    control_dt: float | None = None,
) -> torch.Tensor:
    """Put a planner path on a composite action's control-time grid.

    Some planners preserve their native samples instead of honoring the
    backend-neutral requested count.  When timing is supplied, the requested
    count is only a lower bound: native duration is quantized without speeding
    it up, and any remaining samples are terminal holds.  The legacy
    position-only behavior remains available to callers that have no timing.

    Args:
        trajectory: Batched joint path with shape ``(B, N, D)``.
        sample_count: Required number of output samples.
        dt: Optional native arrival intervals with shape ``(B, N)``.
        control_dt: Destination control interval, required together with ``dt``.

    Returns:
        A control-grid path with at least ``sample_count`` samples when timing
        is supplied, otherwise exactly ``sample_count`` distance samples.

    Raises:
        TypeError: If ``trajectory`` is not a floating-point tensor.
        ValueError: If the path or requested sample count is invalid.
    """
    if not isinstance(trajectory, torch.Tensor):
        raise TypeError("trajectory must be a torch.Tensor.")
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape (B, N, D).")
    if not torch.is_floating_point(trajectory):
        raise TypeError("trajectory must be a floating-point tensor.")
    if trajectory.shape[1] == 0 or trajectory.shape[2] == 0:
        raise ValueError("trajectory must have non-zero N and D dimensions.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer.")
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if trajectory.shape[0] == 0:
        return trajectory.new_empty((0, sample_count, trajectory.shape[2]))
    if (dt is None) != (control_dt is None):
        raise ValueError("dt and control_dt must be supplied together.")
    if dt is not None:
        assert control_dt is not None
        retimed, _, _, _ = retime_to_control_grid(trajectory, dt, control_dt)
        if retimed.shape[1] >= sample_count:
            return retimed
        padding = retimed[:, -1:].expand(-1, sample_count - retimed.shape[1], -1)
        return torch.cat([retimed, padding], dim=1)
    if trajectory.shape[1] == sample_count:
        return trajectory
    if trajectory.shape[1] == 1:
        return trajectory.expand(-1, sample_count, -1).clone()
    resampled = resample_with_distance(
        trajectory,
        sample_count,
        device=trajectory.device,
    )
    # Preserve the planner's exact boundary states even when the Warp path
    # performs its arithmetic in float32 or the Torch fallback rounds a
    # cumulative-distance fraction at an endpoint.
    resampled = resampled.to(dtype=trajectory.dtype).clone()
    resampled[:, 0] = trajectory[:, 0]
    resampled[:, -1] = trajectory[:, -1]
    return resampled


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
    if result.dt is None:
        raise ValueError("Motion planning result must contain arrival intervals.")
    return result.success, resample_planned_trajectory(
        result.positions,
        n_waypoints,
        dt=result.dt,
        control_dt=interpolation_dt,
    )


def pad_trajectory_to_length(
    trajectory: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Pad a joint trajectory with its terminal pose to a shared length."""
    if trajectory.ndim != 3 or trajectory.shape[1] == 0:
        raise ValueError("trajectory must have shape (B, N, D) with N > 0.")
    if sample_count < trajectory.shape[1]:
        raise ValueError("sample_count cannot shorten a trajectory.")
    if sample_count == trajectory.shape[1]:
        return trajectory
    return torch.cat(
        [
            trajectory,
            trajectory[:, -1:].expand(-1, sample_count - trajectory.shape[1], -1),
        ],
        dim=1,
    )


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
    dt: torch.Tensor | None = None,
    control_dt: float | None = None,
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
    if dt is None and control_dt is None:
        return (
            resample_with_distance(first, first_sample_count, device=trajectory.device),
            resample_with_distance(
                second, second_sample_count, device=trajectory.device
            ),
        )
    if dt is None or control_dt is None:
        raise ValueError("dt and control_dt must be supplied together.")
    if dt.shape != trajectory.shape[:2]:
        raise ValueError("dt must have shape (B, N) matching trajectory.")
    first_dt = torch.gather(dt, 1, first_indices).clone()
    first_dt[:, 0] = 0.0
    first_dt = torch.where(
        sample_indices[None] <= split_indices[:, None], first_dt, 0.0
    )
    second_dt = torch.gather(dt, 1, second_indices).clone()
    second_dt[:, 0] = 0.0
    second_dt = torch.where(
        second_indices < trajectory.shape[1] - 1,
        second_dt,
        torch.where(
            sample_indices[None] <= trajectory.shape[1] - 1 - split_indices[:, None],
            second_dt,
            0.0,
        ),
    )
    return (
        resample_planned_trajectory(
            first,
            first_sample_count,
            dt=first_dt,
            control_dt=control_dt,
        ),
        resample_planned_trajectory(
            second,
            second_sample_count,
            dt=second_dt,
            control_dt=control_dt,
        ),
    )


__all__ = [
    "arm_qpos_from_state",
    "assemble_full_robot_trajectory",
    "pad_trajectory_to_length",
    "plan_named_arm_trajectory",
    "resample_planned_trajectory",
    "require_shared_task_state_key",
    "repeat_qpos",
    "resolve_batched_pose",
    "resolve_object_target",
    "split_joint_trajectory_at_pose",
]


def find_symmetric_nearest_xpos(
    target_xpos: torch.Tensor, reference_xpos: torch.Tensor
) -> torch.Tensor:
    """Find the nearest symmetric pose to the reference pose."""
    symmetric_xpos = target_xpos.clone()
    symmetric_xpos[:, :3, 0] = -symmetric_xpos[:, :3, 0]
    symmetric_xpos[:, :3, 1] = -symmetric_xpos[:, :3, 1]
    angle_a = get_relative_rotation(reference_xpos[:, :3, :3], target_xpos[:, :3, :3])
    angle_b = get_relative_rotation(
        reference_xpos[:, :3, :3], symmetric_xpos[:, :3, :3]
    )
    choose_target = (angle_a < angle_b)[..., None, None]
    target_xpos = torch.where(choose_target, target_xpos, symmetric_xpos)
    return target_xpos
