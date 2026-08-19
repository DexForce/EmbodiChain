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

"""Pure target-shaping and trajectory operations for atomic actions."""

from __future__ import annotations

import torch

from embodichain.lab.sim.planners import MoveType, PlanResult, PlanState
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance

from .plans import TimedTrajectory, normalize_success_mask


def resolve_pose_target(
    target: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Validate and copy an end-effector target onto the planning device."""
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be torch.Tensor of shape (4, 4), ({num_envs}, 4, 4), "
            f"or ({num_envs}, n_waypoint, 4, 4)"
        )
    target = target.to(device=device, dtype=torch.float32).clone()
    if target.shape == (4, 4):
        target = target.unsqueeze(0).repeat(num_envs, 1, 1)
    if target.dim() == 3:
        if target.shape != (num_envs, 4, 4):
            raise ValueError(
                f"target tensor must have shape (4, 4) or ({num_envs}, 4, 4), "
                f"but got {target.shape}"
            )
    elif target.dim() == 4:
        if target.shape[0] != num_envs or target.shape[2:] != (4, 4):
            raise ValueError(
                "multi-waypoint target tensor must have shape "
                f"({num_envs}, n_waypoint, 4, 4), but got {target.shape}"
            )
        if target.shape[1] == 0:
            raise ValueError(
                "multi-waypoint target tensor has zero waypoints (shape[1] == 0); "
                "at least one waypoint is required."
            )
    else:
        raise ValueError(
            f"target tensor must be (4, 4), ({num_envs}, 4, 4), or "
            f"({num_envs}, n_waypoint, 4, 4), but got {target.shape}"
        )
    return target


def resolve_joint_target(
    target_qpos: torch.Tensor,
    *,
    num_envs: int,
    joint_dof: int,
    control_part: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Validate and copy a joint target onto the planning device."""
    if not isinstance(target_qpos, torch.Tensor):
        raise TypeError(
            f"target qpos for '{control_part}' must be a torch.Tensor with shape "
            f"({joint_dof},), ({num_envs}, {joint_dof}), or "
            f"({num_envs}, n_waypoint, {joint_dof})"
        )
    target_qpos = target_qpos.to(device=device, dtype=torch.float32).clone()
    if target_qpos.shape == (joint_dof,):
        target_qpos = target_qpos.unsqueeze(0).repeat(num_envs, 1)
    if target_qpos.dim() == 2:
        if target_qpos.shape != (num_envs, joint_dof):
            raise ValueError(
                f"target qpos for '{control_part}' must have shape ({joint_dof},) "
                f"or ({num_envs}, {joint_dof}), but got {target_qpos.shape}"
            )
    elif target_qpos.dim() == 3:
        if target_qpos.shape[0] != num_envs or target_qpos.shape[2] != joint_dof:
            raise ValueError(
                f"multi-waypoint target qpos for '{control_part}' must have shape "
                f"({num_envs}, n_waypoint, {joint_dof}), but got {target_qpos.shape}"
            )
        if target_qpos.shape[1] == 0:
            raise ValueError(
                f"multi-waypoint target qpos for '{control_part}' has zero waypoints "
                "(shape[1] == 0); at least one waypoint is required."
            )
    else:
        raise ValueError(
            f"target qpos for '{control_part}' must be 1D, 2D, or 3D with "
            f"trailing dim {joint_dof}, but got {target_qpos.shape}"
        )
    return target_qpos


def build_pose_plan_states(target_poses: torch.Tensor) -> list[PlanState]:
    """Convert batched pose targets into the planner's waypoint representation.

    Args:
        target_poses: One pose per environment with shape ``(B, 4, 4)`` or a
            waypoint sequence with shape ``(B, N, 4, 4)``.

    Returns:
        One batched :class:`PlanState` per waypoint.
    """
    if target_poses.dim() == 3:
        target_poses = target_poses.unsqueeze(1)
    if target_poses.dim() != 4 or target_poses.shape[2:] != (4, 4):
        raise ValueError("target_poses must have shape (B, 4, 4) or (B, N, 4, 4).")
    if target_poses.shape[1] == 0:
        raise ValueError("target_poses must contain at least one waypoint.")
    return [
        PlanState(xpos=target_poses[:, index], move_type=MoveType.EEF_MOVE)
        for index in range(target_poses.shape[1])
    ]


def build_joint_plan_states(target_qpos: torch.Tensor) -> list[PlanState]:
    """Convert batched joint targets into the planner's waypoint representation.

    Args:
        target_qpos: One target per environment with shape ``(B, D)`` or a
            waypoint sequence with shape ``(B, N, D)``.

    Returns:
        One batched :class:`PlanState` per waypoint.
    """
    if target_qpos.dim() == 2:
        target_qpos = target_qpos.unsqueeze(1)
    if target_qpos.dim() != 3:
        raise ValueError("target_qpos must have shape (B, D) or (B, N, D).")
    if target_qpos.shape[1] == 0:
        raise ValueError("target_qpos must contain at least one waypoint.")
    return [
        PlanState(qpos=target_qpos[:, index], move_type=MoveType.JOINT_MOVE)
        for index in range(target_qpos.shape[1])
    ]


def translate_pose_world(pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Translate batched poses by a world-frame offset."""
    if not (pose.dim() == 3 and pose.shape[1:] == (4, 4)):
        raise ValueError("pose must have shape [N, 4, 4]")
    offset = offset.to(device=pose.device, dtype=pose.dtype)
    if offset.dim() == 1:
        offset = offset.unsqueeze(0)
    if not (offset.dim() == 2 and offset.shape[1] == 3):
        raise ValueError("offset must have shape [N, 3] or [3]")
    if offset.shape[0] not in (1, pose.shape[0]):
        raise ValueError(
            f"offset batch size must be 1 or match pose batch size {pose.shape[0]}, "
            f"but got {offset.shape[0]}"
        )
    result = pose.clone()
    result[:, :3, 3] += offset
    return result


def axis_translation_keyframes(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    axis: torch.Tensor,
    *,
    n_waypoints: int,
) -> torch.Tensor:
    """Build exact Cartesian translation targets along one world-space axis.

    The returned targets exclude ``start_pose`` and include ``end_pose``. This
    matches motion generation, where the observed start configuration is added
    separately. Rotation remains fixed for the entire constrained segment.

    Args:
        start_pose: Batched segment-start poses, shape ``(B, 4, 4)``.
        end_pose: Batched segment-end poses, shape ``(B, 4, 4)``.
        axis: Shared ``(3,)`` or batched ``(B, 3)`` world-space axis.
        n_waypoints: Number of target poses, excluding the segment start.

    Returns:
        Batched keyframes with shape ``(B, n_waypoints, 4, 4)``.

    Raises:
        ValueError: If poses, axis, count, rotation, or displacement are invalid.
    """
    if (
        start_pose.dim() != 3
        or start_pose.shape[1:] != (4, 4)
        or end_pose.shape != start_pose.shape
    ):
        raise ValueError("start_pose and end_pose must have shape (B, 4, 4).")
    if n_waypoints < 1:
        raise ValueError("n_waypoints must be at least 1.")
    axis = axis.to(device=start_pose.device, dtype=start_pose.dtype)
    if axis.shape == (3,):
        axis = axis.unsqueeze(0).expand(start_pose.shape[0], -1)
    if axis.shape != (start_pose.shape[0], 3) or not torch.isfinite(axis).all():
        raise ValueError("axis must be finite with shape (3,) or (B, 3).")
    axis_norm = torch.linalg.vector_norm(axis, dim=1, keepdim=True)
    if torch.any(axis_norm <= 1.0e-6):
        raise ValueError("axis must be non-zero.")
    axis = axis / axis_norm
    if not torch.allclose(
        start_pose[:, :3, :3],
        end_pose[:, :3, :3],
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError("Axis translation requires a fixed segment rotation.")
    displacement = end_pose[:, :3, 3] - start_pose[:, :3, 3]
    orthogonal = displacement - (displacement * axis).sum(dim=1, keepdim=True) * axis
    if torch.any(torch.linalg.vector_norm(orthogonal, dim=1) > 1.0e-5):
        raise ValueError("Segment displacement must be parallel to axis.")

    weights = torch.linspace(
        0.0,
        1.0,
        n_waypoints + 1,
        dtype=start_pose.dtype,
        device=start_pose.device,
    )[1:]
    result = start_pose[:, None].expand(-1, n_waypoints, -1, -1).clone()
    result[:, :, :3, 3] = torch.lerp(
        start_pose[:, None, :3, 3],
        end_pose[:, None, :3, 3],
        weights[None, :, None],
    )
    return result


def split_three_segments(
    sample_count: int,
    hand_interp_steps: int,
    *,
    first_segment_ratio: float = 0.6,
    first_segment_name: str = "first",
    third_segment_name: str = "third",
) -> tuple[int, int, int]:
    """Split a sample budget into motion, hand, and motion segments."""
    first = int(round((sample_count - hand_interp_steps) * first_segment_ratio))
    if first < 2:
        raise ValueError(
            f"Not enough waypoints for {first_segment_name} trajectory. "
            "Increase sample_count or decrease hand_interp_steps."
        )
    second = hand_interp_steps
    third = sample_count - first - second
    if third < 2:
        raise ValueError(
            f"Not enough waypoints for {third_segment_name} trajectory. "
            "Increase sample_count or decrease hand_interp_steps."
        )
    return first, second, third


def interpolate_joint_trajectory(
    start_qpos: torch.Tensor,
    target_qpos: torch.Tensor,
    n_waypoints: int,
) -> torch.Tensor:
    """Interpolate a joint path through one or more exact target waypoints."""
    if target_qpos.dim() == 2:
        target_qpos = target_qpos.unsqueeze(1)
    keyframes = torch.cat([start_qpos.unsqueeze(1), target_qpos], dim=1)
    return interpolate_with_distance(
        trajectory=keyframes,
        interp_num=n_waypoints,
        device=start_qpos.device,
    )


def interpolate_hand_qpos(
    start_hand_qpos: torch.Tensor,
    end_hand_qpos: torch.Tensor,
    *,
    n_waypoints: int,
) -> torch.Tensor:
    """Interpolate hand joint positions between two semantic commands."""
    is_unbatched = start_hand_qpos.dim() == 1 and end_hand_qpos.dim() == 1
    end_hand_qpos = end_hand_qpos.to(
        device=start_hand_qpos.device,
        dtype=start_hand_qpos.dtype,
    )
    if start_hand_qpos.dim() == 1:
        start_hand_qpos = start_hand_qpos.unsqueeze(0)
    if end_hand_qpos.dim() == 1:
        end_hand_qpos = end_hand_qpos.unsqueeze(0)
    weights = torch.linspace(
        0,
        1,
        steps=n_waypoints,
        device=start_hand_qpos.device,
        dtype=start_hand_qpos.dtype,
    )
    result = torch.lerp(
        start_hand_qpos.unsqueeze(1),
        end_hand_qpos.unsqueeze(1),
        weights[None, :, None],
    )
    return result.squeeze(0) if is_unbatched else result


def to_full_robot_trajectory(
    result: PlanResult,
    *,
    base_qpos: torch.Tensor,
    joint_ids: list[int],
    env_ids: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, TimedTrajectory]:
    """Embed a controlled-joint plan into a timed full-robot trajectory."""
    positions = result.positions
    if positions is None or positions.dim() != 3:
        raise ValueError("PlanResult.positions must have shape (B, N, control_dof).")
    if positions.shape[0] != base_qpos.shape[0]:
        raise ValueError("PlanResult and base_qpos batch sizes must match.")
    if positions.shape[2] != len(joint_ids):
        raise ValueError("PlanResult controlled DoF does not match joint_ids.")
    full_positions = base_qpos.unsqueeze(1).expand(-1, positions.shape[1], -1).clone()
    full_positions[:, :, joint_ids] = positions

    def embed_derivative(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        full = torch.zeros_like(full_positions)
        full[:, :, joint_ids] = value
        return full

    duration: float | torch.Tensor | None = result.duration
    if result.dt is None:
        duration_tensor = torch.as_tensor(
            result.duration,
            dtype=torch.float32,
            device=base_qpos.device,
        )
        if not bool((duration_tensor > 0.0).any().item()):
            duration = None
    timed = TimedTrajectory.from_positions(
        full_positions,
        env_ids=env_ids,
        control_dt=control_dt,
        velocities=embed_derivative(result.velocities),
        accelerations=embed_derivative(result.accelerations),
        dt=result.dt,
        duration=duration,
    )
    success = normalize_success_mask(
        result.success,
        num_envs=base_qpos.shape[0],
        device=base_qpos.device,
        name="PlanResult.success",
    )
    return success, timed


__all__ = [
    "axis_translation_keyframes",
    "build_joint_plan_states",
    "build_pose_plan_states",
    "interpolate_hand_qpos",
    "interpolate_joint_trajectory",
    "resolve_joint_target",
    "resolve_pose_target",
    "split_three_segments",
    "to_full_robot_trajectory",
    "translate_pose_world",
]
