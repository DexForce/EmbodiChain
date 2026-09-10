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

"""Private checked IK and anchored Cartesian interpolation shared by skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..bindings import JointPositionTarget
from ..state import PlanningContext

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot


_CANDIDATE_POSITION_TOLERANCE = 1.0e-3
_CANDIDATE_ROTATION_TOLERANCE = 1.0e-2
_CANDIDATE_MAX_JOINT_STEP = 0.5


def _checked_candidate_ik(
    robot: Robot,
    poses: torch.Tensor,
    seed: torch.Tensor,
    manipulator: JointPositionTarget,
    context: PlanningContext,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str | None, ...], ...]]:
    """Check solver flags, finite values, limits, and measured FK per choice."""
    if (
        seed.ndim != 3
        or seed.shape[:2] != active.shape
        or seed.shape[-1] != len(manipulator.joint_ids)
        or poses.shape != (*active.shape, 4, 4)
        or active.dtype != torch.bool
    ):
        raise ValueError("Candidate IK inputs have incompatible batch dimensions.")
    for name, value in (("seed", seed), ("poses", poses)):
        if (
            value.device != context.robot.qpos.device
            or value.dtype != context.robot.qpos.dtype
        ):
            raise ValueError(
                f"Candidate IK {name} must match the context device/dtype."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Candidate IK {name} must contain finite values.")
    if active.device != seed.device:
        raise ValueError("Candidate IK active mask must share the seed device.")
    success = torch.zeros_like(active)
    reasons: list[list[str | None]] = [[None] * active.shape[1] for _ in active]
    if not active.any():
        return success, seed.clone(), tuple(tuple(row) for row in reasons)
    env_ids = context.env_ids.tolist()
    limits = robot.get_qpos_limits(
        joint_ids=list(manipulator.joint_ids), env_ids=env_ids
    )
    if not isinstance(limits, torch.Tensor) or limits.shape != (
        seed.shape[0],
        seed.shape[-1],
        2,
    ):
        raise ValueError("Candidate planning requires per-row joint limits.")
    if limits.device != seed.device or limits.dtype != seed.dtype:
        raise ValueError("Candidate joint limits must match the seed device/dtype.")
    if torch.isnan(limits).any() or (limits[..., 0] > limits[..., 1]).any():
        raise ValueError("Candidate joint limits must be ordered and non-NaN.")
    if ((seed < limits[:, None, :, 0]) | (seed > limits[:, None, :, 1])).any():
        raise ValueError("Candidate IK safe seed is outside joint limits.")
    safe_poses = robot.compute_batch_fk(
        qpos=seed, name=manipulator.control_part, env_ids=env_ids, to_matrix=True
    )
    if (
        not isinstance(safe_poses, torch.Tensor)
        or safe_poses.shape != poses.shape
        or safe_poses.device != seed.device
        or safe_poses.dtype != seed.dtype
        or not torch.isfinite(safe_poses).all()
    ):
        raise ValueError(
            "Candidate safe FK must return finite poses with the requested shape/device/dtype."
        )
    solver_poses = torch.where(active[..., None, None], poses, safe_poses)
    result = robot.compute_batch_ik(
        pose=solver_poses,
        name=manipulator.control_part,
        joint_seed=seed,
        env_ids=env_ids,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Candidate IK must return (success, qpos).")
    flags, raw_qpos = result
    if (
        not isinstance(flags, torch.Tensor)
        or flags.shape != active.shape
        or not isinstance(raw_qpos, torch.Tensor)
        or raw_qpos.shape != seed.shape
    ):
        raise ValueError("Candidate IK returned incompatible batch dimensions.")
    if flags.device != seed.device or raw_qpos.device != seed.device:
        raise ValueError("Candidate IK outputs must share the seed device.")
    if not raw_qpos.is_floating_point() or raw_qpos.dtype != seed.dtype:
        raise ValueError("Candidate IK qpos must use the seed floating dtype.")
    if (
        flags.is_complex()
        or not torch.isfinite(flags).all()
        or not ((flags == 0) | (flags == 1)).all()
    ):
        raise ValueError(
            "Candidate IK success flags must be bool or finite numeric 0/1 values."
        )
    flags = flags.to(dtype=torch.bool)
    finite = torch.isfinite(raw_qpos).all(dim=-1)
    success = active & flags & finite
    qpos = torch.where(success[..., None], raw_qpos, seed)
    within_limits = (
        (qpos >= limits[:, None, :, 0]) & (qpos <= limits[:, None, :, 1])
    ).all(dim=-1)
    success &= within_limits
    qpos = torch.where(success[..., None], qpos, seed)
    actual = robot.compute_batch_fk(
        qpos=qpos, name=manipulator.control_part, env_ids=env_ids, to_matrix=True
    )
    if not isinstance(actual, torch.Tensor) or actual.shape != poses.shape:
        raise ValueError("Candidate FK returned incompatible batch dimensions.")
    if actual.device != seed.device or actual.dtype != seed.dtype:
        raise ValueError("Candidate FK outputs must match the seed device/dtype.")
    position_error = torch.linalg.vector_norm(
        actual[..., :3, 3] - poses[..., :3, 3], dim=-1
    )
    relative_rotation = actual[..., :3, :3].transpose(-2, -1) @ poses[..., :3, :3]
    cosine = (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    rotation_error = torch.acos(cosine.clamp(-1.0, 1.0))
    fk_valid = (
        torch.isfinite(actual).all(dim=(-2, -1))
        & (position_error <= _CANDIDATE_POSITION_TOLERANCE)
        & (rotation_error <= _CANDIDATE_ROTATION_TOLERANCE)
    )
    success &= fk_valid
    for row, column in torch.nonzero(active & ~success).tolist():
        reasons[row][column] = (
            "IK_NOT_FOUND"
            if not flags[row, column]
            else (
                "IK_INVALID_RESULT"
                if not finite[row, column]
                else (
                    "JOINT_LIMIT" if not within_limits[row, column] else "FK_MISMATCH"
                )
            )
        )
    return (
        success,
        torch.where(success[..., None], qpos, seed),
        tuple(tuple(row) for row in reasons),
    )


def _candidate_cartesian_segment(
    robot: Robot,
    begin_pose: torch.Tensor,
    end_pose: torch.Tensor,
    begin_qpos: torch.Tensor,
    end_qpos: torch.Tensor,
    manipulator: JointPositionTarget,
    context: PlanningContext,
    active: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str | None, ...]]:
    """Interpolate translation using previous IK seeds and exact certified endpoints."""
    if count < 2:
        raise ValueError("A Cartesian candidate segment requires at least two samples.")
    if not torch.allclose(
        begin_pose[active, :3, :3], end_pose[active, :3, :3], atol=1e-5, rtol=0
    ):
        raise ValueError("A translation segment must preserve the grasp rotation.")
    success = active.clone()
    reasons: list[str | None] = [None] * context.batch_size
    values = [begin_qpos]
    for waypoint in range(1, count):
        pose = begin_pose.clone()
        pose[:, :3, 3] = torch.lerp(
            begin_pose[:, :3, 3], end_pose[:, :3, 3], waypoint / (count - 1)
        )
        previous = values[-1]
        if waypoint == count - 1:
            # Keep the selected branch's anchor; never re-solve its endpoint.
            qpos, stage_valid = end_qpos.clone(), success.clone()
            stage_reasons = tuple((None,) for _ in range(context.batch_size))
        else:
            stage_valid, solved, stage_reasons = _checked_candidate_ik(
                robot,
                pose[:, None],
                previous[:, None],
                manipulator,
                context,
                success[:, None],
            )
            stage_valid, qpos = stage_valid[:, 0], solved[:, 0]
        continuous = (
            torch.amax(torch.abs(qpos - previous), dim=-1) <= _CANDIDATE_MAX_JOINT_STEP
        )
        for row in (
            torch.nonzero(success & ~(stage_valid & continuous)).flatten().tolist()
        ):
            reasons[row] = stage_reasons[row][0] or "BRANCH_DISCONTINUITY"
        success &= stage_valid & continuous
        values.append(torch.where(success[:, None], qpos, previous))
    return success, torch.stack(values, dim=1), tuple(reasons)


__all__: list[str] = []
