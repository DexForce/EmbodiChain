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

"""Deterministic resource, timing, and state contracts for parallel skills."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from embodichain.lab.sim.atomic_actions import (
    RuntimeCommandFrame,
    StateDelta,
    TaskState,
    TimedCommandSequence,
)

from embodichain.lab.semantic_skills.profiles import ResourceClaim


def _validate_identifier(value: str, *, field_name: str) -> None:
    """Validate one non-empty stable identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable identifier.")


@dataclass(frozen=True, slots=True)
class ParallelTimingPolicy:
    """Strict environment-grid policy for one parallel barrier.

    Fractional frame durations are rejected. Padding repeats the last
    controller target, which is a deterministic position/tool hold; no
    interpolation is hidden inside the scheduler.
    """

    step_dt: float
    tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        for field_name in ("step_dt", "tolerance"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be a number.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True, eq=False)
class ParallelBranchPlan:
    """One independently planned lane entering a common barrier."""

    branch_id: str
    claim: ResourceClaim
    commands: TimedCommandSequence
    expected_effects: StateDelta = StateDelta()

    def __post_init__(self) -> None:
        _validate_identifier(self.branch_id, field_name="branch_id")
        if not isinstance(self.claim, ResourceClaim):
            raise TypeError("claim must be a ResourceClaim.")
        if not isinstance(self.commands, TimedCommandSequence):
            raise TypeError("commands must be a TimedCommandSequence.")
        if not isinstance(self.expected_effects, StateDelta):
            raise TypeError("expected_effects must be a StateDelta.")
        object.__setattr__(self, "commands", self.commands.snapshot())
        object.__setattr__(self, "expected_effects", self.expected_effects.snapshot())


class ParallelConflictError(ValueError):
    """Raised before execution when parallel lanes claim overlapping resources."""


class ParallelTimingError(ValueError):
    """Raised when a command sequence cannot use the environment step grid."""


class ParallelStateConflictError(ValueError):
    """Raised when successful lanes update the same symbolic state row."""


def validate_parallel_claims(branches: tuple[ParallelBranchPlan, ...]) -> None:
    """Reject duplicate IDs and every pair of overlapping physical claims."""
    if not isinstance(branches, tuple) or len(branches) < 2:
        raise ValueError("Parallel execution requires at least two branch plans.")
    if not all(type(branch) is ParallelBranchPlan for branch in branches):
        raise TypeError("branches must contain exact ParallelBranchPlan values.")
    branch_ids = tuple(branch.branch_id for branch in branches)
    if len(set(branch_ids)) != len(branch_ids):
        raise ParallelConflictError("Parallel branch IDs must be unique.")
    for index, left in enumerate(branches):
        for right in branches[index + 1 :]:
            if left.claim.conflicts_with(right.claim):
                raise ParallelConflictError(
                    f"Parallel branches {left.branch_id!r} and "
                    f"{right.branch_id!r} have overlapping physical claims."
                )


def _validate_grid_frame(
    branch_id: str,
    frame_index: int,
    frame: RuntimeCommandFrame,
    policy: ParallelTimingPolicy,
) -> None:
    """Require one frame to occupy exactly one environment control step."""
    durations = frame.hold_duration
    expected = torch.full_like(durations, policy.step_dt)
    if not torch.allclose(durations, expected, atol=policy.tolerance, rtol=0.0):
        values = sorted({float(value) for value in durations.detach().cpu().tolist()})
        raise ParallelTimingError(
            f"Parallel branch {branch_id!r} frame {frame_index} has durations "
            f"{values}; every emitted frame must equal step_dt={policy.step_dt}."
        )


def align_parallel_commands(
    branches: tuple[ParallelBranchPlan, ...],
    policy: ParallelTimingPolicy,
) -> TimedCommandSequence:
    """Merge disjoint lanes on one grid and hold-pad shorter trajectories.

    Each merged frame is a single transport transaction. Runtime frame
    validation independently rejects duplicate destinations or joint overlap,
    defending against an incorrect custom ``ResourceClaim`` implementation.
    """
    if not isinstance(policy, ParallelTimingPolicy):
        raise TypeError("policy must be a ParallelTimingPolicy.")
    validate_parallel_claims(branches)
    first = branches[0].commands
    if any(
        branch.commands.device != first.device
        or not torch.equal(branch.commands.env_ids, first.env_ids)
        for branch in branches[1:]
    ):
        raise ParallelTimingError(
            "Parallel command sequences must share ordered env_ids and device."
        )
    if any(branch.commands.frame_count == 0 for branch in branches):
        raise ParallelTimingError(
            "Parallel branches must emit at least one command frame."
        )
    for branch in branches:
        for frame_index, frame in enumerate(branch.commands.frames):
            _validate_grid_frame(branch.branch_id, frame_index, frame, policy)

    frame_count = max(branch.commands.frame_count for branch in branches)
    merged: list[RuntimeCommandFrame] = []
    for frame_index in range(frame_count):
        lane_frames = tuple(
            branch.commands.frames[min(frame_index, branch.commands.frame_count - 1)]
            for branch in branches
        )
        reference_mask = lane_frames[0].active_mask
        if any(
            not torch.equal(frame.active_mask, reference_mask)
            for frame in lane_frames[1:]
        ):
            raise ParallelTimingError(
                "Parallel lanes cannot merge different per-environment active "
                f"masks at frame {frame_index}; RuntimeCommandFrame owns one "
                "mask for every command in the transaction."
            )
        merged.append(
            RuntimeCommandFrame(
                commands=tuple(
                    command for frame in lane_frames for command in frame.commands
                ),
                active_mask=reference_mask,
                env_ids=first.env_ids,
                hold_duration=torch.full(
                    (first.batch_size,),
                    policy.step_dt,
                    dtype=lane_frames[0].hold_duration.dtype,
                    device=first.device,
                ),
            )
        )
    return TimedCommandSequence(frames=tuple(merged), env_ids=first.env_ids)


def _delta_keys(delta: StateDelta) -> frozenset[tuple[str, object]]:
    """Return domain-qualified symbolic keys written by one delta."""
    return frozenset(
        [("held", key) for key in delta.held_object_updates]
        + [("coordinated", key) for key in delta.coordinated_held_object_updates]
        + [("articulation", key) for key in delta.articulation_joint_updates]
    )


def merge_parallel_effects(
    state: TaskState,
    effects: Mapping[str, tuple[StateDelta, torch.Tensor]],
) -> TaskState:
    """Apply disjoint branch effects with deterministic row-local conflict checks.

    Args:
        state: Verified task state before the barrier.
        effects: Branch ID to ``(delta, verified_success_mask)``.

    Returns:
        New verified task state after all non-conflicting updates.
    """
    if not isinstance(state, TaskState):
        raise TypeError("state must be a TaskState.")
    if not isinstance(effects, Mapping) or not effects:
        raise ValueError("effects must be a non-empty branch mapping.")
    normalized: dict[str, tuple[StateDelta, torch.Tensor]] = {}
    for branch_id, value in effects.items():
        _validate_identifier(branch_id, field_name="effect branch IDs")
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("effect entries must be (StateDelta, success_mask) pairs.")
        delta, mask = value
        if not isinstance(delta, StateDelta):
            raise TypeError("effect deltas must be StateDelta values.")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (state.batch_size,)
            or mask.device != state.device
        ):
            raise ValueError("effect masks must match TaskState batch and device.")
        normalized[branch_id] = delta.snapshot(), mask.clone()

    entries = tuple(normalized.items())
    for index, (left_id, (left_delta, left_mask)) in enumerate(entries):
        for right_id, (right_delta, right_mask) in entries[index + 1 :]:
            overlapping_keys = _delta_keys(left_delta) & _delta_keys(right_delta)
            overlapping_rows = left_mask & right_mask
            if overlapping_keys and overlapping_rows.any():
                raise ParallelStateConflictError(
                    f"Parallel effects {left_id!r} and {right_id!r} write "
                    f"the same symbolic keys on rows "
                    f"{overlapping_rows.nonzero().flatten().tolist()}."
                )
    result = state
    for branch_id in sorted(normalized):
        delta, mask = normalized[branch_id]
        result = delta.apply(result, mask)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ParallelBarrierUpdate:
    """Per-row barrier status after one synchronized lane observation."""

    completed_mask: torch.Tensor
    failure_mask: torch.Tensor
    cancellation_masks: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.completed_mask, torch.Tensor)
            or self.completed_mask.dtype != torch.bool
            or self.completed_mask.dim() != 1
        ):
            raise ValueError("completed_mask must be a one-dimensional bool tensor.")
        if (
            not isinstance(self.failure_mask, torch.Tensor)
            or self.failure_mask.dtype != torch.bool
            or self.failure_mask.shape != self.completed_mask.shape
            or self.failure_mask.device != self.completed_mask.device
        ):
            raise ValueError("failure_mask must match completed_mask.")
        cancellations: dict[str, torch.Tensor] = {}
        for branch_id, mask in self.cancellation_masks.items():
            _validate_identifier(branch_id, field_name="cancellation branch IDs")
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != self.completed_mask.shape
                or mask.device != self.completed_mask.device
            ):
                raise ValueError("cancellation masks must match completed_mask.")
            cancellations[branch_id] = mask.clone()
        object.__setattr__(self, "completed_mask", self.completed_mask.clone())
        object.__setattr__(self, "failure_mask", self.failure_mask.clone())
        object.__setattr__(
            self,
            "cancellation_masks",
            MappingProxyType(cancellations),
        )


def resolve_parallel_barrier(
    *,
    pending_masks: Mapping[str, torch.Tensor],
    success_masks: Mapping[str, torch.Tensor],
    failure_masks: Mapping[str, torch.Tensor],
) -> ParallelBarrierUpdate:
    """Apply deterministic per-row fail-fast semantics at one barrier update."""
    branch_ids = tuple(pending_masks)
    if (
        not branch_ids
        or set(success_masks) != set(branch_ids)
        or set(failure_masks) != set(branch_ids)
    ):
        raise ValueError(
            "pending, success, and failure mappings must share branch IDs."
        )
    reference = pending_masks[branch_ids[0]]
    if not isinstance(reference, torch.Tensor):
        raise TypeError("barrier masks must be torch.Tensor values.")
    for mapping in (pending_masks, success_masks, failure_masks):
        for mask in mapping.values():
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != reference.shape
                or mask.device != reference.device
            ):
                raise ValueError("all barrier masks must share bool shape and device.")
    failed = torch.stack(tuple(failure_masks.values()), dim=0).any(dim=0)
    succeeded_all = torch.stack(tuple(success_masks.values()), dim=0).all(dim=0)
    cancellations = {
        branch_id: failed & pending_masks[branch_id] for branch_id in branch_ids
    }
    return ParallelBarrierUpdate(
        completed_mask=succeeded_all | failed,
        failure_mask=failed,
        cancellation_masks=cancellations,
    )


__all__: list[str] = []
