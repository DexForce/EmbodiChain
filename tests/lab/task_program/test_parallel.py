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

"""Tests for deterministic parallel-skill contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ArticulationJointState,
    EndpointCommand,
    JointPositionPayload,
    JointPositionTarget,
    RuntimeCommandFrame,
    StateDelta,
    TaskState,
    TimedCommandSequence,
)
from embodichain.lab.task_program.runtime.parallel import (
    ParallelBranchPlan,
    ParallelConflictError,
    ParallelStateConflictError,
    ParallelTimingError,
    ParallelTimingPolicy,
    align_parallel_commands,
    merge_parallel_effects,
    resolve_parallel_barrier,
)
from embodichain.lab.task_program.semantics.profiles import ResourceClaim

ENV_IDS = torch.tensor([3, 7], dtype=torch.long)


def _sequence(
    control_part: str,
    joint_id: int,
    frame_count: int,
    *,
    duration: float = 0.1,
) -> TimedCommandSequence:
    target = JointPositionTarget(control_part, (joint_id,))
    frames = tuple(
        RuntimeCommandFrame(
            commands=(
                EndpointCommand(
                    target,
                    JointPositionPayload(torch.full((2, 1), float(index + joint_id))),
                ),
            ),
            active_mask=torch.tensor([True, True]),
            env_ids=ENV_IDS,
            hold_duration=torch.full((2,), duration),
        )
        for index in range(frame_count)
    )
    return TimedCommandSequence(frames, ENV_IDS)


def _branch(
    branch_id: str,
    control_part: str,
    joint_id: int,
    frame_count: int,
    *,
    duration: float = 0.1,
) -> ParallelBranchPlan:
    return ParallelBranchPlan(
        branch_id=branch_id,
        claim=ResourceClaim(frozenset({control_part}), (joint_id,)),
        commands=_sequence(
            control_part,
            joint_id,
            frame_count,
            duration=duration,
        ),
    )


def test_parallel_alignment_hold_pads_shorter_disjoint_branch() -> None:
    merged = align_parallel_commands(
        (
            _branch("left", "left_arm", 0, 2),
            _branch("right", "right_arm", 1, 3),
        ),
        ParallelTimingPolicy(step_dt=0.1),
    )

    assert merged.frame_count == 3
    assert all(len(frame.commands) == 2 for frame in merged.frames)
    left_final = merged.frames[-1].commands[0].payload
    assert isinstance(left_final, JointPositionPayload)
    assert torch.equal(left_final.positions, torch.full((2, 1), 1.0))
    assert torch.equal(merged.frames[-1].active_mask, torch.tensor([True, True]))


def test_parallel_alignment_rejects_claim_and_grid_conflicts() -> None:
    with pytest.raises(ParallelConflictError, match="overlapping"):
        align_parallel_commands(
            (
                _branch("one", "arm", 0, 2),
                _branch("two", "arm", 1, 2),
            ),
            ParallelTimingPolicy(0.1),
        )


def test_parallel_alignment_rejects_different_lane_active_masks() -> None:
    left = _branch("left", "left", 0, 1)
    right = _branch("right", "right", 1, 1)
    right_frame = right.commands.frames[0].with_active_mask(torch.tensor([False, True]))
    right = ParallelBranchPlan(
        branch_id=right.branch_id,
        claim=right.claim,
        commands=TimedCommandSequence((right_frame,), ENV_IDS),
    )

    with pytest.raises(ParallelTimingError, match="active masks"):
        align_parallel_commands(
            (left, right),
            ParallelTimingPolicy(0.1),
        )


def test_parallel_alignment_validates_inactive_row_durations_on_same_grid() -> None:
    left = _branch("left", "left", 0, 1)
    left_frame = RuntimeCommandFrame(
        commands=left.commands.frames[0].commands,
        active_mask=torch.tensor([False, True]),
        env_ids=ENV_IDS,
        hold_duration=torch.tensor([0.2, 0.1]),
    )
    left = ParallelBranchPlan(
        branch_id=left.branch_id,
        claim=left.claim,
        commands=TimedCommandSequence((left_frame,), ENV_IDS),
    )
    right = _branch("right", "right", 1, 1)
    right_frame = RuntimeCommandFrame(
        commands=right.commands.frames[0].commands,
        active_mask=torch.tensor([False, True]),
        env_ids=ENV_IDS,
        hold_duration=torch.tensor([0.1, 0.1]),
    )
    right = ParallelBranchPlan(
        branch_id=right.branch_id,
        claim=right.claim,
        commands=TimedCommandSequence((right_frame,), ENV_IDS),
    )

    with pytest.raises(ParallelTimingError, match="step_dt"):
        align_parallel_commands((left, right), ParallelTimingPolicy(0.1))


def test_parallel_alignment_rejects_off_grid_duration() -> None:
    with pytest.raises(ParallelTimingError, match="step_dt"):
        align_parallel_commands(
            (
                _branch("left", "left", 0, 2, duration=0.05),
                _branch("right", "right", 1, 2),
            ),
            ParallelTimingPolicy(0.1),
        )


def test_parallel_effects_merge_disjoint_keys_by_verified_row() -> None:
    state = TaskState.empty(batch_size=2, device="cpu")
    merged = merge_parallel_effects(
        state,
        {
            "drawer": (
                StateDelta(
                    articulation_joint_updates={
                        ("drawer", "slide"): ArticulationJointState(torch.tensor([0.4]))
                    }
                ),
                torch.tensor([True, False]),
            ),
            "door": (
                StateDelta(
                    articulation_joint_updates={
                        ("door", "hinge"): ArticulationJointState(torch.tensor([1.0]))
                    }
                ),
                torch.tensor([False, True]),
            ),
        },
    )

    drawer = merged.get_articulation_joint_state("drawer", "slide")
    door = merged.get_articulation_joint_state("door", "hinge")
    assert drawer is not None and door is not None
    assert torch.equal(drawer.env_mask, torch.tensor([True, False]))
    assert torch.equal(door.env_mask, torch.tensor([False, True]))


def test_parallel_effects_reject_same_key_on_same_row() -> None:
    delta = StateDelta(
        articulation_joint_updates={
            ("drawer", "slide"): ArticulationJointState(torch.tensor([0.4]))
        }
    )
    with pytest.raises(ParallelStateConflictError, match="same symbolic keys"):
        merge_parallel_effects(
            TaskState.empty(2, "cpu"),
            {
                "one": (delta, torch.tensor([True, False])),
                "two": (delta, torch.tensor([True, True])),
            },
        )


def test_parallel_barrier_cancels_pending_siblings_per_failed_row() -> None:
    update = resolve_parallel_barrier(
        pending_masks={
            "left": torch.tensor([False, True, True]),
            "right": torch.tensor([True, False, True]),
        },
        success_masks={
            "left": torch.tensor([True, False, False]),
            "right": torch.tensor([False, True, False]),
        },
        failure_masks={
            "left": torch.tensor([False, False, True]),
            "right": torch.tensor([False, False, False]),
        },
    )

    assert torch.equal(update.failure_mask, torch.tensor([False, False, True]))
    assert torch.equal(update.completed_mask, torch.tensor([False, False, True]))
    assert torch.equal(
        update.cancellation_masks["right"],
        torch.tensor([False, False, True]),
    )


__all__: list[str] = []
