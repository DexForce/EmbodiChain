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

"""Tests for atomic-action goals, state, policies, effects, and plans."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    Affordance,
    EndEffectorPoseGoal,
    EntityState,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.goals import (
    collect_scene_dependencies,
    resolve_pose_goal,
)


def _semantics(label: str = "object") -> ObjectSemantics:
    return ObjectSemantics(affordance=Affordance(), geometry={}, label=label)


def _held(batch_size: int = 2) -> HeldObjectState:
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    return HeldObjectState(
        semantics=_semantics(),
        object_to_eef=pose,
        grasp_xpos=pose,
    )


def _context(scene: SceneSnapshot | None = None) -> PlanningContext:
    qpos = torch.zeros(2, 4)
    return PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState.empty(batch_size=2, device="cpu"),
        scene=scene or SceneSnapshot.empty(),
        env_ids=torch.tensor([4, 7], dtype=torch.long),
    )


def test_action_binding_is_role_based_and_immutable() -> None:
    binding = ActionBinding(
        manipulators={"primary": "left_arm"},
        end_effectors={"primary": "left_hand"},
    )

    assert binding.manipulator() == "left_arm"
    assert binding.end_effector() == "left_hand"
    with pytest.raises(TypeError):
        binding.manipulators["primary"] = "right_arm"
    with pytest.raises(KeyError, match="destination"):
        binding.manipulator("destination")


def test_invocation_rejects_values_without_goal_contract() -> None:
    with pytest.raises(TypeError, match="goal_kind"):
        ActionInvocation(
            skill_id="move_end_effector",
            goal=object(),  # type: ignore[arg-type]
            binding=ActionBinding(manipulators={"primary": "arm"}),
        )


def test_motion_and_recovery_policy_validate_shared_parameters() -> None:
    policy = MotionPolicy(sample_count=24, control_dt=0.01)
    assert policy.sample_count == 24
    assert policy.control_dt == 0.01
    with pytest.raises(ValueError, match="sample_count"):
        MotionPolicy(sample_count=1)
    with pytest.raises(ValueError, match="max_replans"):
        RecoveryPolicy(max_replans=-1)


def test_task_state_normalizes_held_relations_and_masks_updates() -> None:
    held = _held()
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"left_arm": held},
    )
    replacement = _held()
    updated = StateDelta(
        held_object_updates={
            "left_arm": None,
            "right_arm": replacement,
        }
    ).apply(state, torch.tensor([True, False]))

    left = updated.get_held_object("left_arm")
    right = updated.get_held_object("right_arm")
    assert left is not None and left.env_mask.tolist() == [False, True]
    assert right is not None and right.env_mask.tolist() == [True, False]
    assert state.get_held_object("right_arm") is None


def test_robot_observation_owns_input_tensors() -> None:
    qpos = torch.zeros(2, 4)
    observation = RobotObservation(
        timestamp=0.0,
        qpos=qpos,
        qvel=torch.zeros_like(qpos),
    )
    qpos.fill_(1.0)
    assert torch.count_nonzero(observation.qpos) == 0
    with pytest.raises(FrozenInstanceError):
        observation.timestamp = 2.0


def test_scene_entity_pose_is_resolved_late_from_snapshot() -> None:
    entity_pose = torch.eye(4).repeat(2, 1, 1)
    entity_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    offset = torch.eye(4)
    offset[2, 3] = 0.1
    reference = SceneEntityPose("cup", relative_pose=offset)
    context = _context(
        SceneSnapshot(
            timestamp=1.0,
            version=3,
            entities={"cup": EntityState(entity_pose, confidence=0.9)},
        )
    )

    resolved = resolve_pose_goal(reference, context, name="xpos")

    assert resolved[:, 0, 3].tolist() == pytest.approx([0.2, 0.4])
    assert resolved[:, 2, 3].tolist() == pytest.approx([0.1, 0.1])
    assert collect_scene_dependencies(EndEffectorPoseGoal(reference)) == ("cup",)


def test_scene_entity_pose_enforces_confidence() -> None:
    context = _context(
        SceneSnapshot(
            timestamp=1.0,
            version=1,
            entities={"cup": EntityState(torch.eye(4), confidence=0.2)},
        )
    )
    with pytest.raises(ValueError, match="confidence"):
        resolve_pose_goal(
            SceneEntityPose("cup", minimum_confidence=0.8),
            context,
            name="xpos",
        )


def test_timed_trajectory_synthesizes_timing_and_holds_selected_rows() -> None:
    positions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    trajectory = TimedTrajectory.from_positions(
        positions,
        env_ids=torch.tensor([4, 7]),
        control_dt=0.02,
    )
    held = trajectory.hold_rows(
        torch.tensor([True, False]),
        torch.full((2, 4), -1.0),
    )

    assert trajectory.duration.tolist() == pytest.approx([0.04, 0.04])
    assert torch.equal(held.positions[0], positions[0])
    assert torch.all(held.positions[1] == -1.0)


def test_timed_trajectory_concatenates_metadata() -> None:
    first = TimedTrajectory.from_positions(
        torch.zeros(2, 2, 4),
        env_ids=torch.tensor([0, 1]),
        control_dt=0.1,
    )
    second = TimedTrajectory.from_positions(
        torch.ones(2, 3, 4),
        env_ids=torch.tensor([0, 1]),
        control_dt=0.2,
    )

    result = TimedTrajectory.concatenate((first, second))

    assert result.positions.shape == (2, 5, 4)
    assert result.duration.tolist() == pytest.approx([0.5, 0.5])
