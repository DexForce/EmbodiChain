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
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    Affordance,
    AtomicAction,
    DynamicCollisionMode,
    EndEffectorPoseGoal,
    EntityState,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionBinding,
    ResolvedActionRequest,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    StateDelta,
    TaskState,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.goals import (
    _resolve_object_pose,
    collect_scene_dependencies,
    resolve_pose_goal,
)
from embodichain.lab.sim.common import BatchEntity


def _semantics(
    label: str = "object",
    *,
    entity: BatchEntity | None = None,
    entity_id: str | None = None,
) -> ObjectSemantics:
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label=label,
        entity=entity,
        entity_id=entity_id,
    )


def _held(
    batch_size: int = 2,
    *,
    semantics: ObjectSemantics | None = None,
    env_mask: torch.Tensor | None = None,
) -> HeldObjectState:
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    return HeldObjectState(
        semantics=semantics or _semantics(),
        object_to_eef=pose,
        grasp_xpos=pose,
        env_mask=env_mask,
    )


def _context(scene: SceneSnapshot | None = None) -> PlanningContext:
    qpos = torch.zeros(2, 4)
    return PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState.empty(batch_size=2, device="cpu"),
        scene=scene or SceneSnapshot.empty(),
        env_ids=torch.tensor([4, 7], dtype=torch.long),
    )


class _DependencyAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Minimal action proving that build_plan delegates dependencies to its hook."""

    skill_id = "dependency_test"
    GoalType = EndEffectorPoseGoal
    OptionsType = ActionOptions
    manipulator_roles = ()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def _uses_collision_world(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> bool:
        del request, context
        return False

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
    ) -> tuple[str, ...]:
        dependencies = set(super()._scene_dependencies(request))
        dependencies.add("extra")
        return tuple(sorted(dependencies))

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        raise NotImplementedError


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

@pytest.mark.parametrize("entity_id", ["", "   ", 7])
def test_object_semantics_rejects_invalid_entity_id(entity_id: object) -> None:
    with pytest.raises(ValueError, match="entity_id"):
        ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            entity_id=entity_id,  # type: ignore[arg-type]
        )


def test_object_semantics_identity_fields_are_frozen() -> None:
    semantics = _semantics(entity_id="cube")

    with pytest.raises(FrozenInstanceError):
        semantics.entity_id = "other"  # type: ignore[misc]


def test_motion_and_recovery_policy_validate_shared_parameters() -> None:
    policy = MotionPolicy(sample_count=24, control_dt=0.01)
    assert policy.sample_count == 24
    assert policy.control_dt == 0.01
    assert policy.dynamic_collision_mode is DynamicCollisionMode.AUTO
    with pytest.raises(ValueError, match="sample_count"):
        MotionPolicy(sample_count=1)
    with pytest.raises(ValueError, match="strategy"):
        MotionPolicy(strategy="planner")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_replans"):
        RecoveryPolicy(max_replans=-1)


def test_motion_policy_normalizes_dynamic_collision_mode() -> None:
    assert (
        MotionPolicy(dynamic_collision_mode="off").dynamic_collision_mode
        is DynamicCollisionMode.OFF
    )
    with pytest.raises(ValueError, match="dynamic_collision_mode"):
        MotionPolicy(dynamic_collision_mode="unknown")
    with pytest.raises(TypeError, match="DynamicCollisionMode"):
        MotionPolicy(dynamic_collision_mode=object())  # type: ignore[arg-type]


def test_motion_policy_maps_to_motion_generator_strategy() -> None:
    policy = MotionPolicy(
        strategy="ik_interp",
        sample_count=24,
        velocity_limit=0.2,
        acceleration_limit=0.5,
    )
    start_qpos = torch.zeros(2, 6)

    options = policy.to_motion_gen_options(
        start_qpos=start_qpos,
        control_part="arm",
        sample_count=12,
    )

    assert options.strategy == "ik_interp"
    assert options.sample_count == 12
    assert options.start_qpos is not start_qpos
    assert torch.equal(options.start_qpos, start_qpos)
    assert options.control_part == "arm"
    assert options.velocity_limit == 0.2
    assert options.acceleration_limit == 0.5


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


def test_task_state_reports_per_environment_exclusive_holds() -> None:
    entity = Mock(spec=BatchEntity)
    shared = _semantics("shared", entity=entity)
    same_entity = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="same-entity-alias",
        entity=entity,
    )
    independent = _semantics("shared")
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={
            "left_arm": _held(semantics=shared),
            "right_arm": _held(
                semantics=same_entity,
                env_mask=torch.tensor([True, False]),
            ),
            "third_arm": _held(
                semantics=independent,
                env_mask=torch.tensor([False, True]),
            ),
        },
    )

    assert state.held_object_mask("left_arm").tolist() == [True, True]
    assert state.exclusive_held_object_mask("left_arm").tolist() == [False, True]
    assert state.exclusive_held_object_mask("right_arm").tolist() == [False, False]
    assert state.exclusive_held_object_mask("third_arm").tolist() == [False, True]
    assert state.held_object_mask("missing").tolist() == [False, False]


def test_task_state_treats_matching_entity_ids_as_shared() -> None:
    state = TaskState(
        batch_size=1,
        device="cpu",
        held_objects={
            "left_arm": _held(
                batch_size=1,
                semantics=_semantics(entity_id="tray"),
            ),
            "right_arm": _held(
                batch_size=1,
                semantics=_semantics(entity_id="tray"),
            ),
        },
    )

    assert state.exclusive_held_object_mask("left_arm").tolist() == [False]
    assert state.exclusive_held_object_mask("right_arm").tolist() == [False]


def test_state_delta_merges_distinct_semantics_with_same_entity_id() -> None:
    previous_semantics = _semantics(entity_id="cube")
    candidate_semantics = _semantics(entity_id="cube")
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"arm": _held(semantics=previous_semantics)},
    )

    updated = StateDelta(
        held_object_updates={
            "arm": _held(semantics=candidate_semantics),
        }
    ).apply(state, torch.tensor([True, False]))

    held = updated.get_held_object("arm")
    assert previous_semantics is not candidate_semantics
    assert held is not None and held.semantics is previous_semantics


def test_state_delta_replaces_semantics_when_all_rows_are_updated() -> None:
    previous_semantics = _semantics(entity_id="cube")
    candidate_semantics = _semantics(entity_id="cube")
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"arm": _held(semantics=previous_semantics)},
    )

    updated = StateDelta(
        held_object_updates={
            "arm": _held(semantics=candidate_semantics),
        }
    ).apply(state, torch.tensor([True, True]))

    held = updated.get_held_object("arm")
    assert held is not None and held.semantics is candidate_semantics


def test_state_delta_rejects_partial_merge_of_different_entity_ids() -> None:
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"arm": _held(semantics=_semantics(entity_id="cube"))},
    )
    delta = StateDelta(
        held_object_updates={
            "arm": _held(semantics=_semantics(entity_id="cup")),
        }
    )

    with pytest.raises(ValueError, match="different held-object semantics"):
        delta.apply(state, torch.tensor([True, False]))


def test_state_delta_does_not_match_explicit_id_to_legacy_uid() -> None:
    shared_entity = Mock(uid="cube")
    previous_semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=shared_entity,
        entity_id="cube",
    )
    candidate_semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=shared_entity,
    )
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"arm": _held(semantics=previous_semantics)},
    )
    delta = StateDelta(
        held_object_updates={"arm": _held(semantics=candidate_semantics)}
    )

    with pytest.raises(ValueError, match="different held-object semantics"):
        delta.apply(state, torch.tensor([True, False]))


def test_state_delta_merges_legacy_semantics_with_same_uid() -> None:
    previous_semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=Mock(uid="cube"),
    )
    candidate_semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=Mock(uid="cube"),
    )
    state = TaskState(
        batch_size=2,
        device="cpu",
        held_objects={"arm": _held(semantics=previous_semantics)},
    )

    updated = StateDelta(
        held_object_updates={
            "arm": _held(semantics=candidate_semantics),
        }
    ).apply(state, torch.tensor([True, False]))

    held = updated.get_held_object("arm")
    assert held is not None and held.semantics is previous_semantics


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


def test_object_pose_uses_explicit_scene_id_without_live_fallback() -> None:
    scene_pose = torch.eye(4).repeat(2, 1, 1)
    scene_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    entity = Mock()
    entity.get_local_pose.return_value = torch.full((2, 4, 4), 9.0)
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=entity,
        entity_id="cup",
    )
    context = _context(
        SceneSnapshot(
            timestamp=1.0,
            version=1,
            entities={"cup": EntityState(scene_pose)},
        )
    )

    resolved = _resolve_object_pose(semantics, context)

    assert torch.equal(resolved, scene_pose)
    entity.get_local_pose.assert_not_called()


def test_object_pose_missing_explicit_scene_id_does_not_fall_back() -> None:
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4).repeat(2, 1, 1)
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=entity,
        entity_id="missing",
    )

    with pytest.raises(KeyError, match="unknown scene entity"):
        _resolve_object_pose(semantics, _context())
    entity.get_local_pose.assert_not_called()


def test_object_pose_legacy_entity_warns_and_broadcasts() -> None:
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4)
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity=entity,
    )

    with pytest.warns(DeprecationWarning, match="entity_id"):
        resolved = _resolve_object_pose(semantics, _context())

    assert resolved.shape == (2, 4, 4)
    entity.get_local_pose.assert_called_once_with(to_matrix=True)


def test_dependency_collection_does_not_descend_object_semantics() -> None:
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        properties={"unrelated_pose": SceneEntityPose("hidden")},
        entity_id="object",
    )

    assert collect_scene_dependencies(semantics) == ()


def test_build_plan_uses_action_scene_dependency_hook() -> None:
    context = _context()
    request = ResolvedActionRequest(
        skill_id="dependency_test",
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ResolvedActionBinding(),
        motion_policy=MotionPolicy(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )
    action = _DependencyAction()

    plan = action.build_plan(
        request,
        context,
        success=True,
        trajectory=context.robot.qpos.unsqueeze(1),
        diagnostics=PlannerDiagnostics(backend="test"),
    )

    assert plan.scene_dependencies == ("extra", "tracked")


def test_scene_snapshot_expands_global_collision_world_revision() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    snapshot = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"obstacle": EntityState(pose)},
        collision_world_revision=3,
        collision_entity_ids=("obstacle",),
    )

    assert snapshot.collision_world_revisions(2) == (3, 3)
    obstacle_poses = snapshot.collision_obstacle_poses(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(obstacle_poses["obstacle"], pose)


def test_scene_snapshot_rejects_unknown_collision_entity() -> None:
    with pytest.raises(ValueError, match="missing scene entities"):
        SceneSnapshot(
            timestamp=0.0,
            version=0,
            collision_entity_ids=("missing",),
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


def test_timed_trajectory_snapshot_owns_its_tensor_storage() -> None:
    trajectory = TimedTrajectory.from_positions(
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        env_ids=torch.tensor([4]),
        control_dt=0.02,
    )

    snapshot = trajectory.snapshot()
    snapshot.positions.zero_()
    snapshot.dt.zero_()
    snapshot.env_ids.zero_()

    assert snapshot.duration.item() == 0.0
    assert torch.count_nonzero(trajectory.positions).item() > 0
    assert torch.count_nonzero(trajectory.dt).item() > 0
    assert trajectory.duration.item() > 0.0
    assert trajectory.env_ids.tolist() == [4]


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
