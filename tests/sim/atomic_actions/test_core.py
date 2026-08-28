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

from dataclasses import dataclass, FrozenInstanceError, replace
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
    AtomicActionEngine,
    DynamicCollisionMode,
    EndpointBinding,
    EndpointCommand,
    EndpointTrackingChannelBinding,
    EndpointTrackingFeedbackAddress,
    EndEffectorPoseGoal,
    EntityState,
    EffectVerificationRequirement,
    HeldObjectState,
    JointPositionPayload,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    PlanningFailure,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    RuntimeCommandFrame,
    RuntimeEndpointTarget,
    SceneEntityPose,
    SceneSnapshot,
    SkillBindingContract,
    StateDelta,
    TaskState,
    TimedCommandSequence,
    TimedTerminalAcceptance,
    TimedTrackingSequence,
    TimedTrajectory,
    TrackingFeedbackSourceRef,
    TrackingFrame,
    TrackingPolicy,
    TrackingProjectorRef,
    TrackingSetpoint,
    JointPositionTrackingState,
)
from embodichain.lab.sim.atomic_actions.goals import (
    _resolve_object_pose,
    collect_scene_dependencies,
    resolve_pose_goal,
)
from embodichain.lab.sim.planners import ToppraPlanOptions


def _semantics(
    label: str = "object",
    *,
    entity_id: str | None = None,
) -> ObjectSemantics:
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label=label,
        entity_id=label if entity_id is None else entity_id,
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


def _context(
    scene: SceneSnapshot | None = None,
    *,
    control_dt: float | None = None,
) -> PlanningContext:
    qpos = torch.zeros(2, 4)
    return PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState.empty(batch_size=2, device="cpu"),
        scene=scene or SceneSnapshot.empty(),
        env_ids=torch.tensor([4, 7], dtype=torch.long),
        control_dt=control_dt,
    )


def _command_sequence(
    *,
    env_ids: torch.Tensor,
    frame_count: int,
    targets: tuple[JointPositionTarget, ...] | None = None,
    positions: tuple[torch.Tensor, ...] | None = None,
    velocities: tuple[torch.Tensor | None, ...] | None = None,
) -> TimedCommandSequence:
    batch_size = int(env_ids.shape[0])
    if targets is None:
        target = JointPositionTarget("arm", (0, 1))
        targets = (target,) * frame_count
    if len(targets) != frame_count:
        raise ValueError("targets must contain one value per command frame.")
    if positions is not None and len(positions) != frame_count:
        raise ValueError("positions must contain one value per command frame.")
    if velocities is not None and len(velocities) != frame_count:
        raise ValueError("velocities must contain one value per command frame.")
    frames = tuple(
        RuntimeCommandFrame(
            commands=(
                EndpointCommand(
                    target=targets[index],
                    payload=JointPositionPayload(
                        (
                            torch.full(
                                (batch_size, len(targets[index].joint_ids)),
                                float(index + 1),
                                device=env_ids.device,
                            )
                            if positions is None
                            else positions[index]
                        ),
                        velocities=(None if velocities is None else velocities[index]),
                    ),
                ),
            ),
            active_mask=torch.ones(
                batch_size,
                dtype=torch.bool,
                device=env_ids.device,
            ),
            env_ids=env_ids,
            hold_duration=torch.full(
                (batch_size,),
                0.1,
                device=env_ids.device,
            ),
        )
        for index in range(frame_count)
    )
    return TimedCommandSequence(frames=frames, env_ids=env_ids)


class _AlternateJointPositionTarget(JointPositionTarget):
    """Distinct exact target type sharing joint-position transport semantics."""


@dataclass(frozen=True, slots=True)
class _ClaimedTarget(RuntimeEndpointTarget):
    """Non-joint target used to verify binding claim authorization."""

    name: str

    @property
    def transport_id(self) -> str:
        return JointPositionTarget.TRANSPORT_ID

    @property
    def target_id(self) -> str:
        return self.name


def _action_plan(
    commands: TimedCommandSequence,
    *,
    plan_success: torch.Tensor | None = None,
    joint_trajectory: TimedTrajectory | None = None,
    tracking_policy: TrackingPolicy | None = None,
    tracking: TimedTrackingSequence | None = None,
    expected_effects: StateDelta | None = None,
    effect_candidates: StateDelta | None = None,
    effect_verification: EffectVerificationRequirement | None = None,
    diagnostics: PlannerDiagnostics | None = None,
    scene_dependencies: tuple[str, ...] = (),
    scene_dependency_monitor_until: dict[str, int] | None = None,
) -> ActionPlan:
    if plan_success is None:
        plan_success = torch.ones(
            commands.batch_size,
            dtype=torch.bool,
            device=commands.device,
        )
    return ActionPlan(
        skill_id="test",
        plan_success=plan_success,
        commands=commands,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=(
            TrackingPolicy.timed() if tracking_policy is None else tracking_policy
        ),
        planned_scene_version=0,
        planned_collision_world_revision=(0,) * commands.batch_size,
        diagnostics=(
            PlannerDiagnostics(backend="test") if diagnostics is None else diagnostics
        ),
        tracking=tracking,
        joint_trajectory=joint_trajectory,
        scene_dependencies=scene_dependencies,
        scene_dependency_monitor_until=(
            {}
            if scene_dependency_monitor_until is None
            else scene_dependency_monitor_until
        ),
        expected_effects=StateDelta() if expected_effects is None else expected_effects,
        effect_candidates=(
            StateDelta() if effect_candidates is None else effect_candidates
        ),
        effect_verification=effect_verification,
    )


def _joint_tracking_sequence(
    commands: TimedCommandSequence,
) -> TimedTrackingSequence:
    frames: list[TrackingFrame] = []
    for command_frame in commands.frames:
        setpoints: list[TrackingSetpoint] = []
        for command in command_frame.commands:
            assert isinstance(command.target, JointPositionTarget)
            assert isinstance(command.payload, JointPositionPayload)
            channel = EndpointTrackingChannelBinding(
                channel_id="joint.position",
                source=TrackingFeedbackSourceRef(
                    provider_id="planning_context.robot",
                    revision="1",
                    address=EndpointTrackingFeedbackAddress(
                        target=command.target,
                        channel_id="joint.position",
                    ),
                ),
                projector=TrackingProjectorRef(
                    projector_id="joint_position_payload",
                    revision="1",
                ),
            )
            setpoints.append(
                TrackingSetpoint(
                    endpoint_key=("primary", "motion"),
                    binding=channel,
                    desired=JointPositionTrackingState(command.payload.positions),
                )
            )
        frames.append(TrackingFrame(tuple(setpoints)))
    return TimedTrackingSequence(commands.env_ids, tuple(frames))


@pytest.mark.parametrize("kind", ("", " physical", "physical ", 1, True))
def test_effect_verification_requirement_rejects_invalid_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        EffectVerificationRequirement(kind=kind)  # type: ignore[arg-type]


def test_action_plan_owns_explicit_effect_verification_requirement() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )
    requirement = EffectVerificationRequirement(kind="physical.effect")

    plan = _action_plan(commands, effect_verification=requirement)
    requirement_snapshot = plan.effect_verification

    assert plan.requires_effect_verification is True
    assert requirement_snapshot is not None
    assert requirement_snapshot is not requirement
    assert requirement_snapshot.kind == requirement.kind
    assert requirement_snapshot.snapshot() is not requirement_snapshot


def test_action_plan_implicitly_verifies_nonempty_state_delta() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )
    effects = StateDelta(held_object_updates={"arm": _held(batch_size=1)})

    implicit = _action_plan(commands, expected_effects=effects)
    no_effect = _action_plan(commands)

    assert implicit.effect_verification is None
    assert implicit.requires_effect_verification is True
    assert no_effect.requires_effect_verification is False


def test_action_plan_owns_nonterminal_attachment_candidates() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )
    held = _held(batch_size=1)
    candidates = StateDelta(held_object_updates={"arm": held})

    plan = _action_plan(commands, effect_candidates=candidates)
    owned = plan.effect_candidates.held_object_updates["arm"]

    assert owned is not held
    assert isinstance(owned, HeldObjectState)
    assert plan.requires_effect_verification is False
    assert plan.snapshot().effect_candidates is not plan.effect_candidates


def test_action_plan_rejects_effect_candidate_removals() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )

    with pytest.raises(ValueError, match="attached held-object states"):
        _action_plan(
            commands,
            effect_candidates=StateDelta(held_object_updates={"arm": None}),
        )


def test_action_plan_rejects_untyped_effect_verification_requirement() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )

    with pytest.raises(TypeError, match="EffectVerificationRequirement"):
        _action_plan(
            commands,
            effect_verification=object(),  # type: ignore[arg-type]
        )


class _DependencyAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Minimal action proving that build_plan delegates dependencies to its hook."""

    skill_id = "dependency_test"
    GoalType = EndEffectorPoseGoal
    OptionsType = ActionOptions
    binding_contract = SkillBindingContract()

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


class _RawCommandAction(AtomicAction[EndEffectorPoseGoal, ActionOptions]):
    """Action that deliberately bypasses build_command_plan for validation."""

    skill_id = "raw_command_test"
    GoalType = EndEffectorPoseGoal
    OptionsType = ActionOptions
    binding_contract = SkillBindingContract()

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

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        del request
        return ActionPlan(
            skill_id=self.skill_id,
            plan_success=torch.ones(context.batch_size, dtype=torch.bool),
            commands=_command_sequence(
                env_ids=context.env_ids,
                frame_count=1,
            ),
            recovery_policy=RecoveryPolicy(),
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=context.scene.version,
            planned_collision_world_revision=(0,) * context.batch_size,
            diagnostics=PlannerDiagnostics(backend="test"),
        )


def test_action_binding_is_endpoint_based_and_immutable() -> None:
    endpoint = EndpointBinding(
        slot_id="primary",
        endpoint_id="motion",
        resource_id="left_actor",
        adapter_id="control_part",
        target=JointPositionTarget("left_arm", (0, 1)),
        capabilities=frozenset({"motion.test"}),
        claim_tokens=frozenset({"robot.control_part:left_arm"}),
    )
    binding = ActionBinding(
        owner_id="test-engine",
        endpoints=(endpoint,),
    )

    resolved = binding.endpoint("primary", "motion")
    target = resolved.require_target(JointPositionTarget)
    assert resolved is not binding.endpoints[0]
    assert resolved.target is not binding.endpoints[0].target
    assert target.control_part == "left_arm"
    assert target.joint_ids == (0, 1)
    assert resolved.joint_ids == (0, 1)
    assert resolved.capabilities == frozenset({"motion.test"})
    with pytest.raises(FrozenInstanceError):
        binding.owner_id = "other-engine"  # type: ignore[misc]
    with pytest.raises(KeyError, match="destination.motion"):
        binding.endpoint("destination", "motion")
    with pytest.raises(ValueError, match="must match"):
        EndpointBinding(
            slot_id="primary",
            endpoint_id="motion",
            resource_id="left_actor",
            adapter_id="control_part",
            target=JointPositionTarget("left_arm", (0, 1)),
            joint_ids=(1, 2),
        )


@pytest.mark.parametrize("entity_id", ["", "   ", 7])
def test_object_semantics_rejects_invalid_entity_id(entity_id: object) -> None:
    with pytest.raises(ValueError, match="entity_id"):
        ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            entity_id=entity_id,  # type: ignore[arg-type]
        )


def test_object_semantics_requires_entity_id() -> None:
    with pytest.raises(TypeError, match="entity_id"):
        ObjectSemantics(  # type: ignore[call-arg]
            affordance=Affordance(),
            geometry={},
        )


def test_object_semantics_identity_fields_are_frozen() -> None:
    semantics = _semantics(entity_id="cube")

    with pytest.raises(FrozenInstanceError):
        semantics.entity_id = "other"  # type: ignore[misc]


def test_motion_and_recovery_policy_validate_shared_parameters() -> None:
    policy = MotionPolicy(sample_count=24)
    assert policy.sample_count == 24
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
    planner_options = ToppraPlanOptions(
        constraints={"velocity": 0.2, "acceleration": 0.5}
    )
    policy = MotionPolicy(
        strategy="ik_interp",
        sample_count=24,
        plan_opts=planner_options,
    )
    planner_options.constraints["velocity"] = 1.0
    start_qpos = torch.zeros(2, 6)

    options = policy.to_motion_gen_options(
        start_qpos=start_qpos,
        control_part="arm",
        sample_count=12,
        interpolation_dt=0.02,
    )

    assert options.strategy == "ik_interp"
    assert options.sample_count == 12
    assert options.start_qpos is not start_qpos
    assert torch.equal(options.start_qpos, start_qpos)
    assert options.control_part == "arm"
    assert options.interpolation_dt == pytest.approx(0.02)
    assert options.velocity_limit is None
    assert options.acceleration_limit is None
    assert isinstance(options.plan_opts, ToppraPlanOptions)
    assert options.plan_opts.constraints == {"velocity": 0.2, "acceleration": 0.5}


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
    shared = _semantics("shared", entity_id="shared-object")
    same_entity = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="same-entity-alias",
        entity_id="shared-object",
    )
    independent = _semantics("shared", entity_id="independent-object")
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
    offset[2, 3] = 9.0
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


def test_object_pose_uses_scene_snapshot() -> None:
    scene_pose = torch.eye(4).repeat(2, 1, 1)
    scene_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
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


def test_object_pose_rejects_missing_scene_entity() -> None:
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity_id="missing",
    )

    with pytest.raises(KeyError, match="unknown scene entity"):
        _resolve_object_pose(semantics, _context())


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
    generator = Mock()
    generator.robot = Mock()
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "test"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _DependencyAction()
    engine.register(action)
    request = ResolvedActionRequest(
        skill_id="dependency_test",
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ActionBinding(owner_id=engine.binding_owner_id),
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )

    with pytest.raises(TypeError, match="TimedTrajectory with explicit dt"):
        action.build_plan(
            request,
            context,
            success=True,
            trajectory=context.robot.qpos.unsqueeze(1),  # type: ignore[arg-type]
        )

    plan = action.build_command_plan(
        request,
        context,
        success=True,
        commands=TimedCommandSequence(frames=(), env_ids=context.env_ids),
        diagnostics=PlannerDiagnostics(backend="test"),
    )

    assert plan.scene_dependencies == ("extra", "tracked")


def test_build_segments_omits_zero_length_entry_and_preserves_offsets() -> None:
    approach_length = 2
    release_length = 3
    segment_lengths = {
        "approach": approach_length,
        "hold": 0,
        "release": release_length,
    }

    segments = AtomicAction._build_segments(
        segment_lengths,
        frame_count=sum(segment_lengths.values()),
    )

    assert tuple(
        (segment.name, segment.start, segment.stop) for segment in segments
    ) == (
        ("approach", 0, approach_length),
        ("release", approach_length, approach_length + release_length),
    )


def test_build_command_plan_rejects_unbound_runtime_destination() -> None:
    context = _context()
    generator = Mock()
    generator.robot = Mock()
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "test"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _DependencyAction()
    engine.register(action)
    request = ResolvedActionRequest(
        skill_id="dependency_test",
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ActionBinding(owner_id=engine.binding_owner_id),
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )

    with pytest.raises(ValueError, match="not authorized"):
        action.build_command_plan(
            request,
            context,
            success=True,
            commands=_command_sequence(env_ids=context.env_ids, frame_count=1),
            diagnostics=PlannerDiagnostics(backend="test"),
        )


def test_public_plan_authorizes_raw_action_plan_destinations() -> None:
    context = _context()
    generator = Mock()
    generator.robot = Mock()
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "test"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _RawCommandAction()
    engine.register(action)
    request = ResolvedActionRequest(
        skill_id=action.skill_id,
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ActionBinding(owner_id=engine.binding_owner_id),
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )

    with pytest.raises(ValueError, match="not authorized"):
        action.plan(request, context)


def test_command_target_authorization_rejects_altered_joint_claims() -> None:
    context = _context()
    request = ResolvedActionRequest(
        skill_id="dependency_test",
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ActionBinding(
            owner_id="test-engine",
            endpoints=(
                EndpointBinding(
                    slot_id="primary",
                    endpoint_id="motion",
                    resource_id="arm",
                    adapter_id="control_part",
                    target=JointPositionTarget("arm", (0, 1)),
                ),
            ),
        ),
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )
    frame = RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget("arm", (2, 3)),
                payload=JointPositionPayload(torch.ones(2, 2)),
            ),
        ),
        active_mask=torch.ones(2, dtype=torch.bool),
        env_ids=context.env_ids,
        hold_duration=torch.full((2,), 0.1),
    )

    with pytest.raises(ValueError, match="bound joint-position target"):
        _DependencyAction._authorize_command_targets(
            request,
            TimedCommandSequence(frames=(frame,), env_ids=context.env_ids),
        )


def test_command_target_authorization_rejects_custom_claim_conflicts() -> None:
    context = _context()
    endpoints = tuple(
        EndpointBinding(
            slot_id="primary",
            endpoint_id=name,
            resource_id=name,
            adapter_id="test.claimed",
            target=_ClaimedTarget(name),
            claim_tokens=frozenset({"controller:shared"}),
        )
        for name in ("first", "second")
    )
    request = ResolvedActionRequest(
        skill_id="dependency_test",
        goal=EndEffectorPoseGoal(SceneEntityPose("tracked")),
        binding=ActionBinding(owner_id="test-engine", endpoints=endpoints),
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        recovery_policy=RecoveryPolicy(),
        skill_options=ActionOptions(),
    )
    frame = RuntimeCommandFrame(
        commands=tuple(
            EndpointCommand(
                target=endpoint.target,
                payload=JointPositionPayload(torch.ones(2, 1)),
            )
            for endpoint in endpoints
        ),
        active_mask=torch.ones(2, dtype=torch.bool),
        env_ids=context.env_ids,
        hold_duration=torch.full((2,), 0.1),
    )

    with pytest.raises(ValueError, match="claim tokens.*controller:shared"):
        _DependencyAction._authorize_command_targets(
            request,
            TimedCommandSequence(frames=(frame,), env_ids=context.env_ids),
        )


def test_action_plan_rejects_unknown_scene_dependency_end_segment() -> None:
    plan = _action_plan(
        _command_sequence(
            env_ids=torch.tensor([0, 1], dtype=torch.long),
            frame_count=2,
        )
    )

    with pytest.raises(
        ValueError,
        match="scene_dependency_end_segment must name an ActionPlan segment",
    ):
        replace(
            plan,
            scene_dependencies=("target",),
            scene_dependency_end_segment="approach",
        )


def test_action_plan_owns_commands_and_optional_joint_trajectory() -> None:
    env_ids = torch.tensor([4, 7], dtype=torch.long)
    commands = _command_sequence(env_ids=env_ids, frame_count=2)
    trajectory_positions = torch.stack(
        (
            torch.full((2, 2), 1.0),
            torch.full((2, 2), 2.0),
        ),
        dim=1,
    )
    trajectory = TimedTrajectory.from_uniform_step(
        trajectory_positions,
        env_ids=env_ids,
        step_dt=0.1,
    )
    plan_success = torch.tensor([True, False])

    plan = _action_plan(
        commands,
        plan_success=plan_success,
        joint_trajectory=trajectory,
        diagnostics=PlannerDiagnostics(
            backend="test",
            failure=PlanningFailure("planning_failed", retryable=True),
        ),
    )
    payload = commands.frames[0].commands[0].payload
    assert isinstance(payload, JointPositionPayload)
    plan_success.zero_()
    payload.positions.zero_()
    commands.frames[0].active_mask.zero_()
    commands.frames[0].hold_duration.zero_()
    commands.env_ids.zero_()
    trajectory.positions.zero_()

    owned_payload = plan.commands.frames[0].commands[0].payload
    assert isinstance(owned_payload, JointPositionPayload)
    assert plan.plan_success.tolist() == [True, False]
    assert torch.all(owned_payload.positions == 1.0)
    assert plan.commands.frames[0].active_mask.tolist() == [True, True]
    assert torch.all(plan.commands.frames[0].hold_duration == 0.1)
    assert plan.commands.env_ids.tolist() == [4, 7]
    assert plan.joint_trajectory is not None
    assert torch.equal(plan.joint_trajectory.positions, trajectory_positions)


def test_planner_diagnostics_and_plan_snapshots_own_nested_metadata() -> None:
    nested = {"solver": {"iterations": [3, 5]}}
    diagnostics = PlannerDiagnostics(backend="test", metadata=nested)
    plan = _action_plan(
        _command_sequence(
            env_ids=torch.tensor([4], dtype=torch.long),
            frame_count=1,
        ),
        diagnostics=diagnostics,
    )

    nested["solver"]["iterations"][0] = 99
    diagnostics.metadata["solver"]["iterations"][1] = 77
    snapshot = plan.snapshot()
    plan.diagnostics.metadata["solver"]["iterations"][0] = 42

    assert snapshot.diagnostics.metadata["solver"]["iterations"] == [3, 5]


def test_planner_diagnostics_rejects_non_string_messages() -> None:
    with pytest.raises(TypeError, match="messages must contain strings"):
        PlannerDiagnostics(
            backend="test",
            messages=("valid", 1),  # type: ignore[arg-type]
        )


def test_action_plan_owns_scene_dependency_monitor_cutoffs() -> None:
    source = {"disabled": 0, "full_sequence": 2}
    plan = _action_plan(
        _command_sequence(
            env_ids=torch.tensor([4], dtype=torch.long),
            frame_count=2,
        ),
        scene_dependencies=("disabled", "full_sequence"),
        scene_dependency_monitor_until=source,
    )

    source["disabled"] = 1
    source["full_sequence"] = 1
    snapshot = plan.snapshot()

    assert plan.scene_dependency_monitor_until == {
        "disabled": 0,
        "full_sequence": 2,
    }
    assert snapshot.scene_dependency_monitor_until == {
        "disabled": 0,
        "full_sequence": 2,
    }
    assert snapshot.scene_dependency_monitor_until is not (
        plan.scene_dependency_monitor_until
    )


@pytest.mark.parametrize("waypoint_index", (-1, 3, True, 1.5))
def test_action_plan_rejects_invalid_scene_dependency_monitor_cutoff(
    waypoint_index: object,
) -> None:
    with pytest.raises(ValueError, match="waypoint indices"):
        _action_plan(
            _command_sequence(
                env_ids=torch.tensor([4], dtype=torch.long),
                frame_count=2,
            ),
            scene_dependencies=("tracked",),
            scene_dependency_monitor_until={
                "tracked": waypoint_index  # type: ignore[dict-item]
            },
        )


def test_action_plan_rejects_monitor_cutoff_for_non_dependency() -> None:
    with pytest.raises(ValueError, match="keys must be scene dependencies"):
        _action_plan(
            _command_sequence(
                env_ids=torch.tensor([4], dtype=torch.long),
                frame_count=2,
            ),
            scene_dependencies=("tracked",),
            scene_dependency_monitor_until={"other": 1},
        )


def test_action_plan_allows_timed_commands_without_joint_trajectory() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )

    plan = _action_plan(commands)

    assert plan.commands.frame_count == 1
    assert plan.joint_trajectory is None
    assert isinstance(plan.tracking_policy.terminal, TimedTerminalAcceptance)
    assert plan.tracking is None


def test_action_plan_rejects_command_device_mismatch() -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )

    with pytest.raises(ValueError, match="share a device"):
        _action_plan(
            commands,
            plan_success=torch.ones(1, dtype=torch.bool, device="meta"),
        )


@pytest.mark.parametrize(
    ("trajectory_env_ids", "trajectory_frame_count", "message"),
    [
        (torch.tensor([7], dtype=torch.long), 1, "env_ids must match"),
        (torch.tensor([4], dtype=torch.long), 2, "waypoints must match"),
    ],
)
def test_action_plan_validates_joint_trajectory_against_commands(
    trajectory_env_ids: torch.Tensor,
    trajectory_frame_count: int,
    message: str,
) -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=1,
    )
    trajectory = TimedTrajectory.from_uniform_step(
        torch.ones(1, trajectory_frame_count, 2),
        env_ids=trajectory_env_ids,
        step_dt=0.1,
    )

    with pytest.raises(ValueError, match=message):
        _action_plan(
            commands,
            joint_trajectory=trajectory,
        )


def test_joint_position_plan_rejects_empty_commands_for_successful_rows() -> None:
    env_ids = torch.tensor([4], dtype=torch.long)
    commands = TimedCommandSequence(frames=(), env_ids=env_ids)
    trajectory = TimedTrajectory.empty(
        batch_size=1,
        robot_dof=2,
        device=env_ids.device,
        env_ids=env_ids,
    )

    with pytest.raises(ValueError, match="requires command frames"):
        _action_plan(
            commands,
            plan_success=torch.tensor([True]),
            joint_trajectory=trajectory,
            tracking_policy=TrackingPolicy.joint_position(),
            tracking=_joint_tracking_sequence(commands),
        )


@pytest.mark.parametrize("changed_route", ["source", "projector"])
def test_tracking_plan_rejects_route_changes_between_frames(
    changed_route: str,
) -> None:
    commands = _command_sequence(
        env_ids=torch.tensor([4], dtype=torch.long),
        frame_count=2,
    )
    tracking = _joint_tracking_sequence(commands)
    first_frame, second_frame = tracking.frames
    original = second_frame.setpoints[0]
    source = original.binding.source
    projector = original.binding.projector
    if changed_route == "source":
        source = TrackingFeedbackSourceRef(
            provider_id=source.provider_id,
            revision="alternate",
            address=source.address,
        )
    else:
        projector = TrackingProjectorRef(
            projector_id=projector.projector_id,
            revision="alternate",
        )
    changed = TrackingSetpoint(
        endpoint_key=original.endpoint_key,
        binding=EndpointTrackingChannelBinding(
            channel_id=original.binding.channel_id,
            source=source,
            projector=projector,
        ),
        desired=original.desired,
    )
    changed_tracking = TimedTrackingSequence(
        commands.env_ids,
        (first_frame, TrackingFrame((changed,))),
    )

    with pytest.raises(ValueError, match="source fingerprint and projector route"):
        _action_plan(
            commands,
            tracking_policy=TrackingPolicy.joint_position(),
            tracking=changed_tracking,
        )


def test_joint_position_plan_allows_empty_commands_when_all_rows_fail() -> None:
    env_ids = torch.tensor([4], dtype=torch.long)
    commands = TimedCommandSequence(frames=(), env_ids=env_ids)
    trajectory = TimedTrajectory.empty(
        batch_size=1,
        robot_dof=2,
        device=env_ids.device,
        env_ids=env_ids,
    )

    plan = _action_plan(
        commands,
        plan_success=torch.tensor([False]),
        joint_trajectory=trajectory,
        tracking_policy=TrackingPolicy.joint_position(),
        tracking=_joint_tracking_sequence(commands),
        diagnostics=PlannerDiagnostics(
            backend="test",
            failure=PlanningFailure("planning_failed", retryable=True),
        ),
    )

    assert plan.commands.frame_count == 0


def test_action_plan_requires_explicit_failure_diagnostics() -> None:
    commands = TimedCommandSequence(
        frames=(),
        env_ids=torch.tensor([4], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="failure is required"):
        _action_plan(commands, plan_success=torch.tensor([False]))


def test_action_plan_requires_stable_destination_set() -> None:
    env_ids = torch.tensor([4], dtype=torch.long)
    commands = _command_sequence(
        env_ids=env_ids,
        frame_count=2,
        targets=(
            JointPositionTarget("arm", (0, 1)),
            JointPositionTarget("other_arm", (0, 1)),
        ),
    )
    with pytest.raises(ValueError, match="same destination set"):
        _action_plan(commands)


def test_action_plan_requires_stable_exact_target_type() -> None:
    env_ids = torch.tensor([4], dtype=torch.long)
    commands = _command_sequence(
        env_ids=env_ids,
        frame_count=2,
        targets=(
            JointPositionTarget("arm", (0, 1)),
            _AlternateJointPositionTarget("arm", (0, 1)),
        ),
    )

    with pytest.raises(ValueError, match="exact target type"):
        _action_plan(commands)


def test_action_plan_requires_stable_target_address_fingerprint() -> None:
    env_ids = torch.tensor([4], dtype=torch.long)
    commands = _command_sequence(
        env_ids=env_ids,
        frame_count=2,
        targets=(
            JointPositionTarget("arm", (0, 1)),
            JointPositionTarget("arm", (1, 0)),
        ),
    )

    with pytest.raises(ValueError, match="target address fingerprint"):
        _action_plan(commands)


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


def test_scene_snapshot_owns_entity_state_storage() -> None:
    pose = torch.eye(4)
    state = EntityState(pose)
    snapshot = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"object": state},
    )

    pose.fill_(2.0)
    state.pose.fill_(3.0)

    assert torch.equal(snapshot.entities["object"].pose, torch.eye(4))


def test_scene_snapshot_entity_reads_are_defensive() -> None:
    snapshot = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"object": EntityState(torch.eye(4))},
    )

    first_read = snapshot.entities["object"]
    first_read.pose.fill_(7.0)

    assert torch.equal(snapshot.entities["object"].pose, torch.eye(4))
    with pytest.raises(TypeError):
        snapshot.entities["other"] = EntityState(torch.eye(4))  # type: ignore[index]


def test_scene_snapshot_collision_pose_reads_are_defensive() -> None:
    snapshot = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"obstacle": EntityState(torch.eye(4))},
        collision_entity_ids=("obstacle",),
    )

    obstacle_poses = snapshot.collision_obstacle_poses(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    obstacle_poses["obstacle"].fill_(5.0)

    assert torch.equal(snapshot.entities["obstacle"].pose, torch.eye(4))


def test_scene_snapshot_rejects_unknown_collision_entity() -> None:
    with pytest.raises(ValueError, match="missing scene entities"):
        SceneSnapshot(
            timestamp=0.0,
            version=0,
            collision_entity_ids=("missing",),
        )


def test_timed_trajectory_uses_explicit_uniform_timing_and_holds_rows() -> None:
    positions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    trajectory = TimedTrajectory.from_uniform_step(
        positions,
        env_ids=torch.tensor([4, 7]),
        step_dt=0.02,
    )
    held = trajectory.hold_rows(
        torch.tensor([True, False]),
        torch.full((2, 4), -1.0),
    )

    assert trajectory.duration.tolist() == pytest.approx([0.04, 0.04])
    assert torch.equal(held.positions[0], positions[0])
    assert torch.all(held.positions[1] == -1.0)


def test_timed_trajectory_constructor_detaches_and_owns_all_tensor_fields() -> None:
    positions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    velocities = torch.full_like(positions, 0.5, requires_grad=True)
    accelerations = torch.full_like(positions, 0.25, requires_grad=True)
    dt = torch.tensor([[0.0, 0.1]], requires_grad=True)
    env_ids = torch.tensor([4], dtype=torch.long)
    inputs = {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "dt": dt,
        "env_ids": env_ids,
    }
    expected = {name: value.detach().clone() for name, value in inputs.items()}

    trajectory = TimedTrajectory(**inputs)

    with torch.no_grad():
        for value in inputs.values():
            value.zero_()
    for name, value in expected.items():
        owned = getattr(trajectory, name)
        assert torch.equal(owned, value)
        assert owned.grad_fn is None
        assert not owned.requires_grad


def test_timed_trajectory_rejects_duplicate_environment_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        TimedTrajectory.from_positions(
            torch.zeros(2, 1, 2),
            env_ids=torch.tensor([4, 4], dtype=torch.long),
            dt=torch.zeros(2, 1),
        )


def test_planning_context_requires_explicit_interpolation_period() -> None:
    with pytest.raises(ValueError, match="explicit PlanningContext.control_dt"):
        _context().require_control_dt()

    assert _context(control_dt=0.02).require_control_dt() == pytest.approx(0.02)
    with pytest.raises(ValueError, match="finite and greater than zero"):
        _context(control_dt=0.0)


def test_timed_trajectory_snapshot_owns_its_tensor_storage() -> None:
    trajectory = TimedTrajectory.from_uniform_step(
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        env_ids=torch.tensor([4]),
        step_dt=0.02,
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
    first = TimedTrajectory.from_uniform_step(
        torch.zeros(2, 2, 4),
        env_ids=torch.tensor([0, 1]),
        step_dt=0.1,
    )
    second = TimedTrajectory.from_uniform_step(
        torch.ones(2, 3, 4),
        env_ids=torch.tensor([0, 1]),
        step_dt=0.2,
    )

    result = TimedTrajectory.concatenate((first, second))

    assert result.positions.shape == (2, 5, 4)
    assert result.duration.tolist() == pytest.approx([0.5, 0.5])
