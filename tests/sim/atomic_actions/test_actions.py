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

"""Tests for built-in atomic actions under the plan/invocation contract."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionPlan,
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    AssembleGoal,
    AtomicAction,
    AtomicActionEngine,
    ControlPartCommandProfile,
    CoordinatedPickGoal,
    CoordinatedHeldObjectState,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
    CoordinatedPlacement,
    CoordinatedPlacementGoal,
    CoordinatedPlacementOptions,
    EndEffectorPoseGoal,
    EntityState,
    ExecutionEventKind,
    GraspGoal,
    HandOver,
    HandOverOptions,
    HeldObjectPoseGoal,
    HeldObjectState,
    JointPositionGoal,
    JointPositionPayload,
    JointPositionTarget,
    MotionPolicy,
    MoveEndEffector,
    MoveEndEffectorOptions,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    ObservedArticulationJointState,
    OperateArticulation,
    OperateArticulationGoal,
    OperateArticulationOptions,
    PickUp,
    PickUpOptions,
    Place,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    SlideAffordance,
    Slide,
    SlideGoal,
    SlideOptions,
    RobotObservation,
    SceneArticulationOperationGeometry,
    SceneEntityPose,
    SceneSnapshot,
    TaskState,
    TimedTrajectory,
    TwistAffordance,
    Twist,
    TwistGoal,
    TwistOptions,
)
from embodichain.lab.sim.atomic_actions.goals import collect_scene_dependencies
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.planners import (
    MotionGenerator,
    MoveType,
    PlanOptions,
    PlanResult,
)
from embodichain.utils.math import pose_inv

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
ROBOT_DOF = ARM_DOF + HAND_DOF
CONTROL_DT = 1.0 / 60.0
DUAL_ARM_DOF = 2 * ARM_DOF
DUAL_ROBOT_DOF = DUAL_ARM_DOF + 2 * HAND_DOF

ActionT = TypeVar("ActionT", bound=AtomicAction)
_ACTION_ENGINES: dict[int, AtomicActionEngine] = {}


@pytest.fixture(autouse=True)
def _torch_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small torch interpolation stand-in without initializing Warp."""

    def interpolate(
        trajectory: torch.Tensor,
        interp_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.linspace(
            0,
            trajectory.shape[1] - 1,
            interp_num,
            device=device,
        )
        lower = indices.floor().to(torch.long)
        upper = indices.ceil().to(torch.long)
        weights = (indices - lower).view(1, -1, 1)
        return torch.lerp(trajectory[:, lower], trajectory[:, upper], weights)

    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.trajectory_ops.interpolate_with_distance",
        interpolate,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.planners.motion_generator.interpolate_with_distance",
        interpolate,
    )


def _robot() -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = ROBOT_DOF
    robot.control_parts = {
        "arm": object(),
        "hand": object(),
        "alternate_arm": object(),
        "alternate_hand": object(),
    }

    def get_qpos(name: str | None = None) -> torch.Tensor:
        if name in {"arm", "alternate_arm"}:
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name in {"hand", "alternate_hand"}:
            return torch.zeros(NUM_ENVS, HAND_DOF)
        return torch.zeros(NUM_ENVS, ROBOT_DOF)

    def get_joint_ids(name: str | None = None) -> list[int]:
        if name in {"arm", "alternate_arm"}:
            return list(range(ARM_DOF))
        if name in {"hand", "alternate_hand"}:
            return list(range(ARM_DOF, ROBOT_DOF))
        return list(range(ROBOT_DOF))

    def compute_ik(
        pose: torch.Tensor | None = None,
        name: str | None = None,
        joint_seed: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert joint_seed is not None
        return torch.ones(NUM_ENVS, dtype=torch.bool), joint_seed.clone()

    def compute_fk(
        qpos: torch.Tensor | None = None,
        name: str | None = None,
        to_matrix: bool = True,
    ) -> torch.Tensor:
        count = NUM_ENVS if qpos is None else qpos.shape[0]
        return torch.eye(4).repeat(count, 1, 1)

    robot.get_qpos.side_effect = get_qpos
    robot.get_joint_ids.side_effect = get_joint_ids
    robot.compute_ik.side_effect = compute_ik
    robot.compute_fk.side_effect = compute_fk
    return robot


def _motion_generator() -> MotionGenerator:
    generator = object.__new__(MotionGenerator)
    generator.robot = _robot()
    generator.device = torch.device("cpu")
    generator.planner = Mock()
    generator.planner.cfg.planner_type = "stub"
    generator.planner.collision_world_info = None
    generator.planner.preserve_plan_samples = False
    generator.planner.supports_move_type.return_value = False
    generator.planner.default_plan_options.return_value = PlanOptions()
    generator.planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    return generator


def _bind_action(
    generator: MotionGenerator,
    action: ActionT,
    control_profiles: dict[str, ControlPartCommandProfile] | None = None,
) -> ActionT:
    """Bind one configured action to an engine-owned test backend."""
    profiles = {
        name: ControlPartCommandProfile.joint_positions(
            open=torch.zeros(len(generator.robot.get_joint_ids(name=name))),
            grasp=torch.ones(len(generator.robot.get_joint_ids(name=name))),
        )
        for name in generator.robot.control_parts
        if "hand" in name
    }
    profiles.update({} if control_profiles is None else control_profiles)
    engine = AtomicActionEngine(
        generator,
        control_profiles=profiles,
        load_builtins=False,
    )
    engine.register(action)
    _ACTION_ENGINES[id(action)] = engine
    return action


def _plan_action(
    action: AtomicAction,
    invocation: ActionInvocation,
    context: PlanningContext,
):
    """Resolve a caller-owned invocation before calling the action planner."""
    return action.plan(action.resolve_request(invocation), context)


def _context(
    task: TaskState | None = None,
    *,
    scene: SceneSnapshot | None = None,
    timestamp: float = 0.0,
) -> PlanningContext:
    qpos = torch.zeros(NUM_ENVS, ROBOT_DOF)
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=qpos,
            qvel=torch.zeros_like(qpos),
        ),
        task=task or TaskState.empty(batch_size=NUM_ENVS, device="cpu"),
        scene=SceneSnapshot.empty() if scene is None else scene,
        env_ids=torch.arange(NUM_ENVS),
        control_dt=CONTROL_DT,
    )


def _target_scene(
    pose: torch.Tensor,
    *,
    timestamp: float,
    version: int,
) -> SceneSnapshot:
    """Build a versioned target snapshot for late-bound grasp tests."""
    return SceneSnapshot(
        timestamp=timestamp,
        version=version,
        entities={"target": EntityState(pose)},
    )


def _articulation_geometry() -> SceneArticulationOperationGeometry:
    """Build late-bound identity handle geometry for atomic tests."""
    identity = torch.eye(4)
    return SceneArticulationOperationGeometry(
        handle_pose=SceneEntityPose("drawer_handle"),
        approach_offset=identity,
        contact_offset=identity,
        operation_offset=identity,
        retract_offset=identity,
        operation_axis=torch.tensor((1.0, 0.0, 0.0)),
    )


def _articulation_scene(
    position: torch.Tensor,
    *,
    handle_x: float = 0.0,
    timestamp: float = 0.0,
    version: int = 0,
) -> SceneSnapshot:
    """Build one live handle and articulation-joint snapshot."""
    handle = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    handle[:, 0, 3] = handle_x
    return SceneSnapshot(
        timestamp=timestamp,
        version=version,
        entities={"drawer_handle": EntityState(handle)},
        articulation_joints={
            ("drawer", "slide"): ObservedArticulationJointState(position)
        },
    )


def _binding(
    action: AtomicAction,
    *,
    motion: str = "arm",
    grasp: str = "hand",
    task_state_key: str | None = None,
) -> ActionBinding:
    """Bind one single-participant action through its owning engine."""
    contract = type(action).__dict__.get("binding_contract")
    assert contract is not None
    endpoint_parts = {
        "motion": motion,
        "grasp": grasp,
        "interaction": grasp,
    }
    return _ACTION_ENGINES[id(action)].bind_control_parts(
        action.skill_id,
        {
            slot.slot_id: {
                endpoint.endpoint_id: endpoint_parts[endpoint.endpoint_id]
                for endpoint in slot.endpoints
            }
            for slot in contract.slots
        },
        task_state_keys=(
            None
            if task_state_key is None
            else {slot.slot_id: task_state_key for slot in contract.slots}
        ),
    )


def _invocation(
    action: AtomicAction,
    goal,
    *,
    sample_count: int = 20,
) -> ActionInvocation:
    return ActionInvocation(
        skill_id=action.skill_id,
        goal=goal,
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=sample_count),
    )


def _joint_trajectory(plan: ActionPlan) -> TimedTrajectory:
    """Return the owned planner trajectory for a joint-feedback plan."""
    assert plan.joint_trajectory is not None
    return plan.joint_trajectory


def _joint_command_positions(
    plan: ActionPlan,
    control_part: str,
) -> torch.Tensor:
    """Stack runtime joint commands sent to one concrete control part."""
    return torch.stack(
        [payload.positions for payload in _joint_command_payloads(plan, control_part)],
        dim=1,
    )


def _joint_command_payloads(
    plan: ActionPlan,
    control_part: str,
) -> tuple[JointPositionPayload, ...]:
    """Return runtime joint payloads sent to one concrete control part."""
    payloads: list[JointPositionPayload] = []
    for frame in plan.commands.frames:
        matching = [
            command
            for command in frame.commands
            if isinstance(command.target, JointPositionTarget)
            and command.target.control_part == control_part
        ]
        assert len(matching) == 1
        payload = matching[0].payload
        assert isinstance(payload, JointPositionPayload)
        payloads.append(payload)
    return tuple(payloads)


def _semantics(*, entity_id: str | None = None) -> ObjectSemantics:
    entity = Mock(spec=BatchEntity)
    entity.get_local_pose.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="test_object",
        entity=entity,
        entity_id=entity_id,
    )


def _held(
    semantics: ObjectSemantics | None = None,
    *,
    env_mask: torch.Tensor | None = None,
) -> HeldObjectState:
    poses = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    return HeldObjectState(
        semantics=semantics or _semantics(),
        object_to_eef=poses,
        grasp_xpos=poses,
        env_mask=env_mask,
    )


def _dual_motion_generator() -> MotionGenerator:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = DUAL_ROBOT_DOF
    robot.control_parts = {
        "dual_arm": object(),
        "left_arm": object(),
        "right_arm": object(),
        "left_hand": object(),
        "right_hand": object(),
    }

    def get_qpos(name: str | None = None) -> torch.Tensor:
        if name in {"left_arm", "right_arm"}:
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name == "dual_arm":
            return torch.zeros(NUM_ENVS, DUAL_ARM_DOF)
        if name in {"left_hand", "right_hand"}:
            return torch.zeros(NUM_ENVS, HAND_DOF)
        return torch.zeros(NUM_ENVS, DUAL_ROBOT_DOF)

    def get_joint_ids(name: str | None = None) -> list[int]:
        if name == "left_arm":
            return list(range(ARM_DOF))
        if name == "right_arm":
            return list(range(ARM_DOF, DUAL_ARM_DOF))
        if name == "dual_arm":
            return list(range(DUAL_ARM_DOF))
        if name == "left_hand":
            return list(range(DUAL_ARM_DOF, DUAL_ARM_DOF + HAND_DOF))
        if name == "right_hand":
            return list(range(DUAL_ARM_DOF + HAND_DOF, DUAL_ROBOT_DOF))
        return list(range(DUAL_ROBOT_DOF))

    def compute_ik(
        pose: torch.Tensor | None = None,
        name: str | None = None,
        joint_seed: torch.Tensor | None = None,
        qpos_seed: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed = joint_seed if joint_seed is not None else qpos_seed
        assert seed is not None
        offset = 0.1 if name == "left_arm" else 0.2
        return torch.ones(seed.shape[0], dtype=torch.bool), seed + offset

    def compute_fk(
        qpos: torch.Tensor | None = None,
        name: str | None = None,
        to_matrix: bool = True,
    ) -> torch.Tensor:
        count = NUM_ENVS if qpos is None else qpos.shape[0]
        return torch.eye(4).repeat(count, 1, 1)

    robot.get_qpos.side_effect = get_qpos
    robot.get_joint_ids.side_effect = get_joint_ids
    robot.compute_ik.side_effect = compute_ik
    robot.compute_fk.side_effect = compute_fk

    generator = object.__new__(MotionGenerator)
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner = Mock()
    generator.planner.cfg.planner_type = "stub"
    generator.planner.collision_world_info = None
    generator.planner.preserve_plan_samples = False
    generator.planner.supports_move_type.return_value = False
    generator.planner.default_plan_options.return_value = PlanOptions()
    generator.planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    return generator


def _dual_context(
    task: TaskState | None = None,
    *,
    scene: SceneSnapshot | None = None,
) -> PlanningContext:
    qpos = torch.zeros(NUM_ENVS, DUAL_ROBOT_DOF)
    return PlanningContext(
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
        task=task or TaskState.empty(NUM_ENVS, "cpu"),
        scene=SceneSnapshot.empty() if scene is None else scene,
        env_ids=torch.arange(NUM_ENVS),
        control_dt=CONTROL_DT,
    )


def _dual_binding(
    action: AtomicAction,
    first_slot: str,
    second_slot: str,
    *,
    task_state_keys: dict[str, str] | None = None,
) -> ActionBinding:
    return _ACTION_ENGINES[id(action)].bind_control_parts(
        action.skill_id,
        {
            first_slot: {
                "motion": "left_arm",
                "grasp": "left_hand",
            },
            second_slot: {
                "motion": "right_arm",
                "grasp": "right_hand",
            },
        },
        task_state_keys=task_state_keys,
    )


def _stub_dual_arm_grasp_poses(affordance: AntipodalAffordance) -> None:
    """Stub affordance dual-arm grasp sampling with identity grasp poses."""

    def _sample(obj_poses: torch.Tensor, **_kwargs: object) -> list[dict]:
        arm = {
            "is_success": True,
            "grasp_poses": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "open_lengths": torch.tensor([0.0], dtype=torch.float32),
            "total_cost": torch.tensor([0.0], dtype=torch.float32),
        }
        return [{"left": arm, "right": arm} for _ in range(obj_poses.shape[0])]

    affordance.get_dual_arm_valid_grasp_poses = Mock(side_effect=_sample)


def _plan_segment_contract_case(case_id: str) -> ActionPlan:
    """Plan one built-in used by the Version 1 trajectory-segment contract."""
    generator = _motion_generator()
    sample_count = 20

    if case_id == "move_joints":
        action = _bind_action(generator, MoveJoints())
        goal = JointPositionGoal(torch.zeros(ARM_DOF))
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(),
        )

    if case_id == "move_end_effector":
        action = _bind_action(generator, MoveEndEffector())
        goal = EndEffectorPoseGoal(torch.eye(4))
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(),
        )

    held_task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": _held()},
    )
    if case_id == "move_held_object":
        action = _bind_action(generator, MoveHeldObject())
        goal = HeldObjectPoseGoal(torch.eye(4))
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(held_task),
        )

    if case_id == "place":
        action = _bind_action(generator, Place())
        goal = PlaceGoal(torch.eye(4))
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(held_task),
        )

    if case_id == "assemble":
        action = _bind_action(generator, Place())
        goal = AssembleGoal(
            affordance=AssembleAffordance(
                base_object_entity=Mock(),
                assemble_to_base_pose=torch.eye(4),
            ),
            base_pose=SceneEntityPose("base"),
        )
        base_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
        scene = SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={"base": EntityState(base_pose)},
        )
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(held_task, scene=scene),
        )

    if case_id == "press":
        action = _bind_action(generator, Press())
        semantics = ObjectSemantics(
            affordance=PressAffordance(
                press_axis=torch.tensor([1.0, 0.0, 0.0]),
                press_position=(0.0, 0.0, 0.0),
            ),
            geometry={},
            label="button",
        )
        goal = PressGoal(semantics, torch.eye(4))
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(),
        )

    raise AssertionError(f"Unknown trajectory-segment contract case {case_id!r}.")


def test_builtin_descriptors_expose_goals_not_legacy_targets() -> None:
    assert MoveEndEffector.GoalType is EndEffectorPoseGoal
    assert MoveJoints.GoalType is JointPositionGoal
    assert PickUp.GoalType is GraspGoal
    assert MoveHeldObject.GoalType is HeldObjectPoseGoal
    assert Place.GoalType == (PlaceGoal, AssembleGoal)
    assert Press.GoalType is PressGoal
    assert Slide.GoalType is SlideGoal
    assert Twist.GoalType is TwistGoal
    assert CoordinatedPickment.GoalType is CoordinatedPickGoal
    assert CoordinatedPlacement.GoalType is CoordinatedPlacementGoal
    assert HandOver.GoalType is GraspGoal
    assert OperateArticulation.GoalType is OperateArticulationGoal


@pytest.mark.parametrize(
    ("case_id", "expected_names"),
    (
        ("move_joints", ("move_joints",)),
        ("move_end_effector", ("move_end_effector",)),
        ("move_held_object", ("transport",)),
        ("place", ("approach", "release", "retract")),
        ("assemble", ("approach", "release", "retract")),
        ("press", ("close", "approach", "contact", "press", "retract")),
    ),
)
def test_builtin_trajectory_segment_names_and_ranges_are_stable(
    case_id: str,
    expected_names: tuple[str, ...],
) -> None:
    plan = _plan_segment_contract_case(case_id)

    assert plan.success_all
    assert tuple(segment.name for segment in plan.segments) == expected_names
    assert plan.segments[0].start == 0
    assert all(
        previous.stop == current.start
        for previous, current in zip(plan.segments, plan.segments[1:])
    )
    assert plan.segments[-1].stop == plan.commands.frame_count


def test_interaction_primitives_use_motion_centric_skill_ids() -> None:
    assert (Press.skill_id, Slide.skill_id, Twist.skill_id) == (
        "press",
        "slide",
        "twist",
    )


@pytest.mark.parametrize(
    "options",
    (
        PickUpOptions(),
        MoveHeldObjectOptions(),
        PlaceOptions(),
        PressOptions(),
        SlideOptions(),
        TwistOptions(),
        CoordinatedPickmentOptions(),
        CoordinatedPlacementOptions(),
        HandOverOptions(),
        OperateArticulationOptions(),
    ),
)
def test_action_options_do_not_contain_embodiment_resources(options: object) -> None:
    field_names = getattr(options, "__dataclass_fields__")

    assert "control_part" not in field_names
    assert not any(name.endswith("_control_part") for name in field_names)
    assert not any(name.endswith("_qpos") for name in field_names)


def test_pose_options_own_late_bound_relative_transforms() -> None:
    relative_pose = torch.eye(4)
    target = SceneEntityPose("target", relative_pose=relative_pose)
    pick_options = PickUpOptions(downstream_object_target_poses=(target,))
    handover_options = HandOverOptions(
        middle_object_pose=target,
        final_object_pose=target,
    )

    assert target.relative_pose is not None
    target.relative_pose[0, 3] = 9.0

    pick_target = pick_options.downstream_object_target_poses[0]
    assert type(pick_target) is SceneEntityPose
    assert pick_target.relative_pose is not None
    assert pick_target.relative_pose[0, 3].item() == 0.0
    assert type(handover_options.middle_object_pose) is SceneEntityPose
    assert handover_options.middle_object_pose.relative_pose is not None
    assert handover_options.middle_object_pose.relative_pose[0, 3].item() == 0.0


def test_joint_position_goal_rejects_unsupported_target_type() -> None:
    with pytest.raises(TypeError, match="torch.Tensor or str"):
        JointPositionGoal(target=1.0)  # type: ignore[arg-type]


def test_joint_position_goal_rejects_empty_named_target() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        JointPositionGoal(target="   ")


@pytest.mark.parametrize(
    "target",
    (
        torch.tensor(1.0),
        torch.empty(0),
        torch.zeros(1, 1, 1, 1),
    ),
)
def test_joint_position_goal_rejects_invalid_tensor_shape(
    target: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="Tensor target must have shape"):
        JointPositionGoal(target=target)


def test_move_end_effector_returns_full_robot_timed_plan() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, MoveEndEffector())
    context = _context()

    plan = _plan_action(
        action,
        _invocation(
            action,
            EndEffectorPoseGoal(torch.eye(4)),
            sample_count=10,
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.commands.frame_count == 10
    assert [target.target_id for target in plan.commands.targets] == ["arm"]
    assert _joint_trajectory(plan).duration.tolist() == pytest.approx([0.15, 0.15])
    assert plan.expected_effects.is_empty


def test_move_joints_uses_binding_and_preserves_uncontrolled_joints() -> None:
    generator = _motion_generator()
    named = {"ready": torch.full((ARM_DOF,), 0.4)}
    action = _bind_action(
        generator,
        MoveJoints(),
        control_profiles={
            "arm": ControlPartCommandProfile.joint_positions(**named),
        },
    )
    qpos = torch.zeros(NUM_ENVS, ROBOT_DOF)
    qpos[:, ARM_DOF:] = 0.7
    context = PlanningContext(
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
        task=TaskState.empty(NUM_ENVS, "cpu"),
        scene=SceneSnapshot.empty(),
        env_ids=torch.arange(NUM_ENVS),
        control_dt=CONTROL_DT,
    )

    plan = _plan_action(
        action,
        _invocation(action, JointPositionGoal("ready"), sample_count=8),
        context,
    )

    arm_positions = _joint_command_positions(plan, "arm")
    assert torch.allclose(arm_positions[:, -1], named["ready"])
    assert [target.target_id for target in plan.commands.targets] == ["arm"]


def test_pick_and_place_declare_effects_without_mutating_context() -> None:
    generator = _motion_generator()
    pick = _bind_action(generator, PickUp())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    initial = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))
    semantics = _semantics(entity_id="target")
    grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    pick_plan = _plan_action(
        pick,
        ActionInvocation(
            skill_id="pick_up",
            goal=GraspGoal(semantics=semantics, grasp_xpos=grasp),
            binding=_binding(pick, task_state_key="logical_arm"),
        ),
        initial,
    )
    picked_task = pick_plan.expected_effects.apply(initial.task, pick_plan.plan_success)

    assert initial.task.get_held_object("logical_arm") is None
    assert picked_task.get_held_object("logical_arm") is not None
    assert picked_task.get_held_object("arm") is None

    place = _bind_action(generator, Place())
    picked_context = PlanningContext(
        robot=initial.robot,
        task=picked_task,
        scene=initial.scene,
        env_ids=initial.env_ids,
        control_dt=initial.control_dt,
    )
    place_plan = _plan_action(
        place,
        ActionInvocation(
            skill_id="place",
            goal=PlaceGoal(torch.eye(4)),
            binding=_binding(place, task_state_key="logical_arm"),
        ),
        picked_context,
    )
    placed_task = place_plan.expected_effects.apply(
        picked_task, place_plan.plan_success
    )

    assert picked_task.get_held_object("logical_arm") is not None
    assert placed_task.get_held_object("logical_arm") is None


def test_place_releases_only_exclusively_held_rows() -> None:
    generator = _motion_generator()

    def move_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(NUM_ENVS, dtype=torch.bool), joint_seed + 0.1

    generator.robot.compute_ik.side_effect = move_ik
    action = _bind_action(generator, Place())
    semantics = _semantics()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "arm": _held(semantics),
            "alternate_arm": _held(
                semantics,
                env_mask=torch.tensor([True, False]),
            ),
        },
    )
    context = _context(task)

    plan = _plan_action(
        action,
        _invocation(action, PlaceGoal(torch.eye(4))),
        context,
    )
    projected = plan.expected_effects.apply(task, plan.plan_success)

    assert plan.plan_success.tolist() == [False, True]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(trajectory.waypoint_count, -1),
    )
    primary = projected.get_held_object("arm")
    alternate = projected.get_held_object("alternate_arm")
    assert primary is not None and primary.env_mask.tolist() == [True, False]
    assert alternate is not None and alternate.env_mask.tolist() == [True, False]


def test_move_held_object_requires_projected_attachment() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, MoveHeldObject())
    invocation = _invocation(
        action,
        HeldObjectPoseGoal(torch.eye(4)),
        sample_count=10,
    )
    invocation = replace(
        invocation,
        binding=_binding(action, task_state_key="logical_arm"),
    )

    with pytest.raises(ValueError, match="requires an object held"):
        _plan_action(action, invocation, _context())

    semantics = _semantics()
    held = _held(semantics)
    held.object_to_eef[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    held.object_to_eef[:, 0, 3] = torch.tensor([0.1, 0.2])
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"logical_arm": held},
    )
    eef_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    eef_pose[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    eef_pose[:, 0, 3] = torch.tensor([0.5, 0.8])
    generator.robot.compute_fk.return_value = eef_pose
    generator.robot.compute_fk.side_effect = None
    action._apply_configured_upright_rotation = Mock()
    configured_invocation = ActionInvocation(
        skill_id="move_held_object",
        goal=HeldObjectPoseGoal(torch.eye(4)),
        binding=_binding(action, task_state_key="logical_arm"),
        motion_policy=MotionPolicy(sample_count=10),
        skill_options=MoveHeldObjectOptions(pick_rotate_upright=0.25),
    )

    plan = _plan_action(action, configured_invocation, _context(task))

    assert plan.plan_success.all()
    assert plan.expected_effects.is_empty
    assert generator.robot.compute_fk.call_args.kwargs["name"] == "arm"
    current_object_pose = action._apply_configured_upright_rotation.call_args.args[2]
    assert torch.allclose(
        current_object_pose,
        torch.bmm(eef_pose, pose_inv(held.object_to_eef)),
    )
    semantics.entity.get_local_pose.assert_not_called()


def test_move_held_object_moves_only_exclusively_held_rows() -> None:
    generator = _motion_generator()

    def move_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(NUM_ENVS, dtype=torch.bool), joint_seed + 0.1

    generator.robot.compute_ik.side_effect = move_ik
    action = _bind_action(generator, MoveHeldObject())
    semantics = _semantics()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "arm": _held(semantics),
            "alternate_arm": _held(
                semantics,
                env_mask=torch.tensor([True, False]),
            ),
        },
    )
    context = _context(task)

    plan = _plan_action(
        action,
        _invocation(action, HeldObjectPoseGoal(torch.eye(4))),
        context,
    )

    assert plan.plan_success.tolist() == [False, True]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(trajectory.waypoint_count, -1),
    )
    assert not torch.allclose(trajectory.positions[1], context.robot.qpos[1])


def test_operate_articulation_builds_named_verified_interaction() -> None:
    generator = _motion_generator()
    action = _bind_action(
        generator,
        OperateArticulation(
            OperateArticulationOptions(engage_steps=2, release_steps=2)
        ),
    )
    goal = OperateArticulationGoal(
        articulation_id="drawer",
        joint_id="slide",
        geometry=_articulation_geometry(),
        source_position=torch.tensor([0.0]),
        target_position=torch.tensor([0.4]),
        target_displacement=0.4,
    )

    plan = _plan_action(
        action,
        _invocation(action, goal, sample_count=16),
        _context(scene=_articulation_scene(torch.tensor([0.0]))),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.commands.frame_count == 16
    assert tuple(segment.name for segment in plan.segments) == (
        "approach",
        "engage",
        "operate",
        "release",
        "retract",
    )
    assert plan.requires_effect_verification
    assert plan.scene_dependency_monitor_until == {
        "drawer_handle": plan.segment("operate").start
    }
    assert plan.effect_verification is not None
    assert plan.effect_verification.kind == "articulation.joint_progress"
    update = plan.expected_effects.articulation_joint_updates[("drawer", "slide")]
    assert update is not None
    assert torch.equal(update.position, torch.tensor([0.4]))
    interaction = _joint_command_positions(plan, "hand")
    assert torch.all(interaction[:, -1] == 0.0)
    assert torch.any(interaction == 1.0)


def test_operate_articulation_reports_per_phase_planning_failures() -> None:
    generator = _motion_generator()
    action = _bind_action(
        generator,
        OperateArticulation(
            OperateArticulationOptions(engage_steps=2, release_steps=2)
        ),
    )
    phase_results = []
    for index in range(4):
        phase_results.append(
            PlanResult(
                success=torch.tensor([index != 2, True]),
                positions=torch.zeros(NUM_ENVS, 3, ARM_DOF),
                dt=torch.tensor([[0.0, 0.1, 0.2]]).repeat(NUM_ENVS, 1),
            )
        )
    generator.generate = Mock(side_effect=phase_results)
    goal = OperateArticulationGoal(
        articulation_id="drawer",
        joint_id="slide",
        geometry=_articulation_geometry(),
        source_position=torch.tensor([0.0]),
        target_position=torch.tensor([0.4]),
        target_displacement=0.4,
    )

    plan = _plan_action(
        action,
        _invocation(action, goal, sample_count=16),
        _context(scene=_articulation_scene(torch.tensor([0.0]))),
    )

    assert plan.plan_success.tolist() == [False, True]
    assert plan.diagnostics.messages == (
        "Articulation motion phase 'operate' failed for rows [0].",
    )
    phases = plan.diagnostics.metadata["motion_phases"]
    assert phases["operate"] == {
        "success": [False, True],
        "failed_rows": [0],
        "waypoint_count": 3,
    }


def test_operate_articulation_replan_uses_fresh_handle_and_remaining_stroke() -> None:
    generator = _motion_generator()
    action = _bind_action(
        generator,
        OperateArticulation(
            OperateArticulationOptions(engage_steps=2, release_steps=2)
        ),
    )
    goal = OperateArticulationGoal(
        articulation_id="drawer",
        joint_id="slide",
        geometry=_articulation_geometry(),
        source_position=torch.tensor([[0.0], [0.0]]),
        target_position=torch.tensor([[0.4], [0.4]]),
        target_displacement=0.4,
    )
    invocation = _invocation(action, goal, sample_count=16)
    initial_context = _context(
        scene=_articulation_scene(
            torch.tensor([[0.0], [0.0]]),
            handle_x=0.3,
        )
    )
    session = _ACTION_ENGINES[id(action)].start((invocation,), initial_context)
    first_operation = generator.robot.compute_ik.call_args_list[2].kwargs["pose"]
    assert torch.allclose(first_operation[:, 0, 3], torch.tensor([0.7, 0.7]))
    session.tick(initial_context)

    generator.robot.compute_ik.reset_mock()
    recovered = session.tick(
        _context(
            scene=_articulation_scene(
                torch.tensor([[0.2], [0.4]]),
                handle_x=0.55,
                timestamp=1.0,
                version=1,
            ),
            timestamp=1.0,
        ),
    )
    recovered_operation = generator.robot.compute_ik.call_args_list[2].kwargs["pose"]

    assert torch.allclose(recovered_operation[:, 0, 3], torch.tensor([0.75, 0.55]))
    event_kinds = {event.kind for event in recovered.events}
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in event_kinds
    assert ExecutionEventKind.REPLANNED in event_kinds
    assert session.trajectory_segment("operate").name == "operate"


def test_operate_articulation_requires_live_joint_observation() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, OperateArticulation())
    goal = OperateArticulationGoal(
        articulation_id="drawer",
        joint_id="slide",
        geometry=_articulation_geometry(),
        source_position=torch.tensor([0.0]),
        target_position=torch.tensor([0.4]),
        target_displacement=0.4,
    )
    handle = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"drawer_handle": EntityState(handle)},
    )

    with pytest.raises(ValueError, match="ObservedArticulationJointState"):
        _plan_action(
            action,
            _invocation(action, goal, sample_count=20),
            _context(scene=scene),
        )


def test_operate_articulation_rejects_insufficient_motion_budget() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, OperateArticulation())
    goal = OperateArticulationGoal(
        articulation_id="drawer",
        joint_id="slide",
        geometry=_articulation_geometry(),
        source_position=torch.tensor([0.0]),
        target_position=torch.tensor([0.4]),
        target_displacement=0.4,
    )

    with pytest.raises(ValueError, match="at least two waypoints"):
        _plan_action(
            action,
            _invocation(action, goal, sample_count=17),
            _context(scene=_articulation_scene(torch.tensor([0.0]))),
        )


def test_strategy_and_sample_count_are_not_action_config_fields() -> None:
    with pytest.raises(TypeError):
        MoveEndEffectorOptions(strategy="motion_gen")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MoveJointsOptions(sample_interval=10)  # type: ignore[call-arg]


def test_move_joints_rejects_binding_with_wrong_goal_skill() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_end_effector",
        goal=JointPositionGoal(torch.zeros(ARM_DOF)),
        binding=_binding(action),
    )
    with pytest.raises(ValueError, match="skill_id"):
        action.resolve_request(invocation)


def test_move_joints_rejects_incompatible_goal_at_action_boundary() -> None:
    action = _bind_action(_motion_generator(), MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=object(),  # type: ignore[arg-type]
        binding=_binding(action),
    )

    with pytest.raises(TypeError, match="expects goal JointPositionGoal"):
        action.resolve_request(invocation)


def test_builtin_action_validates_resolved_request_once() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, MoveEndEffector())
    validator = Mock(wraps=action.require_goal)
    action.require_goal = validator  # type: ignore[method-assign]

    _plan_action(
        action,
        _invocation(
            action,
            EndEffectorPoseGoal(torch.eye(4)),
        ),
        _context(),
    )

    validator.assert_called_once()


def test_planner_timing_is_preserved_in_simple_action() -> None:
    generator = _motion_generator()
    generator.planner.cfg.planner_type = "toppra"
    generator.planner.supports_move_type.side_effect = (
        lambda move_type: move_type is MoveType.JOINT_MOVE
    )
    generator.planner.plan.return_value = PlanResult(
        success=torch.ones(NUM_ENVS, dtype=torch.bool),
        positions=torch.ones(NUM_ENVS, 3, ARM_DOF),
        velocities=torch.full((NUM_ENVS, 3, ARM_DOF), 0.5),
        accelerations=torch.zeros(NUM_ENVS, 3, ARM_DOF),
        dt=torch.tensor([[0.0, 0.1, 0.2]]).repeat(NUM_ENVS, 1),
    )
    action = _bind_action(generator, MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.ones(ARM_DOF)),
        binding=_binding(action),
        motion_policy=MotionPolicy(strategy="motion_gen", sample_count=3),
    )

    plan = _plan_action(action, invocation, _context())

    trajectory = _joint_trajectory(plan)
    payloads = _joint_command_payloads(plan, "arm")
    assert trajectory.duration.tolist() == pytest.approx([0.3, 0.3])
    assert all(payload.velocities is not None for payload in payloads)
    assert torch.all(
        torch.stack([payload.velocities for payload in payloads], dim=1) == 0.5
    )


def test_move_end_effector_visits_batched_waypoints_in_order() -> None:
    generator = _motion_generator()
    solved_poses: list[torch.Tensor] = []

    def compute_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        solved_poses.append(pose.clone())
        return torch.ones(NUM_ENVS, dtype=torch.bool), joint_seed + 0.1

    generator.robot.compute_ik.side_effect = compute_ik
    waypoints = torch.eye(4).reshape(1, 1, 4, 4).repeat(NUM_ENVS, 2, 1, 1)
    waypoints[:, 0, 0, 3] = 0.1
    waypoints[:, 1, 0, 3] = 0.3

    action = _bind_action(generator, MoveEndEffector())
    plan = _plan_action(
        action,
        _invocation(
            action,
            EndEffectorPoseGoal(waypoints),
            sample_count=9,
        ),
        _context(),
    )

    assert plan.plan_success.all()
    assert len(solved_poses) == 2
    assert solved_poses[0][:, 0, 3].tolist() == pytest.approx([0.1, 0.1])
    assert solved_poses[1][:, 0, 3].tolist() == pytest.approx([0.3, 0.3])


def test_move_joints_visits_waypoints_and_rejects_unknown_names() -> None:
    generator = _motion_generator()
    action = _bind_action(
        generator,
        MoveJoints(),
        control_profiles={
            "arm": ControlPartCommandProfile.joint_positions(ready=torch.zeros(ARM_DOF))
        },
    )
    waypoints = torch.stack(
        [
            torch.full((NUM_ENVS, ARM_DOF), 0.3),
            torch.full((NUM_ENVS, ARM_DOF), 0.7),
        ],
        dim=1,
    )

    plan = _plan_action(
        action,
        _invocation(
            action,
            JointPositionGoal(waypoints),
            sample_count=7,
        ),
        _context(),
    )

    arm_positions = _joint_command_positions(plan, "arm")
    assert torch.allclose(arm_positions[:, 3], waypoints[:, 0])
    assert torch.allclose(arm_positions[:, -1], waypoints[:, 1])
    with pytest.raises(KeyError, match="has no command"):
        _plan_action(
            action,
            _invocation(action, JointPositionGoal("missing")),
            _context(),
        )


def test_pick_explicit_grasp_bypasses_sampling_and_records_grasp() -> None:
    generator = _motion_generator()
    affordance = AntipodalAffordance()
    affordance.get_valid_grasp_poses = Mock()
    entity = Mock()
    entity.get_local_pose.return_value = torch.full((NUM_ENVS, 4, 4), 9.0)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="explicit-grasp-object",
        entity=entity,
        entity_id="target",
    )
    grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp[:, 0, 3] = torch.tensor([0.1, 0.2])
    action = _bind_action(generator, PickUp())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    object_pose[:, 0, 3] = torch.tensor([0.03, 0.07])
    context = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))

    request = action.resolve_request(
        _invocation(
            action,
            GraspGoal(semantics=semantics, grasp_xpos=grasp),
            sample_count=20,
        )
    )
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    request.goal.semantics.affordance.get_valid_grasp_poses.assert_not_called()
    request.goal.semantics.entity.get_local_pose.assert_not_called()
    held = projected.get_held_object("arm")
    assert held is not None
    assert torch.allclose(held.grasp_xpos, grasp)
    assert torch.allclose(held.object_to_eef, torch.bmm(pose_inv(object_pose), grasp))
    assert plan.scene_dependencies == ("target",)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "close",
        "lift",
    ]
    assert plan.segment("close").stop == plan.segment("lift").start
    assert plan.scene_dependency_monitor_until == {
        "target": plan.segment("close").start
    }


def test_pick_holds_only_environment_without_a_feasible_grasp() -> None:
    generator = _motion_generator()
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="partially-graspable-object",
        entity=entity,
        entity_id="target",
    )
    action = _bind_action(generator, PickUp())
    action._resolve_grasp_pose = Mock(
        return_value=(
            torch.tensor([True, False]),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
        )
    )
    context = _context(
        scene=_target_scene(
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            timestamp=0.0,
            version=0,
        )
    )

    plan = _plan_action(
        action,
        _invocation(action, GraspGoal(semantics=semantics), sample_count=20),
        context,
    )
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(20, -1),
    )
    assert all(not frame.active_mask[1].item() for frame in plan.commands.frames)
    held = projected.get_held_object("arm")
    assert held is not None
    assert held.env_mask.tolist() == [True, False]


def test_pick_resolves_late_bound_scene_grasp_and_declares_dependency() -> None:
    generator = _motion_generator()
    target_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    target_pose[:, 0, 3] = torch.tensor([0.1, 0.2])
    relative_pose = torch.eye(4)
    relative_pose[2, 3] = 0.05
    entity = Mock()
    entity.get_local_pose.return_value = target_pose
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="late-bound-grasp-object",
        entity=entity,
        entity_id="target",
    )
    action = _bind_action(generator, PickUp())
    context = _context(scene=_target_scene(target_pose, timestamp=0.0, version=0))

    plan = _plan_action(
        action,
        _invocation(
            action,
            GraspGoal(
                semantics=semantics,
                grasp_xpos=SceneEntityPose(
                    "target",
                    relative_pose=relative_pose,
                ),
            ),
            sample_count=20,
        ),
        context,
    )
    projected = plan.expected_effects.apply(context.task, plan.plan_success)
    expected_grasp = torch.bmm(
        target_pose,
        relative_pose.unsqueeze(0).expand(NUM_ENVS, -1, -1),
    )

    held = projected.get_held_object("arm")
    assert held is not None
    assert torch.allclose(held.grasp_xpos, expected_grasp)
    assert plan.scene_dependencies == ("target",)
    assert plan.scene_dependency_monitor_until == {
        "target": plan.segment("approach").stop
    }


def test_pick_session_replans_when_late_bound_target_moves() -> None:
    generator = _motion_generator()
    initial_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    moved_pose = initial_pose.clone()
    moved_pose[:, 1, 3] = 0.3
    entity = Mock()
    entity.get_local_pose.return_value = initial_pose
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="moving-grasp-object",
        entity=entity,
        entity_id="target",
    )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(HAND_DOF),
                grasp=torch.ones(HAND_DOF),
            )
        },
        load_builtins=False,
    )
    action = PickUp()
    engine.register(action)
    _ACTION_ENGINES[id(action)] = engine
    invocation = _invocation(
        action,
        GraspGoal(
            semantics=semantics,
            grasp_xpos=SceneEntityPose("target"),
        ),
        sample_count=20,
    )
    initial_context = _context(
        scene=_target_scene(initial_pose, timestamp=0.0, version=0)
    )
    session = engine.start((invocation,), initial_context)
    session.tick(initial_context)
    entity.get_local_pose.return_value = moved_pose

    recovered = session.tick(
        _context(
            scene=_target_scene(moved_pose, timestamp=0.1, version=1),
            timestamp=0.1,
        )
    )

    event_kinds = {event.kind for event in recovered.events}
    assert ExecutionEventKind.DYNAMIC_GOAL_CHANGED in event_kinds
    assert ExecutionEventKind.REPLANNED in event_kinds


@pytest.mark.parametrize(
    ("waypoint_offset", "expects_replan"),
    ((-1, True), (0, False)),
)
def test_pick_scene_monitoring_window_is_exclusive_at_close_boundary(
    waypoint_offset: int,
    expects_replan: bool,
) -> None:
    """External motion replans before close, while grasp-induced motion does not."""
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    engine = _ACTION_ENGINES[id(action)]
    initial_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    moved_pose = initial_pose.clone()
    moved_pose[:, 1, 3] = 0.3
    invocation = _invocation(
        action,
        GraspGoal(
            semantics=_semantics(entity_id="target"),
            grasp_xpos=torch.eye(4),
        ),
        sample_count=20,
    )
    task_state = TaskState.empty(batch_size=NUM_ENVS, device="cpu")
    qpos = torch.zeros(NUM_ENVS, ROBOT_DOF)

    def context_at(
        pose: torch.Tensor,
        *,
        timestamp: float,
        version: int,
    ) -> PlanningContext:
        return PlanningContext(
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=qpos,
                qvel=torch.zeros_like(qpos),
            ),
            task=task_state,
            scene=_target_scene(pose, timestamp=timestamp, version=version),
            env_ids=torch.arange(NUM_ENVS),
            control_dt=1.0 / 60.0,
        )

    session = engine.start(
        (invocation,), context_at(initial_pose, timestamp=0.0, version=0)
    )
    tick = session.tick(context_at(initial_pose, timestamp=0.0, version=0))
    close_start = session.plan_attempts[0].plan.segment("close").start
    commands_to_issue = close_start + waypoint_offset
    issued = 1
    while issued < commands_to_issue:
        assert tick.command is not None
        for command in tick.command.commands:
            assert isinstance(command.target, JointPositionTarget)
            assert isinstance(command.payload, JointPositionPayload)
            qpos[:, list(command.target.joint_ids)] = command.payload.positions
        tick = session.tick(
            context_at(
                initial_pose,
                timestamp=0.04 * issued,
                version=0,
            )
        )
        issued += 1

    assert tick.command is not None
    for command in tick.command.commands:
        assert isinstance(command.target, JointPositionTarget)
        assert isinstance(command.payload, JointPositionPayload)
        qpos[:, list(command.target.joint_ids)] = command.payload.positions
    moved = session.tick(
        context_at(
            moved_pose,
            timestamp=0.04 * commands_to_issue,
            version=1,
        )
    )

    event_kinds = {event.kind for event in moved.events}
    assert (ExecutionEventKind.DYNAMIC_GOAL_CHANGED in event_kinds) is expects_replan
    assert (ExecutionEventKind.REPLANNED in event_kinds) is expects_replan
    assert len(session.plan_attempts) == (2 if expects_replan else 1)


def test_pick_uses_logical_task_state_key_and_physical_control_target() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    invocation = ActionInvocation(
        skill_id="pick_up",
        goal=GraspGoal(
            semantics=_semantics(entity_id="target"),
            grasp_xpos=torch.eye(4),
        ),
        binding=_binding(
            action,
            motion="alternate_arm",
            grasp="alternate_hand",
            task_state_key="logical_picker",
        ),
        motion_policy=MotionPolicy(sample_count=20),
    )
    context = _context(
        scene=_target_scene(
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            timestamp=0.0,
            version=0,
        )
    )

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert projected.get_held_object("logical_picker") is not None
    assert projected.get_held_object("alternate_arm") is None
    assert projected.get_held_object("arm") is None
    assert {target.target_id for target in plan.commands.targets} == {
        "alternate_arm",
        "alternate_hand",
    }


def test_participant_motion_and_grasp_must_share_task_state_key() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    binding = _binding(action)
    mismatched = ActionBinding(
        owner_id=binding.owner_id,
        endpoints=tuple(
            (
                replace(endpoint, task_state_key="other_participant")
                if endpoint.endpoint_id == "grasp"
                else endpoint
            )
            for endpoint in binding.endpoints
        ),
    )
    invocation = ActionInvocation(
        skill_id="pick_up",
        goal=GraspGoal(
            semantics=_semantics(entity_id="target"),
            grasp_xpos=torch.eye(4),
        ),
        binding=mismatched,
    )
    context = _context(
        scene=_target_scene(
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            timestamp=0.0,
            version=0,
        )
    )

    with pytest.raises(ValueError, match="must share one task_state_key"):
        _plan_action(action, invocation, context)


def test_handover_participants_must_use_distinct_task_state_keys() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        HandOver(
            default_options=HandOverOptions(
                middle_object_pose=torch.eye(4),
                final_object_pose=torch.eye(4),
            )
        ),
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=_semantics()),
        binding=_dual_binding(
            action,
            "source",
            "destination",
            task_state_keys={"source": "same", "destination": "same"},
        ),
    )

    with pytest.raises(ValueError, match="different task_state_key"):
        _plan_action(action, invocation, _dual_context())


def test_press_closes_hand_without_changing_projected_attachment() -> None:
    semantics = ObjectSemantics(
        affordance=PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="button",
    )
    held = _held(semantics)
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": held},
    )
    action = _bind_action(
        _motion_generator(),
        Press(default_options=PressOptions(hand_interp_steps=4)),
    )

    plan = _plan_action(
        action,
        _invocation(
            action,
            PressGoal(semantics, torch.eye(4)),
            sample_count=12,
        ),
        _context(task),
    )
    projected = plan.expected_effects.apply(task, plan.plan_success)

    assert torch.all(_joint_command_positions(plan, "hand")[:, -1] == 1.0)
    projected_held = projected.get_held_object("arm")
    assert projected_held is not None
    assert projected_held.semantics is held.semantics
    assert torch.equal(projected_held.object_to_eef, held.object_to_eef)


@pytest.mark.parametrize(
    ("hold_steps", "expected_segments"),
    (
        (0, ("transfer", "approach", "close", "release", "deliver")),
        (
            2,
            ("transfer", "approach", "close", "hold", "release", "deliver"),
        ),
    ),
)
def test_handover_does_not_mutate_cached_final_pose_and_omits_empty_hold(
    hold_steps: int,
    expected_segments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _dual_motion_generator()
    handover_options = HandOverOptions(
        middle_object_pose=torch.eye(4),
        final_object_pose=torch.eye(4),
        hand_interp_steps=4,
        hold_steps=hold_steps,
        retreat_steps=5,
    )
    action = _bind_action(
        generator,
        HandOver(default_options=handover_options),
    )
    assert handover_options.final_object_pose is not None
    original_final_pose = handover_options.final_object_pose.clone()
    semantics = _semantics(entity_id="handover_object")
    held = _held(semantics)
    held.object_to_eef[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    held.object_to_eef[:, 0, 3] = torch.tensor([0.1, 0.2])
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": held},
    )
    current_eef = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    current_eef[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    current_eef[:, 1, 3] = torch.tensor([0.3, 0.5])
    generator.robot.compute_fk.return_value = current_eef
    generator.robot.compute_fk.side_effect = None
    receive_grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    action._resolve_receive_grasp = Mock(
        return_value=(receive_grasp, torch.ones(NUM_ENVS, dtype=torch.bool))
    )

    def plan_from_start(
        motion_generator: MotionGenerator,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        motion_policy: MotionPolicy,
        interpolation_dt: float | None,
    ) -> tuple[bool, torch.Tensor]:
        del interpolation_dt
        return True, start_qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives.hand_over."
        "plan_named_arm_trajectory",
        plan_from_start,
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(
            semantics=semantics,
            grasp_xpos=SceneEntityPose("unused_grasp_pose"),
        ),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context(task))

    assert plan.plan_success.all()
    assert plan.scene_dependencies == ()
    handover_object_pose = action._resolve_receive_grasp.call_args.args[1]
    expected_current_object_pose = torch.bmm(
        current_eef,
        pose_inv(held.object_to_eef),
    )
    assert torch.allclose(
        handover_object_pose[:, :3, :3],
        expected_current_object_pose[:, :3, :3],
    )
    assert torch.equal(handover_options.final_object_pose, original_final_pose)
    semantics.entity.get_local_pose.assert_not_called()
    assert tuple(segment.name for segment in plan.segments) == expected_segments
    assert plan.segments[0].start == 0
    assert all(
        previous.stop == current.start
        for previous, current in zip(plan.segments, plan.segments[1:])
    )
    assert plan.segments[-1].stop == plan.commands.frame_count


def test_handover_replan_resolves_named_targets_from_latest_snapshot() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        HandOver(
            default_options=HandOverOptions(
                middle_object_pose=SceneEntityPose("target"),
                final_object_pose=SceneEntityPose("target"),
            )
        ),
    )
    semantics = _semantics(entity_id="handover_object")
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": _held(semantics)},
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=semantics),
        binding=_dual_binding(action, "source", "destination"),
    )
    request = action.resolve_request(invocation)
    assert action._scene_dependencies(request) == ("target",)
    captured: list[torch.Tensor] = []
    original_resolve_matrix = action._resolve_matrix

    def capture_middle(matrix: torch.Tensor, name: str) -> torch.Tensor:
        if name == "middle_object_pose":
            captured.append(matrix.clone())
            raise RuntimeError("captured target")
        return original_resolve_matrix(matrix, name)

    action._resolve_matrix = capture_middle  # type: ignore[method-assign]
    first_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    first_pose[:, 0, 3] = 0.3
    with pytest.raises(RuntimeError, match="captured target"):
        action.plan(
            request,
            _dual_context(
                task,
                scene=_target_scene(first_pose, timestamp=0.0, version=0),
            ),
        )
    second_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    second_pose[:, 0, 3] = 0.7
    with pytest.raises(RuntimeError, match="captured target"):
        action.plan(
            request,
            _dual_context(
                task,
                scene=_target_scene(second_pose, timestamp=0.0, version=1),
            ),
        )

    torch.testing.assert_close(captured[0], first_pose)
    torch.testing.assert_close(captured[1], second_pose)


def test_handover_holds_only_environment_with_ik_failure() -> None:
    generator = _dual_motion_generator()
    original_compute_ik = generator.robot.compute_ik.side_effect

    def fail_second_receiving_arm(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        success, qpos = original_compute_ik(
            pose=pose,
            name=name,
            joint_seed=joint_seed,
            **kwargs,
        )
        if name == "right_arm":
            success = success.clone()
            success[1] = False
        return success, qpos

    generator.robot.compute_ik.side_effect = fail_second_receiving_arm
    semantics = _semantics(entity_id="handover_object")
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"logical_source": _held(semantics)},
    )
    action = _bind_action(
        generator,
        HandOver(
            default_options=HandOverOptions(
                middle_object_pose=torch.eye(4),
                final_object_pose=torch.eye(4),
                hand_interp_steps=4,
                hold_steps=2,
                retreat_steps=5,
            )
        ),
    )
    action._resolve_receive_grasp = Mock(
        return_value=(
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.ones(NUM_ENVS, dtype=torch.bool),
        )
    )
    context = _dual_context(task)
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=semantics),
        binding=_dual_binding(
            action,
            "source",
            "destination",
            task_state_keys={
                "source": "logical_source",
                "destination": "logical_destination",
            },
        ),
        motion_policy=MotionPolicy(sample_count=30),
    )

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(30, -1),
    )
    assert all(not frame.active_mask[1].item() for frame in plan.commands.frames)
    received = projected.get_held_object("logical_destination")
    assert received is not None
    assert received.env_mask.tolist() == [True, False]
    assert projected.get_held_object("right_arm") is None
    semantics.entity.get_local_pose.assert_not_called()


def test_handover_rejects_goal_for_a_different_held_object() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        HandOver(
            default_options=HandOverOptions(
                middle_object_pose=torch.eye(4),
                final_object_pose=torch.eye(4),
            )
        ),
    )
    held_semantics = _semantics(entity_id="held_object")
    goal_semantics = _semantics(entity_id="other_object")
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": _held(held_semantics)},
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=goal_semantics),
        binding=_dual_binding(action, "source", "destination"),
    )

    with pytest.raises(ValueError, match="must identify the object held"):
        _plan_action(action, invocation, _dual_context(task))

    held_semantics.entity.get_local_pose.assert_not_called()
    goal_semantics.entity.get_local_pose.assert_not_called()


@pytest.mark.parametrize(
    ("hold_steps", "expected_segments"),
    (
        (0, ("approach", "close", "lift", "move")),
        (2, ("approach", "close", "lift", "move", "hold")),
    ),
)
def test_coordinated_pick_returns_full_dof_plan_and_omits_empty_hold(
    hold_steps: int,
    expected_segments: tuple[str, ...],
) -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        CoordinatedPickment(
            default_options=CoordinatedPickmentOptions(
                hand_interp_steps=4,
                hold_steps=hold_steps,
                object_motion_keyframes=3,
            ),
        ),
    )
    affordance = AntipodalAffordance()
    _stub_dual_arm_grasp_poses(affordance)
    entity = Mock()
    entity.get_local_pose.return_value = torch.full((NUM_ENVS, 4, 4), 9.0)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="coordinated-object",
        entity=entity,
        entity_id="coordinated_object",
    )
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=semantics,
            object_target_pose=torch.eye(4),
            object_initial_pose=torch.eye(4),
        ),
        binding=_dual_binding(
            action,
            "left",
            "right",
            task_state_keys={
                "left": "logical_left",
                "right": "logical_right",
            },
        ),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context()

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, True]
    assert _joint_trajectory(plan).positions.shape == (
        NUM_ENVS,
        30,
        DUAL_ROBOT_DOF,
    )
    assert plan.commands.frame_count == 30
    assert {target.target_id for target in plan.commands.targets} == {
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    }
    assert plan.scene_dependencies == ()
    request.goal.semantics.entity.get_local_pose.assert_not_called()
    assert projected.get_held_object("logical_left") is None
    assert projected.get_held_object("logical_right") is None
    assert isinstance(
        projected.get_coordinated_held_object("logical_left", "logical_right"),
        CoordinatedHeldObjectState,
    )
    assert projected.get_coordinated_held_object("left_arm", "right_arm") is None
    assert tuple(segment.name for segment in plan.segments) == expected_segments
    assert plan.segments[0].start == 0
    assert all(
        previous.stop == current.start
        for previous, current in zip(plan.segments, plan.segments[1:])
    )
    assert plan.segments[-1].stop == plan.commands.frame_count


def test_coordinated_pick_implicit_initial_pose_uses_scene_snapshot() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        CoordinatedPickment(
            default_options=CoordinatedPickmentOptions(
                hand_interp_steps=4,
                hold_steps=2,
                object_motion_keyframes=3,
            ),
        ),
    )
    affordance = AntipodalAffordance()
    _stub_dual_arm_grasp_poses(affordance)
    entity = Mock()
    entity.get_local_pose.return_value = torch.full((NUM_ENVS, 4, 4), 9.0)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="snapshot-coordinated-object",
        entity=entity,
        entity_id="target",
    )
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    object_pose[:, 1, 3] = torch.tensor([0.2, 0.4])
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=semantics,
            object_target_pose=object_pose,
        ),
        binding=_dual_binding(action, "left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context(scene=_target_scene(object_pose, timestamp=0.0, version=0))

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    resolved_affordance = request.goal.semantics.affordance
    sampled_pose = resolved_affordance.get_dual_arm_valid_grasp_poses.call_args.kwargs[
        "obj_poses"
    ]
    assert torch.equal(sampled_pose, object_pose)
    assert plan.scene_dependencies == ("target",)
    request.goal.semantics.entity.get_local_pose.assert_not_called()
    coordinated = projected.get_coordinated_held_object("left_arm", "right_arm")
    assert coordinated is not None
    assert torch.allclose(coordinated.left_object_to_eef, pose_inv(object_pose))
    assert torch.allclose(coordinated.right_object_to_eef, pose_inv(object_pose))


def test_assemble_place_uses_explicit_base_snapshot() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, Place())
    base_entity = Mock()
    base_entity.get_local_pose.return_value = torch.full((NUM_ENVS, 4, 4), 9.0)
    relative_pose = torch.eye(4)
    relative_pose[2, 3] = 0.05
    affordance = AssembleAffordance(
        base_object_entity=base_entity,
        assemble_to_base_pose=relative_pose,
    )
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"logical_arm": _held(_semantics(entity_id="assemble_object"))},
    )
    base_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    base_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    context = _context(
        task,
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={"base": EntityState(base_pose)},
        ),
    )

    request = action.resolve_request(
        replace(
            _invocation(
                action,
                AssembleGoal(
                    affordance=affordance,
                    base_pose=SceneEntityPose("base"),
                ),
            ),
            binding=_binding(action, task_state_key="logical_arm"),
        )
    )
    plan = action.plan(request, context)

    assert plan.plan_success.all()
    assert plan.scene_dependencies == ("base",)
    request.goal.affordance.base_object_entity.get_local_pose.assert_not_called()


def test_assemble_place_legacy_base_entity_warns() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, Place())
    base_entity = Mock()
    base_entity.get_local_pose.return_value = torch.eye(4)
    affordance = AssembleAffordance(base_object_entity=base_entity)
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": _held()},
    )

    request = action.resolve_request(
        _invocation(action, AssembleGoal(affordance=affordance))
    )
    with pytest.warns(DeprecationWarning, match="base_pose"):
        plan = action.plan(request, _context(task))

    assert plan.scene_dependencies == ()
    request.goal.affordance.base_object_entity.get_local_pose.assert_called_once_with(
        to_matrix=True
    )


def test_coordinated_pick_holds_only_environment_with_ik_failure() -> None:
    generator = _dual_motion_generator()
    original_compute_ik = generator.robot.compute_ik.side_effect

    def fail_second_environment(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        success, qpos = original_compute_ik(
            pose=pose,
            name=name,
            joint_seed=joint_seed,
            **kwargs,
        )
        if name == "right_arm" and float(pose[1, 0, 3]) > 0.15:
            success = success.clone()
            success[1] = False
        return success, qpos

    generator.robot.compute_ik.side_effect = fail_second_environment
    action = _bind_action(
        generator,
        CoordinatedPickment(
            default_options=CoordinatedPickmentOptions(
                hand_interp_steps=4,
                hold_steps=2,
                object_motion_keyframes=3,
            ),
        ),
    )
    target_pose = torch.eye(4)
    target_pose[0, 3] = 0.3
    affordance = AntipodalAffordance()
    _stub_dual_arm_grasp_poses(affordance)
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=ObjectSemantics(affordance=affordance, geometry={}, label="tray"),
            object_target_pose=target_pose,
            object_initial_pose=torch.eye(4),
        ),
        binding=_dual_binding(action, "left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context()

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).repeat(30, 1),
    )
    assert all(not frame.active_mask[1].item() for frame in plan.commands.frames)
    coordinated = projected.get_coordinated_held_object("left_arm", "right_arm")
    assert coordinated is not None
    assert coordinated.env_mask.tolist() == [True, False]


def test_coordinated_pick_fails_when_affordance_has_no_grasp() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        CoordinatedPickment(
            default_options=CoordinatedPickmentOptions(
                hand_interp_steps=4,
                hold_steps=2,
                object_motion_keyframes=3,
            ),
        ),
    )
    affordance = AntipodalAffordance()
    affordance.get_dual_arm_valid_grasp_poses = Mock(
        return_value=[
            None,
            {
                "left": {
                    "is_success": False,
                    "grasp_poses": torch.eye(4, dtype=torch.float32),
                    "open_lengths": torch.tensor([0.0], dtype=torch.float32),
                    "total_cost": torch.tensor([torch.inf], dtype=torch.float32),
                },
                "right": {
                    "is_success": True,
                    "grasp_poses": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                    "open_lengths": torch.tensor([0.0], dtype=torch.float32),
                    "total_cost": torch.tensor([0.0], dtype=torch.float32),
                },
            },
        ]
    )
    semantics = ObjectSemantics(affordance=affordance, geometry={}, label="object")
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=semantics,
            object_target_pose=torch.eye(4),
            object_initial_pose=torch.eye(4),
        ),
        binding=_dual_binding(action, "left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context())

    assert plan.plan_success.tolist() == [False, False]
    assert plan.commands.frame_count == 0
    assert _joint_trajectory(plan).positions.shape == (
        NUM_ENVS,
        0,
        DUAL_ROBOT_DOF,
    )


@pytest.mark.parametrize(
    ("release", "hold_steps", "expected_segments"),
    (
        (False, 0, ("approach", "retreat")),
        (True, 3, ("approach", "hold", "release", "retreat")),
    ),
)
def test_coordinated_placement_projects_effects_and_omits_empty_segments(
    release: bool,
    hold_steps: int,
    expected_segments: tuple[str, ...],
) -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        CoordinatedPlacement(
            default_options=CoordinatedPlacementOptions(
                release=release,
                hand_interp_steps=4,
                hold_steps=hold_steps,
                retreat_steps=5,
            ),
        ),
    )
    placing = _held(
        ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="placing")
    )
    support = _held(
        ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="support")
    )
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "logical_placing": placing,
            "logical_support": support,
        },
    )
    invocation = ActionInvocation(
        skill_id="coordinated_placement",
        goal=CoordinatedPlacementGoal(
            placing_object_target_pose=torch.eye(4),
            support_object_target_pose=torch.eye(4),
        ),
        binding=_dual_binding(
            action,
            "placing",
            "support",
            task_state_keys={
                "placing": "logical_placing",
                "support": "logical_support",
            },
        ),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context(task)

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, True]
    assert plan.commands.frame_count == 30
    assert {target.target_id for target in plan.commands.targets} == {
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    }
    projected_placing = projected.get_held_object("logical_placing")
    if release:
        assert projected_placing is None
    else:
        assert projected_placing is not None
        assert projected_placing.semantics is placing.semantics
        assert torch.equal(projected_placing.object_to_eef, placing.object_to_eef)
    assert projected.get_held_object("logical_support") is not None
    assert projected.get_held_object("logical_support").semantics is support.semantics
    assert projected.get_held_object("right_arm") is None
    assert tuple(segment.name for segment in plan.segments) == expected_segments
    assert plan.segments[0].start == 0
    assert all(
        previous.stop == current.start
        for previous, current in zip(plan.segments, plan.segments[1:])
    )
    assert plan.segments[-1].stop == plan.commands.frame_count


def test_coordinated_placement_rejects_one_object_held_by_both_arms() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, CoordinatedPlacement())
    semantics = _semantics()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "left_arm": _held(semantics),
            "right_arm": _held(semantics),
        },
    )
    invocation = ActionInvocation(
        skill_id="coordinated_placement",
        goal=CoordinatedPlacementGoal(
            placing_object_target_pose=torch.eye(4),
            support_object_target_pose=torch.eye(4),
        ),
        binding=_dual_binding(action, "placing", "support"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context(task))

    assert plan.plan_success.tolist() == [False, False]
    assert _joint_trajectory(plan).waypoint_count == 0
    generator.robot.compute_ik.assert_not_called()


def test_coordinated_placement_holds_only_environment_with_ik_failure() -> None:
    generator = _dual_motion_generator()
    original_compute_ik = generator.robot.compute_ik.side_effect

    def fail_second_support_arm(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        success, qpos = original_compute_ik(
            pose=pose,
            name=name,
            joint_seed=joint_seed,
            **kwargs,
        )
        if name == "right_arm":
            success = success.clone()
            success[1] = False
        return success, qpos

    generator.robot.compute_ik.side_effect = fail_second_support_arm
    action = _bind_action(
        generator,
        CoordinatedPlacement(
            default_options=CoordinatedPlacementOptions(
                hand_interp_steps=4,
                hold_steps=3,
                retreat_steps=5,
            )
        ),
    )
    placing = _held(
        ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="placing")
    )
    support = _held(
        ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="support")
    )
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": placing, "right_arm": support},
    )
    context = _dual_context(task)
    invocation = ActionInvocation(
        skill_id="coordinated_placement",
        goal=CoordinatedPlacementGoal(
            placing_object_target_pose=torch.eye(4),
            support_object_target_pose=torch.eye(4),
        ),
        binding=_dual_binding(action, "placing", "support"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(30, -1),
    )
    assert all(not frame.active_mask[1].item() for frame in plan.commands.frames)
    supported = projected.get_held_object("right_arm")
    assert supported is not None
    assert supported.env_mask.tolist() == [True, True]
    still_placing = projected.get_held_object("left_arm")
    assert still_placing is not None
    assert still_placing.env_mask.tolist() == [False, True]


def test_coordinated_actions_reject_curobo_motion_generation() -> None:
    generator = _dual_motion_generator()
    generator.planner.cfg.planner_type = "curobo"
    policy = MotionPolicy(strategy="motion_gen", sample_count=30)
    pick = _bind_action(
        generator,
        CoordinatedPickment(),
    )
    pick_invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=ObjectSemantics(
                affordance=AntipodalAffordance(), geometry={}, label="object"
            ),
            object_target_pose=torch.eye(4),
            object_initial_pose=torch.eye(4),
        ),
        binding=_dual_binding(pick, "left", "right"),
        motion_policy=policy,
    )

    with pytest.raises(ValueError, match="not supported"):
        _plan_action(pick, pick_invocation, _dual_context())

    placement = _bind_action(
        generator,
        CoordinatedPlacement(),
    )
    placement_invocation = ActionInvocation(
        skill_id="coordinated_placement",
        goal=CoordinatedPlacementGoal(torch.eye(4), torch.eye(4)),
        binding=_dual_binding(placement, "placing", "support"),
        motion_policy=policy,
    )
    with pytest.raises(ValueError, match="not supported"):
        _plan_action(placement, placement_invocation, _dual_context())
