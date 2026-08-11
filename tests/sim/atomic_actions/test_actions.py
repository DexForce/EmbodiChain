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

from typing import TypeVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    Affordance,
    AntipodalAffordance,
    AssembleGoal,
    AtomicAction,
    AtomicActionEngine,
    ControlPartCommandProfile,
    CoordinatedHeldObjectState,
    CoordinatedPickGoal,
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
    MotionPolicy,
    MoveEndEffector,
    MoveEndEffectorOptions,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    PickUp,
    PickUpOptions,
    Place,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    Press,
    PressGoal,
    PressOptions,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.planners import (
    MotionGenerator,
    MoveType,
    PlanOptions,
    PlanResult,
)

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
ROBOT_DOF = ARM_DOF + HAND_DOF
DUAL_ARM_DOF = 2 * ARM_DOF
DUAL_ROBOT_DOF = DUAL_ARM_DOF + 2 * HAND_DOF

ActionT = TypeVar("ActionT", bound=AtomicAction)


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


def _binding() -> ActionBinding:
    return ActionBinding(
        manipulators={"primary": "arm"},
        end_effectors={"primary": "hand"},
    )


def _invocation(
    skill_id: str,
    goal,
    *,
    sample_count: int = 20,
) -> ActionInvocation:
    return ActionInvocation(
        skill_id=skill_id,
        goal=goal,
        binding=_binding(),
        motion_policy=MotionPolicy(sample_count=sample_count),
    )


def _semantics() -> ObjectSemantics:
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="test_object",
        entity=entity,
    )


def _held(semantics: ObjectSemantics | None = None) -> HeldObjectState:
    poses = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    return HeldObjectState(
        semantics=semantics or _semantics(),
        object_to_eef=poses,
        grasp_xpos=poses,
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
    generator.planner.preserve_plan_samples = False
    generator.planner.supports_move_type.return_value = False
    generator.planner.default_plan_options.return_value = PlanOptions()
    generator.planner.with_motion_context.side_effect = (
        lambda options, *, start_qpos, control_part: options
    )
    return generator


def _dual_context(task: TaskState | None = None) -> PlanningContext:
    qpos = torch.zeros(NUM_ENVS, DUAL_ROBOT_DOF)
    return PlanningContext(
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
        task=task or TaskState.empty(NUM_ENVS, "cpu"),
        scene=SceneSnapshot.empty(),
        env_ids=torch.arange(NUM_ENVS),
    )


def _dual_binding(
    first_role: str,
    second_role: str,
) -> ActionBinding:
    return ActionBinding(
        manipulators={
            first_role: "left_arm",
            second_role: "right_arm",
        },
        end_effectors={
            first_role: "left_hand",
            second_role: "right_hand",
        },
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


def test_builtin_descriptors_expose_goals_not_legacy_targets() -> None:
    assert MoveEndEffector.GoalType is EndEffectorPoseGoal
    assert MoveJoints.GoalType is JointPositionGoal
    assert PickUp.GoalType is GraspGoal
    assert MoveHeldObject.GoalType is HeldObjectPoseGoal
    assert Place.GoalType == (PlaceGoal, AssembleGoal)
    assert Press.GoalType is PressGoal
    assert CoordinatedPickment.GoalType is CoordinatedPickGoal
    assert CoordinatedPlacement.GoalType is CoordinatedPlacementGoal
    assert HandOver.GoalType is GraspGoal


@pytest.mark.parametrize(
    "options",
    (
        PickUpOptions(),
        MoveHeldObjectOptions(),
        PlaceOptions(),
        PressOptions(),
        CoordinatedPickmentOptions(),
        CoordinatedPlacementOptions(),
        HandOverOptions(),
    ),
)
def test_action_options_do_not_contain_embodiment_resources(options: object) -> None:
    field_names = getattr(options, "__dataclass_fields__")

    assert "control_part" not in field_names
    assert not any(name.endswith("_control_part") for name in field_names)
    assert not any(name.endswith("_qpos") for name in field_names)


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
            "move_end_effector",
            EndEffectorPoseGoal(torch.eye(4)),
            sample_count=10,
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 10, ROBOT_DOF)
    assert plan.trajectory.duration.tolist() == pytest.approx([0.15, 0.15])
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
    )

    plan = _plan_action(
        action,
        _invocation("move_joints", JointPositionGoal("ready"), sample_count=8),
        context,
    )

    assert torch.allclose(plan.trajectory.positions[:, -1, :ARM_DOF], named["ready"])
    assert torch.all(plan.trajectory.positions[:, :, ARM_DOF:] == 0.7)


def test_pick_and_place_declare_effects_without_mutating_context() -> None:
    generator = _motion_generator()
    pick = _bind_action(generator, PickUp())
    initial = _context()
    semantics = _semantics()
    grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    pick_plan = _plan_action(
        pick,
        _invocation("pick_up", GraspGoal(semantics=semantics, grasp_xpos=grasp)),
        initial,
    )
    picked_task = pick_plan.expected_effects.apply(initial.task, pick_plan.plan_success)

    assert initial.task.get_held_object("arm") is None
    assert picked_task.get_held_object("arm") is not None

    place = _bind_action(generator, Place())
    picked_context = PlanningContext(
        robot=initial.robot,
        task=picked_task,
        scene=initial.scene,
        env_ids=initial.env_ids,
    )
    place_plan = _plan_action(
        place,
        _invocation("place", PlaceGoal(torch.eye(4))),
        picked_context,
    )
    placed_task = place_plan.expected_effects.apply(
        picked_task, place_plan.plan_success
    )

    assert picked_task.get_held_object("arm") is not None
    assert placed_task.get_held_object("arm") is None


def test_move_held_object_requires_projected_attachment() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, MoveHeldObject())
    invocation = _invocation(
        "move_held_object",
        HeldObjectPoseGoal(torch.eye(4)),
        sample_count=10,
    )

    with pytest.raises(ValueError, match="requires an object held"):
        _plan_action(action, invocation, _context())

    held = _held()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": held},
    )
    plan = _plan_action(action, invocation, _context(task))
    assert plan.plan_success.all()
    assert plan.expected_effects.is_empty


def test_press_uses_invocation_sample_budget() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, Press())

    plan = _plan_action(
        action,
        _invocation("press", PressGoal(torch.eye(4)), sample_count=12),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.waypoint_count == 12
    assert plan.expected_effects.is_empty


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
        binding=ActionBinding(manipulators={"primary": "arm"}),
    )
    with pytest.raises(ValueError, match="skill_id"):
        action.resolve_request(invocation)


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
        duration=torch.full((NUM_ENVS,), 0.3),
    )
    action = _bind_action(generator, MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.ones(ARM_DOF)),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(strategy="motion_gen", sample_count=3),
    )

    plan = _plan_action(action, invocation, _context())

    assert plan.trajectory.duration.tolist() == pytest.approx([0.3, 0.3])
    assert plan.trajectory.velocities is not None
    assert torch.all(plan.trajectory.velocities[:, :, :ARM_DOF] == 0.5)
    assert torch.all(plan.trajectory.velocities[:, :, ARM_DOF:] == 0.0)


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
            "move_end_effector",
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
            "move_joints",
            JointPositionGoal(waypoints),
            sample_count=7,
        ),
        _context(),
    )

    assert torch.allclose(plan.trajectory.positions[:, 3, :ARM_DOF], waypoints[:, 0])
    assert torch.allclose(plan.trajectory.positions[:, -1, :ARM_DOF], waypoints[:, 1])
    with pytest.raises(KeyError, match="has no command"):
        _plan_action(
            action,
            _invocation("move_joints", JointPositionGoal("missing")),
            _context(),
        )


def test_pick_explicit_grasp_bypasses_sampling_and_records_grasp() -> None:
    generator = _motion_generator()
    affordance = AntipodalAffordance()
    affordance.get_valid_grasp_poses = Mock()
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="explicit-grasp-object",
        entity=entity,
    )
    grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp[:, 0, 3] = torch.tensor([0.1, 0.2])
    action = _bind_action(generator, PickUp())

    plan = _plan_action(
        action,
        _invocation(
            "pick_up",
            GraspGoal(semantics=semantics, grasp_xpos=grasp),
            sample_count=20,
        ),
        _context(),
    )
    projected = plan.expected_effects.apply(_context().task, plan.plan_success)

    affordance.get_valid_grasp_poses.assert_not_called()
    held = projected.get_held_object("arm")
    assert held is not None
    assert torch.allclose(held.grasp_xpos, grasp)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "close",
        "lift",
    ]
    assert plan.segment("close").stop == plan.segment("lift").start


def test_pick_holds_only_environment_without_a_feasible_grasp() -> None:
    generator = _motion_generator()
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="partially-graspable-object",
        entity=entity,
    )
    action = _bind_action(generator, PickUp())
    action._resolve_grasp_pose = Mock(
        return_value=(
            torch.tensor([True, False]),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
        )
    )
    context = _context()

    plan = _plan_action(
        action,
        _invocation("pick_up", GraspGoal(semantics=semantics), sample_count=20),
        context,
    )
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(20, -1),
    )
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
    )
    action = _bind_action(generator, PickUp())
    context = _context(scene=_target_scene(target_pose, timestamp=0.0, version=0))

    plan = _plan_action(
        action,
        _invocation(
            "pick_up",
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
    engine.register(PickUp())
    invocation = _invocation(
        "pick_up",
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


def test_pick_uses_binding_control_part_as_effect_resource() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    invocation = ActionInvocation(
        skill_id="pick_up",
        goal=GraspGoal(semantics=_semantics(), grasp_xpos=torch.eye(4)),
        binding=ActionBinding(
            manipulators={"primary": "alternate_arm"},
            end_effectors={"primary": "alternate_hand"},
        ),
        motion_policy=MotionPolicy(sample_count=20),
    )
    context = _context()

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert projected.get_held_object("alternate_arm") is not None
    assert projected.get_held_object("arm") is None


def test_press_closes_hand_without_changing_projected_attachment() -> None:
    held = _held()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": held},
    )
    generator = _motion_generator()
    action = _bind_action(
        generator,
        Press(default_options=PressOptions(hand_interp_steps=4)),
    )

    plan = _plan_action(
        action,
        _invocation("press", PressGoal(torch.eye(4)), sample_count=12),
        _context(task),
    )
    projected = plan.expected_effects.apply(task, plan.plan_success)

    assert torch.all(plan.trajectory.positions[:, -1, ARM_DOF:] == 1.0)
    projected_held = projected.get_held_object("arm")
    assert projected_held is not None
    assert projected_held.semantics is held.semantics
    assert torch.equal(projected_held.object_to_eef, held.object_to_eef)


def test_handover_does_not_mutate_cached_final_pose() -> None:
    generator = _dual_motion_generator()
    handover_options = HandOverOptions(
        middle_object_pose=torch.eye(4),
        final_object_pose=torch.eye(4),
        hand_interp_steps=4,
        hold_steps=2,
        retreat_steps=5,
    )
    action = _bind_action(
        generator,
        HandOver(default_options=handover_options),
    )
    assert handover_options.final_object_pose is not None
    original_final_pose = handover_options.final_object_pose.clone()
    current_object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    current_object_pose[:, :3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    semantics = _semantics()
    semantics.entity.get_local_pose.return_value = current_object_pose
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": _held(semantics)},
    )
    receive_grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    action._resolve_receive_grasp = Mock(
        return_value=(receive_grasp, torch.ones(NUM_ENVS, dtype=torch.bool))
    )

    def plan_from_start(
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        motion_policy: MotionPolicy,
    ) -> tuple[bool, torch.Tensor]:
        return True, start_qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    action._plan_named_arm_trajectory = Mock(side_effect=plan_from_start)
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=semantics),
        binding=_dual_binding("source", "destination"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context(task))

    assert plan.plan_success.all()
    assert torch.equal(handover_options.final_object_pose, original_final_pose)
    assert [segment.name for segment in plan.segments] == [
        "transfer",
        "approach",
        "close",
        "hold",
        "release",
        "deliver",
    ]


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
    semantics = _semantics()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": _held(semantics)},
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
        binding=_dual_binding("source", "destination"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(30, -1),
    )
    received = projected.get_held_object("right_arm")
    assert received is not None
    assert received.env_mask.tolist() == [True, False]


def test_coordinated_pick_returns_full_dof_plan_and_projected_relation() -> None:
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
    semantics = ObjectSemantics(
        affordance=affordance, geometry={}, label="coordinated-object"
    )
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=semantics,
            object_target_pose=torch.eye(4),
            object_initial_pose=torch.eye(4),
        ),
        binding=_dual_binding("left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context()

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 30, DUAL_ROBOT_DOF)
    assert projected.get_held_object("left_arm") is None
    assert projected.get_held_object("right_arm") is None
    assert isinstance(
        projected.get_coordinated_held_object("left_arm", "right_arm"),
        CoordinatedHeldObjectState,
    )
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "close",
        "lift",
        "move",
        "hold",
    ]


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
        binding=_dual_binding("left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context()

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).repeat(30, 1),
    )
    held = projected.get_coordinated_held_object("left_arm", "right_arm")
    assert held is not None
    assert held.env_mask.tolist() == [True, False]


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
        binding=_dual_binding("left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context())

    assert plan.plan_success.tolist() == [False, False]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 0, DUAL_ROBOT_DOF)


def test_coordinated_placement_projects_release_and_support_attachment() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(
        generator,
        CoordinatedPlacement(
            default_options=CoordinatedPlacementOptions(
                hand_interp_steps=4,
                hold_steps=3,
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
        held_objects={"left_arm": placing, "right_arm": support},
    )
    invocation = ActionInvocation(
        skill_id="coordinated_placement",
        goal=CoordinatedPlacementGoal(
            placing_object_target_pose=torch.eye(4),
            support_object_target_pose=torch.eye(4),
        ),
        binding=_dual_binding("placing", "support"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context(task)

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 30, DUAL_ROBOT_DOF)
    assert projected.get_held_object("left_arm") is None
    assert projected.get_held_object("right_arm") is not None
    assert projected.get_held_object("right_arm").semantics is support.semantics
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "hold",
        "release",
        "retreat",
    ]


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
        binding=_dual_binding("placing", "support"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(30, -1),
    )
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
        binding=_dual_binding("left", "right"),
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
        binding=_dual_binding("placing", "support"),
        motion_policy=policy,
    )
    with pytest.raises(ValueError, match="not supported"):
        _plan_action(placement, placement_invocation, _dual_context())
