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

import math
from typing import Literal, TypeVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    AssembleGoal,
    AtomicAction,
    AtomicActionEngine,
    ControlPartCommandProfile,
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
    PressAffordance,
    PressGoal,
    PressOptions,
    SlideAffordance,
    Slide,
    SlideGoal,
    SlideOptions,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    TaskState,
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
    assert Slide.GoalType is SlideGoal
    assert Twist.GoalType is TwistGoal
    assert CoordinatedPickment.GoalType is CoordinatedPickGoal
    assert CoordinatedPlacement.GoalType is CoordinatedPlacementGoal
    assert HandOver.GoalType is GraspGoal


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
        control_dt=CONTROL_DT,
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
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    initial = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))
    semantics = _semantics(entity_id="target")
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
        control_dt=initial.control_dt,
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
        _invocation("place", PlaceGoal(torch.eye(4))),
        context,
    )
    projected = plan.expected_effects.apply(task, plan.plan_success)

    assert plan.plan_success.tolist() == [False, True]
    assert torch.allclose(
        plan.trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(plan.trajectory.waypoint_count, -1),
    )
    primary = projected.get_held_object("arm")
    alternate = projected.get_held_object("alternate_arm")
    assert primary is not None and primary.env_mask.tolist() == [True, False]
    assert alternate is not None and alternate.env_mask.tolist() == [True, False]


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

    semantics = _semantics()
    held = _held(semantics)
    held.object_to_eef[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    held.object_to_eef[:, 0, 3] = torch.tensor([0.1, 0.2])
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": held},
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
        binding=_binding(),
        motion_policy=MotionPolicy(sample_count=10),
        skill_options=MoveHeldObjectOptions(pick_rotate_upright=0.25),
    )

    plan = _plan_action(action, configured_invocation, _context(task))

    assert plan.plan_success.all()
    assert plan.expected_effects.is_empty
    current_object_pose = action._apply_configured_upright_rotation.call_args.args[2]
    assert torch.allclose(
        current_object_pose,
        torch.bmm(eef_pose, pose_inv(held.object_to_eef)),
    )
    semantics.entity.get_local_pose.assert_not_called()


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


def test_move_joints_rejects_incompatible_goal_at_action_boundary() -> None:
    action = _bind_action(_motion_generator(), MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=object(),  # type: ignore[arg-type]
        binding=ActionBinding(manipulators={"primary": "arm"}),
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
            "move_end_effector",
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
            "pick_up",
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
        entity_id="target",
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
        goal=GraspGoal(
            semantics=_semantics(entity_id="target"),
            grasp_xpos=torch.eye(4),
        ),
        binding=ActionBinding(
            manipulators={"primary": "alternate_arm"},
            end_effectors={"primary": "alternate_hand"},
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

    assert projected.get_held_object("alternate_arm") is not None
    assert projected.get_held_object("arm") is None


def test_twist_plans_six_segments_from_articulation_link() -> None:
    affordance = TwistAffordance(
        grasp_position=(0.0, 0.0, 0.0),
        axis_origin=(0.0, 0.0, 0.0),
        twist_axis=torch.tensor([0.0, 1.0, 0.0]),
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="knob",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Twist())

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="twist",
            goal=TwistGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=TwistOptions(hand_interp_steps=3),
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "reach",
        "close",
        "twist",
        "open",
        "retract",
    ]
    assert torch.all(
        plan.trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    assert torch.all(
        plan.trajectory.positions[:, plan.segment("open").stop - 1, ARM_DOF:] == 0.0
    )
    first_target = generator.robot.compute_ik.call_args_list[0].kwargs["pose"]
    grasp_pose = affordance.get_grasp_pose(torch.eye(4).repeat(NUM_ENVS, 1, 1))
    expected_pre_grasp_position = (
        grasp_pose[:, :3, 3] - grasp_pose[:, :3, 2] * TwistOptions().pre_grasp_distance
    )
    assert torch.allclose(first_target[:, :3, 3], expected_pre_grasp_position)


def test_twist_plans_from_explicit_rigid_object_pose_snapshot() -> None:
    semantics = ObjectSemantics(
        affordance=TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(0.0, 0.0, 0.0),
            twist_axis=torch.tensor([0.0, 1.0, 0.0]),
        ),
        geometry={},
        label="rigid-knob",
    )

    plan = _plan_action(
        _bind_action(_motion_generator(), Twist()),
        ActionInvocation(
            skill_id="twist",
            goal=TwistGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=TwistOptions(hand_interp_steps=3),
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]


def test_twist_rotates_grasp_about_explicit_axis_origin() -> None:
    action = _bind_action(_motion_generator(), Twist())
    target_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp_pose = target_pose.clone()
    grasp_pose[:, 0, 3] = 2.0

    twisted = action._twisted_grasp_poses(
        target_pose,
        grasp_pose,
        torch.tensor([0.0, 0.0, 1.0]),
        (1.0, 0.0, 0.0),
        math.pi / 2,
        4,
    )

    assert torch.allclose(
        twisted[:, -1, :3, 3],
        torch.tensor([1.0, 1.0, 0.0]).expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("goal_factory", "affordance"),
    (
        (
            PressGoal,
            PressAffordance(
                press_axis=torch.tensor([1.0, 0.0, 0.0]),
                press_position=(0.0, 0.0, 0.0),
            ),
        ),
        (
            SlideGoal,
            SlideAffordance(
                mesh_vertices=torch.zeros(3, 3),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
            ),
        ),
        (
            TwistGoal,
            TwistAffordance(
                grasp_position=(0.0, 0.0, 0.0),
                axis_origin=(0.0, 0.0, 0.0),
            ),
        ),
    ),
)
def test_interaction_goal_collects_target_scene_dependency(
    goal_factory,
    affordance,
) -> None:
    semantics = ObjectSemantics(affordance=affordance, geometry={}, label="target")
    goal = goal_factory(semantics, SceneEntityPose("target-link"))

    assert collect_scene_dependencies(goal) == ("target-link",)


def test_open_loop_interaction_primitives_are_explicitly_described() -> None:
    assert Press.descriptor().open_loop is True
    assert Slide.descriptor().open_loop is True
    assert Twist.descriptor().open_loop is True


def test_twist_session_replans_when_scene_target_moves() -> None:
    generator = _motion_generator()
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
    engine.register(Twist())
    semantics = ObjectSemantics(
        affordance=TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="moving-knob",
    )
    invocation = ActionInvocation(
        skill_id="twist",
        goal=TwistGoal(semantics, SceneEntityPose("target")),
        binding=_binding(),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=TwistOptions(hand_interp_steps=3),
    )
    initial_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    initial_context = _context(
        scene=_target_scene(initial_pose, timestamp=0.0, version=0)
    )
    session = engine.start((invocation,), initial_context)
    session.tick(initial_context)
    moved_pose = initial_pose.clone()
    moved_pose[:, 1, 3] = 0.3

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
    ("direction", "expected_segments", "translation_sign"),
    (
        ("pull", ["approach", "reach", "close", "pull", "open"], -1.0),
        (
            "push",
            ["approach", "reach", "close", "push", "open", "return"],
            1.0,
        ),
    ),
)
def test_slide_plans_expected_segments(
    direction: Literal["pull", "push"],
    expected_segments: list[str],
    translation_sign: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = torch.tensor(
        [
            [-0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    link_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    affordance = SlideAffordance(
        mesh_vertices=vertices,
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        translation_axis=torch.tensor([0.0, -1.0, 0.0]),
    )
    grasp_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sample_grasp(
        self: SlideAffordance,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_calls.append((obj_poses, approach_direction))
        return (
            torch.ones(NUM_ENVS, dtype=torch.bool),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.full((NUM_ENVS,), 0.03),
        )

    monkeypatch.setattr(
        SlideAffordance,
        "get_best_grasp_poses",
        sample_grasp,
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="drawer_handle",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Slide())
    options = SlideOptions(
        direction=direction,
        hand_interp_steps=3,
        approach_distance=0.1,
        translation_distance=0.15,
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="slide",
            goal=SlideGoal(semantics, link_pose),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == expected_segments
    assert torch.all(
        plan.trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    assert torch.all(
        plan.trajectory.positions[:, plan.segment("open").stop - 1, ARM_DOF:] == 0.0
    )
    assert len(grasp_calls) == 1
    assert torch.equal(grasp_calls[0][0], link_pose)
    assert torch.allclose(
        grasp_calls[0][1],
        torch.tensor([0.0, -1.0, 0.0]).expand(NUM_ENVS, -1),
    )
    planned_targets = [
        call.kwargs["pose"] for call in generator.robot.compute_ik.call_args_list
    ]
    expected_axis = torch.tensor([0.0, -1.0, 0.0])
    motion_lengths = Slide._motion_segment_lengths(
        24,
        options.hand_interp_steps,
        direction=direction,
    )
    assert torch.allclose(
        planned_targets[0][:, :3, 3],
        -expected_axis.expand(NUM_ENVS, -1) * options.approach_distance,
    )
    reach_stop = 1 + motion_lengths[1] - 1
    assert torch.allclose(
        planned_targets[reach_stop - 1][:, :3, 3],
        torch.zeros(NUM_ENVS, 3),
    )
    translate_stop = reach_stop + motion_lengths[2] - 1
    translated_targets = torch.stack(
        [pose[:, :3, 3] for pose in planned_targets[reach_stop:translate_stop]],
        dim=1,
    )
    assert torch.allclose(
        translated_targets[:, -1],
        expected_axis.expand(NUM_ENVS, -1)
        * (translation_sign * options.translation_distance),
    )
    orthogonal = (
        translated_targets
        - (translated_targets * expected_axis).sum(dim=-1, keepdim=True) * expected_axis
    )
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1.0e-6)
    if direction == "push":
        assert torch.allclose(
            planned_targets[-1][:, :3, 3],
            -expected_axis.expand(NUM_ENVS, -1) * options.approach_distance,
        )


def test_slide_holds_failed_environment() -> None:
    affordance = SlideAffordance(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        translation_axis=torch.tensor([0.0, -1.0, 0.0]),
    )
    affordance.get_best_grasp_poses = Mock(
        return_value=(
            torch.tensor([True, False]),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.full((NUM_ENVS,), 0.03),
        )
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="drawer_handle",
    )
    generator = _motion_generator()

    def successful_ik(
        pose: torch.Tensor | None = None,
        name: str | None = None,
        joint_seed: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert joint_seed is not None
        return torch.ones(NUM_ENVS, dtype=torch.bool), torch.ones_like(joint_seed)

    generator.robot.compute_ik.side_effect = successful_ik
    action = _bind_action(generator, Slide())
    context = _context()

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="slide",
            goal=SlideGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=18),
            skill_options=SlideOptions(hand_interp_steps=3),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(18, -1),
    )


def test_slide_fk_path_remains_on_translation_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _motion_generator()

    def position_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = joint_seed.clone()
        qpos[:, :3] = pose[:, :3, 3]
        return torch.ones(NUM_ENVS, dtype=torch.bool), qpos

    def position_fk(
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        pose[:, :3, 3] = qpos[:, :3]
        return pose

    generator.robot.compute_ik.side_effect = position_ik
    generator.robot.compute_fk.side_effect = position_fk
    affordance = SlideAffordance(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        translation_axis=torch.tensor([0.0, -1.0, 0.0]),
    )
    monkeypatch.setattr(
        affordance,
        "get_best_grasp_poses",
        Mock(
            return_value=(
                torch.ones(NUM_ENVS, dtype=torch.bool),
                torch.eye(4).repeat(NUM_ENVS, 1, 1),
                torch.full((NUM_ENVS,), 0.03),
            )
        ),
    )
    semantics = ObjectSemantics(affordance=affordance, geometry={}, label="handle")
    action = _bind_action(generator, Slide())
    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="slide",
            goal=SlideGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=SlideOptions(direction="pull", hand_interp_steps=3),
        ),
        _context(),
    )

    pull_segment = plan.segment("pull")
    arm_path = plan.trajectory.positions[
        :, pull_segment.start : pull_segment.stop, :ARM_DOF
    ]
    fk_path = position_fk(arm_path.reshape(-1, ARM_DOF), "arm", True).reshape(
        NUM_ENVS, -1, 4, 4
    )
    positions = fk_path[:, :, :3, 3]
    axis = torch.tensor([0.0, -1.0, 0.0])
    orthogonal = positions - (positions * axis).sum(dim=-1, keepdim=True) * axis
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1.0e-6)


def test_press_plans_close_approach_press_and_retract() -> None:
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]),
        press_position=(0.0, 0.0, 0.0),
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="button",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Press())
    options = PressOptions(
        hand_interp_steps=3,
        approach_distance=0.1,
        press_distance=0.02,
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == [
        "close",
        "approach",
        "contact",
        "press",
        "retract",
    ]
    assert torch.all(
        plan.trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    contact_pose = affordance.get_press_pose(torch.eye(4).repeat(NUM_ENVS, 1, 1))
    expected_approach = (
        contact_pose[:, :3, 3] - contact_pose[:, :3, 2] * options.approach_distance
    )
    expected_pressed = (
        contact_pose[:, :3, 3] + contact_pose[:, :3, 2] * options.press_distance
    )
    planned_targets = [
        call.kwargs["pose"] for call in generator.robot.compute_ik.call_args_list
    ]
    motion_lengths = Press._motion_segment_lengths(24, options.hand_interp_steps)
    contact_stop = 1 + motion_lengths[1] - 1
    press_stop = contact_stop + motion_lengths[2] - 1
    assert torch.allclose(planned_targets[0][:, :3, 3], expected_approach)
    assert torch.allclose(
        planned_targets[contact_stop - 1][:, :3, 3], contact_pose[:, :3, 3]
    )
    assert torch.allclose(planned_targets[press_stop - 1][:, :3, 3], expected_pressed)
    assert torch.allclose(planned_targets[-1][:, :3, 3], expected_approach)


def test_press_plans_from_rigid_object_pose_snapshot_with_option_position() -> None:
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]),
        press_position=(0.5, 0.5, 0.5),
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="rigid-button",
    )
    generator = _motion_generator()

    plan = _plan_action(
        _bind_action(generator, Press()),
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=PressOptions(
                hand_interp_steps=3,
                press_position=(0.1, 0.2, 0.3),
            ),
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    planned_approach = generator.robot.compute_ik.call_args_list[0].kwargs["pose"]
    assert torch.allclose(
        planned_approach[:, :3, 3],
        torch.tensor([0.0, 0.2, 0.3]).expand(NUM_ENVS, -1),
    )


def test_press_fk_path_passes_contact_and_remains_on_press_axis() -> None:
    generator = _motion_generator()

    def position_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = joint_seed.clone()
        qpos[:, :3] = pose[:, :3, 3]
        return torch.ones(NUM_ENVS, dtype=torch.bool), qpos

    def position_fk(
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        pose[:, :3, 3] = qpos[:, :3]
        return pose

    generator.robot.compute_ik.side_effect = position_ik
    generator.robot.compute_fk.side_effect = position_fk
    semantics = ObjectSemantics(
        affordance=PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="button",
    )
    action = _bind_action(generator, Press())
    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=PressOptions(hand_interp_steps=3, press_distance=0.04),
        ),
        _context(),
    )

    contact_arm = plan.trajectory.positions[
        :, plan.segment("contact").stop - 1, :ARM_DOF
    ]
    contact_fk = position_fk(contact_arm, "arm", True)
    assert torch.allclose(contact_fk[:, :3, 3], torch.zeros(NUM_ENVS, 3))
    press_segment = plan.segment("press")
    press_arm = plan.trajectory.positions[
        :, press_segment.start : press_segment.stop, :ARM_DOF
    ]
    press_fk = position_fk(press_arm.reshape(-1, ARM_DOF), "arm", True).reshape(
        NUM_ENVS, -1, 4, 4
    )
    positions = press_fk[:, :, :3, 3]
    axis = torch.tensor([1.0, 0.0, 0.0])
    orthogonal = positions - (positions * axis).sum(dim=-1, keepdim=True) * axis
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1.0e-6)
    assert torch.allclose(positions[:, -1], torch.tensor([0.04, 0.0, 0.0]))


def test_press_preserves_failed_environment_at_observed_qpos() -> None:
    semantics = ObjectSemantics(
        affordance=PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="button",
    )
    generator = _motion_generator()

    def partial_ik(
        pose: torch.Tensor | None = None,
        name: str | None = None,
        joint_seed: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert joint_seed is not None
        return torch.tensor([True, False]), torch.ones_like(joint_seed)

    generator.robot.compute_ik.side_effect = partial_ik
    action = _bind_action(generator, Press())
    context = _context()

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(),
            motion_policy=MotionPolicy(sample_count=18),
            skill_options=PressOptions(hand_interp_steps=3),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(plan.trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        plan.trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(18, -1),
    )


def test_press_rejects_non_press_affordance() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            mesh_vertices=torch.zeros(8, 3),
            mesh_triangles=torch.zeros(4, 3, dtype=torch.long),
        ),
        geometry={},
        label="mesh-button",
    )
    action = _bind_action(_motion_generator(), Press())

    with pytest.raises(ValueError, match="PressAffordance"):
        _plan_action(
            action,
            _invocation("press", PressGoal(semantics, torch.eye(4))),
            _context(),
        )


def test_press_requires_primary_arm_and_end_effector_bindings() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="button",
    )
    action = _bind_action(_motion_generator(), Press())
    invocation = ActionInvocation(
        skill_id="press",
        goal=PressGoal(semantics, torch.eye(4)),
        binding=ActionBinding(manipulators={"primary": "arm"}),
    )

    with pytest.raises(KeyError, match="No end effector is bound to role 'primary'"):
        action.resolve_request(invocation)


def test_press_axis_belongs_to_affordance_not_action_options() -> None:
    assert "press_axis" not in PressOptions.__dataclass_fields__


@pytest.mark.parametrize(
    "press_position",
    ((0.0, 1.0), (0.0, 1.0, float("nan"))),
)
def test_press_options_reject_invalid_press_position(
    press_position: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="press_position"):
        PressOptions(press_position=press_position)  # type: ignore[arg-type]


def test_twist_rejects_non_twist_affordance() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            mesh_vertices=torch.zeros(8, 3),
            mesh_triangles=torch.zeros(4, 3, dtype=torch.long),
        ),
        geometry={},
        label="mesh-knob",
    )
    action = _bind_action(_motion_generator(), Twist())

    with pytest.raises(ValueError, match="TwistAffordance"):
        _plan_action(
            action,
            _invocation("twist", TwistGoal(semantics, torch.eye(4))),
            _context(),
        )


def test_twist_axis_belongs_to_affordance_not_action_options() -> None:
    assert "twist_axis" not in TwistOptions.__dataclass_fields__
    assert "approach_direction" not in TwistOptions.__dataclass_fields__


def test_twist_options_reject_non_finite_pre_grasp_distance() -> None:
    with pytest.raises(ValueError, match="pre_grasp_distance must be finite"):
        TwistOptions(pre_grasp_distance=float("nan"))


def test_slide_rejects_non_slide_affordance() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            mesh_vertices=torch.zeros(8, 3),
            mesh_triangles=torch.zeros(4, 3, dtype=torch.long),
        ),
        geometry={},
        label="mesh-handle",
    )
    action = _bind_action(_motion_generator(), Slide())

    with pytest.raises(ValueError, match="SlideAffordance"):
        _plan_action(
            action,
            _invocation(
                "slide",
                SlideGoal(semantics, torch.eye(4)),
            ),
            _context(),
        )


def test_slide_requires_primary_end_effector() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="drawer_handle",
    )
    action = _bind_action(_motion_generator(), Slide())
    invocation = ActionInvocation(
        skill_id="slide",
        goal=SlideGoal(semantics, torch.eye(4)),
        binding=ActionBinding(manipulators={"primary": "arm"}),
    )

    with pytest.raises(KeyError, match="No end effector is bound to role 'primary'"):
        action.resolve_request(invocation)


def test_slide_axis_belongs_to_affordance_not_action_options() -> None:
    assert "translation_axis" not in SlideOptions.__dataclass_fields__


def test_slide_options_reject_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        SlideOptions(direction="open")  # type: ignore[arg-type]


def test_handover_does_not_mutate_cached_final_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        binding=_dual_binding("source", "destination"),
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
    semantics = _semantics(entity_id="handover_object")
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

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
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
        binding=_dual_binding("source", "destination"),
    )

    with pytest.raises(ValueError, match="must identify the object held"):
        _plan_action(action, invocation, _dual_context(task))

    held_semantics.entity.get_local_pose.assert_not_called()
    goal_semantics.entity.get_local_pose.assert_not_called()


def test_handover_transfers_only_exclusively_held_rows() -> None:
    generator = _dual_motion_generator()
    semantics = _semantics()
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "left_arm": _held(semantics),
            "right_arm": _held(
                semantics,
                env_mask=torch.tensor([True, False]),
            ),
        },
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

    assert plan.plan_success.tolist() == [False, True]
    assert torch.allclose(
        plan.trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(30, -1),
    )
    transferred = projected.get_held_object("left_arm")
    received = projected.get_held_object("right_arm")
    assert transferred is not None and transferred.env_mask.tolist() == [True, False]
    assert received is not None and received.env_mask.tolist() == [True, True]
    assert received.semantics is semantics


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
        binding=_dual_binding("left", "right"),
        motion_policy=MotionPolicy(sample_count=30),
    )
    context = _dual_context()

    request = action.resolve_request(invocation)
    plan = action.plan(request, context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True, True]
    assert plan.trajectory.positions.shape == (NUM_ENVS, 30, DUAL_ROBOT_DOF)
    left_held = projected.get_held_object("left_arm")
    right_held = projected.get_held_object("right_arm")
    assert isinstance(left_held, HeldObjectState)
    assert isinstance(right_held, HeldObjectState)
    assert left_held.semantics is right_held.semantics
    assert left_held.semantics is not semantics
    assert plan.scene_dependencies == ()
    request.goal.semantics.entity.get_local_pose.assert_not_called()
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "close",
        "lift",
        "move",
        "hold",
    ]


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
        binding=_dual_binding("left", "right"),
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
    left_held = projected.get_held_object("left_arm")
    right_held = projected.get_held_object("right_arm")
    assert left_held is not None and right_held is not None
    assert left_held.semantics is right_held.semantics
    assert torch.allclose(left_held.object_to_eef, pose_inv(object_pose))
    assert torch.allclose(right_held.object_to_eef, pose_inv(object_pose))


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
        held_objects={"arm": _held(_semantics(entity_id="assemble_object"))},
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
        _invocation(
            "place",
            AssembleGoal(
                affordance=affordance,
                base_pose=SceneEntityPose("base"),
            ),
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
        _invocation("place", AssembleGoal(affordance=affordance))
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
    left_held = projected.get_held_object("left_arm")
    right_held = projected.get_held_object("right_arm")
    assert left_held is not None and right_held is not None
    assert left_held.semantics is right_held.semantics
    assert left_held.env_mask.tolist() == [True, False]
    assert right_held.env_mask.tolist() == [True, False]


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
        binding=_dual_binding("placing", "support"),
        motion_policy=MotionPolicy(sample_count=30),
    )

    plan = _plan_action(action, invocation, _dual_context(task))

    assert plan.plan_success.tolist() == [False, False]
    assert plan.trajectory.waypoint_count == 0
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
