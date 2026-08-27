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
from dataclasses import replace
from typing import Literal, TypeVar
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
    AxisAlign,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
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
    HandOverGoal,
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
    OpenDoor,
    OpenDoorAffordance,
    OpenDoorGoal,
    OpenDoorOptions,
    PickUp,
    PickUpOptions,
    Place,
    PlaceGoal,
    PlaceOptions,
    Pour,
    PourGoal,
    PourOptions,
    PlanningContext,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    PushObject,
    PushObjectGoal,
    PushObjectOptions,
    PushObjectToolCalibration,
    SlideAffordance,
    Slide,
    SlideGoal,
    SlideOptions,
    RobotObservation,
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
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.lab.sim.planners import (
    MotionGenerator,
    MoveType,
    PlanOptions,
    PlanResult,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
ROBOT_DOF = ARM_DOF + HAND_DOF
CONTROL_DT = 1.0 / 60.0
DUAL_ARM_DOF = 2 * ARM_DOF
DUAL_ROBOT_DOF = DUAL_ARM_DOF + 2 * HAND_DOF
DOOR_ENTITY_ID = "door"

ActionT = TypeVar("ActionT", bound=AtomicAction)
_ACTION_ENGINES: dict[int, AtomicActionEngine] = {}
_GRASP_GENERATORS: dict[int, _StubGraspPoseGenerator] = {}


class _StubGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Deterministic planning-service double used by atomic-action tests."""

    def __init__(self) -> None:
        super().__init__(ParallelJawGripperModelCfg(model_id="test_parallel_jaw"))

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del (
            mesh_vertices,
            mesh_triangles,
            approach_direction,
            obj_longest_axis,
            is_positive_part,
        )
        return [
            (
                torch.eye(4, dtype=torch.float32, device=obj_poses.device).unsqueeze(0),
                torch.zeros(1, dtype=torch.float32, device=obj_poses.device),
            )
            for _ in range(obj_poses.shape[0])
        ]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mesh_vertices, mesh_triangles, approach_direction
        return (
            torch.ones(obj_poses.shape[0], dtype=torch.bool, device=obj_poses.device),
            torch.eye(4, dtype=torch.float32, device=obj_poses.device).repeat(
                obj_poses.shape[0], 1, 1
            ),
            torch.zeros(obj_poses.shape[0], device=obj_poses.device),
        )

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
    ) -> list[dict[str, dict[str, object]] | None]:
        del (
            mesh_vertices,
            mesh_triangles,
            left_to_right_arm_direction,
            approach_direction,
            middle_empty_ratio,
        )
        arm = {
            "is_success": True,
            "grasp_poses": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "open_lengths": torch.tensor([0.0], dtype=torch.float32),
            "total_cost": torch.tensor([0.0], dtype=torch.float32),
        }
        return [{"left": arm, "right": arm} for _ in range(obj_poses.shape[0])]


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
    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives._helpers.resample_with_distance",
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

    def compute_batch_fk(
        qpos: torch.Tensor,
        name: str | None = None,
        to_matrix: bool = True,
    ) -> torch.Tensor:
        return (
            torch.eye(4).reshape(1, 1, 4, 4).repeat(qpos.shape[0], qpos.shape[1], 1, 1)
        )

    robot.get_qpos.side_effect = get_qpos
    robot.get_joint_ids.side_effect = get_joint_ids
    robot.compute_ik.side_effect = compute_ik
    robot.compute_fk.side_effect = compute_fk
    robot.compute_batch_fk.side_effect = compute_batch_fk
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
    grasp_generator = _StubGraspPoseGenerator()
    engine = AtomicActionEngine(
        generator,
        control_profiles=profiles,
        grasp_pose_generators={
            name: grasp_generator
            for name in generator.robot.control_parts
            if "hand" in name
        },
        load_builtins=False,
    )
    engine.register(action)
    _ACTION_ENGINES[id(action)] = engine
    _GRASP_GENERATORS[id(action)] = grasp_generator
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


def _door_scene(
    pose: torch.Tensor | None = None,
    *,
    hinge_position: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> SceneSnapshot:
    """Build a handle pose and live parent-hinge observation."""
    if pose is None:
        pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    if hinge_position is None:
        hinge_position = torch.zeros(NUM_ENVS, 1)
    return SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"target": EntityState(pose)},
        articulation_joints={
            (DOOR_ENTITY_ID, "door_hinge"): ObservedArticulationJointState(
                hinge_position,
                valid_mask,
            )
        },
    )


def _binding(
    action: AtomicAction,
    *,
    motion: str = "arm",
    grasp: str = "hand",
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


def _semantics(*, entity_id: str = "test_object") -> ObjectSemantics:
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="test_object",
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
    robot.cfg = Mock()
    robot.cfg.solver_cfg = {
        "left_arm": Mock(root_link_name="left_root"),
        "right_arm": Mock(root_link_name="right_root"),
    }

    def get_link_pose(
        link_name: str,
        env_ids: list[int] | None = None,
        to_matrix: bool = True,
    ) -> torch.Tensor:
        del env_ids, to_matrix
        pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
        pose[:, 0, 3] = -1.0 if link_name == "left_root" else 1.0
        return pose

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
    robot.get_link_pose.side_effect = get_link_pose

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
    )


def _stub_dual_arm_grasp_poses(action: AtomicAction) -> None:
    """Stub the engine's dual-arm grasp service with identity grasp poses."""

    def _sample(obj_poses: torch.Tensor, **_kwargs: object) -> list[dict]:
        arm = {
            "is_success": True,
            "grasp_poses": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "open_lengths": torch.tensor([0.0], dtype=torch.float32),
            "total_cost": torch.tensor([0.0], dtype=torch.float32),
        }
        return [{"left": arm, "right": arm} for _ in range(obj_poses.shape[0])]

    _GRASP_GENERATORS[id(action)].get_dual_arm_valid_grasp_poses = Mock(
        side_effect=_sample
    )


def _plan_segment_contract_case(case_id: str) -> ActionPlan:
    """Plan one built-in used by the trajectory-segment contract."""
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
        goal = PressGoal(
            ObjectSemantics(
                affordance=PressAffordance(
                    press_axis=torch.tensor([1.0, 0.0, 0.0]),
                    press_position=(0.0, 0.0, 0.0),
                ),
                geometry={},
                label="button",
                entity_id="button",
            ),
            torch.eye(4),
        )
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(),
        )

    if case_id == "push_object":
        action = _bind_action(generator, PushObject())
        object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
        target_pose = object_pose.clone()
        target_pose[:, 1, 3] = 0.1
        scene = SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={
                "object": EntityState(object_pose),
                "target": EntityState(target_pose),
            },
        )
        goal = PushObjectGoal(
            _semantics(entity_id="object"),
            SceneEntityPose("target"),
        )
        return _plan_action(
            action,
            _invocation(action, goal, sample_count=sample_count),
            _context(scene=scene),
        )

    raise AssertionError(f"Unknown trajectory-segment contract case {case_id!r}.")


def test_builtin_descriptors_expose_strict_goal_types() -> None:
    assert MoveEndEffector.GoalType is EndEffectorPoseGoal
    assert MoveJoints.GoalType is JointPositionGoal
    assert PickUp.GoalType is GraspGoal
    assert AxisAlign.GoalType is AxisAlignGoal
    assert MoveHeldObject.GoalType is HeldObjectPoseGoal
    assert Place.GoalType == (PlaceGoal, AssembleGoal)
    assert Pour.GoalType is PourGoal
    assert Press.GoalType is PressGoal
    assert PushObject.GoalType is PushObjectGoal
    assert Slide.GoalType is SlideGoal
    assert OpenDoor.GoalType is OpenDoorGoal
    assert Twist.GoalType is TwistGoal
    assert CoordinatedPickment.GoalType is CoordinatedPickGoal
    assert CoordinatedPlacement.GoalType is CoordinatedPlacementGoal
    assert HandOver.GoalType is HandOverGoal


@pytest.mark.parametrize(
    ("case_id", "expected_names"),
    (
        ("move_joints", ("move_joints",)),
        ("move_end_effector", ("move_end_effector",)),
        ("move_held_object", ("transport",)),
        ("place", ("approach", "release", "retract")),
        ("assemble", ("approach", "release", "retract")),
        ("press", ("close", "approach", "contact", "press", "retract")),
        ("push_object", ("close", "approach", "contact", "push", "retract")),
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
    assert (Press.skill_id, Slide.skill_id, OpenDoor.skill_id, Twist.skill_id) == (
        "press",
        "slide",
        "open_door",
        "twist",
    )


@pytest.mark.parametrize(
    "options",
    (
        PickUpOptions(),
        AxisAlignOptions(),
        MoveHeldObjectOptions(),
        PlaceOptions(),
        PourOptions(),
        PressOptions(),
        PushObjectOptions(),
        SlideOptions(),
        OpenDoorOptions(),
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


def test_pose_options_own_late_bound_relative_transforms() -> None:
    relative_pose = torch.eye(4)
    target = SceneEntityPose("target", relative_pose=relative_pose)
    pick_options = PickUpOptions(downstream_object_target_poses=(target,))

    assert target.relative_pose is not None
    target.relative_pose[0, 3] = 9.0

    pick_target = pick_options.downstream_object_target_poses[0]
    assert type(pick_target) is SceneEntityPose
    assert pick_target.relative_pose is not None
    assert pick_target.relative_pose[0, 3].item() == 0.0


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
            binding=_binding(pick),
        ),
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
        ActionInvocation(
            skill_id="place",
            goal=PlaceGoal(torch.eye(4)),
            binding=_binding(place),
        ),
        picked_context,
    )
    placed_task = place_plan.expected_effects.apply(
        picked_task, place_plan.plan_success
    )

    assert picked_task.get_held_object("arm") is not None
    assert placed_task.get_held_object("arm") is None


def test_place_holds_fully_open_before_retracting_when_configured() -> None:
    generator = _motion_generator()
    generator.generate = Mock(wraps=generator.generate)
    action = _bind_action(generator, Place())
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": _held()},
    )
    sample_count = 20
    settle_steps = 3
    invocation = ActionInvocation(
        skill_id=action.skill_id,
        goal=PlaceGoal(torch.eye(4)),
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=sample_count),
        skill_options=PlaceOptions(release_settle_steps=settle_steps),
    )

    plan = _plan_action(action, invocation, _context(task))

    release = plan.segment("release")
    assert plan.commands.frame_count == sample_count + settle_steps
    assert release.stop - release.start == 5 + settle_steps
    hand_positions = _joint_command_positions(plan, "hand")
    torch.testing.assert_close(
        hand_positions[:, release.stop - settle_steps : release.stop],
        hand_positions[:, release.stop - 1 : release.stop].expand(
            -1,
            settle_steps,
            -1,
        ),
    )
    generator.generate.assert_called_once()
    target_states = generator.generate.call_args.args[0]
    assert len(target_states) == 3
    assert [state.xpos[0, 2, 3].item() for state in target_states] == pytest.approx(
        [0.1, 0.0, 0.1]
    )
    assert generator.generate.call_args.kwargs["options"].sample_count == (
        sample_count - PlaceOptions().hand_interp_steps
    )


def test_pick_holds_fully_closed_before_lifting_when_configured() -> None:
    generator = _motion_generator()
    generator.generate = Mock(wraps=generator.generate)
    action = _bind_action(generator, PickUp())
    sample_count = 20
    settle_steps = 3
    invocation = ActionInvocation(
        skill_id=action.skill_id,
        goal=GraspGoal(
            semantics=_semantics(entity_id="target"),
            grasp_xpos=torch.eye(4),
        ),
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=sample_count),
        skill_options=PickUpOptions(grasp_settle_steps=settle_steps),
    )

    plan = _plan_action(
        action,
        invocation,
        _context(
            scene=_target_scene(
                torch.eye(4).repeat(NUM_ENVS, 1, 1),
                timestamp=0.0,
                version=0,
            )
        ),
    )

    close = plan.segment("close")
    assert plan.commands.frame_count == sample_count + settle_steps
    assert close.stop - close.start == 5 + settle_steps
    hand_positions = _joint_command_positions(plan, "hand")
    torch.testing.assert_close(
        hand_positions[:, close.stop - settle_steps : close.stop],
        hand_positions[:, close.stop - 1 : close.stop].expand(
            -1,
            settle_steps,
            -1,
        ),
    )
    generator.generate.assert_called_once()
    target_states = generator.generate.call_args.args[0]
    assert len(target_states) == 3
    assert [state.xpos[0, 2, 3].item() for state in target_states] == pytest.approx(
        [0.15, 0.0, 0.1]
    )
    assert generator.generate.call_args.kwargs["options"].sample_count == (
        sample_count - PickUpOptions().hand_interp_steps
    )


def test_pick_combined_motion_gen_preserves_backend_samples_before_split() -> None:
    sample_count = 20
    backend_sample_count = 6
    generator = _motion_generator()
    generator.generate = Mock(
        return_value=PlanResult(
            success=torch.ones(NUM_ENVS, dtype=torch.bool),
            positions=torch.zeros(NUM_ENVS, backend_sample_count, ARM_DOF),
            dt=torch.zeros(NUM_ENVS, backend_sample_count),
        )
    )
    action = _bind_action(generator, PickUp())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="pick_up",
            goal=GraspGoal(
                semantics=_semantics(entity_id="target"),
                grasp_xpos=torch.eye(4),
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=sample_count,
            ),
        ),
        _context(scene=_target_scene(object_pose, timestamp=0.0, version=0)),
    )

    generator.generate.assert_called_once()
    assert generator.generate.call_args.kwargs["options"].sample_count is None
    assert plan.commands.frame_count == sample_count


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


def test_move_held_object_uses_exact_projected_attachment_target() -> None:
    generator = _motion_generator()
    generator.generate = Mock(wraps=generator.generate)
    action = _bind_action(generator, MoveHeldObject())
    invocation = _invocation(
        action,
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
    invocation = ActionInvocation(
        skill_id="move_held_object",
        goal=HeldObjectPoseGoal(torch.eye(4)),
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=10),
        skill_options=MoveHeldObjectOptions(),
    )

    plan = _plan_action(action, invocation, _context(task))

    assert plan.plan_success.all()
    assert plan.expected_effects.is_empty
    target_states = generator.generate.call_args.args[0]
    assert len(target_states) == 1
    assert torch.allclose(target_states[0].xpos, held.object_to_eef)


def test_pour_rotates_held_object_about_internal_axis_and_returns() -> None:
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
    current_eef_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    current_eef_pose[:, 0, 3] = torch.tensor([0.4, 0.7])
    generator.robot.compute_fk.return_value = current_eef_pose
    generator.robot.compute_fk.side_effect = None
    action = _bind_action(generator, Pour())
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0])),
        geometry={},
        label="pourable-object",
        entity_id="pourable-object",
    )
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": _held(semantics)},
    )
    context = _context(task)

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="pour",
            goal=PourGoal(),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=10),
            skill_options=PourOptions(rotate_angle=math.pi / 2.0),
        ),
        context,
    )

    expected_rotation = axis_angle_to_rotation_matrix(
        torch.tensor([math.pi / 2.0, 0.0, 0.0])
    )
    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    assert trajectory.positions.shape == (NUM_ENVS, 10, ROBOT_DOF)
    assert trajectory.duration.tolist() == pytest.approx([9.0 / 60.0] * NUM_ENVS)
    assert [segment.name for segment in plan.segments] == ["pour"]
    assert torch.allclose(
        solved_poses[0][:, :3, :3],
        expected_rotation.expand(NUM_ENVS, -1, -1),
        atol=1.0e-6,
    )
    assert torch.allclose(
        solved_poses[0][:, :3, 3],
        current_eef_pose[:, :3, 3],
    )
    assert len(solved_poses) == 2
    assert torch.allclose(solved_poses[1], current_eef_pose, atol=1.0e-6)
    assert torch.all(trajectory.positions[:, :, ARM_DOF:] == 1.0)
    assert plan.expected_effects.is_empty
    assert context.task is task


def test_pour_reads_held_state_from_the_bound_logical_resource() -> None:
    """Profile resource IDs need not match native motion control-part names."""
    generator = _motion_generator()
    generator.robot.compute_fk.return_value = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    generator.robot.compute_fk.side_effect = None
    action = _bind_action(generator, Pour())
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0])),
        geometry={},
        label="pourable-object",
        entity_id="pourable-object",
    )
    logical_resource = "right_manipulator"
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={logical_resource: _held(semantics)},
    )
    direct_binding = _binding(action)
    binding = ActionBinding(
        owner_id=direct_binding.owner_id,
        endpoints=tuple(
            replace(endpoint, task_state_key=logical_resource)
            for endpoint in direct_binding.endpoints
        ),
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="pour",
            goal=PourGoal(),
            binding=binding,
            motion_policy=MotionPolicy(sample_count=10),
        ),
        _context(task),
    )

    assert plan.plan_success.all()


def test_engine_compiles_pickup_followed_by_pour() -> None:
    generator = _motion_generator()
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(HAND_DOF),
                grasp=torch.ones(HAND_DOF),
            )
        },
    )
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0])),
        geometry={},
        label="pourable-object",
        entity_id="target",
    )
    context = _context(
        scene=_target_scene(
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            timestamp=0.0,
            version=0,
        )
    )

    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics, grasp_xpos=torch.eye(4)),
                control_parts={
                    "primary": {"motion": "arm", "grasp": "hand"},
                },
                motion_policy=MotionPolicy(sample_count=20),
            ),
            engine.make_invocation(
                "pour",
                PourGoal(),
                control_parts={
                    "primary": {"motion": "arm", "grasp": "hand"},
                },
                motion_policy=MotionPolicy(sample_count=10),
                skill_options=PourOptions(rotate_angle=math.pi / 2.0),
            ),
        ),
        context,
    )

    assert compiled.plan_success.tolist() == [True, True]
    assert [plan.skill_id for plan in compiled.action_plans] == ["pick_up", "pour"]
    assert compiled.projected_context.get_held_object("arm") is not None


def test_pour_requires_exclusively_held_axis_align_affordance() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, Pour())
    invocation = ActionInvocation(
        skill_id="pour",
        goal=PourGoal(),
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=10),
    )

    with pytest.raises(ValueError, match="run PickUp first"):
        _plan_action(action, invocation, _context())

    invalid_task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"arm": _held(_semantics())},
    )
    with pytest.raises(ValueError, match="AxisAlignAffordance"):
        _plan_action(action, invocation, _context(invalid_task))

    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(),
        geometry={},
        label="shared-pourable-object",
        entity_id="shared-pourable-object",
    )
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
    plan = _plan_action(action, invocation, _context(task))

    assert plan.plan_success.tolist() == [False, True]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[0],
        _context(task).robot.qpos[0].unsqueeze(0).expand(trajectory.waypoint_count, -1),
    )


def test_pour_options_only_contain_rotate_angle_and_require_finite_value() -> None:
    assert set(PourOptions.__dataclass_fields__) == {"rotate_angle"}
    assert PourOptions().rotate_angle == pytest.approx(math.pi / 4.0)
    with pytest.raises(ValueError, match="rotate_angle must be finite"):
        PourOptions(rotate_angle=float("nan"))


def test_move_held_object_moves_only_exclusively_held_rows() -> None:
    generator = _motion_generator()

    def move_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pose, name
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
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="explicit-grasp-object",
        entity_id="target",
    )
    grasp = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp[:, 0, 3] = torch.tensor([0.1, 0.2])
    action = _bind_action(generator, PickUp())
    grasp_generator = _GRASP_GENERATORS[id(action)]
    grasp_generator.get_valid_grasp_poses = Mock()
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

    grasp_generator.get_valid_grasp_poses.assert_not_called()
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


def test_pick_fixed_object_to_eef_bypasses_sampling_and_adjustments() -> None:
    """A calibrated object-relative grasp is used directly and owned safely."""
    generator = _motion_generator()
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="fixed-grasp-object",
        entity_id="target",
    )
    action = _bind_action(generator, PickUp())
    grasp_generator = _GRASP_GENERATORS[id(action)]
    grasp_generator.get_valid_grasp_poses = Mock()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([0.03, 0.07])
    fixed_object_to_eef = torch.tensor(
        (
            (0.0, 1.0, 0.0, 0.04),
            (-1.0, 0.0, 0.0, 0.02),
            (0.0, 0.0, 1.0, 0.03),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    options = PickUpOptions(
        fixed_object_to_eef=fixed_object_to_eef,
        rotate_upright=math.pi / 2.0,
        grasp_frame_to_eef=torch.tensor(
            (
                (0.0, -1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.05),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    owned_transform = options.fixed_object_to_eef.clone()
    fixed_object_to_eef.zero_()
    context = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id=action.skill_id,
            goal=GraspGoal(semantics=semantics),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=20),
            skill_options=options,
        ),
        context,
    )
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    grasp_generator.get_valid_grasp_poses.assert_not_called()
    held = projected.get_held_object("arm")
    assert held is not None
    expected_grasp = torch.matmul(object_pose, owned_transform)
    torch.testing.assert_close(held.grasp_xpos, expected_grasp)
    torch.testing.assert_close(
        held.object_to_eef,
        owned_transform.expand(NUM_ENVS, -1, -1),
    )


def test_axis_align_plans_two_arm_phases_and_aligns_the_object_axis() -> None:
    generator = _motion_generator()
    generator.generate = Mock(wraps=generator.generate)
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
    action = _bind_action(generator, AxisAlign())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([0.0, 0.0, 1.0])),
        geometry={},
        label="axis-object",
        entity_id="target",
    )
    context = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))
    original_task = context.task

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="axis_align",
            goal=AxisAlignGoal(semantics=semantics, grasp_xpos=torch.eye(4)),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=20),
            skill_options=AxisAlignOptions(
                target_axis=torch.tensor([1.0, 0.0, 0.0]),
                lift_height=0.1,
            ),
        ),
        context,
    )

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    assert trajectory.positions.shape == (NUM_ENVS, 20, ROBOT_DOF)
    assert torch.equal(trajectory.env_ids, context.env_ids)
    assert trajectory.duration.tolist() == pytest.approx([19.0 / 60.0] * NUM_ENVS)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "close",
        "manipulate",
    ]
    assert generator.generate.call_count == 2
    assert len(solved_poses) == 4
    assert plan.expected_effects.is_empty
    assert context.task is original_task
    assert plan.scene_dependencies == ("target",)
    final_object_rotation = solved_poses[-1][:, :3, :3]
    final_world_axis = torch.matmul(
        final_object_rotation,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert torch.allclose(
        final_world_axis,
        torch.tensor([1.0, 0.0, 0.0]).expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )
    assert solved_poses[-1][:, 2, 3].tolist() == pytest.approx([0.1, 0.1])
    assert torch.all(trajectory.positions[:, -1, ARM_DOF:] == 1.0)


def test_axis_align_upright_prefers_perpendicular_grasp_and_pre_rotates() -> None:
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
    action = _bind_action(generator, AxisAlign())
    parallel_grasp = torch.eye(4)
    perpendicular_grasp = torch.eye(4)
    perpendicular_grasp[:3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    candidates = torch.stack([parallel_grasp, perpendicular_grasp])
    affordance = AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0]))
    _GRASP_GENERATORS[id(action)].get_valid_grasp_poses = Mock(
        return_value=[
            (candidates.clone(), torch.tensor([0.0, 10.0])) for _ in range(NUM_ENVS)
        ]
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="upright-axis-object",
        entity_id="target",
    )
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="axis_align",
            goal=AxisAlignGoal(semantics=semantics),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=20),
            skill_options=AxisAlignOptions(
                target_axis=torch.tensor([0.0, 0.0, 1.0]),
            ),
        ),
        _context(scene=_target_scene(object_pose, timestamp=0.0, version=0)),
    )

    assert plan.plan_success.all()
    expected_rotation = (
        axis_angle_to_rotation_matrix(torch.tensor([0.0, math.pi / 4.0, 0.0]))
        @ perpendicular_grasp[:3, :3]
    )
    assert torch.allclose(
        solved_poses[1][:, :3, :3],
        expected_rotation.expand(NUM_ENVS, -1, -1),
        atol=1.0e-6,
    )


def test_axis_align_holds_only_failed_environment_rows() -> None:
    generator = _motion_generator()

    def compute_ik(
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([True, False]), joint_seed + 0.1

    generator.robot.compute_ik.side_effect = compute_ik
    action = _bind_action(generator, AxisAlign())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(),
        geometry={},
        label="partially-alignable-object",
        entity_id="target",
    )
    context = _context(scene=_target_scene(object_pose, timestamp=0.0, version=0))

    plan = _plan_action(
        action,
        _invocation(
            action,
            AxisAlignGoal(semantics=semantics, grasp_xpos=torch.eye(4)),
            sample_count=20,
        ),
        context,
    )

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, False]
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(trajectory.waypoint_count, -1),
    )


def test_axis_align_validates_goal_and_binding_contract() -> None:
    action = _bind_action(_motion_generator(), AxisAlign())
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(),
        geometry={},
        label="axis-object",
        entity_id="axis-object",
    )

    with pytest.raises(TypeError, match="expects goal AxisAlignGoal"):
        action.resolve_request(
            ActionInvocation(
                skill_id="axis_align",
                goal=object(),
                binding=_binding(action),
            )
        )
    with pytest.raises(ValueError, match="missing=.*grasp"):
        action.resolve_request(
            ActionInvocation(
                skill_id="axis_align",
                goal=AxisAlignGoal(semantics),
                binding=ActionBinding(
                    owner_id=_ACTION_ENGINES[id(action)].binding_owner_id,
                ),
            )
        )


def test_axis_align_handles_opposite_axes_without_nan() -> None:
    action = _bind_action(_motion_generator(), AxisAlign())
    identity = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    eef_keyframes = action._axis_alignment_eef_keyframes(
        identity,
        identity,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.0]),
        waypoint_count=3,
    )

    final_axis = torch.matmul(
        eef_keyframes[:, -1, :3, :3], torch.tensor([1.0, 0.0, 0.0])
    )
    assert torch.isfinite(eef_keyframes).all()
    assert torch.allclose(
        final_axis,
        torch.tensor([-1.0, 0.0, 0.0]).expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )


def test_pick_holds_only_environment_without_a_feasible_grasp() -> None:
    generator = _motion_generator()
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="partially-graspable-object",
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


def test_pick_normalizes_integer_batch_ik_success_masks() -> None:
    """DexSim integer IK flags become boolean masks before tensor selection."""
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    poses = torch.eye(4).repeat(NUM_ENVS, 3, 2, 1, 1)
    joint_seed = torch.zeros(NUM_ENVS, ARM_DOF)
    manipulator = JointPositionTarget("arm", tuple(range(ARM_DOF)))

    def compute_batch_ik(
        *,
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "arm"
        return (
            torch.ones(pose.shape[:2], dtype=torch.int32),
            joint_seed.clone(),
        )

    generator.robot.compute_batch_ik.side_effect = compute_batch_ik

    success, qpos = action._compute_batch_candidate_ik(
        poses,
        joint_seed,
        manipulator,
    )

    assert success.dtype is torch.bool
    assert success.shape == (NUM_ENVS, 3, 2)
    assert success.all()
    assert qpos.shape == (NUM_ENVS, 3, 2, ARM_DOF)


def test_pick_maps_canonical_grasp_frames_to_the_robot_eef() -> None:
    """Endpoint calibration is applied after canonical grasp generation."""
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    grasp_xpos = torch.eye(4).repeat(NUM_ENVS, 1, 1, 1)
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    start_qpos = torch.zeros(NUM_ENVS, ARM_DOF)
    manipulator = JointPositionTarget("arm", tuple(range(ARM_DOF)))
    grasp_frame_to_eef = torch.tensor(
        (
            (0.0, 1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    def feasible_ik(
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_seed, manipulator
        return (
            torch.ones(poses.shape[:3], dtype=torch.bool),
            torch.zeros(*poses.shape[:3], ARM_DOF),
        )

    action._compute_batch_candidate_ik = Mock(side_effect=feasible_ik)

    selected, success = action._select_feasible_grasp_variants(
        grasp_xpos,
        start_qpos,
        object_pose,
        manipulator,
        PickUpOptions(grasp_frame_to_eef=grasp_frame_to_eef),
        torch.tensor((0.0, 0.0, -1.0)),
    )

    assert success.all()
    assert torch.allclose(
        selected,
        grasp_frame_to_eef.expand(NUM_ENVS, 1, -1, -1),
    )


def test_pick_candidate_pregrasp_uses_configured_world_approach_direction() -> None:
    """Candidate screening and trajectory generation share one approach frame."""
    generator = _motion_generator()
    action = _bind_action(generator, PickUp())
    grasp_xpos = torch.eye(4).repeat(NUM_ENVS, 1, 1, 1)
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    start_qpos = torch.zeros(NUM_ENVS, ARM_DOF)
    manipulator = JointPositionTarget("arm", tuple(range(ARM_DOF)))
    grasp_frame_to_eef = torch.tensor(
        (
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    def feasible_ik(
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_seed, manipulator
        return (
            torch.ones(poses.shape[:3], dtype=torch.bool),
            torch.zeros(*poses.shape[:3], ARM_DOF),
        )

    action._compute_batch_candidate_ik = Mock(side_effect=feasible_ik)
    options = PickUpOptions(
        pre_grasp_distance=0.15,
        grasp_frame_to_eef=grasp_frame_to_eef,
    )
    approach_direction = torch.tensor((0.0, 0.0, -1.0))

    action._select_feasible_grasp_variants(
        grasp_xpos,
        start_qpos,
        object_pose,
        manipulator,
        options,
        approach_direction,
    )

    pre_grasp_poses = action._compute_batch_candidate_ik.call_args_list[0].args[0]
    torch.testing.assert_close(
        pre_grasp_poses[..., :3, 3],
        torch.tensor((0.0, 0.0, 0.15)).expand(NUM_ENVS, 1, 2, -1),
    )


def test_pick_resolves_late_bound_scene_grasp_and_declares_dependency() -> None:
    generator = _motion_generator()
    target_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    target_pose[:, 0, 3] = torch.tensor([0.1, 0.2])
    relative_pose = torch.eye(4)
    relative_pose[2, 3] = 0.05
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="late-bound-grasp-object",
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
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="moving-grasp-object",
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
            control_dt=CONTROL_DT,
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


def test_pick_uses_selected_control_part_for_state_and_commands() -> None:
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
    assert {target.target_id for target in plan.commands.targets} == {
        "alternate_arm",
        "alternate_hand",
    }


def test_press_closes_hand_without_changing_projected_attachment() -> None:
    semantics = ObjectSemantics(
        affordance=PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="button",
        entity_id="button",
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
        entity_id="knob",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Twist())

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="twist",
            goal=TwistGoal(semantics, torch.eye(4)),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=TwistOptions(hand_interp_steps=3),
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    trajectory = _joint_trajectory(plan)
    assert trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "reach",
        "close",
        "twist",
        "open",
        "retract",
    ]
    assert torch.all(
        trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    assert torch.all(
        trajectory.positions[:, plan.segment("open").stop - 1, ARM_DOF:] == 0.0
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
        entity_id="rigid-knob",
    )

    action = _bind_action(_motion_generator(), Twist())
    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="twist",
            goal=TwistGoal(semantics, torch.eye(4)),
            binding=_binding(action),
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
            lambda semantics, target_pose: OpenDoorGoal(
                semantics,
                target_pose,
                open_fraction=0.5,
            ),
            OpenDoorAffordance(
                mesh_vertices=torch.tensor(
                    [[0.9, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.1, 0.0]]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
                axis_origin=(0.0, 0.0, 0.0),
                joint_name="door_hinge",
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
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        entity_id="target",
        label="target",
    )
    goal = goal_factory(semantics, SceneEntityPose("target-link"))

    assert collect_scene_dependencies(goal) == ("target-link",)


def test_open_loop_interaction_primitives_are_explicitly_described() -> None:
    assert Press.descriptor().open_loop is True
    assert Slide.descriptor().open_loop is True
    assert OpenDoor.descriptor().open_loop is True
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
    action = Twist()
    engine.register(action)
    semantics = ObjectSemantics(
        affordance=TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(0.0, 0.0, 0.0),
        ),
        geometry={},
        label="moving-knob",
        entity_id="moving-knob",
    )
    invocation = ActionInvocation(
        skill_id="twist",
        goal=TwistGoal(semantics, SceneEntityPose("target")),
        binding=engine.bind_control_parts(
            "twist",
            {"primary": {"motion": "arm", "grasp": "hand"}},
        ),
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
        *,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_calls.append((obj_poses, approach_direction))
        return (
            torch.ones(NUM_ENVS, dtype=torch.bool),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.full((NUM_ENVS,), 0.03),
        )

    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="drawer_handle",
        entity_id="drawer_handle",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Slide())
    _GRASP_GENERATORS[id(action)].get_best_grasp_poses = Mock(side_effect=sample_grasp)
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
            goal=SlideGoal(semantics, SceneEntityPose("target")),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(scene=_target_scene(link_pose, timestamp=0.0, version=0)),
    )

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    assert plan.scene_dependencies == ("target",)
    assert plan.scene_dependency_end_segment == "reach"
    assert trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == expected_segments
    assert torch.all(
        trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    assert torch.all(
        trajectory.positions[:, plan.segment("open").stop - 1, ARM_DOF:] == 0.0
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
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="drawer_handle",
        entity_id="drawer_handle",
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
    _GRASP_GENERATORS[id(action)].get_best_grasp_poses = Mock(
        return_value=(
            torch.tensor([True, False]),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.full((NUM_ENVS,), 0.03),
        )
    )
    context = _context()

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="slide",
            goal=SlideGoal(semantics, torch.eye(4)),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=18),
            skill_options=SlideOptions(hand_interp_steps=3),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(18, -1),
    )


def test_slide_fk_path_remains_on_translation_axis() -> None:
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
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        entity_id="handle",
        label="handle",
    )
    action = _bind_action(generator, Slide())
    _GRASP_GENERATORS[id(action)].get_best_grasp_poses = Mock(
        return_value=(
            torch.ones(NUM_ENVS, dtype=torch.bool),
            torch.eye(4).repeat(NUM_ENVS, 1, 1),
            torch.full((NUM_ENVS,), 0.03),
        )
    )
    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="slide",
            goal=SlideGoal(semantics, torch.eye(4)),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=SlideOptions(direction="pull", hand_interp_steps=3),
        ),
        _context(),
    )

    pull_segment = plan.segment("pull")
    arm_path = _joint_trajectory(plan).positions[
        :, pull_segment.start : pull_segment.stop, :ARM_DOF
    ]
    fk_path = position_fk(arm_path.reshape(-1, ARM_DOF), "arm", True).reshape(
        NUM_ENVS, -1, 4, 4
    )
    positions = fk_path[:, :, :3, 3]
    axis = torch.tensor([0.0, -1.0, 0.0])
    orthogonal = positions - (positions * axis).sum(dim=-1, keepdim=True) * axis
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1.0e-6)


def test_push_object_plans_contact_and_target_directed_planar_motion() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PushObject())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([0.4, 0.5])
    object_pose[1, :3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    target_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    target_pose[:, 0, 3] = object_pose[:, 0, 3]
    target_pose[:, 1, 3] += torch.tensor([0.12, -0.12])
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={
            "utensil": EntityState(object_pose),
            "slot": EntityState(target_pose),
        },
    )
    options = PushObjectOptions(
        hand_interp_steps=3,
        approach_height=0.1,
        retract_height=0.08,
        contact_distance=0.03,
        push_overshoot=0.02,
        object_contact_offset=torch.tensor([-0.05, 0.0, 0.0]),
        support_frame_planar_contact_offset=torch.tensor([-0.05, 0.0, 0.0]),
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="push_object",
            goal=PushObjectGoal(
                _semantics(entity_id="utensil"),
                SceneEntityPose("slot"),
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(scene=scene),
    )

    assert plan.plan_success.tolist() == [True, True]
    assert plan.expected_effects.is_empty
    assert plan.scene_dependencies == ("slot", "utensil")
    assert plan.scene_dependency_end_segment == "approach"
    assert [segment.name for segment in plan.segments] == [
        "close",
        "approach",
        "contact",
        "push",
        "retract",
    ]
    trajectory = _joint_trajectory(plan)
    assert trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert torch.all(
        trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )

    planned_targets = [
        call.kwargs["pose"] for call in generator.robot.compute_ik.call_args_list
    ]
    motion_lengths = PushObject._motion_segment_lengths(
        24,
        options.hand_interp_steps,
    )
    contact_stop = 1 + motion_lengths[1] - 1
    push_stop = contact_stop + motion_lengths[2] - 1
    expected_direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    expected_contact = torch.tensor([[0.35, -0.03, 0.0], [0.45, 0.03, 0.0]])
    expected_pushed = expected_contact + expected_direction * 0.14
    assert torch.allclose(
        planned_targets[0][:, :3, 3],
        expected_contact + torch.tensor([0.0, 0.0, 0.1]),
    )
    assert torch.allclose(
        planned_targets[contact_stop - 1][:, :3, 3],
        expected_contact,
    )
    assert torch.allclose(
        planned_targets[push_stop - 1][:, :3, 3],
        expected_pushed,
    )
    assert torch.allclose(
        planned_targets[-1][:, :3, 3],
        expected_pushed + torch.tensor([0.0, 0.0, 0.08]),
    )


def test_push_object_holds_rows_without_a_planar_displacement() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PushObject())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    target_pose = object_pose.clone()
    target_pose[1, 1, 3] = 0.1
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={
            "utensil": EntityState(object_pose),
            "slot": EntityState(target_pose),
        },
    )
    context = _context(scene=scene)

    plan = _plan_action(
        action,
        _invocation(
            action,
            PushObjectGoal(
                _semantics(entity_id="utensil"),
                SceneEntityPose("slot"),
            ),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [False, True]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(trajectory.waypoint_count, -1),
    )


def test_push_object_short_circuits_within_completion_tolerance() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, PushObject())
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    target_pose = object_pose.clone()
    target_pose[:, 1, 3] = 0.02
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={
            "utensil": EntityState(object_pose),
            "slot": EntityState(target_pose),
        },
    )
    context = _context(scene=scene)

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="push_object",
            goal=PushObjectGoal(
                _semantics(entity_id="utensil"),
                SceneEntityPose("slot"),
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=PushObjectOptions(completion_tolerance=0.03),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, True]
    generator.robot.compute_ik.assert_not_called()
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions,
        context.robot.qpos.unsqueeze(1).expand_as(trajectory.positions),
    )


def test_push_object_options_own_calibration_tensors() -> None:
    contact_offset = torch.tensor([-0.05, 0.0, 0.0])
    support_offset = torch.tensor([-0.04, 0.0, 0.0])
    calibration = torch.eye(4)
    tool_calibration = PushObjectToolCalibration(
        control_part="alternate_arm",
        contact_frame_to_eef=calibration,
        contact_distance=0.08,
    )
    options = PushObjectOptions(
        object_contact_offset=contact_offset,
        support_frame_planar_contact_offset=support_offset,
        contact_frame_to_eef=calibration,
        tool_calibrations=(tool_calibration,),
    )

    contact_offset[0] = 9.0
    support_offset[0] = 9.0
    calibration[0, 3] = 9.0
    tool_calibration.contact_frame_to_eef[1, 3] = 9.0

    assert torch.equal(
        options.object_contact_offset,
        torch.tensor([-0.05, 0.0, 0.0]),
    )
    assert torch.equal(
        options.support_frame_planar_contact_offset,
        torch.tensor([-0.04, 0.0, 0.0]),
    )
    assert torch.equal(options.contact_frame_to_eef, torch.eye(4))
    assert torch.equal(
        options.tool_calibrations[0].contact_frame_to_eef,
        torch.eye(4),
    )
    assert options._contact_distance_for("alternate_arm") == pytest.approx(0.08)
    assert options._contact_distance_for("arm") == pytest.approx(0.03)


def _door_affordance(*, opening_direction: int = 1) -> OpenDoorAffordance:
    """Build a handle at local +X around a local +Z hinge for pure tests."""
    return OpenDoorAffordance(
        mesh_vertices=torch.tensor(
            [
                [0.9, -0.1, 0.0],
                [1.1, -0.1, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=(0.0, 0.0, 0.0),
        joint_name="door_hinge",
        joint_limits=(0.0, math.pi / 2),
        opening_direction=opening_direction,
    )


def test_open_door_plans_approach_grasp_arc_release_and_rotated_retract() -> None:
    affordance = _door_affordance()
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    generator = _motion_generator()
    action = _bind_action(generator, OpenDoor())
    grasp_calls: list[torch.Tensor] = []

    def sample_grasp(
        *,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_calls.append(approach_direction)
        grasp_pose = obj_poses.clone()
        grasp_pose[:, 0, 3] = 1.0
        return (
            torch.ones(NUM_ENVS, dtype=torch.bool),
            grasp_pose,
            torch.full((NUM_ENVS,), 0.03),
        )

    _GRASP_GENERATORS[id(action)].get_best_grasp_poses = Mock(side_effect=sample_grasp)
    options = OpenDoorOptions(
        hand_interp_steps=3,
        door_waypoint_count=4,
        approach_distance=0.1,
        retract_distance=0.1,
    )
    link_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(
                semantics,
                SceneEntityPose("target"),
                open_fraction=1.0 / 3.0,
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(scene=_door_scene(link_pose)),
    )

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    assert plan.scene_dependencies == ("target",)
    assert plan.scene_dependency_end_segment == "reach"
    assert trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == [
        "approach",
        "reach",
        "close",
        "open",
        "release",
        "retract",
    ]
    assert torch.all(
        trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
    )
    assert torch.all(
        trajectory.positions[:, plan.segment("release").stop - 1, ARM_DOF:] == 0.0
    )
    assert torch.allclose(
        grasp_calls[0],
        torch.tensor([0.0, -1.0, 0.0]).expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )
    planned_targets = [
        call.kwargs["pose"] for call in generator.robot.compute_ik.call_args_list
    ]
    assert torch.allclose(
        planned_targets[0][:, :3, 3],
        torch.tensor([1.0, 0.1, 0.0]).expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )
    angle = math.pi / 6
    expected_retract = torch.tensor(
        [
            math.cos(angle) - options.retract_distance * math.sin(angle),
            math.sin(angle) + options.retract_distance * math.cos(angle),
            0.0,
        ]
    )
    assert torch.allclose(
        planned_targets[-1][:, :3, 3],
        expected_retract.expand(NUM_ENVS, -1),
        atol=1.0e-6,
    )


def test_open_door_interpolates_link_arc_and_recovers_eef_poses() -> None:
    action = _bind_action(_motion_generator(), OpenDoor())
    link_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp_pose = link_pose.clone()
    grasp_pose[:, 0, 3] = 1.0

    opened_links, opened_eef = action._opened_link_and_eef_poses(
        link_pose,
        grasp_pose,
        torch.tensor([0.0, 0.0, 1.0]),
        (0.0, 0.0, 0.0),
        torch.tensor([math.pi / 6, math.pi / 4]),
        4,
    )

    expected_positions = torch.tensor(
        [
            [math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0],
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0],
        ]
    )
    assert opened_links.shape == (NUM_ENVS, 4, 4, 4)
    assert torch.allclose(
        opened_eef[:, -1, :3, 3],
        expected_positions,
        atol=1.0e-6,
    )


def test_open_door_uses_affordance_owned_negative_opening_direction() -> None:
    action = _bind_action(_motion_generator(), OpenDoor())
    context = _context()

    rotation, active, already_open, valid = action._resolve_hinge_rotation(
        0.5,
        torch.full((NUM_ENVS, 1), math.pi / 2),
        None,
        (0.0, math.pi / 2),
        -1,
        context,
        tolerance=1.0e-4,
    )

    assert torch.allclose(rotation, torch.full((NUM_ENVS,), -math.pi / 4))
    assert active.tolist() == [True, True]
    assert already_open.tolist() == [False, False]
    assert valid.tolist() == [True, True]


def test_open_door_holds_failed_grasp_environment() -> None:
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    generator = _motion_generator()
    action = _bind_action(generator, OpenDoor())
    grasp_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    grasp_pose[:, 0, 3] = 1.0
    _GRASP_GENERATORS[id(action)].get_best_grasp_poses = Mock(
        return_value=(
            torch.tensor([True, False]),
            grasp_pose,
            torch.full((NUM_ENVS,), 0.03),
        )
    )
    context = _context()

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(semantics, torch.eye(4), open_fraction=0.5),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=18),
            skill_options=OpenDoorOptions(
                hand_interp_steps=3,
                door_waypoint_count=4,
            ),
        ),
        replace(context, scene=_door_scene()),
    )

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(18, -1),
    )


def test_open_door_fails_row_whose_target_would_close_hinge() -> None:
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    action = _bind_action(_motion_generator(), OpenDoor())
    context = _context(
        scene=_door_scene(
            hinge_position=torch.tensor([[0.0], [math.pi / 2]]),
        )
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(
                semantics,
                SceneEntityPose("target"),
                open_fraction=torch.tensor([0.5, 0.25]),
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=OpenDoorOptions(
                hand_interp_steps=3,
                door_waypoint_count=4,
            ),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, False]
    assert torch.allclose(
        _joint_trajectory(plan).positions[1],
        context.robot.qpos[1].unsqueeze(0).expand(24, -1),
    )


def test_open_door_fails_rows_with_invalid_fraction_or_out_of_limit_hinge() -> None:
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    action = _bind_action(_motion_generator(), OpenDoor())
    context = _context(
        scene=_door_scene(
            hinge_position=torch.tensor([[0.0], [math.pi]]),
        )
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(
                semantics,
                SceneEntityPose("target"),
                open_fraction=torch.tensor([1.1, 0.5]),
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=OpenDoorOptions(
                hand_interp_steps=3,
                door_waypoint_count=4,
            ),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [False, False]
    assert "invalid" in plan.diagnostics.messages[0]


def test_open_door_holds_row_already_at_requested_open_fraction() -> None:
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    action = _bind_action(_motion_generator(), OpenDoor())
    context = _context(
        scene=_door_scene(
            hinge_position=torch.tensor([[math.pi / 4], [0.0]]),
        )
    )

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(
                semantics,
                SceneEntityPose("target"),
                open_fraction=0.5,
            ),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=OpenDoorOptions(
                hand_interp_steps=3,
                door_waypoint_count=4,
            ),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, True]
    assert torch.allclose(
        _joint_trajectory(plan).positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(24, -1),
    )


def test_open_door_fails_when_live_hinge_observation_is_missing() -> None:
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    action = _bind_action(_motion_generator(), OpenDoor())

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="open_door",
            goal=OpenDoorGoal(semantics, torch.eye(4), open_fraction=0.5),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=OpenDoorOptions(
                hand_interp_steps=3,
                door_waypoint_count=4,
            ),
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [False, False]
    assert plan.diagnostics.messages == (
        "No observed articulation joint named 'door_hinge'.",
    )


def test_open_door_rejects_wrong_affordance_and_invalid_goal_shape() -> None:
    action = _bind_action(_motion_generator(), OpenDoor())
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="handle",
    )

    with pytest.raises(ValueError, match="OpenDoorAffordance"):
        _plan_action(
            action,
            _invocation(
                action,
                OpenDoorGoal(semantics, torch.eye(4), open_fraction=0.5),
            ),
            _context(),
        )
    with pytest.raises(ValueError, match="scalar or have shape"):
        OpenDoorGoal(
            semantics,
            torch.eye(4),
            open_fraction=torch.zeros(2, 1),
        )


def test_open_door_validates_goal_binding_owner_and_endpoint_coverage() -> None:
    action = _bind_action(_motion_generator(), OpenDoor())
    semantics = ObjectSemantics(
        affordance=_door_affordance(),
        geometry={},
        entity_id=DOOR_ENTITY_ID,
        label="door_handle",
    )
    valid_goal = OpenDoorGoal(semantics, torch.eye(4), open_fraction=0.5)

    with pytest.raises(TypeError, match="expects goal OpenDoorGoal"):
        action.resolve_request(
            ActionInvocation(
                skill_id="open_door",
                goal=object(),
                binding=_binding(action),
            )
        )
    with pytest.raises(ValueError, match="another engine instance"):
        action.resolve_request(
            ActionInvocation(
                skill_id="open_door",
                goal=valid_goal,
                binding=replace(_binding(action), owner_id="another-engine"),
            )
        )
    with pytest.raises(ValueError, match="missing=.*grasp"):
        action.resolve_request(
            ActionInvocation(
                skill_id="open_door",
                goal=valid_goal,
                binding=ActionBinding(
                    owner_id=_ACTION_ENGINES[id(action)].binding_owner_id,
                ),
            )
        )


def test_press_plans_close_approach_press_and_retract() -> None:
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]),
        press_position=(0.0, 0.0, 0.0),
    )
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="button",
        entity_id="button",
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
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=options,
        ),
        _context(),
    )

    assert plan.plan_success.tolist() == [True, True]
    trajectory = _joint_trajectory(plan)
    assert trajectory.positions.shape == (NUM_ENVS, 24, ROBOT_DOF)
    assert [segment.name for segment in plan.segments] == [
        "close",
        "approach",
        "contact",
        "press",
        "retract",
    ]
    assert torch.all(
        trajectory.positions[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0
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
        entity_id="rigid-button",
    )
    generator = _motion_generator()
    action = _bind_action(generator, Press())

    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(action),
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
        entity_id="button",
    )
    action = _bind_action(generator, Press())
    plan = _plan_action(
        action,
        ActionInvocation(
            skill_id="press",
            goal=PressGoal(semantics, torch.eye(4)),
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=24),
            skill_options=PressOptions(hand_interp_steps=3, press_distance=0.04),
        ),
        _context(),
    )

    trajectory = _joint_trajectory(plan)
    contact_arm = trajectory.positions[:, plan.segment("contact").stop - 1, :ARM_DOF]
    contact_fk = position_fk(contact_arm, "arm", True)
    assert torch.allclose(contact_fk[:, :3, 3], torch.zeros(NUM_ENVS, 3))
    press_segment = plan.segment("press")
    press_arm = trajectory.positions[
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
        entity_id="button",
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
            binding=_binding(action),
            motion_policy=MotionPolicy(sample_count=18),
            skill_options=PressOptions(hand_interp_steps=3),
        ),
        context,
    )

    assert plan.plan_success.tolist() == [True, False]
    trajectory = _joint_trajectory(plan)
    assert not torch.allclose(trajectory.positions[0], context.robot.qpos[0])
    assert torch.allclose(
        trajectory.positions[1],
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
        entity_id="mesh-button",
    )
    action = _bind_action(_motion_generator(), Press())

    with pytest.raises(ValueError, match="PressAffordance"):
        _plan_action(
            action,
            _invocation(action, PressGoal(semantics, torch.eye(4))),
            _context(),
        )


def test_press_requires_primary_arm_and_end_effector_bindings() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="button",
        entity_id="button",
    )
    action = _bind_action(_motion_generator(), Press())
    invocation = ActionInvocation(
        skill_id="press",
        goal=PressGoal(semantics, torch.eye(4)),
        binding=ActionBinding(
            owner_id=_ACTION_ENGINES[id(action)].binding_owner_id,
        ),
    )

    with pytest.raises(ValueError, match="missing=.*grasp"):
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
        entity_id="mesh-knob",
    )
    action = _bind_action(_motion_generator(), Twist())

    with pytest.raises(ValueError, match="TwistAffordance"):
        _plan_action(
            action,
            _invocation(action, TwistGoal(semantics, torch.eye(4))),
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
        entity_id="mesh-handle",
    )
    action = _bind_action(_motion_generator(), Slide())

    with pytest.raises(ValueError, match="SlideAffordance"):
        _plan_action(
            action,
            _invocation(
                action,
                SlideGoal(semantics, torch.eye(4)),
            ),
            _context(),
        )


def test_slide_requires_primary_end_effector() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="drawer_handle",
        entity_id="drawer_handle",
    )
    action = _bind_action(_motion_generator(), Slide())
    invocation = ActionInvocation(
        skill_id="slide",
        goal=SlideGoal(semantics, torch.eye(4)),
        binding=ActionBinding(
            owner_id=_ACTION_ENGINES[id(action)].binding_owner_id,
        ),
    )

    with pytest.raises(ValueError, match="missing=.*grasp"):
        action.resolve_request(invocation)


def test_slide_axis_belongs_to_affordance_not_action_options() -> None:
    assert "translation_axis" not in SlideOptions.__dataclass_fields__


def test_slide_options_reject_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        SlideOptions(direction="open")  # type: ignore[arg-type]


def _handover_semantics(
    longest_axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0]),
) -> tuple[ObjectSemantics, AntipodalAffordance]:
    affordance = AntipodalAffordance()

    def get_object_longest_axis(
        obj_poses: torch.Tensor,
        *,
        max_points: int,
    ) -> torch.Tensor:
        assert max_points <= 1000
        return longest_axis.to(dtype=torch.float32).expand(obj_poses.shape[0], -1)

    def sample_grasps(
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor,
        is_positive_part: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        assert obj_poses.shape == (1, 4, 4)
        assert approach_direction.shape == (3,)
        assert obj_longest_axis.shape == (3,)
        grasp_poses = obj_poses.clone()
        if is_positive_part:
            grasp_poses[:, :3, :3] = torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
        else:
            grasp_poses[:, :3, :3] = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            )
        return [
            (grasp_poses[index].unsqueeze(0), torch.zeros(1))
            for index in range(obj_poses.shape[0])
        ]

    affordance.get_object_longest_axis = Mock(side_effect=get_object_longest_axis)
    affordance.get_valid_grasp_poses = Mock(side_effect=sample_grasps)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="handover-object",
        entity_id="handover_object",
    )
    return semantics, affordance


def _handover_context(
    object_pose: torch.Tensor,
    task: TaskState | None = None,
) -> PlanningContext:
    return _dual_context(
        task,
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={"handover_object": EntityState(object_pose)},
        ),
    )


def test_handover_picks_with_nearer_arm_and_preserves_waypoint_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, :3, 3] = torch.tensor([-0.8, 0.1, 0.5])
    final_pose = torch.eye(4)
    final_pose[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    final_pose[:3, 3] = torch.tensor([0.4, -0.2, 0.65])
    planned_parts: list[str] = []
    planned_targets: list[torch.Tensor] = []

    def resolve_grasp(
        sampled_affordance: AntipodalAffordance,
        sampled_object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        grasp_target_id: str,
        *,
        obj_longest_axis: torch.Tensor,
        is_positive_part: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sampled_affordance, approach_direction, grasp_target_id, obj_longest_axis
        grasp_pose = sampled_object_pose.clone()
        if bool(is_positive_part[0].item()):
            grasp_pose[:, :3, :3] = torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
        else:
            grasp_pose[:, :3, :3] = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            )
        return grasp_pose, torch.ones(NUM_ENVS, dtype=torch.bool)

    action._resolve_grasp = Mock(side_effect=resolve_grasp)

    def plan_from_start(
        motion_generator: MotionGenerator,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        motion_policy: MotionPolicy,
        interpolation_dt: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del motion_generator, motion_policy, interpolation_dt
        planned_parts.append(control_part)
        planned_targets.append(target_poses.clone())
        trajectory = (start_qpos + 0.1).unsqueeze(1).repeat(1, n_waypoints, 1)
        return torch.ones(NUM_ENVS, dtype=torch.bool), trajectory

    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives.hand_over."
        "plan_named_arm_trajectory",
        plan_from_start,
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=final_pose),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(
            pre_grasp_distance=0.1,
            lift_height=0.2,
            hand_interp_steps=2,
        ),
    )
    context = _handover_context(object_pose)
    original_task = context.task

    plan = _plan_action(action, invocation, context)

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    assert trajectory.positions.shape == (NUM_ENVS, 24, DUAL_ROBOT_DOF)
    assert planned_parts == ["left_arm", "left_arm", "right_arm", "right_arm"]
    assert [segment.name for segment in plan.segments] == [
        "pickup_approach",
        "pickup_close",
        "pickup_transport",
        "receive_approach",
        "receive_close",
        "handover_release",
        "place",
        "receive_release",
    ]

    pickup_call, receive_call = action._resolve_grasp.call_args_list
    expected_axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(NUM_ENVS, -1)
    assert torch.equal(pickup_call.kwargs["obj_longest_axis"], expected_axis)
    assert pickup_call.kwargs["is_positive_part"].tolist() == [False, False]
    diagonal_component = math.sqrt(0.5)
    pickup_horizontal = object_pose[:, :2, 3]
    pickup_horizontal = pickup_horizontal / torch.linalg.vector_norm(
        pickup_horizontal, dim=1, keepdim=True
    )
    expected_pickup_direction = torch.zeros(NUM_ENVS, 3)
    expected_pickup_direction[:, :2] = pickup_horizontal * diagonal_component
    expected_pickup_direction[:, 2] = -diagonal_component
    assert torch.allclose(pickup_call.args[2], expected_pickup_direction)
    assert torch.equal(receive_call.kwargs["obj_longest_axis"], expected_axis)
    assert receive_call.kwargs["is_positive_part"].tolist() == [True, True]
    predicted_middle_pose = receive_call.args[1]
    assert torch.allclose(
        predicted_middle_pose[:, :3, 3],
        torch.tensor([[0.0, 0.1, 0.7], [0.0, 0.1, 0.7]]),
    )
    expected_receive_direction = torch.tensor(
        [
            [0.0, diagonal_component, -diagonal_component],
            [0.0, diagonal_component, -diagonal_component],
        ]
    )
    assert torch.allclose(receive_call.args[2], expected_receive_direction)

    pickup_grasp_rotation = planned_targets[0][:, 1, :3, :3]
    assert torch.allclose(
        planned_targets[1][:, :, :3, :3],
        pickup_grasp_rotation[:, None].expand(-1, 2, -1, -1),
    )
    receive_grasp_rotation = planned_targets[2][:, 1, :3, :3]
    assert torch.allclose(
        planned_targets[3][:, :, :3, :3],
        receive_grasp_rotation[:, None].expand(-1, 2, -1, -1),
    )
    assert torch.allclose(
        planned_targets[3][:, 0, :3, 3],
        torch.tensor([[0.4, -0.2, 0.7], [0.4, -0.2, 0.7]]),
    )
    assert torch.allclose(
        planned_targets[3][:, 1, :3, 3],
        torch.tensor([[0.4, -0.2, 0.65], [0.4, -0.2, 0.65]]),
    )
    assert torch.allclose(
        planned_targets[3][:, 0, 2, 3],
        planned_targets[2][:, 1, 2, 3],
    )

    pickup_close_end = plan.segment("pickup_close").stop - 1
    receive_close_end = plan.segment("receive_close").stop - 1
    handover_release_end = plan.segment("handover_release").stop - 1
    receive_release_end = plan.segment("receive_release").stop - 1
    positions = trajectory.positions
    assert torch.all(
        positions[:, pickup_close_end, DUAL_ARM_DOF : DUAL_ARM_DOF + 2] == 1
    )
    assert torch.all(positions[:, pickup_close_end, DUAL_ARM_DOF + 2 :] == 0)
    assert torch.all(positions[:, receive_close_end, DUAL_ARM_DOF:] == 1)
    assert torch.all(
        positions[:, handover_release_end, DUAL_ARM_DOF : DUAL_ARM_DOF + 2] == 0
    )
    assert torch.all(positions[:, handover_release_end, DUAL_ARM_DOF + 2 :] == 1)
    assert torch.all(positions[:, receive_release_end, DUAL_ARM_DOF:] == 0)
    assert dict(plan.expected_effects.held_object_updates) == {
        "left_arm": None,
        "right_arm": None,
    }
    assert set(plan.effect_candidates.held_object_updates) == {
        "left_arm",
        "right_arm",
    }
    assert all(
        isinstance(candidate, HeldObjectState)
        for candidate in plan.effect_candidates.held_object_updates.values()
    )
    assert context.task is original_task
    assert plan.scene_dependencies == ("handover_object",)
    assert plan.scene_dependency_monitor_until == {
        "handover_object": plan.segment("pickup_close").stop
    }


def test_handover_horizontal_mode_uses_downward_opposite_end_grasps() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics(torch.tensor([1.0, 0.0, 0.0]))
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, :3, 3] = torch.tensor([-0.8, 0.2, 0.5])

    def resolve_grasp(
        affordance: AntipodalAffordance,
        sampled_object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        grasp_target_id: str,
        *,
        obj_longest_axis: torch.Tensor,
        is_positive_part: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            affordance,
            approach_direction,
            grasp_target_id,
            obj_longest_axis,
            is_positive_part,
        )
        return sampled_object_pose.clone(), torch.ones(NUM_ENVS, dtype=torch.bool)

    action._resolve_grasp = Mock(side_effect=resolve_grasp)
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(hand_interp_steps=2),
    )

    plan = _plan_action(action, invocation, _handover_context(object_pose))

    assert plan.plan_success.tolist() == [True, True]
    pickup_call, receive_call = action._resolve_grasp.call_args_list
    downward = torch.tensor([0.0, 0.0, -1.0])
    expected_axis = torch.tensor([[1.0, 0.0, 0.0]]).expand(NUM_ENVS, -1)
    assert torch.equal(pickup_call.args[2], downward.expand(NUM_ENVS, -1))
    assert torch.equal(pickup_call.kwargs["obj_longest_axis"], expected_axis)
    assert pickup_call.kwargs["is_positive_part"].tolist() == [True, True]
    assert torch.equal(receive_call.args[2], downward.expand(NUM_ENVS, -1))
    assert torch.equal(receive_call.kwargs["obj_longest_axis"], expected_axis)
    assert receive_call.kwargs["is_positive_part"].tolist() == [False, False]


def test_handover_selects_arm_per_environment() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([-0.8, 0.8])
    object_pose[:, 1, 3] = 0.2
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(hand_interp_steps=2),
    )

    plan = _plan_action(action, invocation, _handover_context(object_pose))

    trajectory = _joint_trajectory(plan)
    assert plan.plan_success.tolist() == [True, True]
    pickup_close_end = plan.segment("pickup_close").stop - 1
    positions = trajectory.positions[:, pickup_close_end]
    assert torch.all(positions[0, DUAL_ARM_DOF : DUAL_ARM_DOF + 2] == 1)
    assert torch.all(positions[0, DUAL_ARM_DOF + 2 :] == 0)
    assert torch.all(positions[1, DUAL_ARM_DOF : DUAL_ARM_DOF + 2] == 0)
    assert torch.all(positions[1, DUAL_ARM_DOF + 2 :] == 1)


def test_handover_holds_rows_whose_candidate_arm_is_occupied() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = -0.8
    object_pose[:, 1, 3] = 0.2
    occupied = _held(env_mask=torch.tensor([True, False]))
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={"left_arm": occupied},
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(hand_interp_steps=2),
    )
    context = _handover_context(object_pose, task)

    plan = _plan_action(action, invocation, context)

    assert plan.plan_success.tolist() == [False, True]
    trajectory = _joint_trajectory(plan)
    assert torch.allclose(
        trajectory.positions[0],
        context.robot.qpos[0].unsqueeze(0).expand(24, -1),
    )


def test_handover_reports_failed_semantic_waypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = -0.8
    object_pose[:, 1, 3] = 0.2
    grasp_call_count = 0

    def resolve_grasp(
        affordance: AntipodalAffordance,
        sampled_object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        grasp_target_id: str,
        *,
        obj_longest_axis: torch.Tensor,
        is_positive_part: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal grasp_call_count
        del (
            affordance,
            approach_direction,
            grasp_target_id,
            obj_longest_axis,
            is_positive_part,
        )
        grasp_call_count += 1
        success = torch.ones(NUM_ENVS, dtype=torch.bool)
        if grasp_call_count == 2:
            success[0] = False
        return sampled_object_pose.clone(), success

    warnings: list[str] = []
    action._resolve_grasp = Mock(side_effect=resolve_grasp)
    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives.hand_over.logger.log_warning",
        warnings.append,
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(hand_interp_steps=2),
    )

    plan = _plan_action(action, invocation, _handover_context(object_pose))

    assert plan.plan_success.tolist() == [False, True]
    assert any(
        "waypoint 'receive_grasp'" in warning and "env_ids=[0]" in warning
        for warning in warnings
    )


def test_handover_reports_path_failure_between_reachable_waypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics, _ = _handover_semantics()
    object_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    object_pose[:, 0, 3] = -0.8
    object_pose[:, 1, 3] = 0.2
    phase_call_count = 0

    def plan_phase(
        motion_generator: MotionGenerator,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        motion_policy: MotionPolicy,
        interpolation_dt: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal phase_call_count
        del (
            motion_generator,
            control_part,
            target_poses,
            motion_policy,
            interpolation_dt,
        )
        phase_call_count += 1
        success = torch.ones(NUM_ENVS, dtype=torch.bool)
        if phase_call_count == 1:
            success[0] = False
        trajectory = start_qpos.unsqueeze(1).repeat(1, n_waypoints, 1)
        return success, trajectory

    warnings: list[str] = []
    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives.hand_over."
        "plan_named_arm_trajectory",
        plan_phase,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.atomic_actions.primitives.hand_over.logger.log_warning",
        warnings.append,
    )
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HandOverOptions(hand_interp_steps=2),
    )

    plan = _plan_action(action, invocation, _handover_context(object_pose))

    assert plan.plan_success.tolist() == [False, True]
    assert any(
        "phase 'pickup_approach' failed between waypoints" in warning
        and "pickup_pre_grasp" in warning
        and "pickup_grasp" in warning
        and "env_ids=[0]" in warning
        for warning in warnings
    )


def test_handover_requires_antipodal_affordance_and_valid_options() -> None:
    generator = _dual_motion_generator()
    action = _bind_action(generator, HandOver())
    semantics = _semantics(entity_id="target")
    invocation = ActionInvocation(
        skill_id="hand_over",
        goal=HandOverGoal(semantics, target_pose=torch.eye(4)),
        binding=_dual_binding(action, "source", "destination"),
    )

    with pytest.raises(ValueError, match="AntipodalAffordance"):
        _plan_action(
            action,
            invocation,
            _dual_context(
                scene=_target_scene(
                    torch.eye(4).repeat(NUM_ENVS, 1, 1),
                    timestamp=0.0,
                    version=0,
                )
            ),
        )

    with pytest.raises(ValueError, match="lift_height"):
        HandOverOptions(lift_height=float("nan"))
    with pytest.raises(ValueError, match="hand_interp_steps"):
        HandOverOptions(hand_interp_steps=0)


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
    _stub_dual_arm_grasp_poses(action)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="coordinated-object",
        entity_id="coordinated_object",
    )
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
    left_held = projected.get_held_object("left_arm")
    right_held = projected.get_held_object("right_arm")
    assert isinstance(left_held, HeldObjectState)
    assert isinstance(right_held, HeldObjectState)
    assert left_held.semantics is right_held.semantics
    assert left_held.semantics is not semantics
    assert plan.commands.frame_count == 30
    assert {target.target_id for target in plan.commands.targets} == {
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    }
    assert plan.scene_dependencies == ()
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
    _stub_dual_arm_grasp_poses(action)
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        label="snapshot-coordinated-object",
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

    sampled_pose = _GRASP_GENERATORS[
        id(action)
    ].get_dual_arm_valid_grasp_poses.call_args.kwargs["obj_poses"]
    assert torch.equal(sampled_pose, object_pose)
    assert plan.scene_dependencies == ("target",)
    left_held = projected.get_held_object("left_arm")
    right_held = projected.get_held_object("right_arm")
    assert left_held is not None and right_held is not None
    assert left_held.semantics is right_held.semantics
    assert torch.allclose(left_held.object_to_eef, pose_inv(object_pose))
    assert torch.allclose(right_held.object_to_eef, pose_inv(object_pose))


def test_assemble_place_uses_explicit_base_snapshot() -> None:
    generator = _motion_generator()
    action = _bind_action(generator, Place())
    relative_pose = torch.eye(4)
    relative_pose[2, 3] = 0.05
    affordance = AssembleAffordance(
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
            action,
            AssembleGoal(
                affordance=affordance,
                base_pose=SceneEntityPose("base"),
            ),
        )
    )
    plan = action.plan(request, context)

    assert plan.plan_success.all()
    assert plan.scene_dependencies == ("base",)


def test_assemble_goal_requires_snapshot_base_pose() -> None:
    with pytest.raises(TypeError, match="base_pose"):
        AssembleGoal(affordance=AssembleAffordance())  # type: ignore[call-arg]


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
    _stub_dual_arm_grasp_poses(action)
    invocation = ActionInvocation(
        skill_id="coordinated_pickment",
        goal=CoordinatedPickGoal(
            semantics=ObjectSemantics(
                affordance=affordance,
                geometry={},
                entity_id="tray",
                label="tray",
            ),
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
    _GRASP_GENERATORS[id(action)].get_dual_arm_valid_grasp_poses = Mock(
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
    semantics = ObjectSemantics(
        affordance=affordance,
        geometry={},
        entity_id="object",
        label="object",
    )
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
        ObjectSemantics(
            affordance=AntipodalAffordance(),
            geometry={},
            entity_id="placing",
            label="placing",
        )
    )
    support = _held(
        ObjectSemantics(
            affordance=AntipodalAffordance(),
            geometry={},
            entity_id="support",
            label="support",
        )
    )
    task = TaskState(
        batch_size=NUM_ENVS,
        device="cpu",
        held_objects={
            "left_arm": placing,
            "right_arm": support,
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
    projected_placing = projected.get_held_object("left_arm")
    if release:
        assert projected_placing is None
    else:
        assert projected_placing is not None
        assert projected_placing.semantics is placing.semantics
        assert torch.equal(projected_placing.object_to_eef, placing.object_to_eef)
    projected_support = projected.get_held_object("right_arm")
    assert projected_support is not None
    assert projected_support.semantics is support.semantics
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
        ObjectSemantics(
            affordance=AntipodalAffordance(),
            geometry={},
            entity_id="placing",
            label="placing",
        )
    )
    support = _held(
        ObjectSemantics(
            affordance=AntipodalAffordance(),
            geometry={},
            entity_id="support",
            label="support",
        )
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
                affordance=AntipodalAffordance(),
                geometry={},
                entity_id="object",
                label="object",
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
