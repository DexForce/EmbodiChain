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

"""Source-neutral trajectory schemas, snapshots and complete Skill assembly."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    JointPositionGoal,
    MoveJoints,
    Place,
    PlannerDiagnostics,
    RecoveryPolicy,
    SkillPlanSemantics,
    SkillTrajectoryContract,
    SkillTrajectoryInput,
    TimedTrajectory,
    TrajectorySegment,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickTrajectoryGeometry
from embodichain.lab.task_program.integrations.catalog import _canonical_value

CONTROL_DT = 0.01
BATCH_SIZE = 2
ROBOT_DOF = 3
SAMPLE_COUNT = 5


def _motion(**changes: object) -> SkillTrajectoryInput:
    positions = torch.zeros(BATCH_SIZE, SAMPLE_COUNT, ROBOT_DOF)
    positions[:, :, :2] = torch.linspace(0, 0.01, SAMPLE_COUNT)[None, :, None]
    trajectory = TimedTrajectory.from_uniform_step(
        positions,
        env_ids=torch.tensor([7, 3]),
        step_dt=CONTROL_DT,
    )
    values = dict(
        schema_id="move_joints.qpos",
        schema_revision="1",
        trajectory=trajectory,
        landmarks={"goal": SAMPLE_COUNT - 1},
        diagnostics=PlannerDiagnostics("external_fixture"),
    )
    values.update(changes)
    return SkillTrajectoryInput(**values)


def _engine() -> tuple[AtomicActionEngine, object, object]:
    robot = Mock()
    robot.dof = ROBOT_DOF
    robot.device = torch.device("cpu")
    robot.joint_names = ["shoulder", "elbow", "finger"]
    robot.mimic_ids = []
    robot.control_parts = {"custom_motion": object(), "custom_grasp": object()}
    robot.get_qpos.return_value = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    robot.get_qvel.return_value = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    robot.get_joint_ids.side_effect = lambda name=None, **_: (
        [0, 1]
        if name == "custom_motion"
        else [2] if name == "custom_grasp" else [0, 1, 2]
    )
    robot.get_qpos_limits.return_value = torch.tensor([-2.0, 2.0]).expand(
        BATCH_SIZE, ROBOT_DOF, 2
    )
    robot.get_qvel_limits.return_value = torch.full((BATCH_SIZE, ROBOT_DOF), 2.0)
    generator = Mock(robot=robot, device=torch.device("cpu"))
    generator.planner.cfg.planner_type = "fixture"
    generator.planner.collision_world_info = None
    engine = AtomicActionEngine(generator, load_builtins=False)
    engine.register(MoveJoints())
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.full((BATCH_SIZE, 2), 0.01)),
        binding=engine.bind_control_parts(
            "move_joints", {"primary": {"motion": "custom_motion"}}
        ),
        recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
    )
    request = engine.resolve(invocation)
    context = replace(
        engine.initial_context(), env_ids=torch.tensor([7, 3]), control_dt=CONTROL_DT
    )
    return engine, request, context


def test_assembly_input_and_plan_snapshots_own_all_mutable_values() -> None:
    engine, request, context = _engine()
    motion = _motion()
    plan = engine.build_plan_from_trajectory(request, context, motion)
    copied = plan.snapshot()
    motion.trajectory.positions.fill_(0.5)
    motion.planning_success.zero_()
    plan.trajectory_input.trajectory.positions.fill_(0.7)
    assert copied.joint_trajectory.positions[:, -1, 0].tolist() == pytest.approx(
        [0.01, 0.01]
    )
    assert copied.trajectory_input.trajectory.positions[
        :, -1, 0
    ].tolist() == pytest.approx([0.01, 0.01])
    assert copied.plan_success.all()
    with pytest.raises(TypeError):
        copied.trajectory_input.landmarks["goal"] = 0


def test_geometry_is_typed_and_owned_without_becoming_measured_evidence() -> None:
    pose = torch.eye(4)[None]
    geometry = PickTrajectoryGeometry(pose)
    motion = _motion(geometry=geometry)
    pose[:, 0, 3] = 1
    geometry.grasp_pose[:, 0, 3] = 2
    assert motion.geometry.grasp_pose[0, 0, 3] == 0
    with pytest.raises(TypeError, match="geometric facts"):
        MoveJoints.trajectory_contract.validate_input(motion)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"schema_revision": "2"}, "schema identity or revision"),
        ({"schema_id": "pick_up.qpos"}, "schema identity or revision"),
        ({"landmarks": {}}, "event landmarks"),
        ({"landmarks": {"goal": 0}}, "stage boundary"),
        (
            {"segments": (TrajectorySegment("other", 0, SAMPLE_COUNT),)},
            "semantic stages",
        ),
    ],
)
def test_skill_contract_rejects_incompatible_input(changes: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        MoveJoints.trajectory_contract.validate_input(_motion(**changes))


@pytest.mark.parametrize(
    "landmarks", [{"goal": -1}, {"goal": SAMPLE_COUNT}, {"goal": True}]
)
def test_event_indices_are_actual_samples(landmarks: dict) -> None:
    with pytest.raises(ValueError, match="actual trajectory samples"):
        _motion(landmarks=landmarks)


@pytest.mark.parametrize(
    "segments",
    [
        (TrajectorySegment("a", 1, SAMPLE_COUNT),),
        (TrajectorySegment("a", 0, 2), TrajectorySegment("b", 3, SAMPLE_COUNT)),
        (TrajectorySegment("a", 0, 3), TrajectorySegment("b", 2, SAMPLE_COUNT)),
        (TrajectorySegment("a", 0, SAMPLE_COUNT - 1),),
        (TrajectorySegment("a", 0, 2), TrajectorySegment("a", 2, SAMPLE_COUNT)),
    ],
)
def test_semantic_stages_cover_path_without_gaps_or_repeated_ids(
    segments: tuple,
) -> None:
    with pytest.raises(ValueError):
        _motion(segments=segments)


def test_declared_stage_events_follow_current_grid_and_are_fingerprinted() -> None:
    contract = SkillTrajectoryContract(
        "two_stage",
        "1",
        ("approach", "close"),
        ("grasp",),
        {"grasp": "close"},
    )
    short = (TrajectorySegment("approach", 0, 2), TrajectorySegment("close", 2, 5))
    long = (TrajectorySegment("approach", 0, 4), TrajectorySegment("close", 4, 9))
    assert contract.stage_landmarks(short, 5)["grasp"] == 4
    assert contract.stage_landmarks(long, 9)["grasp"] == 8
    descriptor = MoveJoints.descriptor()
    assert _canonical_value(descriptor) != _canonical_value(
        replace(
            descriptor,
            trajectory_contract=replace(descriptor.trajectory_contract, revision="2"),
        )
    )


def test_complete_external_move_plan_preserves_custom_binding_and_env_order() -> None:
    engine, request, context = _engine()
    engine.actions["move_joints"]._plan = Mock(
        side_effect=AssertionError("ordinary planning must not run")
    )
    plan = engine.build_plan_from_trajectory(request, context, _motion())
    assert plan.commands.frame_count == SAMPLE_COUNT
    assert plan.tracking is not None
    assert [target.target_id for target in plan.commands.targets] == ["custom_motion"]
    assert plan.commands.env_ids.tolist() == [7, 3]
    assert [stage.name for stage in plan.segments] == ["move_joints"]
    assert plan.expected_effects.is_empty and not plan.replannable
    assert plan.trajectory_input.landmarks["goal"] == SAMPLE_COUNT - 1


@pytest.mark.parametrize(
    "problem,error",
    [
        ("initial", "initial state"),
        ("position", "position limits"),
        ("velocity", "velocity limits"),
        ("reversal", "interval speeds"),
        ("endpoint", "authorized endpoints"),
        ("goal", "requested joint goal"),
        ("env_order", "ordered env_ids"),
        ("clock", "control grid"),
        ("long_clock", "control grid"),
        ("zero_interval", "positive after"),
    ],
)
def test_external_motion_admission_fails_before_execution(
    problem: str, error: str
) -> None:
    engine, request, context = _engine()
    motion = _motion()
    trajectory = motion.trajectory
    q = trajectory.positions.clone()
    if problem == "initial":
        q[:, 0, 0] = 0.005
    elif problem == "position":
        q[:, 2, 0] = 3.0
    elif problem == "velocity":
        trajectory = replace(trajectory, velocities=torch.full_like(q, 3.0))
    elif problem == "reversal":
        q[:, 2, 0] = 0.03
        trajectory = replace(trajectory, velocities=torch.zeros_like(q))
    elif problem == "endpoint":
        q[:, 1:, 2] = 0.005
    elif problem == "goal":
        q[:, -1, 0] = 0.008
    elif problem == "env_order":
        trajectory = replace(trajectory, env_ids=trajectory.env_ids.flip(0))
    elif problem == "clock":
        trajectory = replace(trajectory, dt=trajectory.dt * 1.5)
    elif problem == "long_clock":
        dt = trajectory.dt.clone()
        dt[:, 1] = 1000.0025  # A quarter tick must remain invalid for a long hold.
        trajectory = replace(trajectory, dt=dt)
    elif problem == "zero_interval":
        dt = trajectory.dt.clone()
        dt[:, 1] = 0
        trajectory = replace(trajectory, dt=dt, velocities=torch.zeros_like(q))
    motion = replace(motion, trajectory=replace(trajectory, positions=q))
    with pytest.raises(ValueError, match=error):
        engine.build_plan_from_trajectory(request, context, motion)


def test_move_checks_ordered_waypoints_and_failed_rows_remain_held() -> None:
    engine, request, context = _engine()
    goals = torch.tensor([[[0.0075, 0.0075], [0.0025, 0.0025], [0.01, 0.01]]]).expand(
        BATCH_SIZE, -1, -1
    )
    request = replace(request, goal=JointPositionGoal(goals))
    with pytest.raises(ValueError, match="waypoints in order"):
        engine.build_plan_from_trajectory(request, context, _motion())
    request = replace(
        request, goal=JointPositionGoal(torch.full((BATCH_SIZE, 2), 0.01))
    )
    motion = _motion(planning_success=torch.tensor([True, False]))
    plan = engine.build_plan_from_trajectory(request, context, motion)
    assert plan.plan_success.tolist() == [True, False]
    assert torch.count_nonzero(plan.joint_trajectory.positions[1]) == 0
    assert plan.trajectory_input.planning_success.tolist() == [True, False]


def test_move_accepts_ordered_polyline_waypoints_between_control_samples() -> None:
    engine, request, context = _engine()
    goals = torch.tensor([[[0.003, 0.003], [0.006, 0.006], [0.01, 0.01]]]).expand(
        BATCH_SIZE, -1, -1
    )
    plan = engine.build_plan_from_trajectory(
        replace(request, goal=JointPositionGoal(goals)), context, _motion()
    )
    assert plan.plan_success.all()


def test_same_grid_variant_is_reassembled_and_changed_grid_is_rejected() -> None:
    engine, request, context = _engine()
    motion = _motion()
    plan = engine.build_plan_from_trajectory(request, context, motion)
    plan = replace(
        plan,
        diagnostics=PlannerDiagnostics(
            "external_fixture", metadata={"candidate_lineage": "seed_7"}
        ),
    )
    positions = plan.joint_trajectory.positions.clone()
    positions[:, 2, 0] += 0.001
    candidate = replace(plan.joint_trajectory, positions=positions, velocities=None)
    rebuilt = engine.rebuild_plan_from_trajectory(request, context, plan, candidate)
    assert rebuilt.trajectory_input.landmarks == motion.landmarks
    assert rebuilt.diagnostics.backend == "external_fixture"
    assert rebuilt.diagnostics.metadata["candidate_lineage"] == "seed_7"
    assert rebuilt.commands.frame_count == SAMPLE_COUNT
    assert rebuilt.commands.frames[2].commands[0].payload.positions[
        :, 0
    ].tolist() == pytest.approx([0.006, 0.006])
    with pytest.raises(ValueError, match="identical dt"):
        engine.rebuild_plan_from_trajectory(
            request, context, plan, replace(candidate, dt=candidate.dt * 2)
        )


def test_contract_can_limit_batch_without_limiting_other_skills() -> None:
    contract = replace(MoveJoints.trajectory_contract, supports_batch=False)
    with pytest.raises(ValueError, match="B=1"):
        contract.validate_input(_motion())


def test_unsupported_skill_rejects_motion_without_planner_fallback() -> None:
    engine, request, context = _engine()
    action = Place()
    action._plan = Mock(side_effect=AssertionError("ordinary planning must not run"))
    engine.register(action)
    with pytest.raises(ValueError, match="does not support trajectory assembly"):
        engine.build_plan_from_trajectory(
            replace(request, skill_id="place"), context, _motion()
        )
    action._plan.assert_not_called()


def test_unimplemented_contract_and_framework_override_are_rejected() -> None:
    class MissingAssembler(AtomicAction):
        skill_id: ClassVar[str] = "missing_assembler"
        GoalType: ClassVar[type] = JointPositionGoal
        trajectory_contract: ClassVar[SkillTrajectoryContract] = (
            SkillTrajectoryContract("missing", "1")
        )

        def _plan(self, request, context) -> ActionPlan:
            raise AssertionError("planning must not run")

    engine, _, _ = _engine()
    with pytest.raises(ValueError, match="semantic assembler"):
        engine.register(MissingAssembler())
    with pytest.raises(TypeError, match="framework-owned"):

        class OverrideAssembler(MoveJoints):
            def assemble_trajectory_plan(self, request, context, motion) -> ActionPlan:
                raise AssertionError


def test_semantic_result_owns_scene_bounds() -> None:
    bounds = {"cube": 3}
    semantics = SkillPlanSemantics(scene_dependency_monitor_until=bounds)
    bounds["cube"] = 4
    assert semantics.scene_dependency_monitor_until["cube"] == 3
