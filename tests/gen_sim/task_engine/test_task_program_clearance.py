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

"""Released-hand clearance binds poses without taking over execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest.mock import Mock
import random

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.release_clearance import (
    _DEFAULT_RETREAT_DISTANCE,
    _ClearReleasedFactory,
    _ClearReleasedLowerer,
    _clearance_poses,
    _object_clearance_height,
)
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseGoal,
    JointPositionGoal,
    JointPositionTarget,
    MoveEndEffectorOptions,
    PlannerDiagnostics,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.task_program.semantics import RegisteredSemanticCall

__all__: list[str] = []


def test_clearance_raises_before_withdrawing_and_preserves_each_row() -> None:
    current = torch.eye(4).repeat(2, 1, 1)
    current[:, 2, 3] = torch.tensor([0.8, 1.4])
    root = torch.eye(4).repeat(2, 1, 1)
    root[:, 0, 3] = torch.tensor([1.0, -1.0])
    before = current.clone()
    poses = _clearance_poses(current, root, height=1.2, retreat=0.1)

    assert poses.shape == (2, 2, 4, 4)
    torch.testing.assert_close(
        poses[:, :, 2, 3], torch.tensor([[1.2, 1.2], [1.4, 1.4]])
    )
    torch.testing.assert_close(poses[:, 0, :2, 3], current[:, :2, 3])
    torch.testing.assert_close(poses[:, 1, 0, 3], torch.tensor([0.1, -0.1]))
    torch.testing.assert_close(
        poses[:, :, :3, :3], current[:, None, :3, :3].expand(-1, 2, -1, -1)
    )
    torch.testing.assert_close(current, before)


def _binding():
    endpoint = SimpleNamespace(
        task_state_key="left",
        runtime_target=SimpleNamespace(control_part="arm", joint_ids=(0, 1)),
    )
    return SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": endpoint,
                        "grasp": SimpleNamespace(
                            runtime_target=SimpleNamespace(target_id="hand")
                        ),
                    }
                )
            }
        )
    )


def test_clearance_rejects_a_live_attachment_before_kinematics() -> None:
    robot = Mock()
    lowerer = _ClearReleasedLowerer(
        (("can", "safe", 1.2),),
        robot,
        retreat_distance=0.1,
        object_vertices={"can": torch.zeros(1, 3)},
        gripper_models={},
    )
    context = SimpleNamespace(
        task=SimpleNamespace(
            get_held_object=lambda key: SimpleNamespace(
                active_mask=torch.tensor([True])
            )
        )
    )
    call = RegisteredSemanticCall(
        call_id="gen_sim.clear_released",
        arguments={"object": "can", "target": "safe"},
    )
    with pytest.raises(ValueError, match="cannot move a held object"):
        lowerer.lower(
            call,
            context=context,
            bound=_binding(),
            option_template=MoveEndEffectorOptions(),
        )
    assert robot.mock_calls == []


def test_clearance_binds_one_shared_cartesian_goal_without_planning() -> None:
    current = torch.eye(4).unsqueeze(0)
    root = current.clone()
    root[:, 0, 3] = 1.0
    robot = Mock()
    robot.compute_fk.return_value = current
    robot.get_link_pose.return_value = root
    robot.cfg.solver_cfg = {"arm": SimpleNamespace(root_link_name="arm_root")}
    lowerer = _ClearReleasedLowerer(
        (("can", "safe", 1.2),),
        robot,
        retreat_distance=0.1,
        object_vertices={"can": torch.zeros(1, 3)},
        gripper_models={"hand": _model()},
    )
    context = SimpleNamespace(
        task=SimpleNamespace(get_held_object=lambda key: None),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        env_ids=torch.tensor([0]),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=current.clone(), confidence=1.0)}
        ),
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.clear_released",
            arguments={"object": "can", "target": "safe"},
        ),
        context=context,
        bound=_binding(),
        option_template=MoveEndEffectorOptions(),
    )
    assert type(result.goal) is EndEffectorPoseGoal
    assert result.goal.xpos.shape == (1, 2, 4, 4)
    assert result.registered_effect is None
    robot.compute_ik.assert_not_called()
    robot.plan.assert_not_called()
    robot.step.assert_not_called()


def test_clearance_factory_requires_the_same_robot() -> None:
    factory = _ClearReleasedFactory((("can", "safe", 1.2),))
    with pytest.raises(ValueError, match="factory's robot"):
        factory.create(
            simulation=None,
            robot=object(),
            scene_registry=Mock(),
            engine=SimpleNamespace(robot=object()),
        )


def test_clearance_default_retreat_preserves_release_separation_margin() -> None:
    factory = _ClearReleasedFactory((("can", "safe", 1.2),))
    assert factory.retreat_distance == pytest.approx(_DEFAULT_RETREAT_DISTANCE)


def _model():
    return SimpleNamespace(
        max_opening_width=0.14,
        finger_thickness=0.01,
        finger_width=0.03,
        finger_length=0.13,
        palm_depth=0.08,
    )


def test_clearance_height_accounts_for_object_top_and_gripper_orientation() -> None:
    current = torch.eye(4).repeat(2, 1, 1)
    current[:, 2, 3] = 0.895
    current[1, 1, 1] = current[1, 2, 2] = -1
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, 2, 3] = 0.682
    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.222]])
    height = _object_clearance_height(current, object_pose, vertices, _model())
    torch.testing.assert_close(height, torch.tensor([1.059, 0.979]))
    poses = _clearance_poses(
        current, torch.eye(4).repeat(2, 1, 1), height=height, retreat=0.15
    )
    torch.testing.assert_close(poses[:, 0, :2, 3], current[:, :2, 3])
    torch.testing.assert_close(poses[:, 0, 2, 3], height)


def _cleanup_program(calls):
    segments = [
        SimpleNamespace(
            name=f"step_{index}",
            segment_id=f"segment-{index}",
            parallel_block=None,
            calls=(SimpleNamespace(call=call, segment_call_index=0),),
        )
        for index, call in enumerate(calls)
    ]
    return SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))


def _cleanup_call(call_id, resource="left_arm"):
    return RegisteredSemanticCall(
        call_id=call_id,
        arguments=(
            {"object": "can", "target": "safe"}
            if call_id
            in {
                "gen_sim.clear_released",
                "simulation.place_relative",
                "gen_sim.place_upright",
                "gen_sim.stack_place",
                "unknown.release",
            }
            else {}
        ),
        resources={} if resource is None else {"primary": resource},
    )


@pytest.mark.parametrize(
    "release_call_id",
    ["simulation.place_relative", "gen_sim.place_upright", "gen_sim.stack_place"],
)
def test_cleanup_recovery_binds_only_exact_adjacent_clearance_and_owning_park(
    release_call_id,
):
    from embodichain.gen_sim.task_engine._task_program.clearance_recovery import (
        bind_clearance_calls,
    )

    park = _cleanup_call("simulation.park")
    calls = [
        _cleanup_call(release_call_id),
        _cleanup_call("gen_sim.clear_released"),
        park,
    ]
    bindings = bind_clearance_calls(_cleanup_program(calls))
    assert len(bindings) == 1
    assert bindings[0].clearance_invocation_id == "p/segment-1:0"
    assert bindings[0].park_invocation_ids == ("p/segment-2:0",)
    assert bindings[0].park_calls == (park,)
    assert bindings[0].park_calls[0].resources == {"primary": "left_arm"}
    assert bindings[0].release_call_id == release_call_id


@pytest.mark.parametrize(
    "calls",
    [
        [_cleanup_call("gen_sim.clear_released")],
        [
            _cleanup_call("gen_sim.clear_released"),
            _cleanup_call("ordinary.move_end_effector"),
            _cleanup_call("simulation.park"),
        ],
        [
            _cleanup_call("gen_sim.clear_released"),
            _cleanup_call("simulation.park", "right_arm"),
        ],
        [
            _cleanup_call("gen_sim.clear_released", None),
            _cleanup_call("simulation.park", None),
        ],
    ],
)
def test_cleanup_binding_rejects_missing_park_or_implicit_cross_arm_resources(calls):
    from embodichain.gen_sim.task_engine._task_program.clearance_recovery import (
        bind_clearance_calls,
    )

    with pytest.raises(ValueError):
        bind_clearance_calls(_cleanup_program(calls))


def test_cleanup_binding_does_not_enable_ordinary_or_unrelated_park_calls():
    from embodichain.gen_sim.task_engine._task_program.clearance_recovery import (
        bind_clearance_calls,
    )

    calls = [
        _cleanup_call("ordinary.move_end_effector"),
        _cleanup_call("simulation.park"),
    ]
    assert bind_clearance_calls(_cleanup_program(calls)) == ()


@pytest.mark.parametrize(
    "release", [None, "other_object", "other_arm", "not_release", "unknown_release"]
)
def test_cleanup_binding_requires_preceding_same_object_and_resource_release(release):
    from embodichain.gen_sim.task_engine._task_program.clearance_recovery import (
        bind_clearance_calls,
    )

    calls = [_cleanup_call("gen_sim.clear_released"), _cleanup_call("simulation.park")]
    if release is not None:
        preceding = _cleanup_call("simulation.place_relative")
        if release == "other_object":
            preceding = replace(
                preceding, arguments={"object": "different", "target": "safe"}
            )
        elif release == "other_arm":
            preceding = replace(preceding, resources={"primary": "right_arm"})
        elif release == "unknown_release":
            preceding = _cleanup_call("unknown.release")
        else:
            preceding = _cleanup_call("ordinary.move_end_effector")
        calls.insert(0, preceding)
    with pytest.raises(ValueError, match="preceding same-object release"):
        bind_clearance_calls(_cleanup_program(calls))


@dataclass(frozen=True)
class _CleanupPlan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: PlannerDiagnostics


@dataclass(frozen=True)
class _CleanupObservation:
    qpos: torch.Tensor
    qvel: torch.Tensor
    timestamp: float = 1.0


@dataclass(frozen=True)
class _CleanupContext:
    robot: _CleanupObservation
    task: object
    env_ids: torch.Tensor
    batch_size: int = 1
    control_dt: float = 0.04
    scene: object = None

    def require_control_dt(self):
        return self.control_dt


@pytest.fixture
def cleanup_gate_fixture(tmp_path):
    robot_file = tmp_path / "cleanup.urdf"
    robot_file.write_text(
        '<robot><joint name="arm"><limit velocity="1" lower="-1" upper="1"/>'
        '</joint><joint name="other"><limit velocity="1" lower="-1" upper="1"/>'
        "</joint></robot>"
    )
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(robot_file)),
        joint_names=["arm", "other"],
        get_joint_ids=lambda **kwargs: [0] if kwargs.get("name") else [0, 1],
        get_qvel_limits=lambda **kwargs: torch.full(
            (1, 1 if kwargs.get("name") else 2), 10.0
        ),
        get_qpos_limits=lambda **kwargs: limits[
            :, list(kwargs.get("joint_ids", [0] if kwargs.get("name") else [0, 1]))
        ].clone(),
    )
    context = _CleanupContext(
        robot=_CleanupObservation(torch.zeros(1, 2), torch.zeros(1, 2)),
        task=SimpleNamespace(get_held_object=lambda key: None),
        env_ids=torch.tensor([0]),
    )
    plan = _CleanupPlan(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]]]),
            dt=torch.tensor([[0.0, 0.04, 0.04]]),
        ),
        diagnostics=PlannerDiagnostics(backend="fixture"),
    )
    return plan, context, robot


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "start_velocity",
        "internal_velocity",
        "other_arm_velocity",
        "joint_limit",
        "nonfinite",
        "collision",
        "missing_collision_dependency",
        "empty_trajectory",
        "missing_trajectory",
        "plan_failed",
        "wrong_env",
        "batch",
    ],
)
def test_cleanup_motion_acceptance_uses_full_robot_limits_and_fails_closed(
    monkeypatch, cleanup_gate_fixture, failure
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    plan, context, robot = cleanup_gate_fixture
    positions = plan.joint_trajectory.positions.clone()
    if failure == "start_velocity":
        positions[:] += 0.2
    elif failure == "internal_velocity":
        positions[0, 1, 0] = 0.2
    elif failure == "other_arm_velocity":
        positions[0, 1, 1] = 0.2
    elif failure == "joint_limit":
        positions[:] = 1.001
        context = replace(
            context,
            robot=replace(context.robot, qpos=torch.full((1, 2), 1.001)),
        )
    elif failure == "nonfinite":
        positions[0, 1, 0] = float("nan")
    elif failure == "wrong_env":
        context = replace(context, env_ids=torch.tensor([1]))
    elif failure == "batch":
        context = replace(context, batch_size=2)
    if failure == "empty_trajectory":
        trajectory = SimpleNamespace(
            positions=torch.zeros(1, 0, 2), dt=torch.zeros(1, 0)
        )
    elif failure == "missing_trajectory":
        trajectory = None
    else:
        trajectory = SimpleNamespace(positions=positions, dt=plan.joint_trajectory.dt)
    plan = replace(
        plan,
        joint_trajectory=trajectory,
        plan_success=torch.tensor([failure != "plan_failed"]),
    )
    seen = []

    def check_collision(actual_robot, simulation, actual_positions, control_part):
        assert actual_robot is robot and control_part == "left_arm"
        seen.append(actual_positions.clone())
        if failure == "missing_collision_dependency":
            raise ImportError("FCL unavailable")
        return {"valid": failure != "collision"}

    monkeypatch.setattr(recovery, "check_free_motion", check_collision)
    result = recovery.validate_motion(plan, context, robot, object(), "left_arm")
    assert result["valid"] is (failure is None)
    if failure is None:
        assert len(seen) == 1
        torch.testing.assert_close(
            seen[0], torch.cat((context.robot.qpos, positions[0]), dim=0)
        )
    assert robot.get_qvel_limits().tolist() == [[10.0, 10.0]]
    assert robot.get_qpos_limits().tolist() == [[[-1.0, 1.0], [-1.0, 1.0]]]


@pytest.mark.parametrize("changed", [False, True])
def test_cleanup_holds_passive_joint_noise_without_issuing_a_passive_command(
    monkeypatch, cleanup_gate_fixture, changed
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    plan, context, robot = cleanup_gate_fixture
    # A passive joint's measured limit noise is an observation, not a new command.
    observed = context.robot.qpos.clone()
    observed[0, 1] = -1.0001
    context = replace(context, robot=replace(context.robot, qpos=observed))
    positions = plan.joint_trajectory.positions.clone()
    positions[0, :, 1] = observed[0, 1]
    if changed:
        positions[0, 1, 1] += 0.001
    plan = replace(
        plan,
        joint_trajectory=SimpleNamespace(
            positions=positions, dt=plan.joint_trajectory.dt
        ),
    )
    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    acceptance = recovery.validate_motion(plan, context, robot, object(), "left_arm")
    assert acceptance["valid"] is not changed
    torch.testing.assert_close(context.robot.qpos, observed)


@dataclass(frozen=True)
class _CleanupRequest:
    goal: object
    binding: object
    invocation_id: str
    motion_policy: MotionPolicy = field(
        default_factory=lambda: MotionPolicy(sample_count=260)
    )
    skill_id: str = "move_end_effector"


@pytest.fixture
def cleanup_recovery_fixture(cleanup_gate_fixture):
    plan, context, robot = cleanup_gate_fixture
    motion = JointPositionTarget(control_part="left_arm", joint_ids=(0,))
    endpoint = SimpleNamespace(
        task_state_key="left_arm", require_target=lambda cls: motion
    )
    binding = SimpleNamespace(endpoint=lambda role, channel: endpoint)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, :, 2, 3] = 1.0
    poses[0, 1, 0, 3] = 0.15
    request = _CleanupRequest(EndEffectorPoseGoal(poses), binding, "clearance")
    park_request = _CleanupRequest(
        JointPositionGoal(torch.zeros(1, 1)), binding, "park", skill_id="move_joints"
    )
    return request, park_request, plan, context, robot


def _park_plan_from(predicted, *, boundary_jump=False):
    start = predicted.robot.qpos.clone()
    if boundary_jump:
        start[0, 0] += 0.2
    return _CleanupPlan(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.stack((start, start), dim=1),
            dt=torch.tensor([[0.0, 0.04]]),
        ),
        diagnostics=PlannerDiagnostics(backend="fixture"),
    )


def _two_arm_park_requests(cleanup_recovery_fixture):
    request, park, plan, context, robot = cleanup_recovery_fixture
    source_motion = JointPositionTarget(control_part="right_arm", joint_ids=(1,))
    source_endpoint = SimpleNamespace(
        task_state_key="right_arm", require_target=lambda cls: source_motion
    )
    source_park = replace(
        park,
        binding=SimpleNamespace(endpoint=lambda role, channel: source_endpoint),
    )
    receiver_park = replace(park, invocation_id="receiver_park")
    robot.get_joint_ids = lambda **kwargs: (
        [0, 1]
        if not kwargs.get("name")
        else [0] if kwargs["name"] == "left_arm" else [1]
    )
    robot.get_qpos_limits = lambda **kwargs: torch.tensor([[[-1.0, 1.0]]])
    return request, (source_park, receiver_park), plan, context, robot


def _moving_park_plan_from(predicted, joint):
    start = predicted.robot.qpos.clone()
    end = start.clone()
    end[:, joint] += 0.01
    return _CleanupPlan(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.stack((start, end), dim=1),
            dt=torch.tensor([[0.0, 0.04]]),
        ),
        diagnostics=PlannerDiagnostics(backend="fixture"),
    )


def test_cleanup_previews_both_parks_from_sequential_predicted_full_robot_qpos(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, parks, original, context, robot = _two_arm_park_requests(
        cleanup_recovery_fixture
    )
    built = []
    checked = []

    def build(selected, predicted):
        assert selected is parks[0] or selected is parks[1]
        index = 1 if selected is parks[0] else 0
        built.append((selected.invocation_id, predicted.robot.qpos.clone()))
        return _moving_park_plan_from(predicted, index)

    def collision(_robot, _simulation, positions, part):
        checked.append((part, positions.clone()))
        return {"valid": True}

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=parks,
        robot=robot,
        simulation=object(),
    )

    assert result.plan_success.all()
    assert [item[0] for item in built] == ["park", "receiver_park"]
    torch.testing.assert_close(built[0][1], torch.tensor([[0.02, 0.0]]))
    torch.testing.assert_close(built[1][1], torch.tensor([[0.02, 0.01]]))
    assert [item[0] for item in checked] == ["left_arm", "right_arm", "left_arm"]
    torch.testing.assert_close(checked[1][1][0], built[0][1][0])
    torch.testing.assert_close(checked[2][1][0], built[1][1][0])
    torch.testing.assert_close(checked[2][1][:, 1], torch.full((3,), 0.01))
    torch.testing.assert_close(context.robot.qpos, torch.zeros(1, 2))


def test_valid_original_cleanup_and_park_are_qualified_without_search(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    built = []
    collision_inputs = []

    def build(selected, predicted):
        assert selected is park_request
        built.append(selected.invocation_id)
        torch.testing.assert_close(
            predicted.robot.qpos, original.joint_trajectory.positions[:, -1]
        )
        assert torch.equal(predicted.robot.qvel, torch.zeros_like(predicted.robot.qvel))
        return _park_plan_from(predicted)

    def collision(_robot, _sim, positions, part):
        assert part == "left_arm"
        collision_inputs.append(positions.clone())
        return {"valid": True}

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )
    assert built == ["park"]
    assert result.plan_success.tolist() == [True]
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    assert report["selected_route"] == "original"
    assert len(report["attempts"]) == 1
    assert len(collision_inputs) == 2
    assert collision_inputs[0].shape == (4, 2)
    assert collision_inputs[1].shape == (3, 2)
    torch.testing.assert_close(
        collision_inputs[1][0], original.joint_trajectory.positions[0, -1]
    )


def test_cleanup_initial_state_collision_stops_before_fallback_search(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    built = []
    collision_inputs = []
    evidence = {"valid": False, "frame": 0, "pair": ["arm_1", "arm_5"]}

    def build(selected, predicted):
        built.append(selected.invocation_id)
        return _park_plan_from(predicted) if selected is park_request else original

    def collision(_robot, _simulation, positions, part):
        assert part == "left_arm"
        collision_inputs.append(positions.clone())
        torch.testing.assert_close(positions[0], context.robot.qpos[0])
        return evidence

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )

    assert built == []
    assert len(collision_inputs) == 1
    assert not result.plan_success.any()
    assert original.plan_success.all()
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert result.diagnostics.failure.retryable is False
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    assert report["accepted"] is False
    assert report["reason"] == "initial_state_collision"
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["reason"] == "initial_state_collision"
    assert report["attempts"][0]["collision"] == evidence


def test_cleanup_park_initial_frame_collision_can_use_another_clearance_route(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    built = []
    collisions = []

    def build(selected, predicted):
        built.append(selected.invocation_id)
        return _park_plan_from(predicted) if selected is park_request else original

    def collision(_robot, _simulation, positions, part):
        assert part == "left_arm"
        collisions.append(positions.clone())
        return {"valid": len(collisions) != 2, "frame": 0}

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )

    assert built == ["park", "clearance", "park"]
    assert result.plan_success.all()
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    assert report["accepted"] is True
    assert report["selected_route"] == "retreat_0.05_extra_height_0.00"
    assert report.get("reason") != "initial_state_collision"
    assert report["attempts"][0]["failed_stage"] == "park"
    assert report["attempts"][0].get("reason") != "initial_state_collision"


def test_cleanup_candidate_initial_state_collision_stops_remaining_routes(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    original = replace(good, plan_success=torch.tensor([False]))
    built = []
    evidence = {"valid": False, "frame": 0, "pair": ["arm_1", "arm_5"]}

    def build(selected, predicted):
        built.append(selected.invocation_id)
        return _park_plan_from(predicted) if selected is park_request else good

    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: evidence)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )

    assert built == ["clearance"]
    assert not result.plan_success.any()
    assert result.diagnostics.failure.retryable is False
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    assert report["reason"] == "initial_state_collision"
    assert [attempt["route"] for attempt in report["attempts"]] == [
        "original",
        "retreat_0.05_extra_height_0.00",
    ]
    assert report["attempts"][-1]["reason"] == "initial_state_collision"
    assert report["attempts"][-1]["collision"] == evidence


@pytest.mark.parametrize("collision_result", [{}, {"valid": False}, None])
def test_cleanup_missing_collision_acceptance_remains_fail_closed(
    monkeypatch, cleanup_recovery_fixture, collision_result
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    built = []

    def build(selected, predicted):
        built.append(selected.invocation_id)
        return _park_plan_from(predicted) if selected is park_request else original

    def collision(*args):
        if collision_result is None:
            raise ImportError("FCL unavailable")
        return collision_result

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )

    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert result.diagnostics.failure.retryable is False
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    assert not report["accepted"]
    assert "initial_state_collision" not in {
        attempt.get("reason") for attempt in report["attempts"]
    }
    if collision_result is None:
        assert built == []
        assert report["attempts"][0]["reason"] == "collision_check_unavailable"


@pytest.mark.parametrize("held,cross_arm", [(True, False), (False, True)])
def test_cleanup_fallback_rejects_held_object_or_cross_arm_successor_before_planning(
    monkeypatch, cleanup_recovery_fixture, held, cross_arm
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    if held:
        context = replace(
            context,
            task=SimpleNamespace(
                get_held_object=lambda key: SimpleNamespace(
                    active_mask=torch.tensor([True])
                )
            ),
        )
    if cross_arm:
        motion = JointPositionTarget(control_part="right_arm", joint_ids=(1,))
        endpoint = SimpleNamespace(require_target=lambda cls: motion)
        park_request = replace(
            park_request,
            binding=SimpleNamespace(endpoint=lambda role, channel: endpoint),
        )
    monkeypatch.setattr(
        recovery,
        "check_free_motion",
        lambda *args: pytest.fail("unexpected collision query"),
    )
    result = recovery.recover(
        request,
        context,
        original,
        build=lambda *_: pytest.fail("unexpected preview or fallback"),
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )
    assert result.plan_success.tolist() == [False]
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert result.diagnostics.failure.retryable is False


@pytest.mark.parametrize(
    "failure",
    [
        "clearance_velocity",
        "clearance_limit",
        "handoff_velocity",
        "clearance_collision",
        "park_collision",
    ],
)
def test_combined_cleanup_rejects_unsafe_candidates_with_a_bounded_fail_closed_search(
    monkeypatch, cleanup_recovery_fixture, failure
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, original, context, robot = cleanup_recovery_fixture
    built = []
    start_goal = request.goal.xpos.clone()
    if failure in {"clearance_velocity", "clearance_limit"}:
        positions = original.joint_trajectory.positions.clone()
        if failure == "clearance_velocity":
            positions[0, 1, 0] = 0.2
        else:
            positions[:] = 1.001
        original = replace(
            original,
            joint_trajectory=SimpleNamespace(
                positions=positions, dt=original.joint_trajectory.dt
            ),
        )

    def build(selected, predicted):
        built.append(selected)
        if selected is park_request:
            return _park_plan_from(
                predicted, boundary_jump=failure == "handoff_velocity"
            )
        assert selected.invocation_id == request.invocation_id
        assert selected.binding is request.binding
        assert selected.motion_policy is request.motion_policy
        assert selected.goal.xpos.shape == request.goal.xpos.shape
        torch.testing.assert_close(
            selected.goal.xpos[:, :, :3, :3], start_goal[:, :, :3, :3]
        )
        torch.testing.assert_close(
            selected.goal.xpos[:, 0, :2, 3], start_goal[:, 0, :2, 3]
        )
        assert (selected.goal.xpos[:, :, 2, 3] >= start_goal[:, :, 2, 3]).all()
        retreat = torch.linalg.vector_norm(
            selected.goal.xpos[:, 1, :2, 3] - selected.goal.xpos[:, 0, :2, 3], dim=-1
        )
        assert retreat.item() <= 0.05 + 1e-6
        return original

    def collision(_robot, _simulation, positions, part):
        assert part == "left_arm"
        if failure == "clearance_collision":
            return {"valid": False, "frame": 1}
        if failure == "park_collision":
            return {"valid": len(positions) != 3, "frame": len(positions) - 1}
        return {"valid": True}

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )
    assert not result.plan_success.any()
    assert original.plan_success.all()
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert result.diagnostics.failure.retryable is False
    report = result.diagnostics.metadata["gen_sim_clearance_recovery"]
    expected_attempts = 7 if failure == "park_collision" else 3
    assert len(report["attempts"]) == expected_attempts
    assert len(built) <= 2 * expected_attempts - 1
    assert not report["accepted"]
    torch.testing.assert_close(request.goal.xpos, start_goal)


@pytest.mark.parametrize("accepted_route", ["original", "short", None])
def test_cleanup_preview_and_fallback_restore_all_random_streams_and_input_state(
    monkeypatch, cleanup_recovery_fixture, accepted_route
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    original = (
        good
        if accepted_route == "original"
        else replace(good, plan_success=torch.tensor([False]))
    )
    before_goal, before_qpos = request.goal.xpos.clone(), context.robot.qpos.clone()
    built = []

    def build(selected, predicted):
        torch.rand(3)
        random.random()
        np.random.rand(2)
        built.append(selected.invocation_id)
        if selected is park_request:
            return _park_plan_from(predicted)
        return good if accepted_route == "short" else original

    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    torch_state, python_state, numpy_state = (
        torch.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(request.goal.xpos, before_goal)
    torch.testing.assert_close(context.robot.qpos, before_qpos)
    assert result.plan_success.tolist() == [accepted_route is not None]
    assert len(built) <= 5


def _cleanup_engine(robot, simulation, request, park_request):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {}
    engine._cartesian_calls = frozenset({request.invocation_id})
    engine._articulation_calls = {}
    engine._clearance_requests = {}
    engine._planning_services = SimpleNamespace(robot=robot)
    engine.bind_release_clearance(
        {request.invocation_id: park_request}, simulation=simulation
    )
    return engine


def test_cleanup_engine_checks_the_transformed_plan_not_the_raw_planner_output(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    engine = _cleanup_engine(robot, object(), request, (park_request,))
    transformed = []

    def transform(selected, current, raw):
        transformed.append(selected.invocation_id)
        if selected.invocation_id == "park":
            return raw
        positions = raw.joint_trajectory.positions.clone()
        positions[0, 1, 0] = 0.2
        return replace(
            raw,
            joint_trajectory=SimpleNamespace(
                positions=positions, dt=raw.joint_trajectory.dt
            ),
        )

    def raw_plan(_engine, selected, current, *, plan_transform=None):
        raw = _park_plan_from(current) if selected.invocation_id == "park" else good
        if selected.invocation_id == "park":
            assert plan_transform is None
            return raw
        assert plan_transform is transform
        return plan_transform(selected, current, raw)

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", raw_plan)
    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    result = engine._plan_request(request, context, plan_transform=transform)
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert transformed.count("clearance") == 3
    assert all(
        not attempt["velocity"]["valid"]
        for attempt in result.diagnostics.metadata["gen_sim_clearance_recovery"][
            "attempts"
        ]
    )
    assert good.joint_trajectory.positions[0, 1, 0] == pytest.approx(0.01)


def test_cleanup_actual_park_revalidates_latest_observation_and_scene(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    simulation = SimpleNamespace(obstacle=False)
    engine = _cleanup_engine(robot, simulation, request, (park_request,))
    observed_contexts = []
    collision_count = 0

    def raw_plan(_engine, selected, current, *, plan_transform=None):
        observed_contexts.append(current)
        return _park_plan_from(current) if selected.invocation_id == "park" else good

    def collision(_robot, actual_simulation, positions, part):
        nonlocal collision_count
        assert actual_simulation is simulation
        collision_count += 1
        return {"valid": not simulation.obstacle, "frame": len(positions) - 1}

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", raw_plan)
    monkeypatch.setattr(recovery, "check_free_motion", collision)
    preview = engine._plan_request(request, context)
    assert preview.plan_success.all()
    assert collision_count == 2
    latest = replace(
        context,
        robot=replace(context.robot, qpos=torch.tensor([[0.019, 0.0]]), timestamp=2.0),
        scene=SimpleNamespace(timestamp=2.0, version=2),
    )
    simulation.obstacle = True
    real_park = engine._plan_request(park_request, latest)
    assert observed_contexts[-1] is latest
    assert collision_count == 3
    assert not real_park.plan_success.any()
    assert real_park.diagnostics.failure.code == "cleanup_park_failed"
    assert real_park.diagnostics.failure.retryable is False
    assert not real_park.diagnostics.metadata["gen_sim_clearance_park"]["valid"]


def test_cleanup_each_park_revalidates_the_latest_other_arm_and_scene(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, parks, original, context, robot = _two_arm_park_requests(
        cleanup_recovery_fixture
    )
    simulation = SimpleNamespace(obstacle=False)
    engine = _cleanup_engine(robot, simulation, request, parks)
    observations = []
    checks = []

    def raw_plan(_engine, selected, current, *, plan_transform=None):
        observations.append((selected.invocation_id, current))
        if selected is request:
            return original
        return _moving_park_plan_from(current, 1 if selected is parks[0] else 0)

    def collision(_robot, actual_simulation, positions, part):
        assert actual_simulation is simulation
        checks.append((part, positions.clone()))
        return {"valid": not simulation.obstacle, "frame": len(positions) - 1}

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", raw_plan)
    monkeypatch.setattr(recovery, "check_free_motion", collision)
    preview = engine._plan_request(request, context)
    assert preview.plan_success.all()
    assert len(checks) == 3

    latest_source = replace(
        context,
        robot=replace(context.robot, qpos=torch.tensor([[0.019, 0.0]]), timestamp=2.0),
        scene=SimpleNamespace(timestamp=2.0, version=2),
    )
    source_park = engine._plan_request(parks[0], latest_source)
    assert source_park.plan_success.all()
    assert observations[-1][1] is latest_source
    assert len(checks) == 4
    latest_receiver = replace(
        latest_source,
        robot=replace(
            latest_source.robot, qpos=torch.tensor([[0.018, 0.009]]), timestamp=3.0
        ),
        scene=SimpleNamespace(timestamp=3.0, version=3),
    )
    simulation.obstacle = True
    receiver_park = engine._plan_request(parks[1], latest_receiver)
    assert observations[-1][1] is latest_receiver
    assert len(checks) == 5
    assert checks[-1][0] == "left_arm"
    torch.testing.assert_close(checks[-1][1][:, 1], torch.full((3,), 0.009))
    assert not receiver_park.plan_success.any()
    assert receiver_park.diagnostics.failure.code == "cleanup_park_failed"
    assert receiver_park.diagnostics.failure.retryable is False


def test_cleanup_scope_does_not_change_an_unbound_gensim_or_shared_action(
    monkeypatch, cleanup_recovery_fixture
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    engine = _cleanup_engine(robot, object(), request, (park_request,))
    ordinary = replace(request, invocation_id="ordinary")
    seen = []

    def raw_plan(_engine, selected, current, *, plan_transform=None):
        seen.append(selected.invocation_id)
        return good

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", raw_plan)
    monkeypatch.setattr(
        recovery, "check_free_motion", lambda *args: pytest.fail("unbound safety query")
    )
    actual = engine._plan_request(ordinary, context)
    shared = AtomicActionEngine._plan_request(engine, ordinary, context)
    assert seen == ["ordinary", "ordinary"]
    torch.testing.assert_close(
        actual.joint_trajectory.positions, shared.joint_trajectory.positions
    )
    assert actual.plan_success.tolist() == shared.plan_success.tolist()
    assert "gen_sim_clearance_recovery" not in actual.diagnostics.metadata
    assert "gen_sim_clearance_park" not in actual.diagnostics.metadata


@pytest.mark.parametrize(
    "release_call_id",
    ["simulation.place_relative", "gen_sim.place_upright", "gen_sim.stack_place"],
)
@pytest.mark.parametrize("verified,has_monitor", [(False, True), (True, False)])
def test_cleanup_park_preflight_requires_verified_release_monitor(
    verified, has_monitor, release_call_id
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )
    from embodichain.lab.task_program.semantics import EffectAssurance

    park = _cleanup_call("simulation.park")
    release = _cleanup_call(release_call_id)
    route = recovery.ClearanceRoute("clearance", ("park",), (park,), release)
    compiler = SimpleNamespace(
        integration=SimpleNamespace(
            link_call=Mock(
                return_value=SimpleNamespace(
                    preset=SimpleNamespace(
                        effect_assurance=(
                            EffectAssurance.VERIFIED
                            if verified
                            else EffectAssurance.PROJECTED
                        ),
                        effect_monitors=(
                            {release_call_id: object()}
                            if has_monitor
                            else {
                                (
                                    "gen_sim.place_upright"
                                    if release_call_id == "simulation.place_relative"
                                    else "simulation.place_relative"
                                ): object()
                            }
                        ),
                    )
                )
            )
        ),
        analyze=Mock(),
    )
    with pytest.raises(ValueError, match="verified preceding release effects"):
        recovery.bind_park_requests(compiler, (route,))
    compiler.integration.link_call.assert_called_once_with(release)
    compiler.analyze.assert_not_called()


@pytest.mark.parametrize(
    "release_call_id",
    ["simulation.place_relative", "gen_sim.place_upright", "gen_sim.stack_place"],
)
def test_cleanup_park_preflight_uses_the_actual_verified_release_alias(
    release_call_id,
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )
    from embodichain.lab.sim.atomic_actions import (
        ActionBinding,
        ActionInvocation,
        RecoveryPolicy,
        TrackingPolicy,
    )
    from embodichain.lab.task_program.semantics import EffectAssurance

    route = recovery.bind_clearance_calls(
        _cleanup_program(
            [
                _cleanup_call(release_call_id),
                _cleanup_call("gen_sim.clear_released"),
                _cleanup_call("simulation.park"),
            ]
        )
    )[0]
    policies = SimpleNamespace(
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors={release_call_id: object()},
        motion_policy=MotionPolicy(),
        tracking_policy=TrackingPolicy.joint_position(),
        recovery_policy=RecoveryPolicy(),
        action_option_template=lambda call_id: None,
    )
    engine = Mock()
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(torch.zeros(1, 1)),
        binding=ActionBinding("test"),
        motion_policy=policies.motion_policy,
        tracking_policy=policies.tracking_policy,
        recovery_policy=policies.recovery_policy,
        invocation_id="compiler_owned:0",
    )
    compiler = SimpleNamespace(
        integration=SimpleNamespace(
            link_call=Mock(return_value=SimpleNamespace(preset=policies)),
            engine=engine,
        ),
        analyze=Mock(return_value=object()),
        ground=Mock(return_value=SimpleNamespace(invocation=invocation)),
    )
    requests = recovery.bind_park_requests(compiler, (route,))
    compiler.integration.link_call.assert_called_once_with(route.release_call)
    compiler.analyze.assert_called_once_with(
        route.park_calls, workflow_id="p/segment-1:0/parks"
    )
    engine.resolve.assert_not_called()
    observed = object()
    preview = requests["p/segment-1:0"][0]
    assert preview.resolve(observed) is engine.resolve.return_value
    compiler.ground.assert_called_once_with(compiler.analyze.return_value, 0, observed)
    invocation = engine.resolve.call_args.args[0]
    assert invocation.invocation_id == "p/segment-2:0"
    assert invocation.skill_id == "move_joints"
    assert invocation.motion_policy is policies.motion_policy


def test_cleanup_factory_binds_canonical_preview_without_rebuilding_lowerers(
    monkeypatch,
):
    from embodichain.gen_sim.task_engine._task_program import assembly
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    from embodichain.lab.task_program.runtime import assembly as runtime_assembly

    profile, registry = object(), object()
    engine = Mock()
    engine._clearance_profile = profile
    semantic = SimpleNamespace(compiler=object())
    factory = object.__new__(assembly._TaskFactory)
    factory._clearance_routes = (object(),)
    factory._alignment_continuation_routes = ()
    factory._simulation = object()
    factory._runtime_preset = "actual_runtime"
    factory._robot_profile_binding = SimpleNamespace(profile_id="robot")
    factory._scene_binding = SimpleNamespace(registry_id="scene")
    factory._registration = SimpleNamespace(
        call_catalog=object(),
        endpoint_adapter_map={"endpoint": object()},
        relation_grounders=(object(),),
        handover_pose_providers=(object(),),
    )
    lowerers = (object(),)
    shared = Mock(return_value=lowerers)
    canonical = Mock(return_value=semantic)
    binding = Mock(return_value={"clearance": (object(),)})
    monkeypatch.setattr(
        assembly.SimulationTaskProgramFactory,
        "create_registered_semantic_lowerers",
        shared,
    )
    monkeypatch.setattr(
        runtime_assembly, "assemble_semantic_executor_components", canonical
    )
    monkeypatch.setattr(recovery, "bind_park_requests", binding)

    assert (
        factory.create_registered_semantic_lowerers(
            scene_registry=registry, engine=engine
        )
        is lowerers
    )
    shared.assert_called_once_with(scene_registry=registry, engine=engine)
    arguments, options = canonical.call_args
    assert arguments[:3] == (registry, profile, engine)
    assert arguments[3].robot_profile == "robot"
    assert arguments[3].scene_registry == "scene"
    assert arguments[3].runtime_preset == "actual_runtime"
    assert options["registered_lowerers"] is lowerers
    assert options["call_catalog"] is factory._registration.call_catalog
    assert options["endpoint_adapters"] is factory._registration.endpoint_adapter_map
    assert options["relation_grounders"] is factory._registration.relation_grounders
    assert (
        options["handover_pose_providers"]
        is factory._registration.handover_pose_providers
    )
    binding.assert_called_once_with(semantic.compiler, factory._clearance_routes)
    engine.bind_release_clearance.assert_called_once_with(
        binding.return_value, simulation=factory._simulation
    )
    assert "_clearance_profile" not in vars(engine)


@pytest.mark.parametrize("has_clearance", [False, True])
def test_cleanup_factory_preserves_the_exact_adapter_type_required_by_embodied_env(
    monkeypatch, has_clearance
):
    from embodichain.gen_sim.task_engine._task_program import assembly

    factory = object.__new__(assembly._TaskFactory)
    factory._clearance_routes = (object(),) if has_clearance else ()
    factory._step_dt = 0.04
    factory._joint_command_mode = object()
    factory._registration = object()
    monkeypatch.setattr(
        assembly.TaskProgramEnvironmentAdapter,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    assert type(factory.create_adapter()) is assembly.TaskProgramEnvironmentAdapter


def test_ordinary_bundles_keep_the_shared_adapter_creation_path(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import assembly

    factory = object.__new__(assembly._TaskFactory)
    shared_adapter = object()
    shared = Mock(return_value=shared_adapter)
    monkeypatch.setattr(assembly.SimulationTaskProgramFactory, "create_adapter", shared)
    assert factory.create_adapter() is shared_adapter
    shared.assert_called_once_with()


def test_preview_and_runtime_compilers_apply_the_same_dynamic_safe_policy():
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )
    from embodichain.lab.sim.atomic_actions import (
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        ControlPartCommandProfile,
        FORWARD_KINEMATICS_CAPABILITY,
        GRASP_CAPABILITY,
        JOINT_POSITION_CAPABILITY,
        MoveJointsOptions,
        PlaceOptions,
    )
    from embodichain.lab.sim.atomic_actions.policies import DynamicCollisionMode
    from embodichain.lab.task_program.integrations._configured_services import (
        _ParkLowerer,
    )
    from embodichain.lab.task_program.language import TaskProgramIntegrationCfg
    from embodichain.lab.task_program.runtime.assembly import (
        assemble_semantic_executor_components,
    )
    from embodichain.lab.task_program.semantics import (
        ControlPartEndpoint,
        EffectAssurance,
        EffectMonitorRef,
        Place,
        RobotResource,
        RobotSkillProfile,
        SceneCollisionRole,
        SceneCollisionWorldMode,
        SceneEntityRegistration,
        SceneObjectRef,
        SceneRegistry,
        SemanticCallDescriptor,
        SemanticPose,
        SkillPolicyPreset,
        builtin_semantic_call_catalog,
    )

    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("can"),
                state_provider=Mock(),
                geometry_provider=Mock(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )
    preset = SkillPolicyPreset(
        "safe",
        effect_assurance=EffectAssurance.VERIFIED,
        motion_policy=MotionPolicy(
            strategy="motion_gen", dynamic_collision_mode=DynamicCollisionMode.OFF
        ),
        action_option_templates={
            "place": PlaceOptions(),
            "simulation.park": MoveJointsOptions(),
        },
        effect_monitors={"place": EffectMonitorRef("builtin.composite_effect", "1")},
    )
    profile = RobotSkillProfile(
        "robot",
        resources={
            "left_arm": RobotResource(
                "left_arm",
                endpoints={
                    "motion": ControlPartEndpoint(
                        "arm",
                        capabilities=frozenset(
                            {
                                BATCH_INVERSE_KINEMATICS_CAPABILITY,
                                CARTESIAN_POSE_CAPABILITY,
                                FORWARD_KINEMATICS_CAPABILITY,
                                JOINT_POSITION_CAPABILITY,
                            }
                        ),
                    ),
                    "grasp": ControlPartEndpoint(
                        "hand", capabilities=frozenset({GRASP_CAPABILITY})
                    ),
                },
            )
        },
        command_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.0]),
                grasp=torch.tensor([1.0]),
            )
        },
        presets={"safe": preset},
        default_preset="safe",
    )
    robot = Mock(device=torch.device("cpu"), dof=2)
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(1, 2)
    robot.get_qvel.return_value = torch.zeros(1, 2)
    robot.get_joint_ids.side_effect = lambda name: [0] if name == "arm" else [1]
    robot.get_solver.side_effect = lambda name=None: object() if name == "arm" else None
    generator = Mock(
        robot=robot, device=torch.device("cpu"), supports_dynamic_collision_world=True
    )
    generator.planner.cfg.planner_type = "stub_planner"
    engine = AtomicActionEngine(
        generator, control_profiles=profile.action_control_profiles()
    )
    lowerer = _ParkLowerer()
    catalog = builtin_semantic_call_catalog().with_descriptor(
        SemanticCallDescriptor(
            call_id=lowerer.call_id,
            spec_type=RegisteredSemanticCall,
            target_descriptor=lowerer.target_descriptor,
        )
    )
    selection = TaskProgramIntegrationCfg(
        robot_profile="robot", scene_registry="scene", runtime_preset="safe"
    )
    arguments = (registry, profile, engine, selection)
    options = {"call_catalog": catalog, "registered_lowerers": (lowerer,)}
    preview = assemble_semantic_executor_components(*arguments, **options)
    runtime = assemble_semantic_executor_components(*arguments, **options)
    release = Place(
        SceneObjectRef("can"),
        at=SemanticPose(
            position=torch.zeros(3), quaternion_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0])
        ),
        resources={"primary": "left_arm"},
    )
    park = _cleanup_call("simulation.park")
    route = recovery.ClearanceRoute("clearance", ("park",), (park,), release)
    requests = recovery.bind_park_requests(preview.compiler, (route,))
    predicted = requests["clearance"][0].workflow.calls[0].bound
    actual = runtime.compiler.analyze((park,)).calls[0].bound
    assert preview.compiler is not runtime.compiler
    assert predicted.preset.motion_policy == actual.preset.motion_policy
    assert (
        predicted.preset.motion_policy.dynamic_collision_mode
        is DynamicCollisionMode.REQUIRED
    )
    assert predicted.preset.tracking_policy == actual.preset.tracking_policy
    assert predicted.preset.recovery_policy == actual.preset.recovery_policy
    assert predicted.binding.action_binding == actual.binding.action_binding


@pytest.mark.parametrize(
    "failure", ["empty_cleanup", "empty_park", "preview_exception"]
)
def test_cleanup_malformed_or_unavailable_preview_stays_fail_closed(
    monkeypatch, cleanup_recovery_fixture, failure
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )

    request, park_request, good, context, robot = cleanup_recovery_fixture
    empty = replace(
        good,
        joint_trajectory=SimpleNamespace(
            positions=torch.zeros(1, 0, 2), dt=torch.zeros(1, 0)
        ),
    )
    original = empty if failure == "empty_cleanup" else good

    def build(selected, predicted):
        if selected is park_request:
            if failure == "preview_exception":
                raise ValueError("preview not available")
            return empty if failure == "empty_park" else _park_plan_from(predicted)
        return original

    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    result = recovery.recover(
        request,
        context,
        original,
        build=build,
        park_request=(park_request,),
        robot=robot,
        simulation=object(),
    )
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "cleanup_clearance_failed"
    assert result.diagnostics.failure.retryable is False
    assert (
        len(result.diagnostics.metadata["gen_sim_clearance_recovery"]["attempts"]) <= 7
    )


def _real_cleanup_plan(fixture, *, park_context=None):
    from embodichain.lab.sim.atomic_actions import (
        ActionBinding,
        ActionPlan,
        EndpointBinding,
        MoveEndEffector,
        TimedTrajectory,
        TrackingPolicy,
    )
    from embodichain.lab.sim.atomic_actions.policies import RecoveryPolicy

    request, _, sample, context, _ = fixture
    if park_context is None:
        positions = sample.joint_trajectory.positions
        dt = sample.joint_trajectory.dt
    else:
        positions = park_context.robot.qpos[:, None].repeat(1, 2, 1)
        dt = torch.tensor([[0.0, 0.04]])
    trajectory = TimedTrajectory.from_positions(
        positions, dt=dt, env_ids=context.env_ids
    )
    binding = ActionBinding(
        owner_id="cleanup_test",
        endpoints=(
            EndpointBinding(
                slot_id="primary",
                endpoint_id="motion",
                resource_id="left_arm",
                adapter_id="test",
                target=JointPositionTarget("left_arm", (0,)),
            ),
        ),
    )
    commands = MoveEndEffector._joint_command_sequence(
        object.__new__(MoveEndEffector),
        replace(request, binding=binding),
        trajectory,
        active_mask=torch.tensor([True]),
    )
    return ActionPlan(
        skill_id="move_end_effector" if park_context is None else "move_joints",
        plan_success=torch.tensor([True]),
        commands=commands,
        joint_trajectory=trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=1,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics(backend="shared_core_fixture"),
    )


@pytest.mark.parametrize("changed", [None, "positions", "hold_duration", "velocities"])
@pytest.mark.parametrize("scope", ["clearance", "park"])
def test_cleanup_acceptance_qualifies_emitted_commands_not_only_trajectory(
    monkeypatch, cleanup_recovery_fixture, changed, scope
):
    from embodichain.gen_sim.task_engine._task_program import (
        clearance_recovery as recovery,
    )
    from embodichain.lab.sim.atomic_actions import JointPositionPayload

    request, park_request, _, context, robot = cleanup_recovery_fixture
    engine = _cleanup_engine(robot, object(), request, (park_request,))
    original = _real_cleanup_plan(cleanup_recovery_fixture)
    transformed_plans = []

    def transform(selected, current, raw):
        if changed is None:
            return raw
        frames = list(raw.commands.frames)
        if changed == "positions":
            frame = frames[1]
            command = frame.commands[0]
            frames[1] = replace(
                frame,
                commands=(
                    replace(
                        command,
                        payload=JointPositionPayload(torch.tensor([[-0.5]])),
                    ),
                ),
            )
        elif changed == "velocities":
            frame = frames[1]
            command = frame.commands[0]
            frames[1] = replace(
                frame,
                commands=(
                    replace(
                        command,
                        payload=JointPositionPayload(
                            command.payload.positions, velocities=torch.tensor([[20.0]])
                        ),
                    ),
                ),
            )
        else:
            frames[0] = replace(frames[0], hold_duration=torch.tensor([0.001]))
        transformed = replace(raw, commands=replace(raw.commands, frames=tuple(frames)))
        torch.testing.assert_close(
            transformed.joint_trajectory.positions, raw.joint_trajectory.positions
        )
        torch.testing.assert_close(
            transformed.joint_trajectory.dt, raw.joint_trajectory.dt
        )
        transformed_plans.append(transformed)
        return transformed

    def plan(_engine, selected, current, *, plan_transform=None):
        raw = (
            original
            if selected.invocation_id == "clearance"
            else _real_cleanup_plan(cleanup_recovery_fixture, park_context=current)
        )
        return raw if plan_transform is None else plan_transform(selected, current, raw)

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    selected = request if scope == "clearance" else park_request
    result = engine._plan_request(selected, context, plan_transform=transform)
    assert result.plan_success.tolist() == [changed is None]
    if changed is not None:
        assert transformed_plans
        assert result.diagnostics.failure.code == (
            "cleanup_clearance_failed"
            if scope == "clearance"
            else "cleanup_park_failed"
        )
        assert result.diagnostics.failure.retryable is False
        if scope == "clearance":
            assert all(
                item["reason"] == "command_trajectory_mismatch"
                for item in result.diagnostics.metadata["gen_sim_clearance_recovery"][
                    "attempts"
                ]
            )
        else:
            assert (
                result.diagnostics.metadata["gen_sim_clearance_park"]["reason"]
                == "command_trajectory_mismatch"
            )
    torch.testing.assert_close(
        original.commands.frames[1].commands[0].payload.positions,
        torch.tensor([[0.01]]),
    )
    assert original.commands.frames[0].hold_duration.item() == pytest.approx(0.04)
