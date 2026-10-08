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

"""GenSim yaw-free held alignment screens candidates without changing shared IK."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import math
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.motion import CheckedMotionGenerator
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanResult, PlanState

__all__: list[str] = []


def _recovery_module():
    return importlib.import_module(
        "embodichain.gen_sim.task_engine._task_program.held_alignment_recovery"
    )


@dataclass(frozen=True)
class _Observation:
    qpos: torch.Tensor
    qvel: torch.Tensor


@dataclass(frozen=True)
class _Context:
    robot: _Observation
    env_ids: torch.Tensor
    batch_size: int = 1


@dataclass(frozen=True)
class _AlignmentRequest:
    goal: HeldObjectPoseGoal
    binding: object


@dataclass(frozen=True)
class _AlignmentPlan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


@pytest.fixture
def alignment_inputs():
    qpos = torch.tensor([[0.0, 0.1, 0.02, 0.5]])
    context = _Context(_Observation(qpos, torch.zeros_like(qpos)), torch.tensor([0]))
    motion = JointPositionTarget(control_part="left_arm", joint_ids=(0, 1))
    hand = JointPositionTarget(control_part="left_hand", joint_ids=(2,))
    arm_endpoint = SimpleNamespace(
        task_state_key="left", require_target=lambda cls: motion
    )
    hand_endpoint = SimpleNamespace(
        task_state_key="left",
        require_target=lambda cls: hand,
        joint_positions=lambda *args, **kwargs: torch.tensor([[0.02]]),
    )
    request = SimpleNamespace(
        goal=HeldObjectPoseGoal(torch.eye(4).repeat(1, 2, 1, 1), world_yaw_free=True),
        binding=SimpleNamespace(
            endpoint=lambda role, channel: (
                arm_endpoint if channel == "motion" else hand_endpoint
            )
        ),
    )
    robot = SimpleNamespace(
        device=torch.device("cpu"),
        dof=4,
        get_joint_ids=lambda name=None: {
            None: [0, 1, 2, 3],
            "left_arm": [0, 1],
            "left_hand": [2],
        }[name],
    )
    generator = object.__new__(CheckedMotionGenerator)
    generator.robot = robot
    generator.device = robot.device
    result = PlanResult(
        success=torch.tensor([True]),
        positions=torch.tensor([[[0.0, 0.1], [0.01, 0.11], [0.02, 0.12]]]),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )
    options = MotionGenOptions(
        strategy="ik_interp",
        start_qpos=qpos[:, :2],
        control_part="left_arm",
        sample_count=3,
        interpolation_dt=0.04,
    )
    return robot, request, context, generator, result, options


def test_unscoped_screen_is_exact_identity_without_mesh_work(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    _, _, _, generator, result, options = alignment_inputs
    monkeypatch.setattr(
        recovery, "check_robot_motion", lambda *args: pytest.fail("unscoped mesh check")
    )
    assert recovery.screen_alignment_result(generator, result, options) is result


def test_exact_heading_does_not_enter_the_new_candidate_policy(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    request.goal = HeldObjectPoseGoal(
        request.goal.object_target_pose, world_yaw_free=False
    )
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: pytest.fail("exact heading changed"),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context):
        assert recovery.screen_alignment_result(generator, result, options) is result


def test_self_unsafe_candidate_masks_only_success_and_preserves_target_samples(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    before_qpos = context.robot.qpos.clone()
    before_positions, before_dt = result.positions.clone(), result.dt.clone()
    seen = []

    def inspect(actual_robot, full, part):
        assert actual_robot is robot and part == "left_arm"
        assert full.shape[-1] == 4
        torch.testing.assert_close(full[:, 2], torch.full_like(full[:, 2], 0.02))
        torch.testing.assert_close(full[:, 3], torch.full_like(full[:, 3], 0.5))
        seen.append(full.clone())
        return {
            "valid": False,
            "frame": 2,
            "kind": "self",
            "pairs": [["fr3_link2_collision", "fr3_link5_collision"]],
        }

    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    with recovery.world_yaw_free_collision_scope(robot, request, context):
        rejected = recovery.screen_alignment_result(generator, result, options)
    assert rejected.success.tolist() == [False]
    assert result.success.tolist() == [True]
    assert len(seen) == 1
    torch.testing.assert_close(rejected.positions, before_positions)
    torch.testing.assert_close(rejected.dt, before_dt)
    torch.testing.assert_close(context.robot.qpos, before_qpos)


def test_candidate_rejection_is_outer_and_does_not_start_density_retries(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    raw_calls = []

    def raw(self, targets, options=None):
        raw_calls.append(options.sample_count)
        return result

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: {"valid": False, "frame": 1, "kind": "self"},
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context):
        rejected = generator.generate([], options)
    assert rejected.success.tolist() == [False]
    assert raw_calls == [3]


def test_original_safe_plan_retains_the_exact_planner_random_stream(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    saved_torch, saved_python, saved_numpy = (
        torch.random.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    try:
        torch.manual_seed(7)
        first, second = torch.rand(1), torch.rand(1)
        torch.manual_seed(7)
        python_before, numpy_before = random.getstate(), pickle.dumps(
            np.random.get_state()
        )
        observed = []

        def raw(self, targets, options=None):
            observed.append(torch.rand(1))
            return result

        def inspect(*args):
            torch.rand(4)
            random.random()
            np.random.random()
            return {"valid": True}

        monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
        monkeypatch.setattr(recovery, "check_robot_motion", inspect)
        with recovery.world_yaw_free_collision_scope(robot, request, context):
            accepted = generator.generate([], options)
        assert accepted.success.tolist() == [True]
        torch.testing.assert_close(observed[0], first)
        torch.testing.assert_close(torch.rand(1), second)
        assert random.getstate() == python_before
        assert pickle.dumps(np.random.get_state()) == numpy_before
    finally:
        torch.random.set_rng_state(saved_torch)
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


def test_exception_restores_scope_before_an_unrelated_generator_call(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    with pytest.raises(RuntimeError, match="scope aborted"):
        with recovery.world_yaw_free_collision_scope(robot, request, context):
            raise RuntimeError("scope aborted")
    monkeypatch.setattr(
        recovery, "check_robot_motion", lambda *args: pytest.fail("scope leaked")
    )
    assert recovery.screen_alignment_result(generator, result, options) is result


def test_actual_ik_failure_does_not_do_an_expensive_mesh_screen(
    monkeypatch, alignment_inputs
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    failed = PlanResult(
        success=torch.tensor([False]), positions=result.positions, dt=result.dt
    )
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: pytest.fail("failed IK candidate mesh work"),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context):
        assert recovery.screen_alignment_result(generator, failed, options) is failed


@pytest.mark.parametrize("failure", ["environment", "batch", "control_part"])
def test_unsupported_alignment_collision_scope_fails_closed(
    monkeypatch, alignment_inputs, failure
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    if failure == "environment":
        context = replace(context, env_ids=torch.tensor([1]))
    elif failure == "batch":
        context = replace(
            context,
            batch_size=2,
            env_ids=torch.tensor([0, 1]),
            robot=replace(
                context.robot,
                qpos=context.robot.qpos.repeat(2, 1),
                qvel=context.robot.qvel.repeat(2, 1),
            ),
        )
    else:
        options = replace(options, control_part="right_arm")
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: pytest.fail("unsupported scope queried geometry"),
    )
    with pytest.raises(recovery.AlignmentCollisionUnavailable):
        with recovery.world_yaw_free_collision_scope(robot, request, context):
            recovery.screen_alignment_result(generator, result, options)
    assert recovery.screen_alignment_result(generator, result, options) is result


@pytest.mark.parametrize("failure", ["unavailable", "malformed_result"])
def test_collision_screen_failure_does_not_blindly_attempt_eight_yaws(
    monkeypatch, alignment_inputs, failure
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    raw_calls = []
    inspections = []

    def raw(self, targets, options=None):
        raw_calls.append(options.sample_count)
        return result

    def inspect(*args):
        inspections.append(True)
        if failure == "unavailable":
            raise ImportError("FCL unavailable")
        return {}

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    with pytest.raises(recovery.AlignmentCollisionUnavailable):
        with recovery.world_yaw_free_collision_scope(robot, request, context):
            for _ in range(8):
                generator.generate([], options)
    assert raw_calls == [3]
    assert len(inspections) == 1
    assert recovery.screen_alignment_result(generator, result, options) is result


@pytest.mark.parametrize("kind", ["scene", "self", "other_arm", None])
def test_collision_wrappers_preserve_the_common_geometry_result(
    monkeypatch, alignment_inputs, kind
):
    from embodichain.gen_sim.task_engine._task_program import articulation_collision

    robot, _, _, _, result, _ = alignment_inputs
    positions = result.positions[0]
    simulation = object()
    evidence = (
        {"valid": True, "frames": 3, "scope": "discrete_free_motion_meshes"}
        if kind is None
        else {"valid": False, "frame": 1, "kind": kind, "pairs": [["a", "b"]]}
    )
    calls = []

    def check(
        actual_robot, actual_simulation, actual_positions, part, *, include_scene
    ):
        calls.append((actual_simulation, include_scene))
        assert actual_robot is robot and actual_positions is positions
        assert part == "left_arm"
        return evidence

    monkeypatch.setattr(articulation_collision, "_check_motion", check)
    assert (
        articulation_collision.check_free_motion(
            robot, simulation, positions, "left_arm"
        )
        is evidence
    )
    assert (
        articulation_collision.check_robot_motion(robot, positions, "left_arm")
        is evidence
    )
    assert calls == [(simulation, True), (None, False)]


@pytest.mark.parametrize(
    "geometry_links,kind",
    [
        (("chain_root", "b"), None),
        (("tool", "hand_3"), None),
        (("chain_root", "c"), "self"),
        (("tool", "passive"), "other_arm"),
    ],
)
def test_robot_mesh_screen_preserves_neighbor_gripper_and_other_arm_rules(
    tmp_path, geometry_links, kind
):
    pytest.importorskip("fcl")
    pytest.importorskip("yourdfpy")
    from embodichain.gen_sim.task_engine._task_program.articulation_collision import (
        check_free_motion,
        check_robot_motion,
    )

    names = [
        "world",
        "chain_root",
        "a",
        "b",
        "c",
        "tool",
        "hand_1",
        "hand_2",
        "hand_3",
        "passive",
    ]
    box = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    links = [
        f'<link name="{name}">{box if name in geometry_links else ""}</link>'
        for name in names
    ]
    joints = [
        '<joint name="mount" type="fixed"><parent link="world"/><child link="chain_root"/><origin xyz="0.5 0 0"/></joint>',
        '<joint name="spin" type="revolute"><parent link="chain_root"/><child link="a"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>',
        '<joint name="passive_mount" type="fixed"><parent link="world"/><child link="passive"/><origin xyz="0.5 0 0"/></joint>',
    ]
    chain = names[2:-1]
    joints.extend(
        f'<joint name="fixed_{child}" type="fixed"><parent link="{parent}"/><child link="{child}"/></joint>'
        for parent, child in zip(chain, chain[1:])
    )
    urdf = tmp_path / "alignment.urdf"
    urdf.write_text('<robot name="alignment">' + "".join(links + joints) + "</robot>")
    eye = torch.eye(4)[None]
    tool = eye.clone()
    tool[:, 0, 3] = 0.5
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(urdf)),
        joint_names=["spin"],
        get_qpos=lambda: torch.zeros(1, 1),
        get_solver=lambda part: SimpleNamespace(
            root_link_name="chain_root", end_link_name="tool"
        ),
        get_local_pose=lambda **kwargs: eye,
        get_link_pose=lambda *args, **kwargs: tool,
    )
    simulation = SimpleNamespace(
        get_rigid_object_uid_list=lambda: [],
        get_articulation_uid_list=lambda: [],
    )
    positions = torch.tensor([[0.0], [0.1]])
    free = check_free_motion(robot, simulation, positions, "arbitrary_arm")
    held = check_robot_motion(robot, positions, "arbitrary_arm")
    assert free["valid"] is (kind is None)
    assert held["valid"] is (kind is None)
    if kind is not None:
        assert free == held
        assert held["kind"] == kind and held["frame"] == 0
    else:
        assert free["frames"] == held["frames"] == 2


def test_robot_mesh_checker_reuses_geometry_and_updates_every_frame(
    tmp_path, monkeypatch
):
    pytest.importorskip("fcl")
    pytest.importorskip("yourdfpy")
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.articulation_collision import (
        check_robot_motion,
        robot_collision_scope,
    )

    path = tmp_path / "moving.urdf"
    box = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    path.write_text(
        '<robot name="moving"><link name="world"/><link name="base"/>'
        f'<link name="tool">{box}</link><link name="passive">{box}</link>'
        '<joint name="mount" type="fixed"><parent link="world"/><child link="base"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>'
        '<joint name="other" type="fixed"><parent link="world"/><child link="passive"/>'
        '<origin xyz="0.5 0 0"/></joint></robot>'
    )
    eye = torch.eye(4)[None]
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(path)),
        joint_names=["slide"],
        get_qpos=lambda: torch.zeros(1, 1),
        get_solver=lambda part: SimpleNamespace(
            root_link_name="base", end_link_name="tool"
        ),
        get_local_pose=lambda **kwargs: eye,
        get_link_pose=lambda *args, **kwargs: eye,
    )
    created, updated = [], []
    add = trimesh.collision.CollisionManager.add_object
    set_transform = trimesh.collision.CollisionManager.set_transform

    def add_object(self, name, mesh, transform=None):
        created.append(name)
        return add(self, name, mesh, transform)

    def update(self, name, transform):
        assert transform.dtype == np.float32
        updated.append(name)
        return set_transform(self, name, transform)

    monkeypatch.setattr(trimesh.collision.CollisionManager, "add_object", add_object)
    monkeypatch.setattr(trimesh.collision.CollisionManager, "set_transform", update)
    with robot_collision_scope():
        first = check_robot_motion(robot, torch.tensor([[0.0], [0.25], [0.5]]), "arm")
        second = check_robot_motion(robot, torch.tensor([[1.0], [0.0]]), "arm")
    assert first["valid"] is False and first["frame"] == 2
    assert first["kind"] == "other_arm"
    assert second["valid"] is True and second["frames"] == 2
    assert len(created) == 2
    assert len(updated) == 10


@pytest.mark.parametrize("screen_candidates", [False, True])
def test_local_branch_predicate_keeps_targets_timing_and_velocity_limits(
    monkeypatch, alignment_inputs, screen_candidates
):
    from embodichain.gen_sim.task_engine._task_program import motion
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator

    robot, _, _, generator, _, options = alignment_inputs
    robot.get_qpos_limits = lambda **kwargs: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 2)
    )
    target = object()
    calls = []
    screened = []

    def generate(self, targets, options=None):
        assert self is generator and targets == [target]
        assert options.preserve_cartesian_samples is True
        assert options.sample_count == 2 and options.interpolation_dt == 0.04
        calls.append(options.start_qpos.clone())
        end = torch.tensor([[0.01 if len(calls) == 1 else 0.02, 0.1]])
        return PlanResult(
            success=torch.tensor([True]),
            positions=torch.stack((options.start_qpos, end), dim=1),
            dt=torch.tensor([[0.0, 0.04]]),
        )

    def predicate(qpos):
        screened.append(qpos.clone())
        return bool(qpos[0, 0] > 0.015)

    monkeypatch.setattr(MotionGenerator, "generate", generate)
    start = options.start_qpos.clone()
    result = motion._local_place_ik(
        generator,
        [target],
        options,
        candidate_predicate=predicate if screen_candidates else None,
    )
    assert result.success.all()
    torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.04]]))
    torch.testing.assert_close(result.positions[:, 0], start)
    assert result.positions[0, -1, 0] == pytest.approx(
        0.02 if screen_candidates else 0.01
    )
    assert len(calls) == (9 if screen_candidates else 1)
    assert len(screened) == (9 if screen_candidates else 0)
    torch.testing.assert_close(options.start_qpos, start)


def _heading_targets(angle):
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[:, -1, :3, :3] = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return [
        PlanState(move_type=MoveType.EEF_MOVE, xpos=poses[:, index])
        for index in range(2)
    ]


@pytest.mark.parametrize("qualified", [True, False])
@pytest.mark.parametrize("last_raw_success", [True, False])
def test_last_allowed_heading_recovery_replays_exact_targets_and_only_once(
    monkeypatch, alignment_inputs, qualified, last_raw_success
):
    recovery = _recovery_module()
    robot, raw_request, context, generator, result, options = alignment_inputs
    request = _AlignmentRequest(raw_request.goal, raw_request.binding)
    options = replace(options, preserve_cartesian_samples=True)
    original_goal = request.goal.object_target_pose.clone()
    original = _AlignmentPlan(
        torch.tensor([False]), PlannerDiagnostics(backend="fixture")
    )
    repaired = replace(result, positions=result.positions.clone())
    repaired.positions[:, 1] = repaired.positions[:, -1]
    robot.compute_fk = lambda **kwargs: torch.eye(4)[None]
    raw_calls, inspected, local_calls, built = [], [], [], []

    def raw(self, targets, options=None):
        raw_calls.append(targets)
        return (
            result
            if len(raw_calls) == 1 or (last_raw_success and len(raw_calls) == 8)
            else replace(result, success=torch.tensor([False]))
        )

    def inspect(_robot, full, part):
        assert part == "left_arm"
        inspected.append(full.clone())
        return {
            "valid": full.shape[0] == 1
            and bool(full[0, 0] != result.positions[0, 1, 0]),
            "kind": "self",
            "frame": 1,
        }

    def local(actual_generator, targets, actual_options, *, candidate_predicate):
        local_calls.append(targets)
        assert actual_generator is generator
        assert actual_options.sample_count == options.sample_count
        assert actual_options.interpolation_dt == options.interpolation_dt
        torch.testing.assert_close(actual_options.start_qpos, options.start_qpos)
        assert len(targets) == len(last_targets)
        for target, last in zip(targets, last_targets):
            torch.testing.assert_close(target.xpos, last.xpos, rtol=0, atol=0)
        if last_raw_success:
            assert candidate_predicate(result.positions[:, 1]) is False
        assert candidate_predicate(repaired.positions[:, -1]) is True
        return repaired

    def build(selected):
        built.append(selected)
        assert selected.goal.world_yaw_free is False
        torch.testing.assert_close(
            selected.goal.object_target_pose[:, :-1], original_goal[:, :-1]
        )
        torch.testing.assert_close(
            selected.goal.object_target_pose[:, -1], last_targets[-1].xpos
        )
        replayed = generator.generate(last_targets, options)
        assert replayed is repaired
        return _AlignmentPlan(
            torch.tensor([True]), PlannerDiagnostics(backend="fixture")
        )

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    monkeypatch.setattr(recovery, "_local_place_ik", local)
    monkeypatch.setattr(
        recovery,
        "qualify_alignment_plan",
        lambda selected, current, plan, actual_robot: replace(
            plan, plan_success=torch.tensor([qualified])
        ),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context) as scope:
        for fraction in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
            last_targets = _heading_targets(math.pi * fraction)
            assert not generator.generate(last_targets, options).success.any()
        recovered = recovery.recover(scope, request, context, original, build)
        assert recovered.plan_success.tolist() == [qualified]
        report = recovered.diagnostics.metadata["gen_sim_held_alignment"]
        assert report["rejected_raw_success"] == (2 if last_raw_success else 1)
        assert report["local_fallback"]["accepted"] is qualified
        assert report["local_fallback"]["predicate_calls"] == (
            2 if last_raw_success else 1
        )
        assert report["local_fallback"]["target_count"] == 2
        assert report["local_fallback"]["sample_count"] == 3
        assert scope.replay_call is None and scope.replay_result is None
        recovery.recover(scope, request, context, original, build)
    assert len(raw_calls) == 8
    assert len(local_calls) == len(built) == 1
    torch.testing.assert_close(request.goal.object_target_pose, original_goal)


@pytest.mark.parametrize("skip", ["short_search", "no_rejection"])
def test_local_recovery_does_not_extend_other_failure_classes(
    monkeypatch, alignment_inputs, skip
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    options = replace(options, preserve_cartesian_samples=True)
    original = _AlignmentPlan(
        torch.tensor([False]), PlannerDiagnostics(backend="fixture")
    )
    raw_calls = []

    def raw(self, targets, options=None):
        raw_calls.append(True)
        ok = len(raw_calls) == 1
        return replace(result, success=torch.tensor([ok]))

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: {"valid": skip == "no_rejection", "kind": "self", "frame": 1},
    )
    monkeypatch.setattr(
        recovery,
        "_local_place_ik",
        lambda *args, **kwargs: pytest.fail("unexpected IK extension"),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context) as scope:
        for _ in range(7 if skip == "short_search" else 8):
            generator.generate(_heading_targets(math.pi), options)
        rejected = recovery.recover(
            scope,
            request,
            context,
            original,
            lambda selected: pytest.fail("unexpected rebuild"),
        )
    assert not rejected.plan_success.any()
    assert (
        "local_fallback" not in rejected.diagnostics.metadata["gen_sim_held_alignment"]
    )


@pytest.mark.parametrize("last_raw_success", [True, False])
def test_local_recovery_rejects_observed_start_collision_without_escaping(
    monkeypatch, alignment_inputs, last_raw_success
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    observed = context.robot.qpos.clone()
    observed[:, 2] = 0.04
    context = replace(context, robot=replace(context.robot, qpos=observed))
    options = replace(options, preserve_cartesian_samples=True)
    robot.compute_fk = lambda **kwargs: torch.eye(4)[None]
    original = _AlignmentPlan(
        torch.tensor([False]), PlannerDiagnostics(backend="fixture")
    )
    raw_calls, starts = [], []

    def raw(self, targets, options=None):
        raw_calls.append(targets)
        ok = len(raw_calls) == 1 or (last_raw_success and len(raw_calls) == 8)
        return replace(result, success=torch.tensor([ok]))

    def inspect(actual_robot, full, part):
        assert actual_robot is robot and part == "left_arm"
        if full.shape[0] == 1:
            starts.append(full.clone())
            torch.testing.assert_close(full, observed)
            return {"valid": False, "kind": "other_arm", "frame": 0}
        return {"valid": False, "kind": "other_arm", "frame": 1}

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    monkeypatch.setattr(
        recovery,
        "_local_place_ik",
        lambda *args, **kwargs: pytest.fail("colliding start must not enter local IK"),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context) as scope:
        for _ in range(8):
            assert not generator.generate(
                _heading_targets(math.pi), options
            ).success.any()
        rejected = recovery.recover(
            scope,
            request,
            context,
            original,
            lambda selected: pytest.fail("colliding start must not rebuild a plan"),
        )
        report = rejected.diagnostics.metadata["gen_sim_held_alignment"]
        assert not rejected.plan_success.any()
        assert report["local_fallback"]["reason"] == "initial_state_collision"
        assert report["local_fallback"]["accepted"] is False
        recovery.recover(scope, request, context, original, lambda selected: None)
    assert len(raw_calls) == 8 and len(starts) == 1


def test_held_alignment_factory_versions_local_branch_recovery():
    from embodichain.gen_sim.task_engine._task_program.align_held import (
        _AlignHeldFactory,
    )

    assert _AlignHeldFactory.revision == "10"


@pytest.mark.parametrize(
    "outcome", ["accepted", "qualification_rejected", "ik_failed", "error"]
)
def test_all_eight_raw_failures_reuse_existing_last_heading_and_restore_legacy_stop(
    monkeypatch, alignment_inputs, outcome
):
    recovery = _recovery_module()
    robot, raw_request, context, generator, result, options = alignment_inputs
    request = _AlignmentRequest(raw_request.goal, raw_request.binding)
    options = replace(options, preserve_cartesian_samples=True)
    robot.compute_fk = lambda **kwargs: torch.eye(4)[None]
    original = _AlignmentPlan(
        torch.tensor([False]), PlannerDiagnostics(backend="fixture")
    )
    original_goal = request.goal.object_target_pose.clone()
    calls, local_calls, rebuilt = [], [], []

    def draw():
        torch.rand(1)
        random.random()
        np.random.random()

    def raw(self, targets, options=None):
        calls.append(targets)
        draw()
        return replace(result, success=torch.tensor([False]))

    def inspect(*args):
        draw()
        return {"valid": True}

    def local(actual_generator, targets, actual_options, *, candidate_predicate):
        local_calls.append(targets)
        assert (
            actual_generator is generator and actual_options is scope.calls[-1].options
        )
        for target, last in zip(targets, last_targets):
            torch.testing.assert_close(target.xpos, last.xpos, rtol=0, atol=0)
        draw()
        assert candidate_predicate(result.positions[:, -1])
        if outcome == "error":
            raise RuntimeError("extra local search aborted")
        return None if outcome == "ik_failed" else result

    def build(selected):
        rebuilt.append(selected)
        draw()
        assert selected.goal.world_yaw_free is False
        assert torch.equal(
            selected.goal.object_target_pose[:, :-1], original_goal[:, :-1]
        )
        torch.testing.assert_close(
            selected.goal.object_target_pose[:, -1], last_targets[-1].xpos
        )
        assert generator.generate(last_targets, options) is result
        return _AlignmentPlan(
            torch.tensor([True]), PlannerDiagnostics(backend="fixture")
        )

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    monkeypatch.setattr(recovery, "_local_place_ik", local)
    monkeypatch.setattr(
        recovery,
        "qualify_alignment_plan",
        lambda selected, current, plan, actual_robot: replace(
            plan, plan_success=torch.tensor([outcome == "accepted"])
        ),
    )
    saved = recovery._RandomState.capture(set())
    try:
        with recovery.world_yaw_free_collision_scope(robot, request, context) as scope:
            for fraction in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
                last_targets = _heading_targets(math.pi * fraction)
                assert not generator.generate(last_targets, options).success.any()
            assert scope.rejected_raw_success == 0 and scope.checkpoint is None
            legacy = recovery._RandomState.capture(set())
            if outcome == "error":
                with pytest.raises(RuntimeError, match="extra local search aborted"):
                    recovery.recover(scope, request, context, original, build)
            else:
                selected = recovery.recover(scope, request, context, original, build)
                assert bool(selected.plan_success.all()) == (outcome == "accepted")
                assert selected.diagnostics.metadata["gen_sim_held_alignment"][
                    "local_fallback"
                ]["accepted"] is (outcome == "accepted")
                recovery.recover(
                    scope,
                    request,
                    context,
                    original,
                    lambda selected: pytest.fail("local search repeated"),
                )
        assert len(calls) == 8 and len(local_calls) == 1
        assert len(rebuilt) == int(outcome in {"accepted", "qualification_rejected"})
        assert scope.checkpoint is not None
        assert torch.equal(torch.random.get_rng_state(), legacy.cpu)
        assert random.getstate() == legacy.python
        assert pickle.dumps(np.random.get_state()) == pickle.dumps(legacy.numpy)
        assert torch.equal(request.goal.object_target_pose, original_goal)
    finally:
        saved.restore()


@pytest.mark.parametrize(
    "skip", ["safe", "short_search", "unsupported", "actual_collision"]
)
def test_all_raw_failure_gate_retains_existing_negative_boundaries(
    monkeypatch, alignment_inputs, skip
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    options = replace(
        options,
        preserve_cartesian_samples=True,
        sample_count=2 if skip == "unsupported" else 3,
    )
    original = _AlignmentPlan(
        torch.tensor([skip == "safe"]), PlannerDiagnostics(backend="fixture")
    )
    monkeypatch.setattr(
        CheckedMotionGenerator,
        "_generate_checked",
        lambda *args, **kwargs: replace(result, success=torch.tensor([False])),
    )
    monkeypatch.setattr(
        recovery,
        "check_robot_motion",
        lambda *args: {"valid": skip != "actual_collision"},
    )
    monkeypatch.setattr(
        recovery,
        "_local_place_ik",
        lambda *args, **kwargs: pytest.fail("negative boundary entered local IK"),
    )
    with recovery.world_yaw_free_collision_scope(robot, request, context) as scope:
        for _ in range(7 if skip == "short_search" else 8):
            generator.generate(_heading_targets(math.pi), options)
        before = recovery._RandomState.capture(set())
        selected = recovery.recover(
            scope,
            request,
            context,
            original,
            lambda selected: pytest.fail("negative boundary rebuilt plan"),
        )
        assert torch.equal(selected.plan_success, original.plan_success)
        assert (
            scope.checkpoint is not None
            if skip == "actual_collision"
            else scope.checkpoint is None
        )
        if skip == "actual_collision":
            assert (
                selected.diagnostics.metadata["gen_sim_held_alignment"][
                    "local_fallback"
                ]["reason"]
                == "initial_state_collision"
            )
        elif skip == "unsupported":
            assert (
                selected.diagnostics.metadata["gen_sim_held_alignment"][
                    "local_fallback"
                ]
                == "unsupported"
            )
        assert torch.equal(torch.random.get_rng_state(), before.cpu)
        assert random.getstate() == before.python
        assert pickle.dumps(np.random.get_state()) == pickle.dumps(before.numpy)


@pytest.mark.parametrize("raise_in_fallback", [True, False])
def test_last_raw_success_local_recovery_restores_legacy_random_stop(
    monkeypatch, alignment_inputs, raise_in_fallback
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    options = replace(options, preserve_cartesian_samples=True)
    robot.compute_fk = lambda **kwargs: torch.eye(4)[None]
    original = _AlignmentPlan(
        torch.tensor([False]), PlannerDiagnostics(backend="fixture")
    )
    saved_torch = torch.random.get_rng_state()
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    raw_calls, local_calls = [], []

    def draw():
        torch.rand(1)
        random.random()
        np.random.random()

    def raw(self, targets, options=None):
        raw_calls.append(targets)
        draw()
        return result

    def inspect(actual_robot, full, part):
        draw()
        return {"valid": full.shape[0] == 1, "kind": "self", "frame": 1}

    def local(*args, **kwargs):
        local_calls.append(args)
        draw()
        if raise_in_fallback:
            raise RuntimeError("local search aborted")
        return None

    monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    monkeypatch.setattr(recovery, "_local_place_ik", local)
    try:
        torch.manual_seed(41)
        random.seed(41)
        np.random.seed(41)
        draw()
        expected_torch = torch.random.get_rng_state()
        expected_python = random.getstate()
        expected_numpy = pickle.dumps(np.random.get_state())
        torch.manual_seed(41)
        random.seed(41)
        np.random.seed(41)

        def run_scope():
            with recovery.world_yaw_free_collision_scope(
                robot, request, context
            ) as scope:
                for _ in range(8):
                    assert not generator.generate(
                        _heading_targets(math.pi), options
                    ).success.any()
                rejected = recovery.recover(
                    scope,
                    request,
                    context,
                    original,
                    lambda selected: pytest.fail(
                        "failed local search must not rebuild"
                    ),
                )
                assert not rejected.plan_success.any()
                recovery.recover(
                    scope, request, context, original, lambda selected: None
                )

        if raise_in_fallback:
            with pytest.raises(RuntimeError, match="local search aborted"):
                run_scope()
        else:
            run_scope()
        assert len(raw_calls) == 8 and len(local_calls) == 1
        torch.testing.assert_close(torch.random.get_rng_state(), expected_torch)
        assert random.getstate() == expected_python
        assert pickle.dumps(np.random.get_state()) == expected_numpy
    finally:
        torch.random.set_rng_state(saved_torch)
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


def _typed_alignment_plan(alignment_inputs, positions=None):
    from embodichain.lab.sim.atomic_actions import (
        ActionPlan,
        RecoveryPolicy,
        TimedTrajectory,
        TrackingPolicy,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
        MoveHeldObject,
    )

    _, raw_request, context, _, result, _ = alignment_inputs
    raw_request.binding.endpoints = tuple(
        raw_request.binding.endpoint("primary", channel)
        for channel in ("motion", "grasp")
    )
    request = _AlignmentRequest(raw_request.goal, raw_request.binding)
    if positions is None:
        positions = context.robot.qpos[:, None].repeat(1, 3, 1)
        positions[:, :, :2] = result.positions
    trajectory = TimedTrajectory.from_positions(
        positions, dt=result.dt, env_ids=context.env_ids
    )
    commands = MoveHeldObject._joint_command_sequence(
        object.__new__(MoveHeldObject),
        request,
        trajectory,
        active_mask=torch.tensor([True]),
    )
    return request, ActionPlan(
        skill_id="move_held_object",
        plan_success=torch.tensor([True]),
        commands=commands,
        joint_trajectory=trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=1,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics(backend="typed_fixture"),
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "start",
        "passive",
        "hand",
        "velocity",
        "limit",
        "payload",
        "schedule",
        "payload_velocity",
        "collision",
        "collision_unavailable",
    ],
)
def test_final_alignment_qualification_checks_real_emitted_commands_and_full_robot(
    monkeypatch, alignment_inputs, failure
):
    from embodichain.lab.sim.atomic_actions import JointPositionPayload

    recovery = _recovery_module()
    robot, _, context, _, _, _ = alignment_inputs
    request, baseline = _typed_alignment_plan(alignment_inputs)
    before_positions = baseline.joint_trajectory.positions.clone()
    plan = baseline
    robot.get_qpos_limits = (
        lambda **kwargs: torch.tensor([[[-1.0, 0.115 if failure == "limit" else 1.0]]])
        .expand(1, 2 if kwargs["name"] == "left_arm" else 1, 2)
        .clone()
    )
    monkeypatch.setattr(
        recovery, "_joint_velocity_limits", lambda *args: torch.ones(1, 4)
    )
    if failure in {"start", "passive", "hand", "velocity"}:
        positions = before_positions.clone()
        if failure == "start":
            positions[0, 0, 0] += 0.001
        elif failure == "passive":
            positions[0, 1, 3] += 0.001
        elif failure == "hand":
            positions[0, 1, 2] += 0.001
        else:
            positions[0, 1, 0] = 0.2
        _, plan = _typed_alignment_plan(alignment_inputs, positions)
    elif failure in {"payload", "schedule", "payload_velocity"}:
        frames = list(plan.commands.frames)
        frame = frames[1]
        if failure == "schedule":
            frames[1] = replace(frame, hold_duration=torch.tensor([0.001]))
        else:
            commands = list(frame.commands)
            payload = commands[0].payload
            if failure == "payload":
                payload = replace(payload, positions=payload.positions + 0.001)
            else:
                payload = JointPositionPayload(
                    payload.positions,
                    velocities=torch.full_like(payload.positions, 20.0),
                )
            commands[0] = replace(commands[0], payload=payload)
            frames[1] = replace(frame, commands=tuple(commands))
        plan = replace(plan, commands=replace(plan.commands, frames=tuple(frames)))
    inspections = []

    def inspect(actual_robot, full, part):
        assert actual_robot is robot and part == "left_arm"
        inspections.append(full.clone())
        if failure == "collision_unavailable":
            raise ImportError("FCL unavailable")
        return {"valid": failure != "collision", "kind": "self", "frame": 1}

    monkeypatch.setattr(recovery, "check_robot_motion", inspect)
    checked = recovery.qualify_alignment_plan(request, context, plan, robot)
    assert checked.plan_success.tolist() == [failure is None]
    assert plan.plan_success.all()
    report = checked.diagnostics.metadata["gen_sim_held_alignment"][
        "final_command_check"
    ]
    assert report["valid"] is (failure is None)
    if failure is not None:
        assert checked.diagnostics.failure.code == "held_alignment_safety_failed"
        assert checked.diagnostics.failure.retryable is False
    assert len(inspections) == (
        1 if failure in {None, "collision", "collision_unavailable"} else 0
    )
    torch.testing.assert_close(baseline.joint_trajectory.positions, before_positions)


@pytest.mark.parametrize("raise_after_extra", [False, True])
def test_extra_yaw_search_restores_the_first_old_stop_checkpoint(
    monkeypatch, alignment_inputs, raise_after_extra
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    saved_torch = torch.random.get_rng_state()
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    try:
        torch.manual_seed(23)
        random.seed(23)
        np.random.seed(23)
        torch.rand(1)
        random.random()
        np.random.random()
        expected_torch = torch.random.get_rng_state()
        expected_python = random.getstate()
        expected_numpy = pickle.dumps(np.random.get_state())
        torch.manual_seed(23)
        random.seed(23)
        np.random.seed(23)
        calls = []

        def raw(self, targets, options=None):
            torch.rand(1)
            random.random()
            np.random.random()
            calls.append(len(calls))
            return result

        def inspect(*args):
            torch.rand(8)
            random.random()
            np.random.random()
            return {"valid": len(calls) > 1, "frame": 1, "kind": "self"}

        monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
        monkeypatch.setattr(recovery, "check_robot_motion", inspect)

        def run_scope():
            with recovery.world_yaw_free_collision_scope(robot, request, context):
                assert generator.generate([], options).success.tolist() == [False]
                assert generator.generate([], options).success.tolist() == [True]
                assert generator.generate([], options).success.tolist() == [True]
                if raise_after_extra:
                    raise RuntimeError("extra search aborted")

        if raise_after_extra:
            with pytest.raises(RuntimeError, match="extra search aborted"):
                run_scope()
        else:
            run_scope()
        assert calls == [0, 1, 2]
        torch.testing.assert_close(torch.random.get_rng_state(), expected_torch)
        assert random.getstate() == expected_python
        assert pickle.dumps(np.random.get_state()) == expected_numpy
        assert recovery.screen_alignment_result(generator, result, options) is result
    finally:
        torch.random.set_rng_state(saved_torch)
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)


@pytest.mark.parametrize("initial_ik_failures", [1, 2])
def test_extra_yaw_search_checkpoints_the_first_raw_success_not_first_attempt(
    monkeypatch, alignment_inputs, initial_ik_failures
):
    recovery = _recovery_module()
    robot, request, context, generator, result, options = alignment_inputs
    saved_torch, saved_python, saved_numpy = (
        torch.random.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    try:
        torch.manual_seed(37)
        random.seed(37)
        np.random.seed(37)
        for _ in range(initial_ik_failures + 1):
            torch.rand(1)
            random.random()
            np.random.random()
        expected_torch = torch.random.get_rng_state()
        expected_python = random.getstate()
        expected_numpy = pickle.dumps(np.random.get_state())
        torch.manual_seed(37)
        random.seed(37)
        np.random.seed(37)
        raw_calls = []
        inspections = []

        def raw(self, targets, options=None):
            torch.rand(1)
            random.random()
            np.random.random()
            raw_calls.append(True)
            if len(raw_calls) <= initial_ik_failures:
                return PlanResult(
                    success=torch.tensor([False]),
                    positions=result.positions,
                    dt=result.dt,
                )
            return result

        def inspect(*args):
            inspections.append(True)
            return {"valid": len(inspections) > 1, "kind": "self", "frame": 1}

        monkeypatch.setattr(CheckedMotionGenerator, "_generate_checked", raw)
        monkeypatch.setattr(recovery, "check_robot_motion", inspect)
        with recovery.world_yaw_free_collision_scope(robot, request, context):
            for _ in range(initial_ik_failures + 1):
                assert not generator.generate([], options).success.any()
            assert generator.generate([], options).success.all()
        assert len(raw_calls) == initial_ik_failures + 2
        assert len(inspections) == 2
        torch.testing.assert_close(torch.random.get_rng_state(), expected_torch)
        assert random.getstate() == expected_python
        assert pickle.dumps(np.random.get_state()) == expected_numpy
    finally:
        torch.random.set_rng_state(saved_torch)
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
