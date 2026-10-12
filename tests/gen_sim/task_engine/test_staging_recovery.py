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

"""Bounded GenSim staging hints must pass unchanged complete-plan safety gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import importlib
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    actions,
    articulation_collision,
)
from embodichain.gen_sim.task_engine._task_program import (
    held_alignment_recovery,
    motion,
    planning_probe,
)
from embodichain.gen_sim.task_engine._task_program.state_handoff import (
    StagingContract,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    StateDelta,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, TaskState


def _module():
    return importlib.import_module(
        "embodichain.gen_sim.task_engine._task_program.staging_recovery"
    )


@dataclass(frozen=True)
class _Request:
    goal: HeldObjectPoseGoal
    binding: object
    motion_policy: MotionPolicy
    skill_options: MoveHeldObjectOptions
    invocation_id: str = "program/align:0"
    skill_id: str = "move_held_object"


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: TimedTrajectory
    commands: object
    expected_effects: StateDelta = StateDelta()


@pytest.fixture
def staging_inputs(monkeypatch):
    qpos = torch.zeros(1, 6)
    qpos[:, 2] = 0.02
    arm, hand = JointPositionTarget("active", (0, 1)), JointPositionTarget("hand", (2,))
    endpoints = tuple(
        SimpleNamespace(
            task_state_key="active",
            require_target=lambda _, target=target: target,
            joint_positions=lambda *args, **kwargs: torch.tensor([[0.02]]),
        )
        for target in (arm, hand)
    )
    object_pose = torch.eye(4)[None]
    object_pose[:, 0, 3] = 0.6
    stage = object_pose.clone()
    stage[:, 2, 3] = 0.3
    final = stage.clone()
    final[:, 0, 3] = 0.5
    request = _Request(
        HeldObjectPoseGoal(torch.stack((stage, final), 1), world_yaw_free=True),
        SimpleNamespace(
            endpoint=lambda _, name: endpoints[name == "grasp"], endpoints=endpoints
        ),
        MotionPolicy(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
        MoveHeldObjectOptions(),
    )
    held = HeldObjectState(
        ObjectSemantics(
            affordance=AntipodalAffordance(
                mesh_vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
            ),
            geometry={},
            entity_id="item",
        ),
        object_to_eef=torch.eye(4)[None],
        grasp_xpos=object_pose,
    )
    task = TaskState(1, "cpu", held_objects={"active": held})
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=qpos, root_pose=torch.eye(4)[None]),
        task=task,
        get_held_object=task.get_held_object,
        scene=SimpleNamespace(
            entities={"item": SimpleNamespace(pose=object_pose, confidence=1.0)}
        ),
    )
    robot = SimpleNamespace(
        device=torch.device("cpu"),
        get_joint_ids=lambda name: {"active": (0, 1), "passive": (3, 4), "hand": (2,)}[
            name
        ],
        get_qpos_limits=lambda name, env_ids=None: torch.tensor([[[-1.0, 1.0]]]).expand(
            1, 2 if name == "active" else 1, 2
        ),
    )
    action = actions.GenSimMoveHeldObject()
    action._planning_services = SimpleNamespace(
        device=robot.device,
        robot=robot,
        motion_generator=SimpleNamespace(direct_transport_planning=False),
    )
    action.staging_contract = StagingContract(
        rigid_entity_ids=frozenset({"item"}),
        invocation_ids=frozenset({request.invocation_id}),
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 6)
    )
    monkeypatch.setattr(
        held_alignment_recovery,
        "_joint_velocity_limits",
        lambda *args: torch.ones(1, 6),
    )
    monkeypatch.setattr(
        planning_probe, "_joint_velocity_limits", lambda *args: torch.ones(1, 6)
    )
    monkeypatch.setattr(
        articulation_collision, "check_robot_motion", lambda *args: {"valid": True}
    )
    monkeypatch.setattr(
        held_alignment_recovery, "check_robot_motion", lambda *args: {"valid": True}
    )

    def make_plan(success):
        positions = qpos[:, None].repeat(1, 3, 1)
        positions[:, 1:, 0] = torch.tensor([0.01, 0.02])
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=context.env_ids, step_dt=0.04
        )
        return _Plan(
            torch.tensor([success]),
            PlannerDiagnostics(backend="probe"),
            trajectory,
            action._joint_command_sequence(
                request, trajectory, active_mask=torch.tensor([success])
            ),
        )

    return SimpleNamespace(
        request=request,
        context=context,
        robot=robot,
        action=action,
        make_plan=make_plan,
    )


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def _state():
    return (
        pickle.dumps((random.getstate(), np.random.get_state())),
        torch.random.get_rng_state().clone(),
    )


def test_owned_scope_requires_factory_rigid_and_invocation_identity(staging_inputs):
    probe = staging_inputs
    recovery = _module()
    assert recovery.owns_staging(
        probe.request,
        probe.context,
        frozenset({"item"}),
        frozenset({probe.request.invocation_id}),
        probe.action.motion_generator,
    )
    assert not recovery.owns_staging(
        probe.request,
        probe.context,
        frozenset(),
        frozenset({probe.request.invocation_id}),
        probe.action.motion_generator,
    )
    assert not recovery.owns_staging(
        probe.request,
        probe.context,
        frozenset({"item"}),
        frozenset(),
        probe.action.motion_generator,
    )


def test_safe_original_does_not_build_hints_or_consume_extra_rng(
    monkeypatch, staging_inputs
):
    probe = staging_inputs
    recovery = _module()
    first = probe.make_plan(True)
    monkeypatch.setattr(MoveHeldObject, "_plan", lambda *args: first)
    monkeypatch.setattr(
        recovery,
        "build_staging_variants",
        lambda *args: pytest.fail("safe original built hints"),
    )
    before = _state()
    result = probe.action._plan(probe.request, probe.context)
    assert result.joint_trajectory is first.joint_trajectory
    assert result.commands is first.commands
    assert (
        result.expected_effects.held_object_updates["active"].semantics.entity_id
        == "item"
    )
    after = _state()
    assert before[0] == after[0] and torch.equal(before[1], after[1])


@pytest.mark.parametrize(
    "exemption",
    [
        "exact",
        "waypoints",
        "batch",
        "environment",
        "motion_gen",
        "dynamic",
        "direct",
        "handoff",
        "missing_held",
        "wrong_invocation",
        "unbound_rigid",
    ],
)
def test_exempt_requests_never_own_staging(staging_inputs, exemption):
    probe = staging_inputs
    request, context, generator = (
        probe.request,
        probe.context,
        probe.action.motion_generator,
    )
    rigid_ids, invocation_ids = frozenset({"item"}), frozenset({request.invocation_id})
    if exemption == "exact":
        request = replace(request, goal=replace(request.goal, world_yaw_free=False))
    elif exemption == "waypoints":
        request = replace(
            request,
            goal=replace(
                request.goal, object_target_pose=request.goal.object_target_pose[:, :1]
            ),
        )
    elif exemption == "batch":
        context.batch_size = 2
    elif exemption == "environment":
        context.env_ids = torch.tensor([1])
    elif exemption == "motion_gen":
        request = replace(
            request, motion_policy=replace(request.motion_policy, strategy="motion_gen")
        )
    elif exemption == "dynamic":
        request = replace(
            request,
            motion_policy=replace(request.motion_policy, dynamic_collision_mode="auto"),
        )
    elif exemption == "direct":
        generator.direct_transport_planning = True
    elif exemption == "handoff":
        held = context.task.get_held_object("active")
        context.task = TaskState(
            1, "cpu", held_objects={"active": held, "passive": held}
        )
    elif exemption == "missing_held":
        context.task = TaskState(1, "cpu")
    elif exemption == "wrong_invocation":
        invocation_ids = frozenset({"unrelated:0"})
    else:
        rigid_ids = frozenset({"other"})
    assert not _module().owns_staging(
        request, context, rigid_ids, invocation_ids, generator
    )


@pytest.fixture
def projected_hint(staging_inputs):
    # Distal meshes are 20 cm wide; route displacement must use measured spans.
    tcp, roots = {}, {}
    for part, x in (("active", 0.6), ("passive", -0.6)):
        tcp[part] = np.eye(4)
        tcp[part][0, 3] = x
        roots[part] = tcp[part].copy()
        roots[part][0, 3] = x / 2
    corners = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]])
    return dict(
        active="active",
        passive="passive",
        tcp=tcp,
        roots=roots,
        points={part: corners + pose[:3, 3] for part, pose in tcp.items()},
        hand_owner_fk_error={"active": 0.05, "passive": 0.0},
    )


def test_two_geometry_routes_preserve_final_attachment_orientation_and_height(
    monkeypatch, staging_inputs, projected_hint
):
    recovery, probe = _module(), staging_inputs
    monkeypatch.setattr(recovery, "_proxy_snapshot", lambda *args: projected_hint)
    original = probe.request.goal.object_target_pose.clone()
    attachment = probe.context.task.get_held_object("active").object_to_eef.clone()
    variants = recovery.build_staging_variants(
        probe.request, probe.context, probe.robot
    )
    assert [label for label, _, _ in variants] == ["B", "C"]
    for _, request, report in variants:
        poses = request.goal.object_target_pose
        assert poses.shape == (1, 3, 4, 4)
        assert torch.equal(poses[:, -1], original[:, -1])
        assert torch.equal(poses[:, 1, 2, 3], original[:, 0, 2, 3])
        assert torch.equal(
            poses[:, :2, :3, :3], original[:, :1, :3, :3].expand(-1, 2, -1, -1)
        )
        assert report["shift_m"] <= report["shift_bound_m"] == pytest.approx(0.4)
        assert report["shift_m"] == pytest.approx(0.1)
        assert (
            request.binding is probe.request.binding
            and request.motion_policy is probe.request.motion_policy
        )
    assert torch.equal(probe.request.goal.object_target_pose, original)
    assert torch.equal(
        probe.context.task.get_held_object("active").object_to_eef, attachment
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "bound",
        "direction",
        "observation",
        "acquisition_rotation",
        "stage_height",
        "attachment",
        "active_ownership",
    ],
)
def test_geometry_refuses_unqualified_hints(
    monkeypatch, staging_inputs, projected_hint, invalid
):
    recovery, probe = _module(), staging_inputs
    monkeypatch.setattr(recovery, "_proxy_snapshot", lambda *args: projected_hint)
    request = probe.request
    if invalid == "bound":
        projected_hint["points"]["passive"][:, 0] += 3
    elif invalid == "direction":
        projected_hint["tcp"]["passive"] = projected_hint["tcp"]["active"].copy()
        projected_hint["roots"]["passive"] = projected_hint["roots"]["active"].copy()
    elif invalid == "observation":
        probe.context.scene.entities["item"].confidence = 0
    elif invalid in {"acquisition_rotation", "stage_height"}:
        targets = request.goal.object_target_pose.clone()
        if invalid == "acquisition_rotation":
            targets[:, 0, :3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        else:
            targets[:, 0, 2, 3] = -0.1
        request = replace(
            request, goal=replace(request.goal, object_target_pose=targets)
        )
    elif invalid == "attachment":
        probe.context.task = SimpleNamespace(
            get_held_object=lambda _: SimpleNamespace(
                semantics=SimpleNamespace(entity_id="item"),
                object_to_eef=torch.zeros(1, 4, 4),
            )
        )
    else:
        probe.robot.get_joint_ids = lambda name: (1, 0)
    if invalid in {"bound", "direction"}:
        assert all(
            candidate is None
            for _, candidate, _ in recovery.build_staging_variants(
                request, probe.context, probe.robot
            )
        )
    else:
        with pytest.raises(ValueError):
            recovery.build_staging_variants(request, probe.context, probe.robot)


@pytest.mark.parametrize("outcome", ["first", "second", "failed", "error", "unsafe"])
def test_recovery_replays_pre_original_rng_and_restores_legacy_stop(
    monkeypatch, staging_inputs, outcome
):
    recovery, probe = _module(), staging_inputs
    before = held_alignment_recovery._RandomState.capture(set())
    expected_draw = _draw()
    after = _state()
    attempts = []
    first = probe.make_plan(False)
    scope = SimpleNamespace(
        rejected_raw_success=1, calls=[None] * 8, fallback_attempted=True
    )
    monkeypatch.setattr(
        recovery,
        "build_staging_variants",
        lambda request, *args: [
            (label, request, {"variant": label}) for label in ("B", "C")
        ],
    )

    def build(request):
        attempts.append(_draw())
        if outcome == "error":
            raise RuntimeError("candidate planner unavailable")
        plan = probe.make_plan(
            outcome in {"first", "unsafe"} or outcome == "second" and len(attempts) == 2
        )
        if outcome == "unsafe":
            frames = list(plan.commands.frames)
            commands = list(frames[1].commands)
            commands[0] = replace(
                commands[0],
                payload=replace(commands[0].payload, positions=torch.ones(1, 2)),
            )
            frames[1] = replace(frames[1], commands=tuple(commands))
            plan = replace(plan, commands=replace(plan.commands, frames=tuple(frames)))
        return plan

    result = recovery.recover_staging(
        first, probe.request, probe.context, probe.robot, scope, before, build
    )
    assert all(draw == expected_draw for draw in attempts)
    restored = _state()
    assert restored[0] == after[0] and torch.equal(restored[1], after[1])
    assert len(attempts) == (1 if outcome in {"first", "error"} else 2)
    assert bool(result.plan_success.all()) == (outcome in {"first", "second"})


@pytest.mark.parametrize("invalid_start", ["collision", "unavailable"])
def test_observed_start_refusal_prevents_geometry_and_planning(
    monkeypatch, staging_inputs, invalid_start
):
    recovery, probe = _module(), staging_inputs
    certificate = (
        {"valid": False, "reason": "collision"}
        if invalid_start == "collision"
        else {"valid": False, "unavailable": "missing FK"}
    )
    monkeypatch.setattr(recovery, "_inspect_pick_motion", lambda *args: certificate)
    monkeypatch.setattr(
        recovery,
        "build_staging_variants",
        lambda *args: pytest.fail("invalid start built hints"),
    )
    first = probe.make_plan(False)
    scope = SimpleNamespace(
        rejected_raw_success=1, calls=[None] * 8, fallback_attempted=True
    )
    result = recovery.recover_staging(
        first,
        probe.request,
        probe.context,
        probe.robot,
        scope,
        held_alignment_recovery._RandomState.capture(set()),
        lambda *args: pytest.fail("invalid start planned"),
    )
    assert not result.plan_success.any()


def test_initial_observed_hand_velocity_boundary_is_not_replaced_by_grasp(
    monkeypatch, staging_inputs
):
    recovery, probe = _module(), staging_inputs
    probe.context.robot.qpos[:, 2] = 0
    positions = probe.context.robot.qpos[:, None].repeat(1, 3, 1)
    positions[:, :, 2] = 0.02
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=probe.context.env_ids, step_dt=0.04
    )
    plan = replace(
        probe.make_plan(True),
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=torch.tensor([True])
        ),
    )
    limits = torch.ones(1, 6)
    limits[:, 2] = 0.1
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *args: limits)
    monkeypatch.setattr(
        held_alignment_recovery, "_joint_velocity_limits", lambda *args: limits
    )
    assert held_alignment_recovery.qualify_alignment_plan(
        probe.request, probe.context, plan, probe.robot
    ).plan_success.all()
    certificate = recovery._inspect_pick_plan(
        probe.robot, probe.request, probe.context, plan
    )
    assert certificate == {"valid": False, "reason": "initial_velocity_rejected"}


def test_failed_original_rebuilds_at_most_two_complete_shared_plans(
    monkeypatch, staging_inputs
):
    probe = staging_inputs
    recovery = _module()
    requests = []

    def planner(self, request, context):
        requests.append(request)
        scope = held_alignment_recovery._SCOPE.get()
        scope.rejected_raw_success = 1
        scope.calls = [None] * 8
        return probe.make_plan(len(requests) == 3)

    def legacy(scope, request, context, plan, build):
        scope.fallback_attempted = True
        return plan

    monkeypatch.setattr(MoveHeldObject, "_plan", planner)
    monkeypatch.setattr(held_alignment_recovery, "recover", legacy)
    monkeypatch.setattr(
        recovery,
        "build_staging_variants",
        lambda request, *args: [
            (label, request, {"variant": label}) for label in ("B", "C")
        ],
    )
    result = probe.action._plan(probe.request, probe.context)
    assert result.plan_success.all(), dict(result.diagnostics.metadata)
    assert len(requests) == 3
    assert (
        result.expected_effects.held_object_updates["active"].semantics.entity_id
        == "item"
    )


@pytest.fixture
def measured_geometry(monkeypatch, staging_inputs):
    probe = staging_inputs
    poses, solvers, joints, mesh_owners = {}, {}, [], {}
    for part, sign in (("active", 1), ("passive", -1)):
        root, middle, end, hand = (
            f"{part}_{name}" for name in ("root", "middle", "end", "hand")
        )
        solvers[part] = SimpleNamespace(
            root_link_name=root, end_link_name=end, get_tcp=lambda: np.eye(4)
        )
        for name, x in ((root, 0.3), (middle, 0.5), (end, 0.6), (hand, 0.6)):
            poses[name] = np.eye(4)
            poses[name][0, 3] = sign * x
        for parent, child, kind in (
            (root, middle, "revolute"),
            (middle, end, "revolute"),
            (end, hand, "fixed"),
        ):
            joints.append(SimpleNamespace(parent=parent, child=child, type=kind))
        mesh_owners[f"{part}_mesh"] = end
        mesh_owners[f"{part}_hand_mesh"] = hand
    corners = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]])
    mesh_local = np.eye(4)
    mesh_local[0, 3] = 0.02

    class Graph:
        nodes_geometry = tuple(mesh_owners)
        transforms = SimpleNamespace(parents=mesh_owners)

        def __getitem__(self, node):
            return poses[mesh_owners[node]] @ mesh_local, "mesh"

    model = SimpleNamespace(
        robot=SimpleNamespace(joints=joints),
        link_map=poses,
        joint_map={},
        update_cfg=lambda value: None,
        get_transform=lambda node, frame_from=None, collision_geometry=False: (
            mesh_local if frame_from is not None else poses[node]
        ),
        collision_scene=SimpleNamespace(
            graph=Graph(), geometry={"mesh": SimpleNamespace(vertices=corners)}
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "yourdfpy",
        SimpleNamespace(URDF=SimpleNamespace(load=lambda *args, **kwargs: model)),
    )
    measured = {link: pose.copy() for link, pose in poses.items()}
    measured["active_hand"][0, 3] += 0.05
    probe.robot.cfg = SimpleNamespace(fpath="owned_robot.urdf")
    probe.robot.control_parts = ("active", "passive", "hand")
    probe.robot.joint_names = tuple(f"joint_{index}" for index in range(6))
    probe.robot.get_solver = lambda part: solvers.get(part)
    probe.robot.get_local_pose = lambda **kwargs: torch.eye(4)[None]
    probe.robot.get_link_pose = lambda link, **kwargs: torch.tensor(measured[link])[
        None
    ]
    probe.robot.compute_fk = lambda qpos, name, **kwargs: torch.tensor(
        poses[solvers[name].end_link_name]
    )[None]
    probe.robot.get_qpos = lambda: probe.context.robot.qpos.clone()
    return SimpleNamespace(probe=probe, poses=poses, measured=measured, model=model)


def test_geometry_uses_measured_owner_times_mesh_local_with_hand_error_diagnostic(
    measured_geometry,
):
    fixture = measured_geometry
    snapshot = _module()._proxy_snapshot(
        fixture.probe.robot, fixture.probe.context, "active"
    )
    assert snapshot["hand_owner_fk_error"]["active"] == pytest.approx(0.05)
    assert snapshot["points"]["active"][:, 0].max() == pytest.approx(0.77)
    assert snapshot["points"]["passive"][:, 0].max() == pytest.approx(-0.48)


@pytest.mark.parametrize(
    "invalid_fk", ["root", "middle", "end", "tcp", "arena_root", "qpos_changed"]
)
def test_geometry_refuses_invalid_arm_chain_root_or_tcp_calibration(
    measured_geometry, invalid_fk
):
    fixture = measured_geometry
    if invalid_fk in {"root", "middle", "end"}:
        fixture.measured[f"active_{invalid_fk}"][0, 3] += 3e-4
    elif invalid_fk == "tcp":
        fixture.probe.robot.compute_fk = lambda *args, **kwargs: torch.eye(4)[None]
    elif invalid_fk == "arena_root":
        fixture.probe.context.robot.root_pose[:, 0, 3] = 3e-4
    else:
        fixture.probe.robot.get_qpos = lambda: fixture.probe.context.robot.qpos + 0.1
    with pytest.raises(ValueError):
        _module()._proxy_snapshot(fixture.probe.robot, fixture.probe.context, "active")


def test_allowlist_uses_compiled_align_occurrences_not_object_name_alone():
    recovery = _module()
    calls = []
    for index, (call_id, object_id, preserve, target) in enumerate(
        [
            ("gen_sim.align_held", "item", False, "target"),
            ("gen_sim.align_held", "item", True, "target"),
            ("gen_sim.align_held", "item", False, "current_object_pose"),
            ("gen_sim.align_held", "articulated", False, "target"),
            ("custom.align", "item", False, "target"),
        ]
    ):
        call = SimpleNamespace(
            call_id=call_id,
            arguments={"object": object_id, "preserve_yaw": preserve, "target": target},
        )
        calls.append(
            SimpleNamespace(
                name=f"segment_{index}",
                segment_id=f"s{index}",
                calls=(SimpleNamespace(call=call, segment_call_index=0),),
            )
        )
    program = SimpleNamespace(program_id="compiled", iter_segments=lambda: iter(calls))
    assert recovery.bind_staging_calls(program, frozenset({"item"})) == (
        "compiled/s0:0",
    )


@pytest.mark.parametrize(
    "failure", [None, "initial_hand", "strip_metadata", "mutate_nested_metadata"]
)
def test_owned_staging_transform_rechecks_real_initial_hand_boundary(
    monkeypatch, staging_inputs, failure
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    probe = staging_inputs
    probe.context.robot.qpos[:, 2] = 0
    positions = probe.context.robot.qpos[:, None].repeat(1, 3, 1)
    positions[:, :, 2] = 0.02

    def make_plan(dt, metadata):
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=probe.context.env_ids, step_dt=dt
        )
        return replace(
            probe.make_plan(True),
            joint_trajectory=trajectory,
            commands=probe.action._joint_command_sequence(
                probe.request, trajectory, active_mask=torch.tensor([True])
            ),
            diagnostics=PlannerDiagnostics(backend="transformed", metadata=metadata),
        )

    first = make_plan(0.04, {"gen_sim_staging_recovery": {"accepted": True}})
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = probe.action._planning_services
    engine._actions = {probe.request.skill_id: probe.action}
    engine._motion_samples, engine._articulation_calls = {}, {}
    engine._cartesian_calls = frozenset()
    monkeypatch.setattr(
        AtomicActionEngine,
        "_plan_request",
        lambda self, selected, context, plan_transform=None: (
            first
            if plan_transform is None
            else plan_transform(selected, context, first)
        ),
    )

    def transform(request, context, plan):
        if failure == "mutate_nested_metadata":
            plan.diagnostics.metadata["gen_sim_staging_recovery"]["accepted"] = False
        metadata = (
            {} if failure == "strip_metadata" else dict(plan.diagnostics.metadata)
        )
        return make_plan(0.04 if failure is None else 0.004, metadata)

    result = engine._plan_request(
        probe.request, probe.context, plan_transform=transform
    )
    assert bool(result.plan_success.all()) == (failure is None)
    if failure is not None:
        assert result.diagnostics.failure.code == "held_staging_safety_failed"
        assert (
            result.diagnostics.metadata["gen_sim_staging_recovery"][
                "post_transform_check"
            ]["reason"]
            == "initial_velocity_rejected"
        )


@pytest.mark.parametrize(
    "exemption", ["no_transform", "safe_original", "unbound_invocation"]
)
def test_nonstaging_transform_does_not_add_boundary_search(
    monkeypatch, staging_inputs, exemption
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.gen_sim.task_engine._task_program import pick_safety
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    probe = staging_inputs
    first = probe.make_plan(True)
    if exemption != "safe_original":
        first = replace(
            first,
            diagnostics=replace(
                first.diagnostics,
                metadata={"gen_sim_staging_recovery": {"accepted": True}},
            ),
        )
    if exemption == "unbound_invocation":
        probe.action.staging_contract = replace(
            probe.action.staging_contract, invocation_ids=frozenset()
        )
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = probe.action._planning_services
    engine._actions = {probe.request.skill_id: probe.action}
    engine._motion_samples, engine._articulation_calls = {}, {}
    engine._cartesian_calls = frozenset()
    monkeypatch.setattr(
        AtomicActionEngine,
        "_plan_request",
        lambda self, selected, context, plan_transform=None: (
            first
            if plan_transform is None
            else plan_transform(selected, context, first)
        ),
    )
    monkeypatch.setattr(
        pick_safety,
        "_inspect_pick_plan",
        lambda *args: pytest.fail("nonstaging extra gate"),
    )
    result = engine._plan_request(
        probe.request,
        probe.context,
        plan_transform=None if exemption == "no_transform" else lambda *args: first,
    )
    assert result.plan_success.all()


@pytest.mark.parametrize("preserve_yaw", [False, True])
def test_staging_revision_fingerprint_is_conditional_and_caller_owned_inputs_unchanged(
    monkeypatch, tmp_path, preserve_yaw
):
    import trimesh

    from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
    from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config

    mesh_path = tmp_path / "item.glb"
    trimesh.creation.box(extents=(0.06, 0.24, 0.06)).export(mesh_path)
    item = {
        "uid": "item",
        "shape": {"shape_type": "Mesh", "fpath": str(mesh_path)},
        "init_pos": [0.1, -0.2, 0.85],
        "init_rot": [90.0, 0.0, 0.0],
    }
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.72]},
        "init_pos": [0.0, 0.0, 0.36],
    }
    scene = PreparedScene(
        source_config_path=tmp_path / "scene.yaml",
        scene_dir=tmp_path,
        planner_objects=(
            {**item, "runtime_uid": "item", "role": "rigid_object"},
            {**table, "runtime_uid": "table", "role": "background"},
        ),
        background=(table,),
        rigid_objects=(item,),
        articulations=(),
        uid_map={"item": "item", "table": "table"},
        table_top_z=0.72,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )
    graph = {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "scope",
        "instruction": "Align the held object",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {
            "staging": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, -0.2, 0.85],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        },
        "nodes": [
            {
                "id": "pick",
                "task_type": "E2",
                "task_instance_id": "group",
                "role": "primary",
                "depends_on": [],
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.pick",
                    "arguments": {"object": "item", "target": "staging"},
                    "resources": {"primary": "left"},
                },
            },
            {
                "id": "align",
                "task_type": "E2",
                "task_instance_id": "group",
                "role": "primary",
                "depends_on": ["pick"],
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.align_held",
                    "arguments": {
                        "object": "item",
                        "target": "staging",
                        "preserve_yaw": preserve_yaw,
                    },
                    "resources": {"primary": "left"},
                },
            },
        ],
        "task_groups": [
            {
                "id": "group",
                "task_type": "E2",
                "node_ids": ["pick", "align"],
                "depends_on": [],
                "success": {"kind": "call_completed"},
            }
        ],
        "success": {"kind": "all_task_groups"},
    }
    graph["targets"] = {
        name: deepcopy(graph["targets"]["staging"])
        for name in ("group_upright_target", "group_upright_staging_target")
    }
    graph["nodes"][0]["call"]["arguments"]["target"] = "group_upright_target"
    graph["nodes"][1]["call"]["arguments"]["target"] = "group_upright_staging_target"
    for name, call_id, arguments in (
        (
            "place",
            "simulation.place_relative",
            {"object": "item", "reference": "table", "relation": "on"},
        ),
        (
            "clear",
            "gen_sim.clear_released",
            {"object": "item", "target": "group_upright_staging_target"},
        ),
        ("park", "simulation.park", {}),
    ):
        graph["nodes"].append(
            {
                "id": name,
                "task_type": "E2",
                "task_instance_id": "group",
                "role": "primary",
                "depends_on": [graph["nodes"][-1]["id"]],
                "call": {
                    "kind": "registered",
                    "call_id": call_id,
                    "arguments": arguments,
                    "resources": {"primary": "left"},
                },
            }
        )
    graph["task_groups"][0]["node_ids"] = [node["id"] for node in graph["nodes"]]
    original = deepcopy(graph)
    _, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
    )
    arguments = dict(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.deployment.parent,
    )
    profile = deepcopy(arguments["skill_profile"])
    first = load_deployment(**arguments)
    assert bool(first.integration.adapter_factory.staging_contract.invocation_ids) is (
        not preserve_yaw
    )
    from embodichain.gen_sim.task_engine._task_program import pick_safety

    opening_revision = pick_safety.PICK_OPENING_RECOVERY_REVISION
    monkeypatch.setattr(pick_safety, "PICK_OPENING_RECOVERY_REVISION", 999)
    opening_changed = load_deployment(**arguments)
    assert (
        opening_changed.integration.integration_fingerprint
        != first.integration.integration_fingerprint
    )
    monkeypatch.setattr(pick_safety, "PICK_OPENING_RECOVERY_REVISION", opening_revision)
    monkeypatch.setattr(_module(), "STAGING_RECOVERY_REVISION", 999)
    second = load_deployment(**arguments)
    assert (
        first.integration.integration_fingerprint
        != second.integration.integration_fingerprint
    ) is not preserve_yaw
    assert graph == original
    assert arguments["skill_profile"] == profile


@pytest.mark.parametrize("has_drawers", [False, True])
def test_factory_binds_only_native_rigids_and_compiled_calls_without_drawer_capability(
    monkeypatch, has_drawers
):
    from unittest.mock import Mock

    from embodichain.gen_sim.task_engine._task_program import assembly, drawer_runtime
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        SimulationRigidObjectBinding,
    )
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        RobotSkillProfile,
        SkillPolicyPreset,
    )

    registration = SimpleNamespace(
        scene_binding=SimpleNamespace(
            rigid_objects=(SimulationRigidObjectBinding("rigid", "native"),)
        ),
        validate_engine=Mock(),
    )
    robot = object()

    def initialize(factory, *args, **kwargs):
        factory._registration, factory._robot, factory._simulation = (
            registration,
            robot,
            object(),
        )
        factory._step_dt, factory._segment_policy_port = 0.04, object()
        factory._robot_profile_binding = SimpleNamespace(profile_id="profile")
        factory._grasp_pose_generators = {}
        factory._create_motion_generator = Mock(
            return_value=SimpleNamespace(
                robot=robot, direct_transport_planning=has_drawers
            )
        )

    monkeypatch.setattr(assembly.SimulationTaskProgramFactory, "__init__", initialize)
    monkeypatch.setattr(assembly, "TaskStabilityPort", Mock())
    monkeypatch.setattr(drawer_runtime, "DrawerObservation", lambda *args: object())
    standard, drawer = Mock(), Mock()
    monkeypatch.setattr(assembly, "GenSimActionEngine", Mock(return_value=standard))
    monkeypatch.setattr(
        drawer_runtime, "DrawerPlacementEngine", Mock(return_value=drawer)
    )
    factory = assembly._TaskFactory(
        constraints={},
        staging_contract=StagingContract(
            rigid_entity_ids=(frozenset() if has_drawers else frozenset({"rigid"})),
            invocation_ids=(
                frozenset() if has_drawers else frozenset({"compiled/align:0"})
            ),
            retention_sources={
                **(
                    {}
                    if has_drawers
                    else {
                        "compiled/align:0": {
                            "predecessor_call_id": "simulation.pick",
                            "predecessor_invocation_id": "compiled/pick:0",
                            "entity_id": "rigid",
                            "resource_id": "active",
                        }
                    }
                ),
            },
        ),
        runtime_preset="verified",
        drawer_routes=(object(),) if has_drawers else (),
    )
    profile = Mock(spec=RobotSkillProfile, profile_id="profile")
    profile.presets = {
        "verified": SkillPolicyPreset(
            preset_id="verified",
            action_option_templates={},
            effect_assurance=EffectAssurance.VERIFIED,
            effect_monitors={
                call_id: EffectMonitorRef("builtin.composite_effect", "1")
                for call_id in ("simulation.pick", "gen_sim.align_held")
            },
        )
    }
    profile.action_control_profiles.return_value = {}
    engine = factory.create_atomic_action_engine(profile)
    transport = next(
        call.args[0]
        for call in engine.register.call_args_list
        if isinstance(call.args[0], actions.GenSimMoveHeldObject)
    )
    assert transport.staging_contract.rigid_entity_ids == (
        frozenset() if has_drawers else frozenset({"rigid"})
    )
    assert transport.staging_contract.invocation_ids == (
        frozenset() if has_drawers else frozenset({"compiled/align:0"})
    )


@pytest.mark.parametrize(
    "scope",
    [
        "verified",
        "projected",
        "unknown",
        "missing_pick_monitor",
        "missing_align_monitor",
        "custom_monitor",
    ],
)
def test_same_command_override_requires_factory_verified_attachment_monitor_bindings(
    scope,
):
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        SkillPolicyPreset,
    )

    monitor = EffectMonitorRef(
        "builtin.composite_effect" if scope != "custom_monitor" else "custom.monitor",
        "1",
    )
    monitors = {
        call_id: monitor for call_id in ("simulation.pick", "gen_sim.align_held")
    }
    if scope == "projected":
        monitors = {}
    elif scope == "missing_pick_monitor":
        monitors.pop("simulation.pick")
    elif scope == "missing_align_monitor":
        monitors.pop("gen_sim.align_held")
    preset = SkillPolicyPreset(
        preset_id="selected",
        action_option_templates={},
        effect_assurance=(
            EffectAssurance.PROJECTED
            if scope == "projected"
            else EffectAssurance.VERIFIED
        ),
        effect_monitors=monitors,
    )
    profile = SimpleNamespace(presets={"selected": preset})
    sources = (
        ("compiled/align:0", "compiled/pick:0", "simulation.pick", "item", "active"),
    )
    actual = _module().verified_retention_sources(
        profile, "unknown" if scope == "unknown" else "selected", sources
    )
    assert bool(actual) == (scope == "verified")


@pytest.fixture
def retained_hand_inputs(monkeypatch, staging_inputs):
    probe = staging_inputs
    probe.context.robot.qpos[:, 2] = 0
    probe.context.robot.qvel = torch.zeros_like(probe.context.robot.qpos)
    desired = probe.context.robot.qpos.clone()
    desired[:, 2] = 0.02
    target_velocity = torch.zeros_like(desired)
    probe.robot.get_qpos = lambda target=False: (
        desired if target else probe.context.robot.qpos
    )
    probe.robot.get_qvel = lambda target=False: (
        target_velocity if target else probe.context.robot.qvel
    )
    limits = torch.ones(1, 6)
    limits[:, 2] = 0.1
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *args: limits)
    monkeypatch.setattr(
        held_alignment_recovery, "_joint_velocity_limits", lambda *args: limits
    )
    positions = probe.context.robot.qpos[:, None].repeat(1, 3, 1)
    positions[:, :, 2] = desired[:, None, 2]
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=probe.context.env_ids, step_dt=0.04
    )
    plan = replace(
        probe.make_plan(True),
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=torch.tensor([True])
        ),
    )
    source = {
        "preset_id": "verified",
        "predecessor_call_id": "simulation.pick",
        "entity_id": "item",
        "resource_id": "active",
        "attachment_monitor": {
            "monitor_id": "builtin.composite_effect",
            "revision": "1",
        },
        "retention_monitor": {
            "monitor_id": "builtin.composite_effect",
            "revision": "1",
        },
    }
    return SimpleNamespace(
        probe=probe,
        plan=plan,
        desired=desired,
        target_velocity=target_velocity,
        source=source,
    )


def test_already_applied_retained_hand_command_uses_previous_desired_boundary_only(
    retained_hand_inputs,
):
    fixture = retained_hand_inputs
    probe = fixture.probe
    before_qpos, before_positions = (
        probe.context.robot.qpos.clone(),
        fixture.plan.joint_trajectory.positions.clone(),
    )
    strict = _module()._inspect_pick_plan(
        probe.robot, probe.request, probe.context, fixture.plan
    )
    assert strict == {"valid": False, "reason": "initial_velocity_rejected"}
    certificate = _module().inspect_staging_plan(
        probe.robot,
        probe.request,
        probe.context,
        fixture.plan,
        retention_source=fixture.source,
    )
    assert certificate["valid"]
    proof = certificate["retained_hand_boundary"]
    assert proof["source_api"] == "Robot.get_qpos(target=True)/get_qvel(target=True)"
    assert proof["prior_target_qpos"] == [[pytest.approx(0.02)]]
    assert proof["actual_qpos"] == [[0.0]]
    assert proof["actual_qvel"] == proof["prior_target_qvel"] == [[0.0]]
    assert torch.equal(probe.context.robot.qpos, before_qpos)
    assert torch.equal(fixture.plan.joint_trajectory.positions, before_positions)


@pytest.mark.parametrize(
    "invalid",
    [
        "changed_target",
        "open_target",
        "unknown_target",
        "target_shape",
        "target_nonfinite",
        "changed_target_velocity",
        "actual_velocity",
        "actual_mismatch",
        "hand_ownership",
        "no_verified_source",
        "target_changes_during_check",
    ],
)
def test_retained_boundary_refuses_unproven_or_new_actuation(
    monkeypatch, retained_hand_inputs, invalid
):
    fixture, recovery = retained_hand_inputs, _module()
    probe, source = fixture.probe, fixture.source
    if invalid == "changed_target":
        fixture.desired[:, 2] += 1e-6
    elif invalid == "open_target":
        fixture.desired[:, 2] = 0
    elif invalid == "unknown_target":
        probe.robot.get_qpos = lambda target=False: (
            (_ for _ in ()).throw(RuntimeError("no native target reader"))
            if target
            else probe.context.robot.qpos
        )
    elif invalid == "target_shape":
        probe.robot.get_qpos = lambda target=False: (
            fixture.desired[:, :3] if target else probe.context.robot.qpos
        )
    elif invalid == "target_nonfinite":
        fixture.desired[:, 2] = torch.nan
    elif invalid == "changed_target_velocity":
        fixture.target_velocity[:, 2] = 0.01
    elif invalid == "actual_velocity":
        probe.context.robot.qvel[:, 2] = 0.101
    elif invalid == "actual_mismatch":
        probe.robot.get_qpos = lambda target=False: (
            fixture.desired if target else probe.context.robot.qpos + 0.01
        )
    elif invalid == "hand_ownership":
        original = probe.robot.get_joint_ids
        probe.robot.get_joint_ids = lambda name: (
            (3,) if name == "hand" else original(name)
        )
    elif invalid == "no_verified_source":
        source = None
    else:
        original = recovery._inspect_pick_plan

        def mutate_after_gate(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("_initial_command_qpos") is not None:
                fixture.desired[:, 2] = 0
            return result

        monkeypatch.setattr(recovery, "_inspect_pick_plan", mutate_after_gate)
    certificate = recovery.inspect_staging_plan(
        probe.robot, probe.request, probe.context, fixture.plan, retention_source=source
    )
    assert not certificate["valid"]


def test_private_boundary_override_cannot_replace_observed_arm_or_passive_joints(
    retained_hand_inputs,
):
    fixture = retained_hand_inputs
    probe = fixture.probe
    for changed in (0, 3):
        baseline = probe.context.robot.qpos.clone()
        baseline[:, 2] = fixture.desired[:, 2]
        baseline[:, changed] += 0.01
        certificate = _module()._inspect_pick_plan(
            probe.robot,
            probe.request,
            probe.context,
            fixture.plan,
            _initial_command_qpos=baseline,
        )
        assert not certificate["valid"] and "unavailable" in certificate


def test_strict_success_does_not_read_native_previous_command(
    monkeypatch, retained_hand_inputs
):
    fixture = retained_hand_inputs
    probe = fixture.probe
    probe.context.robot.qpos[:, 2] = 0.02
    probe.robot.get_qpos = lambda target=False: (
        pytest.fail("strict success read native target")
        if target
        else probe.context.robot.qpos
    )
    probe.robot.get_qvel = lambda *args, **kwargs: pytest.fail(
        "strict success read native velocity"
    )
    certificate = _module().inspect_staging_plan(
        probe.robot,
        probe.request,
        probe.context,
        fixture.plan,
        retention_source=fixture.source,
    )
    assert certificate["valid"]
    assert "retained_hand_boundary" not in certificate


@pytest.mark.parametrize(
    "change", [None, "release", "velocity", "long_hold_release", "unknown_reader"]
)
def test_retained_staging_transform_rechecks_fresh_command_identity(
    monkeypatch, retained_hand_inputs, change
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    fixture, recovery = retained_hand_inputs, _module()
    probe = fixture.probe
    first_certificate = recovery.inspect_staging_plan(
        probe.robot,
        probe.request,
        probe.context,
        fixture.plan,
        retention_source=fixture.source,
    )
    assert first_certificate["valid"]
    first = replace(
        fixture.plan,
        diagnostics=replace(
            fixture.plan.diagnostics,
            metadata={
                "gen_sim_staging_recovery": {
                    "accepted": True,
                    "retained_hand_boundary": first_certificate[
                        "retained_hand_boundary"
                    ],
                }
            },
        ),
    )
    probe.action.staging_contract = replace(
        probe.action.staging_contract,
        invocation_ids=frozenset({probe.request.invocation_id}),
        retention_sources={probe.request.invocation_id: fixture.source},
    )
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services, engine._actions = probe.action._planning_services, {
        probe.request.skill_id: probe.action
    }
    engine._motion_samples, engine._articulation_calls, engine._cartesian_calls = (
        {},
        {},
        frozenset(),
    )
    monkeypatch.setattr(
        AtomicActionEngine,
        "_plan_request",
        lambda self, selected, context, plan_transform=None: plan_transform(
            selected, context, first
        ),
    )

    def transform(request, context, plan):
        if change in {"release", "long_hold_release"}:
            fixture.desired[:, 2] = 0
        elif change == "velocity":
            fixture.target_velocity[:, 2] = 0.01
        elif change == "unknown_reader":
            probe.robot.get_qpos = lambda target=False: (
                (_ for _ in ()).throw(RuntimeError("target reader disappeared"))
                if target
                else context.robot.qpos
            )
        if change == "long_hold_release":
            trajectory = TimedTrajectory.from_uniform_step(
                plan.joint_trajectory.positions, env_ids=context.env_ids, step_dt=0.4
            )
            return replace(
                plan,
                joint_trajectory=trajectory,
                commands=probe.action._joint_command_sequence(
                    request, trajectory, active_mask=torch.tensor([True])
                ),
            )
        return plan

    result = engine._plan_request(
        probe.request, probe.context, plan_transform=transform
    )
    assert bool(result.plan_success.all()) == (change is None)
    if change is not None:
        assert result.diagnostics.failure.code == "held_staging_safety_failed"


def test_retained_boundary_override_keeps_pre_original_and_legacy_rng_contract(
    monkeypatch, retained_hand_inputs
):
    fixture, recovery = retained_hand_inputs, _module()
    probe = fixture.probe
    before = held_alignment_recovery._RandomState.capture(set())
    expected = _draw()
    after = _state()
    original = probe.make_plan(False)
    scope = SimpleNamespace(
        rejected_raw_success=1, calls=[None] * 8, fallback_attempted=True
    )
    monkeypatch.setattr(
        recovery,
        "build_staging_variants",
        lambda request, *args: [
            ("B", request, {"variant": "B"}),
            ("C", request, {"variant": "C"}),
        ],
    )

    def build(request):
        assert _draw() == expected
        return fixture.plan

    result = recovery.recover_staging(
        original,
        probe.request,
        probe.context,
        probe.robot,
        scope,
        before,
        build,
        retention_source=fixture.source,
    )
    assert result.plan_success.all()
    assert result.diagnostics.metadata["gen_sim_staging_recovery"][
        "retained_hand_boundary"
    ]["accepted"]
    restored = _state()
    assert restored[0] == after[0] and torch.equal(restored[1], after[1])
