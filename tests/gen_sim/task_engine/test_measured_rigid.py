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

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from shapely.geometry import Polygon
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program import measured_rigid as gate
from embodichain.gen_sim.task_engine._task_program import placement_evidence
from embodichain.lab.sim.atomic_actions import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    SceneAffordanceRef,
    SceneObjectRef,
)


@pytest.mark.parametrize("value", [None, "p", {}, [True], [""], [" p"], ["p", "p"]])
def test_declarations_reject_ambiguous_names(value):
    with pytest.raises(ValueError, match="exact"):
        gate.declarations(value)


def test_bind_only_exact_ordinary_pick_and_inside_place():
    calls = [
        Pick(SceneObjectRef("item")),
        Place(SceneObjectRef("item"), inside=SceneAffordanceRef("inside")),
    ]
    segments = [
        NS(name=n, segment_id=n, calls=(NS(call=c, segment_call_index=0),))
        for n, c in zip(("p", "q"), calls)
    ]
    program = NS(program_id="test", iter_segments=lambda: iter(segments))
    assert gate.bind_calls([], program, {}) == ()
    assert gate.bind_calls(["p", "q"], program, {"test/p:0": "ordinary"}) == (
        ("test/p:0", "item", ""),
        ("test/q:0", "item", "inside"),
    )
    with pytest.raises(ValueError, match="ordinary"):
        gate.bind_calls(["p"], program, {"test/p:0": "pour"})
    with pytest.raises(ValueError, match="missing"):
        gate.bind_calls(["unknown"], program, {})
    segments[1].calls[0].call = Place(
        SceneObjectRef("item"), on=SceneObjectRef("table")
    )
    with pytest.raises(ValueError, match="inside Place"):
        gate.bind_calls(["q"], program, {})


def test_native_pair_rest_and_source_contact_phase_boundary():
    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    manager = trimesh.collision.CollisionManager()
    target = np.eye(4)
    target[0, 3] = 0.025
    manager.add_object("target", mesh, transform=target)
    scene = object.__new__(gate._Scene)
    scene.obstacles = [
        dict(
            name="target",
            rest=0.001,
            manager=manager,
            bounds=mesh.bounds + target[:3, 3],
        )
    ]
    part = NS(name="pad", meshes=(mesh,), rest_offset=0.001)
    poses = np.tile(np.eye(4), (2, 1, 1, 1))
    assert scene.inspect((part,), poses, "other")["valid"]
    poses[1, 0, 0, 3] = 0.004
    rejected = scene.inspect((part,), poses, "other")
    assert not rejected["valid"] and rejected["frame"] == 1
    assert rejected["required_clearance_m"] == 0.002
    assert scene.inspect((part,), poses, "target", source_contact_from=1)["valid"]
    poses[0, 0, 0, 3] = 0.004
    assert not scene.inspect((part,), poses, "target", source_contact_from=1)["valid"]
    for bad in (float("nan"), -0.001):
        part.rest_offset = bad
        with pytest.raises(ValueError, match="rest"):
            scene.inspect((part,), poses, "target")


@dataclass(frozen=True)
class Request:
    invocation_id: str
    skill_id: str
    goal: object
    skill_options: object
    binding: object
    motion_policy: object


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


def fixture():
    container = NS(entity_id="inside", parent_id="tray", release_clearance=0.12)
    binding = NS(
        rigid_objects=(
            NS(entity_id="item", simulation_uid="native_item"),
            NS(entity_id="tray", simulation_uid="native_tray"),
        ),
        containers=(container,),
    )
    policy = gate.MeasuredRigidPolicy(
        NS(is_default_backend=True), binding, (("place", "item", "inside"),)
    )
    held = NS(semantics=NS(entity_id="item"), object_to_eef=torch.eye(4)[None])
    request = Request(
        "place",
        "place",
        PlaceGoal(SceneEntityPose("inside")),
        PlaceOptions(lift_height=0.05, preserve_current_object_orientation=True),
        NS(
            endpoint=lambda *_: NS(task_state_key="held", target=NS(control_part="arm"))
        ),
        NS(strategy="ik_interp", dynamic_collision_mode="off"),
    )
    context = NS(
        batch_size=1, env_ids=torch.tensor([0]), get_held_object=lambda _: held
    )
    result = Plan(torch.tensor([True]), PlannerDiagnostics(backend="test"))
    failed = Mock(
        return_value=Plan(torch.tensor([False]), PlannerDiagnostics(backend="test"))
    )
    return (
        policy,
        request,
        context,
        NS(actions={"place": NS(failed_plan=failed)}, robot=object()),
        result,
    )


def test_unmarked_and_preview_keep_original_without_scene_queries():
    policy, request, _, engine, result = fixture()
    build = Mock(return_value=result)
    assert (
        policy.plan(engine, replace(request, invocation_id="legacy"), None, build)
        is result
    )
    token = gate._PREVIEW.set(True)
    try:
        assert policy.plan(engine, request, None, build) is result
    finally:
        gate._PREVIEW.reset(token)
    assert build.call_count == 2


@pytest.mark.parametrize("damage", ["batch", "backend", "strategy", "collision"])
def test_unsupported_scope_fails_closed_and_publishes_capture(monkeypatch, damage):
    policy, request, context, engine, _ = fixture()
    if damage == "batch":
        context.batch_size = 2
    if damage == "backend":
        policy.simulation.is_default_backend = False
    if damage == "strategy":
        request.motion_policy.strategy = "motion_gen"
    if damage == "collision":
        request.motion_policy.dynamic_collision_mode = "on"
    capture = NS(accept=Mock(side_effect=lambda req, plan, robot: plan))
    monkeypatch.setattr(gate, "initial_plan_capture", lambda _: capture)
    build = Mock()
    assert not policy.plan(engine, request, context, build).plan_success.any()
    build.assert_not_called()
    capture.accept.assert_called_once()


def test_capture_publishes_final_and_budget_failure_and_restores_scope(monkeypatch):
    policy, request, context, engine, result = fixture()
    capture = NS(accept=Mock(side_effect=lambda req, plan, robot: plan), failed=Mock())
    monkeypatch.setattr(gate, "initial_plan_capture", lambda _: capture)

    def accepted(*_):
        assert gate.defer_initial_capture("place")
        return request, result

    policy._place = accepted
    assert policy.plan(engine, request, context, Mock()) is result
    capture.accept.assert_called_once_with(request, result, engine.robot)
    policy._place = Mock(
        side_effect=gate.PlanningBudgetExceeded("deadline", 60.0, 61.0)
    )
    with pytest.raises(gate.PlanningBudgetExceeded):
        policy.plan(engine, request, context, Mock())
    capture.failed.assert_called_once()
    assert not gate.defer_initial_capture("place")


def test_geometry_failure_is_nonretryable_and_late_success_cannot_dispatch(monkeypatch):
    policy, request, context, engine, result = fixture()
    monkeypatch.setattr(gate, "initial_plan_capture", lambda _: None)
    policy._place = Mock(side_effect=ValueError("no floor"))
    assert not policy.plan(engine, request, context, Mock()).plan_success.any()
    assert engine.actions["place"].failed_plan.call_args.kwargs["retryable"] is False
    policy._place = lambda *_: (request, result)
    clock = iter((0.0, 61.0))
    monkeypatch.setattr(gate, "time", NS(perf_counter=lambda: next(clock)))
    assert not policy.plan(engine, request, context, Mock()).plan_success.any()


@pytest.mark.parametrize("accept_at", [0, 1, 3, None])
def test_place_preserves_nominal_or_selects_first_bounded_clear_candidate(
    monkeypatch, accept_at
):
    policy, request, context, engine, result = fixture()
    policy._landing = lambda req, *_: (req, {"observed_contents": []})
    policy._screen = lambda *_: NS(parts=())
    monkeypatch.setattr(gate, "_Scene", lambda *_: NS(arm_parts=lambda *_: ()))
    heights = []

    def build(req):
        heights.append(req.skill_options.lift_height)
        assert req.goal == request.goal
        return result

    policy._place_check = lambda *_: {"valid": len(heights) - 1 == accept_at}
    if accept_at is None:
        with pytest.raises(ValueError, match="Bounded"):
            policy._place(engine, request, context, build)
        assert len(heights) == 4
    else:
        selected, plan = policy._place(engine, request, context, build)
        assert len(heights) == accept_at + 1
        assert selected.skill_options.lift_height == pytest.approx(
            (0.05, 0.10, 0.15, 0.20)[accept_at]
        )
        assert plan.diagnostics.metadata["gen_sim_measured_rigid"]["accepted"]


@pytest.mark.parametrize("prior_state", ["inside", "outside", "above", "no_space"])
def test_landing_reads_unmarked_actual_occupants_and_preserves_reservations(
    monkeypatch, prior_state
):
    policy, request, context, _, _ = fixture()
    policy.rigids["prior"] = NS(entity_id="prior", simulation_uid="native_prior")
    assert all(source != "prior" for source, _ in policy.calls.values())
    vertices = torch.tensor(
        trimesh.creation.box(extents=(0.08, 0.08, 0.04)).vertices, dtype=torch.float32
    )[None]
    size = 0.055 if prior_state == "no_space" else 0.15
    floor = Polygon([(-size, -size), (size, -size), (size, size), (-size, size)])
    monkeypatch.setattr(
        placement_evidence,
        "_prepare_placement_geometry",
        lambda *_: NS(vertices=vertices, floors=[(floor, 0.0)]),
    )
    prior_pose = torch.eye(4)[None]
    prior_pose[0, :3, 3] = torch.tensor([-0.05, 0.0, 0.02])
    if prior_state == "outside":
        prior_pose[0, 0, 3] = 1.0
    if prior_state == "above":
        prior_pose[0, 2, 3] = 1.0
    if prior_state == "no_space":
        prior_pose[0, 0, 3] = 0.0
    release = torch.eye(4)[None]
    release[0, :3, 3] = torch.tensor([0.05, 0.0, 0.12])
    poses = {"tray": torch.eye(4)[None], "prior": prior_pose, "inside": release}
    monkeypatch.setattr(
        gate,
        "resolve_pose_goal",
        lambda target, *_a, **_k: poses[target.entity_id].clone(),
    )
    policy.simulation.get_rigid_object = lambda _: NS(get_vertices=lambda **_: vertices)
    if prior_state == "no_space":
        with pytest.raises(ValueError, match="5mm"):
            policy._landing(
                request,
                context,
                context.get_held_object("held"),
                policy.containers["inside"],
            )
        return
    selected, report = policy._landing(
        request, context, context.get_held_object("held"), policy.containers["inside"]
    )
    if prior_state in ("outside", "above"):
        assert selected is request and report["observed_contents"] == []
    else:
        assert report["observed_contents"] == ["prior"]
        assert report["floor_boundary_gap_m"] >= 0.005 - 1e-8
        assert report["content_gaps_m"]["prior"] >= 0.005 - 1e-8
        assert selected.goal.xpos.world_displacement[2] == 0


@dataclass(frozen=True)
class PickPlan:
    plan_success: torch.Tensor
    joint_trajectory: object
    segments: tuple
    expected_effects: object
    commands: object
    tracking: object
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize("rejected", [None, "closure", "software", "free", "native"])
def test_measured_pick_keeps_arm_time_effects_and_rebuilds_commands(
    monkeypatch, rejected
):
    from embodichain.gen_sim.task_engine._task_program import articulation_collision
    from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory

    class Binding:
        def __init__(self, opened=None):
            self.opened = torch.zeros(1, 2) if opened is None else opened

        def endpoint(self, _slot, role):
            if role == "motion":
                return NS(task_state_key="held", target=NS(control_part="arm"))
            return NS(
                target=NS(joint_ids=(1, 2)),
                joint_positions=lambda name, **_: (
                    self.opened.clone() if name == "open" else torch.full((1, 2), 0.7)
                ),
            )

        def with_command_overrides(self, values):
            return Binding(values[("primary", "grasp")]["open"].positions)

    policy, _, _, _, _ = fixture()
    policy.calls = {"pick": ("item", "")}
    semantics = NS(entity_id="item")
    request = Request(
        "pick",
        "pick",
        NS(semantics=semantics, grasp_xpos=None, object_pose=None),
        NS(
            fixed_object_to_eef=None,
            pick_object_part="center",
            hand_interp_steps=2,
            grasp_settle_steps=2,
        ),
        Binding(),
        NS(velocity_targets="zero"),
    )
    context = NS(
        robot=NS(qpos=torch.zeros(1, 3)),
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.04,
    )
    q = torch.zeros(1, 14, 3)
    q[0, :, 0] = torch.linspace(0.1, 0.2, 14)
    q[:, 7:, 1:] = 0.7
    trajectory = TimedTrajectory.from_uniform_step(
        q, env_ids=context.env_ids, step_dt=0.04, velocities=torch.zeros_like(q)
    )
    original = PickPlan(
        torch.tensor([True]),
        trajectory,
        (
            NS(name="approach", start=0, stop=6),
            NS(name="close", start=6, stop=10),
            NS(name="lift", start=10, stop=14),
        ),
        NS(
            held_object_updates={
                "held": NS(semantics=semantics, grasp_xpos=torch.eye(4)[None])
            }
        ),
        object(),
        object(),
        PlannerDiagnostics(backend="test"),
    )
    pad = trimesh.creation.box(extents=(0.004, 0.01, 0.01))
    parts = []
    for direction in (-1.0, 1.0):
        poses = np.tile(np.eye(4), (65, 1, 1))
        poses[:, 0, 3] = direction * (0.065 - 0.06 * np.linspace(0.0, 1.0, 65))
        parts.append(NS(pad=True, meshes=(pad,), tcp_poses=poses, rest_offset=0.001))
    screen = NS(parts=parts, steps=65, target_rest_offset=0.001)
    policy._screen = lambda *_: screen
    policy._poses = lambda *_: np.zeros((14, 2, 4, 4))
    scene = NS(
        obstacles=[
            dict(
                name="native_item",
                meshes=(trimesh.creation.box(extents=(0.08, 0.08, 0.04)),),
                pose=np.eye(4),
            )
        ],
        inspect=lambda *_a, **_k: {"valid": rejected != "native"},
    )
    monkeypatch.setattr(gate, "_Scene", lambda *_: scene)
    monkeypatch.setattr(
        gate, "_joint_velocity_limits", lambda *_: torch.full((1, 3), 100.0)
    )
    monkeypatch.setattr(
        gate, "_inspect_pick_plan", lambda *_: {"valid": rejected != "software"}
    )
    monkeypatch.setattr(
        articulation_collision,
        "check_free_motion",
        lambda *_: {"valid": rejected != "free"},
    )
    action = NS(
        inspect_closing_plan=lambda *_a, **_k: {"valid": rejected != "closure"},
        _joint_command_sequence=lambda req, timed, **_: timed.positions.clone(),
        _tracking_sequence=lambda req, commands: commands.clone(),
    )
    engine = NS(robot=object(), actions={"pick_up": action})
    if rejected is not None:
        with pytest.raises(ValueError, match="qualification"):
            policy._pick(engine, request, context, lambda _: original)
        return
    selected, result = policy._pick(engine, request, context, lambda _: original)
    changed = result.joint_trajectory.positions
    assert torch.equal(changed[:, :, 0], q[:, :, 0])
    assert torch.equal(result.joint_trajectory.dt, trajectory.dt)
    assert torch.equal(result.commands, changed) and torch.equal(
        result.tracking, changed
    )
    assert result.expected_effects is original.expected_effects
    assert result.segments is original.segments
    assert not torch.equal(changed[:, :, 1:], q[:, :, 1:])
    assert selected.binding.opened.max() < 0.7
    assert torch.equal(request.binding.opened, torch.zeros(1, 2))


@pytest.mark.parametrize("damage", [None, "missing_link", "missing_body", "fk", "rest"])
def test_native_arm_inventory_and_observed_frames_fail_closed(
    tmp_path, monkeypatch, damage
):
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot><link name="arm"><collision/></link><link name="hand"/>'
        '<joint name="j"><parent link="arm"/><child link="hand"/></joint></robot>'
    )
    root = np.eye(4)
    root[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    root[:3, 3] = [0.4, -0.3, 0.8]
    native_pose = root.copy()
    if damage == "fk":
        native_pose[0, 3] += 0.01
    body = NS(get_rest_offset=lambda: float("nan") if damage == "rest" else 0.001)
    scene = object.__new__(gate._Scene)
    scene.native = NS(
        get_link_names=lambda: [] if damage == "missing_link" else ["arm", "hand"],
        render_articulation=NS(
            get_physical_body=lambda _: None if damage == "missing_body" else body
        ),
        get_link_pose=lambda _: native_pose,
    )
    scene.obstacles = []
    robot = NS(
        cfg=NS(fpath=str(path)),
        joint_names=("j",),
        get_solver=lambda _: NS(root_link_name="arm"),
        get_local_pose=lambda **_: torch.tensor(root)[None],
        get_qpos=lambda: torch.zeros(1, 1),
        compute_fk=lambda **_: torch.eye(4)[None, None],
    )
    monkeypatch.setattr(
        gate,
        "_native_meshes",
        lambda _: (trimesh.creation.box(extents=(0.02, 0.02, 0.02)),),
    )
    if damage in ("missing_link", "missing_body", "fk"):
        with pytest.raises(ValueError):
            scene.arm_parts(robot, "part", (NS(name="hand"),))
        return
    parts = scene.arm_parts(robot, "part", (NS(name="hand"),))
    if damage == "rest":
        with pytest.raises(ValueError, match="rest"):
            scene.inspect(parts, root[None, None], "source")
    else:
        assert [part.name for part in parts] == ["arm"]
        assert scene.inspect(parts, root[None, None], "source")["valid"]
