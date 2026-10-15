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

from dataclasses import replace
import importlib
from types import SimpleNamespace

import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program import scene_collision
from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.bindings import (
    ActionBinding,
    EndpointBinding,
    JointPositionTarget,
)
from embodichain.lab.sim.atomic_actions.control import JointPositionCommand
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
)
from embodichain.lab.sim.atomic_actions.policies import (
    MotionPolicy,
    RecoveryPolicy,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
    GraspGoal,
    PickUp,
    PickUpOptions,
)
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions
from embodichain.lab.sim.atomic_actions.scene import EntityState, SceneSnapshot
from embodichain.lab.sim.atomic_actions.state import (
    HeldObjectState,
    PlanningContext,
    RobotObservation,
    TaskState,
)


def _pose(z):
    value = torch.eye(4)[None]
    value[:, 2, 3] = z
    return value


def _mesh(owner, name, extents):
    native = trimesh.creation.box(extents=extents)
    return scene_collision.CollisionMesh(
        owner,
        owner,
        name,
        torch.tensor(native.vertices, dtype=torch.float64),
        torch.tensor(native.faces, dtype=torch.long),
    )


def _fixture(monkeypatch, kind):
    gate = importlib.import_module(
        "embodichain.gen_sim.task_engine._task_program.rigid_qualification"
    )
    arm, hand = JointPositionTarget("arm", (0, 1, 2)), JointPositionTarget("hand", (3,))
    endpoints = (
        EndpointBinding("primary", "motion", "resource", "joint", arm, task_state_key="held"),
        EndpointBinding(
            "primary", "grasp", "resource", "joint", hand, task_state_key="held",
            commands={"open": JointPositionCommand(torch.zeros(1)), "grasp": JointPositionCommand(torch.ones(1))},
        ),
    )
    semantics = ObjectSemantics(Affordance(), {}, "object")
    held = HeldObjectState(semantics, torch.eye(4)[None], _pose(0.04))
    if kind == "pick_up":
        goal, options = GraspGoal(semantics), PickUpOptions()
        heights = [0.3, 0.04, 0.04, 0.04, 0.14, 0.2]
        segments = (TrajectorySegment("approach", 0, 2), TrajectorySegment("close", 2, 4), TrajectorySegment("lift", 4, 6))
        task = TaskState(1, torch.device("cpu"))
        effects = StateDelta(held_object_updates={"held": held})
        observed_object = _pose(0.04)
    elif kind == "place":
        goal, options = PlaceGoal(_pose(0.04)), PlaceOptions()
        heights = [0.2, 0.06, 0.04, 0.04, 0.04, 0.14]
        segments = (TrajectorySegment("approach", 0, 3), TrajectorySegment("release", 3, 5), TrajectorySegment("retract", 5, 6))
        task = TaskState(1, torch.device("cpu"), held_objects={"held": held})
        effects = StateDelta(held_object_updates={"held": None})
        observed_object = _pose(0.2)
    else:
        goal, options = HeldObjectPoseGoal(_pose(0.2)), MoveHeldObjectOptions()
        heights = [0.2] * 6
        segments = (TrajectorySegment("transport", 0, 6),)
        task = TaskState(1, torch.device("cpu"), held_objects={"held": held})
        effects = StateDelta()
        observed_object = _pose(0.2)
    request = ResolvedActionRequest(
        kind, goal, ActionBinding("owner", endpoints), MotionPolicy(sample_count=6),
        TrackingPolicy(), RecoveryPolicy(), options,
    )
    positions = torch.zeros(1, 6, 4)
    positions[0, :, 2] = torch.tensor(heights)
    if kind != "pick_up":
        positions[0, :, 3] = 1
    if kind == "place":
        positions[0, 3:, 3] = torch.tensor([0.5, 0.0, 0.0])
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=torch.tensor([0]), step_dt=0.04
    )
    commands = PickUp()._joint_command_sequence(request, trajectory, active_mask=torch.tensor([True]))
    plan = ActionPlan(
        kind, torch.tensor([True]), commands, request.recovery_policy,
        request.tracking_policy, 0, (0,), PlannerDiagnostics("portable"),
        joint_trajectory=trajectory, segments=segments, expected_effects=effects,
    )
    context = PlanningContext(
        RobotObservation(timestamp=0.0, qpos=positions[:, 0], root_pose=torch.eye(4)[None]),
        task,
        SceneSnapshot(timestamp=0.0, version=0, entities={
            "object": EntityState(observed_object), "table": EntityState(_pose(-0.05))
        }),
        env_ids=torch.tensor([0]), control_dt=0.04,
    )
    meshes = scene_collision.SceneCollisionMeshes((
        _mesh("object", "object_body", (0.08, 0.08, 0.08)),
        _mesh("table", "table_floor", (2.0, 2.0, 0.1)),
    ))
    records = {}

    def fk(_robot, _context, selected, _part):
        result = torch.eye(4)[None, None].repeat(1, selected.shape[1], 1, 1)
        result[:, :, :3, 3] = selected[:, :, :3]
        return result

    def robot_scene(*_args, **kwargs):
        records["robot_scene"] = kwargs
        return {"valid": True, "scope": "test_robot_scene"}

    def payload_robot(*args, **kwargs):
        records["object_poses"] = args[6].clone()
        records["payload_robot"] = kwargs
        return {"valid": True, "scope": "test_payload_robot"}

    def payload_scene(*args, **kwargs):
        records["payload_scene"] = kwargs
        return {"valid": True, "scope": "test_payload_scene"}

    monkeypatch.setattr(scene_collision, "snapshot_eef_poses", fk, raising=False)
    monkeypatch.setattr(scene_collision, "snapshot_pad_links", lambda *_a: ("active_left_finger_pad", "active_right_finger_pad"), raising=False)
    monkeypatch.setattr(scene_collision, "inspect_snapshot_motion", robot_scene)
    monkeypatch.setattr(scene_collision, "inspect_payload_robot", payload_robot)
    monkeypatch.setattr(scene_collision, "inspect_payload_scene", payload_scene, raising=False)
    robot = SimpleNamespace(get_solver=lambda _part: SimpleNamespace(get_iteration_params=lambda: {"pos_eps": 0.0001, "rot_eps": 0.0001}))
    return gate, robot, request, context, plan, meshes, records


@pytest.mark.parametrize("kind", ["pick_up", "place", "move_held_object"])
def test_emitted_phases_have_exact_conditional_payload_schedule(monkeypatch, kind):
    gate, robot, request, context, plan, meshes, records = _fixture(monkeypatch, kind)
    before = context.task
    result = gate.inspect(robot, request, context, plan, meshes)
    assert result["owned"] and result["valid"]
    assert result["native_effect_verification_required"]
    assert context.task is before
    poses = records["object_poses"]
    if kind == "pick_up":
        assert poses[0, :4, 2, 3].tolist() == pytest.approx([0.04] * 4)
        assert poses[0, 4:, 2, 3].tolist() == pytest.approx([0.14, 0.2])
        assert result["conditional_attachment"]
    elif kind == "place":
        assert poses[0, 2:, 2, 3].tolist() == pytest.approx([0.04] * 4)
        contacts = records["payload_robot"]["pad_contacts"]
        assert all(contact.stop_frame <= 5 for contact in contacts)
    else:
        assert poses[0, :, 2, 3].tolist() == pytest.approx([0.2] * 6)
    assert torch.equal(records["robot_scene"]["object_pose_samples"]["object"], poses)


@pytest.mark.parametrize("query", ["inspect_snapshot_motion", "inspect_payload_robot", "inspect_payload_scene"])
def test_one_failed_oracle_blocks_whole_qualification(monkeypatch, query):
    gate, robot, request, context, plan, meshes, _ = _fixture(monkeypatch, "move_held_object")
    monkeypatch.setattr(scene_collision, query, lambda *_a, **_kw: {"valid": False, "reason": "physical_collision"})
    result = gate.inspect(robot, request, context, plan, meshes)
    assert not result["valid"]


@pytest.mark.parametrize("damage", ["geometry", "held", "segments", "release_motion", "predicted_pick_only"])
def test_unknown_physical_schedule_never_produces_valid_certificate(monkeypatch, damage):
    kind = "pick_up" if damage == "predicted_pick_only" else "place"
    gate, robot, request, context, plan, meshes, _ = _fixture(monkeypatch, kind)
    if damage == "geometry":
        meshes = scene_collision.SceneCollisionMeshes(unavailable="missing mesh")
    elif damage == "held":
        context = replace(context, task=TaskState(1, torch.device("cpu")))
    elif damage == "segments":
        plan = replace(plan, segments=(TrajectorySegment("unknown", 0, 6),))
    elif damage == "release_motion":
        changed = plan.joint_trajectory.positions.clone()
        changed[0, 4, 2] = 0.05
        trajectory = TimedTrajectory.from_uniform_step(changed, env_ids=context.env_ids, step_dt=0.04)
        plan = replace(plan, joint_trajectory=trajectory, commands=PickUp()._joint_command_sequence(request, trajectory, active_mask=plan.plan_success))
    else:
        plan = replace(plan, expected_effects=StateDelta())
    result = gate.inspect(robot, request, context, plan, meshes)
    assert not result["valid"]


def test_floor_contact_is_exact_component_not_whole_container(monkeypatch):
    gate, robot, request, context, plan, meshes, records = _fixture(monkeypatch, "place")
    floor = replace(meshes.components[1], owner_id="cabinet", pose_entity_id="floor_link", name="floor_only")
    wall = _mesh("cabinet", "wall", (0.1, 0.2, 0.4))
    wall = replace(wall, pose_entity_id="wall_link")
    entities = dict(context.scene.entities)
    entities["floor_link"] = entities.pop("table")
    entities["wall_link"] = EntityState(_pose(0.2))
    context = replace(context, scene=replace(context.scene, entities=entities))
    meshes = scene_collision.SceneCollisionMeshes((meshes.components[0], floor, wall))
    result = gate.inspect(robot, request, context, plan, meshes)
    assert result["valid"]
    assert set(records["payload_scene"]["allowed_contacts"]) == {"floor_only"}
    assert records["payload_scene"]["allowed_contacts"]["floor_only"][0, 3:].all()


def test_connected_container_wall_never_becomes_floor_exception(monkeypatch):
    gate, robot, request, context, plan, meshes, records = _fixture(monkeypatch, "place")
    body = trimesh.creation.box(extents=(2.0, 2.0, 0.1))
    wall = trimesh.creation.box(extents=(0.1, 2.0, 0.4))
    wall.apply_translation((0.5, 0.0, 0.25))
    complete = trimesh.util.concatenate((body, wall))
    component = replace(meshes.components[1], vertices=torch.tensor(complete.vertices), faces=torch.tensor(complete.faces))
    meshes = scene_collision.SceneCollisionMeshes((meshes.components[0], component))
    result = gate.inspect(robot, request, context, plan, meshes)
    assert not records["payload_scene"]["allowed_contacts"]
    assert result["floor_contacts"]["not_certified"] == "contact_force"


def test_lift_late_descent_does_not_reuse_observed_source_floor_permission(monkeypatch):
    gate, robot, request, context, plan, meshes, records = _fixture(monkeypatch, "pick_up")
    changed = plan.joint_trajectory.positions.clone()
    changed[0, 5, 2] = 0.04
    trajectory = TimedTrajectory.from_uniform_step(changed, env_ids=context.env_ids, step_dt=0.04)
    plan = replace(plan, joint_trajectory=trajectory, commands=PickUp()._joint_command_sequence(request, trajectory, active_mask=plan.plan_success))
    gate.inspect(robot, request, context, plan, meshes)
    assert not records["payload_scene"]["allowed_contacts"]["table_floor"][0, 5]
