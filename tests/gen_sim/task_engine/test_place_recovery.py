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

from dataclasses import dataclass
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program import place_recovery as gate
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    segments: tuple
    diagnostics: PlannerDiagnostics


def _fixture(monkeypatch):
    vertices = torch.tensor(
        trimesh.creation.box(extents=(0.08, 0.08, 0.08)).vertices, dtype=torch.float32
    )[None]
    pose = torch.eye(4)[None]
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="item"), object_to_eef=pose
    )
    target = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    binding = SimpleNamespace(
        endpoint=lambda *_a: SimpleNamespace(
            task_state_key="held", require_target=lambda _t: target
        )
    )
    request = SimpleNamespace(
        binding=binding,
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
        skill_options=SimpleNamespace(
            preserve_current_object_orientation=True, max_approach_retract_z=None
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        last_qpos=torch.zeros(1, 2),
        get_held_object=lambda _key: held,
        scene=SimpleNamespace(entities={"table": SimpleNamespace(pose=pose)}),
    )

    def fk(qpos, **_kwargs):
        result = torch.eye(4)[None, None].repeat(1, qpos.shape[1], 1, 1)
        result[:, :, 2, 3] = 0.3
        return result

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        compute_batch_fk=fk,
        get_qpos_limits=lambda **_kwargs: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
    )
    action = SimpleNamespace(
        robot=robot, recovery_scene_vertices={"item": vertices, "table": vertices}
    )
    plan = Plan(
        torch.tensor([True]),
        SimpleNamespace(positions=torch.zeros(1, 3, 2)),
        (SimpleNamespace(name="approach", stop=2),),
        PlannerDiagnostics(backend="test"),
    )
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_args: {"valid_mask": torch.tensor([True])},
    )

    def collision(_robot, qpos, _part):
        assert qpos.shape == (3, 2)
        return {"valid": True}

    monkeypatch.setattr(gate, "_inspect_pick_motion", collision)
    return action, request, context, plan


def test_waypoint_recovery_retains_request_start_and_rng_without_committing_effects(
    monkeypatch,
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    before = gate._RandomState.capture(set())

    def build(selected, observed):
        assert selected is request and observed is context
        torch.rand(1)
        np.random.rand()
        random.random()
        return plan

    result = gate.recover(action, request, context, first, build)
    assert result.plan_success.tolist() == [True]
    assert result.joint_trajectory is plan.joint_trajectory
    assert context.last_qpos.tolist() == [[0.0, 0.0]]
    assert torch.equal(before.cpu, torch.random.get_rng_state())
    assert before.python == random.getstate()


@pytest.mark.parametrize(
    "damage",
    [
        "successful",
        "batch",
        "cartesian",
        "ceiling",
        "explicit_orientation",
        "dynamic",
        "pour",
    ],
)
def test_qualified_paths_and_other_owned_contracts_remain_unchanged(
    monkeypatch, damage
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    if damage == "successful":
        first = plan
    elif damage == "batch":
        context.batch_size = 2
    elif damage == "ceiling":
        request.skill_options.max_approach_retract_z = 0.4
    elif damage == "explicit_orientation":
        request.skill_options.preserve_current_object_orientation = False
    elif damage == "dynamic":
        request.motion_policy.dynamic_collision_mode = "auto"
    elif damage == "pour":
        context.get_held_object("held").semantics.affordance = SimpleNamespace(
            get_custom_config=lambda _key: "pour"
        )
    token = gate._CARTESIAN_APPROACH.set(damage == "cartesian")
    try:
        build = Mock()
        assert gate.recover(action, request, context, first, build) is first
        build.assert_not_called()
    finally:
        gate._CARTESIAN_APPROACH.reset(token)


@pytest.mark.parametrize(
    "damage",
    ["velocity", "robot", "joint", "bounds", "scene", "missing_mesh", "missing_pose"],
)
def test_unsafe_recovery_is_never_accepted(monkeypatch, damage) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    if damage == "velocity":
        monkeypatch.setattr(
            gate,
            "_validate_final_plan_velocity",
            lambda *_a: {"valid_mask": torch.tensor([False])},
        )
    elif damage == "robot":
        monkeypatch.setattr(gate, "_inspect_pick_motion", lambda *_a: {"valid": False})
    elif damage == "joint":
        plan.joint_trajectory.positions[0, 1, 0] = 2.0
    elif damage == "bounds":
        action.robot.get_qpos_limits = lambda **_kw: torch.full((1, 2, 2), float("nan"))
    elif damage == "scene":
        action.robot.compute_batch_fk = lambda qpos, **_kw: torch.eye(4)[
            None, None
        ].repeat(1, qpos.shape[1], 1, 1)
    elif damage == "missing_mesh":
        action.recovery_scene_vertices.pop("item")
    else:
        context.scene.entities.clear()
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    assert gate.recover(action, request, context, first, lambda *_a: plan) is first


@pytest.mark.parametrize("damage", [None, "missing", "invalid", "articulation"])
def test_geometry_capture_requires_complete_supported_scene_and_owns_its_copy(
    damage,
) -> None:
    vertices = torch.ones(1, 3, 3)
    native = SimpleNamespace(get_vertices=lambda **_kw: vertices)
    simulation = SimpleNamespace(
        get_rigid_object=lambda uid: native if uid == "a" else None
    )
    bindings = SimpleNamespace(
        rigid_objects=[SimpleNamespace(entity_id="item", simulation_uid="a")],
        articulations=() if damage != "articulation" else (object(),),
    )
    if damage == "missing":
        bindings.rigid_objects.append(
            SimpleNamespace(entity_id="obstacle", simulation_uid="b")
        )
    elif damage == "invalid":
        vertices[0, 0, 0] = float("nan")
    meshes = gate.scene_vertices(simulation, bindings)
    if damage is None:
        assert torch.equal(meshes["item"], vertices)
        assert meshes["item"].data_ptr() != vertices.data_ptr()
    else:
        assert meshes == {}


@pytest.mark.parametrize("ik_success", [True, False])
def test_failure_evidence_distinguishes_waypoint_ik_from_velocity_rejection(
    monkeypatch, ik_success
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    monkeypatch.setattr(
        gate, "inspect", lambda *_a: {"valid": False, "reason": "rejected"}
    )

    def build(*_a):
        gate._WAYPOINT_KINEMATICS.get().append(torch.tensor([ik_success]))
        return plan

    result = gate.recover(action, request, context, first, build)
    assert not result.plan_success.any()
    assert result.joint_trajectory is first.joint_trajectory
    assert (
        result.diagnostics.metadata["gen_sim_place_waypoint_probe"]["kinematic_valid"]
        is ik_success
    )
    assert gate._WAYPOINT_KINEMATICS.get() is None


@pytest.mark.parametrize("ik_success", [True, False])
def test_raw_waypoint_evidence_precedes_gen_sim_velocity_mask(
    monkeypatch, ik_success
) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.CheckedMotionGenerator)
    generator.robot = object()
    raw = motion.PlanResult(
        success=torch.tensor([ik_success]),
        positions=torch.tensor([[[0.0], [2.0], [2.0]]]),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )
    monkeypatch.setattr(motion.MotionGenerator, "generate", Mock(return_value=raw))
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *_a: torch.ones(1, 1))
    target = motion.PlanState(
        move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4)[None]
    )
    options = motion.MotionGenOptions(
        strategy="ik_interp", control_part="arm", sample_count=3
    )
    evidence = []
    token = gate._WAYPOINT_KINEMATICS.set(evidence)
    try:
        result = generator._generate_checked([target], options)
    finally:
        gate._WAYPOINT_KINEMATICS.reset(token)
    assert not result.success.any()
    assert evidence[0].tolist() == [ik_success]
    assert raw.success.tolist() == [ik_success]
