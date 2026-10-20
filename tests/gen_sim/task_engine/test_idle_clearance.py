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

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import idle_clearance as idle
from embodichain.lab.sim.atomic_actions import JointPositionTarget, MoveJointsOptions
from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory
from .test_support_bundle import support_scene, _placement_graph, _load_bundle


def route():
    return {
        "route_id": "outward",
        "clear_segment": "idle_clear",
        "restore_segment": "idle_restore",
        "resource": "right",
        "joint_name": "joint_b",
        "offset_rad": 0.6,
    }


def context():
    return NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.tensor([[0.1, 0.2, 0.3]])),
        task=NS(get_held_object=lambda _: None, coordinated_held_objects={}),
    )


@pytest.mark.parametrize(
    "damage", ["unknown", "bool", "zero", "nonfinite", "duplicate"]
)
def test_idle_route_decoder_is_closed(damage):
    value = route()
    if damage == "unknown":
        value["class_type"] = "untrusted.callable"
    if damage == "bool":
        value["offset_rad"] = True
    if damage == "zero":
        value["offset_rad"] = 0
    if damage == "nonfinite":
        value["offset_rad"] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        idle.decode_routes(
            [value, value] if damage == "duplicate" else [value], path="test"
        )


def test_idle_lowering_owns_one_declared_joint_and_never_mutates_context():
    cfg = idle.decode_routes([route()], path="test")
    robot = NS(joint_names=["joint_a", "joint_b", "joint_c"])
    arm = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    bound = NS(
        binding=NS(
            resources={
                "primary": NS(
                    endpoints={"motion": NS(task_state_key="right", runtime_target=arm)}
                )
            }
        )
    )
    call = NS(arguments={"route": "outward"}, resources={"primary": "right"})
    ctx = context()
    old = ctx.robot.qpos.clone()
    lowerer = idle.IdleFactory(cfg).create(
        simulation=None, robot=robot, scene_registry=None, engine=NS(robot=robot)
    )
    result = lowerer.lower(
        call, context=ctx, bound=bound, option_template=MoveJointsOptions()
    )
    assert torch.allclose(result.goal.target, torch.tensor([[0.1, 0.8]]))
    assert torch.equal(ctx.robot.qpos, old)
    ctx.task.coordinated_held_objects = {
        ("right", "left"): NS(active_mask=torch.tensor([True]))
    }
    with pytest.raises(ValueError, match="coordinated"):
        lowerer.lower(
            call, context=ctx, bound=bound, option_template=MoveJointsOptions()
        )


@pytest.mark.parametrize(
    "damage", [None, "passive", "start", "limits", "velocity", "held"]
)
def test_idle_emitted_motion_keeps_original_gates(monkeypatch, damage):
    ctx = context()
    arm = JointPositionTarget(control_part="arm", joint_ids=(1,))
    request = NS(
        binding=NS(
            endpoint=lambda *_: NS(task_state_key="right", require_target=lambda _: arm)
        )
    )
    values = ctx.robot.qpos[:, None].repeat(1, 4, 1)
    values[0, :, 1] += torch.arange(4) * 0.01
    if damage == "passive":
        values[0, -1, 2] += 0.01
    if damage == "start":
        values[0, 0, 1] += 0.01
    if damage == "limits":
        values[0, -1, 1] = 3
    if damage == "velocity":
        values[0, -1, 1] += 0.5
    if damage == "held":
        ctx.task.get_held_object = lambda _: NS(active_mask=torch.tensor([True]))
    robot = NS(get_qpos_limits=lambda **_: torch.tensor([[[-1.0, 1.0]]]))
    monkeypatch.setattr(idle, "_joint_velocity_limits", lambda *_: torch.ones(1, 3))
    monkeypatch.setattr(idle, "check_free_motion", lambda *_: {"valid": True})
    plan = NS(
        joint_trajectory=TimedTrajectory.from_uniform_step(
            values, step_dt=0.04, env_ids=ctx.env_ids
        )
    )
    if damage:
        with pytest.raises(ValueError):
            idle.qualify_motion(robot, None, request, ctx, plan)
    else:
        assert idle.qualify_motion(robot, None, request, ctx, plan)["valid"]


def test_idle_service_round_trip_binds_only_explicit_clear_and_restore(
    support_scene, tmp_path
):
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )

    graph = _placement_graph(reference="tray")
    clear, restore = [deepcopy(graph["nodes"][0]) for _ in range(2)]
    clear.update(
        id="idle_clear",
        depends_on=[],
        role="idle_clearance",
        call={
            "kind": "registered",
            "call_id": idle.CALL_ID,
            "arguments": {"route": "outward"},
            "resources": {"primary": "right"},
        },
    )
    restore.update(
        id="idle_restore",
        depends_on=["place"],
        role="idle_restore",
        call={
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": "right"},
        },
    )
    graph["nodes"][0]["depends_on"] = ["idle_clear"]
    graph["nodes"] = [clear, *graph["nodes"], restore]
    graph["task_groups"][0]["node_ids"] = [n["id"] for n in graph["nodes"]]
    _, paths = generate_task_program_bundle(
        graph,
        support_scene,
        tmp_path / "idle",
        robot_profile="dual_franka",
        idle_clearance_routes=[route()],
    )
    loaded = _load_bundle(paths).integration.adapter_factory
    assert len(loaded.idle_invocations) == 2
    assert all(resource == "right" for _, resource in loaded.idle_invocations)
    with pytest.raises(ValueError, match="explicit registered route"):
        generate_task_program_bundle(
            graph, support_scene, tmp_path / "missing", robot_profile="dual_franka"
        )


def test_idle_search_uses_fk_ranking_and_skips_colliding_proposals(monkeypatch):
    import numpy as np
    import yourdfpy

    state = {"shoulder": 0.0}

    def transform(name):
        pose = np.eye(4)
        if name == "idle_tcp":
            pose[0, 3] = 1 + state["shoulder"]
        return pose

    model = NS(
        joint_map={"shoulder": NS(limit=NS(lower=-1.0, upper=1.0))},
        update_cfg=lambda value: state.update(value),
        get_transform=transform,
    )
    monkeypatch.setattr(yourdfpy.URDF, "load", lambda *args, **kwargs: model)
    robot = NS(
        cfg=NS(fpath="fixture"),
        joint_names=["shoulder", "passive"],
        get_solver=lambda part: NS(
            end_link_name="idle_tcp" if part == "idle_arm" else "protected_tcp"
        ),
    )
    ctx = NS(robot=NS(qpos=torch.tensor([[0.0, 0.2]])))
    proposals = []

    def check(robot, simulation, sweep, part):
        proposals.append(float(sweep[-1, 0]))
        assert torch.equal(sweep[:, 1], torch.full((101,), 0.2))
        return {"valid": len(proposals) > 1}

    monkeypatch.setattr(idle, "check_free_motion", check)
    cfg = idle.IdleRoute(
        "automatic", "clear", "restore", "idle", "shoulder", 0.6, True, "protected_arm"
    )
    selected = idle.select_offset(robot, None, ctx, NS(control_part="idle_arm"), cfg)
    assert proposals == pytest.approx([0.6, 0.4])
    assert selected == pytest.approx(0.4)
    assert torch.equal(ctx.robot.qpos, torch.tensor([[0.0, 0.2]]))


def test_engine_rechecks_idle_after_standard_plan_transform(monkeypatch):
    from dataclasses import dataclass
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import PlannerDiagnostics

    @dataclass(frozen=True)
    class Plan:
        diagnostics: PlannerDiagnostics

    final = Plan(PlannerDiagnostics(backend="transformed"))
    action = NS(
        idle_invocations={"clear": "empty"}, inspect_final_plan=Mock(return_value=final)
    )
    engine = object.__new__(GenSimActionEngine)
    engine._actions = {"move_joints": action}
    ctx = NS()
    request = NS(skill_id="move_joints", invocation_id="clear")
    monkeypatch.setattr(engine, "initial_context", lambda: ctx)
    monkeypatch.setattr(engine, "_plan_request_impl", lambda *args, **kwargs: final)
    result = engine._plan_request(request, plan_transform=lambda *args: final)
    action.inspect_final_plan.assert_called_once_with(request, ctx, final)
    assert result.diagnostics.backend == "transformed"
