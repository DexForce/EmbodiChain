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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOver


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    commands: object
    joint_trajectory: object


def _setup(monkeypatch, outcomes, *, mode="ends"):
    plans, rng = [], []

    def shared(_self, *_args):
        torch.rand(1)
        selection = actions._RECEIVE_SELECTION.get()
        if selection is not None:
            selection.sampled = True
        plan = _Plan(
            torch.tensor([True]), PlannerDiagnostics("fixture"), object(), object()
        )
        plans.append(plan)
        rng.append(torch.random.get_rng_state().clone())
        return plan

    monkeypatch.setattr(HandOver, "_plan_existing_hold", shared)
    action = actions.GenSimHandOver()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    screen = SimpleNamespace(inspect=Mock(side_effect=[{"valid": v} for v in outcomes]))
    factory = Mock(return_value=screen)
    action.environment_screen_factory = factory
    action._merge_receive_plans = Mock(
        side_effect=lambda _r, _c, _a, b, metadata: replace(
            b, diagnostics=replace(b.diagnostics, metadata=metadata)
        )
    )
    context = SimpleNamespace(
        batch_size=1, robot=SimpleNamespace(qpos=torch.zeros(1, 3))
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(
            affordance=SimpleNamespace(get_custom_config=lambda _: mode)
        )
    )
    return action, screen, factory, context, held, plans, rng


def _post_place_selection(action, *, listed=True, mode="ends"):
    action.post_place_exits = (
        {"exchange:0": ("spoon", "right", "left")} if listed else {}
    )
    resources = SimpleNamespace(
        first=SimpleNamespace(task_state_key="right"),
        second=SimpleNamespace(
            task_state_key="left", arm=SimpleNamespace(control_part="left_arm")
        ),
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(
            entity_id="spoon",
            affordance=SimpleNamespace(get_custom_config=lambda _: mode),
        )
    )
    source, receiver = torch.eye(4)[None], torch.eye(4)[None]
    source[:, 1, 3], receiver[:, 1, 3] = 0.4, -0.4
    return actions._ReceiveSelection(
        resources,
        held,
        torch.tensor([False]),
        SimpleNamespace(invocation_id="exchange:0"),
        None,
        root_poses=(source, receiver),
    )


@pytest.mark.parametrize(
    "damage", [None, "unlisted", "object", "source", "receiver", "ordinary"]
)
def test_only_explicit_same_object_post_place_exit_gets_source_side_withdrawal(damage):
    action = actions.GenSimHandOver()
    selection = _post_place_selection(
        action,
        listed=damage != "unlisted",
        mode="ordinary" if damage == "ordinary" else "ends",
    )
    if damage == "object":
        selection.held.semantics.entity_id = "other"
    elif damage in {"source", "receiver"}:
        selected = (
            selection.resources.first
            if damage == "source"
            else selection.resources.second
        )
        selected.task_state_key = "wrong"
    pose = torch.eye(4)[None]
    kwargs = dict(
        source_fallback=selection.root_poses[0],
        destination_fallback=selection.root_poses[1],
        retreat_distance=0.1,
        lift_height=0.0,
    )
    old = HandOver._source_retreat_waypoints(pose, pose, **kwargs)
    token = actions._RECEIVE_SELECTION.set(selection)
    try:
        new = action._source_retreat_waypoints(pose, pose, **kwargs)
    finally:
        actions._RECEIVE_SELECTION.reset(token)
    assert torch.equal(new[:, :3], old[:, :3])
    assert torch.equal(new[..., :3, :3], old[..., :3, :3])
    if damage is None:
        assert (new[0, -2:, 1, 3] - old[0, -2:, 1, 3]).tolist() == pytest.approx(
            [0.05, 0.1]
        )
    else:
        assert torch.equal(new, old)


@pytest.mark.parametrize(
    "exit_valid,screen_valid", [(True, True), (False, True), (True, False)]
)
def test_post_place_exit_mesh_gate_is_additive_not_a_release_certificate(
    monkeypatch, exit_valid, screen_valid
):
    action = actions.GenSimHandOver()
    action._planning_services = SimpleNamespace(robot=object())
    selection = _post_place_selection(action)
    plan = _Plan(
        torch.tensor([True]),
        PlannerDiagnostics("fixture"),
        object(),
        SimpleNamespace(positions=torch.zeros(1, 3, 4)),
    )
    inspect = Mock(return_value={"valid": exit_valid, "physical_evidence": False})
    monkeypatch.setattr(actions, "_inspect_pick_motion", inspect)
    token = actions._RECEIVE_SELECTION.set(selection)
    try:
        result = action._qualify_receiver_environment(
            plan, SimpleNamespace(inspect=lambda _: {"valid": screen_valid})
        )
    finally:
        actions._RECEIVE_SELECTION.reset(token)
    assert bool(result.plan_success.all()) == (exit_valid and screen_valid)
    inspect.assert_called_once()
    assert result.commands is plan.commands
    assert result.joint_trajectory is plan.joint_trajectory


def test_safe_receiver_keeps_original_commands_and_has_no_direction_retry(monkeypatch):
    action, screen, factory, context, held, plans, rng = _setup(monkeypatch, [True])
    result = action._plan_existing_hold(object(), context, object(), held)
    assert len(plans) == 1
    assert result.commands is plans[0].commands
    assert result.joint_trajectory is plans[0].joint_trajectory
    assert result.plan_success.tolist() == [True]
    assert result.diagnostics.metadata["gen_sim_handover_environment"]["valid"] is True
    assert factory.call_count == screen.inspect.call_count == 1
    action._merge_receive_plans.assert_not_called()
    torch.testing.assert_close(torch.random.get_rng_state(), rng[0])


def test_environment_collision_uses_only_existing_alternate_direction(monkeypatch):
    action, screen, factory, context, held, plans, rng = _setup(
        monkeypatch, [False, True, True]
    )
    result = action._plan_existing_hold(object(), context, object(), held)
    assert len(plans) == 2
    assert factory.call_count == 1
    assert screen.inspect.call_count == 3
    assert result.plan_success.tolist() == [True]
    assert result.commands is plans[1].commands
    assert result.diagnostics.metadata["selected_receive_attempt"] == [1]
    torch.testing.assert_close(torch.random.get_rng_state(), rng[0])
    assert actions._RECEIVE_SELECTION.get() is None


def test_two_unsafe_directions_never_create_a_third_plan(monkeypatch):
    action, screen, _, context, held, plans, _ = _setup(monkeypatch, [False, False])
    result = action._plan_existing_hold(object(), context, object(), held)
    assert len(plans) == screen.inspect.call_count == 2
    assert result.plan_success.tolist() == [False]
    assert (
        result.diagnostics.failure.code == "handover_receiver_environment_unqualified"
    )
    action._merge_receive_plans.assert_not_called()


def test_unconfigured_handover_preserves_shared_plan_identity(monkeypatch):
    action, screen, _, context, held, plans, _ = _setup(monkeypatch, [])
    action.environment_screen_factory = None
    result = action._plan_existing_hold(object(), context, object(), held)
    assert result is plans[0]
    screen.inspect.assert_not_called()


def test_unmarked_semantics_do_not_install_a_new_safety_policy(monkeypatch):
    action, screen, factory, context, held, plans, _ = _setup(
        monkeypatch, [], mode=None
    )
    result = action._plan_existing_hold(object(), context, object(), held)
    assert result is plans[0]
    factory.assert_not_called()
    screen.inspect.assert_not_called()


@pytest.mark.parametrize(
    "error", [ValueError("missing native shape"), RuntimeError("stale snapshot")]
)
def test_unavailable_environment_geometry_fails_before_motion_planning(
    monkeypatch, error
):
    action, _, factory, context, held, plans, _ = _setup(monkeypatch, [])
    factory.side_effect = error
    action.failed_plan = Mock(return_value="failed_without_motion")
    assert (
        action._plan_existing_hold(object(), context, object(), held)
        == "failed_without_motion"
    )
    assert not plans
    action._merge_receive_plans.assert_not_called()


@pytest.mark.parametrize("handover", [False, True])
@pytest.mark.parametrize("revision", ["scene", "exchange", "end_region"])
def test_scene_safety_revision_changes_only_handover_deployment_identity(
    monkeypatch, tmp_path, handover, revision
):
    from embodichain.gen_sim.task_engine._task_program import (
        adaptive_grasp,
        handover_safety,
        handover_exchange,
    )
    from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config
    from .test_semantic_graph import (
        _fresh_handover_graph,
        _inside_orientation_case,
        _prepared_axis_scene,
    )

    if handover:
        scene = _prepared_axis_scene(tmp_path)
        graph = _fresh_handover_graph()
        graph["nodes"][0]["call"] = {
            "kind": "pick",
            "object": "bottle",
            "resources": {"primary": "left"},
        }
        graph["nodes"][1]["call"]["object"] = "bottle"
    else:
        scene, _, graph = _inside_orientation_case(tmp_path)
    _, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "bundle", robot_profile="dual_franka"
    )

    def load():
        return load_deployment(
            task_program=load_config(paths.deployment)["task_program"],
            skill_profile=load_config(paths.embodiment)["skill_profile"],
            base_dir=paths.root,
        ).integration

    before = load()
    assert before.adapter_factory.handover_scene_clearance is handover
    owner, attribute = {
        "scene": (handover_safety, "HANDOVER_SCENE_CLEARANCE_REVISION"),
        "exchange": (handover_exchange, "HANDOVER_EXCHANGE_REVISION"),
        "end_region": (adaptive_grasp, "HANDOVER_END_REGION_REVISION"),
    }[revision]
    monkeypatch.setattr(owner, attribute, getattr(owner, attribute) + 1)
    after = load()
    assert (before.integration_fingerprint != after.integration_fingerprint) is handover
