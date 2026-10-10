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

from embodichain.gen_sim.task_engine._task_program import pick_continuation as gate
from embodichain.lab.sim.atomic_actions import (
    AffordancePoseCandidates,
    JointPositionTarget,
    PlannerDiagnostics,
)
from embodichain.lab.sim.atomic_actions.state import EntityState
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneAffordanceRef,
    EffectAssurance,
)


@pytest.mark.parametrize(
    "damage", [None, "other", "middle", "parallel", "projected", "monitor"]
)
def test_only_owned_adjacent_verified_calls_can_preview(damage) -> None:
    calls = [
        Pick(SceneObjectRef("item"), resources={"primary": "arm"}),
        Place(
            SceneObjectRef("other" if damage == "other" else "item"),
            inside=SceneAffordanceRef("bin"),
            resources={"primary": "arm"},
        ),
    ]
    if damage == "middle":
        calls.insert(
            1,
            RegisteredSemanticCall(
                call_id="simulation.park", resources={"primary": "arm"}
            ),
        )
    segments = [
        SimpleNamespace(
            segment_id=str(i),
            parallel_block=object() if damage == "parallel" else None,
            calls=(SimpleNamespace(call=c, segment_call_index=0),),
        )
        for i, c in enumerate(calls)
    ]
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    preset = SimpleNamespace(
        effect_assurance=(
            EffectAssurance.PROJECTED
            if damage == "projected"
            else EffectAssurance.VERIFIED
        ),
        effect_monitors={} if damage == "monitor" else {"pick": None, "place": None},
    )
    assert bool(gate.bind_calls(program, preset)) is (damage is None)


@dataclass(frozen=True)
class Goal:
    grasp_xpos: object = None


@dataclass(frozen=True)
class Options:
    approach_direction: object


@dataclass(frozen=True)
class Request:
    goal: Goal
    skill_options: object
    motion_policy: object


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    expected_effects: object
    diagnostics: PlannerDiagnostics


def _fixture(monkeypatch):
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    poses[0, 1, 2, 3] = 0.3
    poses[0, 2, 2, 3] = 0.1
    plan = Plan(
        torch.tensor([True]),
        SimpleNamespace(
            held_object_updates={"arm": SimpleNamespace(grasp_xpos=poses[:, 0])}
        ),
        PlannerDiagnostics(backend="test"),
    )
    robot = SimpleNamespace(device=torch.device("cpu"))
    action = SimpleNamespace(
        robot=robot,
        failed_plan=Mock(
            return_value=replace(plan, plan_success=torch.tensor([False]))
        ),
    )
    request = Request(
        Goal(),
        Options(approach_direction=torch.tensor([0.0, 0.0, -1.0])),
        object(),
    )
    context = SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 1)))
    selection = SimpleNamespace(
        direction=torch.tensor([[0.0, 0.0, -1.0]]),
        candidates=AffordancePoseCandidates(
            poses, torch.zeros(1, 3), torch.tensor([[True, False, True]])
        ),
    )
    built = []

    def build(selected, current):
        assert current is context
        built.append(selected)
        torch.rand(1)
        return plan

    preview = SimpleNamespace(
        route=SimpleNamespace(place_id="p/place:0"),
        compiler=SimpleNamespace(
            integration=SimpleNamespace(engine=SimpleNamespace(plan_request=build))
        ),
    )
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_a: {"valid_mask": torch.tensor([True])},
    )
    monkeypatch.setattr(gate, "_inspect_pick_plan", lambda *_a: {"valid": True})
    return action, request, context, plan, selection, preview, built


def test_qualified_original_retains_plan_rng_and_never_rebuilds(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()

    def preview(*_a, **_kw):
        torch.rand(1)
        return {"valid": True}

    monkeypatch.setattr(gate, "preview_place", preview)
    assert gate.recover(*data[:6]) is data[3]
    assert torch.equal(before, torch.random.get_rng_state())
    assert not data[6]


@pytest.mark.parametrize(
    "reason", [{"valid": False}, {"valid": False, "unreachable_waypoints": False}]
)
def test_velocity_or_uncertain_place_prediction_cannot_replace_original_grasp(
    monkeypatch, reason
) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(gate, "preview_place", lambda *_a, **_kw: reason)
    assert gate.recover(*data[:6]) is data[3]
    assert not data[6]


def test_failed_continuation_rebuilds_only_legal_grasp_not_task_goal_or_limits(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()
    monkeypatch.setattr(
        gate,
        "preview_place",
        Mock(
            side_effect=[
                {"valid": False, "unreachable_waypoints": True},
                {"valid": True},
            ]
        ),
    )
    result = gate.recover(*data[:6])
    assert len(data[6]) == 1
    assert torch.equal(data[6][0].goal.grasp_xpos, data[4].candidates.poses[:, 2])
    assert data[6][0].motion_policy is data[1].motion_policy
    assert result.diagnostics.metadata["gen_sim_pick_continuation"]["candidate_id"] == 2
    assert torch.equal(before, torch.random.get_rng_state())


def test_exhausted_or_unsafe_pool_fails_closed_without_acquired_state(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(
        gate,
        "preview_place",
        lambda *_a, **_kw: {"valid": False, "unreachable_waypoints": True},
    )
    monkeypatch.setattr(gate, "_inspect_pick_plan", lambda *_a: {"valid": False})
    result = gate.recover(*data[:6])
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "pick_continuation_unqualified"


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "batch",
        "env",
        "robot",
        "identity",
        "cartesian",
        "explicit",
        "fixed",
        "upright",
        "axis",
        "part",
        "constrained",
        "pool",
        "direction",
        "strategy",
        "dynamic",
        "pour",
        "preview",
    ],
)
def test_pick_recovery_does_not_capture_other_action_contracts(damage) -> None:
    robot = object()
    action = SimpleNamespace(robot=robot)
    engine = SimpleNamespace(robot=robot, _cartesian_calls=())
    preview = SimpleNamespace(
        route=SimpleNamespace(pick_id="p/pick:0"),
        compiler=SimpleNamespace(integration=SimpleNamespace(engine=engine)),
    )
    request = SimpleNamespace(
        invocation_id="p/pick:0",
        goal=SimpleNamespace(grasp_xpos=None),
        skill_options=SimpleNamespace(
            fixed_object_to_eef=None,
            rotate_upright=None,
            obj_upright_direction=None,
            pick_object_part="center",
        ),
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
    )
    context = SimpleNamespace(batch_size=1, env_ids=torch.tensor([0]))
    selection = SimpleNamespace(
        constrained=False, candidates=object(), direction=object()
    )
    if damage == "batch":
        context.batch_size = 2
    elif damage == "env":
        context.env_ids = torch.tensor([1])
    elif damage == "robot":
        action.robot = object()
    elif damage == "identity":
        request.invocation_id = "other"
    elif damage == "pour":
        action.pick_purposes = {request.invocation_id: "pour"}
    elif damage == "cartesian":
        engine._cartesian_calls = (request.invocation_id,)
    elif damage == "explicit":
        request.goal.grasp_xpos = object()
    elif damage in {"fixed", "upright", "axis", "part"}:
        field = {
            "fixed": "fixed_object_to_eef",
            "upright": "rotate_upright",
            "axis": "obj_upright_direction",
            "part": "pick_object_part",
        }[damage]
        setattr(request.skill_options, field, "other")
    elif damage == "constrained":
        selection.constrained = True
    elif damage in {"pool", "direction"}:
        setattr(selection, "candidates" if damage == "pool" else "direction", None)
    elif damage in {"strategy", "dynamic"}:
        setattr(
            request.motion_policy,
            "strategy" if damage == "strategy" else "dynamic_collision_mode",
            "other",
        )
    token = gate._PREVIEW.set(damage == "preview")
    try:
        assert gate.eligible(action, request, context, selection, preview) is (
            damage is None
        )
    finally:
        gate._PREVIEW.reset(token)


@dataclass(frozen=True)
class Scene:
    entities: dict


@dataclass(frozen=True)
class Context:
    robot: object
    task: object
    scene: Scene

    @property
    def last_qpos(self):
        return self.robot.qpos

    def project(self, *, qpos, task):
        return replace(self, robot=SimpleNamespace(qpos=qpos), task=task)


def test_preview_uses_emitted_pick_fk_and_private_state_not_intended_lift(
    monkeypatch,
) -> None:
    arm, hand = JointPositionTarget("arm", (0, 1)), JointPositionTarget("hand", (2,))
    endpoints = [
        SimpleNamespace(
            task_state_key="arm",
            target=t,
            require_target=lambda _cls, t=t: t,
            joint_positions=lambda *_a, **_kw: torch.zeros(1, 1),
        )
        for t in (arm, hand)
    ]
    binding = SimpleNamespace(endpoint=lambda _slot, name: endpoints[name == "grasp"])
    pose, grasp = torch.eye(4)[None], torch.eye(4)[None]
    grasp[:, 2, 3] = 2.0
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="item"),
        object_to_eef=pose.clone(),
        grasp_xpos=grasp,
    )
    initial_task, projected_task = object(), object()
    context = Context(
        SimpleNamespace(qpos=torch.zeros(1, 3)),
        initial_task,
        Scene({"item": EntityState(pose.clone())}),
    )
    picked = SimpleNamespace(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 2, 3)),
        expected_effects=SimpleNamespace(
            held_object_updates={"arm": held}, apply=lambda task, mask: projected_task
        ),
    )
    request = SimpleNamespace(
        binding=binding,
        goal=SimpleNamespace(semantics=held.semantics),
        skill_options=SimpleNamespace(lift_height=10.0),
    )
    place_request = SimpleNamespace(binding=binding)
    placed = SimpleNamespace(
        plan_success=torch.tensor([True]),
        invocation_id="p/place:0",
        skill_id="place",
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 2, 3)),
        expected_effects=SimpleNamespace(
            held_object_updates={"arm": None},
            coordinated_held_object_updates={},
            articulation_joint_updates={},
        ),
    )
    fk = pose.clone()
    fk[:, 2, 3] = 0.42
    robot = SimpleNamespace(compute_fk=Mock(return_value=fk))
    action = SimpleNamespace(robot=robot)
    engine = SimpleNamespace(
        resolve=Mock(return_value=place_request), plan=Mock(return_value=placed)
    )
    compiler = SimpleNamespace(
        ground=Mock(return_value=SimpleNamespace(invocation=object())),
        integration=SimpleNamespace(engine=engine),
    )
    preview = gate.Preview(
        SimpleNamespace(pick_id="p/pick:0", place_id="p/place:0"),
        compiler,
        object(),
        object(),
    )
    monkeypatch.setattr(gate, "_require_monitor", Mock())
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_a: {"valid_mask": torch.tensor([True])},
    )
    assert gate.preview_place(action, request, context, picked, preview)["valid"]
    predicted = compiler.ground.call_args_list[1].args[2]
    assert predicted.task is projected_task and context.task is initial_task
    assert torch.equal(predicted.scene.entities["item"].pose, fk)
    assert torch.equal(context.scene.entities["item"].pose, pose)
    assert torch.equal(context.last_qpos, torch.zeros(1, 3))


def test_failed_preview_restores_rng_and_recursion_guard(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    before = torch.random.get_rng_state()

    def fail(*_args):
        torch.rand(1)
        raise ValueError("canonical_effect_monitor_unavailable")

    monkeypatch.setattr(gate, "preview_place", fail)
    with pytest.raises(ValueError, match="canonical_effect_monitor_unavailable"):
        gate.recover(*data[:6])
    assert torch.equal(before, torch.random.get_rng_state())
    assert not gate._PREVIEW.get()
