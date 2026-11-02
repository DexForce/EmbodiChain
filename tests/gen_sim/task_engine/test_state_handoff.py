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
from types import SimpleNamespace
import pytest
import torch
from embodichain.gen_sim.task_engine._task_program.state_handoff import (
    GenSimStateHandoff,
    require_release_effect,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ObjectSemantics,
    PlannerDiagnostics,
    StateDelta,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, TaskState


def _held(entity_id: str = "item") -> HeldObjectState:
    affordance = AntipodalAffordance(
        mesh_vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        ),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
    )
    return HeldObjectState(
        semantics=ObjectSemantics(
            affordance=affordance, geometry={}, entity_id=entity_id
        ),
        object_to_eef=torch.eye(4)[None],
        grasp_xpos=torch.eye(4)[None],
    )


def _binding(motion_key: str = "right") -> SimpleNamespace:
    endpoints = {
        "motion": SimpleNamespace(task_state_key=motion_key),
        "grasp": SimpleNamespace(task_state_key=motion_key),
    }
    return SimpleNamespace(endpoint=lambda _, name: endpoints[name])


def _context(task: TaskState) -> SimpleNamespace:
    return SimpleNamespace(task=task)


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    expected_effects: StateDelta
    diagnostics: PlannerDiagnostics


def test_handoff_requires_matching_resource_and_exclusive_object() -> None:
    task = TaskState(1, "cpu", held_objects={"right": _held()})
    request = SimpleNamespace(binding=_binding())
    handoff = GenSimStateHandoff.from_request(
        request, _context(task), participant="Pick"
    )
    assert handoff.resource == "right"
    assert handoff.require_held(entity_id="item").semantics.entity_id == "item"
    with pytest.raises(ValueError, match="expected 'other'"):
        handoff.require_held(entity_id="other")


def test_handoff_rejects_reusing_a_held_resource() -> None:
    task = TaskState(1, "cpu", held_objects={"right": _held()})
    handoff = GenSimStateHandoff.from_request(
        SimpleNamespace(binding=_binding()), _context(task), participant="Drawer"
    )
    with pytest.raises(ValueError, match="cannot reuse resource"):
        handoff.require_free(participant="Drawer")


def test_release_contract_is_checked_at_the_action_boundary() -> None:
    plan = _Plan(
        plan_success=torch.tensor([True]),
        expected_effects=StateDelta(held_object_updates={"left": None}),
        diagnostics=PlannerDiagnostics(backend="test"),
    )
    assert require_release_effect(plan, "left") is plan
    rejected = require_release_effect(plan, "right")
    assert not rejected.plan_success.any()
    assert rejected.diagnostics.failure.code == "place_release_contract_missing"


@pytest.mark.parametrize("foreign", [False, True])
def test_pick_rejects_missing_or_foreign_attachment(monkeypatch, foreign):
    from embodichain.gen_sim.task_engine._task_program import actions

    monkeypatch.setattr(actions, "_owns_constrained_pick", lambda *args: True)
    request = SimpleNamespace(
        goal=SimpleNamespace(semantics=_held().semantics),
        skill_options=None,
        binding=_binding(),
    )
    first = _Plan(
        torch.tensor([True]),
        StateDelta(held_object_updates={"right": _held("foreign")} if foreign else {}),
        PlannerDiagnostics(backend="test"),
    )
    result = actions.GenSimPickUp._recover_constrained_pick_roll(
        SimpleNamespace(motion_generator=object()),
        request,
        _context(TaskState(1, "cpu")),
        first,
    )
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "pick_attachment_unavailable"
